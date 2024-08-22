#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
test for sync batchnorm op.
for both FP64 and FP16 input.
"""

import os
import random
import shutil
import sys
import tempfile
import unittest
from shlex import quote

import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, _set_use_system_allocator, convert_float_to_uint16

import paddle
from paddle import base, nn
from paddle.base import core
from paddle.base.data_feeder import convert_dtype
from paddle.base.framework import in_dygraph_mode
from paddle.pir_utils import test_with_pir_api

_set_use_system_allocator(True)


def enable_static():
    if in_dygraph_mode():
        paddle.enable_static()

        def cleanup():
            paddle.disable_static()

    else:

        def cleanup():
            pass

    return cleanup


def convert_numpy_array(array):
    if array.dtype != np.uint16:
        return array

    cleanup = None
    if not in_dygraph_mode():
        paddle.disable_static()
        cleanup = lambda: paddle.enable_static()

    out = paddle.to_tensor(array).astype(paddle.float32).numpy()
    if cleanup is not None:
        cleanup()
    return out


def create_or_get_tensor(scope, var_name, var, place):
    """Get tensor, if not found, create a new one."""
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_recursive_sequence_lengths([])
        tensor.set(var, place)
    return tensor


def clean_dir(path):
    if isinstance(path, tempfile.TemporaryDirectory):
        path = path.name
    for f in os.listdir(path):
        f = os.path.join(path, f)
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)


def concat_cmd(cmd):
    if isinstance(cmd, str):
        return cmd

    return ' '.join([quote(c) for c in cmd])


class TestSyncBatchNormOpTraining(unittest.TestCase):
    """sync_batch_norm op test."""

    def setUp(self):
        """Setup."""
        # self.dtype = np.float32
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 5e-3
        self.data_dir = tempfile.TemporaryDirectory()
        self.fleet_log_dir = tempfile.TemporaryDirectory()
        # nn.Conv2d don't have dtype args to set the dtype of weight and bias
        self.pre_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype(self.dtype)

    def tearDown(self) -> None:
        self.data_dir.cleanup()
        self.fleet_log_dir.cleanup()
        paddle.set_default_dtype(self.pre_dtype)

    def multi_device_run(self, layout, fetch_list, only_forward=False):
        cmds = [
            sys.executable,
            "-m",
            "paddle.distributed.launch",
        ]
        cmds += ["--log_dir", self.fleet_log_dir.name]
        cmds += ["dist_fleet_sync_batch_norm.py"]
        cmds += ["--data_dir", self.data_dir.name]

        dshape = [
            self.N // core.get_cuda_device_count(),
            self.C,
            self.H,
            self.W,
        ]
        cmds += ["--dshape", str(dshape)]
        cmds += ["--dtype", str(self.dtype.__name__)]
        cmds += ["--layout", layout]
        cmds += ["--fetch_list", str(fetch_list)]
        if only_forward:
            cmds += ["--only_forward"]
        if self.dtype == np.float16 or self.dtype == np.uint16:
            cmds += ["--use_cudnn"]
        cmd = concat_cmd(cmds)
        assert os.system(cmd) == 0, cmd

    def _build_program(
        self, place, layout, seed, sync_bn=False, only_forward=False
    ):
        """Build program."""
        main = base.Program()
        startup = base.Program()
        main.random_seed = seed
        startup.random_seed = seed
        use_cudnn = (self.dtype == np.float16) or (self.dtype == np.uint16)
        with base.unique_name.guard():
            with base.program_guard(main, startup):
                data = paddle.static.data(
                    name='input',
                    shape=self.dshape,
                    dtype=self.dtype,
                )
                if not paddle.framework.use_pir_api():
                    data.desc.set_need_check_feed(False)
                conv_layer = paddle.nn.Conv2D(
                    in_channels=data.shape[1],
                    out_channels=32,
                    kernel_size=1,
                    weight_attr=base.ParamAttr(name='conv2d_weight'),
                    bias_attr=False,
                )
                conv_layer._use_cudnn = use_cudnn
                conv = conv_layer(data)
                bn = paddle.nn.BatchNorm(
                    num_channels=(
                        conv.shape[1] if layout == "NCHW" else conv.shape[3]
                    ),
                    param_attr=base.ParamAttr(name='bn_scale'),
                    bias_attr=base.ParamAttr(name='bn_bias'),
                    moving_mean_name='bn_moving_mean',
                    moving_variance_name='bn_moving_variance',
                    data_layout=layout,
                    is_test=only_forward,
                    dtype=(
                        convert_dtype(self.dtype)
                        if self.dtype != np.uint16
                        else "bfloat16"
                    ),
                )(conv)
                if core.is_compiled_with_rocm():
                    bn = paddle.cast(bn, 'float32')
                else:
                    bn = paddle.cast(bn, 'float64')
                sigmoid = paddle.nn.functional.sigmoid(bn)
                out = paddle.sum(sigmoid)
                if not sync_bn:
                    out = out / core.get_cuda_device_count()
                ops = base.default_main_program().global_block().ops
                if not only_forward:
                    sgd_opt = paddle.optimizer.SGD(learning_rate=0.0)
                    sgd_opt.backward(out)
                    ops = base.default_main_program().global_block().ops
        return main, startup, [out, conv, bn, ops]

    @prog_scope()
    def _compare(self, place, layout, only_forward):
        try:
            with paddle.utils.unique_name.guard():
                self._compare_impl(place, layout, only_forward)
        finally:
            clean_dir(self.data_dir)
            clean_dir(self.fleet_log_dir)

    def _compare_impl(self, place, layout, only_forward):
        """Compare results."""
        seed = 10
        os.environ['FLAGS_cudnn_deterministic'] = "1"
        paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        paddle.enable_static()
        scope = core.Scope()
        if self.dtype == np.uint16:
            data = convert_float_to_uint16(
                np.random.random(size=self.dshape).astype(np.float32) * 4.0 - 2
            )
        else:
            data = (
                np.random.random(size=self.dshape).astype(self.dtype) * 4.0 - 2
            )
        stride = self.N // core.get_cuda_device_count()
        for id in range(core.get_cuda_device_count()):
            filepath = os.path.join(
                self.data_dir.name,
                f'input_{id}_{only_forward}_{self.dtype.__name__}_{layout}.npy',
            )
            np.save(filepath, data[id * stride : (id + 1) * stride])
        data = create_or_get_tensor(
            scope, "input", OpTest.np_dtype_to_base_dtype(data), place
        )

        # Single-GPU, N = 32 per GPU
        main, startup, outs = self._build_program(
            place, layout, seed, False, only_forward
        )
        exe = base.Executor(place)
        exe.run(startup)
        (out, conv, bn, ops) = outs
        fetch_names = [out, conv, bn]
        if paddle.framework.use_pir_api():
            fetch_names = [
                *fetch_names,
                ops[1].result(0),
                ops[0].result(0),
                ops[3].result(0),
                ops[2].result(0),
            ]
        else:
            fetch_names = [
                *fetch_names,
                'bn_moving_mean',
                'bn_moving_variance',
                'bn_scale',
                'bn_bias',
            ]
        if not only_forward:
            if paddle.framework.in_pir_mode():
                others = [
                    ops[7].result(0),
                    ops[7].result(1),
                    ops[-2].result(0),
                    ops[-2].result(1),
                    ops[-2].result(2),
                    ops[-1].result(1),
                ]
            else:
                others = [
                    'batch_norm_0.tmp_0',
                    'batch_norm_0.tmp_1',
                    'bn_scale@GRAD',
                    'bn_bias@GRAD',
                    'batch_norm_0.tmp_3@GRAD',
                    'conv2d_0.tmp_0@GRAD',
                ]
            fetch_names += others
        bn_fetches = exe.run(
            program=main, feed={'input': data}, fetch_list=fetch_names
        )

        #####################################################################
        # Multi-GPUs, self.N / core.get_cuda_device_count() per GPU
        assert core.get_cuda_device_count() > 1

        if paddle.framework.in_pir_mode():
            # may be not like this in dist
            fetch_names = [
                ops[1].result(0),
                ops[0].result(0),
                ops[3].result(0),
                ops[2].result(0),
            ]
        else:
            fetch_names = [
                'bn_moving_mean',
                'bn_moving_variance',
                'bn_scale',
                'bn_bias',
            ]
        if not only_forward:
            if paddle.framework.in_pir_mode():
                others = [
                    ops[7].result(0),
                    ops[7].result(1),
                    ops[-2].result(0),
                    ops[-2].result(1),
                    ops[-2].result(2),
                    ops[-1].result(1),
                ]
            else:
                others = [
                    'batch_norm_0.tmp_0',
                    'batch_norm_0.tmp_1',
                    'bn_scale@GRAD',
                    'bn_bias@GRAD',
                    'batch_norm_0.tmp_3@GRAD',
                    'conv2d_0.tmp_0@GRAD',
                ]
            fetch_names += others

        self.multi_device_run(
            layout, fetch_list=fetch_names, only_forward=only_forward
        )

        fetch_names = [out, conv, bn, *fetch_names]

        for i in range(1, len(bn_fetches)):
            bn_val = bn_fetches[i]
            file_path = os.path.join(
                self.data_dir.name,
                f'output_{0}_{only_forward}_{self.dtype.__name__}_{i}.npy',
            )
            sync_bn_val = np.load(file_path)
            if sync_bn_val.shape != bn_val.shape:
                bn_val = bn_val[:stride]
            np.testing.assert_allclose(
                convert_numpy_array(bn_val),
                convert_numpy_array(sync_bn_val),
                rtol=1e-04,
                atol=self.atol,
                err_msg=f"Output ({fetch_names[i]}) has diff. \n\nBN     {bn_val}\nSync BN {sync_bn_val}",
            )

    # @test_with_pir_api
    def test_train(self):
        """Test training."""
        if not core.is_compiled_with_cuda():
            return

        places = [core.CUDAPlace(0)]
        for place in places:
            for layout in ["NHWC", "NCHW"]:
                self._compare(place, layout, False)

    # @test_with_pir_api
    def test_infer(self):
        """Test inference."""
        if not core.is_compiled_with_cuda():
            return

        places = [core.CUDAPlace(0)]
        for place in places:
            for layout in ["NHWC", "NCHW"]:
                self._compare(place, layout, True)


class TestFP16SyncBatchNormOpTraining(TestSyncBatchNormOpTraining):
    """sync_batch_norm op test for FP16 input."""

    def setUp(self):
        """Setup."""
        self.dtype = np.float16
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 5e-3
        self.data_dir = tempfile.TemporaryDirectory()
        self.fleet_log_dir = tempfile.TemporaryDirectory()
        self.pre_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype(self.dtype)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestBF16SyncBatchNormOpTraining(TestSyncBatchNormOpTraining):
    """sync_batch_norm op test for BF16 input."""

    def setUp(self):
        """Setup."""
        self.dtype = np.uint16
        self.N = 8
        self.C = 16
        self.H = 32
        self.W = 32
        self.dshape = [self.N, self.C, self.H, self.W]
        self.atol = 1e-2
        self.data_dir = tempfile.TemporaryDirectory()
        self.fleet_log_dir = tempfile.TemporaryDirectory()
        self.pre_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype(self.dtype)


class TestDygraphSyncBatchNormAPIError(unittest.TestCase):
    @test_with_pir_api
    def test_errors(self):
        if not core.is_compiled_with_cuda():
            return

        cleanup = enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10)
            x1 = base.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CUDAPlace(0)
            )
            self.assertRaises(TypeError, my_sync_batch_norm, x1)

            # the input dtype of SyncBatchNorm must be float16 or float32 or float64
            # float16 only can be set on GPU place
            x2 = paddle.static.data(
                name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
            )
            if not paddle.framework.use_pir_api():
                x2.desc.set_need_check_feed(False)
            self.assertRaises(TypeError, my_sync_batch_norm, x2)
        cleanup()


class TestConvertSyncBatchNorm(unittest.TestCase):
    @test_with_pir_api
    def test_convert(self):
        if not core.is_compiled_with_cuda():
            return

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            compare_model = paddle.nn.Sequential(
                paddle.nn.Conv2D(3, 5, 3),
                paddle.nn.BatchNorm2D(5),
                paddle.nn.BatchNorm2D(5),
            )
            model = paddle.nn.Sequential(
                paddle.nn.Conv2D(3, 5, 3),
                paddle.nn.BatchNorm2D(5),
                paddle.nn.BatchNorm2D(
                    5,
                    weight_attr=base.ParamAttr(name='bn.scale'),
                    bias_attr=base.ParamAttr(name='bn.bias'),
                ),
            )
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            for idx, sublayer in enumerate(compare_model.sublayers()):
                if isinstance(sublayer, paddle.nn.BatchNorm2D):
                    self.assertEqual(
                        isinstance(model[idx], paddle.nn.SyncBatchNorm), True
                    )


class TestConvertSyncBatchNormCast1(unittest.TestCase):
    @test_with_pir_api
    def test_convert(self):
        if not core.is_compiled_with_cuda():
            return

        class Net(nn.Layer):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2D(3, 5, 3)
                self.bn = []
                bn = self.add_sublayer('bn', nn.BatchNorm2D(5))
                self.bn.append(bn)

            def forward(self, x):
                x = self.conv1(x)
                for bn in self.bn:
                    x = bn(x)
                return x

        model = nn.Sequential()
        model.add_sublayer('net1', Net())
        model.add_sublayer('net2', Net())
        compare_model = nn.Sequential()
        compare_model.add_sublayer('net1', Net())
        compare_model.add_sublayer('net2', Net())
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.assertEqual(len(compare_model.sublayers()), len(model.sublayers()))


class TestDygraphSyncBatchNormDataFormatError(unittest.TestCase):
    def test_errors(self):
        if not core.is_compiled_with_cuda():
            return

        with base.dygraph.guard(base.CUDAPlace(0)):
            my_sync_batch_norm = paddle.nn.SyncBatchNorm(10, data_format='CN')
            data = np.random.random([3, 3, 3]).astype('float32')
            x = paddle.to_tensor(data)
            self.assertRaises(ValueError, my_sync_batch_norm, x)


if __name__ == '__main__':
    paddle.seed(0)
    np.random.seed(0)
    random.seed(0)
    unittest.main()
