# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid import unique_name, core
import paddle.fluid as fluid
from paddle.distributed.fleet.meta_optimizers.common import OpRole, OP_ROLE_VAR_KEY, CollectiveHelper
from paddle.distributed.fleet.meta_optimizers.common import is_backward_op, is_optimizer_op, is_update_op
from paddle.distributed.fleet.meta_optimizers.meta_optimizer_base import MetaOptimizerBase
from paddle.distributed.fleet.meta_optimizers.sharding.shard import Shard, ProgramSegment
from paddle.distributed.fleet.meta_optimizers.sharding.fp16_helper import FP16Utils
from paddle.distributed.fleet.meta_optimizers.sharding.weight_decay_helper import WeightDecayHelper
from paddle.distributed.fleet.meta_optimizers.sharding.gradient_clip_helper import GradientClipHelper
from .sharding.offload_helper import OffloadHelper
from paddle.distributed.fleet.meta_optimizers.sharding.prune import ProgramDeps
from paddle.distributed.fleet.meta_optimizers.sharding.utils import *
from paddle.fluid.framework import Program, Variable, name_scope, default_main_program, default_startup_program, device_guard
from paddle.fluid import layers

import logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
from functools import reduce

__all__ = []


class ShardingOptimizer(MetaOptimizerBase):
    """Sharding Optimizer."""

    def __init__(self, optimizer):
        super(ShardingOptimizer, self).__init__(optimizer)
        self.inner_opt = optimizer
        self.meta_optimizers_white_list = [
            "RecomputeOptimizer",
            "AMPOptimizer",
            "LarsOptimizer",
            "LambOptimizer",
            # "ModelParallelOptimizer",
            # "PipelineOptimizer",
        ]
        self.meta_optimizers_black_list = ["GraphExecutionOptimizer", ]
        self._main_program = None
        self._startup_program = None
        self._segments = []
        # params and fp16 params is for broadcast
        self._params = set([])
        self._broadcast_vars = set([])
        # reduced grads to param name
        self._reduced_grads_to_param = {}
        self._shard = Shard()
        self._verbose = False

        # use sharding as outer parallelism (e.g. inner:Megatron & outer sharding)
        self.mp_degree = 1

    def _can_apply(self):
        if not self.role_maker._is_collective:
            return False
        if self.role_maker._worker_num() <= 1:
            return False
        return self.user_defined_strategy.sharding

    def _disable_strategy(self, dist_strategy):
        dist_strategy.sharding = False
        dist_strategy.sharding_configs = {}

    def _enable_strategy(self, dist_strategy, context):
        dist_strategy.sharding = True
        dist_strategy.sharding_configs = {"segment_broadcast_MB": 32}

    def minimize_impl(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        # TODO: (JZ-LIANG) support multiple comm in future
        # self._nrings = self.user_defined_strategy.nccl_comm_num
        self._nrings_sharding = 1
        self._nrings_dp = 1

        # segment
        self._sharding_segment_strategy = str(
            self.user_defined_strategy.sharding_configs[
                "sharding_segment_strategy"])
        if self._sharding_segment_strategy == "segment_broadcast_MB":
            self._broadcast_MB = self.user_defined_strategy.sharding_configs[
                "segment_broadcast_MB"]
            assert self._broadcast_MB > 0, "segment size should larger than zero !"
        elif self._sharding_segment_strategy == "segment_anchors":
            self._sharding_segment_anchors = self.user_defined_strategy.sharding_configs[
                "segment_anchors"]
            assert len(self._sharding_segment_anchors
                       ) > 0, "you should set the sharding segment anchors !"
            self._backward_remain_anchors = self._sharding_segment_anchors[:]
            self._forward_remain_anchors = []
        else:
            raise NotImplementedError(
                "the sharding segment strategy [{}] is not implemented".format(
                    str(self._sharding_segment_strategy)))

        # parallelism
        self.sharding_degree = int(self.user_defined_strategy.sharding_configs[
            "sharding_degree"])
        assert self.sharding_degree > 0, "sharding degree must be larger than zero"
        self.mp_degree = int(self.user_defined_strategy.sharding_configs[
            "mp_degree"])
        # pipeline setting
        # TODO (JZ-LIANG) should revise here for support mix parallelism with pipeline
        self.pp_degree = int(self.user_defined_strategy.sharding_configs[
            "pp_degree"])
        if self.pp_degree > 1:
            assert self.user_defined_strategy.pipeline == True

        self.dp_degree = int(self.user_defined_strategy.sharding_configs[
            'dp_degree'])
        assert self.role_maker._worker_num(
        ) == self.mp_degree * self.sharding_degree * self.pp_degree * self.dp_degree, "global work size [{}], mp_degree [{}], sharding_degree [{}], pp_degree [{}], dp_degree [{}].".format(
            self.role_maker._worker_num(),
            self.mp_degree,
            self.sharding_degree,
            self.pp_degree,
            self.dp_degree, )

        # FIXME (JZ-LIANG) deprecated hybrid_dp
        if self.user_defined_strategy.sharding_configs["hybrid_dp"]:
            logger.warning(
                "[hybrid_dp] API setting is deprecated. Now when dp_degree >= 2, its will be in hybrid dp mode automatically"
            )
            assert self.dp_degree >= 1
        if self.dp_degree > 1:
            self.hybrid_dp = True
        else:
            self.hybrid_dp = False

        # NOTE (JZ-LIANG) 
        # there 2 kind of modes for gradient-merge and hybrid-dp in mixed parallism [sharding] and [pipeline].
        # we distinguish this two modes since the gm/hybrid-dp related allreduce should be insert in different place according different mode to have best performance:
        # sharding: communication within node, and therefore should insert within backward segment to overlap with bw calc, conduct every micro step 
        # pipeline: communication accross nodes, and therefore should insert in update segemnt, conduct just once per global step        
        self.hybrid_dp_mode = None
        # dp here is the pure dp as the outest parallelism
        if self.hybrid_dp:
            assert self.dp_degree > 1, "hybrid dp is on, but dp degree is [{}]".format(
                self.dp_degree)
            if self.pp_degree > 1:
                self.hybrid_dp_mode = "pp_hybrid_dp"
            else:
                assert self.sharding_degree > 1, "by now we only support five kind of hybrid dp: sharding_hybrid_dp, mp_sharding_hybrid_dp, pp_hybrid_dp, mp_sharding_pp_hybrid_dp, sharding_pp_hybrid_dp."
                self.hybrid_dp_mode = "sharding_hybrid_dp"

        # gradient merge
        self._gradient_merge_acc_step = int(
            self.user_defined_strategy.sharding_configs[
                "gradient_merge_acc_step"])
        self.gradient_merge_mode = None
        if self.pp_degree <= 1:
            self.gradient_merge_mode = "sharding_gm"
            self._grad2merged_grad = dict()
        else:
            self.gradient_merge_mode = "pp_gm"
            self._gradient_merge_acc_step = self.user_defined_strategy.pipeline_configs[
                'accumulate_steps']
        if self._gradient_merge_acc_step > 1:
            logger.info("Gradient merge in [{}], acc step = [{}]".format(
                self.gradient_merge_mode, self._gradient_merge_acc_step))

        # optimize offload
        self.optimize_offload = self.user_defined_strategy.sharding_configs[
            "optimize_offload"]

        # this feature is design for ascend, and should NOT be used in GPU training
        self.pp_allreduce_in_optimize = self.user_defined_strategy.sharding_configs[
            "pp_allreduce_in_optimize"]

        if self.inner_opt is None:
            raise ValueError(
                "self.inner_opt of ShardingOptimizer should not be None.")

        if self.pp_degree > 1:
            pp_optimizer = fluid.optimizer.PipelineOptimizer(
                self.inner_opt, self._gradient_merge_acc_step)

            strategy = self.user_defined_strategy
            self.schedule_mode = strategy.pipeline_configs['schedule_mode']
            self.pp_rank_ = self.role_maker._worker_index() // (
                self.sharding_degree * self.mp_degree) % self.pp_degree

            pipeline_opt = dict()
            pipeline_opt['schedule_mode'] = self.schedule_mode
            pipeline_opt['micro_batch_size'] = strategy.pipeline_configs[
                'micro_batch_size']
            pipeline_opt['local_rank'] = self.pp_rank_
            pipeline_opt['global_rank'] = self.role_maker._worker_index()
            pipeline_opt['use_sharding'] = True
            # TODO (JZ-LIANG) should revise here for support mix parallelism with pipeline
            pipeline_opt['ring_id'] = 20
            pipeline_opt['global_ring_id'] = 3
            pipeline_opt['mp_degree'] = self.mp_degree
            pipeline_opt['mp_rank'] = self.role_maker._worker_index(
            ) % self.mp_degree

            main_program = loss.block.program
            main_program._pipeline_opt = pipeline_opt

            optimize_ops, params_grads, program_list, self.pipeline_pair, self.pp_ring_map = pp_optimizer.minimize(
                loss, startup_program, parameter_list, no_grad_set)
            self.pp_degree = len(program_list)
        else:
            optimize_ops, params_grads = self.inner_opt.minimize(
                loss, startup_program, parameter_list, no_grad_set)

        if startup_program is None:
            startup_program = default_startup_program()

        if self.pp_degree > 1:
            startup_program = startup_program._pipeline_opt['startup_program']
            #main_program = main_program._pipeline_opt['section_program']['program']
            print("pp_rank:", self.pp_rank_)
            main_program = program_list[self.pp_rank_]
            with open("main_%d" % self.role_maker._worker_index(), 'w') as f:
                f.writelines(str(main_program))
            main_block = main_program.global_block()
            new_params_grads = []
            for param, grad in params_grads:
                if main_block.has_var(param.name):
                    new_params_grads.append((param, grad))
            params_grads = new_params_grads

        else:
            main_block = loss.block

        startup_block = startup_program.global_block()
        self._main_program = main_block.program
        self._startup_program = startup_program

        if self.pp_degree > 1:
            pp_optimizer._rename_gradient_var_name(main_block)
            with open("main_%d" % self.role_maker._worker_index(), 'w') as f:
                f.writelines(str(main_program))

        # step0: _init_comm
        self._init_comm()

        if self.sharding_degree > 1:

            # step1: build shard
            self._build_shard(params_grads)

            # step2: split_program
            self._split_program(main_block)

            # step3: add broadcast and reduce ops
            self._add_broadcast_allreduce(main_block)
            main_block._sync_with_cpp()
            startup_block._sync_with_cpp()

            main_block._sync_with_cpp()

            # step4: remove unneeded ops and vars from block
            self._prune_main_program(main_block)
            self._prune_startup_program(startup_block)

        if self.pp_degree > 1:
            # sharding-pp related logic
            # pp_optimizer._rename_gradient_var_name(main_block)
            # crop ops
            if self.sharding_degree > 1:
                for idx, op in reversed(list(enumerate(main_block.ops))):
                    if is_update_op(op):
                        op_role_var = op.attr('op_role_var')
                        param_name = op_role_var[0]
                        if not self._shard.has_param(param_name):
                            main_block._remove_op(idx)

                for idx, op in reversed(list(enumerate(main_block.ops))):
                    if op.type != 'cast': continue
                    in_name = op.input_arg_names[0]
                    if in_name not in self._params: continue
                    #if self._shard.has_param(param_name): continue
                    if in_name not in main_block.vars:
                        main_block._remove_op(idx)

            accumulated_grad_names = pp_optimizer._accumulate_gradients(
                main_block)
            # accumulated_grad_names = sorted(accumulated_grad_names)
            if self.pp_allreduce_in_optimize:
                print("persistable FP32 grad: ")
                print(accumulated_grad_names)
                first_optimize_op_index = get_first_check_finite_and_unscale_op_idx(
                    main_block, raise_error=self.user_defined_strategy.amp)
                insert_reduce_ops(
                    main_block,
                    first_optimize_op_index,
                    self.sharding_ring_id,
                    accumulated_grad_names,
                    self._shard,
                    core.op_proto_and_checker_maker.OpRole.Optimize,
                    use_calc_stream=True)
            if self.hybrid_dp and self.hybrid_dp_mode == "pp_hybrid_dp":
                first_optimize_op_index = get_first_check_finite_and_unscale_op_idx(
                    main_block, raise_error=self.user_defined_strategy.amp)
                if first_optimize_op_index >= 0:
                    insert_allreduce_ops(
                        main_block,
                        first_optimize_op_index,
                        self.dp_ring_id,
                        accumulated_grad_names,
                        core.op_proto_and_checker_maker.OpRole.Optimize,
                        use_calc_stream=True,
                        user_defined_strategy=self.user_defined_strategy)

        # if not use sharding, adapt amp/clip, for remain parallelism.
        # cast --> amp --> clip --> opt
        if self.sharding_degree <= 1:
            # amp
            FP16Utils.sync_amp_check_nan_inf(main_block, self.global_ring_id)

            # clip
            gradientclip_helper = GradientClipHelper(self.global_ring_id)
            gradientclip_helper.sync_global_norm(
                main_block, self.global_ring_id, self.dp_degree)

        # step6: loss div dp_degree 
        global_dp_degree = self.sharding_degree * self.dp_degree
        assert int(global_dp_degree) == global_dp_degree
        if global_dp_degree > 1:
            insert_scale_loss_grad_ops(main_block, scale=1.0 / global_dp_degree)

        main_block._sync_with_cpp()

        # TODO(wangxi): add optimize offload
        # opt offload should be enable while gradient merge is enable && acc_step is quite large (e.g. >> 100) 
        # sync its memcpy could not be overlap with calc, otherwise it will slower down training severely. 
        if self.optimize_offload:
            logger.info("Sharding with optimize offload !")
            offload_helper = OffloadHelper()
            offload_helper.offload(main_block, startup_block)
            offload_helper.offload_fp32param(main_block, startup_block)

        # step6: (optional) sharding gradient merge
        if self.gradient_merge_mode == "sharding_gm" and self._gradient_merge_acc_step > 1:
            self._sharding_gradient_merge(main_block)

        # # check op dependecy
        # FIXME (JZ-LIANG) enable checking in future.
        # check_broadcast(main_block)
        # check_allreduce_sum(main_block, self._shard, self.sharding_ring_id,
        #                     self.dp_ring_id)

        if self.hybrid_dp:
            # NOTE(JZ-LIANG) ensure in both sharding_hybrid_dp & pp_hybrid_dp 
            # init param broadcast should be called after startup pruning             
            self._initialization_broadcast(startup_block)

        with open("start_sharding_%d" % self.role_maker._worker_index(),
                  'w') as f:
            f.writelines(str(startup_block.program))
        with open("main_sharding_%d" % self.role_maker._worker_index(),
                  'w') as f:
            f.writelines(str(main_block.program))

        # GPU and NPU need to wait server ready
        self._wait()
        return optimize_ops, params_grads

    def _init_comm(self):

        # config sharding & dp groups
        self._build_groups()

        # sync var
        startup_block = self._startup_program.global_block()
        self.startup_prog_sync_var = startup_block.create_var(
            name="startup_prog_sync_var",
            shape=[1],
            dtype=core.VarDesc.VarType.INT32,
            persistable=False)

        # global ring
        self._collective_helper._init_communicator(
            self._startup_program,
            self.current_endpoint,
            self.global_endpoints,
            self.global_rank,
            self.global_ring_id,
            False,
            global_ring_id=self.global_ring_id,
            sync=False)
        append_naive_sync(startup_block, self.startup_prog_sync_var,
                          self.global_ring_id)

        # mp ring
        if self.mp_degree > 1:
            self._collective_helper._init_communicator(
                self._startup_program,
                self.current_endpoint,
                self.mp_group_endpoints,
                self.mp_rank,
                self.mp_ring_id,
                False,
                global_ring_id=self.global_ring_id,
                sync=False)
            append_naive_sync(startup_block, self.startup_prog_sync_var,
                              self.global_ring_id)

        # sharding ring
        if self.sharding_degree > 1:
            self._collective_helper._init_communicator(
                self._startup_program,
                self.current_endpoint,
                self.sharding_group_endpoints,
                self.sharding_rank,
                self.sharding_ring_id,
                False,
                global_ring_id=self.global_ring_id,
                sync=False)
            append_naive_sync(startup_block, self.startup_prog_sync_var,
                              self.global_ring_id)

        # pp ring
        if self.pp_degree > 1:
            # TODO (JZ-LIANG) to unify this shit
            assert self.pp_rank_ == self.pp_rank, "pp rank for pp opt [{}], pp rank for sharding opt [{}]".format(
                self.pp_rank_, self.pp_rank)

            for pair in self.pipeline_pair:
                pair_key = pair[0] * 1000 + pair[1]
                ring_id = self.pp_ring_map[pair_key]
                print("pp pair:{}, ring_id: {}".format(pair, ring_id))
                if self.pp_rank in pair:
                    pp_group_endpoints = [
                        self.pp_group_endpoints[pair[0]],
                        self.pp_group_endpoints[pair[1]],
                    ]
                    pp_rank = 0 if self.pp_rank == pair[0] else 1
                    self._collective_helper._init_communicator(
                        self._startup_program,
                        self.current_endpoint,
                        pp_group_endpoints,
                        pp_rank,
                        ring_id,
                        False,
                        global_ring_id=self.global_ring_id,
                        sync=False)
                append_naive_sync(startup_block, self.startup_prog_sync_var,
                                  self.global_ring_id)

        # pure dp ring
        if self.dp_degree > 1:
            self._collective_helper._init_communicator(
                self._startup_program,
                self.current_endpoint,
                self.dp_group_endpoints,
                self.dp_rank,
                self.dp_ring_id,
                False,
                global_ring_id=self.global_ring_id,
                sync=False)
            append_naive_sync(startup_block, self.startup_prog_sync_var,
                              self.global_ring_id)

        startup_block._sync_with_cpp()

    def _build_shard(self, params_grads):
        # step 2: split params
        self._params = set([x[0].name for x in params_grads])
        self._shard.setup(params_grads, self.sharding_rank,
                          self.sharding_degree)

        # step 3: get broadcast vars
        self._broadcast_vars = self._shard.find_broadcast_params(
            self._main_program.global_block())

    def _wait(self, ):
        endpoints = self.global_endpoints[:]
        current_endpoint = endpoints[self.global_rank]
        if self.global_rank == 0:
            self._collective_helper._wait(current_endpoint, endpoints)

    def collect_segment(self, segment, op_idx, block):
        segment._start_idx = op_idx + 1
        self._segments.insert(0, segment)
        new_segment = ProgramSegment(block)
        new_segment._end_idx = op_idx + 1

        return new_segment

    def _split_program(self, block):
        for op_idx, op in reversed(list(enumerate(block.ops))):
            if int(op.attr('op_role')) != int(OpRole.Optimize):
                last_backward_op_idx = op_idx + 1
                break

        var2broadcast_time = dict()
        segment = ProgramSegment(block)
        segment._end_idx = last_backward_op_idx
        for op_idx in reversed(range(last_backward_op_idx)):
            op = block.ops[op_idx]
            assert (int(op.attr('op_role')) != int(OpRole.Optimize))
            if self._sharding_segment_strategy == "segment_broadcast_MB":
                if segment._param_mem >= self._broadcast_MB:
                    segment = self.collect_segment(segment, op_idx, block)

            elif self._sharding_segment_strategy == "segment_anchors":
                if int(op.attr('op_role')) == int(OpRole.Backward):
                    for input_name in op.desc.input_arg_names():

                        # NOTE (JZ-LIANG) naive rule to support amp, if amp change, should modify here accordingly
                        if self.user_defined_strategy.amp:
                            if ".cast_fp16@GRAD" not in input_name:
                                continue
                            else:
                                input_name = input_name[:input_name.find(
                                    ".cast_fp16@GRAD")]

                        if input_name in self._backward_remain_anchors:
                            segment = self.collect_segment(segment, op_idx,
                                                           block)
                            assert input_name not in self._forward_remain_anchors, "segment anchor [{}] met twice !".format(
                                input_name)
                            self._backward_remain_anchors.remove(input_name)
                            self._forward_remain_anchors.append(input_name)
                elif int(op.attr('op_role')) == int(OpRole.Forward):
                    for output_name in op.desc.output_arg_names():
                        if output_name in self._forward_remain_anchors:
                            segment = self.collect_segment(segment, op_idx,
                                                           block)
                            self._forward_remain_anchors.remove(output_name)

            # find broadcast vars
            for input_name in op.desc.input_arg_names():
                if input_name not in self._broadcast_vars:
                    continue
                if input_name in segment._param2broadcast:
                    # skip broadcast because it reuse the old broadcast var
                    broadcast_name = segment._param2broadcast[input_name]
                    if input_name != broadcast_name:
                        op._rename_input(input_name, broadcast_name)
                    continue
                if self._shard.has_param(input_name):
                    broadcast_var_name = input_name
                else:
                    broadcast_var_name = unique_name.generate(input_name +
                                                              "@BroadCast")
                    segment._fill_constant_vars.append(broadcast_var_name)

                # (JZ-LIANG) should use Param base name ?
                broadcast_var_base_name = input_name
                if "subprog" in broadcast_var_base_name:
                    # remove suffix
                    broadcast_var_base_name = broadcast_var_base_name[:
                                                                      broadcast_var_base_name.
                                                                      find(
                                                                          ".subprog"
                                                                      )]

                var2broadcast_time[
                    broadcast_var_base_name] = var2broadcast_time.get(
                        broadcast_var_base_name, 0) + 1

                segment._param2broadcast[input_name] = broadcast_var_name
                segment._broadcast_vars.append((broadcast_var_name,
                                                self._shard.device(input_name)))
                segment._param_mem += get_var_size(
                    self._main_program.global_block().var(input_name))

            # find reduce vars
            if self.pp_degree > 1 and self.pp_allreduce_in_optimize:
                # place pipeline gradient allreduce in optimize
                pass
            else:
                if is_backward_op(op) and \
                        OP_ROLE_VAR_KEY in op.attr_names:
                    op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
                    if len(op_role_var) != 0:
                        assert len(op_role_var) % 2 == 0
                        for i in range(0, len(op_role_var), 2):
                            param, reduced_grad = op_role_var[i], op_role_var[
                                i + 1]
                            segment._allreduce_vars.append(reduced_grad)
                            assert (reduced_grad not in
                                    self._reduced_grads_to_param)
                            self._reduced_grads_to_param[reduced_grad] = param

            # find cast op
            if FP16Utils.is_fp16_cast_op(block, op, self._params):
                fp32_param = op.desc.input_arg_names()[0]
                fp16_param = op.desc.output_arg_names()[0]
                if self._shard.has_param(fp32_param):
                    segment._cast_ops[fp16_param] = fp32_param

        if segment._param_mem > 0:
            segment._start_idx = 0
            self._segments.insert(0, segment)

        if self._sharding_segment_strategy == "segment_anchors":
            assert len(
                self._forward_remain_anchors) == 0, "remain anchors {}".format(
                    self._forward_remain_anchors)
            assert len(
                self._backward_remain_anchors) == 0, "remain anchors {}".format(
                    self._backward_remain_anchors)

        if self._verbose:
            for varname in sorted(
                    var2broadcast_time, key=var2broadcast_time.get,
                    reverse=True):
                logger.info("Sharding broadcast: [{}] times [{}]".format(
                    var2broadcast_time[varname], varname))
            for idx_ in range(len(self._segments)):
                logger.info("segment [{}] :".format(idx_))
                logger.info("start op: [{}]  [{}]".format(block.ops[
                    self._segments[idx_]._start_idx].desc.type(), block.ops[
                        self._segments[idx_]._start_idx].desc.input_arg_names(
                        )))
                logger.info("end   op: [{}]  [{}]".format(block.ops[
                    self._segments[idx_]._end_idx].desc.type(), block.ops[
                        self._segments[idx_]._end_idx].desc.input_arg_names()))
        return

    def _prune_main_program(self, block):
        """
        calculate deps from allredce op to optimize op,
        remove ops and vars not needed in this worker

        1. prune regularization (weight decay)
        2. prune cast_fp32_to_fp16; update amp_infine_checking
        3. prune gradient_clip related; update global_norm_sum
        4. prune optimizer op + param + gradient
            
        """
        weightdecay_helper = WeightDecayHelper()
        weightdecay_helper.prune_weight_decay(block, self._shard)
        # NOTE (JZ-LIANG) the sync of FoundInfinite should among one entire Model Parallelism
        # group. and each Data Parallelism group should have its own sync of FoundInfinite
        # amp could use global group for sync
        FP16Utils.prune_fp16(block, self._shard, self._reduced_grads_to_param,
                             self.global_ring_id)
        # clipbyglobalnorm should only use the Model paramllelism group (mp-sharding-pp)
        if self.mp_degree * self.pp_degree == 1:
            # separate the sharding-hybrid senario to keep the accuracy
            gradientclip_helper = GradientClipHelper(self.sharding_ring_id)
            gradientclip_helper.prune_gradient_clip(
                block, self._shard, pure_dp_degree=1)
        else:
            gradientclip_helper = GradientClipHelper(self.global_ring_id)
            gradientclip_helper.prune_gradient_clip(
                block, self._shard, pure_dp_degree=self.dp_degree)

        # build prog deps
        reduced_grads = []
        for idx, op in enumerate(block.ops):
            input_names = op.desc.input_arg_names()
            output_names = op.desc.output_arg_names()
            if op.type == "c_allreduce_sum":
                assert (len(output_names) == 1)
                output_name = output_names[0]
                reduced_grads.append(output_name)

        # prune optimizer state and param
        pruned_opti_vars = []
        for var_name in list(block.vars.keys()):
            if self._shard.is_opti_var(var_name) and \
              not self._shard.has_opt_var(var_name):
                pruned_opti_vars.append(var_name)
        program_deps = ProgramDeps(block, reduced_grads, pruned_opti_vars)

        # Init
        for var_name in program_deps._end_vars:
            program_deps._should_removed_var.add(var_name)

        # Prune
        for idx, op in reversed(list(enumerate(block.ops))):
            if op.type in [
                    "c_allreduce_sum",
                    "c_sync_comm_stream",
                    "c_calc_comm_stream",
                    "c_gen_nccl_id",
                    "c_comm_init",
                    'send_v2',
                    'recv_v2',
            ]:
                pass
            elif op.type == "conditional_block":
                assert (op.desc.has_attr("sub_block"))
                subblock_idx = op.desc.attr("sub_block").id
                subblock_deps = program_deps.get_sub_block_deps(subblock_idx)
                # only prune amp subblock
                if subblock_deps is None or not self._is_amp_subblock(op):
                    continue
                # init
                reversed_output_vars = []
                for output_name in op.desc.output("Out"):
                    if output_name in program_deps._should_removed_var:
                        subblock_deps._should_removed_var.add(output_name)
                        program_deps.crop_output_var_from_op(idx, output_name)
                    else:
                        reversed_output_vars.append(output_name)
                # prune
                for sub_op_idx, _ in reversed(
                        list(enumerate(subblock_deps._block.ops))):
                    if subblock_deps.should_remove_op(sub_op_idx):
                        subblock_deps.remove_op(sub_op_idx)
                reversed_input_vars = []
                for input_name in op.desc.input('Input'):
                    if input_name not in subblock_deps._should_removed_var:
                        reversed_input_vars.append(input_name)
                    else:
                        program_deps.crop_input_var_from_op(idx, input_name)
                op.desc.set_input('Input', reversed_input_vars)
                op.desc.set_output('Out', reversed_output_vars)
            else:
                # if all outputs of this op are in _should_removed_var
                # _should_removed_var: opt state not cur shard
                if program_deps.should_remove_op(idx):
                    program_deps.remove_op(idx)

        # NOTE (JZ-LIANG) revise and unify logic here
        # sharding support fp16_allreduce logic            
        block._sync_with_cpp()
        for idx, op in reversed(list(enumerate(block.ops))):
            if op.type == 'concat' and is_optimizer_op(op):
                # remove inputs that not on this card
                reserved_x = []
                for var_name in op.desc.input("X"):
                    if block.has_var(var_name): reserved_x.append(var_name)
                op.desc.set_input('X', reserved_x)
        block._sync_with_cpp()
        return

    def _add_broadcast_allreduce(self, block):
        """
        add broadcast allreduce op
        if enable gradient_merge, insert related ops

        if combined with pipeline(grad accumulate), 
        the grad allreduce should be done in optimize role
        """
        if len(self._segments) < 1:
            return
        # sharding
        if self.pp_degree > 1 and self.pp_allreduce_in_optimize:
            for idx in range(len(self._segments)):
                assert len(self._segments[idx]._allreduce_vars) == 0

        # NOTE (JZ-LIANG) revise and unify logic here
        # fix the _end_idx for segments[-1] if pp is used.
        new_end_idx = self._segments[-1]._end_idx
        for idx in range(self._segments[-1]._end_idx - 1,
                         self._segments[-1]._start_idx - 1, -1):
            op = block.ops[idx]
            if op.type == "fill_constant" or op.type == "sum":
                if "MERGED" in op.output_arg_names[0]: new_end_idx = idx + 1
            elif op.type == "cast":
                if "@TMP" in op.output_arg_names[0]: new_end_idx = idx + 1
        self._segments[-1]._end_idx = new_end_idx

        if self._segments[-1]._allreduce_vars:
            shard_allredue_vars = self._shard.filter_grads(self._segments[-1]
                                                           ._allreduce_vars)
            if self.gradient_merge_mode != "sharding_gm" or self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and self.hybrid_dp_mode == "sharding_hybrid_dp" and len(
                        shard_allredue_vars) >= 1:
                    insert_sync_comm_ops(block, self._segments[-1]._end_idx,
                                         self.dp_ring_id, shard_allredue_vars)
                    insert_allreduce_ops(
                        block,
                        self._segments[-1]._end_idx,
                        self.dp_ring_id,
                        shard_allredue_vars,
                        user_defined_strategy=self.user_defined_strategy)
            # gradient merge 
            elif self.gradient_merge_mode == "sharding_gm" and self._gradient_merge_acc_step > 1:
                self.create_persistable_gradients_and_insert_merge_ops(
                    block,
                    self._startup_program.global_block(),
                    self._segments[-1]._end_idx, shard_allredue_vars,
                    self._shard)

            insert_sync_comm_ops(block, self._segments[-1]._end_idx,
                                 self.sharding_ring_id,
                                 self._segments[-1]._allreduce_vars)
            # allreduce --> reduce 
            insert_reduce_ops(
                block,
                self._segments[-1]._end_idx,
                self.sharding_ring_id,
                self._segments[-1]._allreduce_vars,
                self._shard,
                op_role=OpRole.Backward,
                use_calc_stream=False)

        for idx, segment in reversed(list(enumerate(self._segments))):
            allreduce_vars = self._segments[
                idx - 1]._allreduce_vars if idx > 0 else []
            broadcast_vars = self._segments[idx +
                                            1]._broadcast_vars if idx < len(
                                                self._segments) - 1 else []
            fill_constant_vars = self._segments[
                idx + 2]._fill_constant_vars if idx < len(
                    self._segments) - 2 else []
            cast_ops = self._segments[idx + 2]._cast_ops if idx < len(
                self._segments) - 2 else {}

            for op_idx in reversed(range(segment._start_idx, segment._end_idx)):
                op = block.ops[op_idx]
                for input_name in op.desc.input_arg_names():
                    if input_name in segment._param2broadcast and \
                        input_name != segment._param2broadcast[input_name]:
                        op._rename_input(input_name,
                                         segment._param2broadcast[input_name])

            for param_name, broadcast_name in segment._param2broadcast.items():
                if param_name != broadcast_name:
                    block.create_var(
                        name=broadcast_name,
                        shape=self._main_program.global_block().var(
                            param_name).shape,
                        dtype=self._main_program.global_block().var(param_name)
                        .dtype,
                        persistable=False)

            # step1: remove cast ops
            block._sync_with_cpp()
            segment._end_idx += FP16Utils.remove_cast_op(block, self._params,
                                                         segment, 0)

            # step2: add Sync ops
            shard_allredue_vars = self._shard.filter_grads(allreduce_vars)

            if self.gradient_merge_mode != "sharding_gm" or self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and self.hybrid_dp_mode == "sharding_hybrid_dp" and len(
                        shard_allredue_vars) >= 1:
                    insert_sync_comm_ops(block, segment._end_idx,
                                         self.dp_ring_id, shard_allredue_vars)

                    broad_cast_vars = [x[0] for x in broadcast_vars]
                    if len(broad_cast_vars) > 0:
                        insert_sync_comm_ops(block, segment._end_idx,
                                             self.sharding_ring_id,
                                             broad_cast_vars)
                else:
                    comm_dep_vars = allreduce_vars + [
                        x[0] for x in broadcast_vars
                    ]
                    if len(comm_dep_vars) > 0:
                        insert_sync_comm_ops(block, segment._end_idx,
                                             self.sharding_ring_id,
                                             comm_dep_vars)
            # gradient merge
            elif self.gradient_merge_mode == "sharding_gm" and self._gradient_merge_acc_step > 1:
                broad_cast_vars = [x[0] for x in broadcast_vars]
                if len(broad_cast_vars) > 0:
                    insert_sync_comm_ops(block, segment._end_idx,
                                         self.sharding_ring_id, broad_cast_vars)

            calc_dep_vars = fill_constant_vars + [
                k for k, v in cast_ops.items()
            ] + self._segments[idx]._allreduce_vars

            if len(calc_dep_vars) > 0:
                insert_sync_calc_op(block, segment._end_idx,
                                    [calc_dep_vars[-1]])

            # step3: insert `fill_constant` ops 
            insert_fill_constant_ops(block, segment._end_idx,
                                     fill_constant_vars)

            # step4: add `cast` ops     
            insert_cast_ops(block, segment._end_idx, cast_ops)

            # step5: add broadcast ops
            # gradient merge
            if self.gradient_merge_mode == "sharding_gm" and self._gradient_merge_acc_step > 1:
                self.create_persistable_gradients_and_insert_merge_ops(
                    block,
                    self._startup_program.global_block(), segment._start_idx,
                    shard_allredue_vars, self._shard)

            insert_broadcast_ops(block, segment._start_idx,
                                 self.sharding_ring_id, broadcast_vars)

            # step6: add all_reduce ops
            # dp
            if self.gradient_merge_mode != "sharding_gm" or self._gradient_merge_acc_step <= 1:
                if self.hybrid_dp and self.hybrid_dp_mode == "sharding_hybrid_dp" and len(
                        shard_allredue_vars) >= 1:
                    insert_allreduce_ops(
                        block,
                        segment._start_idx,
                        self.dp_ring_id,
                        shard_allredue_vars,
                        user_defined_strategy=self.user_defined_strategy)
                    insert_sync_comm_ops(block, segment._start_idx,
                                         self.sharding_ring_id, allreduce_vars)
            # gradient merge
            elif self.gradient_merge_mode == "sharding_gm" and self._gradient_merge_acc_step > 1:
                insert_sync_comm_ops(block, segment._start_idx,
                                     self.sharding_ring_id, allreduce_vars)
            # sharding
            # allreduce --> reduce 
            # TODO temp change
            if len(allreduce_vars) > 0:
                insert_reduce_ops(
                    block,
                    segment._start_idx,
                    self.sharding_ring_id,
                    allreduce_vars,
                    self._shard,
                    op_role=OpRole.Backward,
                    use_calc_stream=False)

            block._sync_with_cpp()

        if self._segments[0]._broadcast_vars:
            broadcast_vars = [x[0] for x in self._segments[0]._broadcast_vars]
            insert_sync_comm_ops(block, self._segments[0]._start_idx,
                                 self.sharding_ring_id, broadcast_vars)
            insert_broadcast_ops(block, self._segments[0]._start_idx,
                                 self.sharding_ring_id,
                                 self._segments[0]._broadcast_vars)

        fill_constant_vars = []
        for x in self._segments[:2]:
            fill_constant_vars += x._fill_constant_vars

        # Join
        cast_ops = {}
        for x in self._segments[:2]:
            for k, v in x._cast_ops.items():
                cast_ops[k] = v

        calc_deps_vars = fill_constant_vars + [k for k, v in cast_ops.items()]
        if fill_constant_vars or cast_ops:
            insert_sync_calc_op(block, self._segments[0]._start_idx,
                                [calc_deps_vars[-1]])

        if fill_constant_vars:
            insert_fill_constant_ops(block, self._segments[0]._start_idx,
                                     fill_constant_vars)

        if cast_ops:
            insert_cast_ops(block, self._segments[0]._start_idx, cast_ops)

        return

    def _prune_startup_program(self, block):
        for idx, op in reversed(list(enumerate(block.ops))):
            for output_name in op.desc.output_arg_names():
                if self._shard.has_var(output_name):
                    continue
                #TODO why do we remove op, when only one var is removed
                block._remove_op(idx, sync=False)
                break

        for var_name in list(block.vars.keys()):
            if self._shard.has_var(var_name):
                continue
            block._remove_var(var_name, sync=False)
        block._sync_with_cpp()

    def _build_groups(self):
        """
        pre-assign ring ids
            mp: 0
            sharding: 1
            pure-dp: 2
            global: 3
            pp: >= 20
        if one parallelism is not enable: -1
        and only support parallelism hierarchy: mp --> sharding --> pp --> dp        
        """
        # step 1: initialize nccl
        self.global_word_size = self.role_maker._worker_num()
        self.global_rank = self.role_maker._worker_index()
        self.global_endpoints = self.role_maker._get_trainer_endpoints()
        self.current_endpoint = self.global_endpoints[self.global_rank]
        self._collective_helper = CollectiveHelper(
            self.role_maker, nrings=self._nrings_sharding)
        assert self.global_word_size % self.mp_degree == 0, \
            "global_word_size: {} should be divisible to the mp_degree: {}".format(self.global_word_size, self.mp_degree)
        assert self.global_word_size % self.sharding_degree == 0, \
            "global_word_size: {} should be divisible to the sharding_degree: {}".format(self.global_word_size, self.sharding_degree)
        assert self.global_word_size % self.pp_degree == 0, \
            "global_word_size: {} should be divisible to the pp_degree: {}".format(self.global_word_size, self.pp_degree)
        assert self.global_word_size % self.dp_degree == 0, \
            "global_word_size: {} should be divisible to the dp_degree: {}".format(self.global_word_size, self.dp_degree)

        # mp group
        if self.mp_degree > 1:
            self.mp_ring_id = 0
            self.mp_rank = self.global_rank % self.mp_degree
            self.mp_group_id = self.global_rank // self.mp_degree
            self.mp_group_endpoints = [
                ep for idx, ep in enumerate(self.global_endpoints)
                if idx // self.mp_degree == self.mp_group_id
            ]
            assert self.current_endpoint in self.mp_group_endpoints
            assert len(
                self.mp_group_endpoints
            ) == self.mp_degree, "num of mp worker in group is [{}], but mp group size is [{}]".format(
                len(self.mp_group_endpoints), self.mp_degree)
        else:
            self.mp_degree = 1
            self.mp_ring_id = -1
            self.mp_rank = -1
            self.mp_group_id = -1
            self.mp_group_endpoints = []

        # sharding 
        if self.sharding_degree > 1:
            self.sharding_ring_id = 1
            self.sharding_rank = (self.global_rank //
                                  self.mp_degree) % self.sharding_degree
            self.sharding_group_id = self.global_rank // (self.mp_degree *
                                                          self.sharding_degree)
            # mp + sharding + ...
            if self.mp_degree > 1:
                self.sharding_group_endpoints = [
                    ep for idx, ep in enumerate(self.global_endpoints)
                    if (idx // (self.mp_degree * self.sharding_degree)) == self.
                    sharding_group_id and idx % self.mp_degree == self.mp_rank
                ]
            # sharding + ...    
            else:
                self.sharding_group_endpoints = [
                    ep for idx, ep in enumerate(self.global_endpoints)
                    if (idx // (self.mp_degree * self.sharding_degree)
                        ) == self.sharding_group_id
                ]
            assert self.current_endpoint in self.sharding_group_endpoints
        else:
            self.sharding_degree = 1
            self.sharding_ring_id = -1
            self.sharding_rank = -1
            self.sharding_group_id = -1
            self.sharding_group_endpoints = []

        # pp
        if self.pp_degree > 1:
            self.pp_ring_id = 20
            self.pp_rank = self.global_rank // (self.sharding_degree *
                                                self.mp_degree) % self.pp_degree
            # (NOTE): Already adjust for (outter-pure) dp
            self.pp_group_id = self.global_rank // (
                self.mp_degree * self.sharding_degree * self.pp_degree)
            pp_first_stage_idx = self.global_rank % (
                self.sharding_degree * self.mp_degree) + self.pp_group_id * (
                    self.mp_degree * self.sharding_degree * self.pp_degree)
            pp_stage_offset = self.sharding_degree * self.mp_degree
            self.pp_group_endpoints = []
            for i in range(self.pp_degree):
                self.pp_group_endpoints.append(self.global_endpoints[
                    pp_first_stage_idx + pp_stage_offset * i])
            assert self.current_endpoint in self.pp_group_endpoints
        else:
            self.pp_degree = 1
            self.pp_ring_id = -1
            self.pp_rank = -1
            self.pp_group_id = -1
            self.pp_group_endpoints = []

        # outter-pure-dp group
        # NOTE (JZ-LIANG) support outter-pure-dp to scale the throughput in 3D parallelism
        # e.g. mp-sharding-pp-dp
        # sharding-hybrid-dp as one senario of outter-pure-dp 
        assert self.global_word_size == self.mp_degree * self.sharding_degree * self.pp_degree * self.dp_degree, "mp_degree: [{}], sharding_degree: [{}], pp_degree: [{}], dp_degree: [{}]; BUT global nrank: [{}]".format(
            self.mp_degree, self.sharding_degree, self.pp_degree,
            self.dp_degree, self.global_word_size)

        if self.dp_degree > 1:
            self.dp_ring_id = 2
            self.dp_rank = self.global_rank // (self.sharding_degree *
                                                self.mp_degree * self.pp_degree)
            dp_first_rank_idx = self.global_rank % (
                self.sharding_degree * self.mp_degree * self.pp_degree)
            dp_offset = (self.sharding_degree * self.mp_degree * self.pp_degree)
            self.dp_group_endpoints = []
            for i in range(self.dp_degree):
                self.dp_group_endpoints.append(self.global_endpoints[
                    dp_first_rank_idx + dp_offset * i])
            assert self.current_endpoint in self.dp_group_endpoints
            logger.info("Hybrid DP mode turn on !")
        else:
            self.dp_ring_id = -1
            self.dp_rank = -1
            self.dp_group_endpoints = []

        # global group
        # use for gen_nccl_comm_sync, amp check nan inf, clip by global norm
        # NOTE (JZ-LIANG) when use global ring for calc global norm and dp_degree > 1, the allreduce result should be devided by dp_degree
        self.global_ring_id = 3

        logger.info("global word size: {}".format(self.global_word_size))
        logger.info("global rank: {}".format(self.global_rank))
        logger.info("global endpoints: {}".format(self.global_endpoints))
        logger.info("global ring id: {}".format(self.global_ring_id))
        logger.info("#####" * 6)

        logger.info("mp group size: {}".format(self.mp_degree))
        logger.info("mp rank: {}".format(self.mp_rank))
        logger.info("mp group id: {}".format(self.mp_group_id))
        logger.info("mp group endpoints: {}".format(self.mp_group_endpoints))
        logger.info("mp ring id: {}".format(self.mp_ring_id))
        logger.info("#####" * 6)

        logger.info("sharding group size: {}".format(self.sharding_degree))
        logger.info("sharding rank: {}".format(self.sharding_rank))
        logger.info("sharding group id: {}".format(self.sharding_group_id))
        logger.info("sharding group endpoints: {}".format(
            self.sharding_group_endpoints))
        logger.info("sharding ring id: {}".format(self.sharding_ring_id))
        logger.info("#####" * 6)

        logger.info("pp group size: {}".format(self.pp_degree))
        logger.info("pp rank: {}".format(self.pp_rank))
        logger.info("pp group id: {}".format(self.pp_group_id))
        logger.info("pp group endpoints: {}".format(self.pp_group_endpoints))
        logger.info("pp ring id: {}".format(self.pp_ring_id))
        logger.info("#####" * 6)

        logger.info("pure dp group size: {}".format(self.dp_degree))
        logger.info("pure dp rank: {}".format(self.dp_rank))
        logger.info("pure dp group endpoints: {}".format(
            self.dp_group_endpoints))
        logger.info("pure dp ring id: {}".format(self.dp_ring_id))
        logger.info("#####" * 6)

        return

    def _initialization_broadcast(self, startup_block):
        """
        this funtion is to ensure the initialization between dp group to be 
        identical when hybrid-dp is used.
        """
        params = []
        for param in startup_block.iter_parameters():
            params.append(param)
            startup_block.append_op(
                type='c_broadcast',
                inputs={'X': param},
                outputs={'Out': param},
                attrs={
                    'ring_id': self.dp_ring_id,
                    'root': 0,
                    OP_ROLE_KEY: OpRole.Forward
                })
        startup_block.append_op(
            type='c_sync_comm_stream',
            inputs={'X': params},
            outputs={'Out': params},
            attrs={'ring_id': self.dp_ring_id,
                   OP_ROLE_KEY: OpRole.Forward})
        # sync within global group
        append_naive_sync(startup_block, self.startup_prog_sync_var,
                          self.global_ring_id)

    # sharding gradient merge
    def create_persistable_gradients_and_insert_merge_ops(
            self, main_block, startup_block, insert_idx, grad_names, shard):

        for grad_name in grad_names:
            assert get_grad_device(
                grad_name, shard
            ) == shard.worker_idx, "try to merge gradient not belong to current shard: [{}]".format(
                grad_name)
            persistable_grad_name = grad_name + '@GradiantMerge'
            assert grad_name not in self._grad2merged_grad, "grad [{}] already in grad2merged_grad, maybe you meet sharing weight case !".format(
                grad_name)
            self._grad2merged_grad[grad_name] = persistable_grad_name
            grad_var = main_block.var(grad_name)
            # create var
            gradient_merge_var = main_block.create_var(
                name=persistable_grad_name,
                shape=grad_var.shape,
                dtype=grad_var.dtype,
                persistable=True)
            startup_gradient_merge_var = startup_block.create_var(
                name=persistable_grad_name,
                shape=grad_var.shape,
                dtype=grad_var.dtype,
                persistable=True)

            # merge gradient
            main_block._insert_op_without_sync(
                insert_idx,
                type="elementwise_add",
                inputs={'X': grad_name,
                        'Y': gradient_merge_var},
                outputs={'Out': gradient_merge_var},
                attrs={
                    'axis': -1,
                    'use_mkldnn': False,
                    OP_ROLE_KEY: OpRole.Backward
                })

            # startup initialization
            startup_block.append_op(
                type="fill_constant",
                outputs={"Out": startup_gradient_merge_var},
                attrs={
                    "shape": grad_var.shape,
                    "dtype": grad_var.dtype,
                    "value": float(0),
                })

        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def _create_gm_cond(self, main_block):
        # Add const var
        acc_step_var = layers.create_global_var(
            name="gradient_merge_acc_step",
            shape=[1],
            value=int(self._gradient_merge_acc_step),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        zero_var = layers.create_global_var(
            name="gradient_merge_zero",
            shape=[1],
            value=int(0),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        # Add step var & cond var
        current_step_var = layers.create_global_var(
            name="gradient_merge_current_step",
            shape=[1],
            value=int(0),
            dtype='int32',
            persistable=True,
            force_cpu=True)

        cond_var = layers.create_global_var(
            name="gradient_merge_cond",
            shape=[1],
            value=bool(0),
            dtype='bool',
            persistable=False,
            force_cpu=True)

        with device_guard("cpu"):
            # step_var = (step_var + 1) % k_step
            main_block.append_op(
                type='increment',
                inputs={'X': [current_step_var]},
                outputs={'Out': [current_step_var]},
                attrs={'step': float(1),
                       OP_ROLE_KEY: OpRole.Optimize})

            main_block.append_op(
                type='elementwise_mod',
                inputs={'X': current_step_var,
                        'Y': acc_step_var},
                outputs={'Out': current_step_var},
                attrs={
                    'axis': -1,
                    OP_ROLE_KEY: OpRole.Optimize,
                    'use_mkldnn': False
                })

            # cond_var = (step_var == 0)
            main_block.append_op(
                type='equal',
                inputs={'X': current_step_var,
                        'Y': zero_var},
                outputs={'Out': cond_var},
                attrs={OP_ROLE_KEY: OpRole.Optimize})
        # paddle.static.Print(current_step_var, message="in FWBW last conditional")
        return cond_var

    def _true_apply_gradient(self):
        """
        allreduce grad@gradientmerge in dp group
        grad@gradientmerge / acc_step
        re-create all optimize ops of origin main block and rename them
            cast(backward)
            amp 
            clip
            opt
        # fill constant grad@gradientmerge

        """
        # current conditional block
        main_block = self._main_program.global_block()
        cur_block_idx = self._main_program.current_block_idx
        cur_block = self._main_program.current_block()
        self.cond_block = self._main_program.current_block()

        # cur_block's forward_block & backward_block is itself
        cur_block._set_forward_block_idx(cur_block_idx)

        # allreduce grad@gradientmerge  
        if self.hybrid_dp:
            assert self.dp_ring_id >= 0, "dp_ring_id should larger than 0 when in sharding&DP mode"
            for grad, merged_grad in self._grad2merged_grad.items():
                merged_grad_var = main_block.var(merged_grad)
                cur_block.append_op(
                    type='c_allreduce_sum',
                    inputs={'X': merged_grad_var},
                    outputs={'Out': merged_grad_var},
                    attrs={
                        'ring_id': self.dp_ring_id,
                        'use_calc_stream': True,
                        OP_ROLE_KEY: OpRole.Optimize
                    })

        # grad@gradientmerge / acc_step
        for grad, merged_grad in self._grad2merged_grad.items():
            # grad /= k_steps
            merged_grad_var = main_block.var(merged_grad)
            cur_block.append_op(
                type='scale',
                inputs={'X': merged_grad_var},
                outputs={'Out': merged_grad_var},
                attrs={
                    'scale': 1.0 / float(self._gradient_merge_acc_step),
                    'bias': 0.0,
                    'bias_after_scale': False,
                    OP_ROLE_KEY: OpRole.Optimize
                })

        # re-create optimize ops
        already_moved_var_names = []
        for op_desc in self.original_optimize_ops_desc:
            new_op_desc = cur_block.desc.append_op()
            new_op_desc.copy_from(op_desc)

            for input_name in new_op_desc.input_arg_names():
                if input_name in self._grad2merged_grad:
                    new_op_desc._rename_input(
                        input_name, self._grad2merged_grad[input_name])

            for output_name in new_op_desc.output_arg_names():
                if output_name in self._grad2merged_grad:
                    new_op_desc._rename_output(
                        output_name, self._grad2merged_grad[output_name])

                # move non temp optimize vars from block0 to cond block
                if output_name not in already_moved_var_names and output_name not in self._grad2merged_grad.keys(
                ):
                    var_ = self._main_program.global_block().var(output_name)
                    if not var_.persistable:
                        # move
                        name_ = var_.name
                        shape_ = var_.shape
                        type_ = var_.dtype
                        self._main_program.global_block()._remove_var(
                            var_.name, sync=False)
                        self.cond_block.create_var(
                            name=name_,
                            shape=shape_,
                            dtype=type_,
                            persistable=False)
                        already_moved_var_names.append(name_)

        self._main_program.global_block()._sync_with_cpp()
        cur_block._sync_with_cpp()

        # fill zero to grad@gradientmerge
        for grad, merged_grad in self._grad2merged_grad.items():
            merged_grad_var = main_block.var(merged_grad)
            cur_block.append_op(
                type='fill_constant',
                outputs={'Out': merged_grad_var},
                attrs={
                    "shape": merged_grad_var.shape,
                    "dtype": merged_grad_var.dtype,
                    "value": float(0),
                    OP_ROLE_KEY: OpRole.Optimize
                })

        # lr_var = main_block.var("gradient_merge_current_step")
        # paddle.static.Print(lr_var, message="in OPTIMIZE last conditional")

    def _sharding_gradient_merge(self, main_block):
        """
        copy all optimize ops in origin main block
        remove all optimize ops in origin main block
        create cond block

        """
        # copy original optimize ops to temp ops desc list
        # remove them from block 0
        tmp_copy_block = self._main_program._create_block()

        self.original_optimize_ops_desc = []
        for op_idx, op in reversed(list(enumerate(main_block.ops))):
            if int(op.attr('op_role')) != int(OpRole.Optimize):
                continue
            else:
                tmp_op_desc = tmp_copy_block.desc.append_op()
                tmp_op_desc.copy_from(op.desc)
                self.original_optimize_ops_desc.append(tmp_op_desc)
                main_block._remove_op(op_idx, sync=False)
        tmp_copy_block._sync_with_cpp()
        self.original_optimize_ops_desc = list(
            reversed(self.original_optimize_ops_desc))

        # back to block 0
        self._main_program._rollback()

        # create cond vars and ops at the end of block 0
        cond = self._create_gm_cond(main_block)

        # create cond block
        cond_block = self._main_program._create_block()
        self._true_apply_gradient()

        # back to block 0
        self._main_program._rollback()

        # cond op
        step_scope = self._main_program.global_block().create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)
        conditional_block_op = self._main_program.global_block().append_op(
            type='conditional_block',
            inputs={
                'Cond': cond,
                'Input': [],
            },
            outputs={'Out': [],
                     'Scope': [step_scope]},
            attrs={
                'sub_block': cond_block,
                'is_scalar_condition': True,
            })
