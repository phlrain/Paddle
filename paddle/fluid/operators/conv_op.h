/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/layout_utils.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int kConvMKLDNNFP32 = 1;
constexpr int kConvMKLDNNINT8 = 2;
constexpr int MaxKeyLength = 256;

// Base convolution operator definations for other conv
// like operators to reuse the implementation.
inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  PADDLE_ENFORCE_GT(
      output_size, 0,
      platform::errors::InvalidArgument(
          "The output's size is expected to be greater than 0. "
          "But recieved: output's size is %d. The output's size is computed by "
          "((input_size + 2 * padding - (dilation * (filter_size - 1) + 1)) / "
          "stride + 1), where input_size is %d, padding is %d, "
          "filter_size is %d, dilation is %d, stride is %d.",
          output_size, input_size, padding, filter_size, dilation, stride));

  return output_size;
}

inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding_1, int padding_2, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + padding_1 + padding_2 - dkernel) / stride + 1;
  PADDLE_ENFORCE_GT(
      output_size, 0,
      platform::errors::InvalidArgument(
          "The output's size is expected to be greater than 0. "
          "But recieved: output's size is %d. The output's size is computed by "
          "((input_size + padding_1 + padding_2 - (dilation * (filter_size - "
          "1) + 1)) / stride + 1), where input_size is %d, padding is "
          "(%d, %d), filter_size is %d, dilation is %d, stride is %d.",
          output_size, input_size, padding_1, padding_2, filter_size, dilation,
          stride));

  return output_size;
}

template <typename T = int>
inline void UpdatePaddingAndDilation(std::vector<T>* paddings,
                                     std::vector<T>* dilation,
                                     const std::string padding_algorithm,
                                     const framework::DDim data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = framework::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        data_dims.size() * 2, paddings->size(),
        platform::errors::InvalidArgument(
            "Attribute padding's size should be the same or twice as the "
            "input's dimension. "
            "But recieved: padding's size is %d, padding is [%s]; input's "
            "dimension is %d, input's shape is [%s].",
            paddings->size(), framework::make_ddim(*paddings), data_dims.size(),
            data_dims));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation->begin() + i) = 1;
    }

  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

inline bool IsExpand(const std::vector<int64_t>& filter_dim,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }
  if (paddings.size() != strides.size()) {
    for (size_t j = 0; j < paddings.size(); ++j) {
      padding_0 = padding_0 && (paddings[j] == 0);
    }
  }
  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

// Define Op classes in .h file so that other conv
// operator implementations can reuse the code.
class Conv2DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

class Conv3DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

class ConvOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{
        {"Input", /*->*/ "Output"}};
    return m;
  }
};

class ConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    std::vector<int64_t> output_shape = ComputeOutputShape(ctx);

    OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output", "Conv");
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
    ctx->ShareLoD("Input", "Output");
  }

  framework::KernelSignature GetExpectedPtenKernelArgs(
      const framework::ExecutionContext& ctx) const override {
    if (ctx.Attr<bool>("use_cudnn")) {
      return framework::KernelSignature(
          "conv2d_cudnn", {"Input", "Filter"},
          {"strides", "paddings", "padding_algorithm", "groups", "dilations",
           "data_format", "use_addto", "workspace_size_MB",
           "exhaustive_search"},
          {"Output"});
    } else {
      return framework::KernelSignature(
          "conv2d", {"Input", "Filter"},
          {"strides", "paddings", "padding_algorithm", "groups", "dilations",
           "data_format", "use_addto", "workspace_size_MB",
           "exhaustive_search"},
          {"Output"});
    }
  }

 protected:
  std::vector<int64_t> ComputeOutputShape(
      framework::InferShapeContext* ctx) const;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class ConvOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

  framework::KernelSignature GetExpectedPtenKernelArgs(
      const framework::ExecutionContext& ctx) const override {
    if (ctx.Attr<bool>("use_cudnn")) {
      return framework::KernelSignature(
          "conv2d_cudnn_grad",
          {framework::GradVarName("Output"), "Input", "Filter"},
          {"strides", "paddings", "padding_algorithm", "groups", "dilations",
           "data_format", "use_addto", "workspace_size_MB",
           "exhaustive_search"},
          {framework::GradVarName("Input"), framework::GradVarName("Filter")});
    } else {
      return framework::KernelSignature(
          "conv2d_grad", {framework::GradVarName("Output"), "Input", "Filter"},
          {"strides", "paddings", "padding_algorithm", "groups", "dilations",
           "data_format", "use_addto", "workspace_size_MB",
           "exhaustive_search"},
          {framework::GradVarName("Input"), framework::GradVarName("Filter")});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class ConvOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class GemmConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    LOG(ERROR) << "run here!!!!!!!!!!!!!";
  }
};

template <typename DeviceContext, typename T>
class GemmConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {}
};

template <typename DeviceContext, typename T>
class GemmConvDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

template <typename DeviceContext, typename T>
class DepthwiseConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    const std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    bool fuse_relu = context.Attr<bool>("fuse_relu_before_depthwise_conv");

    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const std::string data_format = context.Attr<std::string>("data_format");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
    if (channel_last) {
      PADDLE_ENFORCE_EQ(
          output->dims()[output->dims().size() - 1] %
              input->dims()[input->dims().size() - 1],
          0, platform::errors::InvalidArgument(
                 "ShapeError: The output channels must be a multiple of the "
                 "input channels. But receivced output channel number is %d "
                 "and input channel number is %d",
                 output->dims()[output->dims().size() - 1],
                 input->dims()[input->dims().size() - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          output->dims()[1] % input->dims()[1], 0,
          platform::errors::InvalidArgument(
              "ShapeError: The output channels must be a multiple of the "
              "input channels. But receivced output channel number is %d "
              "and input channel number is %d",
              output->dims()[1], input->dims()[1]));
    }

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims;
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_format);
    if (data_layout != framework::DataLayout::kNHWC) {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    }

    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
    if (!is_sys_pad) {
      for (size_t i = 0; i < strides.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();

    if (fuse_relu) {
      math::DepthwiseConvFunctor<DeviceContext, T, true> depthwiseConv;
      depthwiseConv(dev_ctx, *input, filter, strides, paddings, dilations,
                    output, data_layout);
    } else {
      math::DepthwiseConvFunctor<DeviceContext, T, false> depthwiseConv;
      depthwiseConv(dev_ctx, *input, filter, strides, paddings, dilations,
                    output, data_layout);
    }
  }
};

template <typename DeviceContext, typename T>
class DepthwiseConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    Tensor filter = *context.Input<Tensor>("Filter");

    if (!input_grad && !filter_grad) return;

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    bool fuse_relu = context.Attr<bool>("fuse_relu_before_depthwise_conv");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const std::string data_format = context.Attr<std::string>("data_format");

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims;
    const framework::DataLayout data_layout =
        framework::StringToDataLayout(data_format);
    if (data_layout != framework::DataLayout::kNHWC) {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    }
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
    if (!is_sys_pad) {
      for (size_t i = 0; i < strides.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }
    pten::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, input_grad, static_cast<T>(0));

      if (fuse_relu) {
        math::DepthwiseConvInputGradFunctor<DeviceContext, T, true>
            depthwiseConvInputGrad;
        depthwiseConvInputGrad(dev_ctx, *input, filter, *output_grad, strides,
                               paddings, dilations, input_grad, data_layout);
      } else {
        math::DepthwiseConvInputGradFunctor<DeviceContext, T, false>
            depthwiseConvInputGrad;
        depthwiseConvInputGrad(dev_ctx, *input, filter, *output_grad, strides,
                               paddings, dilations, input_grad, data_layout);
      }
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));
      if (fuse_relu) {
        math::DepthwiseConvFilterGradFunctor<DeviceContext, T, true>
            depthwiseConvFilterGrad;
        depthwiseConvFilterGrad(dev_ctx, *input, *output_grad, strides,
                                paddings, dilations, filter_grad, data_layout);
      } else {
        math::DepthwiseConvFilterGradFunctor<DeviceContext, T, false>
            depthwiseConvFilterGrad;
        depthwiseConvFilterGrad(dev_ctx, *input, *output_grad, strides,
                                paddings, dilations, filter_grad, data_layout);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
