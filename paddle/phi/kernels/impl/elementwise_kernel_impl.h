// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#endif

namespace phi {
template <typename T, typename Context>
void ElementwiseFMaxKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           int axis,
                           DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::FMaxFunctor<T>, T, T>(
      dev_ctx, x, y, axis, funcs::FMaxFunctor<T>(), out);
}

template <typename T, typename Context>
void ElementwiseFMinKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           int axis,
                           DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::FMinFunctor<T>, T, T>(
      dev_ctx, x, y, axis, funcs::FMinFunctor<T>(), out);
}

}  // namespace phi
