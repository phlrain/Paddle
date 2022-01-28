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

#include "paddle/pten/core/ddim.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {

inline void GetDims(
    const framework::DDim& dim, int axis, int* pre, int* n, int* post) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  for (int i = 0; i < axis; ++i) {
    (*pre) *= dim[i];
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= dim[i];
  }
}

template <typename T, typename Context>
void NormKernel(const Context& ctx,
                const DenseTensor& x,
                int axis,
                float epsilon,
                bool is_test,
                DenseTensor* out,
                DenseTensor* norm);

}  // namespace pten
