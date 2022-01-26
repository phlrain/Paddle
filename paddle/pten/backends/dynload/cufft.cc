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

#include "paddle/pten/backends/dynload/cufft.h"
#include "paddle/fluid/platform/enforce.h"

namespace pten {
namespace dynload {
std::once_flag cufft_dso_flag;
void* cufft_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

CUFFT_FFT_ROUTINE_EACH(DEFINE_WRAP);

bool HasCUFFT() {
  std::call_once(cufft_dso_flag,
                 []() { cufft_dso_handle = GetCUFFTDsoHandle(); });
  return cufft_dso_handle != nullptr;
}

void EnforceCUFFTLoaded(const char* fn_name) {
  PADDLE_ENFORCE_NOT_NULL(
      cufft_dso_handle,
      paddle::platform::errors::PreconditionNotMet(
          "Cannot load cufft shared library. Cannot invoke method %s.",
          fn_name));
}

}  // namespace dynload
}  // namespace pten
