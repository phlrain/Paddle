// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/core/builtin_type.h"

namespace ir {
std::vector<Type> VectorType::data() const { return storage()->GetAsKey(); }

}  // namespace ir

IR_DEFINE_EXPLICIT_TYPE_ID(ir::UInt8Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::Int8Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::VectorType)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::BFloat16Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::Float16Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::Float32Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::Float64Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::Int16Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::Int32Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::Int64Type)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::BoolType)
