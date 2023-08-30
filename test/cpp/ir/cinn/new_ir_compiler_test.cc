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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"

#include "paddle/cinn/utils/data_util.h"

#include "paddle/cinn/hlir/dialect/runtime_dialect/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime_dialect/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/convert_to_dialect.h"
#include "paddle/cinn/hlir/framework/new_ir_compiler.h"

std::unique_ptr<::ir::Program> BuildProgram() {
  ::ir::IrContext* ctx = ::ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  auto program = std::make_unique<::ir::Program>(ctx);
  ::ir::Builder builder = ::ir::Builder(ctx, program->block());

  const float value = 2.0;
  auto full_op_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 128},
                                             value,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());

  auto full_op_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{128, 64},
                                             value,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());
  // TODO(Aurelius84): test more op
  // auto add_z = builder.Build<paddle::dialect::MatmulOp>(full_op_x->result(0),
  //                                                       full_op_y->result(0));
  return std::move(program);
}

TEST(NewIRCompier, CompilerAndRun) {
  // Step 1: Construct ir::Program
  std::unique_ptr<::ir::Program> program = BuildProgram();
  EXPECT_EQ(program->block()->size(), 2u);

  std::stringstream ss;
  program->Print(ss);
  LOG(INFO) << ss.str();

  // Step 2: Compiler New ir::Program into Runtime Program
  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, *program);
  ASSERT_EQ(scope->var_names().size(), 2);

  cinn::hlir::framework::NewIRCompiler ir_compiler(*program, target, scope);
  auto runtime_program = ir_compiler.Build();

  // Step 3: Execute Runtime Instruction and check Scope.
  ASSERT_NO_THROW(runtime_program->Execute());
  const float value = 2.0;
  for (auto& var_name : scope->var_names()) {
    std::string name = {var_name.begin(), var_name.end()};
    std::vector<float> data =
        cinn::GetTensorData<float>(scope->GetTensor(name), target);
    for (int i = 0; i < data.size(); ++i) {
      LOG_FIRST_N(INFO, 3) << "data: " << data[i];
      ASSERT_NEAR(data[i], value, 1e-5);
    }
  }
}

TEST(RuntimeDialect, CompilerAndRun) {
  // Step 1: Construct ir::Program
  std::unique_ptr<::ir::Program> program = BuildProgram();
  EXPECT_EQ(program->block()->size(), 2u);

  // Step 2: Compiler New ir::Program into Runtime Program
  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, *program);
  ASSERT_EQ(scope->var_names().size(), 2);

  cinn::hlir::framework::NewIRCompiler ir_compiler(*program, target, scope);
  auto runtime_program = ir_compiler.Build();

  // Step 3: Convert into cinn::dialect::RuntimeDialect
  std::unique_ptr<::ir::Program> ir_runtime_program =
      cinn::hlir::framework::ConvertToRuntimeDialect(*runtime_program);

  // Step 4: Run cinn::dialect::RuntimeDialect
  for (auto iter = ir_runtime_program->block()->begin();
       iter != ir_runtime_program->block()->end();
       ++iter) {
    auto op = (*iter)->dyn_cast<cinn::dialect::JitKernelOp>();
    auto* instr = op.instruction();
    instr->Run(/*name2podargs=*/nullptr,
               false,
               /*stream=*/nullptr,
               /*use_cache=*/true);
  }
#ifdef CINN_WITH_CUDA
  CUDA_CALL(cudaDeviceSynchronize());
#endif

  // Step 5: Check Scope Tensor Value.
  const float value = 2.0;
  for (auto& var_name : scope->var_names()) {
    std::string name = {var_name.begin(), var_name.end()};
    std::vector<float> data =
        cinn::GetTensorData<float>(scope->GetTensor(name), target);
    for (int i = 0; i < data.size(); ++i) {
      LOG_FIRST_N(INFO, 3) << "data: " << data[i];
      ASSERT_NEAR(data[i], value, 1e-5);
    }
  }
}
