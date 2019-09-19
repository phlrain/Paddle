// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/imperative/tracer.h"
#include <set>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

struct OpBaseCmp {
  bool operator()(OpBase* first, OpBase* second) {
    return first->id() > second->id();
  }
};

static std::vector<std::unique_ptr<OpBase>> CreateGradOpBases(
    const OpBase* fw_op_base, const NameVarBaseMap& in,
    const NameVarBaseMap& out) {
  if (fw_op_base->Info().dygraph_grad_op_maker_) {
    return fw_op_base->Info().dygraph_grad_op_maker_(fw_op_base, in, out);
  } else {
    return {};
  }
}

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs, framework::AttributeMap attrs,
                     const platform::Place& place, bool trace_backward) {
  platform::RecordEvent event(type);
  VLOG(1) << "Trace Op: " << type;
  size_t op_id = GenerateUniqueId();
  auto op = OpBase::Create(op_id, type, ins, outs, std::move(attrs), place);
  op->Run(ins, outs);

  if (ComputeRequiredGrad(ins, outs, trace_backward)) {
    TraceBackward(op, ins, outs);

    VLOG(6) << "Finish tracking Backward of op: " << type;
  }
  VLOG(6) << "Finish tracing fwd op: " << type;
}

bool Tracer::ComputeRequiredGrad(const NameVarBaseMap& ins,
                                 const NameVarBaseMap& outs,
                                 bool trace_backward) {
  // TODO(jiabin): Implement auto prune here
  return trace_backward;
}

void Tracer::TraceBackward(const std::shared_ptr<OpBase>& fwd_op,
                           const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs) {
  // grad_to_var is a map of framework::GradVarName(in_var_name/out_var_name) ->
  // in_var_name/out_var_name
  std::unordered_map<std::string, std::string> grad_to_var;

  // Get grad_op_desc using fwd_op_desc
  std::vector<std::unique_ptr<OpBase>> grad_op_bases_ =
      CreateGradOpBases(fwd_op.get(), ins, outs);

  size_t grad_op_num = grad_op_bases_.size();

  for (size_t i = 0; i < grad_op_num; ++i) {
    size_t trace_id = fwd_op->id();

    std::shared_ptr<OpBase> grad_op = std::move(grad_op_bases_[i]);
    grad_op->SetId(trace_id);
    grad_op->SetPlace(fwd_op->place());
    grad_op->create_operator_base();

    auto& grad_in = *(grad_op->GetMutableInsMap());
    auto& grad_out = *(grad_op->GetMutableOutsMap());
    for (auto& grad_in_it : grad_in) {
      for (auto& var_base_it : grad_in_it.second) {
        if (var_base_it->IsGradFromGradMaker() == true) {
          var_base_it->AddGradOps(grad_op);
        }
      }
    }
    std::set<OpBase*, OpBaseCmp> visited_preceding_ops;
    for (auto& grad_out_it : grad_out) {
      for (auto& var_base_it : grad_out_it.second) {
        auto preceding_ops = var_base_it->GradOps();

        if (!preceding_ops.empty()) {
          for (const auto& op : preceding_ops) {
            visited_preceding_ops.insert(op);
          }
        }
      }
    }
    std::vector<OpBase*> vec_preceding_ops(visited_preceding_ops.begin(),
                                           visited_preceding_ops.end());

    grad_op->SetGradPendingOps(&vec_preceding_ops);

    // this OpBase* is just used to manage op's life time
    engine_->InsertOp(grad_op.get(), grad_op);
  }
}

}  // namespace imperative
}  // namespace paddle
