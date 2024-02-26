// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/pir/op_lowering_impl.h"

#include <string>

#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/st_shape_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"

PD_DECLARE_bool(cinn_use_cuda_vectorize);
PD_DECLARE_bool(cinn_enable_map_expr);
PD_DECLARE_bool(cinn_enable_map_expr_schedule);
PD_DECLARE_bool(cinn_bucket_compile);
PD_DECLARE_bool(cinn_new_group_scheduler);

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {

using cinn::common::Type;
using cinn::hlir::op::ExternalApiRegistry;
using framework::OpPatternKind;
using framework::StrategyFunction;

namespace details {

NodeAttr CollectAttrs(const ::pir::Operation& op) {
  NodeAttr node_attrs;
  VLOG(4) << "op.attributes():" << op.attributes().size();
  auto attrs = CompatibleInfo::ConvertAttributes(op);
  node_attrs.node_name = CompatibleInfo::OpName(op);
  node_attrs.attr_store = std::move(attrs);

  return node_attrs;
}

}  // namespace details

int64_t Next2Power(int64_t n) {
  if (n == 1) {
    return 1;
  }
  return int64_t(std::pow(2.0, std::ceil(std::log2(n))));
}

Expr BuildOuputExpr(cinn::ir::Tensor tensor) {
  auto axis = tensor->axis();
  int rank = axis.size();
  std::vector<cinn::ir::Expr> indices;
  for (auto& d : axis) {
    // std::cerr << "dd " << d << std::endl;
    indices.push_back(Expr(d));
  }

  auto shape = tensor->shape;

  auto body = ir::Load::Make(tensor, indices);

  auto out_name = tensor->name + "_out";
  auto out_tensor = ir::Tensor(out_name,
                               tensor->type(),
                               tensor->shape,
                               tensor->domain,
                               tensor->operation);
  body = ir::Store::Make(out_tensor, body, indices);

  std::vector<ir::Var> block_vars;
  std::vector<ir::Expr> iter_values;
  std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(rank);
  for (int i = 0; i < shape.size(); ++i) {
    block_vars.push_back(
        Var(Expr(0), shape[i], cinn::UniqName("i" + std::to_string(i)), false));
    optim::ReplaceVarWithExpr(&body, axis[i], block_vars[i]);
    axis_vars[i]->is_reduce_axis = false;
    if (shape[i] == Expr(1)) {
      iter_values.push_back(Expr(0));
    } else {
      iter_values.push_back(axis_vars[i]);
    }
  }
  body = ir::ScheduleBlockRealize::Make(
      iter_values, ir::ScheduleBlock::Make(block_vars, {}, {}, out_name, body));
  for (int i = rank - 1; i >= 0; --i) {
    ir::Var loop_var = axis[i];
    ir::Expr loop_extent = shape[i];
    body = ir::For::Make(loop_var,
                         Expr(0),
                         loop_extent,
                         ir::ForType::Serial,
                         ir::DeviceAPI::CUDA,
                         ir::Block::Make({body}));
  }

  body = ir::ScheduleBlockRealize::Make(
      {}, ir::ScheduleBlock::Make({}, {}, {}, "root_", body));

  return body;
}

std::shared_ptr<cinn::ir::GroupTileInfo> OpLowererImpl::GetGroupTileInfo(
    const GroupPtr& group) {
  auto master_ops = group->master_ops;
  std::shared_ptr<cinn::ir::GroupTileInfo> group_tile_info;
  // PADDLE_ENFORCE_GT(master_ops.size(), 0, "master op MUST great than 0");

  group_tile_info = std::make_shared<cinn::ir::GroupTileInfo>();

  std::stringstream ss;
  ::pir::IrPrinter printer(ss);

  ss << "group\t" << group->group_id << std::endl;
  ss << "kind\t" << group->kind() << std::endl;

  for (auto op : group->ops) {
    printer.PrintOperation(op);
    ss << "\n";
  }

  // std::cerr << ss.str() << std::endl;

  // pir:: first_master_op = *master_ops.begin();
  auto data_dim = group->loop_ranges;
  group_tile_info->data_rank = data_dim.size();
  auto reduce_axis = group->reduce_axis;
  // if (group->kind() == OpPatternKind::kReduction) {
  //   data_dim = first_master_op->operand_source(0)
  //                  .type()
  //                  .dyn_cast<paddle::dialect::DenseTensorType>()
  //                  .dims();
  //   group_tile_info->data_rank = data_dim.size();
  //   reduce_axis = cinn::dialect::ir::GetVectorAttr(first_master_op, "dim");
  // } else if (group->kind() == OpPatternKind::kElementWise) {
  //   data_dim = first_master_op->result(0)
  //                  .type()
  //                  .dyn_cast<paddle::dialect::DenseTensorType>()
  //                  .dims();
  //   // std::cerr << "data dim " << data_dim << std::endl;
  //   group_tile_info->data_rank = data_dim.size();
  // } else if (group->kind() == OpPatternKind::kBroadcast) {
  //   data_dim = first_master_op->result(0)
  //                  .type()
  //                  .dyn_cast<paddle::dialect::DenseTensorType>()
  //                  .dims();
  //   group_tile_info->data_rank = data_dim.size();
  // } else {
  //   PADDLE_THROW(
  //       phi::errors::Unimplemented("only support group kind with reduce, "
  //                                  "elementwise and broadcast for now"));
  // }
  // std::cerr << "data rank " << group_tile_info->data_rank << std::endl;
  // std::cerr << "data dim " << data_dim << std::endl;

  std::set<int64_t> reduce_set;
  for (auto dim : reduce_axis) {
    if (dim < 0) {
      dim += group_tile_info->data_rank;
    }

    group_tile_info->reduce_axis_.push_back(dim);
    reduce_set.insert(dim);
  }

  int64_t flatten_numel = 1;
  int64_t reduce_numel = 1;

  for (int64_t i = 0; i < group_tile_info->data_rank; ++i) {
    if (reduce_set.count(i)) {
      reduce_numel *= data_dim[i];
    } else {
      flatten_numel *= data_dim[i];
    }
  }

  if (reduce_numel < 0 || flatten_numel < 0) {
    std::cerr << "reduce numel " << reduce_numel << "\t" << flatten_numel
              << std::endl;
    throw std::runtime_error("negative reduce numel or flaten numel");
  }

  int64_t reduce_block = 1;
  int64_t flatten_block = 1;

  // std::cerr << "reduce numel " << reduce_numel << "\t" << flatten_numel
  //           << std::endl;
  int64_t reduce_inner_num = 1;
  int64_t flatten_inner_num = 1;
  int warp_num = 1;

  if (reduce_numel == 1) {
    // warp_num * 32 * flattern_inner = flatten_block
    reduce_block = 1;
    flatten_block = Next2Power(flatten_numel);
    if (flatten_block > 1024) {
      flatten_block = 1024;
    }
    reduce_inner_num = 1;
    warp_num = flatten_block / 128;
    if (warp_num == 0) {
      warp_num = 1;
    }
    flatten_inner_num = flatten_block / (warp_num * 32);
    if (flatten_inner_num == 0) {
      flatten_inner_num = 1;
    }

    int64_t block_num = int64_t(std::ceil(flatten_numel * 1.0 / flatten_block));
    group_tile_info->block_num = block_num;
  } else if (reduce_numel <= 256) {
    // warp reduce
    reduce_block = Next2Power(reduce_numel);
    flatten_block = 256 / reduce_block;
    flatten_inner_num = flatten_block;
    reduce_inner_num = reduce_block / 32;
    if (reduce_inner_num == 0) {
      reduce_inner_num = 2;
    }
    warp_num = 8;
  } else if (reduce_numel > 256 && reduce_numel <= 2048) {
    flatten_block = 1;
    reduce_block = int64_t(std::ceil(reduce_numel * 1.0 / 256.0)) * 256;
    warp_num = reduce_block / 256;
    flatten_inner_num = 1;
    reduce_inner_num = 8;
  } else if (reduce_numel > 2048) {
    flatten_block = 1;
    reduce_block = 2048;
    warp_num = 8;
    reduce_inner_num = int64_t(std::ceil(reduce_numel * 1.0 / 256.0));
    flatten_inner_num = 1;
  }

  group_tile_info->reduce_numel = reduce_numel;
  group_tile_info->reduce_block = reduce_block;
  // flatten_block = std::min(flatten_block, flatten_numel);
  // reduce_block = std::min(reduce_block, reduce_numel);

  // int warp_num = (flatten_block * reduce_block) / 128;
  // if ((flatten_block * reduce_block) % 128 != 0) {
  //   std::cerr << "flatten block reduce block " << flatten_block << "\t"
  //             << reduce_block << std::endl;
  //   throw std::runtime_error("flatten block reduce block not divice by 128");
  // }
  // warp_num = next_power_of_2(min(max(warp_num, 2), 8))

  std::cerr << "block num " << group_tile_info->block_num << std::endl;
  std::cerr << "num warp " << warp_num << std::endl;
  std::cerr << "flatten block " << flatten_block << std::endl;
  std::cerr << "reduce block  " << reduce_block << std::endl;
  std::cerr << "flatten inner num " << flatten_inner_num << std::endl;
  std::cerr << "reduce inner num " << reduce_inner_num << std::endl;

  group_tile_info->warp_num = warp_num;
  group_tile_info->flatten_inner_num = flatten_inner_num;
  group_tile_info->reduce_inner_num = reduce_inner_num;

  if (reduce_block > 1 && reduce_block <= 256) {
    group_tile_info->reduce_type = 0;
  }

  for (auto op : group->ops) {
    if (CompatibleInfo::OpKind(*op) == OpPatternKind::kReduction) {
      std::cerr << "reduce var name " << ValueName(op->result(0)) << std::endl;
      group_tile_info->reduce_var_names.insert(ValueName(op->result(0)));
    }

    // if( group->output_ops.count( op ) )
    // {
    //    for( size_t i =0 ; i < op->num_results() ;++i )
    //   {
    //     std::cerr << "output var name " << ValueName(op->result(i) ) <<
    //     std::endl;;
    //   }
    // }
  }

  group_tile_info->shared_var_names = shared_var_names;
  group_tile_info->direct_output_var_names = direct_output_var_names;
  group_tile_info->thread_sync_before_names = thread_sync_before_names;

  group_tile_info->broadcast_info = broadcast_info;
  group_tile_info->broadcast_to_elementwise = broadcast_to_elementwise;

  group_tile_info->copyed_var_names = copyed_var_names;

  return group_tile_info;
}

OpLowererImpl::OpLowererImpl(const Target& target) : target_(target) {
  name_gene_ = new PrettyNamer();
}

std::vector<ir::LoweredFunc> OpLowererImpl::Lower(const GroupPtr& group,
                                                  bool apply_op_schedule,
                                                  bool apply_group_schedule,
                                                  bool apply_pass) {
  VLOG(3) << "Lowering Group : " << group->group_id
          << " , Op Pattern : " << group->op_pattern_kind;
  group->input_names.clear();
  group->output_names.clear();
  switch (group->op_pattern_kind) {
    case framework::kElementWise:
    case framework::kBroadcast:
    case framework::kInjective:
      return LowerGroup(group,
                        apply_op_schedule,
                        apply_group_schedule,
                        &OpLowererImpl::ElementwiseScheduleDetermineFunction);
    case framework::kReduction:
      return LowerGroup(group,
                        apply_op_schedule,
                        apply_group_schedule,
                        &OpLowererImpl::ReduceScheduleDetermineFunction);
    case framework::kOutFusible:
      LOG(FATAL) << "Group Pattern Kind kOutFusible Is Not Implemented!";
    case framework::kNonFusible:
      return LowerGroup(group,
                        apply_op_schedule,
                        apply_group_schedule,
                        &OpLowererImpl::NonFusibleScheduleDetermineFunction);
    default:
      LOG(FATAL) << "Group Pattern Kind Is Unknown!";
  }
}
BucketLoweredFuncsWrapper OpLowererImpl::BucketLower(const GroupPtr& group,
                                                     bool apply_op_schedule,
                                                     bool apply_group_schedule,
                                                     bool apply_pass) {
  // 1.Do compute, lower and schedule for each op.
  auto& ops = group->ops;
  if (ops.size() == 1 && ops[0]->name() == "custom_call") {
    return {{{ir::Expr(1), LowerCustomCall(group)[0]}}, ir::LoweredFunc()};
  }
  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::unordered_map<std::string, ir::Tensor> tmp_tensor_info;
  std::vector<ir::Expr> func_bodies =
      LowerOps(group,
               ops,
               apply_op_schedule,
               &OpLowererImpl::DyShapeScheduleDetermineFunction,
               &group_func_arg_tensors,
               &tensor_map,
               &tmp_tensor_info);

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr(func_bodies);
  // ir::IRSchedule ir_sch(
  //     mod_expr, -1, false, cinn::utils::ErrorMessageLevel::kGeneral, true);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  std::vector<std::pair<ir::SymbolicPredicate, ir::Expr>> cond2func_bodies;
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  if (apply_group_schedule) {
    std::unordered_set<std::string> output_tensor_names;
    for (auto it = group->output_ops.begin(); it != group->output_ops.end();
         ++it) {
      output_tensor_names.insert(ValueName((*it)->result(0)));
    }

    std::shared_ptr<cinn::ir::GroupTileInfo> group_tile_info;
    std::unique_ptr<ir::GroupScheduler> group_scheduler =
        ir::GroupScheduler::Make(&ir_sch,
                                 output_tensor_names,
                                 target_,
                                 /* is_dy_shape = */ true,
                                 group_tile_info);

    group_scheduler->Schedule();

    cond2func_bodies = group_scheduler->GetIRs();
  } else {
    cond2func_bodies.emplace_back(ir::Expr(true),
                                  ir_sch.GetModule().GetExprs()[0]);
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  std::vector<ir::Expr> scheduled_func_bodies;
  for (std::pair<ir::SymbolicPredicate, ir::Expr>& cond2body :
       cond2func_bodies) {
    scheduled_func_bodies.push_back(cond2body.second);
  }
  std::vector<ir::Tensor> group_func_arg_tensors_copy = group_func_arg_tensors;
  std::vector<ir::Argument> group_func_args;
  std::vector<ir::LoweredFunc> funcs = PostProcess(group,
                                                   tensor_map,
                                                   apply_group_schedule,
                                                   {scheduled_func_bodies},
                                                   &group_func_arg_tensors_copy,
                                                   &group_func_args);
  CHECK_EQ(funcs.size(), cond2func_bodies.size());
  BucketLoweredFuncsWrapper funcs_wrapper;
  for (int i = 0; i < funcs.size(); ++i) {
    funcs_wrapper.predicate2funcs.emplace_back(cond2func_bodies[i].first,
                                               funcs[i]);
  }
  funcs_wrapper.infer_shape_func = GenerateInferShapeFunc(
      group, group_func_arg_tensors_copy, group_func_args);

  return funcs_wrapper;
}

void OpLowererImpl::InsertNameGeneToScope(std::shared_ptr<Scope> scope) {
  auto& name_map = name_gene_->GetNameMap();
  for (auto it = name_map.begin(); it != name_map.end(); ++it) {
    auto value = it->first;
    if (!(value) || !(value.type())) {
      return;
    }

    auto& name = it->second;
    auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto* var = scope->Var<Tensor>(name);
    auto& tensor = absl::get<Tensor>(*var);

    std::vector<Shape::dim_t> shape;
    for (auto i = 0; i < type_info.dims().size(); ++i) {
      shape.push_back(Shape::dim_t(type_info.dims()[i]));
    }
    tensor->Resize(Shape{shape});
    tensor->set_type(pir::CompatibleInfo::ConvertIRType(type_info.dtype()));
  }
}

bool OpLowererImpl::ElementwiseScheduleDetermineFunction(::pir::Operation* op) {
  return true;
}

bool OpLowererImpl::ReduceScheduleDetermineFunction(::pir::Operation* op) {
  VLOG(3) << "in ReduceScheduleDetermineFunction";
  return CompatibleInfo::OpKind(*op) == framework::kReduction;
}

bool OpLowererImpl::NonFusibleScheduleDetermineFunction(::pir::Operation* op) {
  return true;
}

bool OpLowererImpl::DyShapeScheduleDetermineFunction(::pir::Operation* op) {
  return false;
}

void OpLowererImpl::LowerOpsForMapExpr(
    const GroupPtr& group,
    const std::vector<::pir::Operation*>& ops,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::unordered_map<std::string, ir::Tensor> tmp_tensor_info;
  for (auto* op : ops) {
    // 1.Select Op impl
    std::vector<Type> out_types;
    std::vector<std::vector<int>> out_shapes;

    CollectOutputInfo(op, &out_types, &out_shapes, group);
    VLOG(4) << "out_types.size(): " << out_types.size();
    NodeAttr node_attrs = details::CollectAttrs(*op);

    std::vector<ir::Tensor> op_func_arg_tensors =
        CollectInputTensor(group, op, group_func_arg_tensors, tensor_map);
    VLOG(4) << "input size:" << op_func_arg_tensors.size();

    std::string cinn_op_name = CompatibleInfo::OpName(*op);
    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    auto op_impl = OpStrategy::SelectImpl(strategy[cinn_op](
        node_attrs, op_func_arg_tensors, out_types, out_shapes, this->target_));
    // 2.Perform the lower process of Op
    std::vector<ir::LoweredFunc> funcs = DoOpLower(
        op_impl, op, tensor_map, &tmp_tensor_info, &op_func_arg_tensors);

    group->mut_map_expr_ctx()->UpdateOpLoweredFuncKey(op, funcs);
  }
}

/* Most of below codes copies from `PostProcess` function */
std::vector<ir::LoweredFunc> OpLowererImpl::LowerMapExpr(
    const GroupPtr& group,
    const std::vector<::pir::Operation*>& ops,
    bool apply_op_schedule,
    bool apply_group_schedule,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  if (FLAGS_cinn_enable_map_expr && FLAGS_cinn_enable_map_expr_schedule) {
    apply_op_schedule = false;
    apply_group_schedule = false;
  }
  VLOG(4) << "FLAGS_cinn_enable_map_expr_schedule = "
          << FLAGS_cinn_enable_map_expr_schedule;
  VLOG(4) << "apply_op_schedule = " << apply_op_schedule;
  VLOG(4) << "apply_group_schedule = " << apply_group_schedule;

  LowerOpsForMapExpr(group, ops, group_func_arg_tensors, tensor_map);

  VLOG(4) << "Begin MapExprToIr";
  ir::Expr func_body = adt::MapExprToIr(group->map_expr_ctx(), target_);

  // 2.Do group schedule.
  ir::ModuleExpr mod_expr({func_body});
  // ir::IRSchedule ir_sch(mod_expr);
  ir::IRSchedule ir_sch(
      mod_expr, -1, false, cinn::utils::ErrorMessageLevel::kGeneral, true);
  ir_sch.MergeExprs();
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  if (apply_group_schedule) {
    std::unordered_set<std::string> output_tensor_names;
    for (auto it = group->output_ops.begin(); it != group->output_ops.end();
         ++it) {
      output_tensor_names.insert(ValueName((*it)->result(0)));
    }
    // std::transform(
    //     group->output_ops.begin(),
    //     group->output_ops.end(),
    //     std::inserter(output_tensor_names, output_tensor_names.begin()),
    //     [](::pir::Operation* node) {
    //       ::pir::Value node_data = node->result(0);
    //       return this->ValueName(node_data);
    //     });
    std::shared_ptr<cinn::ir::GroupTileInfo> group_tile_info;
    ir::StaticShapeGroupScheduler group_scheduler(
        &ir_sch, output_tensor_names, target_, group_tile_info);
    group_scheduler.MapExprSchedule();
    VLOG(3) << "After group schedule, ir is: \n"
            << ir_sch.GetModule().GetExprs().at(0);
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  std::vector<ir::Argument> group_func_args;
  return PostProcess(group,
                     *tensor_map,
                     apply_op_schedule,
                     {ir_sch.GetModule().GetExprs()[0]},
                     group_func_arg_tensors,
                     &group_func_args);
}

std::vector<OpPatternKind> GetOpPatternVector(
    const std::vector<::pir::Operation*> ops) {
  auto& op_pattern_map =
      Operator::GetAttrs<cinn::hlir::framework::OpPatternKind>("OpPattern");
  std::vector<OpPatternKind> op_patterns;
  std::transform(ops.begin(),
                 ops.end(),
                 std::back_inserter(op_patterns),
                 [&op_pattern_map](const ::pir::Operation* op) {
                   std::string cinn_op_name = CompatibleInfo::OpName(*op);
                   const hlir::framework::Operator* cinn_op =
                       Operator::Get(cinn_op_name);
                   return op_pattern_map[cinn_op];
                 });
  for (auto op_ptn : op_patterns) {
    VLOG(4) << "OpPattern is :" << op_ptn;
  }
  return op_patterns;
}

static const int NOT_FOUND = -1;

std::set<Expr*> CollectLoadStoreExprPointer(
    Expr x, std::function<bool(const Expr*)>&& teller) {
  if (!x.defined()) return std::set<Expr*>();
  struct Mutator : public ir::IRMutator<> {
    std::function<bool(const Expr*)> teller;
    std::set<Expr*> exprs;
    explicit Mutator(std::function<bool(const Expr*)>&& teller)
        : teller(std::move(teller)) {}

    void operator()(Expr* expr) { ir::IRMutator<Expr*>::Visit(expr, expr); }

    void Visit(const ir::Load* op, Expr* expr) override {
      if (teller(expr)) {
        exprs.insert(expr);
      }
    }
    void Visit(const ir::Store* op, Expr* expr) override {
      if (teller(expr)) {
        exprs.insert(expr);
      }
    }
  };

  Mutator mutator(std::move(teller));
  mutator(&x);
  return mutator.exprs;
}

class TrivalOp {
 public:
  // Diff here! Don't flatten the output tensor, so we have a list of iter_vars.
  std::vector<ir::Var> output_iter_vars;

  // GetInputIndex
  std::vector<ir::Var> input_vars;
  std::vector<ir::Tensor> input_tensors;
  std::vector<std::vector<ir::Expr>> index_exprs;
  ir::Tensor output_tensor;

  int name_counter = 0;

  // CoreFunction
  ir::Expr core_func;

  TrivalOp(std::vector<ir::Var> output_iter_vars,
           std::vector<ir::Var> input_vars,
           std::vector<ir::Tensor> input_tensors,
           std::vector<std::vector<ir::Expr>> index_exprs,
           ir::Tensor output_tensor,
           int name_counter,
           ir::Expr core_func)
      : output_iter_vars(output_iter_vars),
        input_vars(input_vars),
        index_exprs(index_exprs),
        name_counter(name_counter),
        core_func(core_func),
        input_tensors(input_tensors),
        output_tensor(output_tensor) {}

  TrivalOp(const TrivalOp& other) {
    output_iter_vars = other.output_iter_vars;
    input_tensors.clear();
    for (const auto& input_tensor : other.input_tensors) {
      input_tensors.push_back(input_tensor);
    }
    input_tensors = other.input_tensors;
    index_exprs = other.index_exprs;
    name_counter = other.name_counter;
    core_func = other.core_func;
    output_tensor = other.output_tensor;
  }

  TrivalOp& operator=(const TrivalOp& other) {
    if (this == &other) return *this;
    output_iter_vars = other.output_iter_vars;
    input_tensors.clear();
    for (const auto& input_tensor : other.input_tensors) {
      input_tensors.push_back(input_tensor);
    }
    input_tensors = other.input_tensors;
    index_exprs = other.index_exprs;
    name_counter = other.name_counter;
    core_func = other.core_func;
    output_tensor = other.output_tensor;
    return *this;
  }

  explicit TrivalOp(const ir::Expr& origin_func_body) {
    auto func_body = ir::ir_utils::IRCopy(origin_func_body);
    std::set<Expr> store_tensor_exprs =
        cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
            func_body, [](const Expr* expr) {
              return expr->As<ir::Store>() &&
                     expr->As<ir::Store>()->is_addr_tensor();
            });

    output_tensor =
        (*store_tensor_exprs.begin()).As<ir::Store>()->tensor.as_tensor_ref();

    // initialize for output_iter_vars
    PADDLE_ENFORCE(store_tensor_exprs.size() == 1,
                   "TrivalOp must store for output only once.");
    auto store_iter_vars =
        (*store_tensor_exprs.begin()).As<ir::Store>()->indices;
    for (const auto& var : store_iter_vars) {
      PADDLE_ENFORCE(var.is_var(), "store_iter_vars should be ir::Var");
    }
    output_iter_vars.insert(
        output_iter_vars.end(), store_iter_vars.begin(), store_iter_vars.end());

    // initialize for input_var and index_expr
    const auto& store_value =
        (*store_tensor_exprs.begin()).As<ir::Store>()->value;
    std::set<Expr*> load_tensor_exprs =
        CollectLoadStoreExprPointer(store_value, [](const Expr* expr) {
          return expr->As<ir::Load>() && expr->As<ir::Load>()->is_addr_tensor();
        });
    for (const auto& load_expr : load_tensor_exprs) {
      auto load = load_expr->As<ir::Load>();
      input_tensors.push_back(load->tensor.as_tensor_ref());
      input_vars.emplace_back(CreateVarName(name_counter++));
      index_exprs.push_back(load->indices);

      // replace load_expr with input_var
      // transform A[f(i,j,k)] -> Var('A')
      *load_expr = input_vars.back();
    }

    // initialize for core_func, we have change all the load_expr to input_var
    core_func = store_value;

    DebugPrint();
  }

  void DebugPrint() const {
    VLOG(4) << "output_iter_vars: ";
    for (const auto& var : output_iter_vars) {
      VLOG(4) << "    " << var;
    }
    VLOG(4) << "input_vars: ";
    for (const auto& var : input_vars) {
      VLOG(4) << "    " << var;
    }
    VLOG(4) << "input_tensors: ";
    for (const auto& tensor : input_tensors) {
      VLOG(4) << "    " << tensor;
    }
    VLOG(4) << "output_tensors: ";
    VLOG(4) << "    " << output_tensor;
    VLOG(4) << "index_exprs: ";
    for (const auto& index_expr : index_exprs) {
      for (const auto& expr : index_expr) {
        VLOG(4) << "    " << expr;
      }
    }
    VLOG(4) << "core_func: "
            << "\n    " << core_func;
  }

  ir::Expr to_expr(ir::Expr original) {
    // we reuse the original for loop and replace the body.
    VLOG(4) << "Before to_expr: " << original;
    auto cloned_compute = ir::ir_utils::IRCopy(original);
    std::set<Expr*> stores =
        CollectLoadStoreExprPointer(cloned_compute, [](const Expr* expr) {
          return expr->As<ir::Store>() &&
                 expr->As<ir::Store>()->is_addr_tensor();
        });
    ir::Expr* store_expr = *stores.begin();
    ir::Expr value_expr = [&]() {
      std::vector<ir::Expr> load_with_index_exprs;
      for (size_t i = 0; i < input_tensors.size(); ++i) {
        load_with_index_exprs.push_back(input_tensors[i](index_exprs[i]));
      }
      return CopyedReplaceExpr(core_func, input_vars, load_with_index_exprs);
    }();
    (*store_expr) = ir::Store::Make(store_expr->As<ir::Store>()->tensor,
                                    value_expr,
                                    store_expr->As<ir::Store>()->indices);
    VLOG(4) << "After to_expr: " << cloned_compute;
    return cloned_compute;
  }

  static std::vector<ir::Expr> ComposeIndexExpr(
      const std::vector<ir::Expr>& upstream_index_exprs,
      const std::vector<ir::Expr>& down_stream_index_expr,
      const std::vector<ir::Var>& upstream_iter_vars) {
    // fmap CopyedReplaceExpr [expr]
    VLOG(4) << "ComposeIndexExpr Start.";
    std::vector<ir::Expr> result;
    std::transform(upstream_index_exprs.begin(),
                   upstream_index_exprs.end(),
                   std::back_inserter(result),
                   [&](const ir::Expr& expr) {
                     return CopyedReplaceExpr(
                         expr, upstream_iter_vars, down_stream_index_expr);
                   });
    VLOG(4) << "ComposeIndexExpr End";
    return result;
  }

  static ir::Expr ComposeCoreExpr(
      const ir::Expr& downstream_core_expr,
      const ir::Var& downstream_input_var,
      const ir::Expr& upstream_core_expr,
      const std::vector<ir::Expr>& down_stream_index_expr,
      const std::vector<ir::Var>& upstream_iter_vars,
      const std::vector<ir::Var>& upstream_input_var_name,
      const std::vector<ir::Var>& downstream_input_new_name) {
    // fmap CopyedReplaceExpr [expr]
    const auto index_replaced_upstream_core_expr = [&]() -> ir::Expr {
      return CopyedReplaceExpr(
          upstream_core_expr, upstream_iter_vars, down_stream_index_expr);
    }();

    const auto var_replaced_upstream_core_expr = [&]() -> ir::Expr {
      std::vector<ir::Expr> downstream_new_name_expr;
      std::transform(downstream_input_new_name.begin(),
                     downstream_input_new_name.end(),
                     std::back_inserter(downstream_new_name_expr),
                     [&](const ir::Var& var) { return Expr(var); });
      return CopyedReplaceExpr(index_replaced_upstream_core_expr,
                               upstream_input_var_name,
                               downstream_new_name_expr);
    }();

    return CopyedReplaceExpr(downstream_core_expr,
                             {downstream_input_var},
                             {var_replaced_upstream_core_expr});
  }

  static TrivalOp ComposeSingleConnectIdx(const TrivalOp& upstream,
                                          const TrivalOp& downstream,
                                          int connect_idx) {
    VLOG(4) << "ComposeSingleConnectIdx " << connect_idx;
    std::vector<ir::Var> output_iter_vars = downstream.output_iter_vars;
    std::vector<ir::Var> input_vars = downstream.input_vars;
    std::vector<ir::Tensor> input_tensors = downstream.input_tensors;
    std::vector<std::vector<ir::Expr>> index_exprs = downstream.index_exprs;
    int name_counter = downstream.name_counter;

    // update input_vars, input_tensors, index_exprs
    // Step 1: remove connect_idx
    input_vars.erase(input_vars.begin() + connect_idx);
    input_tensors.erase(input_tensors.begin() + connect_idx);
    index_exprs.erase(index_exprs.begin() + connect_idx);

    // Step2: Inline the upstream input_vars, input_tensors, index_exprs
    size_t inline_start_idx = input_vars.size();
    for (int i = 0; i < upstream.input_vars.size(); ++i) {
      input_vars.push_back(CreateVarName(name_counter++));
      input_tensors.push_back(upstream.input_tensors[i]);
      // index_exprs need compose
      index_exprs.push_back(
          ComposeIndexExpr(upstream.index_exprs[i],
                           downstream.index_exprs[connect_idx],
                           upstream.output_iter_vars));
    }

    // update core_func
    ir::Expr core_func = ComposeCoreExpr(
        downstream.core_func,
        downstream.input_vars[connect_idx],
        upstream.core_func,
        downstream.index_exprs[connect_idx],
        upstream.output_iter_vars,
        std::vector<ir::Var>(upstream.input_vars.begin(),
                             upstream.input_vars.end()),
        std::vector<ir::Var>(input_vars.begin(), input_vars.end()));

    // update output_tensor
    ir::Tensor output_tensor = downstream.output_tensor;

    auto ret = TrivalOp(output_iter_vars,
                        input_vars,
                        input_tensors,
                        index_exprs,
                        output_tensor,
                        name_counter,
                        core_func);
    VLOG(4) << "ComposeSingleConnectIdx done.";
    ret.DebugPrint();
    return ret;
  }

  static TrivalOp Compose(const TrivalOp& upstream,
                          const TrivalOp& downstream) {
    // Get connect_idx.
    std::vector<int> connect_ids;
    PADDLE_ENFORCE_EQ(
        downstream.input_tensors.size(),
        downstream.input_vars.size(),
        "size of input_tensors and input_vars should be the same.");
    VLOG(4) << "==============";
    upstream.DebugPrint();
    VLOG(4) << "==============";
    downstream.DebugPrint();
    VLOG(4) << "==============";
    for (int i = static_cast<int>(downstream.input_tensors.size()); i >= 1;
         --i) {
      VLOG(4) << "i = : " << i - 1;
      if (downstream.input_tensors[i - 1]->name ==
          upstream.output_tensor->name) {
        VLOG(4) << "connect ids: " << i - 1;
        connect_ids.push_back(static_cast<int>(i - 1));
        VLOG(4) << "after connect ids: " << i - 1;
      }
    }
    VLOG(4) << "before assign ";
    TrivalOp iter_op(downstream);
    VLOG(4) << "end assign ";
    for (int i : connect_ids) {
      VLOG(4) << "Start ComposeSingleConnectIdx with connect_idx: " << i;
      iter_op = ComposeSingleConnectIdx(upstream, iter_op, static_cast<int>(i));
      VLOG(4) << "Done ComposeSingleConnectIdx with connect_idx: " << i;
    }
    VLOG(4) << "After OpFusion: ";
    iter_op.DebugPrint();
    return iter_op;
  }

 private:
  static ir::Var CreateVarName(int counter) {
    return ir::Var("var" + std::to_string(counter++), type_of<float>());
  }
  static Expr CopyedReplaceExpr(const Expr& source,
                                const std::vector<Var>& replaced,
                                const std::vector<Expr>& candidates) {
    CHECK_EQ(replaced.size(), candidates.size())
        << "In ReplaceExpr, the size of Vars to be replaced must be equal to "
           "the "
           "size of cadidate Exprs! Please check.";
    auto copyed_source = ir::ir_utils::IRCopy(source);
    if (replaced.empty()) return copyed_source;
    std::map<Var, Expr, ir::CompVar> replacing_map;
    for (int i = 0; i < replaced.size(); ++i) {
      // If the Var to be replaced is equal to the candidate, we skip it.
      if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i])
        continue;
      replacing_map[replaced[i]] = candidates[i];
    }
    ir::MappingVarToExprMutator mapper(replacing_map);
    mapper(&copyed_source);
    return copyed_source;
  }
};

static bool IsAdjecent(const ir::Expr& upstream, const ir::Expr& downstream) {
  // 1. Get inputs / output from Expr, then we can tell whether they are
  // adjecent.
  std::set<Expr> upstream_stores =
      cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
          upstream, [](const Expr* expr) {
            return expr->As<ir::Store>() &&
                   expr->As<ir::Store>()->is_addr_tensor();
          });
  // don't support multi-output yet.
  PADDLE_ENFORCE(upstream_stores.size() == 1,
                 "The expr of injective should have only one store");

  std::set<Expr> downstream_loads =
      cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
          downstream, [](const Expr* expr) {
            return expr->As<ir::Load>() &&
                   expr->As<ir::Load>()->is_addr_tensor();
          });

  for (const auto& upstream_store : upstream_stores) {
    for (const auto& downstream_load : downstream_loads) {
      if (upstream_store.As<ir::Store>()->tensor.As<ir::_Tensor_>()->name ==
          downstream_load.As<ir::Load>()->tensor.As<ir::_Tensor_>()->name) {
        return true;
      }
    }
  }
  return false;
}

std::pair<int, int> SearchForAdjacentInjectives(
    const std::vector<OpPatternKind>& op_patterns,
    const std::vector<ir::Expr>& funcs) {
  int upper_stream = NOT_FOUND;
  int down_stream = NOT_FOUND;
  for (int i = 0; i < op_patterns.size(); i++) {
    if (op_patterns[i] <= OpPatternKind::kInjective) {
      for (int j = i + 1; j < op_patterns.size(); j++) {
        if (op_patterns[j] <= OpPatternKind::kInjective &&
            IsAdjecent(funcs[i], funcs[j])) {
          upper_stream = i;
          down_stream = j;
          return std::make_pair(upper_stream, down_stream);
        }
      }
    }
  }
  return std::make_pair(upper_stream, down_stream);
}

ir::Expr TrivalFusion(ir::Expr upper, ir::Expr down) {
  TrivalOp upper_op(upper);
  TrivalOp down_op(down);
  VLOG(4) << "before TrivalOp::Compute";
  auto fused = TrivalOp::Compose(upper_op, down_op);
  VLOG(4) << "before to_expr";
  return fused.to_expr(down);
}

std::vector<ir::Expr> OpInlineFusion(const GroupPtr& group,
                                     const std::vector<::pir::Operation*> ops,
                                     std::vector<ir::Expr> funcs) {
  for (const auto& func : funcs) {
    VLOG(4) << "OpInlineFusion: {FuncBody is} :" << func;
  }

  auto op_patterns = GetOpPatternVector(ops);

  PADDLE_ENFORCE_EQ(
      op_patterns.size(), funcs.size(), "ops and funcs size not equal");

  while (true) {
    VLOG(4) << "Start search for Injective + Injective";
    std::pair<int, int> idx_pair =
        SearchForAdjacentInjectives(op_patterns, funcs);
    int upper_stream = idx_pair.first, down_stream = idx_pair.second;
    if (upper_stream == NOT_FOUND) {
      VLOG(4) << "Not found Injective + Injective, break.";
      break;
    }
    VLOG(4) << "Find Injective + Injective" << upper_stream << " "
            << down_stream;
    ir::Expr func_body = TrivalFusion(funcs[upper_stream], funcs[down_stream]);

    // update
    auto update_funcs_and_op_patterns = [&]() {
      funcs[down_stream] = func_body;
      op_patterns[down_stream] = OpPatternKind::kInjective;

      funcs.erase(funcs.begin() + upper_stream);
      op_patterns.erase(op_patterns.begin() + upper_stream);
    };
    update_funcs_and_op_patterns();
  }
  return funcs;
}

std::vector<ir::LoweredFunc> OpLowererImpl::LowerGroup(
    const GroupPtr& group,
    bool apply_op_schedule,
    bool apply_group_schedule,
    ScheduleDetermineFunction schedule_determine_func) {
  // 1.Do compute, lower and schedule for each op.
  auto& ops = group->ops;
  if (ops.size() == 1 && ops[0]->name() == "custom_call") {
    return LowerCustomCall(group);
  }
  std::vector<ir::Tensor> group_func_arg_tensors;
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  // for some op, it will output more tmp value and regard as
  // XX_0, XX_1, so we log them in tmp_tensor_info;
  std::unordered_map<std::string, ir::Tensor> tmp_tensor_info;
  bool do_op_schedule = apply_group_schedule || apply_op_schedule;
  if (FLAGS_cinn_enable_map_expr) {
    return LowerMapExpr(group,
                        ops,
                        /*do_op_schedule=*/do_op_schedule,
                        /*apply_group_schedule=*/apply_group_schedule,
                        &group_func_arg_tensors,
                        &tensor_map);
  }

  std::vector<ir::Expr> func_bodies = LowerOps(group,
                                               ops,
                                               do_op_schedule,
                                               schedule_determine_func,
                                               &group_func_arg_tensors,
                                               &tensor_map,
                                               &tmp_tensor_info);

  func_bodies = OpInlineFusion(group, ops, func_bodies);

  std::unordered_set<::pir::Value> inner_genevalue;
  std::unordered_set<::pir::Operation*> ops_set(ops.begin(), ops.end());
  for (auto* op : ops) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      inner_genevalue.insert(op->result(i));
    }
  }

  std::unordered_set<::pir::Operation*> not_used_op;

  auto& align_info = group->alignment_schedule_info;

  for (auto op1 : ops) {
    auto it = align_info.find(op1);
    if (it == align_info.end()) {
      continue;
    }

    std::vector<int64_t> changed_axes;
    std::vector<int64_t> changed_factor;

    if (it->second.size() > 1) {
      for (auto& node : it->second) {
        std::cerr << "info " << node.DebugStr() << std::endl;
      }
      throw std::runtime_error("only suppopt one transform yet");
    }

    std::cerr << "it->second type " << it->second[0].type << std::endl;
    if (it->second[0].type == "broadcast") {
      // get broadcast op
      auto broadcast_axes = it->second[0].axis_info;
      auto output_shape = it->second[0].factor_info;

      std::cerr << "op name " << it->first->name() << std::endl;

      phi::DDim in_dim;

      if (it->first->name() == "cinn_op.reshape") {
        // TODO(phlrain): deal with reshape in a better way
        if (it->first->result(0).use_count() == 1 &&
            it->first->result(0).first_use().owner()->name() == "cf.yield") {
          std::cerr << "skip last reshape\n";
          continue;
        }
      }

      if ((it->first->name() != "cinn_op.reshape") &&
          (it->first->name() != "cinn_op.broadcast") &&
          (it->first->num_operands() == 1)) {
        in_dim = it->first->operand_source(0)
                     .type()
                     .dyn_cast<paddle::dialect::DenseTensorType>()
                     .dims();
      } else {
        in_dim = it->first->result(0)
                     .type()
                     .dyn_cast<paddle::dialect::DenseTensorType>()
                     .dims();
      }
      std::cerr << it->first->name() << "\t in dim " << in_dim << "\t"
                << it->second[0].DebugStr() << std::endl;

      if (in_dim.size() == 1u && in_dim[0] == 1u) {
        std::cerr << "!!!!!!!!!!!!!!!!!!!! " << output_shape.size()
                  << std::endl;
        for (size_t i = 0; i < output_shape.size(); ++i) {
          std::cerr << i << "    shape   " << output_shape[i] << std::endl;
          changed_axes.push_back(i);
          changed_factor.push_back(output_shape[i]);
        }
      } else if (in_dim.size() == broadcast_axes.size()) {
        for (size_t i = 0; i < broadcast_axes.size(); ++i) {
          if (in_dim[i] != output_shape[broadcast_axes[i]]) {
            if (in_dim[i] != 1) {
              throw std::runtime_error("Only support 1 - D broadcast ");
            }
            changed_axes.push_back(i);
            changed_factor.push_back(output_shape[broadcast_axes[i]]);
          }
        }
      } else {
        // only deal with broadcast axes
        std::set<int> axes_set;
        for (size_t i = 0; i < broadcast_axes.size(); ++i) {
          axes_set.insert(broadcast_axes[i]);
          if (in_dim[broadcast_axes[i]] != 1) {
            throw std::runtime_error("Only support 1 - D broadcast ");
          }

          changed_axes.push_back(broadcast_axes[i]);
          changed_factor.push_back(output_shape[broadcast_axes[i]]);
        }
      }

      if (changed_axes.size() == 0) {
        std::cerr << "opname " << it->first->name() << std::endl;
        throw std::runtime_error(" changed axes is zero");
      }
      cinn::ir::BroadcastInfo info{changed_axes, changed_factor};
      if (in_dim.size() == 1u && in_dim[0] == 1u) {
        info.full_broadcast = true;
      }

      for (size_t i = 0; i < it->first->num_operands(); ++i) {
        if (!align_info.count(it->first->operand_source(i).defining_op())) {
          std::cerr << "is first broadcast " << it->first->name() << std::endl;
          info.first_broadcast = true;
          break;
        }
      }

      auto op_out = it->first->result(0);
      std::cerr << "var name " << ValueName(op_out) << std::endl;
      info.op_name = it->first->name();
      broadcast_info[ValueName(op_out)] = info;

      // if( op_out.use_count() > 1 )
      // {
      //   throw std::runtime_error("only support ONE user for now");
      // }

      std::cerr << "op " << it->first->name() << std::endl;

      // std::cerr << "use op name " << op_out.first_use().owner()->name() <<
      // std::endl; std::cerr << "pattern kind " <<  CompatibleInfo::OpKind(
      // *(op_out.first_use().owner()) ) << std::endl;
      for (auto use_it = op_out.use_begin(); use_it != op_out.use_end();
           ++use_it) {
        if (use_it->owner()->name() == "cf.yield") {
          continue;
        }
        if (CompatibleInfo::OpKind(*(use_it->owner())) ==
            framework::kBroadcast) {
          std::cerr << "match broadcast !!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
          if (!info.full_broadcast) {
            broadcast_to_elementwise[ValueName(use_it->owner()->result(0))] =
                info;
          }
        }
      }
    } else {
      std::cerr << "type " << it->second[0].type << std::endl;
      throw std::runtime_error("only supportbroadcast type for now");
    }
  }

  for (auto& op : group->output_ops) {
    if (erase_reshape.count(op)) {
      copyed_var_names.insert(ValueName(op->operand_source(0)));
      continue;
    }
    // collect all output tensor.
    for (auto opresult : op->results()) {
      if (tensor_map.count(opresult) == 0) {
        continue;
      }
      auto tensor = tensor_map.at(opresult);

      if (opresult.use_count() > 1) {
        copyed_var_names.insert(tensor->name);

        if (broadcast_info.count(tensor->name)) {
          auto base_info = broadcast_info[tensor->name];
          base_info.with_constrain = true;
          broadcast_info[tensor->name + "_out"] = base_info;
        }
      } else {
        direct_output_var_names.insert(tensor->name);
      }
    }
  }

  for (size_t i = 0; i < func_bodies.size(); ++i) {
    // std::cerr << ops[i]->name() << std::endl;
    // std::cerr << "var name  " << ValueName(ops[i]->result(0)) <<
    std::cerr << "i " << i << "\n" << func_bodies[i] << std::endl;
  }

  // 2.Do group schedule.
  std::vector<Expr> added_expr;
  for (size_t i = 0; i < func_bodies.size(); ++i) {
    // std::cerr << ops[i]->name() << std::endl;
    // std::cerr << "var name  " << ValueName(ops[i]->result(0)) << std::endl;
    // std::cerr << "i " << i << "\n" << func_bodies[i] << std::endl;

    if (copyed_var_names.count(ValueName(remain_ops[i]->result(0)))) {
      auto tensor = tensor_map.at(remain_ops[i]->result(0));

      auto body = BuildOuputExpr(tensor);

      std::cerr << "oupput body  " << body << std::endl;

      added_expr.push_back(body);
    }
  }

  for (auto expr : added_expr) {
    // std::cerr << "added " << expr << std::endl;
    func_bodies.push_back(expr);
  }

  ir::ModuleExpr mod_expr(func_bodies);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  VLOG(3) << "After lower, ir is: \n" << ir_sch.GetModule().GetExprs().at(0);
  if (apply_group_schedule) {
    DoGroupSchedule(ir_sch, group, tensor_map, tmp_tensor_info);
    VLOG(3) << "After group schedule, ir is: \n"
            << ir_sch.GetModule().GetExprs().at(0);
  }

  // 3.Do post-processing,
  // including preparing function args and temporary variables,
  // applying low-level optimization passes, etc.
  std::vector<ir::Argument> group_func_args;
  return PostProcess(group,
                     tensor_map,
                     do_op_schedule,
                     {ir_sch.GetModule().GetExprs().at(0)},
                     &group_func_arg_tensors,
                     &group_func_args);
}

std::vector<ir::LoweredFunc> OpLowererImpl::LowerCustomCall(
    const GroupPtr& group) {
  auto& ops = group->ops;
  CHECK_EQ(ops.size(), 1);
  ::pir::Operation* op = ops[0];
  std::unordered_map<::pir::Value, ir::Tensor> tensor_map;
  std::vector<ir::Tensor> op_func_arg_tensors =
      CollectInputTensor(group, op, nullptr, &tensor_map);
  VLOG(4) << "inputs.size(): " << op_func_arg_tensors.size();

  std::vector<Type> out_types;
  std::vector<std::vector<int>> out_shapes;
  CollectOutputInfo(op, &out_types, &out_shapes, group);
  VLOG(4) << "out_types.size(): " << out_types.size();

  NodeAttr node_attrs = details::CollectAttrs(*op);

  auto& cinn_strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  const hlir::framework::Operator* cinn_op =
      Operator::Get(node_attrs.node_name);
  auto impl = OpStrategy::SelectImpl(cinn_strategy[cinn_op](
      node_attrs, op_func_arg_tensors, out_types, out_shapes, target_));

  // TODO(Arelius84): Support extern API
  std::string external_api;
  // if (node_attrs.attr_store.count("custom_call")) {
  //   external_api =
  //       absl::get<std::string>(node_attrs.attr_store.at("custom_call"));
  // } else {
  //   external_api = ExternalApiRegistry::Global()->GetExternalApi(node,
  //   target_);
  // }
  std::vector<cinn::common::CINNValue> compute_args = {
      cinn::common::CINNValue(group->FuncName()),
      cinn::common::CINNValue(external_api)};
  cinn::common::CINNValuePack pack =
      impl->fcompute(cinn::common::CINNValuePack{compute_args});
  CHECK_EQ(pack.size(), 1UL);
  // reset input names as extern api input args can't be remove duplicate.
  // group->input_names.clear();
  // for (auto& inode : node->inlinks_in_order()) {
  //   group->input_names.push_back(inode->source()->as<NodeData>()->id());
  // }
  return {pack[0].operator ir::Expr().as_lowered_func_ref()};
}

std::vector<ir::LoweredFunc> OpLowererImpl::PostProcess(
    const GroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    bool done_op_schedule,
    std::vector<ir::Expr> func_bodies,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::vector<ir::Argument>* group_func_args) {
  // 1.Prepare function args
  group->input_names.clear();
  std::unordered_set<std::string> arg_name_set;
  for (auto& arg_tensor : *group_func_arg_tensors) {
    // input data name.
    group->input_names.push_back(arg_tensor->name);
    // input args
    (*group_func_args)
        .emplace_back(arg_tensor->buffer, ir::Argument::IO::kInput);
    arg_name_set.insert(arg_tensor->buffer->name);
  }

  group->output_names.clear();
  // TODO(phlrain): output values not stable here
  for (auto& op : group->output_ops) {
    // collect all output tensor.
    for (auto opresult : op->results()) {
      if (tensor_map.count(opresult) == 0) {
        continue;
      }
      auto tensor = tensor_map.at(opresult);
      if (arg_name_set.count(tensor->buffer->name) != 0) {
        continue;
      }

      group->output_values.push_back(opresult);
      // output arg tensors

      // output args
      // group->output_names.push_back(tensor->name);
      // std::cerr << "tensor name   " << tensor->name << std::endl;
      // std::cerr << "base tensor " << tensor->buffer.defined() << std::endl;
      if (copyed_var_names.count(tensor->name)) {
        // std::cerr << "copyed var name \n";
        auto new_tensor = lang::CreatePlaceHolder(
            tensor->shape, tensor->type(), tensor->name + "_out");
        group_func_arg_tensors->push_back(new_tensor);
        group_func_args->emplace_back(new_tensor->buffer,
                                      ir::Argument::IO::kOutput);
        // std::cerr << "new tensor " << new_tensor->buffer.defined() <<
        // std::endl;
      } else if (erase_reshape.count(op)) {
        if (copyed_var_names.count(ValueName(op->operand_source(0)))) {
          // std::cerr << "rease copyed tensor" << std::endl;
          tensor = tensor_map.at(op->operand_source(0));
          auto new_tensor = lang::CreatePlaceHolder(
              tensor->shape, tensor->type(), tensor->name + "_out");
          group_func_arg_tensors->push_back(new_tensor);
          group_func_args->emplace_back(new_tensor->buffer,
                                        ir::Argument::IO::kOutput);
        } else {
          tensor = tensor_map.at(op->operand_source(0));
          group_func_arg_tensors->push_back(tensor);
          group_func_args->emplace_back(tensor->buffer,
                                        ir::Argument::IO::kOutput);
        }
      } else {
        group_func_arg_tensors->push_back(tensor);
        group_func_args->emplace_back(tensor->buffer,
                                      ir::Argument::IO::kOutput);
      }

      arg_name_set.insert(tensor->buffer->name);
    }
  }

  if (!done_op_schedule) {
    std::unordered_set<std::string> args_set;
    for (auto arg : (*group_func_args)) {
      args_set.insert(arg.name());
    }
    for (auto& op : group->ops) {
      // collect all output tensor.
      for (auto opresult : op->results()) {
        if (tensor_map.count(opresult) == 0) {
          continue;
        }
        auto tensor = tensor_map.at(opresult);
        if (args_set.count("_" + tensor->name) != 0) {
          continue;
        }
        group->output_values.push_back(opresult);
        group_func_arg_tensors->push_back(tensor);
        group->output_names.push_back(tensor->name);
        group_func_args->emplace_back(tensor->buffer,
                                      ir::Argument::IO::kOutput);
      }
    }
  }

  std::map<int, CINNKernelInfo::ArgDimIdx> mps;
  // update args for dynamic dim
  int num_tensor_args = static_cast<int>(group_func_args->size());
  int non_tensor_arg_idx = group_func_args->size();
  std::unordered_set<std::string> int_args_set;
  for (int tensor_arg_idx = 0; tensor_arg_idx < num_tensor_args;
       tensor_arg_idx++) {
    auto tensor_dim = (*group_func_arg_tensors)[tensor_arg_idx]->sym_shape;
    int tensor_dim_size = tensor_dim.size();
    for (int tensor_arg_dim_idx = 0; tensor_arg_dim_idx < tensor_dim_size;
         tensor_arg_dim_idx++) {
      if (tensor_dim[tensor_arg_dim_idx]->IsUniSymbolic()) {
        const std::string symbol_name =
            tensor_dim[tensor_arg_dim_idx]->ToString();
        if (int_args_set.count(symbol_name) != 0) {
          continue;
        }
        int_args_set.insert(symbol_name);
        group_func_args->emplace_back(
            ir::_Var_::Make(symbol_name, cinn::common::Int(64)));
        group->int_args_map[non_tensor_arg_idx++] = {tensor_arg_idx,
                                                     tensor_arg_dim_idx};
        VLOG(4) << "device kernel func's " << non_tensor_arg_idx << " is from "
                << tensor_arg_idx << ".shape(" << tensor_arg_dim_idx << ")";
      }
    }
  }

  std::vector<ir::LoweredFunc> lowered_funcs;
  for (ir::Expr func_body : func_bodies) {
#ifdef CINN_WITH_CUDA
    optim::OptimizeExprGPU(&(func_body));
#endif
    // std::cerr << "fun body " << func_body << std::endl;

    // 2.Prepare temp buffers
    auto temp_buffers =
        lang::GetTempBuffers(*group_func_arg_tensors, func_body);
    // 3.Building LoweredFunc
    auto func = ir::_LoweredFunc_::Make(
        group->FuncName(), *group_func_args, func_body, temp_buffers);
    if (!done_op_schedule) {
      func->PrepareBufferCastExprs();
    }
    // 4.Apply low level pass
    func = optim::Optimize(Expr(func), target_, false).as_lowered_func_ref();
    lowered_funcs.push_back(std::move(func));
  }

  return lowered_funcs;
}

std::vector<ir::Expr> OpLowererImpl::LowerOps(
    const GroupPtr& group,
    const std::vector<::pir::Operation*>& ops,
    bool apply_op_schedule,
    ScheduleDetermineFunction schedule_determine_func,
    std::vector<ir::Tensor>* group_func_arg_tensors,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map,
    std::unordered_map<std::string, ir::Tensor>* tmp_tensor_info) {
  auto& strategy = Operator::GetAttrs<StrategyFunction>("CINNStrategy");
  std::vector<Expr> func_bodies;
  std::unordered_set<::pir::Value> inner_used_value;
  for (auto* op : ops) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      inner_used_value.insert(op->operand_source(i));
    }
  }

  std::unordered_set<::pir::Operation*> not_used_op;
  for (auto* op : ops) {
    bool used = false;
    for (size_t i = 0; i < op->num_results(); ++i) {
      if (inner_used_value.count(op->result(i))) {
        used = true;
        break;
      }
    }

    if (!used) {
      not_used_op.insert(op);
    }
  }

  for (auto* op : ops) {
    std::cerr << "op name " << op->name() << std::endl;
    VLOG(4) << "start lowering op:" << op->name();
    // 1.Select Op impl
    std::vector<ir::Tensor> op_func_arg_tensors =
        CollectInputTensor(group, op, group_func_arg_tensors, tensor_map);
    VLOG(4) << "input size:" << op_func_arg_tensors.size();

    std::string cinn_op_name = CompatibleInfo::OpName(*op);
    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    std::shared_ptr<OpImpl> op_impl = nullptr;
    if (FLAGS_cinn_bucket_compile) {
      std::vector<Type> out_types;
      std::vector<std::vector<ir::Dim>> out_shapes;
      CollectOutputInfo(op, &out_types, &out_shapes, group);
      CHECK_EQ(out_types.size(), out_shapes.size());
      VLOG(4) << "out_types.size(): " << out_types.size();
      NodeAttr node_attrs = details::CollectAttrs(*op);
      auto& strategy_map =
          Operator::GetAttrs<StrategyFunctionSymbolic>("CINNStrategySymbolic");
      StrategyFunctionSymbolic strategy = strategy_map[cinn_op];
      CHECK(static_cast<bool>(strategy))
          << " cinn_op_name: " << cinn_op_name
          << "has no CINNStrategySymbolic registered.";
      op_impl = OpStrategy::SelectImpl(strategy(node_attrs,
                                                op_func_arg_tensors,
                                                out_types,
                                                out_shapes,
                                                this->target_));
    } else {
      std::vector<Type> out_types;
      std::vector<std::vector<int>> out_shapes;
      CollectOutputInfo(op, &out_types, &out_shapes, group);
      VLOG(4) << "out_types.size(): " << out_types.size();
      NodeAttr node_attrs = details::CollectAttrs(*op);
      op_impl = OpStrategy::SelectImpl(strategy[cinn_op](node_attrs,
                                                         op_func_arg_tensors,
                                                         out_types,
                                                         out_shapes,
                                                         this->target_));
    }
    // 2.Perform the lower process of Op
    std::vector<ir::LoweredFunc> funcs = DoOpLower(
        op_impl, op, tensor_map, tmp_tensor_info, &op_func_arg_tensors);

    // if (apply_op_schedule && (this->*schedule_determine_func)(op)) {
    //   // 3.Perform the schedule of Op
    //   func_bodies.push_back(DoOpSchedule(op_impl, op_func_arg_tensors,
    //   funcs));
    // } else
    {
      if (ops.size() > 1 && not_used_op.count(op) &&
          (op->name() == "cinn_op.reshape")) {
        // copyed_var_names.insert( ValueName( op->operand_source(0)));
        erase_reshape.insert(op);
        // auto it = group->output_ops.find( op );
        // if(it != group->output_ops.end() )
        // {
        //   group->output_ops.erase( it );
        // }

        // group->output_ops.insert(
        // op->operand_source(0).dyn_cast<::pir::OpResult>().owner() );
        continue;
      }

      for (const ir::LoweredFunc& func : funcs) {
        func_bodies.push_back(func->body);
      }

      remain_ops.push_back(op);
    }
  }

  VLOG(4) << "group_func_arg_tensors.size(): "
          << group_func_arg_tensors->size();

  return func_bodies;
}

std::vector<ir::LoweredFunc> OpLowererImpl::DoOpLower(
    std::shared_ptr<hlir::framework::OpImpl> op_impl,
    ::pir::Operation* op,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map,
    std::unordered_map<std::string, ir::Tensor>* tmp_tensor_info,
    std::vector<ir::Tensor>* op_func_arg_tensors) {
  VLOG(4) << "Do lower with Compute, op: " << op->name();
  std::vector<cinn::common::CINNValue> cinn_inputs;
  for (const ir::Tensor& tensor : *op_func_arg_tensors) {
    cinn_inputs.push_back(cinn::common::CINNValue(ir::Expr(tensor)));
  }

  // set tensor name = operand hash name
  auto op_results = op->results();
  for (const auto& result : op_results) {
    std::string output_id = ValueName(result);
    cinn_inputs.push_back(cinn::common::CINNValue(output_id));
  }

  // 1.Do compute
  cinn::common::CINNValuePack pack =
      op_impl->fcompute(cinn::common::CINNValuePack{cinn_inputs});

  poly::StageMap tmp_stages = pack.back();
  std::string post = "";
  for (int idx = 0; idx < pack.size() - 1; ++idx) {
    Expr expr = pack[idx];
    // Insert the output tensor defined by Compute into the tensor_map
    if (pack.size() - 1 > op_results.size()) {
      // Some op may output multiple temp tensors in their Compute
      // definition, but only one output  in the graph, and we use id +
      // "_0"/"_1" as key.
      if (idx < op_results.size()) {
        (*tensor_map)[op_results[idx]] = expr.as_tensor_ref();
      }
      std::string tensor_name = ValueName(op_results[0]) + post;
      VLOG(3) << "Add tmp tensor name for reducer op: " << tensor_name;
      (*tmp_tensor_info)[tensor_name] = expr.as_tensor_ref();
      post = "_" + std::to_string(idx);
    } else {
      // If the number of output tensors defined by Compute is less equal than
      // the output node_data on the graph, then there is a one-to-one
      // correspondence, and the redundant output node_data contact empty.
      (*tensor_map)[op_results[idx]] = expr.as_tensor_ref();
    }

    // Insert output tensors into function arg
    if (!expr.as_tensor_ref()->buffer.defined() ||
        this->target_ != cinn::common::DefaultNVGPUTarget()) {
      op_func_arg_tensors->push_back(expr.as_tensor_ref());
      expr.as_tensor_ref()->WithBuffer();
    }
  }

  VLOG(4) << "op_func_arg_tensors.size(): " << op_func_arg_tensors->size();

  // 2.Do lower
  std::string lower_fn_name = CompatibleInfo::OpFuncName(*op);
  ast_gen_ius::TensorGroup tensor_group =
      ast_gen_ius::ConvertStageMapToTensorGroup(tmp_stages);
  std::vector<ir::LoweredFunc> funcs = lang::LowerToAstVec(
      lower_fn_name, *op_func_arg_tensors, {&tensor_group}, this->target_);
  VLOG(4) << "Lower op: " << lower_fn_name << ", get " << funcs.size()
          << " LoweredFunc:\n";
  if (VLOG_IS_ON(4)) {
    for (auto fun : funcs) {
      VLOG(4) << fun;
    }
  }

  op_func_arg_tensors->clear();
  for (int idx = 0; idx < pack.size() - 1; ++idx) {
    CHECK(pack[idx].is_tensor());
    op_func_arg_tensors->push_back(
        pack[idx].operator ir::Expr().as_tensor_ref());
  }

  return funcs;
}

ir::Expr OpLowererImpl::DoOpSchedule(
    std::shared_ptr<hlir::framework::OpImpl> op_impl,
    const std::vector<ir::Tensor>& op_func_arg_tensors,
    const std::vector<ir::LoweredFunc>& lowered_funcs) {
  VLOG(4) << "Do op schedule";
  std::vector<cinn::common::CINNValue> schedule_inputs;
  // 1.Collect tensors
  for (const ir::Tensor& op_func_arg_tensor : op_func_arg_tensors) {
    schedule_inputs.push_back(cinn::common::CINNValue(op_func_arg_tensor));
  }
  // 2.Collect bodies to be scheduled
  for (const ir::LoweredFunc& func : lowered_funcs) {
    schedule_inputs.push_back(cinn::common::CINNValue(func->body));
  }
  // 3.Do schedule on AST
  cinn::common::CINNValuePack expr_pack =
      op_impl->fschedule(cinn::common::CINNValuePack{schedule_inputs});
  VLOG(4) << "After op schedule: " << expr_pack[0].operator ir::Expr();

  return expr_pack[0].operator ir::Expr();
}

ir::Expr OpLowererImpl::DoGroupSchedule(
    ir::IRSchedule& ir_sch,
    const GroupPtr& group,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map,
    const std::unordered_map<std::string, ir::Tensor>& tmp_tensor_info) {
  VLOG(3) << "using StaticShapeGroupScheduler to schedule group.";
  std::cerr << "!!!!!!!!!!!!!!!!!!!!group op kind" << group->op_pattern_kind
            << std::endl;

  std::cerr << "group id " << group->group_id << std::endl;
  std::cerr << "type " << group->op_pattern_kind << std::endl;

  std::cerr << "reduce axis ";
  for (auto d : group->reduce_axis) {
    std::cerr << " " << d;
  }
  std::cerr << std::endl;

  std::cerr << "loop range ";
  for (auto d : group->loop_ranges) {
    std::cerr << " " << d;
  }
  std::cerr << std::endl;

  auto group_tile_info = GetGroupTileInfo(group);

  std::unordered_set<std::string> output_tensor_names;
  std::transform(
      group->output_ops.begin(),
      group->output_ops.end(),
      std::inserter(output_tensor_names, output_tensor_names.begin()),
      [&](::pir::Operation* op) {
        if (erase_reshape.count(op)) {
          return ValueName(op->operand_source(0)) + "_out";
        }
        return ValueName(op->result(0)) + "_out";
      });

  std::unique_ptr<ir::GroupScheduler> group_scheduler =
      ir::GroupScheduler::Make(&ir_sch,
                               output_tensor_names,
                               target_,
                               /* is_dy_shape = */ false,
                               group_tile_info);
  group_scheduler->Schedule();
  return ir_sch.GetModule().GetExprs().at(0);
}

ir::Tensor OpLowererImpl::GetTensor(const GroupPtr& group,
                                    const ::pir::Value& value) {
  auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto dtype = type_info.dtype();
  std::string input_id = ValueName(value);

  auto ForEachDimExpr = [&](const auto& DoEach) {
    const auto& dims = type_info.dims();
    if (::common::contain_unknown_dim(dims)) {  // dynamic shape
      const auto& sym_vec = group->GetShapeOrDataExprs(value).shape();
      for (const auto& dim_expr : sym_vec) {
        DoEach(dim_expr);
      }
    } else {  // static shape
      for (int i = 0; i < dims.size(); ++i) {
        DoEach(::symbol::DimExpr{dims[i]});
      }
    }
  };
  if (FLAGS_cinn_bucket_compile) {
    std::vector<ir::Dim> sym_shape;
    ForEachDimExpr(
        [&](const auto& sym) { sym_shape.emplace_back(input_id, sym); });
    return lang::CreatePlaceHolder(
        sym_shape, CompatibleInfo::ConvertIRType(dtype), input_id);
  } else {
    return lang::CreatePlaceHolder(::common::vectorize<int>(type_info.dims()),
                                   CompatibleInfo::ConvertIRType(dtype),
                                   input_id);
  }
}

std::vector<ir::Tensor> OpLowererImpl::CollectInputTensor(
    const GroupPtr& group,
    const ::pir::Operation* op,
    std::vector<ir::Tensor>* func_args,
    std::unordered_map<::pir::Value, ir::Tensor>* tensor_map) {
  std::vector<ir::Tensor> tensors;
  for (auto in_value : CompatibleInfo::RealOperandSources(*op)) {
    VLOG(4) << "input tensor name: " << ValueName(in_value);
    ir::Tensor tensor = GetTensor(group, in_value);
    VLOG(4) << "shape: " << tensor->shape;
    VLOG(4) << "sym_shape: " << tensor->sym_shape;

    if (!tensor_map->count(in_value)) {
      // record tensor.
      (*tensor_map)[in_value] = tensor;
      // record func input args
      if (func_args != nullptr) {
        func_args->push_back(tensor);
      }
    } else {
      // TODO(6clc): After supporting symbolic calculation,
      // 1. Check that the shape of the tensor with the same name is the same
      // size
      // 2. Or make the symbol expression in compute output tensor consistent
      //    with the one inferred in shape_analysis
      (*tensor_map)[in_value]->sym_shape = tensor->sym_shape;
      (*tensor_map)[in_value]->shape = tensor->shape;
      (*tensor_map)[in_value]->sym_domain = tensor->sym_domain;
      (*tensor_map)[in_value]->domain = tensor->domain;
    }
    tensors.push_back(tensor);
  }
  return tensors;
}

void OpLowererImpl::CollectOutputInfo(::pir::Operation* op,
                                      std::vector<Type>* out_types,
                                      std::vector<std::vector<int>>* out_shapes,
                                      const GroupPtr& group) {
  auto op_results = op->results();
  for (auto& out_value : op_results) {
    std::string output_id = ValueName(out_value);

    auto type_info =
        out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

    out_types->push_back(CompatibleInfo::ConvertIRType(type_info.dtype()));
    auto out_shape = ::common::vectorize<int>(type_info.dims());
    out_shapes->push_back(std::move(out_shape));
  }
}

void OpLowererImpl::CollectOutputInfo(
    ::pir::Operation* op,
    std::vector<Type>* out_types,
    std::vector<std::vector<ir::Dim>>* out_shapes,
    const GroupPtr& group) {
  auto op_results = op->results();
  for (auto& out_value : op_results) {
    std::string output_id = ValueName(out_value);

    auto type_info =
        out_value.type().dyn_cast<paddle::dialect::DenseTensorType>();

    out_types->push_back(CompatibleInfo::ConvertIRType(type_info.dtype()));

    auto ForEachDimExpr = [&](const auto& DoEach) {
      const auto& dims = type_info.dims();
      if (::common::contain_unknown_dim(dims)) {  // dynamic shape
        const auto& sym_vec = group->GetShapeOrDataExprs(out_value).shape();
        std::vector<ir::Dim> sym_shape;
        for (const auto& sym : sym_vec) {
          DoEach(sym);
        }
      } else {  // static shape
        auto out_shape = ::common::vectorize<int64_t>(dims);
        for (int64_t dim : out_shape) {
          DoEach(symbol::DimExpr{dim});
        }
      }
    };
    std::vector<ir::Dim> sym_shape;
    ForEachDimExpr(
        [&](const auto& sym) { sym_shape.emplace_back(output_id, sym); });
    out_shapes->emplace_back(std::move(sym_shape));
  }
}

std::string OpLowererImpl::ValueName(::pir::Value value) {
  auto name = name_gene_->GetOrNew(value, CompatibleInfo::kNamePrefix);

  return name;
}

common::Type OpLowererImpl::GetTensorDtype(
    const std::string& name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  for (auto iter : tensor_map) {
    if (name == ValueName(iter.first)) {
      return GetTensorDtype(iter.first);
    }
  }
  VLOG(4) << name << " is not in tensor map, return FP32 by default.";
  return common::F32();
}

common::Type OpLowererImpl::GetTensorDtype(const ::pir::Value& value) {
  auto type_info = value.type().dyn_cast<paddle::dialect::DenseTensorType>();
  auto in_shape = ::common::vectorize<int>(type_info.dims());
  auto dtype = type_info.dtype();
  return CompatibleInfo::ConvertIRType(dtype);
}

bool OpLowererImpl::IsInTensorMap(
    const std::string& name,
    const std::unordered_map<::pir::Value, ir::Tensor>& tensor_map) {
  for (auto iter : tensor_map) {
    if (name == ValueName(iter.first)) {
      return true;
    }
  }
  return false;
}

ir::LoweredFunc OpLowererImpl::GenerateInferShapeFunc(
    const GroupPtr& group,
    const std::vector<ir::Tensor> group_func_arg_tensors,
    const std::vector<ir::Argument> group_func_args) {
  // CHECK_EQ(group_func_arg_tensors.size(), group_func_args.size());
  std::vector<ir::Expr> ir_bodys;
  int output_tensor_idx = 0;
  for (int tensor_arg_idx = 0; tensor_arg_idx < group_func_arg_tensors.size();
       ++tensor_arg_idx) {
    if (group_func_args[tensor_arg_idx].is_input()) {
      continue;
    }
    auto tensor_dim = group_func_arg_tensors[tensor_arg_idx]->sym_shape;
    int tensor_dim_size = tensor_dim.size();
    auto tensor_shape = group_func_arg_tensors[tensor_arg_idx]->shape;

    ir::Var tensor_shape_args(TENSOR_SHAPE_ARGS, type_of<int64_t**>());
    for (int i = 0; i < tensor_shape.size(); i++) {
      ir::Expr call_set_infer_shape_value =
          ir::Call::Make(type_of<void>(),
                         runtime::intrinsic::infer_shape_set_value,
                         {ir::Expr(output_tensor_idx),
                          ir::Expr(i),
                          tensor_shape[i],
                          tensor_shape_args},
                         {},
                         ir::CallType::Extern,
                         ir::FunctionRef(),
                         0);
      ir_bodys.push_back(call_set_infer_shape_value);
    }
    ++output_tensor_idx;
  }
  ir::LoweredFunc infer_shape_func =
      ir::_LoweredFunc_::Make(group->FuncName() + "_infer_shape",
                              group_func_args,
                              ir::Block::Make(ir_bodys),
                              {});
  return infer_shape_func;
}

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
