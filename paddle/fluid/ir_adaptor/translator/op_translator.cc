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

#include "paddle/fluid/ir_adaptor/translator/op_translator.h"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/fluid/ir_adaptor/translator/attribute_translator.h"
#include "paddle/fluid/ir_adaptor/translator/op_compat_info.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/fluid/ir_adaptor/translator/type_translator.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/value.h"

// NOTE(zhangbo9674): File pd_op.h is generated by op_gen.py, see details in
// paddle/fluid/ir/dialect/CMakeLists.txt.
#include "paddle/fluid/ir/dialect/pd_op.h"

namespace paddle {
namespace translator {

namespace {

using ResultIdx = size_t;
using OpDesc = paddle::framework::OpDesc;
using BlockDesc = paddle::framework::BlockDesc;
using VarDesc = paddle::framework::VarDesc;
using OpOutputTypeList = std::vector<ir::Type>;
using OpOutputMapping = std::unordered_map<std::string, ResultIdx>;
using OpInputInfo = dialect::OpInputInfo;
using OpInputInfoList = std::vector<dialect::OpInputInfo>;
using OpAttributeInfo = dialect::OpAttributeInfo;
using OpAttributeInfoList = std::vector<dialect::OpAttributeInfo>;
using OpOutputInfo = dialect::OpOutputInfo;
using OpOutputInfoList = std::vector<dialect::OpOutputInfo>;
using InputHandleFn = std::function<ir::OpResult(ir::IrContext*,
                                                 TranslationContext*,
                                                 const OpDesc&,
                                                 const std::string&,
                                                 const OpInputInfo&,
                                                 ir::Program*)>;
constexpr char kTargetDialectPrefix[] = "pd.";
constexpr char kEmptyVarName[] = "@EMPTY@";

static const std::unordered_set<std::string> special_inplace_ops = {
    "batch_norm",
};

inline bool IsInplace(const OpDesc& op_desc) {
  bool inplace = false;
  if (special_inplace_ops.count(op_desc.Type())) {
    return inplace;
  }
  auto input_names = op_desc.InputArgumentNames();
  auto output_names = op_desc.OutputArgumentNames();
  if (input_names.size() == 0 || output_names.size() == 0) {
    return inplace;
  }

  std::vector<std::string> name_intersection;
  std::sort(input_names.begin(), input_names.end());
  std::sort(output_names.begin(), output_names.end());
  std::set_intersection(input_names.begin(),
                        input_names.end(),
                        output_names.begin(),
                        output_names.end(),
                        std::back_inserter(name_intersection));

  if (name_intersection.size() > 0) {
    std::string redundant_variables = std::accumulate(
        std::next(name_intersection.begin()),
        name_intersection.end(),
        name_intersection[0],
        [](std::string a, std::string b) { return a + "," + b; });
    VLOG(4) << "Following variables occur both in inputs and outputs: "
            << redundant_variables;
    return true;
  }

  return inplace;
}

inline std::string OpNameCompatibleMapping(std::string op_name) {
  auto& op_normalizer = OpNameNormalizer::instance();
  return op_normalizer[op_name];
}

inline ir::Operation* InsertSliceOperationForTarget(
    ir::IrContext* ctx,
    TranslationContext* param_map,
    ir::Program* program,
    const VariableDefiningInfo& defining_info,
    const std::string& arg_name) {
  std::string slice_op_name(ir::SliceOp::name());
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(slice_op_name);
  std::unordered_map<std::string, ir::Attribute> op_attribute_map = {
      {"index", ir::Int32Attribute::get(ctx, defining_info.idx_in_vector)},
  };
  ir::VectorType src_vec_type =
      defining_info.value.type().dyn_cast<ir::VectorType>();
  ir::Operation* operation =
      ir::Operation::Create({defining_info.value},
                            op_attribute_map,
                            {src_vec_type[defining_info.idx_in_vector]},
                            op_info);
  program->block()->push_back(operation);
  ir::OpResult target_op_result = operation->result(0);
  (*param_map)[arg_name] = VariableDefiningInfo(target_op_result);
  return operation;
}

inline ir::Operation* InsertCombineOperationForTarget(
    ir::IrContext* ctx,
    TranslationContext* param_map,
    ir::Program* program,
    const std::vector<std::string>& args) {
  std::string combine_op_name(ir::CombineOp::name());
  ir::OpInfo op_info = ctx->GetRegisteredOpInfo(combine_op_name);

  std::vector<ir::OpResult> src_values;
  std::vector<ir::Type> types_in_vec;
  for (const auto& arg_name : args) {
    auto defining_info = param_map->at(arg_name);
    src_values.push_back(defining_info.value);
    types_in_vec.push_back(defining_info.value.type());
  }
  ir::Type target_vec_type = ir::VectorType::get(ctx, types_in_vec);
  ir::Operation* operation =
      ir::Operation::Create(src_values, {}, {target_vec_type}, op_info);
  program->block()->push_back(operation);
  return operation;
}

inline ir::Operation* InsertFullOperationForAttributeInput(ir::IrContext* ctx,
                                                           ir::Program* program,
                                                           ir::Attribute attr) {
  float data = 0.0f;
  phi::DataType dtype = phi::DataType::UNDEFINED;
  if (attr.isa<ir::FloatAttribute>()) {
    data = attr.dyn_cast<ir::FloatAttribute>().data();
    dtype = phi::DataType::FLOAT32;
  } else if (attr.isa<ir::DoubleAttribute>()) {
    data = static_cast<float>(attr.dyn_cast<ir::DoubleAttribute>().data());
    dtype = phi::DataType::FLOAT64;
  } else if (attr.isa<ir::Int32Attribute>()) {
    data = static_cast<float>(attr.dyn_cast<ir::Int32Attribute>().data());
    dtype = phi::DataType::INT32;
  } else if (attr.isa<ir::Int64Attribute>()) {
    data = static_cast<float>(attr.dyn_cast<ir::Int64Attribute>().data());
    dtype = phi::DataType::INT64;
  } else if (attr.isa<ir::BoolAttribute>()) {
    data = static_cast<float>(attr.dyn_cast<ir::BoolAttribute>().data());
    dtype = phi::DataType::BOOL;
  }
  ir::Builder builder(ctx, program->block());
  dialect::FullOp full_op = builder.Build<dialect::FullOp>(
      std::vector<int64_t>{1}, data, dtype, phi::CPUPlace());

  return full_op.operation();
}

inline ir::Operation* InsertFullArrayOperationForAttributeInput(
    ir::IrContext* ctx, ir::Program* program, ir::Attribute attr) {
  IR_ENFORCE(attr.isa<dialect::IntArrayAttribute>(),
             "Encounter non IntArray type when trying to insert IntArray "
             "mutable attribute");

  phi::IntArray int_array = attr.dyn_cast<dialect::IntArrayAttribute>().data();

  ir::Builder builder(ctx, program->block());
  dialect::FullIntArrayOp full_int_array_op =
      builder.Build<dialect::FullIntArrayOp>(
          int_array.GetData(), phi::DataType::INT64, phi::CPUPlace());
  return full_int_array_op.operation();
}

inline ir::OpResult GetAttributeAsInput(ir::IrContext* ctx,
                                        ir::Program* program,
                                        const OpDesc& op_desc,
                                        const OpInputInfo& input_info) {
  auto& attribute_translator = AttributeTranslator::instance();
  auto& op_normalizer = OpNameNormalizer::instance();

  auto legacy_attr_name =
      op_normalizer.GetLegacyAttrName(op_desc.Type(), input_info.name);

  if (!op_desc.HasAttr(legacy_attr_name)) {
    IR_THROW("Op %s arg %s should not be zero size",
             op_desc.Type(),
             legacy_attr_name);
  }
  paddle::framework::Attribute legacy_attr = op_desc.GetAttr(legacy_attr_name);
  VLOG(10) << "[" << op_desc.Type() << "][attribute]"
           << " name: " << legacy_attr_name << " " << legacy_attr.index();
  ir::Attribute new_attr =
      attribute_translator(input_info.type_name, legacy_attr);

  ir::Operation* defining_op = nullptr;
  bool is_int_array = (input_info.type_name.find("IntArrayAttribute") !=
                       input_info.type_name.npos);
  if (is_int_array) {
    defining_op =
        InsertFullArrayOperationForAttributeInput(ctx, program, new_attr);
  } else {
    defining_op = InsertFullOperationForAttributeInput(ctx, program, new_attr);
  }

  return defining_op->result(0);
}

}  // namespace

/// @brief This class is used to translate a OpDesc, it's a functor class and
/// should have no non-static data member, since we expected it's stateless.
struct OpTranscriber {
 public:
  virtual ~OpTranscriber() = default;

 public:
  virtual ir::Operation* operator()(ir::IrContext* ctx,
                                    TranslationContext* param_map,
                                    const OpDesc& op_desc,
                                    ir::Program* program);

 public:
  virtual ir::OpInfo LoopkUpOpInfo(ir::IrContext* ctx, const OpDesc& op_desc);
  virtual std::vector<ir::OpResult> GenerateOperationInput(
      ir::IrContext* ctx,
      TranslationContext* param_map,
      const OpDesc& op_desc,
      const std::string& normalized_op_name,
      const OpInputInfoList& input_infos,
      ir::Program* program);
  virtual std::tuple<OpOutputTypeList, OpOutputMapping> GenerateOperationOutput(
      ir::IrContext* ctx,
      const OpDesc& op_desc,
      const OpOutputInfoList& output_infos);
  virtual void HandleNonexistentAttribute(ir::IrContext*,
                                          ir::AttributeMap* attribute_map,
                                          const OpAttributeInfo& info) {
    auto& attribute_translator = AttributeTranslator::instance();
    (*attribute_map)[info.name] =
        attribute_translator(info.type_name, paddle::framework::Attribute());
  }
  virtual ir::AttributeMap TranslateOpAttribute(
      ir::IrContext* ctx,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc);

  virtual void RecordOpResultMapping(TranslationContext* param_map,
                                     const OpDesc& op_desc,
                                     ir::Operation* operation,
                                     const OpOutputMapping& arg_to_idx);

 public:
  virtual InputHandleFn GetSpecialInputHandlers(std::string input_name) {
    return nullptr;
  }
};

ir::OpInfo OpTranscriber::LoopkUpOpInfo(ir::IrContext* ctx,
                                        const OpDesc& op_desc) {
  std::string target_op_name =
      kTargetDialectPrefix + OpNameCompatibleMapping(op_desc.Type());
  if (IsInplace(op_desc)) {
    target_op_name += "_";
  }
  VLOG(6) << "[op name normalizing: " << op_desc.Type() << " to "
          << target_op_name;
  auto op_info = ctx->GetRegisteredOpInfo(target_op_name);
  if (!op_info) {
    IR_THROW("Op %d should have corresponding OpInfo %d",
             op_desc.Type(),
             target_op_name);
  }

  return op_info;
}

std::vector<ir::OpResult> OpTranscriber::GenerateOperationInput(
    ir::IrContext* ctx,
    TranslationContext* param_map,
    const OpDesc& op_desc,
    const std::string& normalized_op_name,
    const OpInputInfoList& input_infos,
    ir::Program* program) {
  VLOG(10) << "[op:" << op_desc.Type() << "][input] entrance";

  auto& op_normalizer = OpNameNormalizer::instance();
  const auto* mutable_attributes =
      op_normalizer.GetMutableAttributes(op_desc.Type());

  std::set<std::string> yaml_input_set;
  for (const auto& info : input_infos) {
    if (auto special_handler = this->GetSpecialInputHandlers(info.name)) {
      continue;
    }

    std::string legacy_input_name =
        op_normalizer.GetLegacyArgName(op_desc.Type(), info.name);

    yaml_input_set.insert(legacy_input_name);
  }
  // scan all inputs to see if any of them is generated as a vector<Tensor>
  // so need an additional `SliceOp` to take it out.
  for (const auto& n : op_desc.Inputs()) {
    auto& name = n.first;
    auto& args = n.second;

    for (const auto& arg_name : args) {
      bool check =
          param_map->count(arg_name) != 0 || !yaml_input_set.count(arg_name);
      IR_ENFORCE(check,
                 "arg %s.%s as input should be exists before prasing %s",
                 name,
                 arg_name,
                 op_desc.Type());
      auto defining_info = (*param_map)[arg_name];
      if (defining_info.generated_by_vector) {
        InsertSliceOperationForTarget(
            ctx, param_map, program, defining_info, arg_name);
      }
    }
  }

  VLOG(10) << "[op:" << op_desc.Type() << "][input] start";

  std::vector<ir::OpResult> op_inputs;

  for (const auto& info : input_infos) {
    if (auto special_handler = this->GetSpecialInputHandlers(info.name)) {
      ir::OpResult ret = special_handler(
          ctx, param_map, op_desc, normalized_op_name, info, program);
      op_inputs.push_back(ret);
      continue;
    }

    std::string legacy_input_name =
        op_normalizer.GetLegacyArgName(op_desc.Type(), info.name);

    VLOG(10) << "[op:" << op_desc.Type() << "][input]" << info.name << " "
             << legacy_input_name;

    std::vector<std::string> legacy_input_vars;
    // return empty OpResult if this arg is optional and not shown in OpDesc
    // TODO(lyk): HasInput doesnot consider variadic attribute
    if (op_desc.HasInput(legacy_input_name, true)) {
      legacy_input_vars = op_desc.Input(legacy_input_name, true);
    }

    if (legacy_input_vars.size() == 0) {
      if (info.optional) {
        op_inputs.push_back(ir::OpResult(nullptr));
        continue;
      }
    }

    VLOG(10) << "[op:" << op_desc.Type() << "][input]" << info.name << " "
             << legacy_input_name << " " << legacy_input_vars.size();

    if (legacy_input_vars.size() == 0 && mutable_attributes != nullptr &&
        mutable_attributes->count(info.name) != 0) {
      const auto& candidate_var_names =
          op_normalizer.GetMutableAttributeInfos(op_desc.Type(), info.name);
      bool found_candidate_var = false;
      for (const auto& var_name : candidate_var_names) {
        VLOG(10) << "[handle mutable attribute][" << info.name << "]["
                 << var_name << "]";
        if (op_desc.HasInput(var_name)) {
          legacy_input_vars = op_desc.Input(var_name, true);
          if (legacy_input_vars.size() == 0) continue;
          found_candidate_var = true;
          break;
        }
      }

      if (!found_candidate_var) {
        auto attribute_input = GetAttributeAsInput(ctx, program, op_desc, info);
        op_inputs.push_back(attribute_input);
        continue;
      }
    }

    bool is_vector = (info.type_name.find("VectorType") != std::string::npos);
    is_vector |=
        (info.type_name.find("IntArrayAttribute") != std::string::npos);
    VLOG(10) << "[op:" << op_desc.Type() << "][input]" << info.name << " "
             << is_vector << " " << info.type_name;

    // Specially process TensorArray, this because we cannot distinguish it with
    // Vector<DenseTensor> by other conditions but we cannot support it like
    // Vector<DenseTensor>
    if (legacy_input_vars.size() == 1) {
      VarDesc* var = op_desc.Block()->FindVarRecursive(legacy_input_vars[0]);
      if (var->GetType() ==
          paddle::framework::proto::VarType::LOD_TENSOR_ARRAY) {
        is_vector = false;
      }
    }

    // if src type is Tensor
    if (!is_vector) {
      auto defining_info = (*param_map)[legacy_input_vars[0]];
      op_inputs.push_back(defining_info.value);

      // if src type is Vector<Tesnor> , need an additional `CombineOp` to
      // assemble them.
    } else {
      auto* combine_op = InsertCombineOperationForTarget(
          ctx, param_map, program, legacy_input_vars);
      op_inputs.push_back(combine_op->result(0));
    }
  }

  return op_inputs;
}

std::tuple<OpOutputTypeList, OpOutputMapping>
OpTranscriber::GenerateOperationOutput(ir::IrContext* ctx,
                                       const OpDesc& op_desc,
                                       const OpOutputInfoList& output_infos) {
  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types = {};

  auto& type_translator = TypeTranslator::instance();
  auto& op_normalizer = OpNameNormalizer::instance();

  const BlockDesc* block = op_desc.Block();

  for (const auto& info : output_infos) {
    size_t cur_output_idx = op_output_types.size();
    std::string legacy_output_name =
        op_normalizer.GetLegacyArgName(op_desc.Type(), info.name);

    // return empty type if this arg is optional and not shown in OpDesc
    if (!op_desc.HasOutput(legacy_output_name)) {
      VLOG(10) << "[output translating]"
               << "[" << op_desc.Type() << "] optional " << info.name << " :"
               << info.type_name << " " << legacy_output_name;
      IR_ENFORCE(info.optional,
                 "Op %s arg %s should be optional if it can be empty",
                 op_desc.Type(),
                 legacy_output_name);
      op_output_types.push_back(ir::Type(nullptr));
      continue;
    }

    const auto& origin_legacy_output_vars = op_desc.Output(legacy_output_name);
    std::vector<std::string> legacy_output_vars;
    std::copy_if(
        origin_legacy_output_vars.begin(),
        origin_legacy_output_vars.end(),
        std::back_inserter(legacy_output_vars),
        [](const auto& var_name) { return var_name != kEmptyVarName; });

    bool is_vector = (info.type_name.find("VectorType") != std::string::npos);

    // Specially process TensorArray, this because we cannot distinguish it with
    // Vector<DenseTensor> by other conditions but we cannot support it like
    // Vector<DenseTensor>
    if (legacy_output_vars.size() == 1) {
      VarDesc* var = block->FindVarRecursive(legacy_output_vars[0]);
      if (var->GetType() ==
          paddle::framework::proto::VarType::LOD_TENSOR_ARRAY) {
        ir::Type translated_var_type =
            type_translator[var->GetType()](ctx, *var);
        op_output_types.push_back(translated_var_type);
        arg_to_idx[var->Name()] = cur_output_idx;
        continue;
      }
    }

    // if src type is Tensor
    if (!is_vector) {
      VLOG(10) << "[output translating]"
               << "[" << op_desc.Type() << "]" << info.name << " :"
               << info.type_name << " " << legacy_output_name << " "
               << legacy_output_vars.size();
      if (legacy_output_vars.size() == 0) {
        op_output_types.push_back(ir::Type(nullptr));
        continue;
      }

      auto& var_name = legacy_output_vars[0];
      VarDesc* var = block->FindVarRecursive(var_name);
      VLOG(10) << "[output translating]"
               << "[" << op_desc.Type() << "]" << info.name << " " << var_name
               << " " << var->GetType();

      ir::Type translated_var_type = type_translator[var->GetType()](ctx, *var);

      arg_to_idx[var_name] = cur_output_idx;
      op_output_types.push_back(translated_var_type);

      // if src type is Vector<Tesnor>
    } else {
      VLOG(10) << "[output translating]"
               << "[" << op_desc.Type() << "]" << info.name << " :"
               << info.type_name << " " << legacy_output_name;
      std::vector<ir::Type> types;
      for (const auto& var_name : legacy_output_vars) {
        VarDesc* var = block->FindVarRecursive(var_name);
        VLOG(10) << "[output translating]"
                 << "[" << op_desc.Type() << "]" << info.name << " " << var_name
                 << " " << var->GetType();
        ir::Type translated_var_type =
            type_translator[var->GetType()](ctx, *var);
        types.push_back(translated_var_type);
        arg_to_idx[var_name] = cur_output_idx;
      }
      ir::Type vec_type = ir::VectorType::get(ctx, types);
      op_output_types.push_back(vec_type);
    }
  }
  return {op_output_types, arg_to_idx};
}

ir::AttributeMap OpTranscriber::TranslateOpAttribute(
    ir::IrContext* ctx,
    const std::string& normalized_op_name,
    const OpAttributeInfoList& op_attr_infos,
    const OpDesc& op_desc) {
  auto& attribute_translator = AttributeTranslator::instance();
  auto& op_normalizer = OpNameNormalizer::instance();
  ir::AttributeMap attribute_map = {};

  for (const auto& info : op_attr_infos) {
    auto legacy_attr_name =
        op_normalizer.GetLegacyAttrName(op_desc.Type(), info.name);

    if (op_desc.HasAttr(legacy_attr_name)) {
      paddle::framework::Attribute legacy_attr =
          op_desc.GetAttr(legacy_attr_name);
      VLOG(10) << "attribute in " << op_desc.Type()
               << " name: " << legacy_attr_name << " " << legacy_attr.index();
      ir::Attribute new_attr =
          attribute_translator(info.type_name, legacy_attr);
      attribute_map[info.name] = new_attr;
      if (!new_attr) {
        VLOG(0) << "empty attribute in " << op_desc.Type()
                << " name: " << info.name;
      }
    } else {
      VLOG(10) << "attribute in " << op_desc.Type()
               << " name: " << legacy_attr_name << " doesn't exist";
      this->HandleNonexistentAttribute(ctx, &attribute_map, info);
    }
  }

  return attribute_map;
}

void OpTranscriber::RecordOpResultMapping(TranslationContext* param_map,
                                          const OpDesc& op_desc,
                                          ir::Operation* operation,
                                          const OpOutputMapping& arg_to_idx) {
  for (const auto& n : op_desc.Outputs()) {
    auto& name = n.first;
    VLOG(10) << "[output recording]"
             << "[" << op_desc.Type() << "]" << name;
    auto& args = n.second;
    size_t idx_in_vector = 0;
    for (const auto& arg_name : args) {
      if (arg_name == kEmptyVarName) {
        continue;
      }
      auto idx_iter = arg_to_idx.find(arg_name);
      if (idx_iter == arg_to_idx.end()) {
        VLOG(4) << "[output recording]"
                << "[" << op_desc.Type() << "][skip]" << arg_name;
        continue;
      }
      auto idx = idx_iter->second;
      VLOG(10) << "[output recording]"
               << "[" << op_desc.Type() << "]" << arg_name << " " << idx;

      ir::OpResult value = operation->result(idx);
      bool generated_by_vector = value.type().isa<ir::VectorType>();

      // Specially process TensorArray, this because we cannot distinguish it
      // with Vector<DenseTensor> by other conditions but we cannot support it
      // like Vector<DenseTensor>
      if (args.size() == 1) {
        VarDesc* var = op_desc.Block()->FindVarRecursive(args[0]);
        if (var->GetType() ==
            paddle::framework::proto::VarType::LOD_TENSOR_ARRAY) {
          generated_by_vector = false;
        }
      }
      (*param_map)[arg_name] = VariableDefiningInfo(
          value, generated_by_vector, generated_by_vector ? idx_in_vector : -1);
      idx_in_vector++;
    }
  }
}

ir::Operation* OpTranscriber::operator()(ir::IrContext* ctx,
                                         TranslationContext* param_map,
                                         const OpDesc& op_desc,
                                         ir::Program* program) {
  auto op_info = this->LoopkUpOpInfo(ctx, op_desc);
  auto* op_info_concept =
      op_info.GetInterfaceImpl<dialect::OpYamlInfoInterface>();

  OpInputInfoList input_infos;
  OpAttributeInfoList attr_infos;
  OpOutputInfoList output_infos;
  std::tie(input_infos, attr_infos, output_infos, std::ignore) =
      op_info_concept->get_op_info_();

  auto op_inputs = this->GenerateOperationInput(
      ctx, param_map, op_desc, op_info.name(), input_infos, program);

  OpOutputMapping arg_to_idx;
  OpOutputTypeList op_output_types;
  std::tie(op_output_types, arg_to_idx) =
      this->GenerateOperationOutput(ctx, op_desc, output_infos);

  auto attribute_map =
      this->TranslateOpAttribute(ctx, op_info.name(), attr_infos, op_desc);
  VLOG(4) << "[general op][" << op_desc.Type() << "] preparation end.";

  ir::Operation* operation =
      ir::Operation::Create(op_inputs, attribute_map, op_output_types, op_info);
  VLOG(4) << "[general op][" << op_desc.Type() << "] opearation creation end.";
  program->block()->push_back(operation);

  VLOG(4) << "[general op][" << op_desc.Type() << "] opearation insertion end.";
  this->RecordOpResultMapping(param_map, op_desc, operation, arg_to_idx);

  return operation;
}

struct CastOpTranscriber : public OpTranscriber {
  ir::AttributeMap TranslateOpAttribute(
      ir::IrContext*,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc) override {
    auto& attribute_translator = AttributeTranslator::instance();
    ir::AttributeMap attribute_map = {};
    const OpAttributeInfo info = op_attr_infos[0];

    std::string legacy_attr_name("out_dtype");

    paddle::framework::Attribute legacy_attr;
    if (op_desc.HasAttr(legacy_attr_name)) {
      legacy_attr = op_desc.GetAttr(legacy_attr_name);
    }
    VLOG(10) << "attribute in " << op_desc.Type()
             << " name: " << legacy_attr_name << " " << legacy_attr.index();
    ir::Attribute new_attr = attribute_translator(info.type_name, legacy_attr);
    attribute_map[info.name] = new_attr;

    return attribute_map;
  }
};

struct EmbeddingOpTranscriber : public OpTranscriber {
  void HandleNonexistentAttribute(ir::IrContext* ctx,
                                  ir::AttributeMap* attribute_map,
                                  const OpAttributeInfo& info) override {
    if (info.name == "padding_idx") {
      (*attribute_map)[info.name] = ir::Int64Attribute::get(ctx, -1);
    } else if (info.name == "sparse") {
      (*attribute_map)[info.name] = ir::BoolAttribute::get(ctx, false);
    }
  }
};

struct IncrementOpTranscriber : public OpTranscriber {
  ir::AttributeMap TranslateOpAttribute(
      ir::IrContext* ctx,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc) override {
    auto& attribute_translator = AttributeTranslator::instance();
    ir::AttributeMap attribute_map = {};

    paddle::framework::Attribute legacy_attr;
    if (op_desc.HasAttr("step")) {
      legacy_attr = op_desc.GetAttr("step");
      VLOG(10) << "attribute in " << op_desc.Type() << " step: "
               << " " << legacy_attr.index();
      ir::Attribute new_attr = attribute_translator(legacy_attr);
      attribute_map["value"] = new_attr;
    } else {
      attribute_map["value"] = ir::FloatAttribute::get(ctx, 1.0f);
    }

    return attribute_map;
  }
};

// The `assign_value` in static_ops.yaml is different from the one in
// `legacy_ops.yaml`. For this op we simulate the logic in
// python/paddle/tensor/creation.py::assign(x, output)
struct AssignValueOpTranscriber : public OpTranscriber {
  ir::OpInfo LoopkUpOpInfo(ir::IrContext* ctx, const OpDesc& op_desc) override {
    std::string target_op_name = "pd.assign_value_";
    const auto& op_info = ctx->GetRegisteredOpInfo(target_op_name);
    if (!op_info) {
      IR_THROW(
          "Op assign_value should have corresponding OpInfo pd.assign_value_");
    }

    return op_info;
  }

  ir::Operation* operator()(ir::IrContext* ctx,
                            TranslationContext* param_map,
                            const OpDesc& op_desc,
                            ir::Program* program) override {
    VLOG(10) << "[op assign_value] start transcribing";
    auto op_info = this->LoopkUpOpInfo(ctx, op_desc);
    auto* op_info_concept =
        op_info.GetInterfaceImpl<dialect::OpYamlInfoInterface>();
    OpInputInfoList input_infos;
    OpAttributeInfoList attr_infos;
    OpOutputInfoList output_infos;
    std::tie(input_infos, attr_infos, output_infos, std::ignore) =
        op_info_concept->get_op_info_();
    std::unordered_map<std::string, OpAttributeInfo> attr_info_maps;
    for (auto info : attr_infos) {
      attr_info_maps.insert({info.name, info});
    }

    auto& attribute_translator = AttributeTranslator::instance();
    ir::AttributeMap attribute_map;

    paddle::framework::Attribute legacy_attr;
    if (op_desc.HasAttr("shape")) {
      legacy_attr = op_desc.GetAttr("shape");
    } else {
      IR_THROW("Op assign_value should have attribute `shape` but not find");
    }
    ir::Attribute attr_shape =
        attribute_translator(attr_info_maps.at("shape").type_name, legacy_attr);
    attribute_map["shape"] = attr_shape;

    if (op_desc.HasAttr("dtype")) {
      legacy_attr = op_desc.GetAttr("dtype");
    } else {
      IR_THROW("Op assign_value should have attribute `dtype` but not find");
    }
    ir::Attribute attr_dtype =
        attribute_translator(attr_info_maps.at("dtype").type_name, legacy_attr);
    attribute_map["dtype"] = attr_dtype;

    ir::Attribute attr_place =
        dialect::PlaceAttribute::get(ctx, phi::CPUPlace());
    attribute_map["place"] = attr_place;

    int dtype = paddle::get<int>(op_desc.GetAttr("dtype"));

    if (dtype == /*BOOL*/ 0) {
      legacy_attr = op_desc.GetAttr("bool_values");
    } else if (dtype == /*INT32*/ 2) {
      legacy_attr = op_desc.GetAttr("int32_values");
    } else if (dtype == /*FP32*/ 5) {
      legacy_attr = op_desc.GetAttr("fp32_values");
    } else if (dtype == /*INT64*/ 3) {
      legacy_attr = op_desc.GetAttr("int64_values");
    } else {
      IR_THROW(
          "Op assign_value should have attribute `**_values` but not find");
    }

    ir::Attribute attr_values = attribute_translator(
        attr_info_maps.at("values").type_name, legacy_attr);
    attribute_map["values"] = attr_values;

    VLOG(10) << "[op assign_value] attribute translation done";

    std::vector<int> src_shape =
        paddle::get<std::vector<int>>(op_desc.GetAttr("shape"));
    std::vector<int64_t> target_shape(src_shape.begin(), src_shape.end());

    ir::Builder builder(ctx, program->block());
    dialect::FullOp full_op = builder.Build<dialect::FullOp>(
        target_shape,
        0.0f,
        attr_dtype.dyn_cast<dialect::DataTypeAttribute>().data(),
        phi::CPUPlace());

    std::vector<ir::OpResult> op_inputs = {full_op->result(0)};

    VLOG(10) << "[op assign_value] insert a full op to get input";

    OpOutputMapping arg_to_idx;
    OpOutputTypeList op_output_types;
    std::tie(op_output_types, arg_to_idx) =
        this->GenerateOperationOutput(ctx, op_desc, output_infos);

    ir::Operation* operation = ir::Operation::Create(
        op_inputs, attribute_map, op_output_types, op_info);
    program->block()->push_back(operation);
    RecordOpResultMapping(param_map, op_desc, operation, arg_to_idx);

    VLOG(10) << "[op assign_value] translation finished";

    return operation;
  }
};

// This input `dropout_state_in` does not exist in static version definition
// So we generate an input by `full` with same type of output `DropoutState` of
// OpDesc And we still should be aware that `DropoutState` is an optional output
// in static graph.
ir::OpResult TranslateDropOutStateIn(ir::IrContext* ctx,
                                     TranslationContext* param_map,
                                     const OpDesc& op_desc,
                                     const std::string& normalized_op_name,
                                     const OpInputInfo& input_info,
                                     ir::Program* program) {
  const std::string legacy_output_name = "DropoutState";
  std::vector<std::string> legacy_output_vars;
  if (op_desc.HasOutput(legacy_output_name)) {
    legacy_output_vars = op_desc.Output(legacy_output_name);
  }

  if (legacy_output_vars.size() == 0) {
    VLOG(3) << "[input translating] not find output variable: DropoutState";
    return ir::OpResult(nullptr);
  }

  // `DropoutState` is a tensor
  VarDesc* dropout_state =
      op_desc.Block()->FindVarRecursive(legacy_output_vars[0]);
  if (dropout_state == nullptr) {
    IR_THROW("Unexpected: Rnn Op should have a non-empty DropoutState");
  }
  auto& type_translator = TypeTranslator::instance();
  ir::Type translated_var_type =
      type_translator[dropout_state->GetType()](ctx, *dropout_state);
  IR_ENFORCE(
      translated_var_type.isa<dialect::DenseTensorType>(),
      "Unexpected: Rnn Op's output DropoutState should be a DenseTensor");
  auto tensor_type = translated_var_type.dyn_cast<dialect::DenseTensorType>();

  ir::Builder builder(ctx, program->block());
  dialect::FullOp full_op = builder.Build<dialect::FullOp>(
      phi::vectorize(tensor_type.dims()),
      0.0f,
      dialect::TransToPhiDataType(tensor_type.dtype()),
      phi::CPUPlace());

  return full_op->result(0);
}

// `rnn` has an aditional input in dynamic graph
struct RnnOpTranscriber : public OpTranscriber {
  InputHandleFn GetSpecialInputHandlers(std::string input_name) override {
    if (input_name != "dropout_state_in") {
      return nullptr;
    }
    return TranslateDropOutStateIn;
  };
};

struct FeedOpTranscriber : public OpTranscriber {
  ir::AttributeMap TranslateOpAttribute(
      ir::IrContext* ctx,
      const std::string& normalized_op_name,
      const OpAttributeInfoList& op_attr_infos,
      const OpDesc& op_desc) override {
    ir::AttributeMap attribute_map = {
        {"name", ir::StrAttribute::get(ctx, op_desc.OutputArgumentNames()[0])},
        {"col",
         ir::Int32Attribute::get(ctx, op_desc.GetAttrIfExists<int>("col"))},
    };

    return attribute_map;
  }

  std::vector<ir::OpResult> GenerateOperationInput(
      ir::IrContext* ctx,
      TranslationContext* param_map,
      const OpDesc& op_desc,
      const std::string& normalized_op_name,
      const OpInputInfoList& input_infos,
      ir::Program* program) override {
    return {};
  }
};

struct FetchOpTranscriber : public OpTranscriber {
  ir::Operation* operator()(ir::IrContext* ctx,
                            TranslationContext* param_map,
                            const OpDesc& op_desc,
                            ir::Program* program) override {
    auto op_info = this->LoopkUpOpInfo(ctx, op_desc);

    auto* op_info_concept =
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
    OpInputInfoList input_infos;
    OpAttributeInfoList attr_infos;
    OpOutputInfoList output_infos;
    std::tie(input_infos, attr_infos, output_infos, std::ignore) =
        op_info_concept->get_op_info_();

    auto op_inputs = this->GenerateOperationInput(
        ctx, param_map, op_desc, op_info.name(), input_infos, program);

    OpOutputTypeList op_output_types;
    ir::AttributeMap attribute_map = {
        {"name", ir::StrAttribute::get(ctx, op_desc.InputArgumentNames()[0])},
        {"col",
         ir::Int32Attribute::get(ctx, op_desc.GetAttrIfExists<int>("col"))},
    };

    op_output_types.push_back(op_inputs[0].type());
    ir::Operation* operation = ir::Operation::Create(
        op_inputs, attribute_map, op_output_types, op_info);
    program->block()->push_back(operation);

    return operation;
  }
};

OpTranslator::OpTranslator() {
  general_handler = OpTranscriber();
  special_handlers["feed"] = FeedOpTranscriber();
  special_handlers["fetch_v2"] = FetchOpTranscriber();
  special_handlers["cast"] = CastOpTranscriber();
  special_handlers["lookup_table_v2"] = EmbeddingOpTranscriber();
  special_handlers["assign_value"] = AssignValueOpTranscriber();
  special_handlers["increment"] = IncrementOpTranscriber();
  special_handlers["rnn"] = RnnOpTranscriber();
}

}  // namespace translator
}  // namespace paddle
