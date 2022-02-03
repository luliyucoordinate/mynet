// Copyright 2021 coordinate
// Author: coordinate

#include "net.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/lib/io.hpp"
#include "insert_splits.hpp"
#include "math_functions.hpp"
#include "op_factory.hpp"

namespace mynet {

template <typename Dtype>
Net<Dtype>::Net(NetParameterT* param) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const std::string& param_file, Phase phase, uint32_t level,
                const std::vector<std::string>* stages) {
  auto param_t = std::make_shared<NetParameterT>();
  auto param = param_t.get();
  ReadNetParamsFromTextFile(param_file, &param);
  // Set phase, stages and level
  param->state = std::make_unique<NetStateT>();
  param->state->phase = phase;
  if (stages != nullptr) {
    for (uint32_t i = 0; i < stages->size(); i++) {
      param->state->stage.emplace_back((*stages)[i]);
    }
  }
  param->state->level = level;
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(NetParameterT* param) {
  // Set phase from the state.
  phase_ = param->state->phase;
  // Filter op based on their include/exclude rules and
  // the current NetState.
  LOG_IF(INFO, Mynet::root_solver())
      << "Initializing net from parameters: " << std::endl
      << param->name;
  // Create a copy of filtered_param with splits added where necessary.
  InsertSplits(param);
  // Basically, build all the op and set up their connections.
  name_ = param->name;
  std::map<std::string, uint32_t> tensor_name_to_idx;
  std::set<std::string> available_tensors;
  memory_used_ = 0;
  // For each op, set up its input and output
  input_vecs_.resize(param->ops.size());
  output_vecs_.resize(param->ops.size());
  input_id_vecs_.resize(param->ops.size());
  param_id_vecs_.resize(param->ops.size());
  output_id_vecs_.resize(param->ops.size());
  input_need_backward_.resize(param->ops.size());
  for (uint32_t op_id = 0; op_id < param->ops.size(); ++op_id) {
    // Inherit phase from net if unset.
    if (param->ops[op_id] == nullptr) {
      param->ops[op_id] = std::make_unique<OpParameterT>();
      param->ops[op_id]->phase = phase_;
    }
    // Setup op.
    const auto& op_param = param->ops[op_id];
    if (!op_param->propagate_down.empty()) {
      DCHECK_EQ(op_param->propagate_down.size(), op_param->input.size())
          << "propagate_down param must be specified "
          << "either 0 or input_size times ";
    }
    ops_.push_back(OpRegistry<Dtype>::CreateOp(op_param.get()));
    op_names_.push_back(op_param->name);
    LOG_IF(INFO, Mynet::root_solver()) << "Creating Op " << op_param->name;
    bool need_backward = false;

    // Figure out this op's input and output
    for (uint32_t input_id = 0; input_id < op_param->input.size(); ++input_id) {
      uint32_t tensor_id = AppendInput(param, op_id, input_id,
                                       &available_tensors, &tensor_name_to_idx);
      // If a tensor needs backward, this op should provide it.
      need_backward |= tensor_need_backward_[tensor_id];
    }
    uint32_t num_output = op_param->output.size();
    for (uint32_t output_id = 0; output_id < num_output; ++output_id) {
      AppendOutput(param, op_id, output_id, &available_tensors,
                   &tensor_name_to_idx);
      // Collect Input op outputs as Net inputs.
      if (op_param->type == "Input") {
        uint32_t tensor_id = tensors_.size() - 1;
        net_input_tensor_indices_.push_back(tensor_id);
        net_input_tensors_.push_back(tensors_[tensor_id].get());
      }
    }
    // If the op specifies that AutoOutputTensors() -> true and the OpParameter
    // specified fewer than the required number (as specified by
    // ExactNumOutputTensors() or MinOutputTensors()), allocate them here.
    auto op = ops_[op_id].get();
    if (op->AutoOutputTensors()) {
      uint32_t needed_num_output =
          std::max(op->MinOutputTensors(), op->ExactNumOutputTensors());
      for (; num_output < needed_num_output; ++num_output) {
        // Add "anonymous" output tensors -- do not modify available_tensors or
        // tensor_name_to_idx as we don't want these tensors to be usable as
        // input to other op.
        AppendOutput(param, op_id, num_output, nullptr, nullptr);
      }
    }
    // After this op is connected, std::set it up.
    ops_[op_id]->SetUp(input_vecs_[op_id], output_vecs_[op_id]);
    LOG_IF(INFO, Mynet::root_solver()) << "Setting up " << op_names_[op_id];
    for (uint32_t output_id = 0; output_id < output_vecs_[op_id].size();
         ++output_id) {
      if (tensor_loss_weights_.size() <= output_id_vecs_[op_id][output_id]) {
        tensor_loss_weights_.resize(output_id_vecs_[op_id][output_id] + 1,
                                    Dtype(0));
      }
      tensor_loss_weights_[output_id_vecs_[op_id][output_id]] =
          op->loss(output_id);
      LOG_IF(INFO, Mynet::root_solver())
          << "Output shape: " << output_vecs_[op_id][output_id]->shape_string();
      if (op->loss(output_id)) {
        LOG_IF(INFO, Mynet::root_solver())
            << "    with loss weight " << op->loss(output_id);
      }
      memory_used_ += output_vecs_[op_id][output_id]->count();
    }

    LOG_IF(INFO, Mynet::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    uint32_t param_size = op_param->param.size();
    uint32_t num_param_tensors = ops_[op_id]->tensors().size();
    DCHECK_LE(param_size, num_param_tensors)
        << "Too many params specified for op " << op_param->name;
    ParamSpecT default_param_spec;
    for (uint32_t param_id = 0; param_id < num_param_tensors; ++param_id) {
      auto param_spec = (param_id < param_size)
                            ? op_param->param[param_id].get()
                            : &default_param_spec;
      bool param_need_backward = param_spec->lr_mult != 0;
      need_backward |= param_need_backward;
      ops_[op_id]->set_param_propagate_down(param_id, param_need_backward);
    }

    for (uint32_t param_id = 0; param_id < num_param_tensors; ++param_id) {
      AppendParam(param, op_id, param_id);
    }
    // Finally, set the backward flag
    op_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (uint32_t output_id = 0; output_id < output_id_vecs_[op_id].size();
           ++output_id) {
        tensor_need_backward_[output_id_vecs_[op_id][output_id]] = true;
      }
    }
  }

  // Go through the net backwards to determine which tensors contribute to the
  // loss.  We can skip backward computation for tensors that don't contribute
  // to the loss.
  // Also checks if all input tensors don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip backward
  // computation for the entire op
  std::set<std::string> tensors_under_loss;
  std::set<std::string> tensors_skip_backp;
  for (int64_t op_id = ops_.size() - 1; op_id >= 0; --op_id) {
    bool op_contributes_loss = false;
    bool op_skip_propagate_down = true;
    for (uint32_t output_id = 0; output_id < output_vecs_[op_id].size();
         ++output_id) {
      const std::string& tensor_name =
          tensor_names_[output_id_vecs_[op_id][output_id]];
      if (ops_[op_id]->loss(output_id) ||
          (tensors_under_loss.find(tensor_name) != tensors_under_loss.end())) {
        op_contributes_loss = true;
      }
      if (tensors_skip_backp.find(tensor_name) == tensors_skip_backp.end()) {
        op_skip_propagate_down = false;
      }
      if (op_contributes_loss && !op_skip_propagate_down) break;
    }
    // If this op can skip backward computation, also all his input tensors
    // don't need backpropagation
    if (op_need_backward_[op_id] && op_skip_propagate_down) {
      op_need_backward_[op_id] = false;
      for (uint32_t input_id = 0; input_id < input_vecs_[op_id].size();
           ++input_id) {
        input_need_backward_[op_id][input_id] = false;
      }
    }
    if (!op_contributes_loss) {
      op_need_backward_[op_id] = false;
    }
    if (Mynet::root_solver()) {
      if (op_need_backward_[op_id]) {
        LOG(INFO) << op_names_[op_id] << " needs backward computation.";
      } else {
        LOG(INFO) << op_names_[op_id] << " does not need backward computation.";
      }
    }
    for (uint32_t input_id = 0; input_id < input_vecs_[op_id].size();
         ++input_id) {
      if (op_contributes_loss) {
        const std::string& tensor_name =
            tensor_names_[input_id_vecs_[op_id][input_id]];
        tensors_under_loss.insert(tensor_name);
      } else {
        input_need_backward_[op_id][input_id] = false;
      }
      if (!input_need_backward_[op_id][input_id]) {
        const std::string& tensor_name =
            tensor_names_[input_id_vecs_[op_id][input_id]];
        tensors_skip_backp.insert(tensor_name);
      }
    }
  }

  // Handle force_backward if needed.
  if (param->force_backward) {
    for (uint32_t op_id = 0; op_id < ops_.size(); ++op_id) {
      op_need_backward_[op_id] = true;
      for (uint32_t input_id = 0; input_id < input_need_backward_[op_id].size();
           ++input_id) {
        input_need_backward_[op_id][input_id] =
            input_need_backward_[op_id][input_id] ||
            ops_[op_id]->AllowForceBackward(input_id);
        tensor_need_backward_[input_id_vecs_[op_id][input_id]] =
            tensor_need_backward_[input_id_vecs_[op_id][input_id]] ||
            input_need_backward_[op_id][input_id];
      }
      for (uint32_t param_id = 0; param_id < ops_[op_id]->tensors().size();
           ++param_id) {
        ops_[op_id]->set_param_propagate_down(param_id, true);
      }
    }
  }

  // In the end, all remaining tensors are considered output tensors.
  for (auto it = available_tensors.begin(); it != available_tensors.end();
       ++it) {
    LOG_IF(INFO, Mynet::root_solver())
        << "This network produces output " << *it;
    net_output_tensors_.push_back(tensors_[tensor_name_to_idx[*it]].get());
    net_output_tensor_indices_.push_back(tensor_name_to_idx[*it]);
  }
  for (size_t tensor_id = 0; tensor_id < tensor_names_.size(); ++tensor_id) {
    tensor_names_index_[tensor_names_[tensor_id]] = tensor_id;
  }
  for (size_t op_id = 0; op_id < op_names_.size(); ++op_id) {
    op_names_index_[op_names_[op_id]] = op_id;
  }
  ShareWeights();
  debug_info_ = param->debug_info;
  LOG_IF(INFO, Mynet::root_solver()) << "Network initialization done.";
}

// Helper for Net::Init: add a new output tensor to the net.
template <typename Dtype>
void Net<Dtype>::AppendOutput(
    const NetParameterT* param, uint32_t op_id, uint32_t output_id,
    std::set<std::string>* available_tensors,
    std::map<std::string, uint32_t>* tensor_name_to_idx) {
  const auto& op_param = param->ops[op_id];
  const std::string& tensor_name = (op_param->output.size() > output_id)
                                       ? op_param->output[output_id]
                                       : "(automatic)";
  // Check if we are doing in-place computation
  if (tensor_name_to_idx && op_param->input.size() > output_id &&
      tensor_name == op_param->input[output_id]) {
    // In-place computation
    LOG_IF(INFO, Mynet::root_solver())
        << op_param->name << " -> " << tensor_name << " (in-place)";
    output_vecs_[op_id].push_back(
        tensors_[(*tensor_name_to_idx)[tensor_name]].get());
    output_id_vecs_[op_id].push_back((*tensor_name_to_idx)[tensor_name]);
  } else if (tensor_name_to_idx && tensor_name_to_idx->find(tensor_name) !=
                                       tensor_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated tensors,
    // raise an error.
    LOG(FATAL) << "Output tensor '" << tensor_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Mynet::root_solver()) {
      LOG(INFO) << op_param->name << " -> " << tensor_name;
    }
    auto tensor_pointer = std::make_shared<Tensor<Dtype>>();
    uint32_t tensor_id = tensors_.size();
    tensors_.push_back(tensor_pointer);
    tensor_names_.push_back(tensor_name);
    tensor_need_backward_.push_back(false);
    if (tensor_name_to_idx) {
      (*tensor_name_to_idx)[tensor_name] = tensor_id;
    }
    output_id_vecs_[op_id].push_back(tensor_id);
    output_vecs_[op_id].push_back(tensor_pointer.get());
  }
  if (available_tensors) {
    available_tensors->insert(tensor_name);
  }
}

// Helper for Net::Init: add a new input tensor to the net.
template <typename Dtype>
uint32_t Net<Dtype>::AppendInput(
    const NetParameterT* param, uint32_t op_id, uint32_t input_id,
    std::set<std::string>* available_tensors,
    std::map<std::string, uint32_t>* tensor_name_to_idx) {
  const auto& op_param = param->ops[op_id];
  const std::string& tensor_name = op_param->input[input_id];
  if (available_tensors != nullptr &&
      available_tensors->find(tensor_name) == available_tensors->end()) {
    LOG(FATAL) << "Unknown input tensor '" << tensor_name << "' (op '"
               << op_param->name << "', input index " << input_id << ")";
  }

  uint32_t tensor_id = 0;
  if (tensor_name_to_idx != nullptr) {
    tensor_id = (*tensor_name_to_idx)[tensor_name];
  }
  LOG_IF(INFO, Mynet::root_solver())
      << op_names_[op_id] << " <- " << tensor_name;
  input_vecs_[op_id].push_back(tensors_[tensor_id].get());
  input_id_vecs_[op_id].push_back(tensor_id);

  if (available_tensors != nullptr) {
    available_tensors->erase(tensor_name);
  }
  bool need_backward = tensor_need_backward_[tensor_id];
  // Check if the backpropagation on input_id should be skipped
  if (op_param->propagate_down.size() > 0) {
    need_backward = op_param->propagate_down[input_id];
  }
  input_need_backward_[op_id].push_back(need_backward);
  return tensor_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameterT* param, uint32_t op_id,
                             uint32_t param_id) {
  auto op_param = ops_[op_id]->op_param();
  uint32_t param_size = op_param->param.size();
  auto param_name =
      (param_size > param_id) ? op_param->param[param_id]->name : "";
  if (!param_name.empty()) {
    param_display_names_.push_back(param_name);
  } else {
    std::ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  uint32_t net_param_id = params_.size();
  params_.push_back(ops_[op_id]->tensors()[param_id]);
  param_id_vecs_[op_id].push_back(net_param_id);
  param_op_indices_.push_back({op_id, param_id});
  ParamSpecT default_param_spec;
  auto param_spec = (op_param->param.size() > param_id)
                        ? op_param->param[param_id].get()
                        : &default_param_spec;
  if (!param_size || !param_name.size() ||
      (param_name.size() &&
       param_names_index_.find(param_name) == param_names_index_.end())) {
    // This op "owns" this parameter tensor -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(INT32_MAX);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    uint32_t learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->lr_mult != 1.0f);
    has_params_decay_.push_back(param_spec->decay_mult != 1.0f);
    params_lr_.push_back(param_spec->lr_mult);
    params_weight_decay_.push_back(param_spec->decay_mult);
  } else {
    // Named param tensor with name we've seen before: share params
    uint32_t owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    auto& owner_index = param_op_indices_[owner_net_param_id];
    uint32_t owner_op_id = owner_index.first;
    uint32_t owner_param_id = owner_index.second;
    LOG_IF(INFO, Mynet::root_solver())
        << "Sharing parameters '" << param_name << "' owned by "
        << "op '" << op_names_[owner_op_id] << "', param "
        << "index " << owner_param_id;
    auto this_tensor = ops_[op_id]->tensors()[param_id].get();
    auto owner_tensor = ops_[owner_op_id]->tensors()[owner_param_id].get();
    uint32_t param_size = op_param->param.size();
    if (param_size > param_id &&
        (op_param->param[param_id]->share_mode == DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      DCHECK_EQ(this_tensor->count(), owner_tensor->count())
          << "Cannot share param '" << param_name << "' owned by op '"
          << op_names_[owner_op_id] << "' with op '" << op_names_[op_id]
          << "'; count mismatch.  Owner op param "
          << "shape is " << owner_tensor->shape_string() << "; sharing op "
          << "shape is " << this_tensor->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_tensor->shape() == owner_tensor->shape())
          << "Cannot share param '" << param_name << "' owned by op '"
          << op_names_[owner_op_id] << "' with op '" << op_names_[op_id]
          << "'; shape mismatch.  Owner op param "
          << "shape is " << owner_tensor->shape_string() << "; sharing op "
          << "expects shape " << this_tensor->shape_string();
    }
    uint32_t learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->lr_mult != 1.0f) {
      if (has_params_lr_[learnable_param_id]) {
        DCHECK_EQ(param_spec->lr_mult, params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult;
      }
    }
    if (param_spec->decay_mult != 1.0f) {
      if (has_params_decay_[learnable_param_id]) {
        DCHECK_EQ(param_spec->decay_mult,
                  params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult;
      }
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(uint32_t start, uint32_t end) {
  DCHECK_LE(start, end);
  DCHECK_LT(end, ops_.size());
  Dtype loss = 0;
  for (uint32_t i = start; i <= end; ++i) {
    for (uint32_t c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype op_loss = ops_[i]->Forward(input_vecs_[i], output_vecs_[i]);
    loss += op_loss;
    if (debug_info_) {
      ForwardDebugInfo(i);
    }
    for (uint32_t c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(uint32_t start) {
  return ForwardFromTo(start, ops_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(uint32_t end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const std::vector<Tensor<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != nullptr) {
    *loss = ForwardFromTo(0, ops_.size() - 1);
  } else {
    ForwardFromTo(0, ops_.size() - 1);
  }
  return net_output_tensors_;
}

template <typename Dtype>
const std::vector<Tensor<Dtype>*>& Net<Dtype>::Forward(
    const std::vector<Tensor<Dtype>*>& input, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000)
      << "DEPRECATED: Forward(input, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy input to net inputs
  for (uint32_t i = 0; i < input.size(); ++i) {
    net_input_tensors_[i]->CopyFrom(*input[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(uint32_t start, uint32_t end) {
  DCHECK_LE(end, start);
  DCHECK_LT(start, ops_.size());
  for (int64_t i = start; i >= end; --i) {
    for (uint32_t c = 0; c < before_backward_.size(); ++c) {
      before_backward_[c]->run(i);
    }
    if (op_need_backward_[i]) {
      ops_[i]->Backward(output_vecs_[i], input_need_backward_[i],
                        input_vecs_[i]);
      if (debug_info_) {
        BackwardDebugInfo(i);
      }
    }
    for (uint32_t c = 0; c < after_backward_.size(); ++c) {
      after_backward_[c]->run(i);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(uint32_t op_id) {
  for (uint32_t output_id = 0; output_id < output_vecs_[op_id].size();
       ++output_id) {
    const auto& tensor = *output_vecs_[op_id][output_id];
    auto& tensor_name = tensor_names_[output_id_vecs_[op_id][output_id]];
    const Dtype data_abs_val_mean = tensor.asum_data() / tensor.count();
    LOG_IF(INFO, Mynet::root_solver())
        << "    [Forward] "
        << "Op " << op_names_[op_id] << ", output tensor " << tensor_name
        << " data: " << data_abs_val_mean;
  }
  for (uint32_t param_id = 0; param_id < ops_[op_id]->tensors().size();
       ++param_id) {
    const auto& tensor = *ops_[op_id]->tensors()[param_id];
    uint32_t net_param_id = param_id_vecs_[op_id][param_id];
    auto& tensor_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = tensor.asum_data() / tensor.count();
    LOG_IF(INFO, Mynet::root_solver())
        << "    [Forward] "
        << "Op " << op_names_[op_id] << ", param tensor " << tensor_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(uint32_t op_id) {
  auto& input_vec = input_vecs_[op_id];
  for (uint32_t input_id = 0; input_id < input_vec.size(); ++input_id) {
    if (!input_need_backward_[op_id][input_id]) {
      continue;
    }
    const auto& tensor = *input_vec[input_id];
    auto& tensor_name = tensor_names_[input_id_vecs_[op_id][input_id]];
    const Dtype diff_abs_val_mean = tensor.asum_diff() / tensor.count();
    LOG_IF(INFO, Mynet::root_solver())
        << "    [Backward] "
        << "Op " << op_names_[op_id] << ", input tensor " << tensor_name
        << " diff: " << diff_abs_val_mean;
  }
  for (uint32_t param_id = 0; param_id < ops_[op_id]->tensors().size();
       ++param_id) {
    if (!ops_[op_id]->param_propagate_down(param_id)) {
      continue;
    }
    const auto& tensor = *ops_[op_id]->tensors()[param_id];
    const Dtype diff_abs_val_mean = tensor.asum_diff() / tensor.count();
    LOG_IF(INFO, Mynet::root_solver())
        << "    [Backward] "
        << "Op " << op_names_[op_id] << ", param tensor " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(uint32_t param_id) {
  const auto& tensor = *params_[param_id];
  uint32_t param_owner = param_owners_[param_id];
  auto& op_name = op_names_[param_op_indices_[param_id].first];
  auto& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = tensor.asum_diff() / tensor.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = tensor.asum_data() / tensor.count();
    LOG_IF(INFO, Mynet::root_solver())
        << "    [Update] Op " << op_name << ", param " << param_display_name
        << " data: " << data_abs_val_mean << "; diff: " << diff_abs_val_mean;
  } else {
    auto& owner_op_name = op_names_[param_op_indices_[param_owner].first];
    LOG_IF(INFO, Mynet::root_solver())
        << "    [Update] Op " << op_name << ", param tensor "
        << param_display_name << " (owned by op " << owner_op_name << ", "
        << "param " << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedOpsWith(const Net* other) {
  uint32_t num_source_op = other->ops().size();
  for (uint32_t i = 0; i < num_source_op; ++i) {
    auto source_op = other->ops()[i].get();
    auto& source_op_name = other->op_names()[i];
    uint32_t target_op_id = 0;
    while (target_op_id != op_names_.size() &&
           op_names_[target_op_id] != source_op_name) {
      ++target_op_id;
    }
    if (target_op_id == op_names_.size()) {
      LOG(INFO) << "Ignoring source op " << source_op_name;
      continue;
    }
    LOG(INFO) << "Copying source op " << source_op_name;
    auto& target_tensors = ops_[target_op_id]->tensors();
    DCHECK_EQ(target_tensors.size(), source_op->tensors().size())
        << "Incompatible number of tensors for op " << source_op_name;
    for (uint32_t j = 0; j < target_tensors.size(); ++j) {
      auto source_tensor = source_op->tensors()[j].get();
      CHECK(target_tensors[j]->shape() == source_tensor->shape())
          << "Cannot share param " << j << " weights from op '"
          << source_op_name << "'; shape mismatch.  Source param shape is "
          << source_tensor->shape_string() << "; target param shape is "
          << target_tensors[j]->shape_string();
      target_tensors[j]->ShareData(*source_tensor);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(uint32_t start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(uint32_t end) {
  BackwardFromTo(ops_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(ops_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (uint32_t i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (uint32_t i = 0; i < ops_.size(); ++i) {
    ops_[i]->Reshape(input_vecs_[i], output_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedOpsFrom(const NetParameterT* param) {
  uint32_t num_source_op = param->ops.size();
  for (uint32_t i = 0; i < num_source_op; ++i) {
    const auto& source_op = param->ops[i];
    const std::string& source_op_name = source_op->name;
    uint32_t target_op_id = 0;
    while (target_op_id != op_names_.size() &&
           op_names_[target_op_id] != source_op_name) {
      ++target_op_id;
    }
    if (target_op_id == op_names_.size()) {
      LOG(INFO) << "Ignoring source op " << source_op_name;
      continue;
    }
    LOG(INFO) << "Copying source op " << source_op_name;
    const auto& target_tensors = ops_[target_op_id]->tensors();
    const auto& source_op_tensors = source_op->tensors;
    DCHECK_EQ(target_tensors.size(), source_op_tensors.size())
        << "Incompatible number of tensors for op " << source_op_name;
    for (uint32_t j = 0; j < target_tensors.size(); ++j) {
      if (!target_tensors[j]->ShapeEquals(source_op_tensors[j].get())) {
        Tensor<Dtype> source_tensor;
        source_tensor.FromFlat(source_op_tensors[j].get());
        LOG(FATAL) << "Cannot copy param " << j << " weights from op '"
                   << source_op_name
                   << "'; shape mismatch.  Source param shape is "
                   << source_tensor.shape_string() << "; target param shape is "
                   << target_tensors[j]->shape_string() << ". "
                   << "To learn this op's parameters from scratch rather than "
                   << "copying from a saved net, rename the op.";
      }
      target_tensors[j]->FromFlat(source_op_tensors[j].get());
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedOpsFrom(const std::string& trained_filename) {
  CopyTrainedOpsFromBinaryFlat(trained_filename);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedOpsFromBinaryFlat(
    const std::string& trained_filename) {
  auto net_param_t = std::make_shared<NetParameterT>();
  auto net_param = net_param_t.get();
  ReadNetParamsFromBinaryFile(trained_filename, &net_param);
  CopyTrainedOpsFrom(net_param);
}

template <typename Dtype>
flatbuffers::DetachedBuffer Net<Dtype>::ToFlat(bool write_diff) {
  NetParameterT net_param;
  net_param.name = name_;
  // Add input and output
  DLOG(INFO) << "Serializing " << ops_.size() << " ops";
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  for (uint32_t i = 0; i < ops_.size(); ++i) {
    flatbuffers::unique_ptr<mynet::OpParameterT> op(
        flatbuffers::GetMutableRoot<OpParameter>(
            ops_[i]->ToFlat(write_diff).data())
            ->UnPack());
    net_param.ops.push_back(std::move(op));
  }
  flatbuffer_builder.Finish(NetParameter::Pack(flatbuffer_builder, &net_param));
  return flatbuffer_builder.Release();
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (uint32_t i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (uint32_t i = 0; i < learnable_params_.size(); ++i) {
    auto tensor = learnable_params_[i];
    switch (Mynet::mode()) {
      case Mynet::CPU:
        mynet_set(tensor->mutable_cpu_diff(), static_cast<Dtype>(0),
                  tensor->count());
        break;
      case Mynet::GPU:
        break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (uint32_t i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] == INT32_MAX) {
      continue;
    }

    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_tensor(const std::string& tensor_name) const {
  return tensor_names_index_.find(tensor_name) != tensor_names_index_.end();
}

template <typename Dtype>
const std::shared_ptr<Tensor<Dtype>> Net<Dtype>::tensor_by_name(
    const std::string& tensor_name) const {
  std::shared_ptr<Tensor<Dtype>> tensor_ptr;
  if (has_tensor(tensor_name)) {
    tensor_ptr = tensors_[tensor_names_index_.find(tensor_name)->second];
  } else {
    tensor_ptr.reset((Tensor<Dtype>*)(nullptr));
    LOG(WARNING) << "Unknown tensor name " << tensor_name;
  }
  return tensor_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_op(const std::string& op_name) const {
  return op_names_index_.find(op_name) != op_names_index_.end();
}

template <typename Dtype>
const std::shared_ptr<Op<Dtype>> Net<Dtype>::op_by_name(
    const std::string& op_name) const {
  std::shared_ptr<Op<Dtype>> op_ptr;
  if (has_op(op_name)) {
    op_ptr = ops_[op_names_index_.find(op_name)->second];
  } else {
    op_ptr.reset((Op<Dtype>*)(nullptr));
    LOG(WARNING) << "Unknown op name " << op_name;
  }
  return op_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace mynet
