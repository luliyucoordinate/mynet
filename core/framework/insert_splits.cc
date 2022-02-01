// Copyright 2021 coordinate
// Author: coordinate

#include "insert_splits.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "common.hpp"

namespace mynet {

void InsertSplits(NetParameterT* param) {
  // Initialize by copying from the input NetParameterT.
  std::map<std::string, std::pair<uint32_t, uint32_t>>
      tensor_name_to_last_output_idx;
  std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>>
      input_idx_to_source_output_idx;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> output_idx_to_input_count;
  std::map<std::pair<uint32_t, uint32_t>, float> output_idx_to_loss_weight;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t>
      output_idx_to_input_split_idx;
  std::map<uint32_t, std::string> op_idx_to_op_name;
  for (uint32_t i = 0; i < param->ops.size(); ++i) {
    const auto& op_param = param->ops[i];
    op_idx_to_op_name[i] = op_param->name;
    for (uint32_t j = 0; j < op_param->input.size(); ++j) {
      const std::string& tensor_name = op_param->input[j];
      if (tensor_name_to_last_output_idx.find(tensor_name) ==
          tensor_name_to_last_output_idx.end()) {
        LOG(FATAL) << "Unknown input tensor '" << tensor_name << "' (op '"
                   << op_param->name << "', input index " << j << ")";
      }
      auto input_idx = std::make_pair(i, j);
      const auto& output_idx = tensor_name_to_last_output_idx[tensor_name];
      input_idx_to_source_output_idx[input_idx] = output_idx;
      ++output_idx_to_input_count[output_idx];
    }
    for (uint32_t j = 0; j < op_param->output.size(); ++j) {
      const std::string& tensor_name = op_param->output[j];
      tensor_name_to_last_output_idx[tensor_name] = std::make_pair(i, j);
    }
    // A use of a output tensor as a loss should be handled similarly to the use
    // of a output tensor as a input tensor to another op.
    uint32_t last_loss =
        std::min(op_param->loss_weight.size(), op_param->output.size());
    for (uint32_t j = 0; j < last_loss; ++j) {
      const std::string& tensor_name = op_param->output[j];
      const auto& output_idx = tensor_name_to_last_output_idx[tensor_name];
      output_idx_to_loss_weight[output_idx] = op_param->loss_weight[j];
      if (output_idx_to_loss_weight[output_idx]) {
        ++output_idx_to_input_count[output_idx];
      }
    }
  }
  for (uint32_t i = 0; i < param->ops.size(); ++i) {
    // Replace any shared input tensors with split op outputs.
    const auto& op_param = param->ops[i];
    for (uint32_t j = 0; j < op_param->input.size(); ++j) {
      const auto& output_idx =
          input_idx_to_source_output_idx[std::make_pair(i, j)];
      uint32_t split_count = output_idx_to_input_count[output_idx];
      if (split_count > 1) {
        const std::string& op_name = op_idx_to_op_name[output_idx.first];
        const std::string& tensor_name = op_param->input[j];
        op_param->input[j] =
            SplitTensorName(op_name, tensor_name, output_idx.second,
                            output_idx_to_input_split_idx[output_idx]++);
      }
    }
    // Create split op for any output tensors used by other op as input
    // tensors more than once.
    for (uint32_t j = 0; j < op_param->output.size(); ++j) {
      auto output_idx = std::make_pair(i, j);
      uint32_t split_count = output_idx_to_input_count[output_idx];
      if (split_count > 1ul) {
        const std::string& op_name = op_idx_to_op_name[i];
        const std::string& tensor_name = op_param->output[j];
        auto split_op_param = std::make_unique<OpParameterT>();
        float loss_weight = output_idx_to_loss_weight[output_idx];
        ConfigureSplitOp(op_name, tensor_name, j, split_count, loss_weight,
                         split_op_param.get());
        param->ops.push_back(std::move(split_op_param));
        if (loss_weight) {
          op_param->loss_weight.clear();
          output_idx_to_input_split_idx[output_idx]++;
        }
      }
    }
  }
}

void ConfigureSplitOp(const std::string& op_name,
                      const std::string& tensor_name, uint32_t tensor_idx,
                      uint32_t split_count, float loss_weight,
                      OpParameterT* split_op_param) {
  split_op_param->input.push_back(tensor_name);
  split_op_param->name = SplitOpName(op_name, tensor_name, tensor_idx);
  split_op_param->type = "Split";
  for (uint32_t k = 0; k < split_count; ++k) {
    split_op_param->output.push_back(
        SplitTensorName(op_name, tensor_name, tensor_idx, k));
    if (loss_weight) {
      if (k == 0) {
        split_op_param->loss_weight.emplace_back(loss_weight);
      } else {
        split_op_param->loss_weight.emplace_back(0ul);
      }
    }
  }
}

std::string SplitOpName(const std::string& op_name,
                        const std::string& tensor_name, uint32_t tensor_idx) {
  std::stringstream split_op_name;
  split_op_name << tensor_name << "_" << op_name << "_" << tensor_idx
                << "_split";
  return split_op_name.str();
}

std::string SplitTensorName(const std::string& op_name,
                            const std::string& tensor_name, uint32_t tensor_idx,
                            uint32_t split_idx) {
  std::stringstream split_tensor_name;
  split_tensor_name << tensor_name << "_" << op_name << "_" << tensor_idx
                    << "_split_" << split_idx;
  return split_tensor_name.str();
}

}  // namespace mynet
