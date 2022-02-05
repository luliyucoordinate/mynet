// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_INSERT_SPLITS_HPP_
#define CORE_FRAMEWORK_INSERT_SPLITS_HPP_

#include <string>

#include "core/schema/filler_generated.h"
#include "core/schema/tensor_generated.h"
#include "core/schema/op_generated.h"
#include "core/schema/net_generated.h"

namespace mynet {

// Copy NetParameterTs with SplitOps added to replace any shared input
// tensors with unique input tensors provided by the SplitOp.
void InsertSplits(NetParameterT* param);

void ConfigureSplitOp(const std::string& op_name,
                      const std::string& tensor_name, uint32_t tensor_idx,
                      uint32_t split_count, const float loss_weight,
                      OpParameterT* split_op_param);

std::string SplitOpName(const std::string& op_name,
                        const std::string& tensor_name, uint32_t tensor_idx);

std::string SplitTensorName(const std::string& op_name,
                            const std::string& tensor_name, uint32_t tensor_idx,
                            uint32_t split_idx);

}  // namespace mynet

#endif  // CORE_FRAMEWORK_INSERT_SPLITS_HPP_
