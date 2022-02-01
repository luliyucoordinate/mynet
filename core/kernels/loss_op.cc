// Copyright 2021 coordinate
// Author: coordinate

#include "loss_op.hpp"

#include <vector>

namespace mynet {

template <typename Dtype>
void LossOp<Dtype>::OpSetUp(const std::vector<Tensor<Dtype>*>& input,
                            const std::vector<Tensor<Dtype>*>& output) {
  // LossOps have a non-zero (1) loss by default.
  if (!this->op_param_->loss_weight.empty()) {
    this->op_param_->loss_weight.emplace_back(Dtype(1));
  }
}

template <typename Dtype>
void LossOp<Dtype>::Reshape(const std::vector<Tensor<Dtype>*>& input,
                            const std::vector<Tensor<Dtype>*>& output) {
  DCHECK_EQ(input[0]->shape(0), input[1]->shape(0))
      << "The data and label should have the same first dimension.";
  std::vector<uint32_t> loss_shape(0);  // Loss ops output a scalar; 0 axes.
  output[0]->Reshape(loss_shape);
}

INSTANTIATE_CLASS(LossOp);

}  // namespace mynet
