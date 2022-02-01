// Copyright 2021 coordinate
// Author: coordinate

#include "euclidean_loss_op.hpp"

#include <vector>

#include "core/framework/math_functions.hpp"
#include "core/framework/op_factory.hpp"

namespace mynet {

template <typename Dtype>
void EuclideanLossOp<Dtype>::Reshape(
    const std::vector<Tensor<Dtype>*>& input,
    const std::vector<Tensor<Dtype>*>& output) {
  LossOp<Dtype>::Reshape(input, output);
  DCHECK_EQ(input[0]->count(1), input[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*input[0]);
}

template <typename Dtype>
void EuclideanLossOp<Dtype>::ForwardCpu(
    const std::vector<Tensor<Dtype>*>& input,
    const std::vector<Tensor<Dtype>*>& output) {
  uint32_t count = input[0]->count();
  mynet_sub(count, input[0]->cpu_data(), input[1]->cpu_data(),
            diff_.mutable_cpu_data());
  Dtype dot = mynet_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / input[0]->num() / Dtype(2);
  output[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossOp<Dtype>::BackwardCpu(
    const std::vector<Tensor<Dtype>*>& output,
    const std::vector<bool>& propagate_down,
    const std::vector<Tensor<Dtype>*>& input) {
  for (uint32_t i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      Dtype sign = (i == 0) ? 1 : -1;
      Dtype alpha = sign * output[0]->cpu_diff()[0] / input[i]->num();
      mynet_cpu_axpby(input[i]->count(),              // count
                      alpha,                          // alpha
                      diff_.cpu_data(),               // a
                      Dtype(0),                       // beta
                      input[i]->mutable_cpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(EuclideanLossOp);
REGISTER_OP_CLASS(EuclideanLoss);

}  // namespace mynet
