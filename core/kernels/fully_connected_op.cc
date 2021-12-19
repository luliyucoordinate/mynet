// Copyright 2021 coordinate
// Author: coordinate

#include "fully_connected_op.hpp"

#include <memory>
#include <vector>

#include "core/framework/filler.hpp"
#include "core/framework/math_functions.hpp"
#include "core/framework/op_factory.hpp"

namespace mynet {

template <typename Dtype>
void FullyConnectedOp<Dtype>::OpSetUp(
    const std::vector<Tensor<Dtype>*>& input,
    const std::vector<Tensor<Dtype>*>& output) {
  auto& fully_connected_param = this->op_param_->fully_connected_param;
  auto num_output = fully_connected_param->num_output;
  bias_term_ = fully_connected_param->bias_term;
  transpose_ = fully_connected_param->transpose;
  N_ = num_output;
  auto axis = input[0]->CanonicalAxisIndex(fully_connected_param->axis);
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ std::vector. For example, if input[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = input[0]->count(axis);
  // Check if we need to set up the weights
  if (this->tensors_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->tensors_.resize(2);
    } else {
      this->tensors_.resize(1);
    }
    // Initialize the weights
    std::vector<uint32_t> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->tensors_[0].reset(new Tensor<Dtype>(weight_shape));
    // fill the weights
    std::shared_ptr<Filler<Dtype>> weight_filler(
        GetFiller<Dtype>(fully_connected_param->weight_filler.get()));
    weight_filler->Fill(this->tensors_[0].get());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      std::vector<uint32_t> bias_shape(1, N_);
      this->tensors_[1].reset(new Tensor<Dtype>(bias_shape));
      std::shared_ptr<Filler<Dtype>> bias_filler(
          GetFiller<Dtype>(fully_connected_param->bias_filler.get()));
      bias_filler->Fill(this->tensors_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->tensors_.size(), true);
}

template <typename Dtype>
void FullyConnectedOp<Dtype>::Reshape(
    const std::vector<Tensor<Dtype>*>& input,
    const std::vector<Tensor<Dtype>*>& output) {
  // Figure out the dimensions
  auto& fully_connected_param = this->op_param_->fully_connected_param;
  auto axis = input[0]->CanonicalAxisIndex(fully_connected_param->axis);
  auto new_K = input[0]->count(axis);
  DCHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = input[0]->count(0, axis);
  // The output shape will be the input shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  std::vector<uint32_t> output_shape = input[0]->shape();
  output_shape.resize(axis + 1);
  output_shape[axis] = N_;
  output[0]->Reshape(output_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    std::vector<uint32_t> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    mynet_set(bias_multiplier_.mutable_cpu_data(), Dtype(1), M_);
  }
}

template <typename Dtype>
void FullyConnectedOp<Dtype>::ForwardCpu(
    const std::vector<Tensor<Dtype>*>& input,
    const std::vector<Tensor<Dtype>*>& output) {
  const Dtype* input_data = input[0]->cpu_data();
  Dtype* output_data = output[0]->mutable_cpu_data();
  const Dtype* weight = this->tensors_[0]->cpu_data();
  mynet_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
                        M_, N_, K_, (Dtype)1., input_data, weight, (Dtype)0.,
                        output_data);
  if (bias_term_) {
    mynet_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                          bias_multiplier_.cpu_data(),
                          this->tensors_[1]->cpu_data(), (Dtype)1.,
                          output_data);
  }
}

template <typename Dtype>
void FullyConnectedOp<Dtype>::BackwardCpu(
    const std::vector<Tensor<Dtype>*>& output,
    const std::vector<bool>& propagate_down,
    const std::vector<Tensor<Dtype>*>& input) {
  if (this->param_propagate_down_[0]) {
    const Dtype* output_diff = output[0]->cpu_diff();
    const Dtype* input_data = input[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      mynet_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
                            input_data, output_diff, (Dtype)1.,
                            this->tensors_[0]->mutable_cpu_diff());
    } else {
      mynet_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
                            output_diff, input_data, (Dtype)1.,
                            this->tensors_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* output_diff = output[0]->cpu_diff();
    // Gradient with respect to bias
    mynet_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., output_diff,
                          bias_multiplier_.cpu_data(), (Dtype)1.,
                          this->tensors_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* output_diff = output[0]->cpu_diff();
    // Gradient with respect to input data
    if (transpose_) {
      mynet_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1.,
                            output_diff, this->tensors_[0]->cpu_data(),
                            (Dtype)0., input[0]->mutable_cpu_diff());
    } else {
      mynet_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
                            output_diff, this->tensors_[0]->cpu_data(),
                            (Dtype)0., input[0]->mutable_cpu_diff());
    }
  }
}

INSTANTIATE_CLASS(FullyConnectedOp);
REGISTER_OP_CLASS(FullyConnected);

}  // namespace mynet
