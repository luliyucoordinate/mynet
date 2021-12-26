// Copyright 2021 coordinate
// Author: coordinate

#include "softmax_op.hpp"

#include <algorithm>
#include <vector>

#include "core/framework/math_functions.hpp"

namespace mynet {

template <typename Dtype>
void SoftmaxOp<Dtype>::Reshape(const std::vector<Tensor<Dtype>*>& input,
                               const std::vector<Tensor<Dtype>*>& output) {
  softmax_axis_ =
      input[0]->CanonicalAxisIndex(this->op_param_->softmax_param->axis);
  output[0]->ReshapeLike(*input[0]);
  std::vector<uint32_t> mult_dims(1, input[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  mynet_set(multiplier_data, Dtype(1), sum_multiplier_.count());
  outer_num_ = input[0]->count(0, softmax_axis_);
  inner_num_ = input[0]->count(softmax_axis_ + 1);
  std::vector<uint32_t> scale_dims = input[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxOp<Dtype>::ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                                  const std::vector<Tensor<Dtype>*>& output) {
  const Dtype* input_data = input[0]->cpu_data();
  Dtype* output_data = output[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  uint32_t channels = input[0]->shape(softmax_axis_);
  uint32_t dim = input[0]->count() / outer_num_;
  mynet_copy(output_data, input_data, input[0]->count());
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (uint32_t i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    mynet_copy(scale_data, input_data + i * dim, inner_num_);
    for (uint32_t j = 0; j < channels; j++) {
      for (uint32_t k = 0; k < inner_num_; k++) {
        scale_data[k] =
            std::max(scale_data[k], input_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    mynet_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
                          -1., sum_multiplier_.cpu_data(), scale_data, 1.,
                          output_data);
    // exponentiation
    mynet_exp<Dtype>(dim, output_data, output_data);
    // sum after exp
    mynet_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1., output_data,
                          sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (uint32_t j = 0; j < channels; j++) {
      mynet_div(inner_num_, output_data, scale_data, output_data);
      output_data += inner_num_;
    }
  }
}

template <typename Dtype>
void SoftmaxOp<Dtype>::BackwardCpu(const std::vector<Tensor<Dtype>*>& output,
                                   const std::vector<bool>& propagate_down,
                                   const std::vector<Tensor<Dtype>*>& input) {
  const Dtype* output_diff = output[0]->cpu_diff();
  const Dtype* output_data = output[0]->cpu_data();
  Dtype* input_diff = input[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  uint32_t channels = output[0]->shape(softmax_axis_);
  uint32_t dim = output[0]->count() / outer_num_;
  mynet_copy(input_diff, output_diff, output[0]->count());
  for (uint32_t i = 0; i < outer_num_; ++i) {
    // compute dot(output_diff, output_data) and subtract them from the input
    // diff
    for (uint32_t k = 0; k < inner_num_; ++k) {
      scale_data[k] = mynet_cpu_strided_dot<Dtype>(
          channels, input_diff + i * dim + k, inner_num_,
          output_data + i * dim + k, inner_num_);
    }
    // subtraction
    mynet_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
                          -1., sum_multiplier_.cpu_data(), scale_data, 1.,
                          input_diff + i * dim);
  }
  // elementwise multiplication
  mynet_mul(output[0]->count(), input_diff, output_data, input_diff);
}

INSTANTIATE_CLASS(SoftmaxOp);

}  // namespace mynet
