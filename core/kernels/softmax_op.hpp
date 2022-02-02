// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_KERNELS_SOFTMAX_OP_HPP_
#define CORE_KERNELS_SOFTMAX_OP_HPP_

#include <vector>

#include "core/framework/op.hpp"
#include "core/framework/tensor.hpp"

namespace mynet {

/**
 * @brief Computes the softmax function.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SoftmaxOp : public Op<Dtype> {
 public:
  explicit SoftmaxOp(OpParameterT* param) : Op<Dtype>(param) {}
  virtual void Reshape(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);

  virtual inline const char* type() const { return "Softmax"; }
  virtual inline uint32_t ExactNumInputTensors() const { return 1ul; }
  virtual inline uint32_t ExactNumOutputTensors() const { return 1ul; }

 protected:
  virtual void ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                          const std::vector<Tensor<Dtype>*>& output);
  virtual void BackwardCpu(const std::vector<Tensor<Dtype>*>& output,
                           const std::vector<bool>& propagate_down,
                           const std::vector<Tensor<Dtype>*>& input);

  uint32_t outer_num_;
  uint32_t inner_num_;
  int32_t softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS
  Tensor<Dtype> sum_multiplier_;
  /// scale is an uint32_termediate Tensor to hold temporary results.
  Tensor<Dtype> scale_;
};
}  // namespace mynet

#endif  // CORE_KERNELS_SOFTMAX_OP_HPP_
