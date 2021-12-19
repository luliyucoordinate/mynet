// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_KERNELS_FULLY_CONNECTED_OP_HPP_
#define CORE_KERNELS_FULLY_CONNECTED_OP_HPP_

#include <vector>

#include "core/framework/op.hpp"
#include "core/framework/tensor.hpp"

namespace mynet {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class FullyConnectedOp : public Op<Dtype> {
 public:
  explicit FullyConnectedOp(OpParameterT* param) : Op<Dtype>(param) {}
  virtual void OpSetUp(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);
  virtual void Reshape(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);

  virtual inline const char* type() const { return "FullyConnected"; }
  virtual inline uint32_t ExactNumInputTensors() const { return 1ul; }
  virtual inline uint32_t ExactNumOutputTensors() const { return 1ul; }

 protected:
  virtual void ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                          const std::vector<Tensor<Dtype>*>& output);
  virtual void BackwardCpu(const std::vector<Tensor<Dtype>*>& output,
                           const std::vector<bool>& propagate_down,
                           const std::vector<Tensor<Dtype>*>& input);

  uint32_t M_;
  uint32_t K_;
  uint32_t N_;
  bool bias_term_;
  Tensor<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
};

}  // namespace mynet

#endif  // CORE_KERNELS_FULLY_CONNECTED_OP_HPP_
