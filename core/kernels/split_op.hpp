// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_KERNELS_SPLIT_OP_HPP_
#define CORE_KERNELS_SPLIT_OP_HPP_

#include <vector>

#include "core/framework/op.hpp"
#include "core/framework/tensor.hpp"

namespace mynet {

/**
 * @brief Creates a "split" path in the network by copying the input Tensor
 *        into multiple output Tensor%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SplitOp : public Op<Dtype> {
 public:
  explicit SplitOp(OpParameterT* param) : Op<Dtype>(param) {}
  virtual void Reshape(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);

  virtual inline const char* type() const { return "Split"; }
  virtual inline uint32_t ExactNumInputTensors() const { return 1ul; }
  virtual inline uint32_t MinOutputTensors() const { return 1ul; }

 protected:
  virtual void ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                          const std::vector<Tensor<Dtype>*>& output);

  virtual void BackwardCpu(const std::vector<Tensor<Dtype>*>& output,
                           const std::vector<bool>& propagate_down,
                           const std::vector<Tensor<Dtype>*>& input);

  uint32_t count_;
};

}  // namespace mynet

#endif  // CORE_KERNELS_SPLIT_OP_HPP_
