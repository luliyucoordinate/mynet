// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_KERNELS_LOSS_OP_HPP_
#define CORE_KERNELS_LOSS_OP_HPP_

#include <vector>

#include "core/framework/op.hpp"
#include "core/framework/tensor.hpp"

namespace mynet {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An uint32_terface for Op%s that take two Tensor%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Tensor representing the loss.
 *
 * LossOps are typically only capable of backpropagating to their first input
 * -- the predictions.
 */
template <typename Dtype>
class LossOp : public Op<Dtype> {
 public:
  explicit LossOp(OpParameterT* param) : Op<Dtype>(param) {}
  virtual void OpSetUp(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);
  virtual void Reshape(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);

  virtual inline uint32_t ExactNumInputTensors() const { return 2; }

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single output Tensor for LossOps, uint32_to
   * which they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */
  virtual inline bool AutoOutputTensors() const { return true; }
  virtual inline uint32_t ExactNumOutputTensors() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  virtual inline bool AllowForceBackward(uint32_t input_index) const {
    return input_index != 1;
  }
};

}  // namespace mynet

#endif  // CORE_KERNELS_LOSS_OP_HPP_
