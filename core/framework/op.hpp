// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_OP_HPP_
#define CORE_FRAMEWORK_OP_HPP_

#include <memory>
#include <utility>
#include <vector>

#include "common.hpp"
#include "core/schema/filler_generated.h"
#include "core/schema/tensor_generated.h"
#include "core/schema/op_generated.h"  // tensor and filler must before than op
#include "math_functions.hpp"
#include "tensor.hpp"

namespace mynet {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Op%s must implement a Forward function, in which they take their input
 * (input) tensor%s (if any) and compute their output tensor%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input tensor%s, given the error gradients
 * with their output tensor%s.
 */
template <typename Dtype>
class Op {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the input tensors are provided to the
   * op.
   */
  explicit Op(OpParameterT* param) : op_param_(param), phase_(param->phase) {
    // Set phase and copy tensors (if there are any).
    // Because unique_ptr can not copy, so use move.
    auto& opt = op_param_->tensors;

    if (opt.size() > 0) {
      tensors_.resize(opt.size());
      for (uint32_t i = 0; i < opt.size(); ++i) {
        tensors_[i].reset(new Tensor<Dtype>());
        tensors_[i]->FromFlat(opt[i].get());
      }
    }
  }
  virtual ~Op() {}

  /**
   * @brief Implements common op setup functionality.
   *
   * @param input the preshaped input tensors
   * @param output
   *     the allocated but unshaped output tensors, to be shaped by Reshape
   *
   * Checks that the number of input and output tensors is correct.
   * Calls OpSetUp to do special op setup for individual op types,
   * followed by Reshape to set up sizes of output tensors and internal buffers.
   * Sets up the loss weight multiplier tensors for any non-zero loss weights.
   * This method may not be overridden.
   */
  void SetUp(const std::vector<Tensor<Dtype>*>& input,
             const std::vector<Tensor<Dtype>*>& output) {
    CheckTensorCounts(input, output);
    OpSetUp(input, output);
    Reshape(input, output);
    SetLossWeights(output);
  }

  /**
   * @brief Does op-specific setup: your op should implement this function
   *        as well as Reshape.
   *
   * @param input
   *     the preshaped input tensors, whose data fields store the input data for
   *     this op
   * @param output
   *     the allocated but unshaped output tensors
   *
   * This method should do one-time op specific setup. This includes reading
   * and processing relevent parameters from the <code>op_param_</code>.
   * Setting up the shapes of output tensors and internal buffers should be done
   * in <code>Reshape</code>, which will be called before the forward pass to
   * adjust the output tensor sizes.
   */
  virtual void OpSetUp(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output) {}

  /**
   * @brief Adjust the shapes of output tensors and internal buffers to
   * accommodate the shapes of the input tensors.
   *
   * @param input the input tensors, with the requested input shapes
   * @param output the output tensors, which should be reshaped as needed
   *
   * This method should reshape output tensors as needed according to the shapes
   * of the input (input) tensors, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the op can
   * accommodate the input tensors.
   */
  virtual void Reshape(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output) = 0;

  /**
   * @brief Given the input tensors, compute the output tensors and the loss.
   *
   * @param input
   *     the input tensors, whose data fields store the input data for this op
   * @param output
   *     the preshaped output tensors, whose data fields will store this ops'
   *     outputs
   * \return The total loss from the op.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (ForwardCpu or ForwardGpu) to compute the output tensor values given the
   * input tensors.  If the op has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your op should implement ForwardCpu and (optionally) ForwardGpu.
   */
  inline Dtype Forward(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);

  /**
   * @brief Given the output tensor error gradients, compute the input tensor
   * error gradients.
   *
   * @param output
   *     the output tensors, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a std::vector with equal length to input, with each index indicating
   *     whether to propagate the error gradients down to the input tensor at
   *     the corresponding index
   * @param input
   *     the input tensors, whose diff fields will store the gradient of the
   * error with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the input tensor diffs given the
   * output tensor diffs.
   *
   * Your op should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const std::vector<Tensor<Dtype>*>& output,
                       const std::vector<bool>& propagate_down,
                       const std::vector<Tensor<Dtype>*>& input);

  /**
   * @brief Returns the std::vector of learnable parameter tensors.
   */
  std::vector<std::shared_ptr<Tensor<Dtype>>>& tensors() { return tensors_; }

  /**
   * @brief Returns the op parameter.
   */
  const OpParameterT* op_param() const { return op_param_; }

  /**
   * @brief Writes the op parameter to a flatbuffers
   */
  virtual flatbuffers::DetachedBuffer ToFlat(bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a output tensor at a given
   * index.
   */
  inline Dtype loss(uint32_t output_index) const {
    return (loss_.size() > output_index) ? loss_[output_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a output tensor at a given index.
   */
  inline void set_loss(uint32_t output_index, const Dtype value) {
    if (loss_.size() <= output_index) {
      loss_.resize(output_index + 1, Dtype(0));
    }
    loss_[output_index] = value;
  }

  /**
   * @brief Returns the op type.
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of input tensors required by the op,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * op expects some exact number of input tensors.
   */
  virtual inline uint32_t ExactNumInputTensors() const { return 0; }
  /**
   * @brief Returns the minimum number of input tensors required by the op,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * op expects some minimum number of input tensors.
   */
  virtual inline uint32_t MinInputTensors() const { return 0; }
  /**
   * @brief Returns the maximum number of input tensors required by the op,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * op expects some maximum number of input tensors.
   */
  virtual inline uint32_t MaxInputTensors() const { return 0; }
  /**
   * @brief Returns the exact number of output tensors required by the op,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * op expects some exact number of output tensors.
   */
  virtual inline uint32_t ExactNumOutputTensors() const { return 0; }
  /**
   * @brief Returns the minimum number of output tensors required by the op,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * op expects some minimum number of output tensors.
   */
  virtual inline uint32_t MinOutputTensors() const { return 0; }
  /**
   * @brief Returns the maximum number of output tensors required by the op,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * op expects some maximum number of output tensors.
   */
  virtual inline uint32_t MaxOutputTensors() const { return 0; }
  /**
   * @brief Returns true if the op requires an equal number of input and
   *        output tensors.
   *
   * This method should be overridden to return true if your op expects an
   * equal number of input and output tensors.
   */
  virtual inline bool EqualNumInputOutputTensors() const { return false; }

  /**
   * @brief Return whether "anonymous" output tensors are created automatically
   *        by the op.
   *
   * If this method returns true, Net::Init will create enough "anonymous"
   * output tensors to fulfill the requirement specified by
   * ExactNumOutputTensors() or EMinOutputTensors().
   */
  virtual inline bool AutoOutputTensors() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given input tensor
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to tensor i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(uint32_t input_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the op should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(uint32_t param_id) {
    return (param_propagate_down_.size() > param_id)
               ? param_propagate_down_[param_id]
               : false;
  }
  /**
   * @brief Sets whether the op should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(uint32_t param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }

 protected:
  /** The flatbuffer that stores the op parameters */
  OpParameterT* op_param_;
  /** The phase: TRAIN or TEST */
  const Phase phase_;
  /** The std::vector that stores the learnable parameters as a set of tensors.
   */
  std::vector<std::shared_ptr<Tensor<Dtype>>> tensors_;
  /** std::vector indicating whether to compute the diff of each param tensor.
   */
  std::vector<bool> param_propagate_down_;

  /** The std::vector that indicates whether each output tensor has a non-zero
   * weight in the objective function. */
  std::vector<Dtype> loss_;

  /** @brief Using the CPU device, compute the op output. */
  virtual void ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                          const std::vector<Tensor<Dtype>*>& output) = 0;

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the input tensors if propagate_down is true.
   */
  virtual void BackwardCpu(const std::vector<Tensor<Dtype>*>& output,
                           const std::vector<bool>& propagate_down,
                           const std::vector<Tensor<Dtype>*>& input) = 0;

  /**
   * Called by the parent op's SetUp to check that the number of input
   * and output tensors provided as input match the expected numbers specified
   * by the {ExactNum,Min,Max}{input,Output}tensors() functions.
   */
  virtual void CheckTensorCounts(const std::vector<Tensor<Dtype>*>& input,
                                 const std::vector<Tensor<Dtype>*>& output) {
    if (ExactNumInputTensors() > 0ul) {
      DCHECK_EQ(ExactNumInputTensors(), input.size())
          << type() << " op takes " << ExactNumInputTensors()
          << " input tensor(s) as input.";
    }

    if (MinInputTensors() > 0ul) {
      DCHECK_LE(MinInputTensors(), input.size())
          << type() << " op takes at least " << MinInputTensors()
          << " input tensor(s) as input.";
    }

    if (MaxInputTensors() > 0ul) {
      DCHECK_GE(MaxInputTensors(), input.size())
          << type() << " op takes at most " << MaxInputTensors()
          << " input tensor(s) as input.";
    }

    if (ExactNumOutputTensors() > 0ul) {
      DCHECK_EQ(ExactNumOutputTensors(), output.size())
          << type() << " op produces " << ExactNumOutputTensors()
          << " output tensor(s) as output.";
    }

    if (MinOutputTensors() > 0ul) {
      DCHECK_LE(MinOutputTensors(), output.size())
          << type() << " op produces at least " << MinOutputTensors()
          << " output tensor(s) as output.";
    }

    if (MaxOutputTensors() > 0ul) {
      DCHECK_GE(MaxOutputTensors(), output.size())
          << type() << " op produces at most " << MaxOutputTensors()
          << " output tensor(s) as output.";
    }

    if (EqualNumInputOutputTensors()) {
      DCHECK_EQ(input.size(), output.size())
          << type() << " op produces one output tensor as output for each "
          << "input tensor input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any output
   * tensors in the loss function. Store non-zero loss weights in the diff
   * tensor.
   */
  inline void SetLossWeights(const std::vector<Tensor<Dtype>*>& output) {
    uint32_t num_loss_weights = op_param_->loss_weight.size();
    if (num_loss_weights) {
      DCHECK_EQ(output.size(), num_loss_weights)
          << "loss_weight must be "
             "unspecified or specified once per output tensor.";
      for (uint32_t output_id = 0; output_id < output.size(); ++output_id) {
        const Dtype loss_weight = op_param_->loss_weight[output_id];
        if (loss_weight == Dtype(0)) {
          continue;
        }

        this->set_loss(output_id, loss_weight);
        uint32_t count = output[output_id]->count();
        Dtype* loss_multiplier = output[output_id]->mutable_cpu_diff();

        for (uint32_t i = 0; i < count; i++) {
          loss_multiplier[i] = loss_weight;
        }
      }
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(Op);
};  // class Op

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Op<Dtype>::Forward(const std::vector<Tensor<Dtype>*>& input,
                                const std::vector<Tensor<Dtype>*>& output) {
  Dtype loss = 0;
  Reshape(input, output);
  switch (Mynet::mode()) {
    case Mynet::CPU:
      ForwardCpu(input, output);
      for (uint32_t output_id = 0; output_id < output.size(); ++output_id) {
        if (!this->loss(output_id)) {
          continue;
        }
        uint32_t count = output[output_id]->count();
        const Dtype* data = output[output_id]->cpu_data();
        const Dtype* loss_weights = output[output_id]->cpu_diff();
        loss += mynet_cpu_dot(count, data, loss_weights);
      }
      break;
    default:
      LOG(FATAL) << "Unknown mynet mode.";
  }
  return loss;
}

template <typename Dtype>
inline void Op<Dtype>::Backward(const std::vector<Tensor<Dtype>*>& output,
                                const std::vector<bool>& propagate_down,
                                const std::vector<Tensor<Dtype>*>& input) {
  switch (Mynet::mode()) {
    case Mynet::CPU:
      BackwardCpu(output, propagate_down, input);
      break;
    default:
      LOG(FATAL) << "Unknown mynet mode.";
  }
}

// Serialize OpParameter to flatbuffer
template <typename Dtype>
flatbuffers::DetachedBuffer Op<Dtype>::ToFlat(bool write_diff) {
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  for (uint32_t i = 0; i < tensors_.size(); ++i) {
    flatbuffers::unique_ptr<mynet::TensorFlatT> tensor(
        flatbuffers::GetMutableRoot<TensorFlat>(
            tensors_[i]->ToFlat(write_diff).data())
            ->UnPack());
    op_param_->tensors.push_back(std::move(tensor));
  }
  flatbuffer_builder.Finish(OpParameter::Pack(flatbuffer_builder, op_param_));
  return flatbuffer_builder.Release();
}

}  // namespace mynet

#endif  // CORE_FRAMEWORK_OP_HPP_
