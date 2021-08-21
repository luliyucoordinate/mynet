#ifndef MYNET_OPS_H_
#define MYNET_OPS_H_


#include "tensor.hpp"
#include "common.hpp"
#include "ops_factory.hpp"
#include "core/protobuf/mynet.pb.h"
#include "math_functions.hpp"

namespace mynet {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Ops%s must implement a Forward function, in which they take their input
 * (bottom) tensor%s (if any) and compute their output tensor%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input tensor%s, given the error gradients with
 * their output tensor%s.
 */
template <typename Dtype>
class Ops {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom tensors are provided to the
   * ops.
   */
  explicit Ops(const OpsParameter& param)
    : ops_param_(param) {
      // Set phase and copy tensors (if there are any).
      phase_ = param.phase();
      if (ops_param_.tensors_size() > 0) {
        tensors_.resize(ops_param_.tensors_size());
        for (size_t i = 0; i < ops_param_.tensors_size(); ++i) {
          tensors_[i].reset(new tensor<Dtype>());
          tensors_[i]->FromProto(ops_param_.tensors(i));
        }
      }
    }
  virtual ~Ops() {}

  /**
   * @brief Implements common ops setup functionality.
   *
   * @param bottom the preshaped input tensors
   * @param top
   *     the allocated but unshaped output tensors, to be shaped by Reshape
   *
   * Checks that the number of bottom and top tensors is correct.
   * Calls OpsSetUp to do special ops setup for individual ops types,
   * followed by Reshape to set up sizes of top tensors and internal buffers.
   * Sets up the loss weight multiplier tensors for any non-zero loss weights.
   * This method may not be overridden.
   */
  void SetUp(const std::vector<Tensor<Dtype>*>& bottom,
      const std::vector<Tensor<Dtype>*>& top) {
    ChecktensorCounts(bottom, top);
    OpsSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }

  /**
   * @brief Does ops-specific setup: your ops should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input tensors, whose data fields store the input data for
   *     this ops
   * @param top
   *     the allocated but unshaped output tensors
   *
   * This method should do one-time ops specific setup. This includes reading
   * and processing relevent parameters from the <code>ops_param_</code>.
   * Setting up the shapes of top tensors and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top tensor sizes.
   */
  virtual void OpsSetUp(const std::vector<Tensor<Dtype>*>& bottom,
      const std::vector<Tensor<Dtype>*>& top) {}

  /**
   * @brief Adjust the shapes of top tensors and internal buffers to accommodate
   *        the shapes of the bottom tensors.
   *
   * @param bottom the input tensors, with the requested input shapes
   * @param top the top tensors, which should be reshaped as needed
   *
   * This method should reshape top tensors as needed according to the shapes
   * of the bottom (input) tensors, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the ops can
   * accommodate the bottom tensors.
   */
  virtual void Reshape(const std::vector<Tensor<Dtype>*>& bottom,
      const std::vector<Tensor<Dtype>*>& top) = 0;

  /**
   * @brief Given the bottom tensors, compute the top tensors and the loss.
   *
   * @param bottom
   *     the input tensors, whose data fields store the input data for this ops
   * @param top
   *     the preshaped output tensors, whose data fields will store this opss'
   *     outputs
   * \return The total loss from the ops.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top tensor values given the
   * bottom tensors.  If the ops has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your ops should implement Forward_cpu and (optionally) Forward_gpu.
   */
  inline Dtype Forward(const std::vector<Tensor<Dtype>*>& bottom,
      const std::vector<Tensor<Dtype>*>& top);

  /**
   * @brief Given the top tensor error gradients, compute the bottom tensor error
   *        gradients.
   *
   * @param top
   *     the output tensors, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a std::vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom tensor at
   *     the corresponding index
   * @param bottom
   *     the input tensors, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom tensor diffs given the
   * top tensor diffs.
   *
   * Your ops should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const std::vector<Tensor<Dtype>*>& top,
      const std::vector<bool>& propagate_down,
      const std::vector<Tensor<Dtype>*>& bottom);

  /**
   * @brief Returns the std::vector of learnable parameter tensors.
   */
  std::vector<std::shared_ptr<Tensor<Dtype>>>& tensors() {
    return tensors_;
  }

  /**
   * @brief Returns the ops parameter.
   */
  const OpsParameter& ops_param() const { return ops_param_; }

  /**
   * @brief Writes the ops parameter to a protocol buffer
   */
  virtual void ToProto(OpsParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top tensor at a given index.
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top tensor at a given index.
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the ops type.
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of bottom tensors required by the ops,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * ops expects some exact number of bottom tensors.
   */
  virtual inline int ExactNumBottomTensors() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom tensors required by the ops,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * ops expects some minimum number of bottom tensors.
   */
  virtual inline int MinBottomTensors() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom tensors required by the ops,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * ops expects some maximum number of bottom tensors.
   */
  virtual inline int MaxBottomTensors() const { return -1; }
  /**
   * @brief Returns the exact number of top tensors required by the ops,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * ops expects some exact number of top tensors.
   */
  virtual inline int ExactNumTopTensors() const { return -1; }
  /**
   * @brief Returns the minimum number of top tensors required by the ops,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * ops expects some minimum number of top tensors.
   */
  virtual inline int EMinTopTensors() const { return -1; }
  /**
   * @brief Returns the maximum number of top tensors required by the ops,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * ops expects some maximum number of top tensors.
   */
  virtual inline int MaxTopTensors() const { return -1; }
  /**
   * @brief Returns true if the ops requires an equal number of bottom and
   *        top tensors.
   *
   * This method should be overridden to return true if your ops expects an
   * equal number of bottom and top tensors.
   */
  virtual inline bool EqualNumBottomTopTensors() const { return false; }

  /**
   * @brief Return whether "anonymous" top tensors are created automatically
   *        by the ops.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * tensors to fulfill the requirement specified by ExactNumTopTensors() or
   * EMinTopTensors().
   */
  virtual inline bool AutoToptensors() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom tensor
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to tensor i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the ops should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }
  /**
   * @brief Sets whether the ops should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }


 protected:
  /** The protobuf that stores the ops parameters */
  OpsParameter ops_param_;
  /** The phase: TRAIN or TEST */
  Phase phase_;
  /** The std::vector that stores the learnable parameters as a set of tensors. */
  std::vector<std::shared_ptr<Tensor<Dtype>>> tensors_;
  /** std::vector indicating whether to compute the diff of each param tensor. */
  std::vector<bool> param_propagate_down_;

  /** The std::vector that indicates whether each top tensor has a non-zero weight in
   *  the objective function. */
  std::vector<Dtype> loss_;

  /** @brief Using the CPU device, compute the ops output. */
  virtual void Forward_cpu(const std::vector<Tensor<Dtype>*>& bottom,
      const std::vector<Tensor<Dtype>*>& top) = 0;

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom tensors if propagate_down is true.
   */
  virtual void Backward_cpu(const std::vector<Tensor<Dtype>*>& top,
      const std::vector<bool>& propagate_down,
      const std::vector<Tensor<Dtype>*>& bottom) = 0;

  /**
   * Called by the parent ops's SetUp to check that the number of bottom
   * and top tensors provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}tensors() functions.
   */
  virtual void ChecktensorCounts(const std::vector<Tensor<Dtype>*>& bottom,
                               const std::vector<Tensor<Dtype>*>& top) {
    if (ExactNumBottomTensors() >= 0) {
      CHECK_EQ(ExactNumBottomTensors(), bottom.size())
          << type() << " ops takes " << ExactNumBottomTensors()
          << " bottom tensor(s) as input.";
    }
    if (MinBottomTensors() >= 0) {
      CHECK_LE(MinBottomTensors(), bottom.size())
          << type() << " ops takes at least " << MinBottomTensors()
          << " bottom tensor(s) as input.";
    }
    if (MaxBottomTensors() >= 0) {
      CHECK_GE(MaxBottomTensors(), bottom.size())
          << type() << " ops takes at most " << MaxBottomTensors()
          << " bottom tensor(s) as input.";
    }
    if (ExactNumTopTensors() >= 0) {
      CHECK_EQ(ExactNumTopTensors(), top.size())
          << type() << " ops produces " << ExactNumTopTensors()
          << " top tensor(s) as output.";
    }
    if (MinTopTensors() >= 0) {
      CHECK_LE(MinTopTensors(), top.size())
          << type() << " ops produces at least " << MinTopTensors()
          << " top tensor(s) as output.";
    }
    if (MaxTopTensors() >= 0) {
      CHECK_GE(MaxTopTensors(), top.size())
          << type() << " ops produces at most " << MaxTopTensors()
          << " top tensor(s) as output.";
    }
    if (EqualNumBottomTopTensors()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " ops produces one top tensor as output for each "
          << "bottom tensor input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top tensors in
   * the loss function. Store non-zero loss weights in the diff tensor.
   */
  inline void SetLossWeights(const std::vector<Tensor<Dtype>*>& top) {
    const int num_loss_weights = ops_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top tensor.";
      for (size_t top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = ops_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) { continue; }
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        mynet_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(ops);
};  // class ops

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Ops<Dtype>::Forward(const std::vector<Tensor<Dtype>*>& bottom,
    const std::vector<Tensor<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Mynet::mode()) {
  case Mynet::CPU:
    Forward_cpu(bottom, top);
    for (size_t top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += mynet_cpu_dot(count, data, loss_weights);
    }
    break;
  default:
    LOG(FATAL) << "Unknown mynet mode.";
  }
  return loss;
}

template <typename Dtype>
inline void Ops<Dtype>::Backward(const std::vector<Tensor<Dtype>*>& top,
    const std::vector<bool>& propagate_down,
    const std::vector<Tensor<Dtype>*>& bottom) {
  switch (Mynet::mode()) {
  case Mynet::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown mynet mode.";
  }
}

// Serialize OpsParameter to protocol buffer
template <typename Dtype>
void Ops<Dtype>::ToProto(OpsParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(ops_param_);
  param->clear_tensors();
  for (size_t i = 0; i < tensors_.size(); ++i) {
    tensors_[i]->ToProto(param->add_tensors(), write_diff);
  }
}

}  // namespace mynet

#endif  // MYNET_OPS_H_