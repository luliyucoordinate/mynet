// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_NET_HPP_
#define CORE_FRAMEWORK_NET_HPP_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"
#include "core/schema/filler_generated.h"
#include "core/schema/tensor_generated.h"
#include "core/schema/op_generated.h"
#include "core/schema/net_generated.h"
#include "op.hpp"
#include "tensor.hpp"

namespace mynet {

/**
 * @brief Connects Op%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameterT.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(NetParameterT* param);
  explicit Net(const std::string& param_file, Phase phase, uint32_t level = 0,
               const std::vector<std::string>* stages = nullptr);
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameterT.
  void Init(NetParameterT* param);

  /**
   * @brief Run Forward and return the result.
   *
   */
  const std::vector<Tensor<Dtype>*>& Forward(Dtype* loss = nullptr);
  /// @brief DEPRECATED; use Forward() instead.
  const std::vector<Tensor<Dtype>*>& ForwardPrefilled(Dtype* loss = nullptr) {
    LOG_EVERY_N(WARNING, 1000)
        << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (outputological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one op to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the op of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(uint32_t start, uint32_t end);
  Dtype ForwardFrom(uint32_t start);
  Dtype ForwardTo(uint32_t end);
  /// @brief DEPRECATED; std::set input tensors then use Forward() instead.
  const std::vector<Tensor<Dtype>*>& Forward(
      const std::vector<Tensor<Dtype>*>& input, Dtype* loss = nullptr);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();
  void BackwardFromTo(uint32_t start, uint32_t end);
  void BackwardFrom(uint32_t start);
  void BackwardTo(uint32_t end);

  /**
   * @brief Reshape all op from input to output.
   *
   * This is useful to propagate changes to op sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner tensors with shared tensors.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained op from another Net.
   */
  void ShareTrainedOpsWith(const Net* other);
  // For an already initialized net, CopyTrainedOpFrom() copies the already
  // trained op from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained op from
   *        another Net.
   */
  void CopyTrainedOpsFrom(const NetParameterT* param);
  void CopyTrainedOpsFrom(const std::string& trained_filename);
  void CopyTrainedOpsFromBinaryFlat(const std::string& trained_filename);
  /// @brief Writes the net to a flat.
  flatbuffers::DetachedBuffer ToFlat(bool write_diff = false);

  /// @brief returns the network name.
  inline const std::string& name() const { return name_; }
  /// @brief returns the op names
  inline const std::vector<std::string>& op_names() const { return op_names_; }
  /// @brief returns the tensor names
  inline const std::vector<std::string>& tensor_names() const {
    return tensor_names_;
  }
  /// @brief returns the tensors
  inline const std::vector<std::shared_ptr<Tensor<Dtype>>>& tensors() const {
    return tensors_;
  }
  /// @brief returns the op
  inline const std::vector<std::shared_ptr<Op<Dtype>>>& ops() const {
    return ops_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the input vecs for each op -- usually you won't
   *        need this unless you do per-op checks such as gradients.
   */
  inline const std::vector<std::vector<Tensor<Dtype>*>>& input_vecs() const {
    return input_vecs_;
  }
  /**
   * @brief returns the output vecs for each op -- usually you won't
   *        need this unless you do per-op checks such as gradients.
   */
  inline const std::vector<std::vector<Tensor<Dtype>*>>& output_vecs() const {
    return output_vecs_;
  }
  /// @brief returns the ids of the output tensors of op i
  inline const std::vector<uint32_t>& output_ids(uint32_t i) const {
    DCHECK_GE(i, 0ul) << "Invalid op id";
    DCHECK_LT(i, output_id_vecs_.size()) << "Invalid op id";
    return output_id_vecs_[i];
  }
  /// @brief returns the ids of the input tensors of op i
  inline const std::vector<uint32_t>& input_ids(uint32_t i) const {
    DCHECK_GE(i, 0ul) << "Invalid op id";
    DCHECK_LT(i, input_id_vecs_.size()) << "Invalid op id";
    return input_id_vecs_[i];
  }
  inline const std::vector<std::vector<bool>>& input_need_backward() const {
    return input_need_backward_;
  }
  inline const std::vector<Dtype>& tensor_loss_weights() const {
    return tensor_loss_weights_;
  }
  inline const std::vector<bool>& op_need_backward() const {
    return op_need_backward_;
  }
  /// @brief returns the parameters
  inline const std::vector<std::shared_ptr<Tensor<Dtype>>>& params() const {
    return params_;
  }
  inline const std::vector<Tensor<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  inline const std::vector<float>& params_lr() const { return params_lr_; }
  inline const std::vector<bool>& has_params_lr() const {
    return has_params_lr_;
  }
  /// @brief returns the learnable parameter decay multipliers
  inline const std::vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const std::vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const std::map<std::string, uint32_t>& param_names_index() const {
    return param_names_index_;
  }
  inline const std::vector<uint32_t>& param_owners() const {
    return param_owners_;
  }
  inline const std::vector<std::string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output tensor numbers
  inline uint32_t num_inputs() const { return net_input_tensors_.size(); }
  inline uint32_t num_outputs() const { return net_output_tensors_.size(); }
  inline const std::vector<Tensor<Dtype>*>& input_tensors() const {
    return net_input_tensors_;
  }
  inline const std::vector<Tensor<Dtype>*>& output_tensors() const {
    return net_output_tensors_;
  }
  inline const std::vector<uint32_t>& input_tensor_indices() const {
    return net_input_tensor_indices_;
  }
  inline const std::vector<uint32_t>& output_tensor_indices() const {
    return net_output_tensor_indices_;
  }
  bool has_tensor(const std::string& tensor_name) const;
  const std::shared_ptr<Tensor<Dtype>> tensor_by_name(
      const std::string& tensor_name) const;
  bool has_op(const std::string& op_name) const;
  const std::shared_ptr<Op<Dtype>> op_by_name(const std::string& op_name) const;

  void set_debug_info(bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove op that the user specified should be excluded given the
   * current phase, level, and stage.
   */
  // TODO(11117913): maybe todo it
  // static void FilterNet(const NetParameterT* param,
  //                       NetParameterT* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  // static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
  //                            const std::string& op_name);

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void run(uint32_t op) = 0;

    template <typename T>
    friend class Net;
  };
  const std::vector<Callback*>& before_forward() const {
    return before_forward_;
  }
  void add_before_forward(Callback* value) { before_forward_.push_back(value); }
  const std::vector<Callback*>& after_forward() const { return after_forward_; }
  void add_after_forward(Callback* value) { after_forward_.push_back(value); }
  const std::vector<Callback*>& before_backward() const {
    return before_backward_;
  }
  void add_before_backward(Callback* value) {
    before_backward_.push_back(value);
  }
  const std::vector<Callback*>& after_backward() const {
    return after_backward_;
  }
  void add_after_backward(Callback* value) { after_backward_.push_back(value); }

 protected:
  // Helpers for Init.
  /// @brief Append a new output tensor to the net.
  void AppendOutput(const NetParameterT* param, uint32_t op_id,
                    uint32_t output_id,
                    std::set<std::string>* available_tensors,
                    std::map<std::string, uint32_t>* tensor_name_to_idx);
  /// @brief Append a new input tensor to the net.
  uint32_t AppendInput(const NetParameterT* param, uint32_t op_id,
                       uint32_t input_id,
                       std::set<std::string>* available_tensors,
                       std::map<std::string, uint32_t>* tensor_name_to_idx);
  /// @brief Append a new parameter tensor to the net.
  void AppendParam(const NetParameterT* param, uint32_t op_id,
                   uint32_t param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(uint32_t op_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(uint32_t op_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(uint32_t param_id);

  /// @brief The network name
  std::string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual op in the net
  std::vector<std::shared_ptr<Op<Dtype>>> ops_;
  std::vector<std::string> op_names_;
  std::map<std::string, uint32_t> op_names_index_;
  std::vector<bool> op_need_backward_;
  /// @brief the tensors storing intermediate results between the op.
  std::vector<std::shared_ptr<Tensor<Dtype>>> tensors_;
  std::vector<std::string> tensor_names_;
  std::map<std::string, uint32_t> tensor_names_index_;
  std::vector<bool> tensor_need_backward_;
  /// input_vecs stores the vectors containing the input for each op.
  /// They don't actually host the tensors (tensors_ does), so we simply store
  /// pointers.
  std::vector<std::vector<Tensor<Dtype>*>> input_vecs_;
  std::vector<std::vector<uint32_t>> input_id_vecs_;
  std::vector<std::vector<bool>> input_need_backward_;
  /// output_vecs stores the vectors containing the output for each op
  std::vector<std::vector<Tensor<Dtype>*>> output_vecs_;
  std::vector<std::vector<uint32_t>> output_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net tensor,
  /// indexed by tensor_id.
  std::vector<Dtype> tensor_loss_weights_;
  std::vector<std::vector<uint32_t>> param_id_vecs_;
  std::vector<uint32_t> param_owners_;
  std::vector<std::string> param_display_names_;
  std::vector<std::pair<uint32_t, uint32_t>> param_op_indices_;
  std::map<std::string, uint32_t> param_names_index_;
  /// tensor indices for the input and the output of the net
  std::vector<uint32_t> net_input_tensor_indices_;
  std::vector<uint32_t> net_output_tensor_indices_;
  std::vector<Tensor<Dtype>*> net_input_tensors_;
  std::vector<Tensor<Dtype>*> net_output_tensors_;
  /// The parameters in the network.
  std::vector<std::shared_ptr<Tensor<Dtype>>> params_;
  std::vector<Tensor<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  std::vector<uint32_t> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  std::vector<float> params_lr_;
  std::vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  std::vector<float> params_weight_decay_;
  std::vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  // Callbacks
  std::vector<Callback*> before_forward_;
  std::vector<Callback*> after_forward_;
  std::vector<Callback*> before_backward_;
  std::vector<Callback*> after_backward_;

  DISABLE_COPY_AND_ASSIGN(Net);
};

}  // namespace mynet

#endif  // CORE_FRAMEWORK_NET_HPP_
