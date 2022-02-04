// Copyright 2021 coordinate
// Author: coordinate

#include "sgd_solver.hpp"

#include <flatbuffers/idl.h>
#include <flatbuffers/util.h>

#include <string>
#include <utility>
#include <vector>

#include "core/lib/io.hpp"
#include "solver_factory.hpp"

namespace mynet {

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const std::string& lr_policy = this->param_->lr_policy;
  if (lr_policy == "fixed") {
    rate = this->param_->base_lr;
  } else if (lr_policy == "step") {
    DCHECK_GT(this->param_->step_size, 0ul);
    this->current_step_ = this->iter_ / this->param_->step_size;
    DCHECK_GE(this->param_->gamma, 0ul);
    rate = this->param_->base_lr *
           std::pow(this->param_->gamma, this->current_step_);
  } else if (lr_policy == "exp") {
    DCHECK_GE(this->param_->gamma, 0ul);
    rate = this->param_->base_lr * std::pow(this->param_->gamma, this->iter_);
  } else if (lr_policy == "inv") {
    DCHECK_GE(this->param_->gamma, 0ul);
    rate = this->param_->base_lr *
           std::pow(Dtype(1) + this->param_->gamma * this->iter_,
                    -this->param_->power);
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_->step_value.size() &&
        this->iter_ >= this->param_->step_value[this->current_step_]) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " << this->iter_
                << ", step = " << this->current_step_;
    }
    DCHECK_GE(this->param_->gamma, 0ul);
    rate = this->param_->base_lr *
           std::pow(this->param_->gamma, this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_->base_lr *
           std::pow(
               Dtype(1.) - (Dtype(this->iter_) / Dtype(this->param_->max_iter)),
               this->param_->power);
  } else if (lr_policy == "sigmoid") {
    DCHECK_GE(this->param_->gamma, 0ul);
    DCHECK_GT(this->param_->step_size, 0ul);
    rate = this->param_->base_lr *
           (Dtype(1.) /
            (Dtype(1.) +
             std::exp(-this->param_->gamma *
                      (Dtype(this->iter_) - Dtype(this->param_->step_size)))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const auto& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (uint32_t i = 0; i < net_params.size(); ++i) {
    const auto& shape = net_params[i]->shape();
    history_.push_back(
        std::shared_ptr<Tensor<Dtype>>(new Tensor<Dtype>(shape)));
    update_.push_back(std::shared_ptr<Tensor<Dtype>>(new Tensor<Dtype>(shape)));
    temp_.push_back(std::shared_ptr<Tensor<Dtype>>(new Tensor<Dtype>(shape)));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  Dtype clip_gradients = this->param_->clip_gradients;
  if (clip_gradients < 0) {
    return;
  }
  const auto& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (uint32_t i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
              << l2norm_diff << " > " << clip_gradients << ") "
              << "by scale factor " << scale_factor;
    for (uint32_t i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  Dtype rate = GetLearningRate();
  if (this->param_->display && this->iter_ % this->param_->display == 0) {
    LOG_IF(INFO, Mynet::root_solver())
        << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  ClipGradients();
  for (uint32_t param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    Normalize(param_id);
    Regularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  this->net_->Update();

  // Increment the internal iter_ counter -- its value should always indicate
  // the number of times the weights have been updated.
  ++this->iter_;
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(uint32_t param_id) {
  if (this->param_->iter_size == 1ul) {
    return;
  }
  // Scale gradient to counterbalance accumulation.
  const auto& net_params = this->net_->learnable_params();
  Dtype accum_normalization = Dtype(1.) / this->param_->iter_size;
  switch (Mynet::mode()) {
    case Mynet::CPU: {
      mynet_scal(net_params[param_id]->count(), accum_normalization,
                 net_params[param_id]->mutable_cpu_diff());
      break;
    }
    default:
      LOG(FATAL) << "Unknown mynet mode: " << Mynet::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(uint32_t param_id) {
  const auto& net_params = this->net_->learnable_params();
  const auto& net_params_weight_decay = this->net_->params_weight_decay();
  Dtype weight_decay = this->param_->weight_decay;
  std::string regularization_type = this->param_->regularization_type;
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  switch (Mynet::mode()) {
    case Mynet::CPU: {
      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          mynet_axpy(net_params[param_id]->count(), local_decay,
                     net_params[param_id]->cpu_data(),
                     net_params[param_id]->mutable_cpu_diff());
        } else if (regularization_type == "L1") {
          mynet_cpu_sign(net_params[param_id]->count(),
                         net_params[param_id]->cpu_data(),
                         temp_[param_id]->mutable_cpu_data());
          mynet_axpy(net_params[param_id]->count(), local_decay,
                     temp_[param_id]->cpu_data(),
                     net_params[param_id]->mutable_cpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }
      break;
    }
    default:
      LOG(FATAL) << "Unknown mynet mode: " << Mynet::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(uint32_t param_id, Dtype rate) {
  const auto& net_params = this->net_->learnable_params();
  const auto& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_->momentum;
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Mynet::mode()) {
    case Mynet::CPU: {
      mynet_cpu_axpby(net_params[param_id]->count(), local_rate,
                      net_params[param_id]->cpu_diff(), momentum,
                      history_[param_id]->mutable_cpu_data());
      mynet_copy(net_params[param_id]->mutable_cpu_diff(),
                 history_[param_id]->cpu_data(), net_params[param_id]->count());
      break;
    }
    default:
      LOG(FATAL) << "Unknown mynet mode: " << Mynet::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const std::string& model_filename) {
  switch (this->param_->snapshot_format) {
    case SnapshotFormat_BINARY:
      SnapshotSolverStateToBinaryFlat(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryFlat(
    const std::string& model_filename) {
  SolverStateT state;
  state.iter = this->iter_;
  state.learned_net = model_filename;
  state.current_step = this->current_step_;
  for (uint32_t i = 0; i < history_.size(); ++i) {
    flatbuffers::unique_ptr<mynet::TensorFlatT> tensor_flat(
        flatbuffers::GetMutableRoot<TensorFlat>(history_[i]->ToFlat().data())
            ->UnPack());
    // Add history
    state.history.push_back(std::move(tensor_flat));
  }
  std::string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO) << "Snapshotting solver state to binary proto file "
            << snapshot_filename;
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(SolverState::Pack(fbb, &state));
  DCHECK(flatbuffers::SaveFile(snapshot_filename.c_str(),
                               reinterpret_cast<char*>(fbb.GetBufferPointer()),
                               fbb.GetSize(), true));
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryFlat(
    const std::string& state_file) {
  std::string data;
  DCHECK(flatbuffers::LoadFile(state_file.c_str(), true, &data));
  auto state = flatbuffers::GetRoot<SolverState>(data.c_str())->UnPack();

  this->iter_ = state->iter;
  if (!state->learned_net.empty()) {
    auto net_param_t = std::make_shared<NetParameterT>();
    auto net_param = net_param_t.get();
    ReadNetParamsFromBinaryFile(state->learned_net.c_str(), &net_param);
    this->net_->CopyTrainedOpsFrom(net_param);
  }
  this->current_step_ = state->current_step;
  DCHECK_EQ(state->history.size(), history_.size())
      << "Incorrect length of history tensors.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (uint32_t i = 0; i < history_.size(); ++i) {
    history_[i]->FromFlat(state->history[i].get());
  }
}

INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace mynet
