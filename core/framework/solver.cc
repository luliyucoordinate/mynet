// Copyright 2021 coordinate
// Author: coordinate

#include "solver.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/common.hpp"
#include "core/lib/io.hpp"

namespace mynet {

template <typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template <typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(SolverParameterT* param)
    : net_(), callbacks_(), requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const std::string& param_file)
    : net_(), callbacks_(), requested_early_exit_(false) {
  auto param_t = std::make_shared<SolverParameterT>();
  auto param = param_t.get();
  ReadSolverParamsFromTextFile(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(SolverParameterT* param) {
  LOG_IF(INFO, Mynet::root_solver())
      << "Initializing solver from parameters: " << std::endl
      << param->net;
  param_ = param;
  DCHECK_GE(param_->average_loss, 1ul)
      << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (param_->random_seed >= 0ul) {
    Mynet::set_random_seed(param_->random_seed + Mynet::solver_rank());
  }
  // Scaffolding code
  InitTrainNet();
  InitTestNets();
  if (Mynet::root_solver()) {
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

// Load weights from the mynetmodel(s) specified in "weights" solver parameter
// into the train and test nets.
template <typename Dtype>
void LoadNetWeights(std::shared_ptr<Net<Dtype>> net,
                    const std::string& model_list) {
  std::vector<std::string> model_names = split(model_list, ',');
  for (uint32_t i = 0; i < model_names.size(); ++i) {
    trim(&model_names[i]);
    LOG(INFO) << "Finetuning from " << model_names[i];
    net->CopyTrainedOpsFrom(model_names[i]);
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  uint32_t num_train_nets =
      (!param_->net.empty()) + (param_->net_param != nullptr) +
      (!param_->train_net.empty()) + (param_->train_net_param != nullptr);
  const std::string field_names = "net, net_param, train_net, train_net_param";
  DCHECK_GE(num_train_nets, 1ul)
      << "SolverParameterT must specify a train net "
      << "using one of these fields: " << field_names;
  DCHECK_LE(num_train_nets, 1ul)
      << "SolverParameterT must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;

  NetParameterT* net_param;
  if (param_->train_net_param != nullptr) {
    LOG_IF(INFO, Mynet::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param = param_->train_net_param.get();
  } else if (!param_->train_net.empty()) {
    LOG_IF(INFO, Mynet::root_solver())
        << "Creating training net from train_net file: " << param_->train_net;
    ReadNetParamsFromTextFile(param_->train_net, &net_param);
  }
  if (param_->net_param != nullptr) {
    LOG_IF(INFO, Mynet::root_solver())
        << "Creating training net specified in net_param.";
    net_param = param_->net_param.get();
  }
  if (!param_->net.empty()) {
    LOG_IF(INFO, Mynet::root_solver())
        << "Creating training net from net file: " << param_->net;
    ReadNetParamsFromTextFile(param_->net, &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  auto net_state = std::make_unique<NetStateT>();
  net_state->phase = Phase_TRAIN;
  net_state->stage = net_param->state->stage;
  net_state->level = net_param->state->level;
  const auto& param_train_state_stage = param_->train_state->stage;
  net_state->stage.insert(net_state->stage.end(),
                          param_train_state_stage.begin(),
                          param_train_state_stage.end());
  net_param->state = std::move(net_state);
  net_.reset(new Net<Dtype>(net_param));
  for (uint32_t w_idx = 0; w_idx < param_->weights.size(); ++w_idx) {
    LoadNetWeights(net_, param_->weights[w_idx]);
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  bool has_net_param = param_->net_param != nullptr;
  bool has_net_file = !param_->net.empty();
  uint32_t num_generic_nets = has_net_param + has_net_file;
  DCHECK_LE(num_generic_nets, 1ul)
      << "Both net_param and net_file may not be specified.";
  uint32_t num_test_net_params = param_->test_net_param.size();
  uint32_t num_test_net_files = param_->test_net.size();
  uint32_t num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
    DCHECK_GE(param_->test_iter.size(), num_test_nets)
        << "test_iter must be specified for each test network.";
  } else {
    DCHECK_EQ(param_->test_iter.size(), num_test_nets)
        << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  uint32_t num_generic_net_instances = param_->test_iter.size() - num_test_nets;
  uint32_t num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_->test_state.size()) {
    DCHECK_EQ(param_->test_state.size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    DCHECK_GT(param_->test_interval, 0ul);
  }
  uint32_t test_net_id = 0;
  std::vector<std::string> sources(num_test_net_instances);
  std::vector<NetParameterT*> net_params(num_test_net_instances);
  for (uint32_t i = 0; i < num_test_net_params; ++i, ++test_net_id) {
    sources[test_net_id] = "test_net_param";
    net_params[test_net_id] = param_->test_net_param[i].get();
  }
  for (uint32_t i = 0; i < num_test_net_files; ++i, ++test_net_id) {
    sources[test_net_id] = "test_net file: " + param_->test_net[i];
    ReadNetParamsFromTextFile(param_->test_net[i], &net_params[test_net_id]);
  }
  uint32_t remaining_test_nets = param_->test_iter.size() - test_net_id;
  if (has_net_param) {
    for (uint32_t i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id] = param_->net_param.get();
    }
  }
  if (has_net_file) {
    for (uint32_t i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_->net;
      ReadNetParamsFromTextFile(param_->net, &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (uint32_t i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    auto net_state = std::make_unique<NetStateT>();
    net_state->phase = Phase_TEST;
    net_state->stage = net_params[i]->state->stage;
    net_state->level = net_params[i]->state->level;

    if (!param_->test_state.empty()) {
      const auto& param_test_state_stage = param_->test_state[i]->stage;
      net_state->stage.insert(net_state->stage.end(),
                              param_test_state_stage.begin(),
                              param_test_state_stage.end());
    }
    net_params[i]->state = std::move(net_state);

    LOG(INFO) << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    test_nets_[i]->set_debug_info(param_->debug_info);
    for (uint32_t w_idx = 0; w_idx < param_->weights.size(); ++w_idx) {
      LoadNetWeights(test_nets_[i], param_->weights[w_idx]);
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(uint32_t iters) {
  uint32_t start_iter = iter_;
  uint32_t stop_iter = iter_ + iters;
  uint32_t average_loss = this->param_->average_loss;
  losses_.clear();
  smoothed_loss_ = 0;
  auto start = std::chrono::steady_clock::now();

  while (iter_ < stop_iter) {
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_->test_interval && iter_ % param_->test_interval == 0 &&
        (iter_ > 0 || param_->test_initialization)) {
      if (Mynet::root_solver()) {
        TestAll();
      }
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    for (uint32_t i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    bool display = param_->display && iter_ % param_->display == 0;
    net_->set_debug_info(display && param_->debug_info);
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (uint32_t i = 0; i < param_->iter_size; ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_->iter_size;
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      auto end = std::chrono::steady_clock::now();
      float lapse =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      LOG_IF(INFO, Mynet::root_solver())
          << "Iteration " << iter_ << " (" << per_s << " iter/s, " << lapse
          << "s/" << param_->display << " iters), loss = " << smoothed_loss_;
      iterations_last_ = iter_;
      const auto& result = net_->output_tensors();
      uint32_t score_index = 0;
      for (uint32_t j = 0; j < result.size(); ++j) {
        auto result_vec = result[j]->cpu_data();
        const std::string& output_name =
            net_->tensor_names()[net_->output_tensor_indices()[j]];
        Dtype loss_weight =
            net_->tensor_loss_weights()[net_->output_tensor_indices()[j]];
        for (uint32_t k = 0; k < result[j]->count(); ++k) {
          std::ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight << " = "
                            << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Mynet::root_solver())
              << "    Train net output #" << score_index++ << ": "
              << output_name << " = " << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (uint32_t i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    ApplyUpdate();

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_->snapshot && iter_ % param_->snapshot == 0 &&
         Mynet::root_solver()) ||
        (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  DCHECK(Mynet::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_->lr_policy;

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no input or output vecs
  // should be given, and we will just provide dummy vecs.
  uint32_t start_iter = iter_;
  Step(param_->max_iter - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_->snapshot_after_train &&
      (!param_->snapshot || iter_ % param_->snapshot != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_->display && iter_ % param_->display == 0) {
    uint32_t average_loss = this->param_->average_loss;
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_->test_interval && iter_ % param_->test_interval == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (uint32_t test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const uint32_t test_net_id) {
  DCHECK(Mynet::root_solver());
  LOG(INFO) << "Iteration " << iter_ << ", Testing net (#" << test_net_id
            << ")";
  DCHECK_NOTNULL(test_nets_[test_net_id].get())
      ->ShareTrainedOpsWith(net_.get());
  std::vector<Dtype> test_score;
  std::vector<uint32_t> test_score_output_id;
  const auto& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (uint32_t i = 0; i < param_->test_iter[test_net_id]; ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
      if (SolverAction::SNAPSHOT == request) {
        Snapshot();
      } else if (SolverAction::STOP == request) {
        requested_early_exit_ = true;
      }
      request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const auto& result = test_net->Forward(&iter_loss);
    if (param_->test_compute_loss) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (uint32_t j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (uint32_t k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      uint32_t idx = 0;
      for (uint32_t j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (uint32_t k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  if (param_->test_compute_loss) {
    loss /= param_->test_iter[test_net_id];
    LOG(INFO) << "Test loss: " << loss;
  }
  for (uint32_t i = 0; i < test_score.size(); ++i) {
    uint32_t output_tensor_index =
        test_net->output_tensor_indices()[test_score_output_id[i]];
    const std::string& output_name =
        test_net->tensor_names()[output_tensor_index];
    Dtype loss_weight = test_net->tensor_loss_weights()[output_tensor_index];
    std::ostringstream loss_msg_stream;
    Dtype mean_score = test_score[i] / param_->test_iter[test_net_id];
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight << " = "
                      << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  DCHECK(Mynet::root_solver());
  std::string model_filename;
  switch (param_->snapshot_format) {
    case SnapshotFormat_BINARY:
      model_filename = SnapshotToBinaryFlat();
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Mynet::root_solver() && param_->snapshot) {
    DCHECK(!param_->snapshot_prefix.empty())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    std::string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
                 << param_->snapshot_prefix << "'.  Make sure "
                 << "that the directory exists and is writable.";
    }
  }
}

template <typename Dtype>
std::string Solver<Dtype>::SnapshotFilename(const std::string& extension) {
  return param_->snapshot_prefix + "_iter_" + format_int(iter_) + extension;
}

template <typename Dtype>
std::string Solver<Dtype>::SnapshotToBinaryFlat() {
  std::string model_filename = SnapshotFilename(".mynetmodel");
  LOG(INFO) << "Snapshotting to binary flat file " << model_filename;
  flatbuffers::unique_ptr<mynet::NetParameterT> net_param(
      flatbuffers::GetMutableRoot<NetParameter>(
          net_->ToFlat(param_->snapshot_diff).data())
          ->UnPack());
  WriteNetParamsToBinaryFile(net_param.get(), model_filename.c_str());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  std::string state_filename(state_file);
  RestoreSolverStateFromBinaryFlat(state_filename);
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, uint32_t start_iter,
                                       uint32_t average_loss) {
  if (losses_.size() < average_loss) {
    losses_.emplace_back(loss);
    uint32_t size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    uint32_t idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace mynet
