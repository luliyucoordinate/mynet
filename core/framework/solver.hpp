// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_SOLVER_HPP_
#define CORE_FRAMEWORK_SOLVER_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

// #include "benchmark.hpp"
#include "core/schema/filler_generated.h"
#include "core/schema/tensor_generated.h"
#include "core/schema/op_generated.h"
#include "core/schema/net_generated.h"
#include "core/schema/solver_generated.h"
#include "net.hpp"

namespace mynet {

/**
 * @brief Enumeration of actions that a client of the Solver may request by
 * implementing the Solver's action request function, which a
 * client may optionally provide in order to request early termination
 * or saving a snapshot without exiting. In the executable mynet, this
 * mechanism is used to allow the snapshot to be saved when stopping
 * execution with a SIGINT (Ctrl-C).
 */
namespace SolverAction {
enum Enum {
  NONE = 0,     // Take no special action.
  STOP = 1,     // Stop training. snapshot_after_train controls whether a
                // snapshot is created.
  SNAPSHOT = 2  // Take a snapshot, and keep training.
};
}

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef std::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(SolverParameterT* param);
  explicit Solver(const std::string& param_file);
  void Init(SolverParameterT* param);
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = nullptr);
  inline void Solve(const std::string& resume_file) {
    Solve(resume_file.c_str());
  }
  void Step(uint32_t iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();
  virtual ~Solver() {}
  inline const SolverParameterT* param() const { return param_; }
  inline std::shared_ptr<Net<Dtype>> net() { return net_; }
  inline const std::vector<std::shared_ptr<Net<Dtype>>>& test_nets() {
    return test_nets_;
  }
  uint32_t iter() const { return iter_; }

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const std::vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) { callbacks_.push_back(value); }

  void CheckSnapshotWritePermissions();
  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }

  // Make and apply the update value for the current iteration.
  virtual void ApplyUpdate() = 0;

 protected:
  std::string SnapshotFilename(const std::string& extension);
  std::string SnapshotToBinaryFlat();
  // The test routine
  void TestAll();
  void Test(const uint32_t test_net_id = 0);
  virtual void SnapshotSolverState(const std::string& model_filename) = 0;
  virtual void RestoreSolverStateFromBinaryFlat(
      const std::string& state_file) = 0;
  void DisplayOutputTensors(const uint32_t net_id);
  void UpdateSmoothedLoss(Dtype loss, uint32_t start_iter,
                          uint32_t average_loss);

  SolverParameterT* param_;
  uint32_t iter_;
  uint32_t current_step_;
  std::shared_ptr<Net<Dtype>> net_;
  std::vector<std::shared_ptr<Net<Dtype>>> test_nets_;
  std::vector<Callback*> callbacks_;
  std::vector<Dtype> losses_;
  Dtype smoothed_loss_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // Timing information, handy to tune e.g. nbr of GPUs
  float iterations_last_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

}  // namespace mynet

#endif  // CORE_FRAMEWORK_SOLVER_HPP_
