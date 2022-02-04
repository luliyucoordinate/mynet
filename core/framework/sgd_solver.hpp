// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_SGD_SOLVER_HPP_
#define CORE_FRAMEWORK_SGD_SOLVER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "core/schema/filler_generated.h"
#include "core/schema/tensor_generated.h"
#include "core/schema/op_generated.h"
#include "core/schema/mynet_generated.h"
#include "core/schema/solver_generated.h"
#include "solver.hpp"

namespace mynet {

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(SolverParameterT* param) : Solver<Dtype>(param) {
    PreSolve();
  }
  explicit SGDSolver(const std::string& param_file)
      : Solver<Dtype>(param_file) {
    PreSolve();
  }
  virtual inline const char* type() const { return "SGD"; }

  const std::vector<std::shared_ptr<Tensor<Dtype>>>& history() {
    return history_;
  }

  virtual void ApplyUpdate();
  Dtype GetLearningRate();

 protected:
  void PreSolve();
  virtual void Normalize(uint32_t param_id);
  virtual void Regularize(uint32_t param_id);
  virtual void ComputeUpdateValue(uint32_t param_id, Dtype rate);
  virtual void ClipGradients();
  virtual void SnapshotSolverState(const std::string& model_filename);
  virtual void SnapshotSolverStateToBinaryFlat(
      const std::string& model_filename);
  virtual void RestoreSolverStateFromBinaryFlat(const std::string& state_file);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  std::vector<std::shared_ptr<Tensor<Dtype>>> history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

}  // namespace mynet

#endif  // CORE_FRAMEWORK_SGD_SOLVER_HPP_
