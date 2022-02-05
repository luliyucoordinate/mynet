// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_SOLVER_FACTORY_HPP_
#define CORE_FRAMEWORK_SOLVER_FACTORY_HPP_

#include <map>
#include <string>
#include <vector>

#include "common.hpp"
#include "core/schema/filler_generated.h"
#include "core/schema/tensor_generated.h"
#include "core/schema/op_generated.h"
#include "core/schema/net_generated.h"
#include "core/schema/solver_generated.h"
#include "solver.hpp"

namespace mynet {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
  typedef Solver<Dtype>* (*Creator)(SolverParameterT*);
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const std::string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    DCHECK_EQ(registry.count(type), 0ul)
        << "Solver type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a solver using a SolverParameter.
  static Solver<Dtype>* CreateSolver(SolverParameterT* param) {
    const std::string& type = param->type;
    CreatorRegistry& registry = Registry();
    DCHECK_EQ(registry.count(type), 1ul)
        << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
    return registry[type](param);
  }

  static std::vector<std::string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    std::vector<std::string> solver_types;
    for (auto iter = registry.begin(); iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  SolverRegistry() {}

  static std::string SolverTypeListString() {
    std::vector<std::string> solver_types = SolverTypeList();
    std::string solver_types_str;
    for (auto iter = solver_types.begin(); iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};

template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const std::string& type,
                   Solver<Dtype>* (*creator)(SolverParameterT*)) {
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};

#define REGISTER_SOLVER_CREATOR(type, creator)                              \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>); \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)

#define REGISTER_SOLVER_CLASS(type)                                \
  template <typename Dtype>                                        \
  Solver<Dtype>* Creator_##type##Solver(SolverParameterT* param) { \
    return new type##Solver<Dtype>(param);                         \
  }                                                                \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace mynet

#endif  // CORE_FRAMEWORK_SOLVER_FACTORY_HPP_
