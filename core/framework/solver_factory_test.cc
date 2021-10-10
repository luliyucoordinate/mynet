// Copyright 2021 coordinate
// Author: coordinate

#include "solver_factory.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "common.hpp"
#include "mynet_test_main.hpp"
#include "solver.hpp"

namespace mynet {

template <typename TypeParam>
class SolverFactoryTest : public MultiDeviceTest<TypeParam> {
 protected:
  SolverParameterT* simple_solver_param() {
    solver_param_.reset(new SolverParameterT());
    solver_param_->train_net_param = std::make_unique<NetParameterT>();
    auto ops_param = std::make_unique<OpsParameterT>();
    ops_param->name = "data";
    ops_param->type = "DummyData";
    ops_param->output.push_back("data");
    auto dummy_data_param = std::make_unique<DummyDataParameterT>();
    auto shape = std::make_unique<TensorShapeT>();
    shape->dim.push_back(1ul);
    dummy_data_param->shape.push_back(std::move(shape));
    ops_param->dummy_data_param = std::move(dummy_data_param);
    solver_param_->train_net_param->ops.push_back(std::move(ops_param));
    return solver_param_.get();
  }

  std::shared_ptr<SolverParameterT> solver_param_;
};

TYPED_TEST_CASE(SolverFactoryTest, TestDtypesAndDevices);

TYPED_TEST(SolverFactoryTest, TestCreateSolver) {
  typedef typename TypeParam::Dtype Dtype;
  auto& registry = SolverRegistry<Dtype>::Registry();
  std::shared_ptr<Solver<Dtype>> solver;
  SolverParameterT* solver_param = this->simple_solver_param();
  for (auto iter = registry.begin(); iter != registry.end(); ++iter) {
    solver_param->type = iter->first;
    solver.reset(SolverRegistry<Dtype>::CreateSolver(solver_param));
    EXPECT_EQ(iter->first, solver->type());
  }
}

}  // namespace mynet
