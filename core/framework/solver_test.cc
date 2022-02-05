// Copyright 2021 coordinate
// Author: coordinate

#include "solver.hpp"

#include <flatbuffers/idl.h>
#include <flatbuffers/util.h>

#include <string>
#include <utility>
#include <vector>

#include "common.hpp"
#include "core/lib/io.hpp"
#include "mynet_test_main.hpp"
#include "sgd_solver.hpp"

namespace mynet {

template <typename TypeParam>
class SolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolverFromFlatString(const std::string& filename) {
    auto param_t = std::make_shared<SolverParameterT>();
    auto param = param_t.get();
    ReadSolverParamsFromTextFile(filename, &param);
    // Set the solver_mode according to current Mynet::mode.
    switch (Mynet::mode()) {
      case Mynet::CPU:
        param->solver_mode = SolverMode_CPU;
        break;
      default:
        LOG(FATAL) << "Unknown Mynet mode: " << Mynet::mode();
    }
    solver_.reset(new SGDSolver<Dtype>(param));
  }

  std::shared_ptr<Solver<Dtype>> solver_;
};

TYPED_TEST_CASE(SolverTest, TestDtypesAndDevices);

TYPED_TEST(SolverTest, TestInitTrainTestNets) {
  const std::string& filename = "core/test_data/solver.json";
  this->InitSolverFromFlatString(filename);
  ASSERT_TRUE(this->solver_->net() != nullptr);
  EXPECT_TRUE(this->solver_->net()->has_op("loss"));
  EXPECT_FALSE(this->solver_->net()->has_op("accuracy"));
  ASSERT_EQ(1ul, this->solver_->test_nets().size());
  EXPECT_TRUE(this->solver_->test_nets()[0]->has_op("loss"));
}

}  // namespace mynet
