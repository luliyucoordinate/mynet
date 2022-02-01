// Copyright 2021 coordinate
// Author: coordinate

#include "split_op.hpp"

#include <string>
#include <vector>

#include "core/framework/common.hpp"
#include "core/framework/filler.hpp"
#include "core/framework/mynet_test_main.hpp"
#include "core/framework/tensor.hpp"
#include "gradient_check_util.hpp"

namespace mynet {

template <typename TypeParam>
class SplitOpTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SplitOpTest()
      : tensor_input_(new Tensor<Dtype>(2ul, 3ul, 6ul, 5ul)),
        tensor_output_a_(new Tensor<Dtype>()),
        tensor_output_b_(new Tensor<Dtype>()) {
    // fill the values
    FillerParameterT filler_param;
    GaussianFiller<Dtype> filler(&filler_param);
    filler.Fill(this->tensor_input_);
    tensor_input_vec_.push_back(tensor_input_);
    tensor_output_vec_.push_back(tensor_output_a_);
    tensor_output_vec_.push_back(tensor_output_b_);
  }
  virtual ~SplitOpTest() {
    delete tensor_input_;
    delete tensor_output_a_;
    delete tensor_output_b_;
  }
  Tensor<Dtype>* const tensor_input_;
  Tensor<Dtype>* const tensor_output_a_;
  Tensor<Dtype>* const tensor_output_b_;
  std::vector<Tensor<Dtype>*> tensor_input_vec_;
  std::vector<Tensor<Dtype>*> tensor_output_vec_;
};

TYPED_TEST_CASE(SplitOpTest, TestDtypesAndDevices);

TYPED_TEST(SplitOpTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  SplitOp<Dtype> op(&op_param);
  op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(this->tensor_output_a_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_a_->channels(), 3ul);
  EXPECT_EQ(this->tensor_output_a_->height(), 6ul);
  EXPECT_EQ(this->tensor_output_a_->width(), 5ul);
  EXPECT_EQ(this->tensor_output_b_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_b_->channels(), 3ul);
  EXPECT_EQ(this->tensor_output_b_->height(), 6ul);
  EXPECT_EQ(this->tensor_output_b_->width(), 5ul);
}

TYPED_TEST(SplitOpTest, Test) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  SplitOp<Dtype> op(&op_param);
  op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  for (uint32_t i = 0; i < this->tensor_input_->count(); ++i) {
    Dtype input_value = this->tensor_input_->cpu_data()[i];
    EXPECT_EQ(input_value, this->tensor_output_a_->cpu_data()[i]);
    EXPECT_EQ(input_value, this->tensor_output_b_->cpu_data()[i]);
  }
}

TYPED_TEST(SplitOpTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  SplitOp<Dtype> op(&op_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&op, this->tensor_input_vec_,
                               this->tensor_output_vec_);
}

}  // namespace mynet
