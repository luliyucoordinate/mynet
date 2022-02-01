// Copyright 2021 coordinate
// Author: coordinate

#include "euclidean_loss_op.hpp"

#include <cmath>
#include <vector>

#include "core/framework/filler.hpp"
#include "core/framework/mynet_test_main.hpp"
#include "core/framework/tensor.hpp"
#include "gradient_check_util.hpp"

namespace mynet {

template <typename TypeParam>
class EuclideanLossOpTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  EuclideanLossOpTest()
      : tensor_input_data_(new Tensor<Dtype>(10ul, 5ul, 1ul, 1ul)),
        tensor_input_label_(new Tensor<Dtype>(10ul, 5ul, 1ul, 1ul)),
        tensor_output_loss_(new Tensor<Dtype>()) {
    // fill the values
    FillerParameterT filler_param;
    GaussianFiller<Dtype> filler(&filler_param);
    filler.Fill(this->tensor_input_data_);
    tensor_input_vec_.push_back(tensor_input_data_);
    filler.Fill(this->tensor_input_label_);
    tensor_input_vec_.push_back(tensor_input_label_);
    tensor_output_vec_.push_back(tensor_output_loss_);
  }
  virtual ~EuclideanLossOpTest() {
    delete tensor_input_data_;
    delete tensor_input_label_;
    delete tensor_output_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifying a weight of 1.
    OpParameterT op_param;
    EuclideanLossOp<Dtype> op_weight_1(&op_param);
    op_weight_1.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    Dtype loss_weight_1 =
        op_weight_1.Forward(this->tensor_input_vec_, this->tensor_output_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    Dtype kLossWeight = 3.7;
    op_param.loss_weight.clear();
    op_param.loss_weight.emplace_back(kLossWeight);
    EuclideanLossOp<Dtype> op_weight_2(&op_param);
    op_weight_2.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    Dtype loss_weight_2 =
        op_weight_2.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Tensor<Dtype>* const tensor_input_data_;
  Tensor<Dtype>* const tensor_input_label_;
  Tensor<Dtype>* const tensor_output_loss_;
  std::vector<Tensor<Dtype>*> tensor_input_vec_;
  std::vector<Tensor<Dtype>*> tensor_output_vec_;
};

TYPED_TEST_CASE(EuclideanLossOpTest, TestDtypesAndDevices);

TYPED_TEST(EuclideanLossOpTest, TestForward) { this->TestForward(); }

TYPED_TEST(EuclideanLossOpTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  Dtype kLossWeight = 3.7;
  op_param.loss_weight.emplace_back(kLossWeight);
  EuclideanLossOp<Dtype> op(&op_param);
  op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
                                  this->tensor_output_vec_);
}

}  // namespace mynet
