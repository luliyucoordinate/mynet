// Copyright 2021 coordinate
// Author: coordinate

#include "softmax_op.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include "core/framework/common.hpp"
#include "core/framework/filler.hpp"
#include "core/framework/mynet_test_main.hpp"
#include "core/framework/tensor.hpp"
#include "gradient_check_util.hpp"

namespace mynet {

template <typename TypeParam>
class SoftmaxOpTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxOpTest()
      : tensor_input_(new Tensor<Dtype>(2ul, 10ul, 2ul, 3ul)),
        tensor_output_(new Tensor<Dtype>()) {
    // fill the values
    FillerParameterT filler_param;
    GaussianFiller<Dtype> filler(&filler_param);
    filler.Fill(this->tensor_input_);
    tensor_input_vec_.push_back(tensor_input_);
    tensor_output_vec_.push_back(tensor_output_);
  }
  virtual ~SoftmaxOpTest() {
    delete tensor_input_;
    delete tensor_output_;
  }
  Tensor<Dtype>* const tensor_input_;
  Tensor<Dtype>* const tensor_output_;
  std::vector<Tensor<Dtype>*> tensor_input_vec_;
  std::vector<Tensor<Dtype>*> tensor_output_vec_;
};

TYPED_TEST_CASE(SoftmaxOpTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxOpTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  op_param.softmax_param = std::make_unique<SoftmaxParameterT>();
  SoftmaxOp<Dtype> op(&op_param);
  op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Test sum
  for (uint32_t i = 0; i < this->tensor_input_->num(); ++i) {
    for (uint32_t k = 0; k < this->tensor_input_->height(); ++k) {
      for (uint32_t l = 0; l < this->tensor_input_->width(); ++l) {
        Dtype sum = 0;
        for (uint32_t j = 0; j < this->tensor_output_->channels(); ++j) {
          sum += this->tensor_output_->data_at(i, j, k, l);
        }
        EXPECT_GE(sum, 0.999);
        EXPECT_LE(sum, 1.001);
        // Test exact values
        Dtype scale = 0;
        for (uint32_t j = 0; j < this->tensor_input_->channels(); ++j) {
          scale += exp(this->tensor_input_->data_at(i, j, k, l));
        }
        for (uint32_t j = 0; j < this->tensor_input_->channels(); ++j) {
          EXPECT_GE(this->tensor_output_->data_at(i, j, k, l) + 1e-4,
                    exp(this->tensor_input_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
          EXPECT_LE(this->tensor_output_->data_at(i, j, k, l) - 1e-4,
                    exp(this->tensor_input_->data_at(i, j, k, l)) / scale)
              << "debug: " << i << " " << j;
        }
      }
    }
  }
}

TYPED_TEST(SoftmaxOpTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  op_param.softmax_param = std::make_unique<SoftmaxParameterT>();
  SoftmaxOp<Dtype> op(&op_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
                                  this->tensor_output_vec_);
}

}  // namespace mynet
