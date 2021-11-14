// Copyright 2021 coordinate
// Author: coordinate

#include "dummy_data_op.hpp"

#include <string>
#include <utility>
#include <vector>

#include "core/framework/common.hpp"
#include "core/framework/mynet_test_main.hpp"
#include "core/framework/tensor.hpp"

namespace mynet {

template <typename TypeParam>
class DummyDataOpTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DummyDataOpTest()
      : tensor_output_a_(new Tensor<Dtype>()),
        tensor_output_b_(new Tensor<Dtype>()),
        tensor_output_c_(new Tensor<Dtype>()) {}

  virtual void SetUp() {
    tensor_input_vec_.clear();
    tensor_output_vec_.clear();
    tensor_output_vec_.push_back(tensor_output_a_);
    tensor_output_vec_.push_back(tensor_output_b_);
    tensor_output_vec_.push_back(tensor_output_c_);
  }

  virtual ~DummyDataOpTest() {
    delete tensor_output_a_;
    delete tensor_output_b_;
    delete tensor_output_c_;
  }

  Tensor<Dtype>* const tensor_output_a_;
  Tensor<Dtype>* const tensor_output_b_;
  Tensor<Dtype>* const tensor_output_c_;
  std::vector<Tensor<Dtype>*> tensor_input_vec_;
  std::vector<Tensor<Dtype>*> tensor_output_vec_;
};

TYPED_TEST_CASE(DummyDataOpTest, TestDtypesAndDevices);

TYPED_TEST(DummyDataOpTest, TestOneOutputConstant) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT param;
  param.dummy_data_param = std::make_unique<DummyDataParameterT>();
  param.dummy_data_param->shape.push_back(std::make_unique<TensorShapeT>());
  param.dummy_data_param->shape[0]->dim = {5ul, 3ul, 2ul, 4ul};
  this->tensor_output_vec_.resize(1);
  DummyDataOp<Dtype> op(&param);
  op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  auto& tensor_output_a_shape = this->tensor_output_a_->shape();
  EXPECT_EQ(tensor_output_a_shape[0], 5ul);
  EXPECT_EQ(tensor_output_a_shape[1], 3ul);
  EXPECT_EQ(tensor_output_a_shape[2], 2ul);
  EXPECT_EQ(tensor_output_a_shape[3], 4ul);
  EXPECT_EQ(this->tensor_output_b_->count(), 0ul);
  EXPECT_EQ(this->tensor_output_c_->count(), 0ul);
  for (uint32_t i = 0; i < this->tensor_output_vec_.size(); ++i) {
    for (uint32_t j = 0; j < this->tensor_output_vec_[i]->count(); ++j) {
      EXPECT_EQ(0, this->tensor_output_vec_[i]->cpu_data()[j]);
    }
  }
  op.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  for (uint32_t i = 0; i < this->tensor_output_vec_.size(); ++i) {
    for (uint32_t j = 0; j < this->tensor_output_vec_[i]->count(); ++j) {
      EXPECT_EQ(0, this->tensor_output_vec_[i]->cpu_data()[j]);
    }
  }
}

TYPED_TEST(DummyDataOpTest, TestTwoOutputConstant) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT param;
  param.dummy_data_param = std::make_unique<DummyDataParameterT>();
  param.dummy_data_param->shape.push_back(std::make_unique<TensorShapeT>());
  param.dummy_data_param->shape[0]->dim = {5ul, 3ul, 2ul, 4ul};
  param.dummy_data_param->shape.push_back(std::make_unique<TensorShapeT>());
  param.dummy_data_param->shape[1]->dim = {5ul, 3ul, 1ul, 4ul};

  auto data_filler_param = std::make_unique<FillerParameterT>();
  data_filler_param->type = "constant";
  data_filler_param->value = 7.0f;
  param.dummy_data_param->data_filler.push_back(std::move(data_filler_param));
  this->tensor_output_vec_.resize(2);
  DummyDataOp<Dtype> op(&param);
  op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  auto& tensor_output_a_shape = this->tensor_output_a_->shape();
  EXPECT_EQ(tensor_output_a_shape[0], 5ul);
  EXPECT_EQ(tensor_output_a_shape[1], 3ul);
  EXPECT_EQ(tensor_output_a_shape[2], 2ul);
  EXPECT_EQ(tensor_output_a_shape[3], 4ul);
  auto& tensor_output_b_shape = this->tensor_output_b_->shape();
  EXPECT_EQ(tensor_output_b_shape[0], 5ul);
  EXPECT_EQ(tensor_output_b_shape[1], 3ul);
  EXPECT_EQ(tensor_output_b_shape[2], 1ul);
  EXPECT_EQ(tensor_output_b_shape[3], 4ul);
  EXPECT_EQ(this->tensor_output_c_->count(), 0ul);
  for (uint32_t i = 0; i < this->tensor_output_vec_.size(); ++i) {
    for (uint32_t j = 0; j < this->tensor_output_vec_[i]->count(); ++j) {
      EXPECT_EQ(7, this->tensor_output_vec_[i]->cpu_data()[j]);
    }
  }
  op.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  for (uint32_t i = 0; i < this->tensor_output_vec_.size(); ++i) {
    for (uint32_t j = 0; j < this->tensor_output_vec_[i]->count(); ++j) {
      EXPECT_EQ(7, this->tensor_output_vec_[i]->cpu_data()[j]);
    }
  }
}

TYPED_TEST(DummyDataOpTest, TestThreeOutputConstantGaussianConstant) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT param;
  param.dummy_data_param = std::make_unique<DummyDataParameterT>();
  param.dummy_data_param->shape.push_back(std::make_unique<TensorShapeT>());
  param.dummy_data_param->shape[0]->dim = {5ul, 3ul, 2ul, 4ul};
  auto data_filler_param_a = std::make_unique<FillerParameterT>();
  data_filler_param_a->type = "constant";
  data_filler_param_a->value = 7.0f;
  param.dummy_data_param->data_filler.push_back(std::move(data_filler_param_a));
  Dtype gaussian_mean = 3.0f;
  Dtype gaussian_std = 0.01f;
  auto data_filler_param_b = std::make_unique<FillerParameterT>();
  data_filler_param_b->type = "gaussian";
  data_filler_param_b->mean = gaussian_mean;
  data_filler_param_b->std = gaussian_std;
  param.dummy_data_param->data_filler.push_back(std::move(data_filler_param_b));
  auto data_filler_param_c = std::make_unique<FillerParameterT>();
  data_filler_param_c->type = "constant";
  data_filler_param_c->value = 9.0f;
  param.dummy_data_param->data_filler.push_back(std::move(data_filler_param_c));
  DummyDataOp<Dtype> op(&param);
  op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  auto& tensor_output_a_shape = this->tensor_output_a_->shape();
  EXPECT_EQ(tensor_output_a_shape[0], 5ul);
  EXPECT_EQ(tensor_output_a_shape[1], 3ul);
  EXPECT_EQ(tensor_output_a_shape[2], 2ul);
  EXPECT_EQ(tensor_output_a_shape[3], 4ul);
  auto& tensor_output_b_shape = this->tensor_output_b_->shape();
  EXPECT_EQ(tensor_output_b_shape[0], 5ul);
  EXPECT_EQ(tensor_output_b_shape[1], 3ul);
  EXPECT_EQ(tensor_output_b_shape[2], 2ul);
  EXPECT_EQ(tensor_output_b_shape[3], 4ul);
  auto& tensor_output_c_shape = this->tensor_output_c_->shape();
  EXPECT_EQ(tensor_output_c_shape[0], 5ul);
  EXPECT_EQ(tensor_output_c_shape[1], 3ul);
  EXPECT_EQ(tensor_output_c_shape[2], 2ul);
  EXPECT_EQ(tensor_output_c_shape[3], 4ul);
  for (uint32_t i = 0; i < this->tensor_output_a_->count(); ++i) {
    EXPECT_EQ(7ul, this->tensor_output_a_->cpu_data()[i]);
  }
  // Tensor b uses a Gaussian filler, so SetUp should not have initialized it.
  // Tensor b's data should therefore be the default Tensor data value: 0.
  for (uint32_t i = 0; i < this->tensor_output_b_->count(); ++i) {
    EXPECT_EQ(0ul, this->tensor_output_b_->cpu_data()[i]);
  }
  for (uint32_t i = 0; i < this->tensor_output_c_->count(); ++i) {
    EXPECT_EQ(9ul, this->tensor_output_c_->cpu_data()[i]);
  }

  // Do a Forward pass to fill in Tensor b with Gaussian data.
  op.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  for (uint32_t i = 0; i < this->tensor_output_a_->count(); ++i) {
    EXPECT_EQ(7, this->tensor_output_a_->cpu_data()[i]);
  }
  // Check that the Gaussian's data has been filled in with values within
  // 10 standard deviations of the mean. Record the first and last sample.
  // to check that they're different after the next Forward pass.
  for (uint32_t i = 0; i < this->tensor_output_b_->count(); ++i) {
    EXPECT_NEAR(gaussian_mean, this->tensor_output_b_->cpu_data()[i],
                gaussian_std * 10);
  }
  Dtype first_gaussian_sample = this->tensor_output_b_->cpu_data()[0];
  Dtype last_gaussian_sample =
      this->tensor_output_b_->cpu_data()[this->tensor_output_b_->count() - 1];
  for (uint32_t i = 0; i < this->tensor_output_c_->count(); ++i) {
    EXPECT_EQ(9, this->tensor_output_c_->cpu_data()[i]);
  }

  // Do another Forward pass to fill in Tensor b with Gaussian data again,
  // checking that we get different values.
  op.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  for (uint32_t i = 0; i < this->tensor_output_a_->count(); ++i) {
    EXPECT_EQ(7, this->tensor_output_a_->cpu_data()[i]);
  }
  for (uint32_t i = 0; i < this->tensor_output_b_->count(); ++i) {
    EXPECT_NEAR(gaussian_mean, this->tensor_output_b_->cpu_data()[i],
                gaussian_std * 10);
  }
  EXPECT_NE(first_gaussian_sample, this->tensor_output_b_->cpu_data()[0]);
  EXPECT_NE(
      last_gaussian_sample,
      this->tensor_output_b_->cpu_data()[this->tensor_output_b_->count() - 1]);
  for (uint32_t i = 0; i < this->tensor_output_c_->count(); ++i) {
    EXPECT_EQ(9, this->tensor_output_c_->cpu_data()[i]);
  }
}

}  // namespace mynet
