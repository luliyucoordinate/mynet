// Copyright 2021 coordinate
// Author: coordinate

#include "filler.hpp"

#include <memory>
#include <vector>

#include "common.hpp"
#include "mynet_test_main.hpp"

namespace mynet {

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
  ConstantFillerTest() : tensor_(new Tensor<Dtype>()) {
    filler_param_.reset(new FillerParameterT());
    filler_param_->value = 10.0f;
    filler_.reset(new ConstantFiller<Dtype>(filler_param_.get()));
  }
  virtual void test_params(const std::vector<uint32_t>& shape) {
    EXPECT_TRUE(tensor_);
    tensor_->Reshape(shape);
    filler_->Fill(tensor_);
    const uint32_t count = tensor_->count();
    const Dtype* data = tensor_->cpu_data();
    for (uint32_t i = 0; i < count; ++i) {
      EXPECT_EQ(data[i], filler_param_->value);
    }
  }
  virtual ~ConstantFillerTest() { delete tensor_; }
  Tensor<Dtype>* const tensor_;
  std::shared_ptr<FillerParameterT> filler_param_;
  std::shared_ptr<ConstantFiller<Dtype>> filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
  std::vector<uint32_t> tensor_shape = {2ul, 3ul, 4ul, 5ul};
  this->test_params(tensor_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill1D) {
  std::vector<uint32_t> tensor_shape = {15ul};
  this->test_params(tensor_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill2D) {
  std::vector<uint32_t> tensor_shape = {8ul, 3ul};
  this->test_params(tensor_shape);
}

TYPED_TEST(ConstantFillerTest, TestFill5D) {
  std::vector<uint32_t> tensor_shape = {2ul, 3ul, 4ul, 5ul, 2ul};
  this->test_params(tensor_shape);
}

template <typename Dtype>
class UniformFillerTest : public ::testing::Test {
 protected:
  UniformFillerTest() : tensor_(new Tensor<Dtype>()) {
    filler_param_.reset(new FillerParameterT());
    filler_param_->min = 1.0;
    filler_param_->max = 2.0;
    filler_.reset(new UniformFiller<Dtype>(filler_param_.get()));
  }

  virtual void test_params(const std::vector<uint32_t>& shape) {
    EXPECT_TRUE(tensor_);
    tensor_->Reshape(shape);
    filler_->Fill(tensor_);
    const uint32_t count = tensor_->count();
    const Dtype* data = tensor_->cpu_data();
    for (uint32_t i = 0; i < count; ++i) {
      EXPECT_GE(data[i], filler_param_->min);
      EXPECT_LE(data[i], filler_param_->max);
    }
  }

  virtual ~UniformFillerTest() { delete tensor_; }
  Tensor<Dtype>* const tensor_;
  std::shared_ptr<FillerParameterT> filler_param_;
  std::shared_ptr<UniformFiller<Dtype>> filler_;
};

TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

TYPED_TEST(UniformFillerTest, TestFill) {
  std::vector<uint32_t> tensor_shape = {2ul, 3ul, 4ul, 5ul};
  this->test_params(tensor_shape);
}

TYPED_TEST(UniformFillerTest, TestFill1D) {
  std::vector<uint32_t> tensor_shape(1ul, 15ul);
  this->test_params(tensor_shape);
}

TYPED_TEST(UniformFillerTest, TestFill2D) {
  std::vector<uint32_t> tensor_shape = {8ul, 3ul};
  this->test_params(tensor_shape);
}

TYPED_TEST(UniformFillerTest, TestFill5D) {
  std::vector<uint32_t> tensor_shape = {2ul, 3ul, 4ul, 5ul, 2ul};
  this->test_params(tensor_shape);
}

}  // namespace mynet
