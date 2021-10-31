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
    filler_param_ = std::make_shared<FillerParameterT>();
    filler_param_->value = 10.0f;
    filler_ = std::make_shared<ConstantFiller<Dtype>>(filler_param_.get());
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
    filler_param_ = std::make_shared<FillerParameterT>();
    filler_param_->min = 1.0;
    filler_param_->max = 2.0;
    filler_ = std::make_shared<UniformFiller<Dtype>>(filler_param_.get());
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
  std::vector<uint32_t> tensor_shape = {15ul};
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

template <typename Dtype>
class GaussianFillerTest : public ::testing::Test {
 protected:
  GaussianFillerTest() : tensor_(new Tensor<Dtype>()) {
    filler_param_ = std::make_shared<FillerParameterT>();
    filler_param_->mean = 10.0;
    filler_param_->std = 0.1;
    filler_ = std::make_shared<GaussianFiller<Dtype>>(filler_param_.get());
  }
  virtual void test_params(const std::vector<uint32_t>& shape,
                           Dtype tolerance = Dtype(5),
                           uint32_t repetitions = 100) {
    // Tests for statistical properties should be ran multiple times.
    EXPECT_TRUE(tensor_);
    tensor_->Reshape(shape);
    for (uint32_t i = 0; i < repetitions; ++i) {
      test_params_iter(shape, tolerance);
    }
  }
  virtual void test_params_iter(const std::vector<uint32_t>& shape,
                                Dtype tolerance) {
    // This test has a configurable tolerance parameter - by default it was
    // equal to 5.0 which is very loose - allowing some tuning (e.g. for tests
    // on smaller blobs the actual variance will be larger than desired, so the
    // tolerance can be increased to account for that).
    filler_->Fill(tensor_);
    uint32_t count = tensor_->count();
    const Dtype* data = tensor_->cpu_data();
    Dtype mean = Dtype(0);
    Dtype var = Dtype(0);
    for (uint32_t i = 0; i < count; ++i) {
      mean += data[i];
      var += data[i] * data[i];
    }
    mean /= count;
    var /= count;
    var -= mean * mean;
    EXPECT_GE(mean, filler_param_->mean - filler_param_->std * tolerance);
    EXPECT_LE(mean, filler_param_->mean + filler_param_->std * tolerance);
    Dtype target_var = filler_param_->std * filler_param_->std;
    EXPECT_GE(var, target_var / tolerance);
    EXPECT_LE(var, target_var * tolerance);
  }
  virtual ~GaussianFillerTest() { delete tensor_; }
  Tensor<Dtype>* const tensor_;
  std::shared_ptr<FillerParameterT> filler_param_;
  std::shared_ptr<GaussianFiller<Dtype>> filler_;
};

TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);

TYPED_TEST(GaussianFillerTest, TestFill) {
  std::vector<uint32_t> tensor_shape = {2ul, 3ul, 4ul, 5ul};
  TypeParam tolerance = TypeParam(3);  // enough for a 120-element tensor
  this->test_params(tensor_shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill1D) {
  std::vector<uint32_t> tensor_shape = {125ul};
  TypeParam tolerance = TypeParam(3);
  this->test_params(tensor_shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill2D) {
  std::vector<uint32_t> tensor_shape = {8ul, 15ul};
  TypeParam tolerance = TypeParam(3);
  this->test_params(tensor_shape, tolerance);
}

TYPED_TEST(GaussianFillerTest, TestFill5D) {
  std::vector<uint32_t> tensor_shape = {2ul, 3ul, 4ul, 5ul, 2ul};
  TypeParam tolerance = TypeParam(2);
  this->test_params(tensor_shape, tolerance);
}

}  // namespace mynet
