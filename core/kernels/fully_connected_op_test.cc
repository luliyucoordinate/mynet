// Copyright 2021 coordinate
// Author: coordinate

#include "fully_connected_op.hpp"

#include <memory>
#include <vector>

#include "core/framework/common.hpp"
#include "core/framework/filler.hpp"
#include "core/framework/mynet_test_main.hpp"
#include "core/framework/tensor.hpp"
#include "gradient_check_util.hpp"

namespace mynet {

template <typename TypeParam>
class FullyConnectedOpTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FullyConnectedOpTest()
      : tensor_input_(new Tensor<Dtype>(2, 3, 4, 5)),
        tensor_input_nobatch_(new Tensor<Dtype>(1, 2, 3, 4)),
        tensor_output_(new Tensor<Dtype>()) {
    // fill the values
    FillerParameterT filler_param;
    UniformFiller<Dtype> filler(&filler_param);
    filler.Fill(this->tensor_input_);
    tensor_output_vec_.push_back(tensor_output_);
  }
  virtual ~FullyConnectedOpTest() {
    delete tensor_input_;
    delete tensor_input_nobatch_;
    delete tensor_output_;
  }
  Tensor<Dtype>* const tensor_input_;
  Tensor<Dtype>* const tensor_input_nobatch_;
  Tensor<Dtype>* const tensor_output_;
  std::vector<Tensor<Dtype>*> tensor_input_vec_;
  std::vector<Tensor<Dtype>*> tensor_output_vec_;
};

TYPED_TEST_CASE(FullyConnectedOpTest, TestDtypesAndDevices);

TYPED_TEST(FullyConnectedOpTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_);
  OpParameterT op_param;
  op_param.fully_connected_param = std::make_unique<FullyConnectedParameterT>();
  op_param.fully_connected_param->num_output = 10ul;
  op_param.fully_connected_param->weight_filler =
      std::make_unique<FillerParameterT>();
  op_param.fully_connected_param->weight_filler->type = "constant";
  op_param.fully_connected_param->bias_filler =
      std::make_unique<FillerParameterT>();
  op_param.fully_connected_param->bias_filler->type = "constant";
  std::shared_ptr<FullyConnectedOp<Dtype>> op =
      std::make_shared<FullyConnectedOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(this->tensor_output_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_->height(), 1ul);
  EXPECT_EQ(this->tensor_output_->width(), 1ul);
  EXPECT_EQ(this->tensor_output_->channels(), 10ul);
}

/** @brief TestSetUp while toggling transpose flag
 */
TYPED_TEST(FullyConnectedOpTest, TestSetUpTransposeFalse) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_);
  OpParameterT op_param;
  op_param.fully_connected_param = std::make_unique<FullyConnectedParameterT>();
  op_param.fully_connected_param->num_output = 10ul;
  op_param.fully_connected_param->weight_filler =
      std::make_unique<FillerParameterT>();
  op_param.fully_connected_param->weight_filler->type = "uniform";
  op_param.fully_connected_param->bias_filler =
      std::make_unique<FillerParameterT>();
  op_param.fully_connected_param->bias_filler->type = "uniform";
  op_param.fully_connected_param->bias_filler->min = 1.0f;
  op_param.fully_connected_param->bias_filler->max = 2.0f;
  op_param.fully_connected_param->transpose = false;
  std::shared_ptr<FullyConnectedOp<Dtype>> op =
      std::make_shared<FullyConnectedOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(2ul, this->tensor_output_->num());
  EXPECT_EQ(1ul, this->tensor_output_->height());
  EXPECT_EQ(1ul, this->tensor_output_->width());
  EXPECT_EQ(10ul, this->tensor_output_->channels());
  EXPECT_EQ(2ul, op->tensors()[0]->num_axes());
  EXPECT_EQ(10ul, op->tensors()[0]->shape(0));
  EXPECT_EQ(60ul, op->tensors()[0]->shape(1));
}

/** @brief TestSetUp while toggling transpose flag
 */
TYPED_TEST(FullyConnectedOpTest, TestSetUpTransposeTrue) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_);
  OpParameterT op_param;
  op_param.fully_connected_param = std::make_unique<FullyConnectedParameterT>();
  op_param.fully_connected_param->num_output = 10ul;
  op_param.fully_connected_param->transpose = true;
  op_param.fully_connected_param->weight_filler =
      std::make_unique<FillerParameterT>();
  op_param.fully_connected_param->weight_filler->type = "constant";
  op_param.fully_connected_param->bias_filler =
      std::make_unique<FillerParameterT>();
  op_param.fully_connected_param->bias_filler->type = "constant";
  std::shared_ptr<FullyConnectedOp<Dtype>> op =
      std::make_shared<FullyConnectedOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(2ul, this->tensor_output_->num());
  EXPECT_EQ(1ul, this->tensor_output_->height());
  EXPECT_EQ(1ul, this->tensor_output_->width());
  EXPECT_EQ(10ul, this->tensor_output_->channels());
  EXPECT_EQ(2ul, op->tensors()[0]->num_axes());
  EXPECT_EQ(60ul, op->tensors()[0]->shape(0));
  EXPECT_EQ(10ul, op->tensors()[0]->shape(1));
}

TYPED_TEST(FullyConnectedOpTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_);

  if (Mynet::mode() == Mynet::CPU || sizeof(Dtype) == 4) {
    OpParameterT op_param;
    op_param.fully_connected_param =
        std::make_unique<FullyConnectedParameterT>();
    op_param.fully_connected_param->num_output = 10ul;
    op_param.fully_connected_param->weight_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->bias_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->weight_filler->type = "uniform";
    op_param.fully_connected_param->bias_filler->type = "uniform";
    op_param.fully_connected_param->bias_filler->min = 1.0f;
    op_param.fully_connected_param->bias_filler->max = 2.0f;
    std::shared_ptr<FullyConnectedOp<Dtype>> op =
        std::make_shared<FullyConnectedOp<Dtype>>(&op_param);

    op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    const Dtype* data = this->tensor_output_->cpu_data();
    auto count = this->tensor_output_->count();
    for (uint32_t i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.0f);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

/**
 * @brief Init. an IP op without transpose + random weights,
 * run Forward, save the result.
 * Init. another IP op with transpose.
 * manually copy and transpose the weights from the first IP op,
 * then run Forward on the same input and check that the result is the same
 */
TYPED_TEST(FullyConnectedOpTest, TestForwardTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_);

  if (Mynet::mode() == Mynet::CPU || sizeof(Dtype) == 4) {
    OpParameterT op_param;
    op_param.fully_connected_param =
        std::make_unique<FullyConnectedParameterT>();
    op_param.fully_connected_param->num_output = 10ul;
    op_param.fully_connected_param->weight_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->bias_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->weight_filler->type = "uniform";
    op_param.fully_connected_param->bias_filler->type = "uniform";
    op_param.fully_connected_param->bias_filler->min = 1.0f;
    op_param.fully_connected_param->bias_filler->max = 2.0f;
    std::shared_ptr<FullyConnectedOp<Dtype>> op =
        std::make_shared<FullyConnectedOp<Dtype>>(&op_param);

    op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    auto count = this->tensor_output_->count();
    Tensor<Dtype>* const input = new Tensor<Dtype>();
    input->ReshapeLike(*this->tensor_output_);
    mynet_copy(input->mutable_cpu_data(), this->tensor_output_->cpu_data(),
               count);
    this->tensor_output_vec_.clear();
    this->tensor_output_vec_.push_back(new Tensor<Dtype>());
    op_param.fully_connected_param->transpose = true;
    std::shared_ptr<FullyConnectedOp<Dtype>> ip_t =
        std::make_shared<FullyConnectedOp<Dtype>>(&op_param);

    ip_t->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    auto count_w = op->tensors()[0]->count();
    EXPECT_EQ(count_w, ip_t->tensors()[0]->count());
    // manually copy and transpose the weights from 1st IP op into 2nd
    const Dtype* w = op->tensors()[0]->cpu_data();
    Dtype* w_t = ip_t->tensors()[0]->mutable_cpu_data();
    auto width = op->tensors()[0]->shape(1);
    auto width_t = ip_t->tensors()[0]->shape(1);
    for (uint32_t i = 0; i < count_w; ++i) {
      uint32_t r = i / width;
      uint32_t c = i % width;
      w_t[c * width_t + r] = w[r * width + c];  // copy while transposing
    }
    // copy bias from 1st IP op to 2nd IP op
    ASSERT_EQ(op->tensors()[1]->count(), ip_t->tensors()[1]->count());
    mynet_copy(ip_t->tensors()[1]->mutable_cpu_data(),
               op->tensors()[1]->cpu_data(), op->tensors()[1]->count());
    ip_t->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    EXPECT_EQ(count, this->tensor_output_->count())
        << "Invalid count for input tensor for IP with transpose.";
    Tensor<Dtype>* const output_t = new Tensor<Dtype>();
    output_t->ReshapeLike(*this->tensor_output_vec_[0]);
    mynet_copy(output_t->mutable_cpu_data(),
               this->tensor_output_vec_[0]->cpu_data(), count);
    const Dtype* data = input->cpu_data();
    const Dtype* data_t = output_t->cpu_data();
    for (uint32_t i = 0; i < count; ++i) {
      EXPECT_FLOAT_EQ(data[i], data_t[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(FullyConnectedOpTest, TestForwardNoBatch) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_nobatch_);

  if (Mynet::mode() == Mynet::CPU || sizeof(Dtype) == 4) {
    OpParameterT op_param;
    op_param.fully_connected_param =
        std::make_unique<FullyConnectedParameterT>();
    op_param.fully_connected_param->num_output = 10ul;
    op_param.fully_connected_param->weight_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->bias_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->weight_filler->type = "uniform";
    op_param.fully_connected_param->bias_filler->type = "uniform";
    op_param.fully_connected_param->bias_filler->min = 1.0f;
    op_param.fully_connected_param->bias_filler->max = 2.0f;
    std::shared_ptr<FullyConnectedOp<Dtype>> op =
        std::make_shared<FullyConnectedOp<Dtype>>(&op_param);

    op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    const Dtype* data = this->tensor_output_->cpu_data();
    auto count = this->tensor_output_->count();
    for (uint32_t i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(FullyConnectedOpTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_);

  if (Mynet::mode() == Mynet::CPU || sizeof(Dtype) == 4) {
    OpParameterT op_param;
    op_param.fully_connected_param =
        std::make_unique<FullyConnectedParameterT>();
    op_param.fully_connected_param->num_output = 10ul;
    op_param.fully_connected_param->weight_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->bias_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->weight_filler->type = "gaussian";
    op_param.fully_connected_param->bias_filler->type = "gaussian";
    op_param.fully_connected_param->bias_filler->min = 1.0f;
    op_param.fully_connected_param->bias_filler->max = 2.0f;
    FullyConnectedOp<Dtype> op(&op_param);

    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
                                    this->tensor_output_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(FullyConnectedOpTest, TestGradientTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_);

  if (Mynet::mode() == Mynet::CPU || sizeof(Dtype) == 4) {
    OpParameterT op_param;
    op_param.fully_connected_param =
        std::make_unique<FullyConnectedParameterT>();
    op_param.fully_connected_param->num_output = 10ul;
    op_param.fully_connected_param->weight_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->bias_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->weight_filler->type = "gaussian";
    op_param.fully_connected_param->bias_filler->type = "gaussian";
    op_param.fully_connected_param->bias_filler->min = 1.0f;
    op_param.fully_connected_param->bias_filler->max = 2.0f;
    op_param.fully_connected_param->transpose = true;
    FullyConnectedOp<Dtype> op(&op_param);

    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
                                    this->tensor_output_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(FullyConnectedOpTest, TestBackwardTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_);

  if (Mynet::mode() == Mynet::CPU || sizeof(Dtype) == 4) {
    OpParameterT op_param;
    op_param.fully_connected_param =
        std::make_unique<FullyConnectedParameterT>();
    op_param.fully_connected_param->num_output = 10ul;
    op_param.fully_connected_param->weight_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->bias_filler =
        std::make_unique<FillerParameterT>();
    op_param.fully_connected_param->weight_filler->type = "uniform";
    op_param.fully_connected_param->bias_filler->type = "uniform";
    op_param.fully_connected_param->bias_filler->min = 1.0f;
    op_param.fully_connected_param->bias_filler->max = 2.0f;
    op_param.fully_connected_param->transpose = false;

    std::shared_ptr<FullyConnectedOp<Dtype>> op =
        std::make_shared<FullyConnectedOp<Dtype>>(&op_param);
    op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    // copy input tensor
    Tensor<Dtype> input;
    input.CopyFrom(*this->tensor_output_, false, true);
    // fake input diff
    Tensor<Dtype> diff;
    diff.ReshapeLike(*this->tensor_output_);
    {
      FillerParameterT filler_param;
      UniformFiller<Dtype> filler(&filler_param);
      filler.Fill(&diff);
    }
    mynet_copy(this->tensor_output_vec_[0]->mutable_cpu_diff(), diff.cpu_data(),
               this->tensor_output_vec_[0]->count());
    std::vector<bool> propagate_down(1, true);
    op->Backward(this->tensor_output_vec_, propagate_down,
                 this->tensor_input_vec_);
    // copy first ip's weights and their diffs
    Tensor<Dtype> w;
    w.CopyFrom(*op->tensors()[0], false, true);
    w.CopyFrom(*op->tensors()[0], true, true);
    // copy output diffs
    Tensor<Dtype> input_diff;
    input_diff.CopyFrom(*this->tensor_input_vec_[0], true, true);
    // repeat original input with transposed ip
    this->tensor_output_vec_.clear();
    this->tensor_output_vec_.push_back(new Tensor<Dtype>());
    op_param.fully_connected_param->transpose = true;
    std::shared_ptr<FullyConnectedOp<Dtype>> ip_t =
        std::make_shared<FullyConnectedOp<Dtype>>(&op_param);
    ip_t->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    // manually copy and transpose the weights from 1st IP op into 2nd
    {
      const Dtype* w_src = w.cpu_data();
      Dtype* w_t = ip_t->tensors()[0]->mutable_cpu_data();
      auto width = op->tensors()[0]->shape(1);
      auto width_t = ip_t->tensors()[0]->shape(1);
      for (uint32_t i = 0; i < op->tensors()[0]->count(); ++i) {
        uint32_t r = i / width;
        uint32_t c = i % width;
        w_t[c * width_t + r] = w_src[r * width + c];  // copy while transposing
      }
      // copy bias from 1st IP op to 2nd IP op
      ASSERT_EQ(op->tensors()[1]->count(), ip_t->tensors()[1]->count());
      mynet_copy(ip_t->tensors()[1]->mutable_cpu_data(),
                 op->tensors()[1]->cpu_data(), op->tensors()[1]->count());
    }
    ip_t->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    mynet_copy(this->tensor_output_vec_[0]->mutable_cpu_diff(), diff.cpu_data(),
               this->tensor_output_vec_[0]->count());
    ip_t->Backward(this->tensor_output_vec_, propagate_down,
                   this->tensor_input_vec_);
    const Dtype* data = w.cpu_diff();
    const Dtype* data_t = ip_t->tensors()[0]->cpu_diff();
    auto WIDTH = op->tensors()[0]->shape(1);
    auto WIDTH_T = ip_t->tensors()[0]->shape(1);
    for (uint32_t i = 0; i < op->tensors()[0]->count(); ++i) {
      uint32_t r = i / WIDTH;
      uint32_t c = i % WIDTH;
      EXPECT_NE(Dtype(0.), data[r * WIDTH + c]);
      EXPECT_FLOAT_EQ(data[r * WIDTH + c], data_t[c * WIDTH_T + r]);
    }
    data = input_diff.cpu_diff();
    data_t = this->tensor_input_vec_[0]->cpu_diff();
    for (uint32_t i = 0; i < this->tensor_input_vec_[0]->count(); ++i) {
      EXPECT_NE(Dtype(0.), data[i]);
      EXPECT_FLOAT_EQ(data[i], data_t[i]);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace mynet
