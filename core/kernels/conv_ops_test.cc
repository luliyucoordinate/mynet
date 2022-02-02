// Copyright 2021 coordinate
// Author: coordinate

#include "conv_ops.hpp"

#include <memory>
#include <vector>

#include "core/framework/common.hpp"
#include "core/framework/filler.hpp"
#include "core/framework/mynet_test_main.hpp"
#include "core/framework/tensor.hpp"

namespace mynet {

// Reference conv for checking results:
// accumulate through explicit loop over input, output, and filters.
template <typename Dtype>
void mynet_conv(const Tensor<Dtype>* in, ConvParameterT* conv_param,
                const std::vector<std::shared_ptr<Tensor<Dtype>>>& weights,
                Tensor<Dtype>* out) {
  DCHECK(conv_param);
  const bool has_depth = (out->num_axes() == 5ul);
  if (!has_depth) {
    DCHECK_EQ(4ul, out->num_axes());
  }
  // Kernel size, stride, and pad
  uint32_t kernel_h, kernel_w;
  if (conv_param->kernel_h || conv_param->kernel_w) {
    kernel_h = conv_param->kernel_h;
    kernel_w = conv_param->kernel_w;
  } else {
    kernel_h = kernel_w = conv_param->kernel_size[0];
  }

  uint32_t pad_h, pad_w;
  if (conv_param->pad_h || conv_param->pad_w) {
    pad_h = conv_param->pad_h;
    pad_w = conv_param->pad_w;
  } else {
    pad_h = pad_w = conv_param->pad.size() ? conv_param->pad[0] : 0ul;
  }
  uint32_t stride_h, stride_w;
  if (conv_param->stride_h || conv_param->stride_w) {
    stride_h = conv_param->stride_h;
    stride_w = conv_param->stride_w;
  } else {
    stride_h = stride_w =
        conv_param->stride.size() ? conv_param->stride[0] : 1ul;
  }
  uint32_t dilation_h, dilation_w;
  dilation_h = dilation_w =
      conv_param->dilation.size() ? conv_param->dilation[0] : 1ul;
  uint32_t kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  uint32_t groups = conv_param->group;
  uint32_t o_g = out->shape(1) / groups;
  uint32_t k_g = in->shape(1) / groups;
  uint32_t o_head, k_head;
  // Conv
  std::vector<uint32_t> weight_offset(4 + has_depth);
  std::vector<uint32_t> in_offset(4 + has_depth);
  std::vector<uint32_t> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (uint32_t n = 0; n < out->shape(0); n++) {
    for (uint32_t g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (uint32_t o = 0; o < o_g; o++) {
        for (uint32_t k = 0; k < k_g; k++) {
          for (uint32_t z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (uint32_t y = 0; y < out->shape(2 + has_depth); y++) {
              for (uint32_t x = 0; x < out->shape(3 + has_depth); x++) {
                for (uint32_t r = 0; r < kernel_d; r++) {
                  for (uint32_t p = 0; p < kernel_h; p++) {
                    for (uint32_t q = 0; q < kernel_w; q++) {
                      uint32_t in_z = z * stride_d - pad_d + r * dilation_d;
                      uint32_t in_y = y * stride_h - pad_h + p * dilation_h;
                      uint32_t in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1) &&
                          in_y >= 0 && in_y < in->shape(2 + has_depth) &&
                          in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) {
                          weight_offset[2] = r;
                        }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) {
                          in_offset[2] = in_z;
                        }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) {
                          out_offset[2] = z;
                        }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset) *
                            weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (uint32_t n = 0; n < out->shape(0); n++) {
      for (uint32_t o = 0; o < out->shape(1); o++) {
        for (uint32_t z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (uint32_t y = 0; y < out->shape(2 + has_depth); y++) {
            for (uint32_t x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) {
                out_offset[2] = z;
              }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }
}

template void mynet_conv(
    const Tensor<float>* in, ConvParameterT* conv_param,
    const std::vector<std::shared_ptr<Tensor<float>>>& weights,
    Tensor<float>* out);
template void mynet_conv(
    const Tensor<double>* in, ConvParameterT* conv_param,
    const std::vector<std::shared_ptr<Tensor<double>>>& weights,
    Tensor<double>* out);

template <typename TypeParam>
class ConvOpTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvOpTest()
      : tensor_input_(new Tensor<Dtype>(2ul, 3ul, 6ul, 4ul)),
        tensor_input_2_(new Tensor<Dtype>(2ul, 3ul, 6ul, 4ul)),
        tensor_output_(new Tensor<Dtype>()),
        tensor_output_2_(new Tensor<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameterT filler_param;
    filler_param.value = 1.0f;
    UniformFiller<Dtype> filler(&filler_param);
    filler.Fill(tensor_input_);
    filler.Fill(tensor_input_2_);
    tensor_input_vec_.push_back(tensor_input_);
    tensor_output_vec_.push_back(tensor_output_);
  }

  virtual ~ConvOpTest() {
    delete tensor_input_;
    delete tensor_input_2_;
    delete tensor_output_;
    delete tensor_output_2_;
  }

  virtual Tensor<Dtype>* MakeReferenceInput(Tensor<Dtype>* output) {
    this->ref_tensor_output_.reset(new Tensor<Dtype>());
    this->ref_tensor_output_->ReshapeLike(*output);
    return this->ref_tensor_output_.get();
  }

  Tensor<Dtype>* const tensor_input_;
  Tensor<Dtype>* const tensor_input_2_;
  Tensor<Dtype>* const tensor_output_;
  Tensor<Dtype>* const tensor_output_2_;
  std::shared_ptr<Tensor<Dtype>> ref_tensor_output_;
  std::vector<Tensor<Dtype>*> tensor_input_vec_;
  std::vector<Tensor<Dtype>*> tensor_output_vec_;
};

TYPED_TEST_CASE(ConvOpTest, TestDtypesAndDevices);

TYPED_TEST(ConvOpTest, TestConvSetup) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->weight_filler->type = "constant";
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler->type = "constant";
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->stride.push_back(2ul);
  op_param.conv_param->num_output = 4ul;
  this->tensor_input_vec_.push_back(this->tensor_input_2_);
  this->tensor_output_vec_.push_back(this->tensor_output_2_);
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(this->tensor_output_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_->channels(), 4ul);
  EXPECT_EQ(this->tensor_output_->height(), 2ul);
  EXPECT_EQ(this->tensor_output_->width(), 1ul);
  EXPECT_EQ(this->tensor_output_2_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_2_->channels(), 4ul);
  EXPECT_EQ(this->tensor_output_2_->height(), 2ul);
  EXPECT_EQ(this->tensor_output_2_->width(), 1ul);
  // setting group should not change the shape
  op_param.conv_param->num_output = 3ul;
  op_param.conv_param->group = 3ul;
  op.reset(new ConvOp<Dtype>(&op_param));
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(this->tensor_output_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_->channels(), 3ul);
  EXPECT_EQ(this->tensor_output_->height(), 2ul);
  EXPECT_EQ(this->tensor_output_->width(), 1ul);
  EXPECT_EQ(this->tensor_output_2_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_2_->channels(), 3ul);
  EXPECT_EQ(this->tensor_output_2_->height(), 2ul);
  EXPECT_EQ(this->tensor_output_2_->width(), 1ul);
}

TYPED_TEST(ConvOpTest, TestSimpleConv) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_2_);
  this->tensor_output_vec_.push_back(this->tensor_output_2_);
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->stride.push_back(2ul);
  op_param.conv_param->num_output = 4ul;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->bias_filler->type = "constant";
  op_param.conv_param->bias_filler->value = 0.1f;
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Check against reference conv.
  const Dtype* output_data;
  const Dtype* ref_output_data;
  mynet_conv(this->tensor_input_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_));
  output_data = this->tensor_output_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
  mynet_conv(this->tensor_input_2_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_2_));
  output_data = this->tensor_output_2_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpTest, TestDilatedConv) {
  typedef typename TypeParam::Dtype Dtype;
  std::vector<uint32_t> input_shape = {2ul, 3ul, 8ul, 7ul};
  this->tensor_input_vec_.push_back(this->tensor_input_2_);
  this->tensor_output_vec_.push_back(this->tensor_output_2_);
  for (uint32_t i = 0; i < this->tensor_input_vec_.size(); ++i) {
    this->tensor_input_vec_[i]->Reshape(input_shape);
  }
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->dilation.push_back(2ul);
  op_param.conv_param->num_output = 4ul;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->bias_filler->type = "constant";
  op_param.conv_param->bias_filler->value = 0.1f;
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Check against reference conv.
  const Dtype* output_data;
  const Dtype* ref_output_data;
  mynet_conv(this->tensor_input_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_));
  output_data = this->tensor_output_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
  mynet_conv(this->tensor_input_2_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_2_));
  output_data = this->tensor_output_2_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpTest, Test0DConv) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();

  const uint32_t kNumOutput = 3ul;
  op_param.conv_param->num_output = kNumOutput;
  op_param.conv_param->axis = 3;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->bias_filler->type = "uniform";
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  std::vector<uint32_t> output_shape = this->tensor_input_->shape();
  output_shape[3] = kNumOutput;
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(output_shape, this->tensor_output_->shape());
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Check against reference conv.
  std::vector<uint32_t> weight_offset(2);
  const Tensor<Dtype>* weight = op->tensors()[0].get();
  const Tensor<Dtype>* bias = op->tensors()[1].get();
  const uint32_t num = this->tensor_output_->count(3);
  const uint32_t dim = this->tensor_output_->shape(3);
  const uint32_t input_dim = this->tensor_input_->shape(3);
  for (uint32_t n = 0; n < num; ++n) {
    for (uint32_t d = 0; d < dim; ++d) {
      weight_offset[0] = d;
      Dtype value = bias->cpu_data()[d];
      for (uint32_t input_d = 0; input_d < input_dim; ++input_d) {
        weight_offset[1] = input_d;
        value += weight->data_at(weight_offset) *
                 this->tensor_input_->cpu_data()[n * input_dim + input_d];
      }
      EXPECT_NEAR(value, this->tensor_output_->cpu_data()[n * dim + d], 1e-4);
    }
  }
}

TYPED_TEST(ConvOpTest, TestSimple3DConv) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_2_);
  this->tensor_output_vec_.push_back(this->tensor_output_2_);
  std::vector<uint32_t> input_shape(5);
  input_shape[0] = this->tensor_input_vec_[0]->shape(0);
  input_shape[1] = this->tensor_input_vec_[0]->shape(1);
  input_shape[2] = 5ul;
  input_shape[3] = this->tensor_input_vec_[0]->shape(2);
  input_shape[4] = this->tensor_input_vec_[0]->shape(3);
  FillerParameterT filler_param;
  UniformFiller<Dtype> filler(&filler_param);
  for (uint32_t i = 0; i < this->tensor_input_vec_.size(); ++i) {
    this->tensor_input_vec_[i]->Reshape(input_shape);
    filler.Fill(this->tensor_input_vec_[i]);
  }
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->stride.push_back(2ul);
  op_param.conv_param->num_output = 4ul;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->bias_filler->type = "uniform";
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Check against reference conv.
  const Dtype* output_data;
  const Dtype* ref_output_data;
  mynet_conv(this->tensor_input_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_));
  output_data = this->tensor_output_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
  mynet_conv(this->tensor_input_2_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_2_));
  output_data = this->tensor_output_2_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpTest, TestDilated3DConv) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_2_);
  this->tensor_output_vec_.push_back(this->tensor_output_2_);
  std::vector<uint32_t> input_shape(5);
  input_shape[0] = this->tensor_input_vec_[0]->shape(0);
  input_shape[1] = this->tensor_input_vec_[0]->shape(1);
  input_shape[2] = 6ul;
  input_shape[3] = 7ul;
  input_shape[4] = 8ul;
  FillerParameterT filler_param;
  UniformFiller<Dtype> filler(&filler_param);
  for (uint32_t i = 0; i < this->tensor_input_vec_.size(); ++i) {
    this->tensor_input_vec_[i]->Reshape(input_shape);
    filler.Fill(this->tensor_input_vec_[i]);
  }
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->stride.push_back(2ul);
  op_param.conv_param->num_output = 4ul;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->bias_filler->type = "uniform";
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Check against reference conv.
  const Dtype* output_data;
  const Dtype* ref_output_data;
  mynet_conv(this->tensor_input_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_));
  output_data = this->tensor_output_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
  mynet_conv(this->tensor_input_2_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_2_));
  output_data = this->tensor_output_2_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpTest, Test1x1Conv) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->kernel_size.push_back(1ul);
  op_param.conv_param->stride.push_back(1ul);
  op_param.conv_param->num_output = 4ul;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->bias_filler->type = "constant";
  op_param.conv_param->bias_filler->value = 0.1f;
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Check against reference conv.
  const Dtype* output_data;
  const Dtype* ref_output_data;
  mynet_conv(this->tensor_input_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_));
  output_data = this->tensor_output_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpTest, TestSimpleConvGroup) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->stride.push_back(2ul);
  op_param.conv_param->num_output = 3ul;
  op_param.conv_param->group = 3ul;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->bias_filler->type = "constant";
  op_param.conv_param->bias_filler->value = 0.1f;
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Check against reference conv.
  const Dtype* output_data;
  const Dtype* ref_output_data;
  mynet_conv(this->tensor_input_, op_param.conv_param.get(), op->tensors(),
             this->MakeReferenceInput(this->tensor_output_));
  output_data = this->tensor_output_->cpu_data();
  ref_output_data = this->ref_tensor_output_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], ref_output_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpTest, TestSobelConv) {
  // Test separable conv by computing the Sobel operator
  // as a single filter then comparing the result
  // as the conv of two rectangular filters.
  typedef typename TypeParam::Dtype Dtype;
  // Fill inputs with identical Gaussian noise.
  FillerParameterT filler_param;
  filler_param.value = 1.0f;
  auto filler = std::make_shared<UniformFiller<Dtype>>(&filler_param);
  filler->Fill(this->tensor_input_);
  this->tensor_input_2_->CopyFrom(*this->tensor_input_);
  // Compute Sobel G_x operator as 3 x 3 conv.
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->weight_filler->type = "constant";
  op_param.conv_param->bias_filler->type = "constant";
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->stride.push_back(2ul);
  op_param.conv_param->num_output = 1ul;
  op_param.conv_param->bias_term = false;
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->tensors().resize(1);
  op->tensors()[0].reset(new Tensor<Dtype>(1ul, 3ul, 3ul, 3ul));
  Dtype* weights = op->tensors()[0]->mutable_cpu_data();
  for (uint32_t c = 0; c < 3; ++c) {
    uint32_t i = c * 9;  // 3 x 3 filter
    weights[i + 0] = -1;
    weights[i + 1] = 0;
    weights[i + 2] = 1;
    weights[i + 3] = -2;
    weights[i + 4] = 0;
    weights[i + 5] = 2;
    weights[i + 6] = -1;
    weights[i + 7] = 0;
    weights[i + 8] = 1;
  }
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convs.
  // (1) the [1 2 1] column filter
  std::vector<Tensor<Dtype>*> sep_tensor_input_vec;
  std::vector<Tensor<Dtype>*> sep_tensor_output_vec;
  auto tensor_sep = std::make_shared<Tensor<Dtype>>();
  sep_tensor_input_vec.push_back(this->tensor_input_2_);
  sep_tensor_output_vec.push_back(this->tensor_output_2_);
  op_param.conv_param->kernel_size.clear();
  op_param.conv_param->stride.clear();
  op_param.conv_param->kernel_h = 3ul;
  op_param.conv_param->kernel_w = 1ul;
  op_param.conv_param->stride_h = 2ul;
  op_param.conv_param->stride_w = 1ul;
  op_param.conv_param->num_output = 1ul;
  op_param.conv_param->bias_term = false;
  op.reset(new ConvOp<Dtype>(&op_param));
  op->tensors().resize(1);
  op->tensors()[0].reset(new Tensor<Dtype>(1ul, 3ul, 3ul, 1ul));
  Dtype* weights_1 = op->tensors()[0]->mutable_cpu_data();
  for (uint32_t c = 0; c < 3; ++c) {
    uint32_t i = c * 3;  // 3 x 1 filter
    weights_1[i + 0] = 1;
    weights_1[i + 1] = 2;
    weights_1[i + 2] = 1;
  }
  op->SetUp(sep_tensor_input_vec, sep_tensor_output_vec);
  op->Forward(sep_tensor_input_vec, sep_tensor_output_vec);
  // (2) the [-1 0 1] row filter
  tensor_sep->CopyFrom(*this->tensor_output_2_, false, true);
  sep_tensor_input_vec.clear();
  sep_tensor_input_vec.push_back(tensor_sep.get());
  op_param.conv_param->kernel_h = 1ul;
  op_param.conv_param->kernel_w = 3ul;
  op_param.conv_param->stride_h = 1ul;
  op_param.conv_param->stride_w = 2ul;
  op_param.conv_param->num_output = 1ul;
  op_param.conv_param->bias_term = false;
  op.reset(new ConvOp<Dtype>(&op_param));
  op->tensors().resize(1);
  op->tensors()[0].reset(new Tensor<Dtype>(1ul, 1ul, 1ul, 3ul));
  Dtype* weights_2 = op->tensors()[0]->mutable_cpu_data();
  weights_2[0] = -1;
  weights_2[1] = 0;
  weights_2[2] = 1;
  op->SetUp(sep_tensor_input_vec, sep_tensor_output_vec);
  op->Forward(sep_tensor_input_vec, sep_tensor_output_vec);
  // Test equivalence of full and separable filters.
  const Dtype* output_data = this->tensor_output_->cpu_data();
  const Dtype* sep_output_data = this->tensor_output_2_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_output_->count(); ++i) {
    EXPECT_NEAR(output_data[i], sep_output_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpTest, TestNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const uint32_t kernel_h = 11ul;
  const uint32_t kernel_w = 13ul;
  std::vector<uint32_t> input_shape(4);
  input_shape[0] = 15ul;
  input_shape[1] = 18ul;
  input_shape[2] = kernel_h * 2ul;
  input_shape[3] = kernel_w * 2ul;
  FillerParameterT filler_param;
  UniformFiller<Dtype> filler(&filler_param);
  for (uint32_t i = 0; i < this->tensor_input_vec_.size(); ++i) {
    this->tensor_input_vec_[i]->Reshape(input_shape);
    filler.Fill(this->tensor_input_vec_[i]);
  }
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->num_output = 12ul;
  op_param.conv_param->bias_term = false;
  op_param.conv_param->group = 6ul;
  op_param.conv_param->kernel_h = kernel_h;
  op_param.conv_param->kernel_w = kernel_w;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->bias_filler->type = "constant";
  Tensor<Dtype> weights;
  Tensor<Dtype> output_diff;
  // Shape and fill weights and output_diff.
  bool copy_diff;
  bool reshape;
  {
    ConvOp<Dtype> op(&op_param);
    op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    output_diff.ReshapeLike(*this->tensor_output_);
    filler.Fill(&output_diff);
    ASSERT_EQ(1, op.tensors().size());
    copy_diff = false;
    reshape = true;
    weights.CopyFrom(*op.tensors()[0], copy_diff, reshape);
  }
  std::vector<bool> propagate_down(1, true);
  Tensor<Dtype> result_2d;
  Tensor<Dtype> backward_result_2d;
  Tensor<Dtype> backward_weight_result_2d;
  // Test with 2D im2col
  {
    mynet_set(this->tensor_output_->mutable_cpu_data(), Dtype(0),
              this->tensor_output_->count());
    mynet_set(this->tensor_input_->mutable_cpu_diff(), Dtype(0),
              this->tensor_input_->count());
    mynet_set(weights.mutable_cpu_diff(), Dtype(0), weights.count());
    // Do SetUp and Forward; save Forward result in result_2d.
    op_param.conv_param->force_nd_im2col = false;
    ConvOp<Dtype> conv_2d(&op_param);
    conv_2d.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    ASSERT_EQ(1ul, conv_2d.tensors().size());
    copy_diff = false;
    reshape = false;
    conv_2d.tensors()[0]->CopyFrom(weights, copy_diff, reshape);
    conv_2d.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    copy_diff = false;
    reshape = true;
    result_2d.CopyFrom(*this->tensor_output_, copy_diff, reshape);
    // Copy pre-generated output diff into actual output diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->tensor_output_->shape(), output_diff.shape());
    mynet_copy(this->tensor_output_->mutable_cpu_diff(), output_diff.cpu_data(),
               output_diff.count());
    conv_2d.Backward(this->tensor_output_vec_, propagate_down,
                     this->tensor_input_vec_);
    copy_diff = true;
    reshape = true;
    backward_result_2d.CopyFrom(*this->tensor_input_, copy_diff, reshape);
    backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
  }
  Tensor<Dtype> result_nd;
  Tensor<Dtype> backward_result_nd;
  Tensor<Dtype> backward_weight_result_nd;
  // Test with ND im2col
  {
    mynet_set(this->tensor_output_->mutable_cpu_data(), Dtype(0),
              this->tensor_output_->count());
    mynet_set(this->tensor_input_->mutable_cpu_diff(), Dtype(0),
              this->tensor_input_->count());
    mynet_set(weights.mutable_cpu_diff(), Dtype(0), weights.count());
    // Do SetUp and Forward; save Forward result in result_nd.
    op_param.conv_param->force_nd_im2col = true;
    ConvOp<Dtype> conv_nd(&op_param);
    conv_nd.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    ASSERT_EQ(1ul, conv_nd.tensors().size());
    copy_diff = false;
    reshape = false;
    conv_nd.tensors()[0]->CopyFrom(weights, copy_diff, reshape);
    conv_nd.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    copy_diff = false;
    reshape = true;
    result_nd.CopyFrom(*this->tensor_output_, copy_diff, reshape);
    // Copy pre-generated output diff into actual output diff;
    // do Backward and save result in backward_result_nd.
    ASSERT_EQ(this->tensor_output_->shape(), output_diff.shape());
    mynet_copy(this->tensor_output_->mutable_cpu_diff(), output_diff.cpu_data(),
               output_diff.count());
    conv_nd.Backward(this->tensor_output_vec_, propagate_down,
                     this->tensor_input_vec_);
    copy_diff = true;
    reshape = true;
    backward_result_nd.CopyFrom(*this->tensor_input_, copy_diff, reshape);
    backward_weight_result_nd.CopyFrom(weights, copy_diff, reshape);
  }
  ASSERT_EQ(result_nd.count(), result_2d.count());
  for (uint32_t i = 0; i < result_2d.count(); ++i) {
    EXPECT_EQ(result_2d.cpu_data()[i], result_nd.cpu_data()[i]);
  }
  ASSERT_EQ(backward_result_nd.count(), backward_result_2d.count());
  for (uint32_t i = 0; i < backward_result_2d.count(); ++i) {
    EXPECT_FLOAT_EQ(backward_result_2d.cpu_diff()[i],
                    backward_result_nd.cpu_diff()[i]);
  }
  ASSERT_EQ(backward_weight_result_nd.count(),
            backward_weight_result_2d.count());
  for (uint32_t i = 0; i < backward_weight_result_2d.count(); ++i) {
    EXPECT_EQ(backward_weight_result_2d.cpu_diff()[i],
              backward_weight_result_nd.cpu_diff()[i]);
  }
}

// TYPED_TEST(ConvOpTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpParameterT op_param;
//   op_param.conv_param = std::make_unique<ConvParameterT>();
//   op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   this->tensor_input_vec_.push_back(this->tensor_input_2_);
//   this->tensor_output_vec_.push_back(this->tensor_output_2_);
//   op_param.conv_param->kernel_size.push_back(3ul);
//   op_param.conv_param->stride.push_back(2ul);
//   op_param.conv_param->num_output = 2ul;
//   op_param.conv_param->weight_filler->type = "uniform";
//   op_param.conv_param->bias_filler->type = "uniform";
//   ConvOp<Dtype> op(&op_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
//       this->tensor_output_vec_);
// }

// TYPED_TEST(ConvOpTest, TestDilatedGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpParameterT op_param;
//   op_param.conv_param = std::make_unique<ConvParameterT>();
//   op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   std::vector<uint32_t> input_shape;
//   input_shape.push_back(2ul);
//   input_shape.push_back(3ul);
//   input_shape.push_back(5ul);
//   input_shape.push_back(6ul);
//   for (uint32_t i = 0; i < this->tensor_input_vec_.size(); ++i) {
//     this->tensor_input_vec_[i]->Reshape(input_shape);
//   }
//   op_param.conv_param->kernel_size.push_back(3ul);
//   op_param.conv_param->dilation.push_back(2ul);
//   op_param.conv_param->num_output = 2ul;
//   op_param.conv_param->weight_filler->type = "uniform";
//   op_param.conv_param->bias_filler->type = "uniform";
//   ConvOp<Dtype> op(&op_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
//                                   this->tensor_output_vec_);
// }

// TYPED_TEST(ConvOpTest, TestGradient3D) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpParameterT op_param;
//   op_param.conv_param = std::make_unique<ConvParameterT>();
//   op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   std::vector<uint32_t> input_shape(5);
//   input_shape[0] = this->tensor_input_vec_[0]->shape(0);
//   input_shape[1] = this->tensor_input_vec_[0]->shape(1);
//   input_shape[2] = 5ul;
//   input_shape[3] = this->tensor_input_vec_[0]->shape(2);
//   input_shape[4] = this->tensor_input_vec_[0]->shape(3);
//   FillerParameterT filler_param;
//   UniformFiller<Dtype> filler(&filler_param);
//   for (uint32_t i = 0; i < this->tensor_input_vec_.size(); ++i) {
//     this->tensor_input_vec_[i]->Reshape(input_shape);
//     filler.Fill(this->tensor_input_vec_[i]);
//   }
//   op_param.conv_param->kernel_size.push_back(3ul);
//   op_param.conv_param->stride.push_back(2ul);
//   op_param.conv_param->num_output = 2ul;
//   op_param.conv_param->weight_filler->type = "uniform";
//   op_param.conv_param->bias_filler->type = "uniform";
//   ConvOp<Dtype> op(&op_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
//       this->tensor_output_vec_);
// }

// TYPED_TEST(ConvOpTest, Test1x1Gradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpParameterT op_param;
//   op_param.conv_param = std::make_unique<ConvParameterT>();
//   op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   this->tensor_input_vec_.push_back(this->tensor_input_2_);
//   this->tensor_output_vec_.push_back(this->tensor_output_2_);
//   op_param.conv_param->kernel_size.push_back(1ul);
//   op_param.conv_param->stride.push_back(1ul);
//   op_param.conv_param->num_output = 2ul;
//   op_param.conv_param->weight_filler->type = "uniform";
//   op_param.conv_param->bias_filler->type = "uniform";
//   ConvOp<Dtype> op(&op_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
//       this->tensor_output_vec_);
// }

// TYPED_TEST(ConvOpTest, TestGradientGroup) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpParameterT op_param;
//   op_param.conv_param = std::make_unique<ConvParameterT>();
//   op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   op_param.conv_param->kernel_size.push_back(3ul);
//   op_param.conv_param->stride.push_back(2ul);
//   op_param.conv_param->num_output = 3ul;
//   op_param.conv_param->group = 3ul;
//   op_param.conv_param->weight_filler->type = "uniform";
//   op_param.conv_param->bias_filler->type = "uniform";
//   ConvOp<Dtype> op(&op_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
//       this->tensor_output_vec_);
// }

TYPED_TEST(ConvOpTest, TestDeConvSetup) {
  typedef typename TypeParam::Dtype Dtype;
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->weight_filler->type = "constant";
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler->type = "constant";
  op_param.conv_param->transpose = true;
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->stride.push_back(2ul);
  op_param.conv_param->num_output = 4ul;
  this->tensor_input_vec_.push_back(this->tensor_input_2_);
  this->tensor_output_vec_.push_back(this->tensor_output_2_);
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(this->tensor_output_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_->channels(), 4ul);
  EXPECT_EQ(this->tensor_output_->height(), 13ul);
  EXPECT_EQ(this->tensor_output_->width(), 9ul);
  EXPECT_EQ(this->tensor_output_2_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_2_->channels(), 4ul);
  EXPECT_EQ(this->tensor_output_2_->height(), 13ul);
  EXPECT_EQ(this->tensor_output_2_->width(), 9ul);
  // setting group should not change the shape
  op_param.conv_param->num_output = 3ul;
  op_param.conv_param->group = 3ul;
  op.reset(new ConvOp<Dtype>(&op_param));
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  EXPECT_EQ(this->tensor_output_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_->channels(), 3ul);
  EXPECT_EQ(this->tensor_output_->height(), 13ul);
  EXPECT_EQ(this->tensor_output_->width(), 9ul);
  EXPECT_EQ(this->tensor_output_2_->num(), 2ul);
  EXPECT_EQ(this->tensor_output_2_->channels(), 3ul);
  EXPECT_EQ(this->tensor_output_2_->height(), 13ul);
  EXPECT_EQ(this->tensor_output_2_->width(), 9ul);
}

TYPED_TEST(ConvOpTest, TestSimpleDeconv) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_input_vec_.push_back(this->tensor_input_2_);
  this->tensor_output_vec_.push_back(this->tensor_output_2_);
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->kernel_size.push_back(3ul);
  op_param.conv_param->stride.push_back(2ul);
  op_param.conv_param->num_output = 4ul;
  op_param.conv_param->weight_filler->type = "constant";
  op_param.conv_param->weight_filler->value = 1.0f;
  op_param.conv_param->bias_filler->type = "constant";
  op_param.conv_param->bias_filler->value = 0.1f;
  op_param.conv_param->transpose = true;
  std::shared_ptr<Op<Dtype>> op = std::make_shared<ConvOp<Dtype>>(&op_param);
  op->SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
  // constant-fill the input tensors
  FillerParameterT filler_param;
  filler_param.value = 1.0f;
  ConstantFiller<Dtype> filler(&filler_param);
  filler.Fill(this->tensor_input_);
  filler.Fill(this->tensor_input_2_);
  op->Forward(this->tensor_input_vec_, this->tensor_output_vec_);
  // simply check that accumulation works with overlapping filters
  const Dtype* output_data = this->tensor_output_->cpu_data();
  for (uint32_t n = 0; n < this->tensor_output_->num(); ++n) {
    for (uint32_t c = 0; c < this->tensor_output_->channels(); ++c) {
      for (uint32_t h = 0; h < this->tensor_output_->height(); ++h) {
        for (uint32_t w = 0; w < this->tensor_output_->width(); ++w) {
          Dtype expected = 3.1;
          bool h_overlap =
              h % 2 == 0 && h > 0 && h < this->tensor_output_->height() - 1;
          bool w_overlap =
              w % 2 == 0 && w > 0 && w < this->tensor_output_->width() - 1;
          if (h_overlap && w_overlap) {
            expected += 9;
          } else if (h_overlap || w_overlap) {
            expected += 3;
          }
          EXPECT_NEAR(output_data[this->tensor_output_->offset(n, c, h, w)],
                      expected, 1e-4);
        }
      }
    }
  }
}

// TYPED_TEST(ConvOpTest, TestDeconvGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpParameterT op_param;
//   op_param.conv_param = std::make_unique<ConvParameterT>();
//   op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   this->tensor_input_vec_.push_back(this->tensor_input_2_);
//   this->tensor_output_vec_.push_back(this->tensor_output_2_);
//   op_param.conv_param->kernel_size.push_back(2ul);
//   op_param.conv_param->stride.push_back(1ul);
//   op_param.conv_param->num_output = 1ul;
//   op_param.conv_param->weight_filler->type = "uniform";
//   op_param.conv_param->bias_filler->type = "uniform";
//   op_param.conv_param->transpose = true;
//   ConvOp<Dtype> op(&op_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
//       this->tensor_output_vec_);
// }

TYPED_TEST(ConvOpTest, TestDeconvNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const uint32_t kernel_h = 11ul;
  const uint32_t kernel_w = 13ul;
  std::vector<uint32_t> input_shape(4);
  input_shape[0] = 15ul;
  input_shape[1] = 12ul;
  input_shape[2] = kernel_h * 2;
  input_shape[3] = kernel_w * 2;
  FillerParameterT filler_param;
  UniformFiller<Dtype> filler(&filler_param);
  for (uint32_t i = 0; i < this->tensor_input_vec_.size(); ++i) {
    this->tensor_input_vec_[i]->Reshape(input_shape);
    filler.Fill(this->tensor_input_vec_[i]);
  }
  OpParameterT op_param;
  op_param.conv_param = std::make_unique<ConvParameterT>();
  op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  op_param.conv_param->num_output = 18ul;
  op_param.conv_param->bias_term = false;
  op_param.conv_param->group = 6ul;
  op_param.conv_param->kernel_h = kernel_h;
  op_param.conv_param->kernel_w = kernel_w;
  op_param.conv_param->weight_filler->type = "uniform";
  op_param.conv_param->transpose = true;
  Tensor<Dtype> weights;
  Tensor<Dtype> output_diff;
  // Shape and fill weights and output_diff.
  bool copy_diff;
  bool reshape;
  {
    ConvOp<Dtype> op(&op_param);
    op.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    output_diff.ReshapeLike(*this->tensor_output_);
    filler.Fill(&output_diff);
    ASSERT_EQ(1ul, op.tensors().size());
    copy_diff = false;
    reshape = true;
    weights.CopyFrom(*op.tensors()[0], copy_diff, reshape);
  }
  std::vector<bool> propagate_down(1, true);
  Tensor<Dtype> result_2d;
  Tensor<Dtype> backward_result_2d;
  Tensor<Dtype> backward_weight_result_2d;
  // Test with 2D im2col
  {
    mynet_set(this->tensor_output_->mutable_cpu_data(), Dtype(0),
              this->tensor_output_->count());
    mynet_set(this->tensor_input_->mutable_cpu_diff(), Dtype(0),
              this->tensor_input_->count());
    mynet_set(weights.mutable_cpu_diff(), Dtype(0), weights.count());
    // Do SetUp and Forward; save Forward result in result_2d.
    op_param.conv_param->force_nd_im2col = false;
    ConvOp<Dtype> op_2d(&op_param);
    op_2d.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    ASSERT_EQ(1ul, op_2d.tensors().size());
    copy_diff = false;
    reshape = false;
    op_2d.tensors()[0]->CopyFrom(weights, copy_diff, reshape);
    op_2d.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    copy_diff = false;
    reshape = true;
    result_2d.CopyFrom(*this->tensor_output_, copy_diff, reshape);
    // Copy pre-generated output diff into actual output diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->tensor_output_->shape(), output_diff.shape());
    mynet_copy(this->tensor_output_->mutable_cpu_diff(), output_diff.cpu_data(),
               output_diff.count());
    op_2d.Backward(this->tensor_output_vec_, propagate_down,
                   this->tensor_input_vec_);
    copy_diff = true;
    reshape = true;
    backward_result_2d.CopyFrom(*this->tensor_input_, copy_diff, reshape);
    backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
  }
  Tensor<Dtype> result_nd;
  Tensor<Dtype> backward_result_nd;
  Tensor<Dtype> backward_weight_result_nd;
  // Test with ND im2col
  {
    mynet_set(this->tensor_output_->mutable_cpu_data(), Dtype(0),
              this->tensor_output_->count());
    mynet_set(this->tensor_input_->mutable_cpu_diff(), Dtype(0),
              this->tensor_input_->count());
    mynet_set(weights.mutable_cpu_diff(), Dtype(0), weights.count());
    // Do SetUp and Forward; save Forward result in result_nd.
    op_param.conv_param->force_nd_im2col = true;
    ConvOp<Dtype> op_nd(&op_param);
    op_nd.SetUp(this->tensor_input_vec_, this->tensor_output_vec_);
    ASSERT_EQ(1, op_nd.tensors().size());
    copy_diff = false;
    reshape = false;
    op_nd.tensors()[0]->CopyFrom(weights, copy_diff, reshape);
    op_nd.Forward(this->tensor_input_vec_, this->tensor_output_vec_);
    copy_diff = false;
    reshape = true;
    result_nd.CopyFrom(*this->tensor_output_, copy_diff, reshape);
    // Copy pre-generated output diff into actual output diff;
    // do Backward and save result in backward_result_nd.
    ASSERT_EQ(this->tensor_output_->shape(), output_diff.shape());
    mynet_copy(this->tensor_output_->mutable_cpu_diff(), output_diff.cpu_data(),
               output_diff.count());
    op_nd.Backward(this->tensor_output_vec_, propagate_down,
                   this->tensor_input_vec_);
    copy_diff = true;
    reshape = true;
    backward_result_nd.CopyFrom(*this->tensor_input_, copy_diff, reshape);
    backward_weight_result_nd.CopyFrom(weights, copy_diff, reshape);
  }
  ASSERT_EQ(result_nd.count(), result_2d.count());
  for (uint32_t i = 0; i < result_2d.count(); ++i) {
    EXPECT_EQ(result_2d.cpu_data()[i], result_nd.cpu_data()[i]);
  }
  ASSERT_EQ(backward_result_nd.count(), backward_result_2d.count());
  for (uint32_t i = 0; i < backward_result_2d.count(); ++i) {
    EXPECT_EQ(backward_result_2d.cpu_diff()[i],
              backward_result_nd.cpu_diff()[i]);
  }
  ASSERT_EQ(backward_weight_result_nd.count(),
            backward_weight_result_2d.count());
  for (uint32_t i = 0; i < backward_weight_result_2d.count(); ++i) {
    EXPECT_EQ(backward_weight_result_2d.cpu_diff()[i],
              backward_weight_result_nd.cpu_diff()[i]);
  }
}

// TYPED_TEST(ConvOpTest, TestDeconvGradient3D) {
//   typedef typename TypeParam::Dtype Dtype;
//   std::vector<uint32_t> input_shape(5);
//   input_shape[0] = this->tensor_input_vec_[0]->shape(0);
//   input_shape[1] = this->tensor_input_vec_[0]->shape(1);
//   input_shape[2] = 2ul;
//   input_shape[3] = 3ul;
//   input_shape[4] = 2ul;
//   FillerParameterT filler_param;
//   UniformFiller<Dtype> filler(&filler_param);
//   for (uint32_t i = 0; i < this->tensor_input_vec_.size(); ++i) {
//     this->tensor_input_vec_[i]->Reshape(input_shape);
//     filler.Fill(this->tensor_input_vec_[i]);
//   }
//   OpParameterT op_param;
//   op_param.conv_param = std::make_unique<ConvParameterT>();
//   op_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   op_param.conv_param->kernel_size.push_back(2ul);
//   op_param.conv_param->stride.push_back(2ul);
//   op_param.conv_param->pad.push_back(1ul);
//   op_param.conv_param->num_output = 2ul;
//   op_param.conv_param->weight_filler->type = "uniform";
//   op_param.conv_param->bias_filler->type = "uniform";
//   op_param.conv_param->transpose = true;
//   ConvOp<Dtype> op(&op_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&op, this->tensor_input_vec_,
//       this->tensor_output_vec_);
// }

}  // namespace mynet
