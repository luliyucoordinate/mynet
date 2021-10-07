#include <vector>

#include "conv_ops.hpp"
#include "core/framework/tensor.hpp"
#include "core/framework/common.hpp"
#include "core/framework/filler.hpp"
#include "core/framework/mynet_test_main.hpp"


namespace mynet {

// Reference conv for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void mynet_conv(const Tensor<Dtype>* in, ConvParameterT* conv_param,
    const std::vector<std::shared_ptr<Tensor<Dtype>>>& weights,
    Tensor<Dtype>* out) {
  DCHECK(conv_param);
  const bool has_depth = (out->num_axes() == 5ul);
  if (!has_depth) { DCHECK_EQ(4ul, out->num_axes()); }
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
    stride_h = stride_w = conv_param->stride.size() ? conv_param->stride[0] : 1ul;
  }
  uint32_t dilation_h, dilation_w;
  dilation_h = dilation_w = conv_param->dilation.size() ?
                            conv_param->dilation[0] : 1ul;
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
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
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
              if (has_depth) { out_offset[2] = z; }
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

template void mynet_conv(const Tensor<float>* in,
    ConvParameterT* conv_param,
    const std::vector<std::shared_ptr<Tensor<float> > >& weights,
    Tensor<float>* out);
template void mynet_conv(const Tensor<double>* in,
    ConvParameterT* conv_param,
    const std::vector<std::shared_ptr<Tensor<double> > >& weights,
    Tensor<double>* out);

template <typename TypeParam>
class ConvOpsTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ConvOpsTest()
      : tensor_bottom_(new Tensor<Dtype>(2ul, 3ul, 6ul, 4ul)),
        tensor_bottom_2_(new Tensor<Dtype>(2ul, 3ul, 6ul, 4ul)),
        tensor_top_(new Tensor<Dtype>()),
        tensor_top_2_(new Tensor<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameterT filler_param;
    filler_param.value = 1.0f;
    UniformFiller<Dtype> filler(&filler_param);
    filler.Fill(tensor_bottom_);
    filler.Fill(tensor_bottom_2_);
    tensor_bottom_vec_.push_back(tensor_bottom_);
    tensor_top_vec_.push_back(tensor_top_);
  }

  virtual ~ConvOpsTest() {
    delete tensor_bottom_;
    delete tensor_bottom_2_;
    delete tensor_top_;
    delete tensor_top_2_;
  }

  virtual Tensor<Dtype>* MakeReferenceTop(Tensor<Dtype>* top) {
    this->ref_tensor_top_.reset(new Tensor<Dtype>());
    this->ref_tensor_top_->ReshapeLike(*top);
    return this->ref_tensor_top_.get();
  }

  Tensor<Dtype>* const tensor_bottom_;
  Tensor<Dtype>* const tensor_bottom_2_;
  Tensor<Dtype>* const tensor_top_;
  Tensor<Dtype>* const tensor_top_2_;
  std::shared_ptr<Tensor<Dtype>> ref_tensor_top_;
  std::vector<Tensor<Dtype>*> tensor_bottom_vec_;
  std::vector<Tensor<Dtype>*> tensor_top_vec_;
};

TYPED_TEST_CASE(ConvOpsTest, TestDtypesAndDevices);

TYPED_TEST(ConvOpsTest, TestConvSetup) {
  typedef typename TypeParam::Dtype Dtype;
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->weight_filler->type = "constant";
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler->type = "constant";
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->stride.push_back(2ul);
  ops_param.conv_param->num_output = 4ul;
  this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
  this->tensor_top_vec_.push_back(this->tensor_top_2_);
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  EXPECT_EQ(this->tensor_top_->num(), 2ul);
  EXPECT_EQ(this->tensor_top_->channels(), 4ul);
  EXPECT_EQ(this->tensor_top_->height(), 2ul);
  EXPECT_EQ(this->tensor_top_->width(), 1ul);
  EXPECT_EQ(this->tensor_top_2_->num(), 2ul);
  EXPECT_EQ(this->tensor_top_2_->channels(), 4ul);
  EXPECT_EQ(this->tensor_top_2_->height(), 2ul);
  EXPECT_EQ(this->tensor_top_2_->width(), 1ul);
  // setting group should not change the shape
  ops_param.conv_param->num_output = 3ul;
  ops_param.conv_param->group = 3ul;
  ops.reset(new ConvOps<Dtype>(&ops_param));
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  EXPECT_EQ(this->tensor_top_->num(), 2ul);
  EXPECT_EQ(this->tensor_top_->channels(), 3ul);
  EXPECT_EQ(this->tensor_top_->height(), 2ul);
  EXPECT_EQ(this->tensor_top_->width(), 1ul);
  EXPECT_EQ(this->tensor_top_2_->num(), 2ul);
  EXPECT_EQ(this->tensor_top_2_->channels(), 3ul);
  EXPECT_EQ(this->tensor_top_2_->height(), 2ul);
  EXPECT_EQ(this->tensor_top_2_->width(), 1ul);
}

TYPED_TEST(ConvOpsTest, TestSimpleConv) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
  this->tensor_top_vec_.push_back(this->tensor_top_2_);
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->stride.push_back(2ul);
  ops_param.conv_param->num_output = 4ul;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->bias_filler->type = "constant";
  ops_param.conv_param->bias_filler->value = 0.1f;
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // Check against reference conv.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  mynet_conv(this->tensor_bottom_, ops_param.conv_param.get(), ops->tensors(),
      this->MakeReferenceTop(this->tensor_top_));
  top_data = this->tensor_top_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  mynet_conv(this->tensor_bottom_2_, ops_param.conv_param.get(), ops->tensors(),
      this->MakeReferenceTop(this->tensor_top_2_));
  top_data = this->tensor_top_2_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpsTest, TestDilatedConv) {
  typedef typename TypeParam::Dtype Dtype;
  std::vector<uint32_t> bottom_shape = {2ul, 3ul, 8ul, 7ul};
  this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
  this->tensor_top_vec_.push_back(this->tensor_top_2_);
  for (uint32_t i = 0; i < this->tensor_bottom_vec_.size(); ++i) {
    this->tensor_bottom_vec_[i]->Reshape(bottom_shape);
  }
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->dilation.push_back(2ul);
  ops_param.conv_param->num_output = 4ul;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->bias_filler->type = "constant";
  ops_param.conv_param->bias_filler->value = 0.1f;
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // Check against reference conv.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  mynet_conv(this->tensor_bottom_, ops_param.conv_param.get(), ops->tensors(),
             this->MakeReferenceTop(this->tensor_top_));
  top_data = this->tensor_top_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  mynet_conv(this->tensor_bottom_2_, ops_param.conv_param.get(), ops->tensors(),
             this->MakeReferenceTop(this->tensor_top_2_));
  top_data = this->tensor_top_2_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpsTest, Test0DConv) {
  typedef typename TypeParam::Dtype Dtype;
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();

  const uint32_t kNumOutput = 3ul;
  ops_param.conv_param->num_output = kNumOutput;
  ops_param.conv_param->axis = 3;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->bias_filler->type = "uniform";
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  std::vector<uint32_t> top_shape = this->tensor_bottom_->shape();
  top_shape[3] = kNumOutput;
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  EXPECT_EQ(top_shape, this->tensor_top_->shape());
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // Check against reference conv.
  std::vector<uint32_t> weight_offset(2);
  const Tensor<Dtype>* weight = ops->tensors()[0].get();
  const Tensor<Dtype>* bias = ops->tensors()[1].get();
  const uint32_t num = this->tensor_top_->count(3);
  const uint32_t dim = this->tensor_top_->shape(3);
  const uint32_t bottom_dim = this->tensor_bottom_->shape(3);
  for (uint32_t n = 0; n < num; ++n) {
    for (uint32_t d = 0; d < dim; ++d) {
      weight_offset[0] = d;
      Dtype value = bias->cpu_data()[d];
      for (uint32_t bottom_d = 0; bottom_d < bottom_dim; ++bottom_d) {
        weight_offset[1] = bottom_d;
        value += weight->data_at(weight_offset) *
                 this->tensor_bottom_->cpu_data()[n * bottom_dim + bottom_d];
      }
      EXPECT_NEAR(value, this->tensor_top_->cpu_data()[n * dim + d], 1e-4);
    }
  }
}

TYPED_TEST(ConvOpsTest, TestSimple3DConv) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
  this->tensor_top_vec_.push_back(this->tensor_top_2_);
  std::vector<uint32_t> bottom_shape(5);
  bottom_shape[0] = this->tensor_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->tensor_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 5ul;
  bottom_shape[3] = this->tensor_bottom_vec_[0]->shape(2);
  bottom_shape[4] = this->tensor_bottom_vec_[0]->shape(3);
  FillerParameterT filler_param;
  UniformFiller<Dtype> filler(&filler_param);
  for (uint32_t i = 0; i < this->tensor_bottom_vec_.size(); ++i) {
    this->tensor_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->tensor_bottom_vec_[i]);
  }
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->stride.push_back(2ul);
  ops_param.conv_param->num_output = 4ul;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->bias_filler->type = "uniform";
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // Check against reference conv.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  mynet_conv(this->tensor_bottom_, ops_param.conv_param.get(), ops->tensors(),
      this->MakeReferenceTop(this->tensor_top_));
  top_data = this->tensor_top_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  mynet_conv(this->tensor_bottom_2_, ops_param.conv_param.get(), ops->tensors(),
      this->MakeReferenceTop(this->tensor_top_2_));
  top_data = this->tensor_top_2_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpsTest, TestDilated3DConv) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
  this->tensor_top_vec_.push_back(this->tensor_top_2_);
  std::vector<uint32_t> bottom_shape(5);
  bottom_shape[0] = this->tensor_bottom_vec_[0]->shape(0);
  bottom_shape[1] = this->tensor_bottom_vec_[0]->shape(1);
  bottom_shape[2] = 6ul;
  bottom_shape[3] = 7ul;
  bottom_shape[4] = 8ul;
  FillerParameterT filler_param;
  UniformFiller<Dtype> filler(&filler_param);
  for (uint32_t i = 0; i < this->tensor_bottom_vec_.size(); ++i) {
    this->tensor_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->tensor_bottom_vec_[i]);
  }
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->stride.push_back(2ul);
  ops_param.conv_param->num_output = 4ul;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->bias_filler->type = "uniform";
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // Check against reference conv.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  mynet_conv(this->tensor_bottom_, ops_param.conv_param.get(), ops->tensors(),
             this->MakeReferenceTop(this->tensor_top_));
  top_data = this->tensor_top_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  mynet_conv(this->tensor_bottom_2_, ops_param.conv_param.get(), ops->tensors(),
             this->MakeReferenceTop(this->tensor_top_2_));
  top_data = this->tensor_top_2_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpsTest, Test1x1Conv) {
  typedef typename TypeParam::Dtype Dtype;
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->kernel_size.push_back(1ul);
  ops_param.conv_param->stride.push_back(1ul);
  ops_param.conv_param->num_output = 4ul;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->bias_filler->type = "constant";
  ops_param.conv_param->bias_filler->value = 0.1f;
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // Check against reference conv.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  mynet_conv(this->tensor_bottom_, ops_param.conv_param.get(), ops->tensors(),
      this->MakeReferenceTop(this->tensor_top_));
  top_data = this->tensor_top_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpsTest, TestSimpleConvGroup) {
  typedef typename TypeParam::Dtype Dtype;
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->stride.push_back(2ul);
  ops_param.conv_param->num_output = 3ul;
  ops_param.conv_param->group = 3ul;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->bias_filler->type = "constant";
  ops_param.conv_param->bias_filler->value = 0.1f;
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // Check against reference conv.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  mynet_conv(this->tensor_bottom_, ops_param.conv_param.get(), ops->tensors(),
      this->MakeReferenceTop(this->tensor_top_));
  top_data = this->tensor_top_->cpu_data();
  ref_top_data = this->ref_tensor_top_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpsTest, TestSobelConv) {
  // Test separable conv by computing the Sobel operator
  // as a single filter then comparing the result
  // as the conv of two rectangular filters.
  typedef typename TypeParam::Dtype Dtype;
  // Fill bottoms with identical Gaussian noise.
  FillerParameterT filler_param;
  filler_param.value = 1.0f;
  auto filler = std::make_shared<UniformFiller<Dtype>>(&filler_param);
  filler->Fill(this->tensor_bottom_);
  this->tensor_bottom_2_->CopyFrom(*this->tensor_bottom_);
  // Compute Sobel G_x operator as 3 x 3 conv.
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->weight_filler->type = "constant";
  ops_param.conv_param->bias_filler->type = "constant";
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->stride.push_back(2ul);
  ops_param.conv_param->num_output = 1ul;
  ops_param.conv_param->bias_term = false;
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->tensors().resize(1);
  ops->tensors()[0].reset(new Tensor<Dtype>(1ul, 3ul, 3ul, 3ul));
  Dtype* weights = ops->tensors()[0]->mutable_cpu_data();
  for (uint32_t c = 0; c < 3; ++c) {
    uint32_t i = c * 9;  // 3 x 3 filter
    weights[i +  0] = -1;
    weights[i +  1] =  0;
    weights[i +  2] =  1;
    weights[i +  3] = -2;
    weights[i +  4] =  0;
    weights[i +  5] =  2;
    weights[i +  6] = -1;
    weights[i +  7] =  0;
    weights[i +  8] =  1;
  }
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convs.
  // (1) the [1 2 1] column filter
  std::vector<Tensor<Dtype>*> sep_tensor_bottom_vec;
  std::vector<Tensor<Dtype>*> sep_tensor_top_vec;
  auto tensor_sep = std::make_shared<Tensor<Dtype>>();
  sep_tensor_bottom_vec.push_back(this->tensor_bottom_2_);
  sep_tensor_top_vec.push_back(this->tensor_top_2_);
  ops_param.conv_param->kernel_size.clear();
  ops_param.conv_param->stride.clear();
  ops_param.conv_param->kernel_h = 3ul;
  ops_param.conv_param->kernel_w = 1ul;
  ops_param.conv_param->stride_h = 2ul;
  ops_param.conv_param->stride_w = 1ul;
  ops_param.conv_param->num_output = 1ul;
  ops_param.conv_param->bias_term = false;
  ops.reset(new ConvOps<Dtype>(&ops_param));
  ops->tensors().resize(1);
  ops->tensors()[0].reset(new Tensor<Dtype>(1ul, 3ul, 3ul, 1ul));
  Dtype* weights_1 = ops->tensors()[0]->mutable_cpu_data();
  for (uint32_t c = 0; c < 3; ++c) {
    uint32_t i = c * 3;  // 3 x 1 filter
    weights_1[i +  0] = 1;
    weights_1[i +  1] = 2;
    weights_1[i +  2] = 1;
  }
  ops->SetUp(sep_tensor_bottom_vec, sep_tensor_top_vec);
  ops->Forward(sep_tensor_bottom_vec, sep_tensor_top_vec);
  // (2) the [-1 0 1] row filter
  tensor_sep->CopyFrom(*this->tensor_top_2_, false, true);
  sep_tensor_bottom_vec.clear();
  sep_tensor_bottom_vec.push_back(tensor_sep.get());
  ops_param.conv_param->kernel_h = 1ul;
  ops_param.conv_param->kernel_w = 3ul;
  ops_param.conv_param->stride_h = 1ul;
  ops_param.conv_param->stride_w = 2ul;
  ops_param.conv_param->num_output = 1ul;
  ops_param.conv_param->bias_term = false;
  ops.reset(new ConvOps<Dtype>(&ops_param));
  ops->tensors().resize(1);
  ops->tensors()[0].reset(new Tensor<Dtype>(1ul, 1ul, 1ul, 3ul));
  Dtype* weights_2 = ops->tensors()[0]->mutable_cpu_data();
  weights_2[0] = -1;
  weights_2[1] =  0;
  weights_2[2] =  1;
  ops->SetUp(sep_tensor_bottom_vec, sep_tensor_top_vec);
  ops->Forward(sep_tensor_bottom_vec, sep_tensor_top_vec);
  // Test equivalence of full and separable filters.
  const Dtype* top_data = this->tensor_top_->cpu_data();
  const Dtype* sep_top_data = this->tensor_top_2_->cpu_data();
  for (uint32_t i = 0; i < this->tensor_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
  }
}

TYPED_TEST(ConvOpsTest, TestNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const uint32_t kernel_h = 11ul;
  const uint32_t kernel_w = 13ul;
  std::vector<uint32_t> bottom_shape(4);
  bottom_shape[0] = 15ul;
  bottom_shape[1] = 18ul;
  bottom_shape[2] = kernel_h * 2ul;
  bottom_shape[3] = kernel_w * 2ul;
  FillerParameterT filler_param;
  UniformFiller<Dtype> filler(&filler_param);
  for (uint32_t i = 0; i < this->tensor_bottom_vec_.size(); ++i) {
    this->tensor_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->tensor_bottom_vec_[i]);
  }
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->num_output = 12ul;
  ops_param.conv_param->bias_term = false;
  ops_param.conv_param->group = 6ul;
  ops_param.conv_param->kernel_h = kernel_h;
  ops_param.conv_param->kernel_w = kernel_w;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->bias_filler->type = "constant";
  Tensor<Dtype> weights;
  Tensor<Dtype> top_diff;
  // Shape and fill weights and top_diff.
  bool copy_diff;
  bool reshape;
  {
    ConvOps<Dtype> ops(&ops_param);
    ops.SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
    top_diff.ReshapeLike(*this->tensor_top_);
    filler.Fill(&top_diff);
    ASSERT_EQ(1, ops.tensors().size());
    copy_diff = false; reshape = true;
    weights.CopyFrom(*ops.tensors()[0], copy_diff, reshape);
  }
  std::vector<bool> propagate_down(1, true);
  Tensor<Dtype> result_2d;
  Tensor<Dtype> backward_result_2d;
  Tensor<Dtype> backward_weight_result_2d;
  // Test with 2D im2col
  {
    mynet_set(this->tensor_top_->mutable_cpu_data(), Dtype(0),
              this->tensor_top_->count());
    mynet_set(this->tensor_bottom_->mutable_cpu_diff(), Dtype(0),
              this->tensor_bottom_->count());
    mynet_set(weights.mutable_cpu_diff(), Dtype(0), weights.count());
    // Do SetUp and Forward; save Forward result in result_2d.
    ops_param.conv_param->force_nd_im2col = false;
    ConvOps<Dtype> conv_2d(&ops_param);
    conv_2d.SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
    ASSERT_EQ(1ul, conv_2d.tensors().size());
    copy_diff = false; reshape = false;
    conv_2d.tensors()[0]->CopyFrom(weights, copy_diff, reshape);
    conv_2d.Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
    copy_diff = false; reshape = true;
    result_2d.CopyFrom(*this->tensor_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->tensor_top_->shape(), top_diff.shape());
    mynet_copy(this->tensor_top_->mutable_cpu_diff(), top_diff.cpu_data(),
               top_diff.count());
    conv_2d.Backward(this->tensor_top_vec_, propagate_down,
                      this->tensor_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_2d.CopyFrom(*this->tensor_bottom_, copy_diff, reshape);
    backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
  }
  Tensor<Dtype> result_nd;
  Tensor<Dtype> backward_result_nd;
  Tensor<Dtype> backward_weight_result_nd;
  // Test with ND im2col
  {
    mynet_set(this->tensor_top_->mutable_cpu_data(), Dtype(0),
              this->tensor_top_->count());
    mynet_set(this->tensor_bottom_->mutable_cpu_diff(), Dtype(0),
              this->tensor_bottom_->count());
    mynet_set(weights.mutable_cpu_diff(), Dtype(0), weights.count());
    // Do SetUp and Forward; save Forward result in result_nd.
    ops_param.conv_param->force_nd_im2col = true;
    ConvOps<Dtype> conv_nd(&ops_param);
    conv_nd.SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
    ASSERT_EQ(1ul, conv_nd.tensors().size());
    copy_diff = false; reshape = false;
    conv_nd.tensors()[0]->CopyFrom(weights, copy_diff, reshape);
    conv_nd.Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
    copy_diff = false; reshape = true;
    result_nd.CopyFrom(*this->tensor_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_nd.
    ASSERT_EQ(this->tensor_top_->shape(), top_diff.shape());
    mynet_copy(this->tensor_top_->mutable_cpu_diff(), top_diff.cpu_data(),
               top_diff.count());
    conv_nd.Backward(this->tensor_top_vec_, propagate_down,
                      this->tensor_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_nd.CopyFrom(*this->tensor_bottom_, copy_diff, reshape);
    backward_weight_result_nd.CopyFrom(weights, copy_diff, reshape);
  }
  ASSERT_EQ(result_nd.count(), result_2d.count());
  for (uint32_t i = 0; i < result_2d.count(); ++i)  {
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

// TYPED_TEST(ConvOpsTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpsParameterT ops_param;
//   ops_param.conv_param = std::make_unique<ConvParameterT>();
//   ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
//   this->tensor_top_vec_.push_back(this->tensor_top_2_);
//   ops_param.conv_param->kernel_size.push_back(3ul);
//   ops_param.conv_param->stride.push_back(2ul);
//   ops_param.conv_param->num_output = 2ul;
//   ops_param.conv_param->weight_filler->type = "uniform";
//   ops_param.conv_param->bias_filler->type = "uniform";
//   ConvOps<Dtype> ops(&ops_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&ops, this->tensor_bottom_vec_,
//       this->tensor_top_vec_);
// }

// TYPED_TEST(ConvOpsTest, TestDilatedGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpsParameterT ops_param;
//   ops_param.conv_param = std::make_unique<ConvParameterT>();
//   ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   std::vector<uint32_t> bottom_shape;
//   bottom_shape.push_back(2ul);
//   bottom_shape.push_back(3ul);
//   bottom_shape.push_back(5ul);
//   bottom_shape.push_back(6ul);
//   for (uint32_t i = 0; i < this->tensor_bottom_vec_.size(); ++i) {
//     this->tensor_bottom_vec_[i]->Reshape(bottom_shape);
//   }
//   ops_param.conv_param->kernel_size.push_back(3ul);
//   ops_param.conv_param->dilation.push_back(2ul);
//   ops_param.conv_param->num_output = 2ul;
//   ops_param.conv_param->weight_filler->type = "uniform";
//   ops_param.conv_param->bias_filler->type = "uniform";
//   ConvOps<Dtype> ops(&ops_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&ops, this->tensor_bottom_vec_,
//                                   this->tensor_top_vec_);
// }

// TYPED_TEST(ConvOpsTest, TestGradient3D) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpsParameterT ops_param;
//   ops_param.conv_param = std::make_unique<ConvParameterT>();
//   ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   std::vector<uint32_t> bottom_shape(5);
//   bottom_shape[0] = this->tensor_bottom_vec_[0]->shape(0);
//   bottom_shape[1] = this->tensor_bottom_vec_[0]->shape(1);
//   bottom_shape[2] = 5ul;
//   bottom_shape[3] = this->tensor_bottom_vec_[0]->shape(2);
//   bottom_shape[4] = this->tensor_bottom_vec_[0]->shape(3);
//   FillerParameterT filler_param;
//   UniformFiller<Dtype> filler(&filler_param);
//   for (uint32_t i = 0; i < this->tensor_bottom_vec_.size(); ++i) {
//     this->tensor_bottom_vec_[i]->Reshape(bottom_shape);
//     filler.Fill(this->tensor_bottom_vec_[i]);
//   }
//   ops_param.conv_param->kernel_size.push_back(3ul);
//   ops_param.conv_param->stride.push_back(2ul);
//   ops_param.conv_param->num_output = 2ul;
//   ops_param.conv_param->weight_filler->type = "uniform";
//   ops_param.conv_param->bias_filler->type = "uniform";
//   ConvOps<Dtype> ops(&ops_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&ops, this->tensor_bottom_vec_,
//       this->tensor_top_vec_);
// }

// TYPED_TEST(ConvOpsTest, Test1x1Gradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpsParameterT ops_param;
//   ops_param.conv_param = std::make_unique<ConvParameterT>();
//   ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
//   this->tensor_top_vec_.push_back(this->tensor_top_2_);
//   ops_param.conv_param->kernel_size.push_back(1ul);
//   ops_param.conv_param->stride.push_back(1ul);
//   ops_param.conv_param->num_output = 2ul;
//   ops_param.conv_param->weight_filler->type = "uniform";
//   ops_param.conv_param->bias_filler->type = "uniform";
//   ConvOps<Dtype> ops(&ops_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&ops, this->tensor_bottom_vec_,
//       this->tensor_top_vec_);
// }

// TYPED_TEST(ConvOpsTest, TestGradientGroup) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpsParameterT ops_param;
//   ops_param.conv_param = std::make_unique<ConvParameterT>();
//   ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   ops_param.conv_param->kernel_size.push_back(3ul);
//   ops_param.conv_param->stride.push_back(2ul);
//   ops_param.conv_param->num_output = 3ul;
//   ops_param.conv_param->group = 3ul;
//   ops_param.conv_param->weight_filler->type = "uniform";
//   ops_param.conv_param->bias_filler->type = "uniform";
//   ConvOps<Dtype> ops(&ops_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&ops, this->tensor_bottom_vec_,
//       this->tensor_top_vec_);
// }

TYPED_TEST(ConvOpsTest, TestDeConvSetup) {
  typedef typename TypeParam::Dtype Dtype;
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->weight_filler->type = "constant";
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler->type = "constant";
  ops_param.conv_param->transpose = true;
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->stride.push_back(2ul);
  ops_param.conv_param->num_output = 4ul;
  this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
  this->tensor_top_vec_.push_back(this->tensor_top_2_);
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  EXPECT_EQ(this->tensor_top_->num(), 2ul);
  EXPECT_EQ(this->tensor_top_->channels(), 4ul);
  EXPECT_EQ(this->tensor_top_->height(), 13ul);
  EXPECT_EQ(this->tensor_top_->width(), 9ul);
  EXPECT_EQ(this->tensor_top_2_->num(), 2ul);
  EXPECT_EQ(this->tensor_top_2_->channels(), 4ul);
  EXPECT_EQ(this->tensor_top_2_->height(), 13ul);
  EXPECT_EQ(this->tensor_top_2_->width(), 9ul);
  // setting group should not change the shape
  ops_param.conv_param->num_output = 3ul;
  ops_param.conv_param->group = 3ul;
  ops.reset(new ConvOps<Dtype>(&ops_param));
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  EXPECT_EQ(this->tensor_top_->num(), 2ul);
  EXPECT_EQ(this->tensor_top_->channels(), 3ul);
  EXPECT_EQ(this->tensor_top_->height(), 13ul);
  EXPECT_EQ(this->tensor_top_->width(), 9ul);
  EXPECT_EQ(this->tensor_top_2_->num(), 2ul);
  EXPECT_EQ(this->tensor_top_2_->channels(), 3ul);
  EXPECT_EQ(this->tensor_top_2_->height(), 13ul);
  EXPECT_EQ(this->tensor_top_2_->width(), 9ul);
}

TYPED_TEST(ConvOpsTest, TestSimpleDeconv) {
  typedef typename TypeParam::Dtype Dtype;
  this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
  this->tensor_top_vec_.push_back(this->tensor_top_2_);
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->kernel_size.push_back(3ul);
  ops_param.conv_param->stride.push_back(2ul);
  ops_param.conv_param->num_output = 4ul;
  ops_param.conv_param->weight_filler->type = "constant";
  ops_param.conv_param->weight_filler->value = 1.0f;
  ops_param.conv_param->bias_filler->type = "constant";
  ops_param.conv_param->bias_filler->value = 0.1f;
  ops_param.conv_param->transpose = true;
  std::shared_ptr<Ops<Dtype>> ops = std::make_shared<ConvOps<Dtype>>(&ops_param);
  ops->SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // constant-fill the bottom tensors
  FillerParameterT filler_param;
  filler_param.value = 1.0f;
  ConstantFiller<Dtype> filler(&filler_param);
  filler.Fill(this->tensor_bottom_);
  filler.Fill(this->tensor_bottom_2_);
  ops->Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
  // simply check that accumulation works with overlapping filters
  const Dtype* top_data = this->tensor_top_->cpu_data();
  for (uint32_t n = 0; n < this->tensor_top_->num(); ++n) {
    for (uint32_t c = 0; c < this->tensor_top_->channels(); ++c) {
      for (uint32_t h = 0; h < this->tensor_top_->height(); ++h) {
        for (uint32_t w = 0; w < this->tensor_top_->width(); ++w) {
          Dtype expected = 3.1;
          bool h_overlap = h % 2 == 0 && h > 0
            && h < this->tensor_top_->height() - 1;
          bool w_overlap = w % 2 == 0 && w > 0
            && w < this->tensor_top_->width() - 1;
          if (h_overlap && w_overlap) {
            expected += 9;
          } else if (h_overlap || w_overlap) {
            expected += 3;
          }
          EXPECT_NEAR(top_data[this->tensor_top_->offset(n, c, h, w)],
              expected, 1e-4);
        }
      }
    }
  }
}

// TYPED_TEST(ConvOpsTest, TestDeconvGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   OpsParameterT ops_param;
//   ops_param.conv_param = std::make_unique<ConvParameterT>();
//   ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   this->tensor_bottom_vec_.push_back(this->tensor_bottom_2_);
//   this->tensor_top_vec_.push_back(this->tensor_top_2_);
//   ops_param.conv_param->kernel_size.push_back(2ul);
//   ops_param.conv_param->stride.push_back(1ul);
//   ops_param.conv_param->num_output = 1ul;
//   ops_param.conv_param->weight_filler->type = "uniform";
//   ops_param.conv_param->bias_filler->type = "uniform";
//   ops_param.conv_param->transpose = true;
//   ConvOps<Dtype> ops(&ops_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&ops, this->tensor_bottom_vec_,
//       this->tensor_top_vec_);
// }

TYPED_TEST(ConvOpsTest, TestDeconvNDAgainst2D) {
  typedef typename TypeParam::Dtype Dtype;
  const uint32_t kernel_h = 11ul;
  const uint32_t kernel_w = 13ul;
  std::vector<uint32_t> bottom_shape(4);
  bottom_shape[0] = 15ul;
  bottom_shape[1] = 12ul;
  bottom_shape[2] = kernel_h * 2;
  bottom_shape[3] = kernel_w * 2;
  FillerParameterT filler_param;
  UniformFiller<Dtype> filler(&filler_param);
  for (uint32_t i = 0; i < this->tensor_bottom_vec_.size(); ++i) {
    this->tensor_bottom_vec_[i]->Reshape(bottom_shape);
    filler.Fill(this->tensor_bottom_vec_[i]);
  }
  OpsParameterT ops_param;
  ops_param.conv_param = std::make_unique<ConvParameterT>();
  ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->bias_filler = std::make_unique<FillerParameterT>();
  ops_param.conv_param->num_output = 18ul;
  ops_param.conv_param->bias_term = false;
  ops_param.conv_param->group = 6ul;
  ops_param.conv_param->kernel_h = kernel_h;
  ops_param.conv_param->kernel_w = kernel_w;
  ops_param.conv_param->weight_filler->type = "uniform";
  ops_param.conv_param->transpose = true;
  Tensor<Dtype> weights;
  Tensor<Dtype> top_diff;
  // Shape and fill weights and top_diff.
  bool copy_diff;
  bool reshape;
  {
    ConvOps<Dtype> ops(&ops_param);
    ops.SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
    top_diff.ReshapeLike(*this->tensor_top_);
    filler.Fill(&top_diff);
    ASSERT_EQ(1ul, ops.tensors().size());
    copy_diff = false; reshape = true;
    weights.CopyFrom(*ops.tensors()[0], copy_diff, reshape);
  }
  std::vector<bool> propagate_down(1, true);
  Tensor<Dtype> result_2d;
  Tensor<Dtype> backward_result_2d;
  Tensor<Dtype> backward_weight_result_2d;
  // Test with 2D im2col
  {
    mynet_set(this->tensor_top_->mutable_cpu_data(), Dtype(0),
              this->tensor_top_->count());
    mynet_set(this->tensor_bottom_->mutable_cpu_diff(), Dtype(0),
              this->tensor_bottom_->count());
    mynet_set(weights.mutable_cpu_diff(), Dtype(0), weights.count());
    // Do SetUp and Forward; save Forward result in result_2d.
    ops_param.conv_param->force_nd_im2col = false;
    ConvOps<Dtype> ops_2d(&ops_param);
    ops_2d.SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
    ASSERT_EQ(1ul, ops_2d.tensors().size());
    copy_diff = false; reshape = false;
    ops_2d.tensors()[0]->CopyFrom(weights, copy_diff, reshape);
    ops_2d.Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
    copy_diff = false; reshape = true;
    result_2d.CopyFrom(*this->tensor_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_2d.
    ASSERT_EQ(this->tensor_top_->shape(), top_diff.shape());
    mynet_copy(this->tensor_top_->mutable_cpu_diff(), top_diff.cpu_data(),
               top_diff.count());
    ops_2d.Backward(this->tensor_top_vec_, propagate_down,
                      this->tensor_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_2d.CopyFrom(*this->tensor_bottom_, copy_diff, reshape);
    backward_weight_result_2d.CopyFrom(weights, copy_diff, reshape);
  }
  Tensor<Dtype> result_nd;
  Tensor<Dtype> backward_result_nd;
  Tensor<Dtype> backward_weight_result_nd;
  // Test with ND im2col
  {
    mynet_set(this->tensor_top_->mutable_cpu_data(), Dtype(0),
              this->tensor_top_->count());
    mynet_set(this->tensor_bottom_->mutable_cpu_diff(), Dtype(0),
              this->tensor_bottom_->count());
    mynet_set(weights.mutable_cpu_diff(), Dtype(0), weights.count());
    // Do SetUp and Forward; save Forward result in result_nd.
    ops_param.conv_param->force_nd_im2col = true;
    ConvOps<Dtype> ops_nd(&ops_param);
    ops_nd.SetUp(this->tensor_bottom_vec_, this->tensor_top_vec_);
    ASSERT_EQ(1, ops_nd.tensors().size());
    copy_diff = false; reshape = false;
    ops_nd.tensors()[0]->CopyFrom(weights, copy_diff, reshape);
    ops_nd.Forward(this->tensor_bottom_vec_, this->tensor_top_vec_);
    copy_diff = false; reshape = true;
    result_nd.CopyFrom(*this->tensor_top_, copy_diff, reshape);
    // Copy pre-generated top diff into actual top diff;
    // do Backward and save result in backward_result_nd.
    ASSERT_EQ(this->tensor_top_->shape(), top_diff.shape());
    mynet_copy(this->tensor_top_->mutable_cpu_diff(), top_diff.cpu_data(),
               top_diff.count());
    ops_nd.Backward(this->tensor_top_vec_, propagate_down,
                      this->tensor_bottom_vec_);
    copy_diff = true; reshape = true;
    backward_result_nd.CopyFrom(*this->tensor_bottom_, copy_diff, reshape);
    backward_weight_result_nd.CopyFrom(weights, copy_diff, reshape);
  }
  ASSERT_EQ(result_nd.count(), result_2d.count());
  for (uint32_t i = 0; i < result_2d.count(); ++i)  {
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

// TYPED_TEST(ConvOpsTest, TestDeconvGradient3D) {
//   typedef typename TypeParam::Dtype Dtype;
//   std::vector<uint32_t> bottom_shape(5);
//   bottom_shape[0] = this->tensor_bottom_vec_[0]->shape(0);
//   bottom_shape[1] = this->tensor_bottom_vec_[0]->shape(1);
//   bottom_shape[2] = 2ul;
//   bottom_shape[3] = 3ul;
//   bottom_shape[4] = 2ul;
//   FillerParameterT filler_param;
//   UniformFiller<Dtype> filler(&filler_param);
//   for (uint32_t i = 0; i < this->tensor_bottom_vec_.size(); ++i) {
//     this->tensor_bottom_vec_[i]->Reshape(bottom_shape);
//     filler.Fill(this->tensor_bottom_vec_[i]);
//   }
//   OpsParameterT ops_param;
//   ops_param.conv_param = std::make_unique<ConvParameterT>();
//   ops_param.conv_param->weight_filler = std::make_unique<FillerParameterT>();
//   ops_param.conv_param->kernel_size.push_back(2ul);
//   ops_param.conv_param->stride.push_back(2ul);
//   ops_param.conv_param->pad.push_back(1ul);
//   ops_param.conv_param->num_output = 2ul;
//   ops_param.conv_param->weight_filler->type = "uniform";
//   ops_param.conv_param->bias_filler->type = "uniform";
//   ops_param.conv_param->transpose = true;
//   ConvOps<Dtype> ops(&ops_param);
//   GradientChecker<Dtype> checker(1e-2, 1e-3);
//   checker.CheckGradientExhaustive(&ops, this->tensor_bottom_vec_,
//       this->tensor_top_vec_);
// }

}  // namespace mynet
