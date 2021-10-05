#include "tensor.hpp"
#include "common.hpp"
#include "mynet_test_main.hpp"
#include "filler.hpp"

namespace mynet {

template <typename Dtype>
class TensorSimpleTest : public ::testing::Test {
 protected:
  TensorSimpleTest()
      : tensor_(new Tensor<Dtype>()),
        tensor_preshaped_(new Tensor<Dtype>(2, 3, 4, 5)) {}
  virtual ~TensorSimpleTest() { delete tensor_; delete tensor_preshaped_; }
  Tensor<Dtype>* const tensor_;
  Tensor<Dtype>* const tensor_preshaped_;
};

TYPED_TEST_CASE(TensorSimpleTest, TestDtypes);

TYPED_TEST(TensorSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->tensor_);
  EXPECT_TRUE(this->tensor_preshaped_);
  EXPECT_EQ(this->tensor_preshaped_->num(), 2);
  EXPECT_EQ(this->tensor_preshaped_->channels(), 3);
  EXPECT_EQ(this->tensor_preshaped_->height(), 4);
  EXPECT_EQ(this->tensor_preshaped_->width(), 5);
  EXPECT_EQ(this->tensor_preshaped_->count(), 120);
  EXPECT_EQ(this->tensor_->num_axes(), 0);
  EXPECT_EQ(this->tensor_->count(), 0);
}

TYPED_TEST(TensorSimpleTest, TestCopyFrom) {
  this->tensor_->CopyFrom(*(this->tensor_preshaped_), true, true);
  EXPECT_EQ(this->tensor_->num(), 2);
  EXPECT_EQ(this->tensor_->channels(), 3);
  EXPECT_EQ(this->tensor_->height(), 4);
  EXPECT_EQ(this->tensor_->width(), 5);
  EXPECT_EQ(this->tensor_->count(), 120);
}

TYPED_TEST(TensorSimpleTest, TestReshape) {
  this->tensor_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->tensor_->num(), 2);
  EXPECT_EQ(this->tensor_->channels(), 3);
  EXPECT_EQ(this->tensor_->height(), 4);
  EXPECT_EQ(this->tensor_->width(), 5);
  EXPECT_EQ(this->tensor_->count(), 120);

  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  std::vector<uint32_t> shape = {5, 4, 3, 1};
  auto ts = CreateTensorShapeDirect(flatbuffer_builder, &shape);
  flatbuffer_builder.Finish(ts);
  
  auto tensor_shape = flatbuffers::GetRoot<TensorShape>(flatbuffer_builder.GetBufferPointer());
  this->tensor_->Reshape(tensor_shape->UnPack());
  EXPECT_EQ(this->tensor_->num(), 5);
  EXPECT_EQ(this->tensor_->channels(), 4);
  EXPECT_EQ(this->tensor_->height(), 3);
  EXPECT_EQ(this->tensor_->width(), 1);
  EXPECT_EQ(this->tensor_->count(), 60);
}

TYPED_TEST(TensorSimpleTest, TestReshapeZero) {
  std::vector<uint32_t> shape(2);
  shape[0] = 0;
  shape[1] = 5;
  this->tensor_->Reshape(shape);
  EXPECT_EQ(this->tensor_->count(), 0);
}

TYPED_TEST(TensorSimpleTest, TestToFlat) {
  std::vector<uint32_t> shape(2);
  shape[0] = 3;
  shape[1] = 2;
  this->tensor_->Reshape(shape);
  
  auto tensor_flat = flatbuffers::GetRoot<TensorFlat>(this->tensor_->ToFlat(true).data())->UnPack();
  EXPECT_EQ(tensor_flat->num, 3);
  EXPECT_EQ(tensor_flat->channels, 2);
}

TYPED_TEST(TensorSimpleTest, TestLegacytensorFlatShapeEquals) {
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  flatbuffer_builder.ForceDefaults(true);
  auto ff = CreateTensorFlat(flatbuffer_builder);
  flatbuffer_builder.Finish(ff);
  
  auto tensor_flat = flatbuffers::GetMutableRoot<TensorFlat>(flatbuffer_builder.GetBufferPointer())->UnPack();
  // Reshape to (3 x 2).
  std::vector<uint32_t> shape(2);
  shape[0] = 3;
  shape[1] = 2;
  this->tensor_->Reshape(shape);

  // (3 x 2) tensor == (1 x 1 x 3 x 2) legacy tensor
  tensor_flat->num = 1;
  tensor_flat->channels = 1;
  tensor_flat->height = 3;
  tensor_flat->width = 2;
  EXPECT_TRUE(this->tensor_->ShapeEquals(tensor_flat));

  // (3 x 2) tensor != (0 x 1 x 3 x 2) legacy tensor
  tensor_flat->num = 0;
  tensor_flat->channels = 1;
  tensor_flat->height = 3;
  tensor_flat->width = 2;
  EXPECT_FALSE(this->tensor_->ShapeEquals(tensor_flat));

  // (3 x 2) tensor != (3 x 1 x 3 x 2) legacy tensor
  tensor_flat->num = 3;
  tensor_flat->channels = 1;
  tensor_flat->height = 3;
  tensor_flat->width = 2;
  EXPECT_FALSE(this->tensor_->ShapeEquals(tensor_flat));

  // Reshape to (1 x 3 x 2).
  shape.insert(shape.begin(), 1);
  this->tensor_->Reshape(shape);

  // (1 x 3 x 2) tensor == (1 x 1 x 3 x 2) legacy tensor
  tensor_flat->num = 1;
  tensor_flat->channels = 1;
  tensor_flat->height = 3;
  tensor_flat->width = 2;
  EXPECT_TRUE(this->tensor_->ShapeEquals(tensor_flat));

  // Reshape to (2 x 3 x 2).
  shape[0] = 2;
  this->tensor_->Reshape(shape);

  // (2 x 3 x 2) tensor != (1 x 1 x 3 x 2) legacy tensor
  tensor_flat->num = 1;
  tensor_flat->channels = 1;
  tensor_flat->height = 3;
  tensor_flat->width = 2;
  EXPECT_FALSE(this->tensor_->ShapeEquals(tensor_flat));
}

template <typename TypeParam>
class TensorMathTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
protected:
  TensorMathTest()
      : tensor_(new Tensor<Dtype>(2, 3, 4, 5)),
        epsilon_(1e-6) {}

  virtual ~TensorMathTest() { delete tensor_; }
  Tensor<Dtype>* const tensor_;
  Dtype epsilon_;
};

TYPED_TEST_CASE(TensorMathTest, TestDtypesAndDevices);

TYPED_TEST(TensorMathTest, TestSumOfSquares) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized tensor should have sum of squares == 0.
  EXPECT_EQ(0, this->tensor_->sumsq_data());
  EXPECT_EQ(0, this->tensor_->sumsq_diff());
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  auto fp = CreateFillerParameterDirect(flatbuffer_builder, nullptr, 0, -3, 3);
  flatbuffer_builder.Finish(fp);
  
  auto filler_param = flatbuffers::GetMutableRoot<FillerParameter>(flatbuffer_builder.GetBufferPointer())->UnPack();
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->tensor_);
  Dtype expected_sumsq = 0;
  const Dtype* data = this->tensor_->cpu_data();
  for (size_t i = 0; i < this->tensor_->count(); ++i) {
    expected_sumsq += data[i] * data[i];
  }
  // Do a mutable access on the current device,
  // so that the sumsq computation is done on that device.
  // (Otherwise, this would only check the CPU sumsq implementation.)
  switch (TypeParam::device) {
  case Mynet::CPU:
    this->tensor_->mutable_cpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_sumsq, this->tensor_->sumsq_data(),
              this->epsilon_ * expected_sumsq);
  EXPECT_EQ(0, this->tensor_->sumsq_diff());

  // Check sumsq_diff too.
  const Dtype kDiffScaleFactor = 7;
  mynet_cpu_scale(this->tensor_->count(), kDiffScaleFactor, data,
                  this->tensor_->mutable_cpu_diff());
  switch (TypeParam::device) {
  case Mynet::CPU:
    this->tensor_->mutable_cpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_sumsq, this->tensor_->sumsq_data(),
              this->epsilon_ * expected_sumsq);
  const Dtype expected_sumsq_diff =
      expected_sumsq * kDiffScaleFactor * kDiffScaleFactor;
  EXPECT_NEAR(expected_sumsq_diff, this->tensor_->sumsq_diff(),
              this->epsilon_ * expected_sumsq_diff);
}

TYPED_TEST(TensorMathTest, TestAsum) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized tensor should have asum == 0.
  EXPECT_EQ(0, this->tensor_->asum_data());
  EXPECT_EQ(0, this->tensor_->asum_diff());
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  auto fp = CreateFillerParameterDirect(flatbuffer_builder, nullptr, 0, -3, 3);
  flatbuffer_builder.Finish(fp);
  
  auto filler_param = flatbuffers::GetMutableRoot<FillerParameter>(flatbuffer_builder.GetBufferPointer())->UnPack();
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->tensor_);
  Dtype expected_asum = 0;
  const Dtype* data = this->tensor_->cpu_data();
  for (size_t i = 0; i < this->tensor_->count(); ++i) {
    expected_asum += std::fabs(data[i]);
  }
  // Do a mutable access on the current device,
  // so that the asum computation is done on that device.
  // (Otherwise, this would only check the CPU asum implementation.)
  switch (TypeParam::device) {
  case Mynet::CPU:
    this->tensor_->mutable_cpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_asum, this->tensor_->asum_data(),
              this->epsilon_ * expected_asum);
  EXPECT_EQ(0, this->tensor_->asum_diff());

  // Check asum_diff too.
  const Dtype kDiffScaleFactor = 7;
  mynet_cpu_scale(this->tensor_->count(), kDiffScaleFactor, data,
                  this->tensor_->mutable_cpu_diff());
  switch (TypeParam::device) {
  case Mynet::CPU:
    this->tensor_->mutable_cpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_NEAR(expected_asum, this->tensor_->asum_data(),
              this->epsilon_ * expected_asum);
  const Dtype expected_diff_asum = expected_asum * kDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum, this->tensor_->asum_diff(),
              this->epsilon_ * expected_diff_asum);
}

TYPED_TEST(TensorMathTest, TestScaleData) {
  typedef typename TypeParam::Dtype Dtype;

  EXPECT_EQ(0, this->tensor_->asum_data());
  EXPECT_EQ(0, this->tensor_->asum_diff());
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  auto fp = CreateFillerParameterDirect(flatbuffer_builder, nullptr, 0, -3, 3);
  flatbuffer_builder.Finish(fp);
  
  auto filler_param = flatbuffers::GetMutableRoot<FillerParameter>(flatbuffer_builder.GetBufferPointer())->UnPack();
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->tensor_);
  const Dtype asum_before_scale = this->tensor_->asum_data();
  // Do a mutable access on the current device,
  // so that the asum computation is done on that device.
  // (Otherwise, this would only check the CPU asum implementation.)
  switch (TypeParam::device) {
  case Mynet::CPU:
    this->tensor_->mutable_cpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  const Dtype kDataScaleFactor = 3;
  this->tensor_->scale_data(kDataScaleFactor);
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, this->tensor_->asum_data(),
              this->epsilon_ * asum_before_scale * kDataScaleFactor);
  EXPECT_EQ(0, this->tensor_->asum_diff());

  // Check scale_diff too.
  const Dtype kDataToDiffScaleFactor = 7;
  const Dtype* data = this->tensor_->cpu_data();
  mynet_cpu_scale(this->tensor_->count(), kDataToDiffScaleFactor, data,
                  this->tensor_->mutable_cpu_diff());
  const Dtype expected_asum_before_scale = asum_before_scale * kDataScaleFactor;
  EXPECT_NEAR(expected_asum_before_scale, this->tensor_->asum_data(),
              this->epsilon_ * expected_asum_before_scale);
  const Dtype expected_diff_asum_before_scale =
      asum_before_scale * kDataScaleFactor * kDataToDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum_before_scale, this->tensor_->asum_diff(),
              this->epsilon_ * expected_diff_asum_before_scale);
  switch (TypeParam::device) {
  case Mynet::CPU:
    this->tensor_->mutable_cpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  const Dtype kDiffScaleFactor = 3;
  this->tensor_->scale_diff(kDiffScaleFactor);
  EXPECT_NEAR(asum_before_scale * kDataScaleFactor, this->tensor_->asum_data(),
              this->epsilon_ * asum_before_scale * kDataScaleFactor);
  const Dtype expected_diff_asum =
      expected_diff_asum_before_scale * kDiffScaleFactor;
  EXPECT_NEAR(expected_diff_asum, this->tensor_->asum_diff(),
              this->epsilon_ * expected_diff_asum);
}

}  // namespace mynet
