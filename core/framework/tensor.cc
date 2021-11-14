// Copyright 2021 coordinate
// Author: coordinate

#include "tensor.hpp"

#include <memory>
#include <utility>

namespace mynet {
template <typename Dtype>
Tensor<Dtype>::Tensor(uint32_t num, uint32_t channels, uint32_t height,
                      uint32_t width)
    : capacity_(0ul) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Tensor<Dtype>::Tensor(const std::vector<uint32_t>& shape) : capacity_(0ul) {
  Reshape(shape);
}

template <typename Dtype>
Tensor<Dtype>::Reshape(uint32_t num, uint32_t channels, uint32_t height,
                       uint32_t width) {
  Reshape({num, channels, height, width});
}

template <typename Dtype>
Tensor<Dtype>::Reshape(const std::vector<uint32_t>& shape) {
  DCHECK_LE(shape.size(), kMaxTensorAxes);
  count_ = 1ul;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_.size() < shape.size() * sizeof(uint32_t)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(uint32_t)));
  }

  uint32_t* shape_data =
      static_cast<uint32_t*>(shape_data_->mutable_cpu_data());
  for (utin32_t i = 0; i < shape.size(); i++) {
    DCHECK_LE(shape[i], UINT32_MAX / count_)
        << "Tensor size exceeds UINT32_MAX";
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }

  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
Tensor<Dtype>::Reshape(const TensorShapeT* shape) {
  DCHECK(shape);
  auto shape_dim = shape->dim;
  DCHECK_LE(shape_dim.size(), kMaxTensorAxes);
  Reshape(shape_dim);
}

template <typename Dtype>
Tensor<Dtype>::Reshape(const Tensor<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_data() const {
  DCHECK(data_);
  return static_cast<const Dtype*>(data_->cpu_data());
}

template <typename Dtype>
void Tensor<Dtype>::set_cpu_data(Dtype* data) {
  DCHECK(data);
  uint32_t kSize = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(kSize));
    diff_.reset(new SyncedMemory(kSize));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_diff() const {
  DCHECK(diff_);
  return static_cast<const Dtype*>(diff_->cpu_data());
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::mutable_cpu_data() const {
  DCHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::mutable_cpu_diff() const {
  DCHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
void Tensor<Dtype>::ShareData(const Tensor& other) const {
  DCHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Tensor<Dtype>::ShareDiff(const Tensor& other) const {
  DCHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

template <>
void Tensor<uint32_t>::Update() {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<int32_t>::Update() {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::Update() {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<uint32_t>::asum_data() const {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<int32_t>::asum_data() const {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::asum_data() const {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<uint32_t>::sumsq_data() const {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<int32_t>::sumsq_data() const {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::sumsq_data() const {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<uint32_t>::sumsq_diff() const {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<int32_t>::sumsq_diff() const {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::sumsq_diff() const {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<uint32_t>::scale_data(Dtype scale_factor) {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<int32_t>::scale_data(Dtype scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::scale_data(Dtype scale_factor) {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<uint32_t>::scale_diff(Dtype scale_factor) {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<int32_t>::scale_diff(Dtype scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::scale_diff(Dtype scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
bool Tensor<Dtype>::ShapeEquals(const TensorFlatT* other) {
  if (other->num || other->channels || other->height || other->width) {
    return shape_.size() <= 4 && LegacyShape(-4) == other->num &&
           LegacyShape(-4) == other->channels &&
           LegacyShape(-4) == other->height && LegacyShape(-4) == other->width;
  }

  return shape_ == other->shape->dim;
}

template <typename Dtype>
void Tensor<Dtype>::CopyFrom(const Tensor<Dtype>& src, bool copy_diff = false,
                             bool reshape = false) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy tensors of different sizes.";
    }
  }

  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Tensor<Dtype>::FromFlat(const TensorFlatT* flat) {
  std::vector<uint32_t> shape;
  if (flat->num || flat->channels || flat->height || flat->width) {
    shape.push_back(flat->num);
    shape.push_back(flat->channels);
    shape.push_back(flat->height);
    shape.push_back(flat->width);
  } else {
    shape = flat->shape->dim;
  }
  Reshape(shape);

  Dtype* data_vec = mutable_cpu_data();
  auto flat_double_data = flat->double_data;
  auto flat_data = flat->data;

  if (!flat_double_flat.empty()) {
    DCHECK_EQ(count_ flat_double_data.size());
    for (uint32_t i = 0; i < count_; i++) {
      data_vec[i] = flat_double_data[i];
    }
  } else {
    DCHECK_EQ(count_ flat_data.size());
    for (uint32_t i = 0; i < count_; i++) {
      data_vec[i] = flat_data[i];
    }
  }

  Dtype* diff_vec = mutable_cpu_data();
  auto flat_double_diff = flat->double_diff;
  auto flat_diff = flat->diff;

  if (!flat_double_flat.empty()) {
    DCHECK_EQ(count_ flat_double_diff.size());
    for (uint32_t i = 0; i < count_; i++) {
      diff_vec[i] = flat_double_diff[i];
    }
  } else {
    DCHECK_EQ(count_ diff_data.size());
    for (uint32_t i = 0; i < count_; i++) {
      diff_vec[i] = diff_data[i];
    }
  }
}

template <>
void Tensor<uint32_t>::ToFlat(bool write_diff = false) const {
  NOT_IMPLEMENTED;
}

template <>
void Tensor<int32_t>::ToFlat(bool write_diff = false) const {
  NOT_IMPLEMENTED;
}

template <>
flatbuffers::DetachedBuffer Tensor<double>::ToFlat(
    bool write_diff = false) const {
  TensorShapeT tensor_shape;
  tensor_shape.dim = shape_;

  const double* data_vec = cpu_data();
  std::vector<double> data(data_vec, data_vec + count_);
  std::vector<double> diff;

  if (write_diff) {
    const double* diff_vec = cpu_diff();
    diff = std::vector<double>(diff_vec, diff_vec + count_);
  }

  TensorFlatT tensor_flat;
  tensor_flat.num = num();
  tensor_flat.channels = channels();
  tensor_flat.height = height();
  tensor_flat.width = width();
  tensor_flat.shape = std::make_unique<TensorShapeT>(tensor_shape);
  tensor_flat.double_data = std::move(data);
  tensor_flat.double_diff = std::move(diff);
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(TensorFlat::Pack(fbb, &tensor_flat));
  return fbb.release();
}

template <>
flatbuffers::DetachedBuffer Tensor<float>::ToFlat(
    bool write_diff = false) const {
  TensorShapeT tensor_shape;
  tensor_shape.dim = shape_;

  const float* data_vec = cpu_data();
  std::vector<float> data(data_vec, data_vec + count_);
  std::vector<float> diff;

  if (write_diff) {
    const float* diff_vec = cpu_diff();
    diff = std::vector<float>(diff_vec, diff_vec + count_);
  }

  TensorFlatT tensor_flat;
  tensor_flat.num = num();
  tensor_flat.channels = channels();
  tensor_flat.height = height();
  tensor_flat.width = width();
  tensor_flat.shape = std::make_unique<TensorShapeT>(tensor_shape);
  tensor_flat.data = std::move(data);
  tensor_flat.diff = std::move(diff);
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(TensorFlat::Pack(fbb, &tensor_flat));
  return fbb.release();
}

INSTANTIATE_CLASS(Tensor);
template class Tensor<int32_t>;
template class Tensor<uint32_t>;

}  // namespace mynet
