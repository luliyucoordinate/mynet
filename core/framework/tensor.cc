// Copyright 2021 coordinate
// Author: coordinate

#include "tensor.hpp"

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

}  // namespace mynet
