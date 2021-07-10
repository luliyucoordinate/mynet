#include <climits>

#include "tensor.hpp"
#include "common.hpp"
#include "syncedmem.hpp"
#include "math_functions.hpp"

namespace mynet {

template <typename Dtype>
void Tensor<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  std::vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Tensor<Dtype>::Reshape(const std::vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxTensorAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (size_t i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "Tensor size exceeds INT_MAX";
    }
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
void Tensor<Dtype>::Reshape(const TensorShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxTensorAxes);
  std::vector<int> shape_vec(shape.dim_size());
  for (size_t i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Tensor<Dtype>::ReshapeLike(const Tensor<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Tensor<Dtype>::Tensor(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Tensor<Dtype>::Tensor(const std::vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Tensor<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
void Tensor<Dtype>::ShareData(const Tensor& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Tensor<Dtype>::ShareDiff(const Tensor& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

template <typename Dtype>
void Tensor<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    mynet_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::SYNCED:
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <typename Dtype>
Dtype Tensor<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return mynet_cpu_asum(count_, cpu_data());
  case SyncedMemory::SYNCED:
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <typename Dtype>
Dtype Tensor<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return mynet_cpu_asum(count_, cpu_diff());
  case SyncedMemory::SYNCED:
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <typename Dtype>
Dtype Tensor<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = mynet_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::SYNCED:
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <typename Dtype>
Dtype Tensor<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = mynet_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::SYNCED:
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <typename Dtype>
void Tensor<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    mynet_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::SYNCED:
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <typename Dtype>
void Tensor<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    mynet_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::SYNCED:
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Tensor<Dtype>::ShapeEquals(const TensorProto& other) {
  if (other.num() || other.channels() ||
      other.height() || other.width()) {
    // Using deprecated 4D Tensor dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Tensor::num(), Tensor::channels(), etc.
    // methods as these index from the beginning of the Tensor shape, where legacy
    // parameter Tensors were indexed from the end of the Tensor shape (e.g., bias
    // Tensor shape (1 x 1 x 1 x N), IP layer weight Tensor shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  std::vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Tensor<Dtype>::CopyFrom(const Tensor& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy Tensors of different sizes.";
    }
  }
  switch (Mynet::mode()) {
  case Mynet::CPU:
    if (copy_diff) {
      mynet_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      mynet_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown mynet mode.";
  }
}

template <typename Dtype>
void Tensor<Dtype>::FromProto(const TensorProto& proto, bool reshape) {
  if (reshape) {
    std::vector<int> shape;
    if (proto.num() || proto.channels() ||
        proto.height() || proto.width()) {
      // Using deprecated 4D Tensor dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  if (proto.double_data_size() > 0) {
    CHECK_EQ(count_, proto.double_data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.double_data(i);
    }
  } else {
    CHECK_EQ(count_, proto.data_size());
    for (int i = 0; i < count_; ++i) {
      data_vec[i] = proto.data(i);
    }
  }
  if (proto.double_diff_size() > 0) {
    CHECK_EQ(count_, proto.double_diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.double_diff(i);
    }
  } else if (proto.diff_size() > 0) {
    CHECK_EQ(count_, proto.diff_size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (int i = 0; i < count_; ++i) {
      diff_vec[i] = proto.diff(i);
    }
  }
}

template <>
void Tensor<double>::ToProto(TensorProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (size_t i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Tensor<float>::ToProto(TensorProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (size_t i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
  if (write_diff) {
    const float* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_diff(diff_vec[i]);
    }
  }
}

INSTANTIATE_CLASS(Tensor);
template class Tensor<int>;
template class Tensor<unsigned int>;

}  // namespace mynet

