#include "tensor.hpp"
#include "common.hpp"
#include "syncedmem.hpp"
#include "math_functions.hpp"
#include "flatbuffers/flatbuffers.h"

namespace mynet {

template <typename Dtype>
Tensor<Dtype>::Tensor(uint32_t num, uint32_t channels, uint32_t height,
    uint32_t width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0ul) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Tensor<Dtype>::Tensor(const std::vector<uint32_t>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0ul) {
  Reshape(shape);
}

template <typename Dtype>
void Tensor<Dtype>::Reshape(uint32_t num, uint32_t channels, uint32_t height,
    uint32_t width) {
  std::vector<uint32_t> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Tensor<Dtype>::Reshape(const std::vector<uint32_t>& shape) {
  DCHECK_LE(shape.size(), kMaxTensorAxes);
  count_ = 1ul;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(uint32_t)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(uint32_t)));
  }
  uint32_t* shape_data = static_cast<uint32_t*>(shape_data_->mutable_cpu_data());
  for (uint32_t i = 0; i < shape.size(); ++i) {
    // TODO: DCHECK_GT ? Should be zero ?
    // DCHECK_GE(shape[i], 0); 
    if (count_ > 0) {
      DCHECK_LE(shape[i], UINT32_MAX / count_) << "Tensor size exceeds UINT32_MAX";
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
void Tensor<Dtype>::Reshape(const TensorShapeT* shape) {
  DCHECK(shape);
  auto shape_dim = shape->dim;
  DCHECK_LE(shape_dim.size(), kMaxTensorAxes);
  std::vector<uint32_t> shape_vec(shape_dim.size());
  for (uint32_t i = 0; i < shape_dim.size(); ++i) {
    shape_vec[i] = shape_dim[i];
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Tensor<Dtype>::ReshapeLike(const Tensor<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_data() const {
  DCHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Tensor<Dtype>::set_cpu_data(Dtype* data) {
  DCHECK(data);
  // Make sure CPU and GPU sizes remain equal
  uint32_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Tensor<Dtype>::cpu_diff() const {
  DCHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_cpu_data() {
  DCHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Tensor<Dtype>::mutable_cpu_diff() {
  DCHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
void Tensor<Dtype>::ShareData(const Tensor& other) {
  DCHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Tensor<Dtype>::ShareDiff(const Tensor& other) {
  DCHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

// The "update" method is used for parameter tensors in a Net, which are stored
// as Tensor<float> or Tensor<double> -- hence we do not define it for
// Tensor<int32_t> or Tensor<uint32_t>.
template <> void Tensor<uint32_t>::Update() { NOT_IMPLEMENTED; }
template <> void Tensor<int32_t>::Update() { NOT_IMPLEMENTED; }

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

template <> uint32_t Tensor<uint32_t>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int32_t Tensor<int32_t>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
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

template <> uint32_t Tensor<uint32_t>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int32_t Tensor<int32_t>::asum_diff() const {
  NOT_IMPLEMENTED;
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

template <> uint32_t Tensor<uint32_t>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int32_t Tensor<int32_t>::sumsq_data() const {
  NOT_IMPLEMENTED;
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

template <> uint32_t Tensor<uint32_t>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int32_t Tensor<int32_t>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
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

template <> void Tensor<uint32_t>::scale_data(uint32_t scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Tensor<int32_t>::scale_data(int32_t scale_factor) {
  NOT_IMPLEMENTED;
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

template <> void Tensor<uint32_t>::scale_diff(uint32_t scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Tensor<int32_t>::scale_diff(int32_t scale_factor) {
  NOT_IMPLEMENTED;
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
bool Tensor<Dtype>::ShapeEquals(const TensorFlatT* other) {
  if (other->num || other->channels ||
      other->height || other->width) {
    // Using deprecated 4D Tensor dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Tensor::num(), Tensor::channels(), etc.
    // methods as these index from the beginning of the Tensor shape, where legacy
    // parameter Tensors were indexed from the end of the Tensor shape (e.g., bias
    // Tensor shape (1 x 1 x 1 x N), IP layer weight Tensor shape (1 x 1 x M x N)).

    return shape_.size() <= 4 &&
           LegacyShape(-4) == other->num &&
           LegacyShape(-3) == other->channels &&
           LegacyShape(-2) == other->height &&
           LegacyShape(-1) == other->width;
  }

  auto other_shape_dim = other->shape->dim;

  std::vector<uint32_t> other_shape(other_shape_dim.size());
  for (uint32_t i = 0; i < other_shape_dim.size(); ++i) {
    other_shape[i] = other_shape_dim[i];
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
      mynet_copy(static_cast<Dtype*>(diff_->mutable_cpu_data()), source.cpu_diff(), count_);
    } else {
      mynet_copy(static_cast<Dtype*>(data_->mutable_cpu_data()), source.cpu_data(), count_);
    }
    break;
  default:
    LOG(FATAL) << "Unknown mynet mode.";
  }
}

template <typename Dtype>
void Tensor<Dtype>::FromFlat(const TensorFlatT* flat, bool reshape) {
  if (reshape) {
    std::vector<uint32_t> shape;
    if (flat->num || flat->channels ||
        flat->height || flat->width) {
      // Using deprecated 4D Tensor dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = flat->num;
      shape[1] = flat->channels;
      shape[2] = flat->height;
      shape[3] = flat->width;
    } else {
      auto flat_shape_dim = flat->shape->dim;
      shape.resize(flat_shape_dim.size());
      for (uint32_t i = 0; i < flat_shape_dim.size(); ++i) {
        shape[i] = flat_shape_dim[i];
      }
    }
    Reshape(shape);
  } else {
    DCHECK(ShapeEquals(flat)) << "shape mismatch (reshape not set)";
  }
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  auto flat_double_data = flat->double_data;
  auto flat_data = flat->data;
  if (flat_double_data.size() > 0) {
    DCHECK_EQ(count_, flat_double_data.size());
    for (uint32_t i = 0; i < count_; ++i) {
      data_vec[i] = flat_double_data[i];
    }
  } else {
    DCHECK_EQ(count_, flat_data.size());
    for (uint32_t i = 0; i < count_; ++i) {
      data_vec[i] = flat_data[i];
    }
  }

  auto flat_double_diff = flat->double_diff;
  auto flat_diff = flat->diff;
  if (flat_double_diff.size() > 0) {
    DCHECK_EQ(count_, flat_double_diff.size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (uint32_t i = 0; i < count_; ++i) {
      diff_vec[i] = flat_double_diff[i];
    }
  } else if (flat_diff.size() > 0) {
    DCHECK_EQ(count_, flat_diff.size());
    Dtype* diff_vec = mutable_cpu_diff();
    for (uint32_t i = 0; i < count_; ++i) {
      diff_vec[i] = flat_diff[i];
    }
  }
}

template <>
flatbuffers::DetachedBuffer Tensor<double>::ToFlat(bool write_diff) const {
  // auto tensor_shape = CreateTensorShapeDirect(flatbuffer_builder, &shape_);
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
  tensor_flat.double_data = data;
  tensor_flat.double_diff = diff;
  // auto tensor_flat = CreateTensorFlatDirect(flatbuffer_builder, num(), channels(), height(), width(), nullptr, nullptr, tensor_shape, data_ptr, diff_ptr);
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  flatbuffer_builder.Finish(TensorFlat::Pack(flatbuffer_builder, &tensor_flat));
  return flatbuffer_builder.Release();
}

template <>
flatbuffers::DetachedBuffer Tensor<float>::ToFlat(bool write_diff) const {
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
  tensor_flat.data = data;
  tensor_flat.diff = diff;
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  flatbuffer_builder.Finish(TensorFlat::Pack(flatbuffer_builder, &tensor_flat));
  return flatbuffer_builder.Release();
}

INSTANTIATE_CLASS(Tensor);
template class Tensor<int32_t>;
template class Tensor<uint32_t>;

}  // namespace mynet

