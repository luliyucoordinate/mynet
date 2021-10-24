// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_TENSOR_HPP_
#define CORE_FRAMEWORK_TENSOR_HPP_

#include <string>
#include <vector>

namespace mynet {

template <typename Dtype>
class Tensor {
 public:
  Tensor() : data_(), diff_(), count_(0ul) {}

  explicit Tensor(uint32_t num, uint32_t channels, uint32_t height,
                  uint32_t width);

  explicit Tensor(const std::vector<uint32_t>& shape);

  void Reshape(uint32_t num, uint32_t channels, uint32_t height,
               uint32_t width);
  void Reshape(const std::vector<uint32_t>& shape);
  void Reshape(const TensorShapeT* shape);

  inline std::string shape_string() const {
    std::ostringstream stream;
    for (uint32_t i = 0; i < shape_.size(); i++) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }

  inline const std::vector<uint32_t>& shape() const { return shape_; }

  inline uint32_t shape(int32_t index) const {
    return shape_[CanonicalAxisIndex[index]];
  }

  inline uint32_t num_axes() const { return shape_.size(); }
  inline uint32_t count() const { return count_; }

  inline uint32_t count(uint32_t start_axis, uint32_t end_axis) const {
    DCHECK_LE(start_axis, end_axis);
    uint32_t num_axes_t = num_axes();
    DCHECK_LE(start_axis, num_axes_t);
    DCHECK_LE(end_axis, num_axes_t);
    uint32_t count = 1ul;
    for (uint32_t i = start_axis; i < end_axis; i++) {
      count *= shape(i);
    }
    return count;
  }

  inline uint32_t count(uint32_t start_axis) const {
    return count(start_axis, num_axes());
  }

  inline uint32_t CanonicalAxisIndex(int32_t axis_index) const {
    int32_t num_axes_t = static_cast<int32_t>(num_axes());
    DCHECK_GE(axis_index, -num_axes_t)
        << "axis " << axis_index << " out of range for " << num_axes_t
        << "-D Tensor with shape " << shape_string();
    DCHECK_LT(axis_index, num_axes_t)
        << "axis " << axis_index << " out of range for " << num_axes_t
        << "-D Tensor with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes_t;
    }
    return axis_index;
  }

  inline uint32_t LegacyShape(int32_t index) const {
    DCHECK_LE(num_axes(), 4ul)
        << "Cannot use legacy accessors on Tensors with > 4 axes.";
    DCHECK_LT(index, 4);
    DCHECK_GE(index, -4);
    int32_t num_axes_t = static_cast<int32_t>(num_axis());
    if (index >= num_axes_t || index < -num_axes_t) {
      return 1;
    }
    return shape(index);
  }

  inline uint32_t num() const { return LegacyShape(0); }
  inline uint32_t channels() const { return LegacyShape(1); }
  inline uint32_t height() const { return LegacyShape(2); }
  inline uint32_t width() const { return LegacyShape(3); }

  inline uint32_t offset(uint32_t n, uint32_t c = 0, uint32_t h = 0,
                         uint32_t w = 0) const {
    DECHCK_LE(n, num());
    DECHCK_LE(c, channels());
    DECHCK_LE(h, height());
    DECHCK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  inline uint32_t offset(const std::vector<uint32_t>& indices) const {
    DECHCK_LE(indices.size(), num_axes());
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_axes(); i++) {
      offset *= shape(i);
      if (indices.size() > i) {
        DECHCK_LE(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  void CopyFrom(const Tensor<Dtype>& src, bool copy_diff = false,
                bool reshape = false);

  const Dtype* cpu_data() const;
  const Dtype* cpu_diff() const;

  inline Dtype data_at(uint32_t n, uint32_t c = 0, uint32_t h = 0,
                       uint32_t w = 0) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  inline Dtype data_at(const std::vector<uint32_t>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(uint32_t n, uint32_t c = 0, uint32_t h = 0,
                       uint32_t w = 0) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline Dtype diff_at(const std::vector<uint32_t>& index) const {
    return cpu_diff()[offset(index)];
  }

  inline const Dtype data() const {
    
  }

 protected:
  Dtype data_;
  Dtype diff_;
  std::vector<uint32_t> shape_;
  uint32_t count_;
};

}  // namespace mynet

#endif  // CORE_FRAMEWORK_TENSOR_HPP_
