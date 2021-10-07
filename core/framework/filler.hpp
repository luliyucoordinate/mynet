// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_FILLER_HPP_
#define CORE_FRAMEWORK_FILLER_HPP_

#include <string>
#include <memory>

#include "math_functions.hpp"
#include "tensor.hpp"

namespace mynet {

/// @brief Fills a Tensor with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameterT* param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void Fill(Tensor<Dtype>* Tensor) = 0;

 protected:
  const FillerParameterT* filler_param_;
};  // class Filler

/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameterT* param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Tensor<Dtype>* tensor) {
    DCHECK(tensor);
    Dtype* data = tensor->mutable_cpu_data();
    const uint32_t count = tensor->count();
    const Dtype value = this->filler_param_->value;
    DCHECK(count);
    for (uint32_t i = 0; i < count; ++i) {
      data[i] = value;
    }
    DCHECK_EQ(this->filler_param_->sparse, -1)
        << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Tensor with uniformly distributed values @f$ x\sim U(a, b)
/// @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameterT* param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Tensor<Dtype>* tensor) {
    DCHECK(tensor);
    DCHECK(tensor->count());
    mynet_rng_uniform<Dtype>(tensor->count(), Dtype(this->filler_param_->min),
                             Dtype(this->filler_param_->max),
                             tensor->mutable_cpu_data());
    // DCHECK_EQ(this->filler_param_.sparse(), -1)
    //      << "Sparsity not supported by this Filler.";
  }
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
std::shared_ptr<Filler<Dtype>> GetFiller(const FillerParameterT* param) {
  DCHECK(param);
  const std::string& type = param->type;
  if (type == "constant") {
    return std::make_shared<ConstantFiller<Dtype>>(param);
  } else if (type == "uniform") {
    return std::make_shared<UniformFiller<Dtype>>(param);
  } else {
    DCHECK(false) << "Unknown filler name: " << type;
  }
  return nullptr;
}

}  // namespace mynet

#endif  // CORE_FRAMEWORK_FILLER_HPP_
