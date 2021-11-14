// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_FILLER_HPP_
#define CORE_FRAMEWORK_FILLER_HPP_

#include <memory>
#include <string>

#include "core/schema/filler_generated.h"
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

/// @brief Fills a Tensor with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameterT* param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Tensor<Dtype>* tensor) {
    DCHECK(tensor);
    Dtype* data = tensor->mutable_cpu_data();
    uint32_t count = tensor->count();
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

/// @brief Fills a Tensor with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameterT* param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Tensor<Dtype>* tensor) {
    Dtype* data = tensor->mutable_cpu_data();
    DCHECK(tensor->count());
    mynet_rng_gaussian<Dtype>(tensor->count(), Dtype(this->filler_param_->mean),
                              Dtype(this->filler_param_->std),
                              tensor->mutable_cpu_data());
    int32_t sparse = this->filler_param_->sparse;
    DCHECK_GE(sparse, -1);
    if (sparse >= 0) {
      // Sparse initialization is implemented for "weight" blobs; i.e. matrices.
      // These have num == channels == 1; width is number of inputs; height is
      // number of outputs.  The 'sparse' variable specifies the mean number
      // of non-zero input weights for a given output.
      DCHECK_GE(tensor->num_axes(), 1ul);
      uint32_t num_outputs = tensor->shape(0);
      Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
      rand_vec_.reset(new SyncedMemory(tensor->count() * sizeof(uint32_t)));
      uint32_t* mask =
          reinterpret_cast<uint32_t*>(rand_vec_->mutable_cpu_data());
      mynet_rng_bernoulli(tensor->count(), non_zero_probability, mask);
      for (uint32_t i = 0; i < tensor->count(); ++i) {
        data[i] *= mask[i];
      }
    }
  }

 protected:
  std::shared_ptr<SyncedMemory> rand_vec_;
};

/**
 * @brief Get a specific filler from the specification given in FillerParameter.
 *
 * Ideally this would be replaced by a factory pattern, but we will leave it
 * this way for now.
 */
template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameterT* param) {
  DCHECK(param);
  const std::string& type = param->type;
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else {
    DCHECK(false) << "Unknown filler name: " << type;
  }
  return nullptr;
}

}  // namespace mynet

#endif  // CORE_FRAMEWORK_FILLER_HPP_
