// Fillers are random number generators that fills a Tensor using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef MYNET_CC_FILLER_HPP_
#define MYNET_CC_FILLER_HPP_

#include <string>

#include "tensor.hpp"
#include "math_functions.hpp"

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

/// @brief Fills a Tensor with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameterT* param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Tensor<Dtype>* Tensor) {
    DCHECK(Tensor->count());
    mynet_rng_uniform<Dtype>(Tensor->count(), Dtype(this->filler_param_->min),
        Dtype(this->filler_param_->max), Tensor->mutable_cpu_data());
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
Filler<Dtype>* GetFiller(const FillerParameterT* param) {
  const std::string& type = param->type;

  if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else {
    DCHECK(false) << "Unknown filler name: " << type;
  }
  return (Filler<Dtype>*)(NULL);
}

}  // namespace mynet

#endif  // MYNET_CC_FILLER_HPP_
