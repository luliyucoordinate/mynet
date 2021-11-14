// Copyright 2021 coordinate
// Author: coordinate

#include "dummy_data_op.hpp"

#include <vector>

#include "core/framework/op_factory.hpp"
#include "core/schema/filler_generated.h"
#include "core/schema/op_generated.h"

namespace mynet {

template <typename Dtype>
void DummyDataOp<Dtype>::OpSetUp(const std::vector<Tensor<Dtype>*>& input,
                                 const std::vector<Tensor<Dtype>*>& output) {
  uint32_t num_output = output.size();
  const auto& param = this->op_param_->dummy_data_param;
  uint32_t num_data_filler = param->data_filler.size();
  DCHECK(num_data_filler == 0ul || num_data_filler == 1ul ||
         num_data_filler == num_output)
      << "Number of data fillers must be 0, 1 or equal to the number of "
         "output: "
      << num_output << "; you specified " << num_data_filler
      << " data fillers.";

  DCHECK(param->shape.size() == 1ul || param->shape.size() == num_output)
      << "Must specify 'shape' once, or once per output tensor "
      << "(" << num_output << "); specified " << param->shape.size() << ".";
  // refill_[i] tells Forward i whether or not to actually refill output Tensor
  // i. If refill_[i] is false, Forward does nothing for Tensor i. We use this
  // to avoid wastefully refilling "constant" Tensors in every forward pass. We
  // first fill refill_ in with the INVERSE of its final values. The first time
  // we run Forward from the OpSetUp method, we'll fill only Tensors for which
  // refill_ is normally false.  These Tensors will never be filled again.
  refill_.clear();
  fillers_.clear();
  if (num_data_filler <= 1ul) {
    FillerParameterT filler_param;
    if (num_data_filler == 0ul) {
      filler_param.type = "constant";
      filler_param.value = 0.0f;
    } else {
      // TODO(coordinate): deep copy
      filler_param = *(param->data_filler[0]);
    }
    // Refill on each iteration iff not using a constant filler,
    // but use the inverse of this rule for the first run.
    refill_.resize(1);
    refill_[0] = (filler_param.type == "constant");
    fillers_.resize(1);
    fillers_[0].reset(GetFiller<Dtype>(&filler_param));
  } else {
    refill_.resize(num_output);
    fillers_.resize(num_output);
    for (uint32_t i = 0; i < num_output; ++i) {
      fillers_[i].reset(GetFiller<Dtype>(param->data_filler[i].get()));
      // Refill on each iteration iff not using a constant filler,
      // but use the inverse of this rule for the first run.
      refill_[i] = (param->data_filler[i]->type == "constant");
    }
  }
  for (uint32_t i = 0; i < num_output; ++i) {
    uint32_t shape_index = (param->shape.size() == 1ul) ? 0ul : i;
    output[i]->Reshape(param->shape[shape_index].get());
  }
  // Run Forward once, with refill_ inverted, to fill the constant Tensors.
  this->Forward(input, output);
  // Invert the inverted refill_ values to refill the desired (non-constant)
  // Tensors in every usual forward pass.
  for (uint32_t i = 0; i < refill_.size(); ++i) {
    refill_[i] = !refill_[i];
  }
}

template <typename Dtype>
void DummyDataOp<Dtype>::ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                                    const std::vector<Tensor<Dtype>*>& output) {
  for (uint32_t i = 0; i < output.size(); ++i) {
    uint32_t filler_id = (fillers_.size() > 1ul) ? i : 0ul;
    if (refill_[filler_id]) {
      fillers_[filler_id]->Fill(output[i]);
    }
  }
}

INSTANTIATE_CLASS(DummyDataOp);
REGISTER_OP_CLASS(DummyData);

}  // namespace mynet
