// Copyright 2021 coordinate
// Author: coordinate

#include "split_op.hpp"

#include <vector>

#include "core/framework/math_functions.hpp"
#include "core/framework/op_factory.hpp"

namespace mynet {

template <typename Dtype>
void SplitOp<Dtype>::Reshape(const std::vector<Tensor<Dtype>*>& input,
                             const std::vector<Tensor<Dtype>*>& output) {
  count_ = input[0]->count();
  for (uint32_t i = 0; i < output.size(); ++i) {
    // Do not allow in-place computation in the SplitOp.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    DCHECK_NE(output[i], input[0]) << this->type()
                                   << " Op does not "
                                      "allow in-place computation.";
    output[i]->ReshapeLike(*input[0]);
    DCHECK_EQ(count_, output[i]->count());
  }
}

template <typename Dtype>
void SplitOp<Dtype>::ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                                const std::vector<Tensor<Dtype>*>& output) {
  for (uint32_t i = 0; i < output.size(); ++i) {
    output[i]->ShareData(*input[0]);
  }
}

template <typename Dtype>
void SplitOp<Dtype>::BackwardCpu(const std::vector<Tensor<Dtype>*>& output,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Tensor<Dtype>*>& input) {
  if (!propagate_down[0]) {
    return;
  }
  if (output.size() == 1) {
    mynet_copy(input[0]->mutable_cpu_diff(), output[0]->cpu_diff(), count_);
    return;
  }
  mynet_add(count_, output[0]->cpu_diff(), output[1]->cpu_diff(),
            input[0]->mutable_cpu_diff());
  // Add remaining output blob diffs.
  for (uint32_t i = 2; i < output.size(); ++i) {
    const Dtype* output_diff = output[i]->cpu_diff();
    Dtype* input_diff = input[0]->mutable_cpu_diff();
    mynet_axpy(count_, Dtype(1.), output_diff, input_diff);
  }
}

INSTANTIATE_CLASS(SplitOp);
REGISTER_OP_CLASS(Split);

}  // namespace mynet
