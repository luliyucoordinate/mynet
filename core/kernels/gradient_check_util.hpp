// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_KERNELS_GRADIENT_CHECK_UTIL_HPP_
#define CORE_KERNELS_GRADIENT_CHECK_UTIL_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "core/framework/op.hpp"
#include "core/framework/tensor.hpp"

namespace mynet {

// The gradient checker adds a L2 normalization loss function on output of the
// output tensors, and checks the gradient.
template <typename Dtype>
class GradientChecker {
 public:
  // kink and kink_range specify an ignored nonsmooth region of the form
  // kink - kink_range <= |feature value| <= kink + kink_range,
  // which accounts for all nonsmoothness in use by mynet
  GradientChecker(Dtype stepsize, Dtype threshold, uint32_t seed = 1701,
                  Dtype kink = 0.0f, Dtype kink_range = -1.0f)
      : stepsize_(stepsize),
        threshold_(threshold),
        seed_(seed),
        kink_(kink),
        kink_range_(kink_range) {}
  // Checks the gradient of a op, with provided input ops and output
  // ops.
  // Note that after the gradient check, we do not guarantee that the data
  // stored in the op parameters and the tensors are unchanged.
  void CheckGradient(Op<Dtype>* op, const std::vector<Tensor<Dtype>*>& input,
                     const std::vector<Tensor<Dtype>*>& output,
                     int32_t check_input = -1) {
    op->SetUp(input, output);
    CheckGradientSingle(op, input, output, check_input, -1, -1);
  }
  void CheckGradientExhaustive(Op<Dtype>* op,
                               const std::vector<Tensor<Dtype>*>& input,
                               const std::vector<Tensor<Dtype>*>& output,
                               int32_t check_input = -1);

  // CheckGradientEltwise can be used to test ops that perform element-wise
  // computation only (e.g., neuron ops) -- where (d y_i) / (d x_j) = 0 when
  // i != j.
  void CheckGradientEltwise(Op<Dtype>* op,
                            const std::vector<Tensor<Dtype>*>& input,
                            const std::vector<Tensor<Dtype>*>& output);

  // Checks the gradient of a single input with respect to particular output
  // tensor(s).  If check_input = i >= 0, check only the ith input Tensor.
  // If check_input == -1, check everything -- all input Tensors and all
  // param Tensors.  Otherwise (if check_input < -1), check only param Tensors.
  void CheckGradientSingle(Op<Dtype>* op,
                           const std::vector<Tensor<Dtype>*>& input,
                           const std::vector<Tensor<Dtype>*>& output,
                           int32_t check_input, int32_t output_id,
                           int32_t output_data_id, bool element_wise = false);

  // Checks the gradient of a network. This network should not have any data
  // ops or loss ops, since the function does not explicitly deal with
  // such cases yet. All output tensors and parameter tensors are going to be
  // checked, op-by-op to avoid numerical problems to accumulate.
  // void CheckGradientNet(const Net<Dtype>& net,
  //                       const std::vector<Tensor<Dtype>*>& output);

 protected:
  Dtype GetObjAndGradient(const Op<Dtype>& op,
                          const std::vector<Tensor<Dtype>*>& output,
                          int32_t output_id = -1, int32_t output_data_id = -1);
  Dtype stepsize_;
  Dtype threshold_;
  uint32_t seed_;
  Dtype kink_;
  Dtype kink_range_;
};

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientSingle(
    Op<Dtype>* op, const std::vector<Tensor<Dtype>*>& input,
    const std::vector<Tensor<Dtype>*>& output, int32_t check_input,
    int32_t output_id, int32_t output_data_id, bool element_wise) {
  if (element_wise) {
    DCHECK_EQ(0ul, op->tensors().size());
    DCHECK_LE(0, output_id);
    DCHECK_LE(0, output_data_id);
    auto output_count = output[static_cast<uint32_t>(output_id)]->count();
    for (uint32_t tensor_id = 0; tensor_id < input.size(); ++tensor_id) {
      DCHECK_EQ(output_count, input[tensor_id]->count());
    }
  }
  // First, figure out what tensors we need to check against, and zero init
  // parameter tensors.
  std::vector<Tensor<Dtype>*> tensors_to_check;
  std::vector<bool> propagate_down(input.size(), check_input == -1);
  for (uint32_t i = 0; i < op->tensors().size(); ++i) {
    Tensor<Dtype>* tensor = op->tensors()[i].get();
    mynet_set(tensor->mutable_cpu_diff(), static_cast<Dtype>(0),
              tensor->count());
    tensors_to_check.push_back(tensor);
  }
  if (check_input == -1) {
    for (uint32_t i = 0; i < input.size(); ++i) {
      tensors_to_check.push_back(input[i]);
    }
  } else if (check_input >= 0) {
    uint32_t check_input_u = static_cast<uint32_t>(check_input);
    DCHECK_LT(check_input_u, input.size());
    tensors_to_check.push_back(input[check_input_u]);
    propagate_down[check_input_u] = true;
  }
  DCHECK_GT(tensors_to_check.size(), 0ul) << "No tensors to check.";
  // Compute the gradient analytically using Backward
  Mynet::set_random_seed(seed_);
  // Ignore the loss from the op (it's just the weighted sum of the losses
  // from the output tensors, whose gradients we may want to test individually).
  op->Forward(input, output);
  // Get additional loss from the objective
  GetObjAndGradient(*op, output, output_id, output_data_id);
  op->Backward(output, propagate_down, input);
  // Store computed gradients for all checked tensors
  std::vector<std::shared_ptr<Tensor<Dtype>>> computed_gradient_tensors(
      tensors_to_check.size());
  for (uint32_t tensor_id = 0; tensor_id < tensors_to_check.size();
       ++tensor_id) {
    Tensor<Dtype>* current_tensor = tensors_to_check[tensor_id];
    computed_gradient_tensors[tensor_id].reset(new Tensor<Dtype>());
    computed_gradient_tensors[tensor_id]->ReshapeLike(*current_tensor);
    auto count = tensors_to_check[tensor_id]->count();
    const Dtype* diff = tensors_to_check[tensor_id]->cpu_diff();
    Dtype* computed_gradients =
        computed_gradient_tensors[tensor_id]->mutable_cpu_data();
    mynet_copy(computed_gradients, diff, count);
  }
  // Compute derivative of output w.r.t. each input and parameter output using
  // finite differencing.
  // LOG(ERROR) << "Checking " << tensors_to_check.size() << " tensors.";
  for (uint32_t tensor_id = 0; tensor_id < tensors_to_check.size();
       ++tensor_id) {
    Tensor<Dtype>* current_tensor = tensors_to_check[tensor_id];
    const Dtype* computed_gradients =
        computed_gradient_tensors[tensor_id]->cpu_data();
    // LOG(ERROR) << "Tensor " << tensor_id << ": checking "
    //     << current_tensor->count() << " parameters.";
    for (uint32_t feat_id = 0; feat_id < current_tensor->count(); ++feat_id) {
      // For an element-wise op, we only need to do finite differencing to
      // compute the derivative of output[output_id][output_data_id] w.r.t.
      // input[tensor_id][i] only for i == output_data_id.  For any other
      // i != output_data_id, we know the derivative is 0 by definition, and
      // simply check that that's true.
      Dtype estimated_gradient = 0;
      Dtype positive_objective = 0;
      Dtype negative_objective = 0;
      if (!element_wise || (static_cast<int32_t>(feat_id) == output_data_id)) {
        // Do finite differencing.
        // Compute loss with stepsize_ added to output.
        current_tensor->mutable_cpu_data()[feat_id] += stepsize_;
        Mynet::set_random_seed(seed_);
        op->Forward(input, output);
        positive_objective =
            GetObjAndGradient(*op, output, output_id, output_data_id);
        // Compute loss with stepsize_ subtracted from output.
        current_tensor->mutable_cpu_data()[feat_id] -= stepsize_ * 2;
        Mynet::set_random_seed(seed_);
        op->Forward(input, output);
        negative_objective =
            GetObjAndGradient(*op, output, output_id, output_data_id);
        // Recover original output value.
        current_tensor->mutable_cpu_data()[feat_id] += stepsize_;
        estimated_gradient =
            (positive_objective - negative_objective) / stepsize_ / 2.;
      }
      Dtype computed_gradient = computed_gradients[feat_id];
      Dtype feature = current_tensor->cpu_data()[feat_id];
      // LOG(ERROR) << "debug: " << current_tensor->cpu_data()[feat_id] << " "
      //     << current_tensor->cpu_diff()[feat_id];
      if (kink_ - kink_range_ > fabs(feature) ||
          fabs(feature) > kink_ + kink_range_) {
        // We check relative accuracy, but for too small values, we threshold
        // the scale factor by 1.
        Dtype scale = std::max<Dtype>(
            std::max(fabs(computed_gradient), fabs(estimated_gradient)),
            Dtype(1.));
        EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
            << "debug: (output_id, output_data_id, tensor_id, feat_id)="
            << output_id << "," << output_data_id << "," << tensor_id << ","
            << feat_id << "; feat = " << feature
            << "; objective+ = " << positive_objective
            << "; objective- = " << negative_objective;
      }
      // LOG(ERROR) << "Feature: " << current_tensor->cpu_data()[feat_id];
      // LOG(ERROR) << "computed gradient: " << computed_gradient
      //    << " estimated_gradient: " << estimated_gradient;
    }
  }
}

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientExhaustive(
    Op<Dtype>* op, const std::vector<Tensor<Dtype>*>& input,
    const std::vector<Tensor<Dtype>*>& output, int32_t check_input) {
  op->SetUp(input, output);
  DCHECK_GT(output.size(), 0ul)
      << "Exhaustive mode requires at least one output tensor.";
  // LOG(ERROR) << "Exhaustive Mode.";
  for (uint32_t i = 0; i < output.size(); ++i) {
    // LOG(ERROR) << "Exhaustive: tensor " << i << " size " <<
    // output[i]->count();
    for (uint32_t j = 0; j < output[i]->count(); ++j) {
      // LOG(ERROR) << "Exhaustive: tensor " << i << " data " << j;
      CheckGradientSingle(op, input, output, check_input,
                          static_cast<int32_t>(i), static_cast<int32_t>(j));
    }
  }
}

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientEltwise(
    Op<Dtype>* op, const std::vector<Tensor<Dtype>*>& input,
    const std::vector<Tensor<Dtype>*>& output) {
  op->SetUp(input, output);
  DCHECK_GT(output.size(), 0ul)
      << "Eltwise mode requires at least one output tensor.";
  int32_t check_input = -1;
  bool element_wise = true;
  for (uint32_t i = 0; i < output.size(); ++i) {
    for (uint32_t j = 0; j < output[i]->count(); ++j) {
      CheckGradientSingle(op, input, output, check_input,
                          static_cast<int32_t>(i), static_cast<int32_t>(j),
                          element_wise);
    }
  }
}

// template <typename Dtype>
// void GradientChecker<Dtype>::CheckGradientNet(
//     const Net<Dtype>& net, const std::vector<Tensor<Dtype>*>& output) {
//   const std::vector<std::shared_ptr<Op<Dtype> > >& ops = net.ops();
//   std::vector<std::vector<Tensor<Dtype>*> >& input_vecs = net.input_vecs();
//   std::vector<std::vector<Tensor<Dtype>*> >& output_vecs = net.output_vecs();
//   for (int32_t i = 0; i < ops.size(); ++i) {
//     net.Forward(output);
//     LOG(ERROR) << "Checking gradient for " <<
//     ops[i]->op_param().name();
//     CheckGradientExhaustive(*(ops[i].get()), input_vecs[i], output_vecs[i]);
//   }
// }

template <typename Dtype>
Dtype GradientChecker<Dtype>::GetObjAndGradient(
    const Op<Dtype>& op, const std::vector<Tensor<Dtype>*>& output,
    int32_t output_id, int32_t output_data_id) {
  Dtype loss = 0;
  if (output_id < 0) {
    // the loss will be half of the sum of squares of all inputs
    for (uint32_t i = 0; i < output.size(); ++i) {
      Tensor<Dtype>* output_tensor = output[i];
      const Dtype* output_tensor_data = output_tensor->cpu_data();
      Dtype* output_tensor_diff = output_tensor->mutable_cpu_diff();
      auto count = output_tensor->count();
      for (uint32_t j = 0; j < count; ++j) {
        loss += output_tensor_data[j] * output_tensor_data[j];
      }
      // set the diff: simply the data.
      mynet_copy(output_tensor_diff, output_tensor_data,
                 output_tensor->count());
    }
    loss /= 2.;
  } else {
    // the loss will be the output_data_id-th element in the output_id-th
    // tensor.
    for (uint32_t i = 0; i < output.size(); ++i) {
      Tensor<Dtype>* output_tensor = output[i];
      Dtype* output_tensor_diff = output_tensor->mutable_cpu_diff();
      mynet_set(output_tensor_diff, Dtype(0), output_tensor->count());
    }
    Dtype loss_weight = 2;
    loss = output[output_id]->cpu_data()[output_data_id] * loss_weight;
    output[output_id]->mutable_cpu_diff()[output_data_id] = loss_weight;
  }
  return loss;
}

}  // namespace mynet

#endif  // CORE_KERNELS_GRADIENT_CHECK_UTIL_HPP_
