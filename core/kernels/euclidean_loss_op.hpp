// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_KERNELS_EUCLIDEAN_LOSS_OP_HPP_
#define CORE_KERNELS_EUCLIDEAN_LOSS_OP_HPP_

#include <vector>

#include "core/framework/tensor.hpp"
#include "core/kernels/loss_op.hpp"

namespace mynet {

/**
 * @brief Computes the Euclidean (Euclidean) loss @f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$ for real-valued regression tasks.
 *
 * @param input input Tensor vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param output output Tensor vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Euclidean loss: @f$ E =
 *          \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$
 *
 * This can be used for least-squares regression tasks.  An InnerProductOp
 * input to a EuclideanLossOp exactly formulates a linear least squares
 * regression problem. With non-zero weight decay the problem becomes one of
 * ridge regression -- see src/mynet/test/test_gradient_based_solver.cpp for a
 * concrete example wherein we check that the gradients computed for a Net with
 * exactly this structure match hand-computed gradient formulas for ridge
 * regression.
 *
 * (Note: Caffe, and SGD in general, is certainly \b not the best way to solve
 * linear least squares problems! We use it only as an instructive example.)
 */
template <typename Dtype>
class EuclideanLossOp : public LossOp<Dtype> {
 public:
  explicit EuclideanLossOp(OpParameterT* param)
      : LossOp<Dtype>(param), diff_() {}
  virtual void Reshape(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);

  virtual inline const char* type() const { return "EuclideanLoss"; }
  /**
   * Unlike most loss layers, in the EuclideanLossOp we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(uint32_t input_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanLossOp
  virtual void ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                          const std::vector<Tensor<Dtype>*>& output);

  /**
   * @brief Computes the Euclidean error gradient w.r.t. the inputs.
   *
   * Unlike other children of LossOp, EuclideanLossOp \b can compute
   * gradients with respect to the label inputs input[1] (but still only will
   * if propagate_down[1] is set, due to being produced by learnable parameters
   * or if force_backward is set). In fact, this layer is "commutative" -- the
   * result is the same regardless of the order of the two inputs.
   *
   * @param output output Tensor std::vector (length 1), providing the error
   * gradient with respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Tensor's diff will simply contain the loss_weight* @f$ \lambda
   * @f$, as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this output Tensor is not used as a input (input) by
   * any other layer of the Net.)
   * @param propagate_down see Op::Backward.
   * @param input input Tensor std::vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
   *      @f$ if propagate_down[1]
   */
  virtual void BackwardCpu(const std::vector<Tensor<Dtype>*>& output,
                           const std::vector<bool>& propagate_down,
                           const std::vector<Tensor<Dtype>*>& input);

  Tensor<Dtype> diff_;
};

}  // namespace mynet

#endif  // CORE_KERNELS_EUCLIDEAN_LOSS_OP_HPP_
