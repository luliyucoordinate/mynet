// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_KERNELS_CONV_OPS_HPP_
#define CORE_KERNELS_CONV_OPS_HPP_

#include <vector>

#include "core/framework/ops.hpp"
#include "core/framework/tensor.hpp"
#include "core/schema/mynet_generated.h"
#include "im2col.hpp"

namespace mynet {

template <typename Dtype>
class ConvOps : public Ops<Dtype> {
 public:
  explicit ConvOps(OpsParameterT* param) : Ops<Dtype>(param) {}
  virtual void OpsSetUp(const std::vector<Tensor<Dtype>*>& input,
                        const std::vector<Tensor<Dtype>*>& output);
  virtual void Reshape(const std::vector<Tensor<Dtype>*>& input,
                       const std::vector<Tensor<Dtype>*>& output);

  virtual inline uint32_t MinBottomTensors() const { return 1; }
  virtual inline uint32_t MinTopTensors() const { return 1; }
  virtual inline bool EqualNumBottomTopTensors() const { return true; }
  virtual inline const char* type() const { return "Conv"; }
  // virtual inline bool reverse_dimensions() { return false; } // TODO: or not

 protected:
  virtual void ForwardCpu(const std::vector<Tensor<Dtype>*>& input,
                          const std::vector<Tensor<Dtype>*>& output);
  virtual void BackwardCpu(const std::vector<Tensor<Dtype>*>& output,
                           const std::vector<bool>& propagate_down,
                           const std::vector<Tensor<Dtype>*>& input);
  virtual void compute_output_shape();

  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output,
                        bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
                         Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

  /// @brief The spatial dimensions of the input.
  inline uint32_t input_shape(uint32_t i) {
    return bottom_shape_[channel_axis_ + i];
  }

  /// @brief The spatial dimensions of a filter kernel.
  Tensor<uint32_t> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Tensor<uint32_t> stride_;
  /// @brief The spatial dimensions of the padding.
  Tensor<uint32_t> pad_;
  /// @brief The spatial dimensions of the dilation.
  Tensor<uint32_t> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  Tensor<uint32_t> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  std::vector<uint32_t> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  std::vector<uint32_t> output_shape_;
  std::vector<uint32_t> bottom_shape_;

  uint32_t num_spatial_axes_;
  uint32_t bottom_dim_;
  uint32_t top_dim_;

  uint32_t channel_axis_;
  uint32_t num_;
  uint32_t channels_;
  uint32_t group_;
  uint32_t out_spatial_dim_;
  uint32_t weight_offset_;
  uint32_t num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;
  bool transpose_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_, conv_input_shape_.cpu_data()[1],
                 conv_input_shape_.cpu_data()[2], kernel_shape_.cpu_data()[0],
                 kernel_shape_.cpu_data()[1], pad_.cpu_data()[0],
                 pad_.cpu_data()[1], stride_.cpu_data()[0],
                 stride_.cpu_data()[1], dilation_.cpu_data()[0],
                 dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
                    col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                    pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(),
                    col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_, conv_input_shape_.cpu_data()[1],
                 conv_input_shape_.cpu_data()[2], kernel_shape_.cpu_data()[0],
                 kernel_shape_.cpu_data()[1], pad_.cpu_data()[0],
                 pad_.cpu_data()[1], stride_.cpu_data()[0],
                 stride_.cpu_data()[1], dilation_.cpu_data()[0],
                 dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
                    col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                    pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(),
                    data);
    }
  }

  uint32_t num_kernels_im2col_;
  uint32_t num_kernels_col2im_;
  uint32_t conv_out_channels_;
  uint32_t conv_in_channels_;
  uint32_t conv_out_spatial_dim_;
  uint32_t kernel_dim_;
  uint32_t col_offset_;
  uint32_t output_offset_;

  Tensor<Dtype> col_buffer_;
  Tensor<Dtype> bias_multiplier_;
};

}  // namespace mynet

#endif  // CORE_KERNELS_CONV_OPS_HPP_
