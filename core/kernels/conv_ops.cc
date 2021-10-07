// Copyright 2021 coordinate
// Author: coordinate

#include "conv_ops.hpp"

#include <algorithm>
#include <vector>

#include "core/framework/common.hpp"
#include "core/framework/filler.hpp"
#include "core/framework/math_functions.hpp"

namespace mynet {

template <typename Dtype>
void ConvOps<Dtype>::OpsSetUp(const std::vector<Tensor<Dtype>*>& bottom,
                              const std::vector<Tensor<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  auto& conv_param = this->ops_param_->conv_param;
  force_nd_im2col_ = conv_param->force_nd_im2col;
  transpose_ = conv_param->transpose;
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param->axis);
  auto first_spatial_axis = channel_axis_ + 1;
  auto num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  std::vector<uint32_t> spatial_dim_tensor_shape(
      1, std::max(num_spatial_axes_, 1u));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_tensor_shape);
  uint32_t* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param->kernel_h || conv_param->kernel_w) {
    DCHECK_EQ(num_spatial_axes_, 2ul)
        << "kernel_h & kernel_w can only be used for 2D conv.";
    DCHECK_EQ(0ul, conv_param->kernel_size.size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param->kernel_h;
    kernel_shape_data[1] = conv_param->kernel_w;
  } else {
    auto num_kernel_dims = conv_param->kernel_size.size();
    DCHECK(num_kernel_dims == 1ul || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] =
          conv_param->kernel_size[(num_kernel_dims == 1ul) ? 0ul : i];
    }
  }

  for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
    DCHECK_GT(kernel_shape_data[i], 0ul)
        << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_tensor_shape);
  uint32_t* stride_data = stride_.mutable_cpu_data();
  if (conv_param->stride_h || conv_param->stride_w) {
    DCHECK_EQ(num_spatial_axes_, 2ul)
        << "stride_h & stride_w can only be used for 2D conv.";
    DCHECK_EQ(0ul, conv_param->stride.size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param->stride_h;
    stride_data[1] = conv_param->stride_w;
  } else {
    auto num_stride_dims = conv_param->stride.size();
    DCHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
           num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const uint32_t default_stride = 1;
    for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0ul)
                           ? default_stride
                           : conv_param->stride[(num_stride_dims == 1) ? 0 : i];
      DCHECK_GT(stride_data[i], 0ul) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_tensor_shape);
  uint32_t* pad_data = pad_.mutable_cpu_data();
  if (conv_param->pad_h || conv_param->pad_w) {
    DCHECK_EQ(num_spatial_axes_, 2ul)
        << "pad_h & pad_w can only be used for 2D conv.";
    DCHECK_EQ(0ul, conv_param->pad.size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param->pad_h;
    pad_data[1] = conv_param->pad_w;
  } else {
    auto num_pad_dims = conv_param->pad.size();
    DCHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
           num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; " << num_spatial_axes_
        << " spatial dims).";
    const uint32_t default_pad = 0;
    for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0ul)
                        ? default_pad
                        : conv_param->pad[(num_pad_dims == 1) ? 0 : i];
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_tensor_shape);
  uint32_t* dilation_data = dilation_.mutable_cpu_data();
  auto num_dilation_dims = conv_param->dilation.size();
  DCHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
         num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const uint32_t default_dilation = 1;
  for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] =
        (num_dilation_dims == 0ul)
            ? default_dilation
            : conv_param->dilation[(num_dilation_dims == 1) ? 0 : i];
  }
  // Special case: im2col is the identity for 1x1 conv with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) {
      break;
    }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = conv_param->num_output;
  DCHECK_GT(num_output_, 0ul);
  group_ = conv_param->group;
  DCHECK_EQ(channels_ % group_, 0ul);
  DCHECK_EQ(num_output_ % group_, 0ul)
      << "Number of output should be multiples of group.";
  if (transpose_) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - tensors_[0] holds the filter weights
  // - tensors_[1] holds the biases (optional)
  std::vector<uint32_t> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = conv_param->bias_term;
  std::vector<uint32_t> bias_shape(bias_term_, num_output_);
  if (this->tensors_.size() > 0ul) {
    DCHECK_EQ(1ul + bias_term_, this->tensors_.size())
        << "Incorrect number of weight tensors.";
    if (weight_shape != this->tensors_[0]->shape()) {
      Tensor<Dtype> weight_shaped_tensor(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
                 << weight_shaped_tensor.shape_string()
                 << "; instead, shape was "
                 << this->tensors_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->tensors_[1]->shape()) {
      Tensor<Dtype> bias_shaped_tensor(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
                 << bias_shaped_tensor.shape_string() << "; instead, shape was "
                 << this->tensors_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->tensors_.resize(2);
    } else {
      this->tensors_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->tensors_[0].reset(new Tensor<Dtype>(weight_shape));
    auto weight_filler = GetFiller<Dtype>(conv_param->weight_filler.get());
    weight_filler->Fill(this->tensors_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->tensors_[1].reset(new Tensor<Dtype>(bias_shape));
      auto bias_filler = GetFiller<Dtype>(conv_param->bias_filler.get());
      bias_filler->Fill(this->tensors_[1].get());
    }
  }
  kernel_dim_ = this->tensors_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->tensors_.size(), true);
}

template <typename Dtype>
void ConvOps<Dtype>::Reshape(const std::vector<Tensor<Dtype>*>& bottom,
                             const std::vector<Tensor<Dtype>*>& top) {
  const uint32_t first_spatial_axis = channel_axis_ + 1;
  DCHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  DCHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with conv kernel.";
  // TODO(coordinate): generalize to handle inputs of different shapes.
  for (uint32_t bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    DCHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "shape mismatch - bottom[0]: " << bottom[0]->shape_string()
        << " vs. bottom[" << bottom_id
        << "]: " << bottom[bottom_id]->shape_string();
  }
  // Shape the tops.
  bottom_shape_ = bottom[0]->shape();
  compute_output_shape();
  std::vector<uint32_t> top_shape(bottom[0]->shape().begin(),
                                  bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (uint32_t top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (transpose_) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  std::vector<uint32_t> bottom_dim_tensor_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_tensor_shape);
  uint32_t* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (uint32_t i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (transpose_) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 conv
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (uint32_t i = 0; i < num_spatial_axes_; ++i) {
    if (transpose_) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = transpose_ ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    std::vector<uint32_t> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    auto bias_multiplier_cpu_data = bias_multiplier_.mutable_cpu_data();
    for (uint32_t i = 0; i < bias_multiplier_.count(); i++) {
      bias_multiplier_cpu_data[i] = 1;
    }
  }
}

template <typename Dtype>
void ConvOps<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* weights,
                                      Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (uint32_t g = 0; g < group_; ++g) {
    mynet_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          conv_out_channels_ / group_, conv_out_spatial_dim_,
                          kernel_dim_, (Dtype)1., weights + weight_offset_ * g,
                          col_buff + col_offset_ * g, (Dtype)0.,
                          output + output_offset_ * g);
  }
}

template <typename Dtype>
void ConvOps<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias) {
  mynet_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                        out_spatial_dim_, 1, (Dtype)1., bias,
                        bias_multiplier_.cpu_data(), (Dtype)1., output);
}

template <typename Dtype>
void ConvOps<Dtype>::backward_cpu_gemm(const Dtype* output,
                                       const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (uint32_t g = 0; g < group_; ++g) {
    mynet_cpu_gemm<Dtype>(
        CblasTrans, CblasNoTrans, kernel_dim_, conv_out_spatial_dim_,
        conv_out_channels_ / group_, (Dtype)1., weights + weight_offset_ * g,
        output + output_offset_ * g, (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void ConvOps<Dtype>::weight_cpu_gemm(const Dtype* input, const Dtype* output,
                                     Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (uint32_t g = 0; g < group_; ++g) {
    mynet_cpu_gemm<Dtype>(
        CblasNoTrans, CblasTrans, conv_out_channels_ / group_, kernel_dim_,
        conv_out_spatial_dim_, (Dtype)1., output + output_offset_ * g,
        col_buff + col_offset_ * g, (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void ConvOps<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input) {
  mynet_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1., input,
                        bias_multiplier_.cpu_data(), 1., bias);
}

template <typename Dtype>
void ConvOps<Dtype>::compute_output_shape() {
  const uint32_t* kernel_shape_data = this->kernel_shape_.cpu_data();
  const uint32_t* stride_data = this->stride_.cpu_data();
  const uint32_t* pad_data = this->pad_.cpu_data();
  const uint32_t* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (uint32_t i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const uint32_t input_dim = this->input_shape(i + 1);
    const uint32_t kernel_extent =
        dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    uint32_t output_dim = 0;
    if (transpose_) {
      output_dim =
          stride_data[i] * (input_dim - 1) + kernel_extent - 2 * pad_data[i];
    } else {
      output_dim =
          (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    }
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvOps<Dtype>::ForwardCpu(const std::vector<Tensor<Dtype>*>& bottom,
                                const std::vector<Tensor<Dtype>*>& top) {
  const Dtype* weight = this->tensors_[0]->cpu_data();
  for (uint32_t i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (uint32_t n = 0; n < this->num_; ++n) {
      if (transpose_) {
        this->backward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                top_data + n * this->top_dim_);
      } else {
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                               top_data + n * this->top_dim_);
      }
      if (this->bias_term_) {
        const Dtype* bias = this->tensors_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvOps<Dtype>::BackwardCpu(const std::vector<Tensor<Dtype>*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Tensor<Dtype>*>& bottom) {
  const Dtype* weight = this->tensors_[0]->cpu_data();
  Dtype* weight_diff = this->tensors_[0]->mutable_cpu_diff();
  for (uint32_t i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->tensors_[1]->mutable_cpu_diff();
      for (uint32_t n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (uint32_t n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          if (transpose_) {
            this->weight_cpu_gemm(top_diff + n * this->top_dim_,
                                  bottom_data + n * this->bottom_dim_,
                                  weight_diff);
          } else {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                  top_diff + n * this->top_dim_, weight_diff);
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          if (transpose_) {
            this->forward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                   bottom_diff + n * this->bottom_dim_,
                                   this->param_propagate_down_[0]);
          } else {
            this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                                    bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(ConvOps);

}  // namespace mynet
