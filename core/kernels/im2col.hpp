// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_KERNELS_IM2COL_HPP_
#define CORE_KERNELS_IM2COL_HPP_

#include <cstdint>

namespace mynet {

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, uint32_t num_spatial_axes,
    const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, Dtype* data_col);

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t pad_h, uint32_t pad_w, uint32_t stride_h,
    uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w,
    Dtype* data_col);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, uint32_t num_spatial_axes,
    const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, Dtype* data_im);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t pad_h, uint32_t pad_w, uint32_t stride_h,
    uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w,
    Dtype* data_im);

}  // namespace mynet

#endif  // CORE_KERNELS_IM2COL_HPP_
