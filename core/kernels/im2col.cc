#include <vector>
#include <cstring> // memset
#include <glog/logging.h>

#include "im2col.hpp"

namespace mynet {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t pad_h, uint32_t pad_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t dilation_h, uint32_t dilation_w,
    Dtype* data_col) {
  DCHECK(data_im);
  uint32_t output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  uint32_t output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  uint32_t channel_size = height * width;
  for (uint32_t channel = 0; channel < channels; data_im += channel_size, channel++) {
    for (uint32_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (uint32_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        uint32_t input_row = -pad_h + kernel_row * dilation_h; // will be neg
        for (uint32_t output_rows = 0; output_rows < output_h; output_rows++) {
          if (input_row >= height) {
            for (uint32_t output_cols = 0; output_cols < output_w; output_cols++) {
              *(data_col++) = 0;
            }
          } else {
            uint32_t input_col = -pad_w + kernel_col * dilation_w; // will be neg
            for (uint32_t output_col = 0; output_col < output_w; output_col++) {
              if (input_col < width) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t pad_h, uint32_t pad_w, uint32_t stride_h,
    uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t pad_h, uint32_t pad_w, uint32_t stride_h,
    uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w,
    double* data_col);

template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, bool im2col,
    uint32_t num_spatial_axes, const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, Dtype* data_output) {
  if (!im2col) {
    uint32_t im_size = im_shape[0];
    for (uint32_t i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    std::memset(data_output, 0, sizeof(Dtype) * im_size);
  }
  uint32_t kernel_size = 1;
  for (uint32_t i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  uint32_t channels_col = col_shape[0];
  std::vector<uint32_t> d_offset(num_spatial_axes, 0);
  std::vector<uint32_t> d_iter(num_spatial_axes, 0);
  for (uint32_t c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    uint32_t offset = c_col;
    DCHECK_LE(num_spatial_axes - 1, UINT32_MAX);
    int32_t num_spatial_axes_sub = static_cast<int32_t>(num_spatial_axes - 1);
    for (int32_t d_i = num_spatial_axes_sub; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes_sub) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      uint32_t index_col = c_col;
      uint32_t index_im = c_col / kernel_size;
      bool is_padding = false;
      for (uint32_t d_i = 0; d_i < num_spatial_axes; ++d_i) {
        uint32_t d = d_iter[d_i];
        uint32_t d_im = d * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      DCHECK_LE(num_spatial_axes - 1, UINT32_MAX);
      for (uint32_t d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        uint32_t d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int32_t c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, uint32_t num_spatial_axes,
    const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, Dtype* data_col) {
  im2col_nd_core_cpu(data_im, true, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, dilation, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    uint32_t num_spatial_axes,
    const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    uint32_t num_spatial_axes,
    const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t pad_h, uint32_t pad_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t dilation_h, uint32_t dilation_w,
    Dtype* data_im) {
  std::memset(data_im, 0, sizeof(Dtype) * height * width * channels);
  uint32_t output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  uint32_t output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  uint32_t channel_size = height * width;
  for (uint32_t channel = 0; channel < channels; data_im += channel_size) {
    for (uint32_t kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (uint32_t kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        uint32_t input_row = -pad_h + kernel_row * dilation_h;
        for (uint32_t output_rows = 0; output_rows < output_h; output_rows++) {
          if (input_row >= height) {
            data_col += output_w;
          } else {
            uint32_t input_col = -pad_w + kernel_col * dilation_w;
            for (uint32_t output_col = 0; output_col < output_w; output_col++) {
              if (input_col < width) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t pad_h, uint32_t pad_w, uint32_t stride_h,
    uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, uint32_t channels,
    uint32_t height, uint32_t width, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t pad_h, uint32_t pad_w, uint32_t stride_h,
    uint32_t stride_w, uint32_t dilation_h, uint32_t dilation_w,
    double* data_im);

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, uint32_t num_spatial_axes,
    const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, Dtype* data_im) {
  im2col_nd_core_cpu(data_col, false, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    uint32_t num_spatial_axes,
    const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    uint32_t num_spatial_axes,
    const uint32_t* im_shape, const uint32_t* col_shape,
    const uint32_t* kernel_shape, const uint32_t* pad, const uint32_t* stride,
    const uint32_t* dilation, double* data_im);


}  // namespace mynet
