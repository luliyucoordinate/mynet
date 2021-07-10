#ifndef MYNET_CC_MATH_FUNCTIONS_H_
#define MYNET_CC_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <openblas/cblas.h>
#include "common.hpp"

namespace mynet {

template <typename Dtype>
void mynet_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void mynet_copy(const int N, const Dtype *X, Dtype *Y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype mynet_cpu_asum(const int n, const Dtype* x);

template <typename Dtype>
Dtype mynet_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype mynet_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

template <typename Dtype>
void mynet_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void mynet_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

template <typename Dtype>
void mynet_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

}  // namespace mynet

#endif  // MYNET_CC_MATH_FUNCTIONS_H_
