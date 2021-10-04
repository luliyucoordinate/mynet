#ifndef MYNET_CC_MATH_FUNCTIONS_H_
#define MYNET_CC_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <openblas/cblas.h>
#include "common.hpp"

namespace mynet {

// Mynet gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void mynet_cpu_gemm(CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K,
    Dtype alpha, const Dtype* A, const Dtype* B, Dtype beta,
    Dtype* C);

template <typename Dtype>
void mynet_cpu_gemv(CBLAS_TRANSPOSE TransA, size_t M, size_t N,
    Dtype alpha, const Dtype* A, const Dtype* x, Dtype beta,
    Dtype* y);

template <typename Dtype>
void mynet_axpy(size_t N, Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void mynet_copy(size_t N, const Dtype *X, Dtype *Y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype mynet_cpu_asum(size_t n, const Dtype* x);

template <typename Dtype>
Dtype mynet_cpu_dot(size_t n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype mynet_cpu_strided_dot(size_t n, const Dtype* x, size_t incx,
    const Dtype* y, size_t incy);

template <typename Dtype>
void mynet_scal(size_t N, Dtype alpha, Dtype *X);

template <typename Dtype>
void mynet_cpu_scale(size_t n, Dtype alpha, const Dtype *x, Dtype* y);

template <typename Dtype>
void mynet_rng_uniform(size_t n, Dtype a, Dtype b, Dtype* r);

}  // namespace mynet

#endif  // MYNET_CC_MATH_FUNCTIONS_H_
