// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_MATH_FUNCTIONS_HPP_
#define CORE_FRAMEWORK_MATH_FUNCTIONS_HPP_

#include <openblas/cblas.h>
#include <stdint.h>

#include "common.hpp"

namespace mynet {

// Mynet gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void mynet_cpu_gemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, uint32_t M,
                    uint32_t N, uint32_t K, Dtype alpha, const Dtype* A,
                    const Dtype* B, Dtype beta, Dtype* C);

template <typename Dtype>
void mynet_cpu_gemv(CBLAS_TRANSPOSE TransA, uint32_t M, uint32_t N, Dtype alpha,
                    const Dtype* A, const Dtype* x, Dtype beta, Dtype* y);

template <typename Dtype>
void mynet_axpy(uint32_t N, Dtype alpha, const Dtype* X, Dtype* Y);

template <typename Dtype>
void mynet_cpu_axpby(uint32_t N, Dtype alpha, const Dtype* X, Dtype beta,
                     Dtype* Y);

template <typename Dtype>
void mynet_copy(Dtype* Y, const Dtype* X, uint32_t N);

// no imply for void Y
template <typename Dtype>
void mynet_set(Dtype* Y, Dtype alpha, uint32_t N);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype mynet_cpu_asum(uint32_t n, const Dtype* x);

template <typename Dtype>
Dtype mynet_cpu_dot(uint32_t n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype mynet_cpu_strided_dot(uint32_t n, const Dtype* x, uint32_t incx,
                            const Dtype* y, uint32_t incy);

template <typename Dtype>
void mynet_scal(uint32_t N, Dtype alpha, Dtype* X);

template <typename Dtype>
void mynet_cpu_scale(uint32_t n, Dtype alpha, const Dtype* x, Dtype* y);

template <typename Dtype>
void mynet_rng_uniform(uint32_t n, Dtype a, Dtype b, Dtype* r);

template <typename Dtype>
void mynet_rng_gaussian(uint32_t n, Dtype mu, Dtype sigma, Dtype* r);

template <typename Dtype>
void mynet_rng_bernoulli(uint32_t n, Dtype p, uint32_t* r);

template <typename Dtype>
void mynet_exp(uint32_t n, const Dtype* a, Dtype* y);

template <typename Dtype>
void mynet_div(uint32_t n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void mynet_mul(uint32_t n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void mynet_add(uint32_t n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void mynet_sub(uint32_t n, const Dtype* a, const Dtype* b, Dtype* y);

}  // namespace mynet

#endif  // CORE_FRAMEWORK_MATH_FUNCTIONS_HPP_
