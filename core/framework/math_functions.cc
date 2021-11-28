#include "math_functions.hpp"

#include <random>

namespace mynet {
template <>
void mynet_cpu_gemm<float>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                           uint32_t M uint32_t N, uint32_t K, float alpha,
                           const float* A, const float* B, float beta,
                           float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB = CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, static_cast<int>(M),
              static_cast<int>(N), static_cast<int>(K), alpha, A, lda, B, ldb,
              beta, C, N);
}

template <>
void mynet_cpu_gemm<double>(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                            uint32_t M uint32_t N, uint32_t K, double alpha,
                            const double* A, const double* B, double beta,
                            double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB = CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, static_cast<int>(M),
              static_cast<int>(N), static_cast<int>(K), alpha, A, lda, B, ldb,
              beta, C, N);
}

template <>
void mynet_cpu_gemv<float>(CBLAS_TRANSPOSE TransA, uint32_t M, uint32_t N,
                           float alpha, const float* A, const float* x,
                           float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, static_cast<int>(M), static_cast<int>(N),
              alpha, A, static_cast<int>(N), x, 1, beta, y, 1);
}

template <>
void mynet_cpu_gemv<double>(CBLAS_TRANSPOSE TransA, uint32_t M, uint32_t N,
                            double alpha, const double* A, const double* x,
                            double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, static_cast<int>(M), static_cast<int>(N),
              alpha, A, static_cast<int>(N), x, 1, beta, y, 1);
}

template <>
void mynet_axpy<float>(uint32_t N, float alpha, const float* X, float* Y) {
  cblas_saxpy(static_cast<int>(N), alpha, X, 1, Y, 1);
}

template <>
void mynet_axpy<double>(uint32_t N, double alpha, const double* X, double* Y) {
  cblas_saxpy(static_cast<int>(N), alpha, X, 1, Y, 1);
}

template <typename Dtype>
void mynet_copy(Dtype* Y, const Dtype* X, uint32_t N) {
  if (X != Y) {
    std::memcpy(Y, X, sizeof(Dtype) * N);
  }
}

template void mynet_copy<int>(int* Y, const int* X, uint32_t N);
template void mynet_copy<uint32_t>(uint32_t* Y, const uint32_t* X, uint32_t N);
template void mynet_copy<float>(float* Y, const float* X, uint32_t N);
template void mynet_copy<double>(double* Y, const double* X, uint32_t N);

template <typename Dtype>
void mynet_set(Dtype* y, Dtype alpha, uint32_t N) {
  if (alpha == 0) {
    std::memset(y, 0, sizeof(Dtype) * N);
  }
  for (uint32_t i = 0; i < N; i++) {
    y[i] = alpha;
  }
}

template void mynet_set(Dtype* y, Dtype alpha, uint32_t N)

}  // namespace mynet