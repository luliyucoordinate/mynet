#include "math_functions.hpp"
#include <random>

namespace mynet {

template<>
void mynet_cpu_gemm<float>(CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB, uint32_t M, uint32_t N, uint32_t K,
    float alpha, const float* A, const float* B, float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, (int)M, (int)N, (int)K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void mynet_cpu_gemm<double>(CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB, uint32_t M, uint32_t N, uint32_t K,
    double alpha, const double* A, const double* B, double beta,
    double* C) {
  int lda = (int)((TransA == CblasNoTrans) ? K : M);
  int ldb = (int)((TransB == CblasNoTrans) ? N : K);
  cblas_dgemm(CblasRowMajor, TransA, TransB, (int)M, (int)N, (int)K, alpha, A, lda, B,
      ldb, beta, C, (int)N);
}

template <>
void mynet_cpu_gemv<float>(CBLAS_TRANSPOSE TransA, uint32_t M,
    uint32_t N, float alpha, const float* A, const float* x,
    float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, (int)M, (int)N, alpha, A, (int)N, x, 1, beta, y, 1);
}

template <>
void mynet_cpu_gemv<double>(CBLAS_TRANSPOSE TransA, uint32_t M,
    uint32_t N, double alpha, const double* A, const double* x,
    double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, (int)M, (int)N, alpha, A, (int)N, x, 1, beta, y, 1);
}

template <>
void mynet_axpy<float>(uint32_t N, float alpha, const float* X,
    float* Y) { cblas_saxpy((int)N, alpha, X, 1, Y, 1); }

template <>
void mynet_axpy<double>(uint32_t N, double alpha, const double* X,
    double* Y) { cblas_daxpy((int)N, alpha, X, 1, Y, 1); }

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
void mynet_set(Dtype* Y, Dtype alpha, uint32_t N) {
  if (alpha == 0) {
    std::memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (uint32_t i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void mynet_set<int>(int* Y, int alpha, uint32_t N);
template void mynet_set<uint32_t>(uint32_t* Y, uint32_t alpha, uint32_t N);
template void mynet_set<float>(float* Y, float alpha, uint32_t N);
template void mynet_set<double>(double* Y, double alpha, uint32_t N);

template <>
float mynet_cpu_asum<float>(uint32_t n, const float* x) {
  return cblas_sasum((int)n, x, 1);
}

template <>
double mynet_cpu_asum<double>(uint32_t n, const double* x) {
  return cblas_dasum((int)n, x, 1);
}

template <>
float mynet_cpu_strided_dot<float>(uint32_t n, const float* x, uint32_t incx,
    const float* y, uint32_t incy) {
  return cblas_sdot((int)n, x, (int)incx, y, (int)incy);
}

template <>
double mynet_cpu_strided_dot<double>(uint32_t n, const double* x,
    uint32_t incx, const double* y, uint32_t incy) {
  return cblas_ddot((int)n, x, (int)incx, y, (int)incy);
}

template <typename Dtype>
Dtype mynet_cpu_dot(uint32_t n, const Dtype* x, const Dtype* y) {
  return mynet_cpu_strided_dot((int)n, x, 1, y, 1);
}

template 
float mynet_cpu_dot<float>(uint32_t n, const float* x, const float* y);

template 
double mynet_cpu_dot<double>(uint32_t n, const double* x, const double* y);

template <>
void mynet_scal<float>(uint32_t N, float alpha, float *X) {
  cblas_sscal((int)N, alpha, X, 1);
}

template <>
void mynet_scal<double>(uint32_t N, double alpha, double *X) {
  cblas_dscal((int)N, alpha, X, 1);
}

template <>
void mynet_cpu_scale<float>(uint32_t n, float alpha, const float *x,
                            float* y) {
  cblas_scopy((int)n, x, 1, y, 1);
  cblas_sscal((int)n, alpha, y, 1);
}

template <>
void mynet_cpu_scale<double>(uint32_t n, double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <typename Dtype>
void mynet_rng_uniform(uint32_t n, Dtype a, Dtype b, Dtype* r) {
  CHECK_GE(n, 0ul);
  CHECK(r);
  CHECK_LE(a, b);

  std::random_device rd;
  std::mt19937 gen{rd()};

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  std::uniform_real_distribution<Dtype> dis(a, b);
  for (uint32_t i = 0; i < n; ++i) {
    r[i] = dis(gen);
  }
}

template
void mynet_rng_uniform<float>(uint32_t n, float a, float b,
                              float* r);

template
void mynet_rng_uniform<double>(uint32_t n, double a, double b,
                               double* r);

}  // namespace mynet