#include "math_functions.hpp"
#include <random>

namespace mynet {

template <>
void mynet_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void mynet_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void mynet_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    std::memcpy(Y, X, sizeof(Dtype) * N);  
  }
}

template void mynet_copy<int>(const int N, const int* X, int* Y);
template void mynet_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void mynet_copy<float>(const int N, const float* X, float* Y);
template void mynet_copy<double>(const int N, const double* X, double* Y);

template <>
float mynet_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double mynet_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
float mynet_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double mynet_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype mynet_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return mynet_cpu_strided_dot(n, x, 1, y, 1);
}

template 
float mynet_cpu_dot<float>(const int n, const float* x, const float* y);

template 
double mynet_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
void mynet_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void mynet_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void mynet_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void mynet_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <typename Dtype>
void mynet_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);

  std::random_device rd;
  std::mt19937 gen{rd()};

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  std::uniform_real_distribution<Dtype> dis(a, b);
  for (int i = 0; i < n; ++i) {
    r[i] = dis(gen);
  }
}

template
void mynet_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void mynet_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

}  // namespace mynet