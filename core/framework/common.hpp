#ifndef MYNET_CC_COMMON_HPP_
#define MYNET_CC_COMMON_HPP_

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cmath>
#include <fstream>  
#include <iostream> 
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <cstring>  // memset
#include <utility>  // pair
#include <vector>
#include <thread>


#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

namespace mynet {

void GlobalInit(int* pargc, char*** pargv);


// A singleton class to hold common mynet stuff, such as the handler that
// mynet is going to use for cublas, curand, etc.
class Mynet {
public:
  ~Mynet();

  enum Mode { CPU, GPU };

  // Thread local context for Mynet. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  static Mynet& Get();

  inline static Mode mode() { return Get().mode_; }
  inline static bool root_solver() { return Get().solver_rank_ == 0ul; }

protected:
  Mode mode_;

  // Parallel training
  uint32_t solver_count_;
  uint32_t solver_rank_;
  bool multiprocess_;

private:
  // The private constructor to avoid duplicate instantiation.
  Mynet();

  DISABLE_COPY_AND_ASSIGN(Mynet);
};

} // namespace mynet

#endif  // MYNET_CC_COMMON_HPP_
