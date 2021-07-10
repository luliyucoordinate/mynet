#ifndef MYNET_CC_COMMON_HPP_
#define MYNET_CC_COMMON_HPP_

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <climits>
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

protected:
  Mode mode_;

private:
  // The private constructor to avoid duplicate instantiation.
  Mynet();

  DISABLE_COPY_AND_ASSIGN(Mynet);
};

} // namespace mynet

#endif  // MYNET_CC_COMMON_HPP_
