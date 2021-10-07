// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef MYNET_CORE_FRAMEWORK_MYNET_MAIN_HPP_
#define MYNET_CORE_FRAMEWORK_MYNET_MAIN_HPP_

#include "common.hpp"

int main(int argc, char** argv);

namespace mynet {

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
public:
  typedef typename TypeParam::Dtype Dtype;
protected:
  MultiDeviceTest() {
    // Mynet::set_mode(TypeParam::device);
  }
  virtual ~MultiDeviceTest() {}
};

typedef ::testing::Types<float, double> TestDtypes;

template <typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static const Mynet::Mode device = Mynet::CPU;
};

typedef ::testing::Types<CPUDevice<float>,
                         CPUDevice<double>> TestDtypesAndDevices;
} 

#endif  // MYNET_CORE_FRAMEWORK_MYNET_MAIN_HPP_