// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_MYNET_TEST_MAIN_HPP_
#define CORE_FRAMEWORK_MYNET_TEST_MAIN_HPP_

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

typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>>
    TestDtypesAndDevices;
}  // namespace mynet

#endif  // CORE_FRAMEWORK_MYNET_TEST_MAIN_HPP_
