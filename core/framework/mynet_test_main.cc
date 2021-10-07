// Copyright 2021 coordinate
// Author: coordinate

#include "mynet_test_main.hpp"

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  mynet::GlobalInit(&argc, &argv);
  return RUN_ALL_TESTS();
}
