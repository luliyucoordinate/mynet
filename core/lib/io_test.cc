// Copyright 2021 coordinate
// Author: coordinate

#include "io.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "core/framework/mynet_test_main.hpp"

namespace mynet {

class IOTest : public ::testing::Test {};

TEST_F(IOTest, ReadWriteNetParamsFromTextFile) {
  auto net_param_t = std::make_shared<NetParameterT>();
  auto net_param = net_param_t.get();
  const char* filename_r = "core/test_data/io_read_test.json";
  ReadNetParamsFromTextFile(filename_r, &net_param);
  const char* filename_w = "core/test_data/io_write_test.json";
  WriteNetParamsToTextFile(net_param, filename_w);
}

TEST_F(IOTest, ReadWriteNetParamsFromBinaryFile) {
  auto net_param_t = std::make_shared<NetParameterT>();
  auto net_param = net_param_t.get();
  const char* filename_r = "core/test_data/io_read_test.bin";
  ReadNetParamsFromBinaryFile(filename_r, &net_param);
  const char* filename_w = "core/test_data/io_write_test.bin";
  WriteNetParamsToBinaryFile(net_param, filename_w);
}

}  // namespace mynet
