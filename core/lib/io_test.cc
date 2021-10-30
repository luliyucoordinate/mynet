// Copyright 2021 coordinate
// Author: coordinate

#include "io.hpp"

#include <gtest/gtest.h>

#include "core/framework/mynet_test_main.hpp"

namespace mynet {

class IOTest : public ::testing::Test {};

TEST_F(IOTest, ReadWriteFlatFromTextFile) {
  TensorFlatT* tensor_flat = nullptr;
  const char* filename_r = "core/test_data/io_read_test.json";
  ReadFlatFromTextFile(filename_r, &tensor_flat);
  const char* filename_w = "core/test_data/io_write_test.json";
  WriteFlatToTextFile(tensor_flat, filename_w);
}

TEST_F(IOTest, ReadWriteFlatFromBinaryFile) {
  TensorFlatT* tensor_flat = nullptr;
  const char* filename_r = "core/test_data/io_read_test.bin";
  ReadFlatFromBinaryFile(filename_r, &tensor_flat);
  const char* filename_w = "core/test_data/io_write_test.bin";
  WriteFlatToBinaryFile(tensor_flat, filename_w);
}

}  // namespace mynet
