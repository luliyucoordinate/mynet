// Copyright 2021 coordinate
// Author: coordinate

#include "syncedmem.hpp"

#include <memory>

#include "common.hpp"

namespace mynet {
class SyncedMemoryTest : public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization) {
  SyncedMemory mem(10ul);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 10ul);
  auto p_mem = std::make_shared<SyncedMemory>(10ul * sizeof(float));
  EXPECT_EQ(p_mem->size(), 10ul * sizeof(float));
}

TEST_F(SyncedMemoryTest, TestAllocationCPU) {
  SyncedMemory mem(10ul);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
}

TEST_F(SyncedMemoryTest, TestCPUWrite) {
  SyncedMemory mem(10ul);
  auto cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  std::memset(cpu_data, 1, mem.size());
  for (uint32_t i = 0; i < mem.size(); i++) {
    EXPECT_EQ(static_cast<char*>(cpu_data)[i], 1);
  }

  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  std::memset(cpu_data, 2, mem.size());
  for (uint32_t i = 0; i < mem.size(); i++) {
    EXPECT_EQ(static_cast<char*>(cpu_data)[i], 2);
  }
}

}  // namespace mynet
