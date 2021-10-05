#include <vector>

#include "common.hpp"
#include "syncedmem.hpp"

namespace mynet {

class SyncedMemoryTest : public ::testing::Test {};

TEST_F(SyncedMemoryTest, TestInitialization) {
  SyncedMemory mem(10);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), 10);
  SyncedMemory* p_mem = new SyncedMemory(10 * sizeof(float));
  EXPECT_EQ(p_mem->size(), 10 * sizeof(float));
  delete p_mem;
}

TEST_F(SyncedMemoryTest, TestAllocationCPU) {
  SyncedMemory mem(10);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
}

TEST_F(SyncedMemoryTest, TestCPUWrite) {
  SyncedMemory mem(10);
  void* cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  std::memset(cpu_data, 1, mem.size());
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(cpu_data))[i], 1);
  }
  // do another round
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  std::memset(cpu_data, 2, mem.size());
  for (uint32_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ((static_cast<char*>(cpu_data))[i], 2);
  }
}

}  // namespace mynet
