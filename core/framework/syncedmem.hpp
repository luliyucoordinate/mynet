// Copyright 2021 coordinate
// Author: coordinate

#ifndef CORE_FRAMEWORK_SYNCEDMEM_HPP_
#define CORE_FRAMEWORK_SYNCEDMEM_HPP_

#include <cstdlib>

#include "common.hpp"

namespace mynet {

inline void MynetMallocHost(void** ptr, uint32_t size) {
  *ptr = malloc(size);
  DCHECK(*ptr) << "host allocation of size" << size << "failed";
}

inline void MynetFreeHost(void* ptr) { free(ptr); }

class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(uint32_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() const { return head_; }
  uint32_t size() const { return size_; }

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  SyncedHead head_;
  uint32_t size_;
  bool own_cpu_data_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};
}  // namespace mynet

#endif  // CORE_FRAMEWORK_SYNCEDMEM_HPP_
