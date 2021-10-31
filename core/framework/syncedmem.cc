// Copyright 2021 coordinate
// Author: coordinate

#include "syncedmem.hpp"

#include "common.hpp"

namespace mynet {
SyncedMemory::SyncedMemory()
    : cpu_ptr_(nullptr), size_(0), head_(UNINITIALIZED), own_cpu_data_(false) {}

SyncedMemory::SyncedMemory(uint32_t size)
    : cpu_ptr_(nullptr),
      size_(size),
      head_(UNINITIALIZED),
      own_cpu_data_(false) {}

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_) {
    MynetFreeHost(cpu_ptr_);
  }
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
    case UNINITIALIZED:
      MynetMallocHost(&cpu_ptr_, size_);
      std::memset(cpu_ptr_, 0, size_);
      head_ = HEAD_AT_CPU;
      own_cpu_data_ = true;
      break;
    case HEAD_AT_GPU:
    case HEAD_AT_CPU:
    case SYNCED:
      break;
  }
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return const_cast<const void*>(cpu_ptr_);
}

void SyncedMemory::set_cpu_data(void* data) {
  DCHECK(data);
  if (own_cpu_data_) {
    MynetFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

}  // namespace mynet
