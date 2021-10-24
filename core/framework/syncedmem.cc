// Copyright 2021 coordinate
// Author: coordinate

#include "syncedmem.hpp"

namespace mynet {

SyncedMemory::SyncedMemory()
    : cpu_ptr(nullptr),
      size_(0ul),
      head_(UNINITIALIZED),
      own_cpu_data_(false) {}

SyncedMemory::SyncedMemory(uint32_t size)
    : cpu_ptr(nullptr),
      size_(size),
      head_(UNINITIALIZED),
      own_cpu_data_(false) {}

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_) {
    MynetFreeHost(cpu_ptr_);
  }
}

}  // namespace mynet
