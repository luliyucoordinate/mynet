#ifndef MYNET_CC_SYNCEDMEM_HPP_
#define MYNET_CC_SYNCEDMEM_HPP_

#include <cstdlib>
#include "common.hpp"

namespace mynet {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void MynetMallocHost(void** ptr, uint32_t size) {
  *ptr = malloc(size);
  DCHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void MynetFreeHost(void* ptr) {
  free(ptr);
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
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
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, SYNCED };
  SyncedHead head() const { return head_; }
  uint32_t size() const { return size_; }

 private:

  void to_cpu();
  void* cpu_ptr_;
  uint32_t size_;
  SyncedHead head_;
  bool own_cpu_data_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace mynet

#endif  // MYNET_CC_SYNCEDMEM_HPP_
