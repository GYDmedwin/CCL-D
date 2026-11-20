#ifndef NCCL_CCLD_ALLOC_H_
#define NCCL_CCLD_ALLOC_H_

#include <stddef.h>
#include <string.h>

#include <hip/hip_runtime.h>

// CCL-D deliberately uses the explicit mapped-host path first. This keeps the
// host pointer/device alias pair visible and avoids assuming that RCCL's
// fine-grained managed allocation has the same lifetime semantics.
struct CcldMappedAllocation {
  void* host = nullptr;
  void* device = nullptr;
  size_t bytes = 0;

  void reset() {
    host = nullptr;
    device = nullptr;
    bytes = 0;
  }

  bool empty() const { return host == nullptr && device == nullptr && bytes == 0; }
};

inline hipError_t ccldMappedHostAlloc(CcldMappedAllocation* allocation, size_t bytes) {
  if (allocation == nullptr) return hipErrorInvalidValue;
  allocation->reset();
  if (bytes == 0) return hipSuccess;

  hipError_t error = hipHostMalloc(&allocation->host, bytes, hipHostMallocMapped);
  if (error != hipSuccess) {
    allocation->reset();
    return error;
  }
  error = hipHostGetDevicePointer(&allocation->device, allocation->host, 0);
  if (error != hipSuccess) {
    hipHostFree(allocation->host);
    allocation->reset();
    return error;
  }
  allocation->bytes = bytes;
  memset(allocation->host, 0, bytes);
  return hipSuccess;
}

inline hipError_t ccldMappedHostFree(CcldMappedAllocation* allocation) {
  if (allocation == nullptr) return hipErrorInvalidValue;
  if (allocation->host == nullptr && allocation->device == nullptr && allocation->bytes == 0) {
    return hipSuccess;
  }
  if (allocation->host == nullptr) {
    allocation->reset();
    return hipErrorInvalidValue;
  }
  hipError_t error = hipHostFree(allocation->host);
  if (error == hipSuccess) allocation->reset();
  return error;
}

#endif
