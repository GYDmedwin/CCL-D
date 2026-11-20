#include <assert.h>

#include "ccld_alloc.h"

__global__ static void writeMappedValue(uint64_t* value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *value = 0xCC1D936ull;
}

int main() {
  CcldMappedAllocation allocation;
  assert(allocation.empty());
  assert(ccldMappedHostAlloc(&allocation, 0) == hipSuccess);
  assert(allocation.empty());
  assert(ccldMappedHostFree(&allocation) == hipSuccess);

  int deviceCount = 0;
  assert(hipGetDeviceCount(&deviceCount) == hipSuccess && deviceCount > 0);
  assert(ccldMappedHostAlloc(&allocation, sizeof(uint64_t)) == hipSuccess);
  assert(allocation.host != nullptr && allocation.device != nullptr);
  assert(allocation.bytes == sizeof(uint64_t));
  assert(*static_cast<uint64_t*>(allocation.host) == 0);
  writeMappedValue<<<1, 1>>>(static_cast<uint64_t*>(allocation.device));
  assert(hipDeviceSynchronize() == hipSuccess);
  assert(*static_cast<uint64_t*>(allocation.host) == 0xCC1D936ull);
  assert(ccldMappedHostFree(&allocation) == hipSuccess);
  assert(allocation.empty());

  CcldMappedAllocation* nullAllocation = nullptr;
  assert(ccldMappedHostAlloc(nullAllocation, 0) == hipErrorInvalidValue);
  assert(ccldMappedHostFree(nullAllocation) == hipErrorInvalidValue);
  return 0;
}
