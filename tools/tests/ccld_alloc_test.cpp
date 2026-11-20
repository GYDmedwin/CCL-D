#include <assert.h>

#include "ccld_alloc.h"

int main() {
  CcldMappedAllocation allocation;
  assert(allocation.empty());
  assert(ccldMappedHostAlloc(&allocation, 0) == hipSuccess);
  assert(allocation.empty());
  assert(ccldMappedHostFree(&allocation) == hipSuccess);

  CcldMappedAllocation* nullAllocation = nullptr;
  assert(ccldMappedHostAlloc(nullAllocation, 0) == hipErrorInvalidValue);
  assert(ccldMappedHostFree(nullAllocation) == hipErrorInvalidValue);
  return 0;
}
