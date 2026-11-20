#include <hip/hip_runtime.h>
#include <rccl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call) do { \
  hipError_t e = (call); \
  if (e != hipSuccess) { \
    fprintf(stderr, "%s:%d HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    return 2; \
  } \
} while (0)

#define RCCL_CHECK(call) do { \
  ncclResult_t e = (call); \
  if (e != ncclSuccess) { \
    fprintf(stderr, "%s:%d RCCL error: %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); \
    return 3; \
  } \
} while (0)

int main() {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    fprintf(stderr, "CCL-D Simple Ring AllReduce test requires at least two gfx936 devices\n");
    return 4;
  }
  const int ranks = deviceCount < 4 ? deviceCount : 4;
  const size_t count = 1 << 16;
  std::vector<ncclComm_t> comms(ranks);
  std::vector<hipStream_t> streams(ranks);
  std::vector<float*> input(ranks), output(ranks);
  std::vector<float> hostInput(count, 1.0f), hostOutput(count);

  RCCL_CHECK(ncclCommInitAll(comms.data(), ranks, nullptr));
  for (int rank = 0; rank < ranks; ++rank) {
    HIP_CHECK(hipSetDevice(rank));
    HIP_CHECK(hipStreamCreate(&streams[rank]));
    HIP_CHECK(hipMalloc(&input[rank], count * sizeof(float)));
    HIP_CHECK(hipMalloc(&output[rank], count * sizeof(float)));
    HIP_CHECK(hipMemcpy(input[rank], hostInput.data(), count * sizeof(float), hipMemcpyHostToDevice));
  }

  for (int iteration = 0; iteration < 3; ++iteration) {
    RCCL_CHECK(ncclGroupStart());
    for (int rank = 0; rank < ranks; ++rank) {
      HIP_CHECK(hipSetDevice(rank));
      RCCL_CHECK(ncclAllReduce(input[rank], output[rank], count, ncclFloat, ncclSum,
                               comms[rank], streams[rank]));
    }
    RCCL_CHECK(ncclGroupEnd());
  }

  for (int rank = 0; rank < ranks; ++rank) {
    HIP_CHECK(hipSetDevice(rank));
    HIP_CHECK(hipStreamSynchronize(streams[rank]));
    HIP_CHECK(hipMemcpy(hostOutput.data(), output[rank], count * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
      if (hostOutput[i] != static_cast<float>(ranks)) {
        fprintf(stderr, "rank %d result[%zu]=%g, expected %d\n", rank, i, hostOutput[i], ranks);
        return 5;
      }
    }
  }
  printf("Simple Ring AllReduce passed: ranks=%d count=%zu iterations=3\n", ranks, count);

  for (int rank = 0; rank < ranks; ++rank) {
    HIP_CHECK(hipSetDevice(rank));
    HIP_CHECK(hipFree(input[rank]));
    HIP_CHECK(hipFree(output[rank]));
    HIP_CHECK(hipStreamDestroy(streams[rank]));
    RCCL_CHECK(ncclCommDestroy(comms[rank]));
  }
  return 0;
}
