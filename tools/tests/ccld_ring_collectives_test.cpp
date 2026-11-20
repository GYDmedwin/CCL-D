#include <hip/hip_runtime.h>
#include <rccl.h>

#include <cstdio>
#include <vector>

#define HIP_CHECK(call) do { \
  hipError_t e = (call); \
  if (e != hipSuccess) { fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); return 2; } \
} while (0)
#define RCCL_CHECK(call) do { \
  ncclResult_t e = (call); \
  if (e != ncclSuccess) { fprintf(stderr, "RCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); return 3; } \
} while (0)

int main() {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) return 4;
  const int ranks = deviceCount < 4 ? deviceCount : 4;
  const size_t count = 4096;
  const size_t gathered = count * static_cast<size_t>(ranks);
  std::vector<ncclComm_t> comms(ranks);
  std::vector<hipStream_t> streams(ranks);
  std::vector<float*> agIn(ranks), agOut(ranks), rsIn(ranks), rsOut(ranks);
  RCCL_CHECK(ncclCommInitAll(comms.data(), ranks, nullptr));

  for (int rank = 0; rank < ranks; ++rank) {
    HIP_CHECK(hipSetDevice(rank));
    HIP_CHECK(hipStreamCreate(&streams[rank]));
    HIP_CHECK(hipMalloc(&agIn[rank], count * sizeof(float)));
    HIP_CHECK(hipMalloc(&agOut[rank], gathered * sizeof(float)));
    HIP_CHECK(hipMalloc(&rsIn[rank], gathered * sizeof(float)));
    HIP_CHECK(hipMalloc(&rsOut[rank], count * sizeof(float)));
    std::vector<float> agHost(count, static_cast<float>(rank + 1));
    std::vector<float> rsHost(gathered, static_cast<float>(rank + 1));
    HIP_CHECK(hipMemcpy(agIn[rank], agHost.data(), count * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(rsIn[rank], rsHost.data(), gathered * sizeof(float), hipMemcpyHostToDevice));
  }

  RCCL_CHECK(ncclGroupStart());
  for (int rank = 0; rank < ranks; ++rank) {
    HIP_CHECK(hipSetDevice(rank));
    RCCL_CHECK(ncclAllGather(agIn[rank], agOut[rank], count, ncclFloat, comms[rank], streams[rank]));
  }
  RCCL_CHECK(ncclGroupEnd());

  RCCL_CHECK(ncclGroupStart());
  for (int rank = 0; rank < ranks; ++rank) {
    HIP_CHECK(hipSetDevice(rank));
    RCCL_CHECK(ncclReduceScatter(rsIn[rank], rsOut[rank], count, ncclFloat, ncclSum,
                                 comms[rank], streams[rank]));
  }
  RCCL_CHECK(ncclGroupEnd());

  const float expectedRs = static_cast<float>(ranks * (ranks + 1) / 2);
  std::vector<float> host(gathered);
  for (int rank = 0; rank < ranks; ++rank) {
    HIP_CHECK(hipSetDevice(rank));
    HIP_CHECK(hipStreamSynchronize(streams[rank]));
    HIP_CHECK(hipMemcpy(host.data(), agOut[rank], gathered * sizeof(float), hipMemcpyDeviceToHost));
    for (int src = 0; src < ranks; ++src)
      for (size_t i = 0; i < count; ++i)
        if (host[static_cast<size_t>(src) * count + i] != static_cast<float>(src + 1)) return 5;
    HIP_CHECK(hipMemcpy(host.data(), rsOut[rank], count * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) if (host[i] != expectedRs) return 6;
  }
  printf("Ring AllGather and ReduceScatter passed: ranks=%d count=%zu\n", ranks, count);

  for (int rank = 0; rank < ranks; ++rank) {
    HIP_CHECK(hipSetDevice(rank));
    HIP_CHECK(hipFree(agIn[rank])); HIP_CHECK(hipFree(agOut[rank]));
    HIP_CHECK(hipFree(rsIn[rank])); HIP_CHECK(hipFree(rsOut[rank]));
    HIP_CHECK(hipStreamDestroy(streams[rank]));
    RCCL_CHECK(ncclCommDestroy(comms[rank]));
  }
  return 0;
}
