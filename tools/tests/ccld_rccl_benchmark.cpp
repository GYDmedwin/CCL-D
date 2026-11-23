#include <hip/hip_runtime.h>
#include <rccl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <time.h>

#define HIP(call) do { hipError_t e = (call); if (e != hipSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (false)
#define RCCL(call) do { ncclResult_t e = (call); if (e != ncclSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, ncclGetErrorString(e)); return 3; } } while (false)

double cpuTime() {
  timespec stamp{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stamp);
  return stamp.tv_sec + stamp.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s BYTES_PER_RANK ar|ag|rs|a2a\n", argv[0]);
    return 1;
  }
  const std::string op = argv[2];
  if (op != "ar" && op != "ag" && op != "rs" && op != "a2a") return 1;
  const size_t bytes = std::strtoull(argv[1], nullptr, 10);
  if (bytes < 65536 || bytes > (256u << 20) || bytes % 16) return 1;
  const size_t count = bytes / sizeof(float);
  constexpr int ranks = 4, warmup = 20, repeats = 60;
  using Clock = std::chrono::steady_clock;
  int devices = 0;
  HIP(hipGetDeviceCount(&devices));
  if (devices != ranks) return 4;
  const size_t segment = count / ranks;
  const size_t sendCount = op == "ag" ? segment : count;
  const size_t recvCount = op == "rs" ? segment : count;
  std::vector<ncclComm_t> comms(ranks);
  std::vector<hipStream_t> streams(ranks);
  std::vector<float*> inputs(ranks), outputs(ranks);
  std::vector<float> host(count, 1.0f);
  RCCL(ncclCommInitAll(comms.data(), ranks, nullptr));
  for (int r = 0; r < ranks; ++r) {
    HIP(hipSetDevice(r));
    HIP(hipStreamCreate(&streams[r]));
    HIP(hipMalloc(&inputs[r], bytes));
    HIP(hipMalloc(&outputs[r], bytes));
    for (size_t i = 0; i < sendCount; ++i) {
      const int dst = static_cast<int>(i / segment);
      host[i] = op == "a2a" ? 10 * r + dst + 1 : r + 1;
    }
    HIP(hipMemcpy(inputs[r], host.data(), sendCount * sizeof(float), hipMemcpyHostToDevice));
  }
  std::vector<double> durations;
  durations.reserve(repeats);
  double cpuStart = 0;
  for (int i = 0; i < repeats + warmup; ++i) {
    if (i == warmup) cpuStart = cpuTime();
    const auto start = Clock::now();
    RCCL(ncclGroupStart());
    for (int r = 0; r < ranks; ++r) {
      HIP(hipSetDevice(r));
      if (op == "ar") {
        RCCL(ncclAllReduce(inputs[r], outputs[r], count, ncclFloat, ncclSum, comms[r], streams[r]));
      } else if (op == "ag") {
        RCCL(ncclAllGather(inputs[r], outputs[r], segment, ncclFloat, comms[r], streams[r]));
      } else if (op == "rs") {
        RCCL(ncclReduceScatter(inputs[r], outputs[r], segment, ncclFloat, ncclSum, comms[r], streams[r]));
      } else {
        RCCL(ncclAllToAll(inputs[r], outputs[r], segment, ncclFloat, comms[r], streams[r]));
      }
    }
    RCCL(ncclGroupEnd());
    for (int r = 0; r < ranks; ++r) {
      HIP(hipSetDevice(r));
      HIP(hipStreamSynchronize(streams[r]));
    }
    if (i >= warmup) durations.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  }
  const double cpuS = cpuTime() - cpuStart;
  rusage memory{};
  getrusage(RUSAGE_SELF, &memory);
  for (int r = 0; r < ranks; ++r) {
    HIP(hipSetDevice(r));
    HIP(hipMemcpy(host.data(), outputs[r], recvCount * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < recvCount; ++i) {
      const int src = i / segment;
      int expected = ranks * (ranks + 1) / 2;
      if (op == "ag") expected = src + 1;
      if (op == "a2a") expected = 10 * src + r + 1;
      if (host[i] != expected) {
        std::fprintf(stderr, "%s rank %d element %zu expected %d got %f\n",
                     op.c_str(), r, i, expected, host[i]);
        return 5;
      }
    }
    HIP(hipFree(inputs[r]));
    HIP(hipFree(outputs[r]));
    HIP(hipStreamDestroy(streams[r]));
    RCCL(ncclCommDestroy(comms[r]));
  }
  std::sort(durations.begin(), durations.end());
  const double avg = std::accumulate(durations.begin(), durations.end(), 0.) / repeats;
  std::printf("RESULT {\"bytes\": %zu, \"mean_ms\": %.8f, \"p50_ms\": %.8f, \"p95_ms\": %.8f, \"cpu_s\": %.8f, \"max_rss_kib\": %ld, \"iterations\": %d}\n",
      bytes, avg, (durations[repeats/2 - 1] + durations[repeats/2])/2,
      durations[(repeats * 95 + 99)/100 - 1], cpuS, memory.ru_maxrss, repeats);
}
