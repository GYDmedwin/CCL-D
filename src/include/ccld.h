#ifndef NCCL_CCLD_H_
#define NCCL_CCLD_H_

#include <stddef.h>
#include <stdint.h>

// Keep the frame ABI aligned with the current NCCL CCL-D implementation.
// Its variable tail stores only the communicator's active channels.
#define CCLD_BLOCKS 8
#define CCLD_MAX_BLOCKS 512
#define CCLD_MAX_CHANNELS 32
#define CCLD_GROUPS 2

struct CcldChannel {
  uint64_t send, recv;
};

struct CcldWaitChannel {
  uint64_t data, credit;
};

static inline void ccldWaitTotals(const CcldWaitChannel* waits, unsigned groups,
                                  uint64_t* data, uint64_t* credit) {
  uint64_t dataTotal = 0, creditTotal = 0;
  for (unsigned group = 0; group < groups; ++group) {
    dataTotal += *reinterpret_cast<const volatile uint64_t*>(&waits[group].data);
    creditTotal += *reinterpret_cast<const volatile uint64_t*>(&waits[group].credit);
  }
  *data = dataTotal;
  *credit = creditTotal;
}

struct CcldBlock {
  uint64_t commHash, opCount;
  struct CcldChannel channel[1];
};

struct CcldFrame {
  uint64_t kernelIndex, mode, counter, detectChannel;
  unsigned char blocks[1];
};

static_assert(offsetof(CcldFrame, blocks) == 32, "frame header must be 32 bytes");
static_assert(offsetof(CcldBlock, channel) == 16, "dynamic block header must be 16 bytes");

static inline size_t ccldBlockBytes(unsigned channels, unsigned groups) {
  return offsetof(CcldBlock, channel) + size_t(channels) * groups * sizeof(CcldChannel);
}

static inline size_t ccldFrameBytes(unsigned channels, unsigned groups, unsigned blocks) {
  return offsetof(CcldFrame, blocks) + blocks * ccldBlockBytes(channels, groups);
}

static inline CcldBlock* ccldFrameBlock(CcldFrame* frame, unsigned index,
                                        unsigned channels, unsigned groups) {
  return reinterpret_cast<CcldBlock*>(reinterpret_cast<unsigned char*>(frame) +
      offsetof(CcldFrame, blocks) + index * ccldBlockBytes(channels, groups));
}

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#else
typedef void* cudaStream_t;
#endif
struct ncclComm;
struct ncclInfo;
struct ncclKernelPlan;
void ccldInit(struct ncclComm* comm);
void ccldFree(struct ncclComm* comm);
void ccldDeviceConfig(struct ncclComm* comm, CcldFrame** frame, uint32_t* blockBytes,
                      uint32_t* groups, CcldWaitChannel** waitFrame,
                      uint32_t* waitBlockBytes);
uint64_t ccldEnter(struct ncclComm* comm, const struct ncclInfo* info);
void ccldDiscardGroupedTask(struct ncclComm* comm, uint64_t seq);
void ccldPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);
void ccldCallbackEnqueue(struct ncclComm* comm, struct ncclKernelPlan* plan,
                         cudaStream_t stream);

#endif
