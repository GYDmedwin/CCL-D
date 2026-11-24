#ifndef NCCL_CCLD_H_
#define NCCL_CCLD_H_

#include <stdint.h>
#include <stddef.h>

// The variable tail contains only the communicator's actual channels. A
// second subgroup is retained whenever Tree can be selected.
#define CCLD_BLOCKS 8
#define CCLD_MAX_BLOCKS 512
#define CCLD_MAX_CHANNELS 32
#define CCLD_GROUPS 2
struct CcldChannel {
  uint64_t send, recv;
};
// Optional Ring-only wait counter slots, separately allocated when enabled.
struct CcldWaitChannel {
  uint64_t data, credit;
};
constexpr unsigned CCLD_TREE_PEERS = 4;
struct CcldTreeWaitChannel {
  // Parent first, then the three child slots in device Tree order.
  CcldWaitChannel peers[CCLD_TREE_PEERS];
};
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
static inline CcldBlock* ccldFrameBlock(CcldFrame* frame, unsigned index, unsigned channels, unsigned groups) {
  return reinterpret_cast<CcldBlock*>(reinterpret_cast<unsigned char*>(frame) +
      offsetof(CcldFrame, blocks) + index * ccldBlockBytes(channels, groups));
}

#ifndef __CUDACC__
#include <cuda_runtime.h>
struct ncclComm;
struct ncclInfo;
struct ncclTaskColl;
struct ncclDevWorkColl;
struct ncclKernelPlan;
void ccldInit(struct ncclComm* comm);
void ccldFree(struct ncclComm* comm);
void ccldDeviceConfig(struct ncclComm* comm, CcldFrame** frame, uint32_t* blockBytes, uint32_t* groups,
                      CcldWaitChannel** waitFrame, uint32_t* waitBlockBytes);
uint64_t ccldEnter(struct ncclComm* comm, const struct ncclInfo* info);
void ccldPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);
void ccldCallbackEnqueue(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
#endif
#endif
