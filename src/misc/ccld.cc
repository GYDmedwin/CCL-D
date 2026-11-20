#include "ccld.h"
#include "ccld_alloc.h"
#include "ccld_pending.h"
#include "comm.h"
#include "collectives.h"
#include "debug.h"
#include "param.h"
#include "utils.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <strings.h>

namespace {

static uint64_t ccldEnvNumber(const char* name, uint64_t fallback,
                              uint64_t minimum, uint64_t maximum) {
  const char* value = ncclGetEnv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const unsigned long long parsed = strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed < minimum || parsed > maximum) return fallback;
  return static_cast<uint64_t>(parsed);
}

struct CcldState {
  CcldMappedAllocation frameAllocation;
  CcldMappedAllocation waitAllocation;
  CcldFrame* frame = nullptr;
  CcldWaitChannel* waitFrame = nullptr;
  unsigned frameChannels = 0;
  unsigned frameGroups = 0;
  unsigned frameBlocks = CCLD_BLOCKS;
  size_t frameBytes = 0;
  size_t waitBlockBytes = 0;
  size_t waitBytes = 0;
  uint64_t mode = 0;
  CcldSequence sequence;
  CcldPendingTable<256> pending;
};

static CcldState* ccldState(struct ncclComm* comm) {
  return comm == nullptr ? nullptr : static_cast<CcldState*>(comm->ccldState);
}

static bool ccldEnabled() {
  return ccldEnvNumber("NCCL_CCLD_ENABLE", 0, 0, 1) == 1;
}

} // namespace

void ccldInit(struct ncclComm* comm) {
  if (comm == nullptr || !ccldEnabled()) return;
  const uint64_t mode = ccldEnvNumber("NCCL_CCLD_MODE", 1, 0, 2);
  if (mode != 1) {
    WARN("CCL-D RCCL prototype supports mode 1 only; telemetry disabled");
    return;
  }
  if (comm->nChannels < 1 || comm->nChannels > CCLD_MAX_CHANNELS) {
    WARN("CCL-D channel count %d is outside the supported frame range", comm->nChannels);
    return;
  }

  CcldState* state = new (std::nothrow) CcldState();
  if (state == nullptr) {
    WARN("CCL-D state allocation failed");
    return;
  }
  state->mode = mode;
  state->frameChannels = static_cast<unsigned>(comm->nChannels);
  const char* algorithm = ncclGetEnv("NCCL_ALGO");
  state->frameGroups = algorithm != nullptr && strcasecmp(algorithm, "Ring") == 0 ? 1 : CCLD_GROUPS;
  state->frameBlocks = static_cast<unsigned>(ccldEnvNumber(
      "NCCL_CCLD_FRAME_BLOCKS", CCLD_BLOCKS, CCLD_BLOCKS, CCLD_MAX_BLOCKS));
  state->frameBytes = ccldFrameBytes(state->frameChannels, state->frameGroups, state->frameBlocks);

  hipError_t error = ccldMappedHostAlloc(&state->frameAllocation, state->frameBytes);
  if (error != hipSuccess) {
    WARN("CCL-D mapped frame allocation failed: %s", hipGetErrorString(error));
    delete state;
    return;
  }
  state->frame = static_cast<CcldFrame*>(state->frameAllocation.host);
  state->frame->mode = mode;
  state->frame->detectChannel = state->frameChannels;

  const bool waitRequested = ccldEnvNumber("NCCL_CCLD_WAIT_ENABLE", 0, 0, 1) == 1;
  if (waitRequested && state->frameGroups == 1) {
    state->waitBlockBytes = size_t(state->frameChannels) * sizeof(CcldWaitChannel);
    state->waitBytes = state->frameBlocks * state->waitBlockBytes;
    error = ccldMappedHostAlloc(&state->waitAllocation, state->waitBytes);
    if (error != hipSuccess) {
      WARN("CCL-D wait frame allocation failed: %s", hipGetErrorString(error));
      ccldMappedHostFree(&state->frameAllocation);
      delete state;
      return;
    }
    state->waitFrame = static_cast<CcldWaitChannel*>(state->waitAllocation.host);
  }
  comm->ccldState = state;
}

void ccldFree(struct ncclComm* comm) {
  CcldState* state = ccldState(comm);
  if (state == nullptr) return;
  comm->ccldState = nullptr;
  ccldMappedHostFree(&state->waitAllocation);
  ccldMappedHostFree(&state->frameAllocation);
  delete state;
}

void ccldDeviceConfig(struct ncclComm* comm, CcldFrame** frame, uint32_t* blockBytes,
                      uint32_t* groups, CcldWaitChannel** waitFrame,
                      uint32_t* waitBlockBytes) {
  CcldState* state = ccldState(comm);
  if (frame == nullptr || blockBytes == nullptr || groups == nullptr ||
      waitFrame == nullptr || waitBlockBytes == nullptr) return;
  *frame = state == nullptr ? nullptr : static_cast<CcldFrame*>(state->frameAllocation.device);
  *blockBytes = state == nullptr ? 0 : static_cast<uint32_t>(
      ccldBlockBytes(state->frameChannels, state->frameGroups));
  *groups = state == nullptr ? 0 : state->frameGroups;
  *waitFrame = state == nullptr ? nullptr : static_cast<CcldWaitChannel*>(state->waitAllocation.device);
  *waitBlockBytes = state == nullptr ? 0 : static_cast<uint32_t>(state->waitBlockBytes);
}

uint64_t ccldEnter(struct ncclComm* comm, const struct ncclInfo* info) {
  CcldState* state = ccldState(comm);
  if (state == nullptr || info == nullptr) return 0;
  bool valid = false;
  const uint64_t seq = state->sequence.next(&valid);
  if (!valid) return seq;
  uint64_t bytes = static_cast<uint64_t>(info->count) * static_cast<uint64_t>(ncclTypeSize(info->datatype));
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncReduceScatter) {
    bytes *= static_cast<uint64_t>(comm->nRanks);
  }
  CcldStart start{clockNano(), static_cast<uint64_t>(info->count), bytes,
                  ncclFuncToString(info->coll), ncclDatatypeToString(info->datatype)};
  if (!state->pending.publish(seq, start)) return seq;
  return seq;
}

void ccldPlan(struct ncclComm* /*comm*/, struct ncclKernelPlan* /*plan*/) {}

void ccldCallbackEnqueue(struct ncclComm* /*comm*/, struct ncclKernelPlan* /*plan*/,
                         cudaStream_t /*stream*/) {}
