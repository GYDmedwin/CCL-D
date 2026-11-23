#include "ccld.h"
#include "ccld_alloc.h"
#include "ccld_log.h"
#include "ccld_pending.h"
#include "ccld_plan.h"
#include "ccld_queue.h"
#include "ccld_stream.h"
#include "comm.h"
#include "collectives.h"
#include "debug.h"
#include "param.h"
#include "utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <mutex>
#include <new>
#include <thread>
#include <strings.h>
#include <unistd.h>

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
  struct ChannelSample {
    int parent, ringPrev, ringNext;
    uint64_t waitData, waitCredit;
    uint64_t lastSend, lastRecv, dumpSend, dumpRecv, changesSend, changesRecv;
  };
  struct Slot {
    std::atomic<int> phase{0};
    std::atomic<bool> done{false};
    std::atomic<bool> eventRecorded{false};
    std::atomic<uint64_t> endNs{0};
    hipEvent_t completionEvent = nullptr;
    std::atomic<bool> eventFailed{false};
    uint64_t seq = 0;
    uint64_t startNs = 0, lastDumpNs = 0, firstProgressNs = 0;
    uint64_t count = 0, bytes = 0;
    uint32_t builtinReduction = 0;
    const char* op = nullptr;
    const char* datatype = nullptr;
    const char* algorithm = nullptr;
    const char* protocol = nullptr;
    int channelLo = 0, channelHi = -1;
    ChannelSample* channels = nullptr;
  };
  std::unique_ptr<Slot[]> slots;
  std::unique_ptr<ChannelSample[]> samples;
  std::thread sampler;
  std::mutex mutex;
  std::condition_variable cv;
  std::mutex outputMutex;
  std::atomic<bool> stop{false};
  std::atomic<bool> writerStop{false};
  std::atomic<bool> ioFailed{false};
  std::atomic<uint64_t> logDrops{0};
  std::atomic<uint64_t> lastDroppedSeq{0};
  std::atomic<int> callbacksOutstanding{0};
  std::unique_ptr<CcldQueue<CcldLogEvent>> logQueue;
  size_t queueSlots = 0;
  std::thread writer;
  CcldNodeStream nodeStream;
  FILE* output = nullptr;
  uint64_t commHash = 0;
  uint64_t pollUs = 200;
  uint64_t dumpNs = 1000000000ull;
  int rank = 0, nranks = 0, device = -1;
  bool useEvent = true;
  char host[64] = {};
#ifdef CCLD_FAULT_INJECTION
  uint64_t faultKind = 0, faultRank = UINT64_MAX, faultOp = 0, faultCycles = 0, faultRepeats = 1;
#endif
};

struct CcldCallback { CcldState* state; unsigned slot; };

static CcldState* ccldState(struct ncclComm* comm) {
  return comm == nullptr ? nullptr : static_cast<CcldState*>(comm->ccldState);
}

static bool ccldEnabled() {
  return ccldEnvNumber("NCCL_CCLD_ENABLE", 0, 0, 1) == 1;
}

static uint64_t ccldNow() { return clockNano(); }

static CcldLogEvent ccldLogBase(CcldState* state, CcldLogType type,
                                uint64_t seq, uint64_t now) {
  CcldLogEvent event{};
  event.magic = 0x444c4343u;
  event.version = 1;
  event.type = type;
  event.hash = state->commHash;
  event.seq = seq;
  event.nowNs = now;
  event.rank = state->rank;
  event.nranks = state->nranks;
  snprintf(event.host, sizeof(event.host), "%s", state->host);
  return event;
}

static void ccldWrite(CcldState* state, const CcldLogEvent* events, size_t count) {
  if (state->logQueue != nullptr && state->logQueue->pushBatch(events, count)) return;
  state->lastDroppedSeq.store(events[count - 1].seq, std::memory_order_relaxed);
  state->logDrops.fetch_add(count, std::memory_order_release);
}

static void ccldWrite(CcldState* state, const CcldLogEvent& event) {
  ccldWrite(state, &event, 1);
}

static void ccldWriter(CcldState* state) {
  CcldLogEvent batch[16];
  for (;;) {
    size_t count = 0;
    const uint64_t drops = state->logDrops.exchange(0, std::memory_order_acq_rel);
    if (drops != 0) {
      batch[count] = ccldLogBase(state, CcldLogDropped,
                                 state->lastDroppedSeq.load(std::memory_order_relaxed), ccldNow());
      batch[count].count = drops;
      snprintf(batch[count].reason, sizeof(batch[count].reason), "%s", "binary_queue_overflow");
      ++count;
    }
    while (count < 16 && state->logQueue->pop(batch[count])) ++count;
    if (count != 0) {
      bool ok = false;
      {
        std::lock_guard<std::mutex> lock(state->outputMutex);
        if (state->output != nullptr)
          ok = fwrite(batch, sizeof(CcldLogEvent), count, state->output) == count && fflush(state->output) == 0;
      }
      if (state->output == nullptr && !state->ioFailed.load(std::memory_order_relaxed))
        ok = state->nodeStream.write(batch, count * sizeof(CcldLogEvent));
      if (!ok) {
        state->ioFailed.store(true, std::memory_order_release);
        state->nodeStream.close();
      }
    } else if (state->writerStop.load(std::memory_order_acquire)) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  if (state->output == nullptr && !state->ioFailed.load(std::memory_order_acquire)) {
    CcldLogEvent end = ccldLogBase(state, CcldLogEnd, state->sequence.next(), ccldNow());
    state->nodeStream.write(&end, sizeof(end));
    state->nodeStream.close();
  }
}

static void ccldDropped(CcldState* state, uint64_t seq, const char* reason) {
  CcldLogEvent event = ccldLogBase(state, CcldLogDropped, seq, ccldNow());
  snprintf(event.reason, sizeof(event.reason), "%s", reason);
  ccldWrite(state, event);
}

static uint64_t ccldLoad(const uint64_t* value) {
  return *reinterpret_cast<const volatile uint64_t*>(value);
}

static void ccldSample(CcldState* state, unsigned index, bool done) {
  CcldState::Slot& slot = state->slots[index];
  CcldBlock* block = ccldFrameBlock(state->frame, index, state->frameChannels, state->frameGroups);
  CcldChannel* channels = reinterpret_cast<CcldChannel*>(
      reinterpret_cast<unsigned char*>(block) + offsetof(CcldBlock, channel));
  const CcldWaitChannel* waits = state->waitFrame == nullptr ? nullptr :
      reinterpret_cast<const CcldWaitChannel*>(
          reinterpret_cast<const unsigned char*>(state->waitFrame) + index * state->waitBlockBytes);
  uint64_t progress = 0;
  for (int channel = slot.channelLo; channel <= slot.channelHi; ++channel) {
    CcldState::ChannelSample& sample = slot.channels[channel];
    uint64_t send = 0, recv = 0;
    for (unsigned group = 0; group < state->frameGroups; ++group) {
      const CcldChannel& counts = channels[channel * state->frameGroups + group];
      send += ccldLoad(&counts.send);
      recv += ccldLoad(&counts.recv);
    }
    if (send != sample.lastSend) ++sample.changesSend;
    if (recv != sample.lastRecv) ++sample.changesRecv;
    sample.lastSend = send;
    sample.lastRecv = recv;
    progress |= send | recv;
    if (waits != nullptr) {
      ccldWaitTotals(waits + channel * state->frameGroups, state->frameGroups,
                     &sample.waitData, &sample.waitCredit);
    }
  }
  if (!done && slot.firstProgressNs == 0 && progress != 0)
    slot.firstProgressNs = ccldNow();
}

static size_t ccldEmitPart(CcldState* state, CcldState::Slot& slot, CcldLogType type,
                        uint64_t now, unsigned part, unsigned parts, CcldLogEvent* records) {
  CcldLogEvent* wait = state->waitFrame != nullptr ? records : nullptr;
  CcldLogEvent& event = records[wait != nullptr];
  event = ccldLogBase(state, type, slot.seq, now);
  if (parts > 1) {
    event.version = CCLD_FRAGMENT_VERSION;
    event.reserved = (parts << 16) | (part << 8) | type;
  }
  event.startNs = slot.startNs;
  event.windowNs = now > slot.lastDumpNs ? now - slot.lastDumpNs : 1;
  if (slot.firstProgressNs >= slot.startNs && slot.firstProgressNs < now)
    ccldLogFirstProgress(event, slot.firstProgressNs);
  ccldLogReduction(event, slot.builtinReduction);
  event.count = slot.count;
  event.bytes = slot.bytes;
  const unsigned offset = part * CCLD_RECORD_CHANNELS;
  const unsigned channels = slot.channelHi - slot.channelLo + 1;
  event.nChannels = std::min(CCLD_RECORD_CHANNELS, channels - offset);
  snprintf(event.op, sizeof(event.op), "%s", slot.op ? slot.op : "");
  snprintf(event.algorithm, sizeof(event.algorithm), "%s", slot.algorithm ? slot.algorithm : "");
  snprintf(event.protocol, sizeof(event.protocol), "%s", slot.protocol ? slot.protocol : "");
  snprintf(event.datatype, sizeof(event.datatype), "%s", slot.datatype ? slot.datatype : "");
  if (wait != nullptr) {
    *wait = ccldLogBase(state, CcldLogRingWait, slot.seq, now);
    wait->version = event.version;
    wait->reserved = parts > 1 ? event.reserved : type;
    wait->nChannels = event.nChannels;
    snprintf(event.reason, sizeof(event.reason), "%s", "ring_wait_v1");
  }
  for (unsigned i = 0; i < event.nChannels; ++i) {
    const unsigned channel = slot.channelLo + offset + i;
    auto& sample = slot.channels[channel];
    event.channels[i] = {static_cast<int32_t>(channel), sample.parent, sample.lastSend,
        sample.lastRecv, sample.changesSend, sample.changesRecv,
        sample.lastSend - sample.dumpSend, sample.lastRecv - sample.dumpRecv};
    sample.dumpSend = sample.lastSend;
    sample.dumpRecv = sample.lastRecv;
    sample.changesSend = sample.changesRecv = 0;
    if (wait != nullptr) {
      wait->channels[i] = {static_cast<int32_t>(channel), sample.ringPrev,
          sample.waitData, sample.waitCredit, static_cast<uint64_t>(sample.ringNext), 0, 0, 0};
    }
  }
  return wait != nullptr ? 2 : 1;
}

static void ccldEmitWide(CcldState* state, CcldState::Slot& slot, CcldLogType type,
                        uint64_t now, unsigned parts) {
  constexpr unsigned maxParts = (MAXCHANNELS + CCLD_RECORD_CHANNELS - 1) / CCLD_RECORD_CHANNELS;
  CcldLogEvent records[2 * maxParts];
  size_t count = 0;
  for (unsigned part = 0; part < parts; ++part)
    count += ccldEmitPart(state, slot, type, now, part, parts, records + count);
  ccldWrite(state, records, count);
}

static void ccldEmit(CcldState* state, unsigned index, bool complete, uint64_t now) {
  CcldState::Slot& slot = state->slots[index];
  const auto type = complete ? CcldLogComplete : CcldLogSnapshot;
  const unsigned channels = slot.channelHi - slot.channelLo + 1;
  const unsigned parts = (channels + CCLD_RECORD_CHANNELS - 1) / CCLD_RECORD_CHANNELS;
  if (parts == 1) {
    CcldLogEvent records[2];
    const size_t count = ccldEmitPart(state, slot, type, now, 0, 1, records);
    ccldWrite(state, records, count);
  } else {
    ccldEmitWide(state, slot, type, now, parts);
  }
  slot.lastDumpNs = now;
}

static void ccldPollSlot(CcldState* state, unsigned index, bool eventDeviceReady, uint64_t now,
                         bool finishing) {
  CcldState::Slot& slot = state->slots[index];
  if (slot.phase.load(std::memory_order_acquire) != 2) return;
  if (state->useEvent && slot.eventRecorded.load(std::memory_order_acquire) &&
      !slot.done.load(std::memory_order_relaxed) &&
      !slot.eventFailed.load(std::memory_order_relaxed)) {
    if (!eventDeviceReady) {
      slot.eventFailed.store(true, std::memory_order_release);
      ccldDropped(state, slot.seq, "completion_event_worker_device_setup_failed");
    } else {
      const hipError_t query = hipEventQuery(slot.completionEvent);
      if (query == hipSuccess) {
        slot.endNs.store(ccldNow(), std::memory_order_relaxed);
        slot.done.store(true, std::memory_order_release);
      } else if (query != hipErrorNotReady) {
        slot.eventFailed.store(true, std::memory_order_release);
        ccldDropped(state, slot.seq, "completion_event_query_failed");
      }
    }
  }
  const bool done = slot.done.load(std::memory_order_acquire);
  ccldSample(state, index, done);
  if (done) {
    ccldEmit(state, index, true, slot.endNs.load(std::memory_order_relaxed));
    slot.phase.store(0, std::memory_order_release);
  } else if (finishing || (now >= slot.lastDumpNs && now - slot.lastDumpNs >= state->dumpNs)) {
    ccldEmit(state, index, false, ccldNow());
  }
}

static void ccldSampler(CcldState* state) {
  const bool eventDeviceReady = !state->useEvent || hipSetDevice(state->device) == hipSuccess;
  if (!eventDeviceReady)
    WARN("CCL-D rank %d sampler could not select device %d", state->rank, state->device);
  std::unique_lock<std::mutex> lock(state->mutex);
  while (!state->stop.load(std::memory_order_acquire)) {
    state->cv.wait_for(lock, std::chrono::microseconds(state->pollUs));
    const uint64_t now = ccldNow();
    for (unsigned i = 0; i < state->frameBlocks; ++i)
      ccldPollSlot(state, i, eventDeviceReady, now, false);
  }
  const uint64_t now = ccldNow();
  for (unsigned i = 0; i < state->frameBlocks; ++i)
    ccldPollSlot(state, i, eventDeviceReady, now, true);
}

static void ccldDone(void* data) {
  CcldCallback* callback = static_cast<CcldCallback*>(data);
  CcldState* state = callback->state;
  CcldState::Slot& slot = state->slots[callback->slot];
  slot.endNs.store(ccldNow(), std::memory_order_relaxed);
  slot.done.store(true, std::memory_order_release);
  state->cv.notify_one();
  delete callback;
  state->callbacksOutstanding.fetch_sub(1, std::memory_order_release);
}

} // namespace

void ccldInit(struct ncclComm* comm) {
  if (comm == nullptr || !ccldEnabled()) return;
  const uint64_t mode = ccldEnvNumber("NCCL_CCLD_MODE", 1, 0, 2);
  if (mode != 1) {
    WARN("CCL-D RCCL prototype supports mode 1 only; telemetry disabled");
    return;
  }
  if (comm->nChannels < 1 || comm->nChannels > MAXCHANNELS) {
    WARN("CCL-D channel count %d is outside the supported frame range", comm->nChannels);
    return;
  }

  CcldState* state = new (std::nothrow) CcldState();
  if (state == nullptr) {
    WARN("CCL-D state allocation failed");
    return;
  }
  state->mode = mode;
  state->commHash = comm->commHash;
  state->rank = comm->rank;
  state->nranks = comm->nRanks;
  state->device = comm->cudaDev;
  gethostname(state->host, sizeof(state->host) - 1);
  state->pollUs = ccldEnvNumber("NCCL_CCLD_POLL_US", 200, 20, 1000000);
  state->dumpNs = ccldEnvNumber("NCCL_CCLD_DUMP_MS", 1000, 10, 60000) * 1000000ull;
  const char* completion = ncclGetEnv("NCCL_CCLD_COMPLETION");
  state->useEvent = completion == nullptr || strcmp(completion, "hostfunc") != 0;
#ifdef CCLD_FAULT_INJECTION
  const char* faultKind = ncclGetEnv("NCCL_CCLD_FAULT_KIND");
  if (faultKind != nullptr && strcmp(faultKind, "H3") == 0) state->faultKind = 1;
  else if (faultKind != nullptr && strcmp(faultKind, "S2") == 0) state->faultKind = 2;
  state->faultRank = ccldEnvNumber("NCCL_CCLD_FAULT_RANK", UINT64_MAX, 0,
                                   static_cast<uint64_t>(comm->nRanks - 1));
  state->faultOp = ccldEnvNumber("NCCL_CCLD_FAULT_OP", 0, 1, UINT64_MAX);
  state->faultRepeats = ccldEnvNumber("NCCL_CCLD_FAULT_REPEATS", 1, 1, 100);
  state->faultCycles = ccldEnvNumber("NCCL_CCLD_FAULT_CYCLES", 0, 1, 30000000000ull);
  if (state->faultKind != 0 && (state->faultRank == UINT64_MAX || state->faultOp == 0 || state->faultCycles == 0)) {
    WARN("CCL-D fault simulation disabled: rank, operation and bounded cycles are required");
    state->faultKind = 0;
  }
#else
  if (ncclGetEnv("NCCL_CCLD_FAULT_KIND") != nullptr)
    WARN("CCL-D fault injection is not compiled into this build");
#endif
  state->frameChannels = static_cast<unsigned>(comm->nChannels);
  const char* algorithm = ncclGetEnv("NCCL_ALGO");
  state->frameGroups = algorithm != nullptr && strcasecmp(algorithm, "Ring") == 0 ? 1 : CCLD_GROUPS;
  const bool waitRequested = ccldEnvNumber("NCCL_CCLD_WAIT_ENABLE", 0, 0, 1) == 1;
  if (waitRequested && state->frameGroups != 1)
    WARN("CCL-D wait probe requires NCCL_ALGO=Ring");
  state->frameBlocks = static_cast<unsigned>(ccldEnvNumber(
      "NCCL_CCLD_FRAME_BLOCKS", CCLD_BLOCKS, CCLD_BLOCKS, CCLD_MAX_BLOCKS));
  state->slots.reset(new (std::nothrow) CcldState::Slot[state->frameBlocks]);
  state->samples.reset(new (std::nothrow)
      CcldState::ChannelSample[state->frameBlocks * state->frameChannels]);
  if (!state->slots || !state->samples) {
    WARN("CCL-D slot metadata allocation failed");
    delete state;
    return;
  }
  for (unsigned i = 0; i < state->frameBlocks; ++i)
    state->slots[i].channels = state->samples.get() + i * state->frameChannels;
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
  if (waitRequested && state->frameGroups == 1) {
    state->waitBlockBytes = state->frameChannels * state->frameGroups * sizeof(CcldWaitChannel);
    state->waitBytes = state->frameBlocks * state->waitBlockBytes;
    error = ccldMappedHostAlloc(&state->waitAllocation, state->waitBytes);
    if (error != hipSuccess) {
      WARN("CCL-D wait frame allocation failed: %s", hipGetErrorString(error));
      ccldMappedHostFree(&state->frameAllocation);
      delete state;
      return;
    }
    state->waitFrame = static_cast<CcldWaitChannel*>(state->waitAllocation.host);
    memset(state->waitFrame, 0, state->waitBytes);
  }

  const char* outputDir = ncclGetEnv("NCCL_CCLD_OUT_DIR");
  const char* nodeSocket = ncclGetEnv("NCCL_CCLD_NODE_SOCKET");
  state->queueSlots = static_cast<size_t>(ccldEnvNumber("NCCL_CCLD_QUEUE_SLOTS", 128, 128, 4096));
  state->logQueue.reset(new (std::nothrow) CcldQueue<CcldLogEvent>(state->queueSlots));
  if (!state->logQueue || !state->logQueue->valid() || !state->logQueue->lockFree()) {
    WARN("CCL-D binary queue is unavailable");
    ccldMappedHostFree(&state->waitAllocation);
    ccldMappedHostFree(&state->frameAllocation);
    delete state;
    return;
  }
  if (outputDir != nullptr && *outputDir != '\0') {
    char path[4096];
    const int length = snprintf(path, sizeof(path), "%s/ccld-%016" PRIx64 "-rank%d-pid%ld.ccldbin",
                                outputDir, state->commHash, state->rank, static_cast<long>(getpid()));
    if (length < 0 || static_cast<size_t>(length) >= sizeof(path) ||
        (state->output = fopen(path, "wb")) == nullptr) {
      WARN("CCL-D rank %d could not open telemetry output", state->rank);
      ccldMappedHostFree(&state->waitAllocation);
      ccldMappedHostFree(&state->frameAllocation);
      delete state;
      return;
    }
  }
  if (state->output == nullptr && (nodeSocket == nullptr || !state->nodeStream.open(nodeSocket))) {
    WARN("CCL-D requires NCCL_CCLD_NODE_SOCKET or NCCL_CCLD_OUT_DIR for binary telemetry");
    ccldMappedHostFree(&state->waitAllocation);
    ccldMappedHostFree(&state->frameAllocation);
    delete state;
    return;
  }

  if (state->useEvent) {
    for (unsigned i = 0; i < state->frameBlocks; ++i) {
      error = hipEventCreateWithFlags(&state->slots[i].completionEvent, hipEventDisableTiming);
      if (error != hipSuccess) {
        WARN("CCL-D completion event allocation failed: %s", hipGetErrorString(error));
        for (unsigned j = 0; j < i; ++j) hipEventDestroy(state->slots[j].completionEvent);
        ccldMappedHostFree(&state->waitAllocation);
        ccldMappedHostFree(&state->frameAllocation);
        if (state->output != nullptr) fclose(state->output);
        delete state;
        return;
      }
    }
  }
  memset(state->frame, 0, state->frameBytes);
  state->frame->mode = mode;
  CcldLogEvent init = ccldLogBase(state, CcldLogInit, 0, ccldNow());
  init.reserved = state->frameBlocks;
  init.frameBytes = static_cast<uint32_t>(state->frameBytes);
  init.frameChannels = state->frameChannels;
  init.frameGroups = state->frameGroups;
  init.pollUs = static_cast<uint32_t>(state->pollUs);
  init.mode = static_cast<uint32_t>(state->mode);
  init.count = state->waitBytes;
  init.bytes = state->queueSlots;
  init.windowNs = state->logQueue->allocatedBytes();
  snprintf(init.reason, sizeof(init.reason), "%s", state->useEvent ? "event_query" : "hostfunc");
  ccldWrite(state, init);
  try {
    state->writer = std::thread(ccldWriter, state);
    state->sampler = std::thread(ccldSampler, state);
  } catch (const std::exception& errorException) {
    WARN("CCL-D sampler creation failed: %s", errorException.what());
    state->writerStop.store(true, std::memory_order_release);
    if (state->writer.joinable()) state->writer.join();
    if (state->useEvent) for (unsigned i = 0; i < state->frameBlocks; ++i)
      hipEventDestroy(state->slots[i].completionEvent);
    ccldMappedHostFree(&state->waitAllocation);
    ccldMappedHostFree(&state->frameAllocation);
    if (state->output != nullptr) fclose(state->output);
    delete state;
    return;
  }
  comm->ccldState = state;
}

void ccldFree(struct ncclComm* comm) {
  CcldState* state = ccldState(comm);
  if (state == nullptr) return;
  state->stop.store(true, std::memory_order_release);
  state->cv.notify_one();
  if (state->sampler.joinable()) state->sampler.join();
  if (state->output != nullptr) {
    CcldLogEvent end = ccldLogBase(state, CcldLogEnd, state->sequence.next(), ccldNow());
    ccldWrite(state, end);
  }
  state->writerStop.store(true, std::memory_order_release);
  if (state->writer.joinable()) state->writer.join();
  bool active = state->callbacksOutstanding.load(std::memory_order_acquire) != 0;
  for (unsigned i = 0; i < state->frameBlocks; ++i)
    active = active || state->slots[i].phase.load(std::memory_order_acquire) != 0;
  comm->ccldState = nullptr;
  if (active) {
    WARN("CCL-D retains mapped storage while completion callback is active");
    return;
  }
  if (state->useEvent) for (unsigned i = 0; i < state->frameBlocks; ++i)
    hipEventDestroy(state->slots[i].completionEvent);
  if (state->output != nullptr) {
    fclose(state->output);
    state->output = nullptr;
  }
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
  if ((info->coll == ncclFuncAllReduce || info->coll == ncclFuncReduceScatter ||
       info->coll == ncclFuncReduce) && info->op >= ncclSum && info->op < ncclNumOps)
    start.builtinReduction = static_cast<uint32_t>(info->op) + 1;
  if (!state->pending.publish(seq, start)) {
    ccldDropped(state, seq, "pending_ring_full_or_counter_wrap");
    return seq;
  }
  CcldLogEvent entered = ccldLogBase(state, CcldLogEntered, seq, start.ns);
  entered.count = start.count;
  entered.bytes = start.bytes;
  ccldLogReduction(entered, start.builtinReduction);
  snprintf(entered.op, sizeof(entered.op), "%s", start.op ? start.op : "");
  snprintf(entered.datatype, sizeof(entered.datatype), "%s", start.datatype ? start.datatype : "");
  ccldWrite(state, entered);
  return seq;
}

void ccldDiscardGroupedTask(struct ncclComm* comm, uint64_t seq) {
  CcldState* state = ccldState(comm);
  if (state != nullptr && state->pending.forget(seq)) {
    ccldDropped(state, seq, "multiple_collectives");
  }
}

void ccldPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  CcldState* state = ccldState(comm);
  if (state == nullptr || plan == nullptr || plan->ccldSeq == 0) return;

  CcldPlanDescriptor descriptor;
  descriptor.persistent = plan->persistent;
  descriptor.multipleCollectives = plan->ccldTaskCount != 1;
  descriptor.collective = plan->ccldFunc == ncclFuncAllReduce ? CcldCollective::AllReduce :
      plan->ccldFunc == ncclFuncAllGather ? CcldCollective::AllGather :
      plan->ccldFunc == ncclFuncReduceScatter ? CcldCollective::ReduceScatter : CcldCollective::Other;
  descriptor.algorithm = plan->ccldAlgorithm == NCCL_ALGO_RING ? CcldAlgorithm::Ring :
      plan->ccldAlgorithm == NCCL_ALGO_TREE ? CcldAlgorithm::Tree : CcldAlgorithm::Other;

  ncclWorkList* first = ncclIntruQueueHead(&plan->workQueue);
  ncclWorkList* node = first;
  unsigned workCount = 0;
  while (node != nullptr) {
    ++workCount;
    if (node->workType == ncclDevWorkTypeP2p) descriptor.p2pPresent = true;
    node = node->next;
  }
  descriptor.multipleWork = workCount != 1;
  if (first != nullptr && first->workType == ncclDevWorkTypeColl) {
    descriptor.workType = CcldWorkType::Collective;
  } else if (first != nullptr && first->workType == ncclDevWorkTypeCollReg) {
    descriptor.workType = CcldWorkType::CollectiveRegistered;
  }

  if (first != nullptr && !descriptor.p2pPresent &&
      (first->workType == ncclDevWorkTypeColl || first->workType == ncclDevWorkTypeCollReg)) {
    ncclDevWorkColl* work = reinterpret_cast<ncclDevWorkColl*>(first + 1);
    descriptor.channelLo = work->channelLo;
    descriptor.channelHi = work->channelHi;
  }
  descriptor.frameChannels = state->frameChannels;
  descriptor.frameGroups = state->frameGroups;
  const CcldPlanDecision decision = ccldPlanDecision(descriptor);
  if (!decision.supported()) {
    if (state->pending.forget(plan->ccldSeq)) {
      ccldDropped(state, plan->ccldSeq, ccldPlanReasonString(decision.reason));
    }
    return;
  }
  if (plan->ccldProtocol == NCCL_PROTO_LL128) {
    state->pending.forget(plan->ccldSeq);
    ccldDropped(state, plan->ccldSeq, "ll128_unsupported_gfx936");
    return;
  }
  if (plan->ccldProtocol != NCCL_PROTO_SIMPLE && plan->ccldProtocol != NCCL_PROTO_LL) {
    state->pending.forget(plan->ccldSeq);
    ccldDropped(state, plan->ccldSeq, "unsupported_protocol");
    return;
  }
  const bool ringCollective = plan->ccldAlgorithm == NCCL_ALGO_RING &&
      (plan->ccldFunc == ncclFuncAllReduce || plan->ccldFunc == ncclFuncAllGather ||
       plan->ccldFunc == ncclFuncReduceScatter);
  const bool treeAllReduce = plan->ccldAlgorithm == NCCL_ALGO_TREE &&
      plan->ccldFunc == ncclFuncAllReduce;
  if (!ringCollective && !treeAllReduce) {
    state->pending.forget(plan->ccldSeq);
    ccldDropped(state, plan->ccldSeq, "unsupported_collective_algorithm");
    return;
  }

  CcldStart start{};
  if (!state->pending.take(plan->ccldSeq, &start)) {
    ccldDropped(state, plan->ccldSeq, "pending_metadata_missing");
    return;
  }
  const unsigned index = static_cast<unsigned>(plan->ccldSeq % state->frameBlocks);
  CcldState::Slot& slot = state->slots[index];
  int freePhase = 0;
  if (!slot.phase.compare_exchange_strong(freePhase, 1, std::memory_order_acq_rel)) {
    ccldDropped(state, plan->ccldSeq, "frame_slot_busy");
    return;
  }

  ncclDevWorkColl* work = reinterpret_cast<ncclDevWorkColl*>(first + 1);
  CcldBlock* block = ccldFrameBlock(state->frame, index, state->frameChannels, state->frameGroups);
  memset(block, 0, ccldBlockBytes(state->frameChannels, state->frameGroups));
  if (state->waitFrame != nullptr)
    memset(reinterpret_cast<unsigned char*>(state->waitFrame) + index * state->waitBlockBytes,
           0, state->waitBlockBytes);
  block->commHash = comm->commHash;
  block->opCount = plan->ccldSeq;
#ifdef CCLD_FAULT_INJECTION
  work->ccldFaultKind = 0;
  work->ccldFaultCycles = 0;
  if (state->faultKind != 0 && state->faultRank == static_cast<uint64_t>(state->rank) &&
      plan->ccldSeq >= state->faultOp && plan->ccldSeq - state->faultOp < state->faultRepeats) {
    work->ccldFaultKind = static_cast<uint32_t>(state->faultKind);
    work->ccldFaultCycles = state->faultCycles;
  }
#endif
  state->frame->kernelIndex = index;
  state->frame->counter = plan->ccldSeq;
  state->frame->detectChannel = descriptor.channelHi - descriptor.channelLo + 1;
  slot.seq = plan->ccldSeq;
  slot.startNs = start.ns;
  slot.firstProgressNs = 0;
  slot.count = start.count;
  slot.bytes = start.bytes;
  slot.builtinReduction = start.builtinReduction;
  slot.op = start.op;
  slot.datatype = start.datatype;
  slot.algorithm = plan->ccldAlgorithm == NCCL_ALGO_RING ? "Ring" : "Tree";
  slot.protocol = plan->ccldProtocol == NCCL_PROTO_SIMPLE ? "Simple" :
      plan->ccldProtocol == NCCL_PROTO_LL ? "LL" : "LL128";
  slot.channelLo = static_cast<int>(descriptor.channelLo);
  slot.channelHi = static_cast<int>(descriptor.channelHi);
  for (int channel = slot.channelLo; channel <= slot.channelHi; ++channel) {
    auto& sample = slot.channels[channel];
    sample = {};
    sample.parent = plan->ccldAlgorithm == NCCL_ALGO_TREE
        ? comm->channels[channel].tree.up : -1;
    sample.ringPrev = comm->channels[channel].ring.prev;
    sample.ringNext = comm->channels[channel].ring.next;
  }
  slot.lastDumpNs = slot.startNs;
  slot.done.store(false, std::memory_order_relaxed);
  slot.eventRecorded.store(false, std::memory_order_relaxed);
  slot.eventFailed.store(false, std::memory_order_relaxed);
  slot.phase.store(2, std::memory_order_release);
  work->ccldSlotPlus1 = static_cast<uint16_t>(index + 1);
  plan->ccldSlotPlus1 = static_cast<uint16_t>(index + 1);
  state->cv.notify_one();
}

void ccldCallbackEnqueue(struct ncclComm* comm, struct ncclKernelPlan* plan,
                         cudaStream_t stream) {
  CcldState* state = ccldState(comm);
  if (state == nullptr || plan == nullptr || plan->ccldSlotPlus1 == 0) return;
  const unsigned index = plan->ccldSlotPlus1 - 1;
  CcldState::Slot& slot = state->slots[index];
  if (state->useEvent) {
    hipError_t error = hipEventRecord(slot.completionEvent, stream);
    if (error == hipSuccess) {
      slot.eventRecorded.store(true, std::memory_order_release);
      state->cv.notify_one();
    } else {
      slot.eventFailed.store(true, std::memory_order_release);
      ccldDropped(state, slot.seq, "completion_event_record_failed");
    }
    return;
  }
  CcldCallback* callback = new (std::nothrow) CcldCallback{state, index};
  if (callback == nullptr) {
    ccldDropped(state, slot.seq, "host_callback_allocation_failed");
    return;
  }
  state->callbacksOutstanding.fetch_add(1, std::memory_order_relaxed);
  hipError_t error = hipLaunchHostFunc(stream, ccldDone, callback);
  if (error != hipSuccess) {
    delete callback;
    state->callbacksOutstanding.fetch_sub(1, std::memory_order_release);
    ccldDropped(state, slot.seq, "host_callback_enqueue_failed");
  }
}
