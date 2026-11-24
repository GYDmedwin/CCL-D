#include "ccld.h"
#include "ccld_log.h"
#include "ccld_queue.h"
#include "ccld_stream.h"
#include "comm.h"
#include "info.h"
#include "collectives.h"
#include "debug.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <inttypes.h>
#include <memory>
#include <mutex>
#include <new>
#include <pthread.h>
#include <thread>
#include <time.h>
#include <unistd.h>

static uint64_t ccldNow() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + t.tv_nsec;
}

struct CcldStart {
  uint64_t ns, count, bytes;
  const char* op;
  const char* dtype;
};
struct CcldPending {
  std::atomic<uint64_t> owner{0};
  CcldStart start{};
};
struct CcldSlot {
  // 0=free, 1=producer preparing fields, 2=visible to the sampler.
  std::atomic<int> phase{0};
  std::atomic<bool> done{false};
  std::atomic<bool> eventRecorded{false};
  std::atomic<uint64_t> endNs{0};
  cudaEvent_t completionEvent = nullptr;
  bool eventFailed = false;
  uint64_t seq = 0, startNs = 0, lastDumpNs = 0;
  uint64_t count = 0, bytes = 0;
  const char* op = nullptr;
  const char* dtype = nullptr;
  const char* algo = nullptr;
  const char* proto = nullptr;
  int lo = 0, hi = -1;
  const CcldTreeWaitChannel* treeWait = nullptr;
  int parents[CCLD_MAX_CHANNELS] = {};
  int ringPrev[CCLD_MAX_CHANNELS] = {}, ringNext[CCLD_MAX_CHANNELS] = {};
  uint64_t lastSend[CCLD_MAX_CHANNELS] = {}, lastRecv[CCLD_MAX_CHANNELS] = {};
  uint64_t lastWaitData[CCLD_MAX_CHANNELS] = {}, lastWaitCredit[CCLD_MAX_CHANNELS] = {};
  uint64_t dumpSend[CCLD_MAX_CHANNELS] = {}, dumpRecv[CCLD_MAX_CHANNELS] = {};
  uint64_t changesSend[CCLD_MAX_CHANNELS] = {}, changesRecv[CCLD_MAX_CHANNELS] = {};
};
struct CcldState {
  std::mutex waitMu;
  std::condition_variable cv;
  std::atomic<bool> stop{false}, writerStop{false}, ioFailed{false};
  std::atomic<int> callbacksOutstanding{0};
  std::atomic<uint64_t> counter{0}, logDrops{0}, lastDroppedSeq{0};
  FILE* out = nullptr;
  CcldNodeStream nodeStream;
  CcldFrame* frame = nullptr;
  CcldFrame* device = nullptr;
  CcldWaitChannel* waitFrame = nullptr;
  CcldWaitChannel* waitDevice = nullptr;
  std::unique_ptr<ncclTree[]> treeTopology;
  size_t waitBytes = 0, waitBlockBytes = 0;
  size_t frameBytes = 0;
  unsigned frameChannels = 0, frameGroups = 0, frameBlocks = CCLD_BLOCKS;
  std::thread thread, writer;
  std::unique_ptr<CcldSlot[]> slots;
  CcldPending pending[256];
  std::unique_ptr<CcldQueue<CcldLogEvent>> logQueue;
  size_t queueSlots = 0;
  uint64_t hash = 0, pollUs = 200, dumpNs = 1000000000ull, treeDumpNs = 1000000000ull;
  uint64_t mode = 1;
  int cudaDev = -1;
  bool useEvent = true;
  char host[64] = {};
#ifdef CCLD_FAULT_INJECTION
  uint64_t faultKind = 0, faultRank = UINT64_MAX, faultOp = 0, faultCycles = 0, faultRepeats = 1;
#endif
  int rank = 0, nranks = 0;
  ~CcldState() { if (out) fclose(out); }
};
struct CcldCallback { CcldState* state; int slot; };

static uint64_t envNumber(const char* key, uint64_t fallback, uint64_t lo, uint64_t hi) {
  const char* s = getenv(key);
  if (!s || !*s) return fallback;
  char* end = nullptr;
  unsigned long long v = strtoull(s, &end, 10);
  return end != s && *end == 0 && v >= lo && v <= hi ? v : fallback;
}
static void ccldCopyString(char* dst, size_t bytes, const char* src) {
  if (src && bytes) { strncpy(dst, src, bytes-1); dst[bytes-1] = '\0'; }
}
static CcldLogEvent ccldLogBase(CcldState* s, CcldLogType type, uint64_t seq, uint64_t now) {
  CcldLogEvent e{};
  e.magic = 0x444c4343u; e.version = 1; e.type = type;
  e.hash = s->hash; e.seq = seq; e.nowNs = now; e.rank = s->rank; e.nranks = s->nranks;
  return e;
}
static void ccldLog(CcldState* s, const CcldLogEvent& e) {
  if (!s->logQueue->push(e)) {
    s->lastDroppedSeq.store(e.seq, std::memory_order_relaxed);
    s->logDrops.fetch_add(1, std::memory_order_release);
  }
}
static void ccldStatus(CcldState* s, const char*, uint64_t seq, const char* reason) {
  CcldLogEvent e = ccldLogBase(s, CcldLogDropped, seq, ccldNow());
  ccldCopyString(e.reason, sizeof(e.reason), reason);
  ccldLog(s, e);
}
static void ccldWriter(CcldState* s) {
  (void)pthread_setname_np(pthread_self(), "ccld-writer");
  CcldLogEvent batch[16];
  for (;;) {
    size_t n = 0;
    uint64_t lost = s->logDrops.exchange(0, std::memory_order_acq_rel);
    if (lost) {
      batch[n] = ccldLogBase(s, CcldLogDropped, s->lastDroppedSeq.load(std::memory_order_relaxed), ccldNow());
      batch[n].count = lost;
      ccldCopyString(batch[n].reason, sizeof(batch[n].reason), "binary_queue_overflow");
      ++n;
    }
    while (n < 16 && s->logQueue->pop(batch[n])) ++n;
    if (n) {
      if (s->ioFailed.load(std::memory_order_relaxed)) continue;
      const bool sent = s->out ?
          (fwrite(batch, sizeof(CcldLogEvent), n, s->out) == n && fflush(s->out) == 0) :
          s->nodeStream.write(batch, n * sizeof(CcldLogEvent));
      if (!sent) {
        s->ioFailed.store(true, std::memory_order_relaxed);
        s->nodeStream.close();
        WARN("CCL-D telemetry writer lost its sink");
      }
    } else if (s->writerStop.load(std::memory_order_acquire)) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  if (!s->out && !s->ioFailed.load(std::memory_order_relaxed)) {
    CcldLogEvent end = ccldLogBase(s, CcldLogEnd, s->counter.load(std::memory_order_relaxed), ccldNow());
    if (!s->nodeStream.write(&end, sizeof(end))) WARN("CCL-D could not deliver final telemetry marker");
    s->nodeStream.close();
  }
}
static uint64_t ccldLoad(const uint64_t* p) {
  return *(const volatile uint64_t*)p;
}
static void ccldSample(CcldState* s, int idx) {
  CcldSlot& slot = s->slots[idx];
  const CcldBlock& b = *ccldFrameBlock(s->frame, idx, s->frameChannels, s->frameGroups);
  const CcldChannel* channels = reinterpret_cast<const CcldChannel*>(reinterpret_cast<const unsigned char*>(&b) + offsetof(CcldBlock, channel));
  for (int c = slot.lo; c <= slot.hi; ++c) {
    uint64_t send = 0, recv = 0;
    for (unsigned g = 0; g < s->frameGroups; ++g) {
      const CcldChannel& channel = channels[c * s->frameGroups + g];
      send += ccldLoad(&channel.send);
      recv += ccldLoad(&channel.recv);
    }
    if (send != slot.lastSend[c]) ++slot.changesSend[c];
    if (recv != slot.lastRecv[c]) ++slot.changesRecv[c];
    slot.lastSend[c] = send;
    slot.lastRecv[c] = recv;
    if (s->waitFrame && !s->treeTopology) {
      const CcldWaitChannel* waits = reinterpret_cast<const CcldWaitChannel*>(
          reinterpret_cast<const unsigned char*>(s->waitFrame) + size_t(idx) * s->waitBlockBytes);
      uint64_t data = 0, credit = 0;
      for (unsigned g = 0; g < s->frameGroups; ++g) {
        const CcldWaitChannel& wait = waits[c * s->frameGroups + g];
        data += ccldLoad(&wait.data); credit += ccldLoad(&wait.credit);
      }
      slot.lastWaitData[c] = data; slot.lastWaitCredit[c] = credit;
    }
  }
}
static void ccldEmit(CcldState* s, int idx, const char* type, uint64_t now) {
  CcldSlot& slot = s->slots[idx];
  const uint64_t window = now > slot.lastDumpNs ? now - slot.lastDumpNs : 1;
  CcldLogEvent e = ccldLogBase(s, strcmp(type, "complete") == 0 ? CcldLogComplete : CcldLogSnapshot, slot.seq, now);
  e.startNs = slot.startNs; e.windowNs = window; e.count = slot.count; e.bytes = slot.bytes;
  e.nChannels = slot.hi-slot.lo+1;
  ccldCopyString(e.op, sizeof(e.op), slot.op); ccldCopyString(e.algorithm, sizeof(e.algorithm), slot.algo);
  ccldCopyString(e.protocol, sizeof(e.protocol), slot.proto); ccldCopyString(e.datatype, sizeof(e.datatype), slot.dtype);
  CcldLogEvent w{};
  const bool ringWait = s->waitFrame && !s->treeTopology;
  const bool treeWait = slot.treeWait != nullptr;
  if (ringWait || treeWait) {
    ccldCopyString(e.reason, sizeof(e.reason), treeWait ? "tree_wait_v1" : "ring_wait_v1");
    w = ccldLogBase(s, treeWait ? CcldLogTreeWait : CcldLogRingWait, slot.seq, now);
    w.reserved = e.type; w.nChannels = e.nChannels;
  }
  for (int c = slot.lo; c <= slot.hi; ++c) {
    CcldLogChannel& channel = e.channels[c - slot.lo];
    channel.id = c; channel.parent = slot.parents[c];
    channel.send = slot.lastSend[c]; channel.recv = slot.lastRecv[c];
    channel.sendChanges = slot.changesSend[c]; channel.recvChanges = slot.changesRecv[c];
    channel.sendDelta = slot.lastSend[c] - slot.dumpSend[c]; channel.recvDelta = slot.lastRecv[c] - slot.dumpRecv[c];
    if (treeWait) {
      const ncclTree& topology = s->treeTopology[c];
      CcldTreeLogChannel row{};
      row.id = c;
      row.parent = topology.up;
      static_assert(CCLD_TREE_PEERS - 1 == NCCL_MAX_TREE_ARITY, "Tree peer count differs");
      for (unsigned p = 0; p < CCLD_TREE_PEERS - 1; ++p) row.children[p] = topology.down[p];
      for (unsigned p = 0; p < CCLD_TREE_PEERS; ++p) {
        const auto& edge = slot.treeWait[c].peers[p];
        row.data[p] = std::min<uint64_t>(ccldLoad(&edge.data), UINT32_MAX);
        row.credit[p] = std::min<uint64_t>(ccldLoad(&edge.credit), UINT32_MAX);
      }
      memcpy(&w.channels[c - slot.lo], &row, sizeof(row));
    } else if (ringWait) {
      CcldLogChannel& wait = w.channels[c - slot.lo];
      wait.id = c; wait.parent = slot.ringPrev[c]; wait.sendChanges = uint64_t(slot.ringNext[c]);
      wait.send = slot.lastWaitData[c]; wait.recv = slot.lastWaitCredit[c];
    }
    slot.dumpSend[c] = slot.lastSend[c];
    slot.dumpRecv[c] = slot.lastRecv[c];
    slot.changesSend[c] = slot.changesRecv[c] = 0;
  }
  if (ringWait || treeWait) ccldLog(s, w); // Paired by rank, sequence, timestamp and base type.
  ccldLog(s, e);
  slot.lastDumpNs = now;
}
static void ccldWorker(CcldState* s) {
  (void)pthread_setname_np(pthread_self(), "ccld-probe");
  const bool deviceReady = !s->useEvent || cudaSetDevice(s->cudaDev) == cudaSuccess;
  std::unique_lock<std::mutex> lock(s->waitMu);
  if (!deviceReady) ccldStatus(s, "dropped", 0, "event_worker_device_setup_failed");
  uint64_t lastHeartbeat = ccldNow();
  while (!s->stop.load(std::memory_order_acquire)) {
    s->cv.wait_for(lock, std::chrono::microseconds(s->pollUs));
    uint64_t now = ccldNow();
    for (unsigned i = 0; i < s->frameBlocks; ++i) {
      CcldSlot& slot = s->slots[i];
      if (slot.phase.load(std::memory_order_acquire) != 2) continue;
      ccldSample(s, i);
      if (s->useEvent && deviceReady && slot.eventRecorded.load(std::memory_order_acquire) &&
          !slot.done.load(std::memory_order_relaxed) && !slot.eventFailed) {
        cudaError_t query = cudaEventQuery(slot.completionEvent);
        if (query == cudaSuccess) {
          ccldSample(s, i); // Re-read after CUDA confirms the kernel has completed.
          slot.endNs.store(ccldNow(), std::memory_order_relaxed);
          slot.done.store(true, std::memory_order_release);
        } else if (query != cudaErrorNotReady) {
          slot.eventFailed = true;
          ccldStatus(s, "dropped", slot.seq, "completion_event_query_failed");
        }
      }
      if (slot.done.load(std::memory_order_acquire)) {
        uint64_t end = slot.endNs.load(std::memory_order_relaxed);
        ccldEmit(s, i, "complete", end);
        slot.phase.store(0, std::memory_order_release);
      } else if (now >= slot.lastDumpNs &&
                 now - slot.lastDumpNs >= (slot.treeWait ? s->treeDumpNs : s->dumpNs)) {
        ccldEmit(s, i, "snapshot", now);
      }
    }
    if (now - lastHeartbeat >= s->dumpNs) {
      ccldLog(s, ccldLogBase(s, CcldLogHeartbeat, s->counter.load(std::memory_order_relaxed), now));
      lastHeartbeat = now;
    }
  }
  // Drain already completed slots even if shutdown follows a callback quickly.
  for (unsigned i = 0; i < s->frameBlocks; ++i) {
    CcldSlot& slot = s->slots[i];
    if (slot.phase.load(std::memory_order_acquire) != 2) continue;
    ccldSample(s, i);
    if (s->useEvent && deviceReady && slot.eventRecorded.load(std::memory_order_acquire) &&
        !slot.done.load(std::memory_order_relaxed) && !slot.eventFailed && cudaEventQuery(slot.completionEvent) == cudaSuccess) {
      ccldSample(s, i);
      slot.endNs.store(ccldNow(), std::memory_order_relaxed);
      slot.done.store(true, std::memory_order_release);
    }
    if (slot.done.load(std::memory_order_acquire)) {
      ccldEmit(s, i, "complete", slot.endNs.load(std::memory_order_relaxed));
      slot.phase.store(0, std::memory_order_release);
    } else {
      ccldEmit(s, i, "snapshot", ccldNow());
    }
  }
}
void ccldInit(ncclComm* comm) {
  static_assert(MAXCHANNELS <= CCLD_MAX_CHANNELS, "increase CCLD_MAX_CHANNELS");
  if (envNumber("NCCL_CCLD_ENABLE", 0, 0, 1) != 1) return;
  uint64_t mode = envNumber("NCCL_CCLD_MODE", 1, 0, 2);
  if (mode == 0) return;
  if (mode != 1) {
    WARN("CCL-D store-only prototype supports mode 1 only; telemetry disabled");
    return;
  }
  const char* dir = getenv("NCCL_CCLD_OUT_DIR");
  const char* nodeSocket = getenv("NCCL_CCLD_NODE_SOCKET");
  const bool inMemory = nodeSocket && *nodeSocket;
  if (!inMemory && (!dir || !*dir)) {
    WARN("CCL-D requires NCCL_CCLD_NODE_SOCKET or NCCL_CCLD_OUT_DIR; telemetry disabled");
    return;
  }
  CcldState* s = new CcldState();
  s->queueSlots = size_t(envNumber("NCCL_CCLD_QUEUE_SLOTS", 128, 128, 4096));
  s->logQueue.reset(new (std::nothrow) CcldQueue<CcldLogEvent>(s->queueSlots));
  if (!s->logQueue || !s->logQueue->valid() || !s->logQueue->lockFree() || !s->counter.is_lock_free()) {
    WARN("CCL-D requires lock-free host queue and counter atomics"); delete s; return;
  }
  s->rank = comm->rank; s->nranks = comm->nRanks; s->hash = comm->commHash; s->mode = mode;
  s->cudaDev = comm->cudaDev;
  const char* completion = getenv("NCCL_CCLD_COMPLETION");
  if (completion && strcmp(completion, "hostfunc") == 0) s->useEvent = false;
  else if (completion && strcmp(completion, "event") != 0) WARN("CCL-D unknown completion mode; using event");
#ifdef CCLD_FAULT_INJECTION
  const char* kind = getenv("NCCL_CCLD_FAULT_KIND");
  if (kind && strcmp(kind, "H3") == 0) s->faultKind = 1;
  else if (kind && strcmp(kind, "S2") == 0) s->faultKind = 2;
  s->faultRank = envNumber("NCCL_CCLD_FAULT_RANK", UINT64_MAX, 0, uint64_t(comm->nRanks-1));
  s->faultOp = envNumber("NCCL_CCLD_FAULT_OP", 0, 1, UINT64_MAX);
  s->faultRepeats = envNumber("NCCL_CCLD_FAULT_REPEATS", 1, 1, 100);
  // Bound the simulated stall: user may choose up to 30 billion GPU clock cycles.
  s->faultCycles = envNumber("NCCL_CCLD_FAULT_CYCLES", 0, 1, 30000000000ull);
  if (s->faultKind && (s->faultRank == UINT64_MAX || s->faultOp == 0 || s->faultCycles == 0)) {
    WARN("CCL-D fault simulation disabled: rank, operation and bounded cycles are required");
    s->faultKind = 0;
  }
#else
  if (getenv("NCCL_CCLD_FAULT_KIND")) WARN("CCL-D fault injection is not compiled into this build");
#endif
  s->pollUs = envNumber("NCCL_CCLD_POLL_US", 200, 20, 1000000);
  const uint64_t dumpMs = envNumber("NCCL_CCLD_DUMP_MS", 1000, 10, 60000);
  s->dumpNs = dumpMs * 1000000ull;
  s->treeDumpNs = envNumber("NCCL_CCLD_TREE_SNAPSHOT_MS", dumpMs, 1, 60000) * 1000000ull;
  if (inMemory) {
    if (!s->nodeStream.open(nodeSocket)) {
      WARN("CCL-D cannot connect to local node agent; telemetry disabled"); delete s; return;
    }
  } else {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/ccld-%016" PRIx64 "-rank%d-pid%ld.ccldbin", dir, s->hash, s->rank, (long)getpid());
    if (n < 0 || size_t(n) >= sizeof(path) || (s->out = fopen(path, "wb")) == nullptr) {
      WARN("CCL-D cannot open output; telemetry disabled"); delete s; return;
    }
  }
  if (comm->nChannels < 1 || comm->nChannels > CCLD_MAX_CHANNELS) {
    WARN("CCL-D channel count is outside the supported frame range");
    delete s; return;
  }
  s->frameChannels = unsigned(comm->nChannels);
  const char* selectedAlgo = getenv("NCCL_ALGO");
  s->frameGroups = selectedAlgo && strcasecmp(selectedAlgo, "Ring") == 0 ? 1 : CCLD_GROUPS;
  const bool waitRequested = envNumber("NCCL_CCLD_WAIT_ENABLE", 0, 0, 1) == 1;
  if (waitRequested && s->frameGroups != 1) WARN("CCL-D wait experiment requires NCCL_ALGO=Ring");
  const bool treeRequested = envNumber("NCCL_CCLD_TREE_WAIT_ENABLE", 0, 0, 1) == 1;
  const bool treeWait = treeRequested && selectedAlgo && strcasecmp(selectedAlgo, "Tree") == 0;
  if (treeRequested && !treeWait) WARN("CCL-D Tree wait requires NCCL_ALGO=Tree");
  if (treeWait) {
    s->treeTopology.reset(new (std::nothrow) ncclTree[s->frameChannels]);
    if (!s->treeTopology) {
      WARN("CCL-D Tree topology allocation failed");
      delete s; return;
    }
    for (unsigned c = 0; c < s->frameChannels; ++c) s->treeTopology[c] = comm->channels[c].tree;
  }
  s->frameBlocks = unsigned(envNumber("NCCL_CCLD_FRAME_BLOCKS", CCLD_BLOCKS, CCLD_BLOCKS, CCLD_MAX_BLOCKS));
  s->slots.reset(new (std::nothrow) CcldSlot[s->frameBlocks]);
  if (!s->slots) {
    WARN("CCL-D frame slot metadata allocation failed");
    delete s; return;
  }
  s->frameBytes = ccldFrameBytes(s->frameChannels, s->frameGroups, s->frameBlocks);
  cudaError_t err = cudaHostAlloc((void**)&s->frame, s->frameBytes, cudaHostAllocMapped);
  if (err == cudaSuccess) err = cudaHostGetDevicePointer((void**)&s->device, s->frame, 0);
  if (err != cudaSuccess) {
    WARN("CCL-D mapped allocation failed: %s", cudaGetErrorString(err));
    if (s->frame) cudaFreeHost(s->frame);
    delete s; return;
  }
  if (treeWait || (waitRequested && s->frameGroups == 1)) {
    s->waitBlockBytes = size_t(s->frameChannels) * (treeWait ? sizeof(CcldTreeWaitChannel) : sizeof(CcldWaitChannel));
    s->waitBytes = s->frameBlocks * s->waitBlockBytes;
    err = cudaHostAlloc((void**)&s->waitFrame, s->waitBytes, cudaHostAllocMapped);
    if (err == cudaSuccess) err = cudaHostGetDevicePointer((void**)&s->waitDevice, s->waitFrame, 0);
    if (err != cudaSuccess) {
      WARN("CCL-D wait frame allocation failed: %s", cudaGetErrorString(err));
      if (s->waitFrame) cudaFreeHost(s->waitFrame);
      cudaFreeHost(s->frame); delete s; return;
    }
    memset(s->waitFrame, 0, s->waitBytes);
  }
  if (s->useEvent) {
    for (unsigned i = 0; i < s->frameBlocks; ++i) {
      err = cudaEventCreateWithFlags(&s->slots[i].completionEvent, cudaEventDisableTiming);
      if (err != cudaSuccess) {
        WARN("CCL-D event allocation failed: %s", cudaGetErrorString(err));
        for (unsigned j = 0; j < i; ++j) cudaEventDestroy(s->slots[j].completionEvent);
        if (s->waitFrame) cudaFreeHost(s->waitFrame);
        cudaFreeHost(s->frame); delete s; return;
      }
    }
  }
  memset(s->frame, 0, s->frameBytes); s->frame->mode = mode;
  gethostname(s->host, sizeof(s->host)-1);
  CcldLogEvent init = ccldLogBase(s, CcldLogInit, 0, ccldNow());
  init.reserved = s->frameBlocks; // Version-1 init records used zero before this extension.
  init.bytes = s->queueSlots; init.windowNs = s->logQueue->allocatedBytes();
  init.count = s->waitBytes;
  init.frameBytes = s->frameBytes; init.frameChannels = s->frameChannels; init.frameGroups = s->frameGroups;
  init.pollUs = s->pollUs; init.mode = s->mode;
  ccldCopyString(init.host, sizeof(init.host), s->host);
  ccldCopyString(init.reason, sizeof(init.reason), s->useEvent ? "event_query" : "hostfunc");
  ccldLog(s, init);
  try {
    s->writer = std::thread(ccldWriter, s);
    s->thread = std::thread(ccldWorker, s);
  } catch (const std::exception& e) {
    WARN("CCL-D thread creation failed: %s", e.what());
    s->writerStop.store(true, std::memory_order_release);
    if (s->writer.joinable()) s->writer.join();
    if (s->useEvent) for (unsigned i = 0; i < s->frameBlocks; ++i) cudaEventDestroy(s->slots[i].completionEvent);
    if (s->waitFrame) cudaFreeHost(s->waitFrame);
    cudaFreeHost(s->frame); delete s; return;
  }
  comm->ccldState = s;
}
void ccldDeviceConfig(ncclComm* comm, CcldFrame** frame, uint32_t* blockBytes, uint32_t* groups,
                      CcldWaitChannel** waitFrame, uint32_t* waitBlockBytes) {
  CcldState* s = static_cast<CcldState*>(comm->ccldState);
  *frame = s ? s->device : nullptr;
  *blockBytes = s ? uint32_t(ccldBlockBytes(s->frameChannels, s->frameGroups)) : 0;
  *groups = s ? s->frameGroups : 0;
  *waitFrame = s ? s->waitDevice : nullptr;
  *waitBlockBytes = s ? uint32_t(s->waitBlockBytes) : 0;
}
uint64_t ccldEnter(ncclComm* comm, const ncclInfo* info) {
  CcldState* s = static_cast<CcldState*>(comm->ccldState);
  uint64_t seq = s->counter.fetch_add(1, std::memory_order_relaxed) + 1;
  uint64_t now = ccldNow();
  uint64_t bytes = info->count * uint64_t(ncclTypeSize(info->datatype));
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncReduceScatter) bytes *= comm->nRanks;
  CcldStart start{now, info->count, bytes,
                  ncclFuncToString(info->coll), ncclDatatypeToString(info->datatype)};
  CcldPending& pending = s->pending[seq % 256];
  uint64_t empty = 0;
  if (seq == 0 || seq == UINT64_MAX || !pending.owner.compare_exchange_strong(empty, UINT64_MAX, std::memory_order_acq_rel)) {
    ccldStatus(s, "dropped", seq, "pending_ring_full_or_counter_wrap");
  } else {
    pending.start = start;
    pending.owner.store(seq, std::memory_order_release);
  }
  CcldLogEvent entered = ccldLogBase(s, CcldLogEntered, seq, now);
  entered.count = start.count; entered.bytes = start.bytes;
  ccldCopyString(entered.op, sizeof(entered.op), start.op);
  ccldCopyString(entered.datatype, sizeof(entered.datatype), start.dtype);
  ccldLog(s, entered);
  return seq;
}
static void ccldForgetPending(CcldState* s, uint64_t seq) {
  CcldPending& pending = s->pending[seq % 256];
  uint64_t expected = seq;
  pending.owner.compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
}
void ccldPlan(ncclComm* comm, ncclKernelPlan* plan) {
  CcldState* s = static_cast<CcldState*>(comm->ccldState);
  ncclTaskColl* task = ncclIntruQueueHead(&plan->collTaskQueue);
  if (!task) return;
  const bool supported = !plan->persistent && !task->next && ncclIntruQueueEmpty(&plan->p2pTaskQueue) &&
      (task->func == ncclFuncAllReduce || task->func == ncclFuncAllGather || task->func == ncclFuncReduceScatter) &&
      (task->algorithm == NCCL_ALGO_RING || (task->func == ncclFuncAllReduce && task->algorithm == NCCL_ALGO_TREE));
  if (!supported) {
    for (ncclTaskColl* t = task; t; t = t->next) {
      ccldStatus(s, "dropped", t->ccldSeq, "unsupported_group_graph_or_operation");
      ccldForgetPending(s, t->ccldSeq);
    }
    return;
  }
  CcldPending& pending = s->pending[task->ccldSeq % 256];
  uint64_t expectedSeq = task->ccldSeq;
  if (!pending.owner.compare_exchange_strong(expectedSeq, UINT64_MAX, std::memory_order_acq_rel)) {
    ccldStatus(s, "dropped", task->ccldSeq, "pending_metadata_missing");
    return;
  }
  CcldStart start = pending.start;
  pending.owner.store(0, std::memory_order_release);
  const int index = int(task->ccldSeq % s->frameBlocks);
  CcldSlot& slot = s->slots[index];
  ncclWorkList* node = ncclIntruQueueHead(&plan->workQueue);
  if (!node || (node->workType != ncclDevWorkTypeColl && node->workType != ncclDevWorkTypeCollReg)) {
    ccldStatus(s, "dropped", task->ccldSeq, "unexpected_work_type");
    return;
  }
  if (node->next) {
    ccldStatus(s, "dropped", task->ccldSeq, "store_only_multi_work_unsupported");
    return;
  }
  ncclDevWorkColl* work = reinterpret_cast<ncclDevWorkColl*>(node + 1);
  if (work->channelLo > work->channelHi || work->channelHi >= s->frameChannels ||
      (task->algorithm == NCCL_ALGO_TREE && s->frameGroups < 2)) {
    ccldStatus(s, "dropped", task->ccldSeq, "dynamic_frame_capacity_mismatch");
    return;
  }
  int freePhase = 0;
  if (!slot.phase.compare_exchange_strong(freePhase, 1, std::memory_order_acq_rel)) {
    ccldStatus(s, "dropped", task->ccldSeq, "frame_slot_busy");
    return;
  }
  CcldBlock* block = ccldFrameBlock(s->frame, index, s->frameChannels, s->frameGroups);
  memset(block, 0, ccldBlockBytes(s->frameChannels, s->frameGroups));
  if (s->waitFrame) memset(reinterpret_cast<unsigned char*>(s->waitFrame) + size_t(index) * s->waitBlockBytes, 0, s->waitBlockBytes);
  block->commHash = s->hash;
  block->opCount = task->ccldSeq;
#ifdef CCLD_FAULT_INJECTION
  if (s->faultKind && s->faultRank == uint64_t(s->rank) && task->ccldSeq >= s->faultOp && task->ccldSeq - s->faultOp < s->faultRepeats) {
    work->ccldFaultKind = s->faultKind;
    work->ccldFaultCycles = s->faultCycles;
  }
#endif
  __atomic_store_n(&s->frame->kernelIndex, uint64_t(index), __ATOMIC_RELAXED);
  __atomic_store_n(&s->frame->counter, task->ccldSeq, __ATOMIC_RELAXED);
  __atomic_store_n(&s->frame->detectChannel, uint64_t(work->channelHi - work->channelLo + 1), __ATOMIC_RELAXED);
  slot.seq = task->ccldSeq; slot.startNs = start.ns; slot.lastDumpNs = slot.startNs;
  slot.count = start.count; slot.bytes = start.bytes;
  slot.op = start.op; slot.dtype = start.dtype;
  slot.algo = ncclAlgoToString(task->algorithm); slot.proto = ncclProtoToString(task->protocol);
  slot.lo = work->channelLo; slot.hi = work->channelHi;
  slot.treeWait = s->treeTopology && task->algorithm == NCCL_ALGO_TREE && task->protocol == NCCL_PROTO_SIMPLE ?
      reinterpret_cast<const CcldTreeWaitChannel*>(
          reinterpret_cast<const unsigned char*>(s->waitFrame) + size_t(index) * s->waitBlockBytes) : nullptr;
  for (int c = 0; c < CCLD_MAX_CHANNELS; ++c) {
    slot.parents[c] = comm->channels[c].tree.up;
    slot.ringPrev[c] = comm->channels[c].ring.prev; slot.ringNext[c] = comm->channels[c].ring.next;
    slot.lastWaitData[c] = slot.lastWaitCredit[c] = 0;
    slot.lastSend[c] = slot.lastRecv[c] = slot.dumpSend[c] = slot.dumpRecv[c] = 0;
    slot.changesSend[c] = slot.changesRecv[c] = 0;
  }
  slot.done.store(false, std::memory_order_relaxed); slot.endNs.store(0, std::memory_order_relaxed);
  slot.eventRecorded.store(false, std::memory_order_relaxed); slot.eventFailed = false;
  slot.phase.store(2, std::memory_order_release);
  work->ccldSlotPlus1 = index + 1;
  plan->ccldSlotPlus1 = index + 1;
  s->cv.notify_one();
}
static void CUDART_CB ccldDone(void* data) {
  CcldCallback* cb = static_cast<CcldCallback*>(data);
  CcldState* s = cb->state;
  CcldSlot& slot = s->slots[cb->slot];
  slot.endNs.store(ccldNow(), std::memory_order_relaxed);
  slot.done.store(true, std::memory_order_release);
  s->cv.notify_one();
  delete cb;
  // This must be the callback's last access to s; destruction checks it.
  s->callbacksOutstanding.fetch_sub(1, std::memory_order_release);
}
void ccldCallbackEnqueue(ncclComm* comm, ncclKernelPlan* plan, cudaStream_t stream) {
  CcldState* s = static_cast<CcldState*>(comm->ccldState);
  if (s->useEvent) {
    CcldSlot& slot = s->slots[plan->ccldSlotPlus1-1];
    cudaError_t err = cudaEventRecord(slot.completionEvent, stream);
    if (err == cudaSuccess) {
      slot.eventRecorded.store(true, std::memory_order_release);
      s->cv.notify_one();
    } else {
      ccldStatus(s, "dropped", slot.seq, "completion_event_record_failed");
      WARN("CCL-D event record failed: %s", cudaGetErrorString(err));
      // Do not reuse the slot while a kernel may still be writing to it.
    }
    return;
  }
  CcldCallback* cb = new CcldCallback{s, plan->ccldSlotPlus1-1};
  s->callbacksOutstanding.fetch_add(1, std::memory_order_relaxed);
  cudaError_t err = cudaLaunchHostFunc(stream, ccldDone, cb);
  if (err != cudaSuccess) {
    delete cb;
    s->callbacksOutstanding.fetch_sub(1, std::memory_order_release);
    ccldStatus(s, "dropped", s->slots[plan->ccldSlotPlus1-1].seq, "host_callback_failed");
    WARN("CCL-D callback enqueue failed: %s", cudaGetErrorString(err));
    // Do not reuse the slot: the kernel may still be writing to it.
  }
}
void ccldFree(ncclComm* comm) {
  CcldState* s = static_cast<CcldState*>(comm->ccldState);
  if (!s) return;
  s->stop.store(true, std::memory_order_release);
  s->cv.notify_one();
  s->thread.join();
  s->writerStop.store(true, std::memory_order_release);
  s->writer.join();
  bool active = false;
  for (unsigned i = 0; i < s->frameBlocks; ++i) active |= s->slots[i].phase.load(std::memory_order_acquire) != 0;
  active |= s->callbacksOutstanding.load(std::memory_order_acquire) != 0;
  if (s->out) {
    fclose(s->out);
    s->out = nullptr;
  }
  comm->ccldState = nullptr;
  if (active) {
    // An aborted/erroring CUDA stream may suppress callbacks. Keep storage alive
    // until process exit rather than risking a device or callback use-after-free.
    WARN("CCL-D retains pinned storage after incomplete/aborted kernel");
    return;
  }
  if (s->useEvent) for (unsigned i = 0; i < s->frameBlocks; ++i) cudaEventDestroy(s->slots[i].completionEvent);
  if (s->waitFrame) cudaFreeHost(s->waitFrame);
  cudaFreeHost(s->frame);
  delete s;
}
