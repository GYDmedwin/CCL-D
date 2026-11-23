#ifndef NCCL_CCLD_LOG_H_
#define NCCL_CCLD_LOG_H_

#include <stdint.h>
#include <type_traits>
#include "ccld.h"

// Keep the fixed wire stride. V2 snapshots use consecutive parts when
// the active channels do not fit in one V1 record.
constexpr unsigned CCLD_RECORD_CHANNELS = 32;
constexpr uint32_t CCLD_FRAGMENT_VERSION = 2;
constexpr uint32_t CCLD_OBSERVED_PROGRESS = 1u << 31;
enum CcldLogType : uint32_t {
  CcldLogInit = 1,
  CcldLogEntered = 2,
  CcldLogSnapshot = 3,
  CcldLogComplete = 4,
  CcldLogHeartbeat = 5,
  CcldLogDropped = 6,
  CcldLogRingWait = 7,
  CcldLogEnd = 8
};

struct CcldLogChannel {
  int32_t id, parent;
  uint64_t send, recv, sendChanges, recvChanges, sendDelta, recvDelta;
};

struct CcldLogEvent {
  uint32_t magic, version, type, reserved;
  uint64_t hash, seq, nowNs, startNs, count, bytes, windowNs;
  uint32_t rank, nranks, frameBytes, frameChannels, frameGroups, pollUs, mode, nChannels;
  char host[64], op[16], algorithm[16], protocol[16], datatype[16], reason[64];
  CcldLogChannel channels[CCLD_RECORD_CHANNELS];
};

static_assert(sizeof(CcldLogChannel) == 56, "binary channel layout changed");
static_assert(sizeof(CcldLogEvent) == 2088, "binary event layout changed");
static_assert(std::is_trivially_copyable<CcldLogEvent>::value,
              "binary records must be trivially copyable");

// Progress records do not use the initialization record's frame dimensions.
inline void ccldLogFirstProgress(CcldLogEvent& event, uint64_t monoNs) {
  event.mode |= CCLD_OBSERVED_PROGRESS;
  event.frameBytes = static_cast<uint32_t>(monoNs);
  event.frameChannels = static_cast<uint32_t>(monoNs >> 32);
}

#endif
