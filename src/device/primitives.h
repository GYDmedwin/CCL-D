/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common_kernel.h"
#include "common.h"
#include "ccld.h"

__device__ __forceinline__ CcldChannel* ccldOutput(ncclDevWorkColl* work, int tid, int group) {
  if (tid != 0 || work == nullptr || work->ccldSlotPlus1 == 0 || ncclShmem.comm.ccldFrame == nullptr) return nullptr;
  unsigned g = group == 0 ? 0 : 1;
  if (g >= ncclShmem.comm.ccldGroups) return nullptr;
  unsigned char* base = reinterpret_cast<unsigned char*>(ncclShmem.comm.ccldFrame) +
      offsetof(CcldFrame, blocks) + (work->ccldSlotPlus1 - 1) * ncclShmem.comm.ccldBlockBytes +
      offsetof(CcldBlock, channel);
  CcldChannel* channels = reinterpret_cast<CcldChannel*>(base);
  return &channels[ncclShmem.channelId * ncclShmem.comm.ccldGroups + g];
}

__device__ __forceinline__ void ccldCount(CcldChannel* channel, int tid, int group,
                                          bool send, bool recv) {
  if (tid != 0 || channel == nullptr) return;
  const unsigned g = group == 0 ? 0 : 1;
  CcldChannel& local = ncclShmem.ccldCounts[g];
  if (send) *(volatile uint64_t*)&channel->send = ++local.send;
  if (recv) *(volatile uint64_t*)&channel->recv = ++local.recv;
}

__device__ __forceinline__ CcldWaitChannel* ccldWaitOutput(ncclDevWorkColl* work, int group) {
  if (work == nullptr || work->ccldSlotPlus1 == 0 || ncclShmem.comm.ccldWaitFrame == nullptr) return nullptr;
  const unsigned g = group == 0 ? 0 : 1;
  if (g >= ncclShmem.comm.ccldGroups) return nullptr;
  unsigned char* base = reinterpret_cast<unsigned char*>(ncclShmem.comm.ccldWaitFrame) +
      (work->ccldSlotPlus1 - 1) * ncclShmem.comm.ccldWaitBlockBytes +
      (ncclShmem.channelId * ncclShmem.comm.ccldGroups + g) * sizeof(CcldWaitChannel);
  return reinterpret_cast<CcldWaitChannel*>(base);
}

__device__ __forceinline__ void ccldWaitAdd(CcldWaitChannel* output, int group, bool credit, uint64_t delta) {
  if (output == nullptr || delta == 0) return;
  CcldWaitChannel& local = ncclShmem.ccldWaitCounts[group == 0 ? 0 : 1];
  if (credit) *(volatile uint64_t*)&output->credit = (local.credit += delta);
  else *(volatile uint64_t*)&output->data = (local.data += delta);
}

#ifdef CCLD_FAULT_INJECTION
// NV CCL-D H3/S2 experiment: bounded, abort-aware device delay. H3 executes
// once per primitive subgroup; S2 executes for each send operation.
__device__ __forceinline__ void ccldInject(uint32_t kind, uint64_t cycles, int tid, bool& hangDone) {
  if (tid != 0 || kind == 0 || cycles == 0) return;
  if (kind == 1 && hangDone) return;
  if (kind == 1) hangDone = true;
  const unsigned long long started = clock64();
  while (clock64() - started < cycles) {
    if (*ncclShmem.comm.abortFlag) break;
  }
}
#endif

#define NCCL_SPINS_BEFORE_CHECK_ABORT 1000000

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define __THREAD_FENCE __threadfence_block()
#else
#define __THREAD_FENCE __threadfence()
#endif

#define barrier_by_group() do { \
  if (nthreads == NCCL_MAX_NTHREADS) { \
    __THREAD_FENCE; __builtin_amdgcn_s_barrier(); \
  } else { \
    const int w = threadIdx.x/WARP_SIZE; \
    const int wid = threadIdx.x%WARP_SIZE; \
    if (wid == 0) { \
      barrier_next += nthreads/WARP_SIZE; \
      __hip_atomic_fetch_add(barriers, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP); \
      int spins = 0; \
      int rate_limit = 50; \
      while (__hip_atomic_load(barriers, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) < barrier_next) { \
        spins++; \
        if (spins == NCCL_SPINS_BEFORE_CHECK_ABORT) { \
          if (__atomic_load_n(ncclShmem.comm.abortFlag, __ATOMIC_SEQ_CST)) { \
            ncclShmem.aborted = 1; \
            break; \
          } \
          spins = 0; \
        } \
        if (spins == 0 && rate_limit > 0) { \
          rate_limit --; \
          traceData(__LINE__, threadIdx.x, __hip_atomic_load(barriers, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP), barrier_next); \
        } \
        __builtin_amdgcn_s_sleep(1); \
      } \
      __asm__ __volatile__("s_wakeup"); \
      __THREAD_FENCE; \
    } \
  } \
} while (0)

/* Protocol classes: ProtoSimple, ProtoLL, ProtoLL128
 * We use these as template args to the Primtiives class instead of integral
 * enums (e.g. NCCL_PROTO_LL) because for SIMPLE we need to carry a few extra
 * numbers. Also these types hold methods which let us compute numbers important
 * to how that protocol operates with a consistent interface so that our
 * algorithm code can operate protocol parametrically.
 */
template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1, int MultimemSrcs_1 = 0, int MultimemDsts_1 = 0>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;
  static constexpr int MultimemSrcs = MultimemSrcs_1;
  static constexpr int MultimemDsts = MultimemDsts_1;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};

struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2; // Half is data
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // One 16-byte line has 8-bytes of data
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};

struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_DATAELEMS*sizeof(uint64_t)/NCCL_LL128_LINEELEMS;
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};

/* Fan (as in fan-in & fan-out) classes hold recv and send counts. The template
 * arguments are static bounds on the maximum values. Asymmetric counts are
 * independent. Symmetric is a static guarantee that nrecv==nsend, so it only
 * stores one value at runtime. This optimization save 32-bit register, but more
 * importantly uses fewer predicate registers when unrolling loops.
 */
template<int MaxRecv_, int MaxSend_>
struct FanAsymmetric {
  static constexpr int MaxRecv = MaxRecv_, MaxSend = MaxSend_;
  int nr, ns;
  FanAsymmetric() = default;
  __device__ FanAsymmetric(int nrecv, int nsend): nr(nrecv), ns(nsend) {
    // assert(nrecv <= MaxRecv && nsend <= MaxSend);
  }
  __device__ int nrecv() const { return MaxRecv ? nr : 0; }
  __device__ int nsend() const { return MaxSend ? ns : 0; }
};

template<int MaxArity>
struct FanSymmetric {
  static constexpr int MaxRecv = MaxArity, MaxSend = MaxArity;
  int n;
  FanSymmetric() = default;
  __device__ FanSymmetric(int nrecv, int nsend): n(nrecv) {
    // assert(nrecv == nsend && nrecv <= MaxArity);
  }
  __device__ int nrecv() const { return n; }
  __device__ int nsend() const { return n; }
};

// The primitives class. Specialized per protocol in the other headers.
template<typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p>
class Primitives;

// Used by LL & LL128 to implement direct members in the naive way.
template<typename RealPrimitives>
struct PrimitivesWithoutDirect {
  __device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }
  __device__ void directSendFromOutput(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->sendFromOutput(outIx, eltN);
  }
  __device__ void directRecv(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->recv(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->copySend(inpIx, outIx, eltN, postOp);
  }
  __device__ void directRecvCopySend(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->recvCopySend(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directRecvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
};

#include "prims_simple.h"
#include "prims_ll.h"
#include "prims_ll128.h"
#endif
