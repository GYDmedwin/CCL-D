#ifndef NCCL_CCLD_PENDING_H_
#define NCCL_CCLD_PENDING_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

struct CcldStart {
  uint64_t ns;
  uint64_t count;
  uint64_t bytes;
  const char* op;
  const char* datatype;
};

// Rank-local operation sequence used to correlate API entry and kernel plan.
class CcldSequence {
 public:
  explicit CcldSequence(uint64_t initial = 0) : counter(initial) {}

  uint64_t next(bool* valid = nullptr) {
    const uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool ok = seq != 0 && seq != std::numeric_limits<uint64_t>::max();
    if (valid) *valid = ok;
    return seq;
  }

  bool lockFree() const { return counter.is_lock_free(); }

 private:
  std::atomic<uint64_t> counter;
};

// Fixed-size pending ring matching NCCL CCL-D. owner=0 is empty and
// owner=UINT64_MAX marks a slot being published or consumed.
template <size_t Capacity = 256>
class CcldPendingTable {
  static_assert(Capacity >= 2, "pending ring needs at least two entries");
  static constexpr uint64_t Busy = std::numeric_limits<uint64_t>::max();

  struct Entry {
    std::atomic<uint64_t> owner{0};
    CcldStart start{};
  };

 public:
  bool publish(uint64_t seq, const CcldStart& start) {
    if (seq == 0 || seq == Busy) return false;
    Entry& entry = entries[seq % Capacity];
    uint64_t empty = 0;
    if (!entry.owner.compare_exchange_strong(empty, Busy, std::memory_order_acq_rel)) return false;
    entry.start = start;
    entry.owner.store(seq, std::memory_order_release);
    return true;
  }

  bool take(uint64_t seq, CcldStart* start) {
    if (seq == 0 || seq == Busy || start == nullptr) return false;
    Entry& entry = entries[seq % Capacity];
    uint64_t expected = seq;
    if (!entry.owner.compare_exchange_strong(expected, Busy, std::memory_order_acq_rel)) return false;
    *start = entry.start;
    entry.owner.store(0, std::memory_order_release);
    return true;
  }

  bool forget(uint64_t seq) {
    if (seq == 0 || seq == Busy) return false;
    Entry& entry = entries[seq % Capacity];
    uint64_t expected = seq;
    return entry.owner.compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
  }

 private:
  Entry entries[Capacity];
};

#endif
