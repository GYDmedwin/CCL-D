#ifndef NCCL_CCLD_QUEUE_H_
#define NCCL_CCLD_QUEUE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <limits>

// Bounded MPSC queue: producers never wait for the consumer when full.
template <typename T, size_t Capacity = 0>
class CcldQueue {
  static_assert(Capacity == 0 || Capacity >= 2, "queue needs at least two cells");

  struct Cell {
    std::atomic<size_t> sequence;
    T value;
  };

  const size_t capacity;
  std::unique_ptr<Cell[]> cells;
  std::atomic<size_t> producer{0};
  size_t consumer = 0;

 public:
  explicit CcldQueue(size_t requested = Capacity)
      : capacity(requested),
        cells(requested >= 2 && requested <= std::numeric_limits<size_t>::max() / sizeof(Cell)
                  ? new (std::nothrow) Cell[requested]
                  : nullptr) {
    if (cells) {
      for (size_t i = 0; i < capacity; ++i) {
        cells[i].sequence.store(i, std::memory_order_relaxed);
      }
    }
  }

  bool valid() const { return cells != nullptr; }
  size_t allocatedBytes() const { return cells ? capacity * sizeof(Cell) : 0; }
  bool lockFree() const { return producer.is_lock_free(); }

  bool push(const T& value) {
    if (!cells) return false;
    size_t pos = producer.load(std::memory_order_relaxed);
    Cell* cell;
    for (;;) {
      cell = &cells[pos % capacity];
      const size_t sequence = cell->sequence.load(std::memory_order_acquire);
      const intptr_t diff = intptr_t(sequence) - intptr_t(pos);
      if (diff == 0) {
        if (producer.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (diff < 0) {
        return false;
      } else {
        pos = producer.load(std::memory_order_relaxed);
      }
    }
    cell->value = value;
    cell->sequence.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool pop(T& value) {
    if (!cells) return false;
    Cell& cell = cells[consumer % capacity];
    const size_t sequence = cell.sequence.load(std::memory_order_acquire);
    if (intptr_t(sequence) - intptr_t(consumer + 1) != 0) return false;
    value = cell.value;
    cell.sequence.store(consumer + capacity, std::memory_order_release);
    ++consumer;
    return true;
  }
};

#endif
