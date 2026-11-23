#include <assert.h>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "ccld_queue.h"

static void test_capacity_and_wrap() {
  CcldQueue<uint64_t> queue(2);
  assert(queue.valid());
  assert(queue.allocatedBytes() >= 2 * sizeof(uint64_t));
  uint64_t value = 0;
  assert(!queue.pop(value));
  assert(queue.push(10));
  assert(queue.push(11));
  assert(!queue.push(12));
  assert(queue.pop(value) && value == 10);
  assert(queue.push(12));
  assert(queue.pop(value) && value == 11);
  assert(queue.pop(value) && value == 12);
  assert(!queue.pop(value));
}

static void test_invalid_capacity() {
  CcldQueue<uint64_t> tooSmall(1);
  assert(!tooSmall.valid());
  assert(tooSmall.allocatedBytes() == 0);
  assert(!tooSmall.push(1));
  uint64_t value = 0;
  assert(!tooSmall.pop(value));
}

static void test_batch_capacity_and_wrap() {
  CcldQueue<uint64_t> queue(4);
  uint64_t value = 0;
  const uint64_t first[] = {1, 2, 3};
  const uint64_t second[] = {4, 5, 6};
  assert(queue.pushBatch(first, 3));
  assert(!queue.pushBatch(second, 3));
  assert(queue.pop(value) && value == 1);
  assert(queue.pop(value) && value == 2);
  assert(queue.pushBatch(second, 3));
  for (uint64_t expected = 3; expected <= 6; ++expected)
    assert(queue.pop(value) && value == expected);
  assert(!queue.pop(value));
}

static void test_multiple_producers() {
  const uint64_t producerCount = 4;
  const uint64_t valuesPerProducer = 20000;
  CcldQueue<uint64_t> queue(256);
  assert(queue.valid());
  std::atomic<uint64_t> consumed{0};
  std::vector<std::thread> producers;
  for (uint64_t producer = 0; producer < producerCount; ++producer) {
    producers.emplace_back([&, producer] {
      for (uint64_t i = 0; i < valuesPerProducer; i += 2) {
        const uint64_t first = producer * valuesPerProducer + i;
        const uint64_t batch[] = {first, first + 1};
        while (!queue.pushBatch(batch, 2)) std::this_thread::yield();
      }
    });
  }

  std::vector<uint8_t> seen(producerCount * valuesPerProducer, 0);
  uint64_t value = 0;
  while (consumed.load(std::memory_order_relaxed) < producerCount * valuesPerProducer) {
    if (queue.pop(value)) {
      assert(value < seen.size() && value % 2 == 0);
      assert(seen[value] == 0);
      seen[value] = 1;
      uint64_t second = 0;
      assert(queue.pop(second) && second == value + 1 && seen[second] == 0);
      seen[second] = 1;
      consumed.fetch_add(2, std::memory_order_relaxed);
    } else {
      std::this_thread::yield();
    }
  }
  for (auto& producer : producers) producer.join();
  for (uint8_t observed : seen) assert(observed == 1);
}

int main() {
  test_capacity_and_wrap();
  test_invalid_capacity();
  test_batch_capacity_and_wrap();
  test_multiple_producers();
  return 0;
}
