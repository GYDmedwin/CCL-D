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

static void test_multiple_producers() {
  const uint64_t producerCount = 4;
  const uint64_t valuesPerProducer = 20000;
  CcldQueue<uint64_t> queue(256);
  assert(queue.valid());
  std::atomic<uint64_t> consumed{0};
  std::vector<std::thread> producers;
  for (uint64_t producer = 0; producer < producerCount; ++producer) {
    producers.emplace_back([&, producer] {
      for (uint64_t i = 0; i < valuesPerProducer; ++i) {
        const uint64_t value = producer * valuesPerProducer + i;
        while (!queue.push(value)) std::this_thread::yield();
      }
    });
  }

  std::vector<uint8_t> seen(producerCount * valuesPerProducer, 0);
  uint64_t value = 0;
  while (consumed.load(std::memory_order_relaxed) < producerCount * valuesPerProducer) {
    if (queue.pop(value)) {
      assert(value < seen.size());
      assert(seen[value] == 0);
      seen[value] = 1;
      consumed.fetch_add(1, std::memory_order_relaxed);
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
  test_multiple_producers();
  return 0;
}
