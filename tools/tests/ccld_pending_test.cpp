#include <assert.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#include "ccld_pending.h"

static void test_sequence_and_wrap() {
  CcldSequence sequence;
  bool valid = false;
  assert(sequence.next(&valid) == 1 && valid);
  assert(sequence.next(&valid) == 2 && valid);

  CcldSequence nearWrap(std::numeric_limits<uint64_t>::max() - 1);
  assert(nearWrap.next(&valid) == std::numeric_limits<uint64_t>::max() && !valid);
  assert(nearWrap.next(&valid) == 0 && !valid);
}

static void test_pending_publish_take_and_forget() {
  CcldPendingTable<4> pending;
  const CcldStart input{123, 17, 272, "AllReduce", "Float32"};
  CcldStart output{};
  assert(!pending.take(1, &output));
  assert(pending.publish(1, input));
  assert(!pending.publish(1, input));
  assert(pending.take(1, &output));
  assert(output.ns == input.ns && output.count == input.count && output.bytes == input.bytes);
  assert(output.op == input.op && output.datatype == input.datatype);
  assert(!pending.take(1, &output));

  assert(pending.publish(2, input));
  assert(pending.forget(2));
  assert(!pending.take(2, &output));
  assert(!pending.forget(2));
}

static void test_ring_collision_and_reserved_sequences() {
  CcldPendingTable<4> pending;
  const CcldStart input{1, 2, 8, "AllReduce", "Int32"};
  CcldStart output{};
  assert(pending.publish(1, input));
  assert(!pending.publish(5, input)); // Same slot while seq 1 is still pending.
  assert(!pending.publish(0, input));
  assert(!pending.publish(std::numeric_limits<uint64_t>::max(), input));
  assert(!pending.take(5, &output));
  assert(pending.take(1, &output));
  assert(pending.publish(5, input));
}

static void test_sequence_concurrency() {
  const uint64_t threadCount = 4;
  const uint64_t perThread = 10000;
  CcldSequence sequence;
  std::vector<uint64_t> values(threadCount * perThread);
  std::vector<std::thread> threads;
  for (uint64_t t = 0; t < threadCount; ++t) {
    threads.emplace_back([&, t] {
      for (uint64_t i = 0; i < perThread; ++i) {
        bool valid = false;
        values[t * perThread + i] = sequence.next(&valid);
        assert(valid);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  std::sort(values.begin(), values.end());
  for (uint64_t i = 0; i < values.size(); ++i) assert(values[i] == i + 1);
}

int main() {
  test_sequence_and_wrap();
  test_pending_publish_take_and_forget();
  test_ring_collision_and_reserved_sequences();
  test_sequence_concurrency();
  return 0;
}
