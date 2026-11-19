#include "ccld_queue.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>
struct Item { unsigned producer, index; };
int main() {
  CcldQueue<Item> invalid(1);
  assert(!invalid.valid() && invalid.allocatedBytes() == 0);
  CcldQueue<Item> dynamic(2048);
  assert(dynamic.valid() && dynamic.allocatedBytes() >= 2048 * sizeof(Item));
  for (unsigned i = 0; i < 2048; ++i) assert(dynamic.push({1, i}));
  assert(!dynamic.push({1, 2048}));
  for (unsigned i = 0; i < 2048; ++i) { Item item; assert(dynamic.pop(item) && item.index == i); }
  CcldQueue<Item, 2> small;
  assert(small.push({0, 0})); assert(small.push({0, 1})); assert(!small.push({0, 2}));
  Item v; assert(small.pop(v) && v.index == 0); assert(small.push({0, 2}));
  assert(small.pop(v) && v.index == 1); assert(small.pop(v) && v.index == 2); assert(!small.pop(v));
  const unsigned producers = 4, n = 20000;
  CcldQueue<Item, 128> queue;
  assert(queue.lockFree());
  std::atomic<unsigned> finished{0};
  std::vector<std::thread> threads;
  for (unsigned p = 0; p < producers; ++p) threads.emplace_back([&, p] {
    for (unsigned i = 0; i < n; ++i) while (!queue.push({p, i})) std::this_thread::yield();
    finished.fetch_add(1, std::memory_order_release);
  });
  std::vector<unsigned> next(producers, 0);
  unsigned consumed = 0;
  while (consumed < producers * n) {
    Item item;
    if (queue.pop(item)) {
      assert(item.producer < producers && item.index == next[item.producer]);
      ++next[item.producer]; ++consumed;
    } else std::this_thread::yield();
  }
  for (auto& thread : threads) thread.join();
  assert(finished.load(std::memory_order_acquire) == producers && !queue.pop(v));
  printf("bounded queue: 80000 ordered events, dynamic capacity and full/reuse checks passed\n");
}
