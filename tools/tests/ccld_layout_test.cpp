#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "ccld.h"
#include "ccld_log.h"

int main() {
  assert(sizeof(CcldChannel) == 16);
  assert(sizeof(CcldWaitChannel) == 16);
  assert(offsetof(CcldFrame, blocks) == 32);
  assert(offsetof(CcldBlock, channel) == 16);
  assert(sizeof(CcldLogChannel) == 56);
  assert(sizeof(CcldLogEvent) == 2088);

  const unsigned channels = 7;
  const unsigned groups = 2;
  const unsigned blocks = 5;
  const size_t blockBytes = 16 + channels * groups * sizeof(CcldChannel);
  assert(ccldBlockBytes(channels, groups) == blockBytes);
  assert(ccldFrameBytes(channels, groups, blocks) == 32 + blocks * blockBytes);

  CcldWaitChannel waitGroups[2] = {{11, 13}, {17, 19}};
  uint64_t waitData = 0, waitCredit = 0;
  ccldWaitTotals(waitGroups, 2, &waitData, &waitCredit);
  assert(waitData == 28 && waitCredit == 32);
  ccldWaitTotals(waitGroups, 1, &waitData, &waitCredit);
  assert(waitData == 11 && waitCredit == 13);

  unsigned char storage[32 + blocks * blockBytes] = {};
  CcldFrame* frame = reinterpret_cast<CcldFrame*>(storage);
  for (unsigned i = 0; i < blocks; ++i) {
    CcldBlock* block = ccldFrameBlock(frame, i, channels, groups);
    assert(reinterpret_cast<unsigned char*>(block) == storage + 32 + i * blockBytes);
    assert(reinterpret_cast<unsigned char*>(block + 1) <= storage + 32 + (i + 1) * blockBytes);
  }

  assert(CcldLogInit == 1 && CcldLogRingWait == 7 && CcldLogEnd == 8);
  return 0;
}
