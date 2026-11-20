#include <stddef.h>
#include <stdint.h>

#include "device.h"
#include "comm.h"

static_assert(offsetof(ncclDevWorkCollReg, coll) == 0,
              "registered work must embed collective work first");
static_assert(sizeof(ncclDevWorkColl) % 16 == 0,
              "collective work must preserve 16-byte alignment");
static_assert(sizeof(((ncclDevWorkColl*)0)->ccldSlotPlus1) == sizeof(uint16_t),
              "CCL-D slot field width changed");
static_assert(offsetof(ncclDevCommAndChannels, comm) == 0,
              "device comm must remain the first member");

int main() {
  return (sizeof(ncclDevWorkCollReg) >= sizeof(ncclDevWorkColl) &&
          sizeof(ncclDevCommAndChannels) >= sizeof(ncclDevComm)) ? 0 : 1;
}
