#define __CUDACC__ 1
#include "ccld_log.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>

int main(int argc, char** argv) {
  assert(argc == 2);
  CcldLogEvent wait{}, done{};
  for (CcldLogEvent* e : {&wait, &done}) {
    e->magic = 0x444c4343u; e->version = 1; e->hash = 123; e->seq = 4;
    e->nowNs = 1000; e->rank = 0; e->nranks = 8; e->nChannels = 1;
    e->channels[0].id = 2;
  }
  wait.type = CcldLogRingWait; wait.reserved = CcldLogComplete;
  wait.channels[0].parent = 7; wait.channels[0].sendChanges = 1;
  wait.channels[0].send = 37; wait.channels[0].recv = 42;
  done.type = CcldLogComplete; done.startNs = 100; done.windowNs = 500;
  strcpy(done.reason, "ring_wait_v1"); strcpy(done.op, "AllReduce");
  strcpy(done.algorithm, "RING"); strcpy(done.protocol, "SIMPLE");
  FILE* out = fopen(argv[1], "wb"); assert(out);
  assert(fwrite(&wait, sizeof(wait), 1, out) == 1);
  assert(fwrite(&done, sizeof(done), 1, out) == 1);
  assert(fclose(out) == 0);
}
