#include "ccld_stream.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

int main() {
  char path[sizeof(sockaddr_un::sun_path)];
  std::snprintf(path, sizeof(path), "/tmp/ccld-stream-%ld.sock", static_cast<long>(getpid()));
  const int server = socket(AF_UNIX, SOCK_STREAM, 0);
  assert(server >= 0);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strcpy(address.sun_path, path);
  assert(bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  assert(listen(server, 1) == 0);

  const std::string payload(8192, 'c');
  std::string received;
  std::thread receiver([&] {
    const int client = accept(server, nullptr, nullptr);
    assert(client >= 0);
    char buffer[257];
    for (;;) {
      const ssize_t bytes = recv(client, buffer, sizeof(buffer), 0);
      if (bytes == 0) break;
      assert(bytes > 0);
      received.append(buffer, static_cast<size_t>(bytes));
    }
    close(client);
  });

  CcldNodeStream stream;
  assert(stream.open(path));
  assert(stream.write(payload.data(), payload.size()));
  stream.close();
  receiver.join();
  close(server);
  unlink(path);
  assert(received == payload);

  CcldNodeStream missing;
  assert(!missing.open("/tmp/ccld-stream-socket-does-not-exist.sock"));
  return 0;
}
