#ifndef NCCL_CCLD_STREAM_H_
#define NCCL_CCLD_STREAM_H_

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// The writer thread owns the socket; sampler and communication paths enqueue
// fixed-size events and never perform socket I/O.
class CcldNodeStream {
  int fd = -1;

 public:
  CcldNodeStream() = default;
  CcldNodeStream(const CcldNodeStream&) = delete;
  CcldNodeStream& operator=(const CcldNodeStream&) = delete;
  ~CcldNodeStream() { close(); }

  bool open(const char* path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path == nullptr || std::strlen(path) >= sizeof(address.sun_path)) return false;
    std::strcpy(address.sun_path, path);
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0 || fcntl(fd, F_SETFL, O_NONBLOCK) != 0 ||
        connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
      close();
      return false;
    }
    return true;
  }

  bool write(const void* data, size_t bytes) {
    const char* cursor = static_cast<const char*>(data);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (bytes != 0) {
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now()).count();
      if (left <= 0) return false;
#ifdef MSG_NOSIGNAL
      const ssize_t written = send(fd, cursor, bytes, MSG_NOSIGNAL);
#else
      const ssize_t written = send(fd, cursor, bytes, 0);
#endif
      if (written > 0) {
        cursor += written;
        bytes -= static_cast<size_t>(written);
      } else if (written < 0 && errno == EINTR) {
        continue;
      } else if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        pollfd wake{fd, POLLOUT, 0};
        if (poll(&wake, 1, static_cast<int>(left)) < 0 && errno != EINTR) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  void close() {
    if (fd >= 0) ::close(fd);
    fd = -1;
  }
};

#endif
