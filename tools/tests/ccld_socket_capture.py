#!/usr/bin/env python3
import pathlib
import socket
import sys
import threading

sock_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
expected = int(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(sock_path))
server.listen(expected)
errors = []
threads = []

def capture(conn, index):
    try:
        with conn, (out_dir / f"rank-stream-{index}.ccldbin").open("wb") as output:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                output.write(chunk)
    except Exception as error:  # surfaced in the parent after joining
        errors.append(error)

for index in range(expected):
    conn, _ = server.accept()
    thread = threading.Thread(target=capture, args=(conn, index))
    thread.start()
    threads.append(thread)
server.close()
for thread in threads:
    thread.join()
if errors:
    raise errors[0]
