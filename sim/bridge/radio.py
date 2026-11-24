"""Radio input and read-only flight status for the local Betaflight SITL process."""

import socket
import struct
import threading
import time

NEUTRAL = (1500, 1500, 1000, 1500)


class Radio:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.disarm()
        self.thread = threading.Thread(target=self._transmit, daemon=True)
        self.thread.start()

    def disarm(self):
        with self.lock:
            self.rc = NEUTRAL
            self.arm_switch = False
            self.deadline = 0.0

    def command(self, rc):
        with self.lock:
            self.rc = tuple(rc)
            self.arm_switch = True
            self.deadline = time.monotonic() + 10.0

    def state(self):
        with self.lock:
            if time.monotonic() >= self.deadline:
                self.arm_switch = False
                self.rc = NEUTRAL
            return self.rc, self.arm_switch

    def _transmit(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            while not self.stop.is_set():
                rc, arm = self.state()
                packet = struct.pack("<d16H", time.monotonic(), *rc, 2000 if arm else 1000,
                                     *([1000] * 11))
                sock.sendto(packet, ("127.0.0.1", 9004))
                self.stop.wait(0.01)
