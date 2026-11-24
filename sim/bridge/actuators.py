"""Bridge the simulator's physics clock to the FC's independently scheduled motor updates."""

import select
import socket
import threading
import time


class ActuatorLink:
    """Forward every FDM sample and hold the latest actual FC output for each physics tick."""

    def __init__(self):
        self.stop = threading.Event()
        self.enabled = False
        self.last_motor = 0.0
        self.motor = bytes(16)
        self.fdm = self._bind(("127.0.0.1", 9003))
        self.commands = self._bind(("127.0.0.2", 9002))
        threading.Thread(target=self._exchange, daemon=True).start()

    @staticmethod
    def _bind(address):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # SITL binds 0.0.0.0:9003 with SO_REUSEADDR. Unicast for .1 reaches
        # the address-specific socket here; unicast for .2 reaches SITL.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
        return sock

    def _exchange(self):
        gazebo = ("127.0.0.1", 9002)
        fc = ("127.0.0.2", 9003)
        last_fdm = 0
        while not self.stop.is_set():
            readable, _, _ = select.select([self.commands, self.fdm], [], [], 0.1)
            now = time.monotonic()
            if self.commands in readable:
                packet = self.commands.recv(256)
                if len(packet) == 16 and self.enabled:
                    self.motor, self.last_motor = packet, now
            if self.fdm in readable:
                packet = self.fdm.recv(2048)
                if self.enabled:
                    self.fdm.sendto(packet, fc)
                    # Keep physics and rate feedback in step. A bounded wait also
                    # lets Gazebo advance while the FC is still booting.
                    if select.select([self.commands], [], [], .006)[0]:
                        latest = self.commands.recv(256)
                        if len(latest) == 16:
                            self.motor, self.last_motor = latest, time.monotonic()
                # One reply per tick prevents queued stale commands when rendering slows down.
                motor = self.motor if self.enabled and time.monotonic() - self.last_motor < 0.25 else bytes(16)
                self.commands.sendto(motor, gazebo)
                last_fdm = now
            elif now - last_fdm > 0.2:
                # Start the official plugin's request/reply loop with stopped motors.
                self.commands.sendto(bytes(16), gazebo)
                last_fdm = now
