"""Synchronous client for the local SITL bridge."""

import base64
import threading
import struct
import time
from collections import deque
from functools import lru_cache
import cv2
from sim.optics import WIDTH, HEIGHT, CX, CY, FOCAL, K1, EDGE_SCALE
from dataclasses import dataclass
import json
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


@dataclass
class Sample:
    rgb: np.ndarray
    xyz: np.ndarray
    sim_time: float
    armed: bool = True
    quaternion: np.ndarray | None = None
    applied: float | None = None
    image_time: float | None = None
    image_count: int = 0
    sent_wall: float = 0.0


class Bridge:
    def __init__(self, url="http://127.0.0.1:8765", timeout=180):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.owner = uuid4().hex

    def _post(self, path, body):
        request = Request(
            self.url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Controller": self.owner},
            method="POST",
        )
        for attempt in range(4):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except HTTPError as error:
                detail = error.read(2048).decode(errors="replace")
                # A busy reply is sent before processing the request. An accepted
                # step or a different controller's ownership must never be retried.
                if attempt < 3 and error.code == 409 and "processing another command" in detail:
                    time.sleep(0.01)
                    continue
                raise RuntimeError(f"{path} returned HTTP {error.code}: {detail}") from error
            except (URLError, TimeoutError) as error:
                raise RuntimeError(f"Could not reach SITL bridge at {self.url}{path}: {error}") from error

    def _sample(self, path, body):
        data = json.loads(self._post(path, body))
        width, height = data["width"], data["height"]
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1:
            raise ValueError("Invalid camera dimensions")
        rgb = np.frombuffer(base64.b64decode(data["rgb"], validate=True), dtype=np.uint8)
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        sim_time = float(data["sim_time"])
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(sim_time):
            raise ValueError("Invalid Gazebo pose or timestamp")
        self.session = data["session"]
        return Sample(
            rgb.reshape(height, width, 3),
            xyz,
            sim_time,
            data["armed"],
            np.asarray(data["quaternion"]),
            image_time=sim_time,
        )

    def reset(self, settings=None):
        return self._sample("/reset", {"settings": settings})

    def step(self, rc):
        return self._sample("/step", {"rc": rc})

    def disarm(self):
        self._post("/disarm", {})


class FlightBridge(Bridge):
    """Independent latest-frame receiver and paced, lightweight RC request channel."""

    def __init__(self, url, timeout=180):
        super().__init__(url, timeout)
        self.reader = None

    def reset(self, settings=None):
        self._stop_camera()
        sample = super().reset(settings)
        self.frames = deque([(sample.sim_time, sample.rgb, 0)], maxlen=4)
        self.current_frame = self.frames[0]
        self.frame_lock = threading.Lock()
        self.failure = None
        self.camera_stop = threading.Event()
        self.reader = threading.Thread(target=self._receive, args=(self.session,), daemon=True)
        self.received = 0
        self.sim_rx_fps = 0.0
        self.receiver_start = time.monotonic()
        self.work_period = 1 / 60
        self.reader.start()
        self.sent = time.monotonic()
        return sample

    def _receive(self, session):
        after = 0
        first_capture = None
        try:
            mx, my = sensor_maps()
            while not self.camera_stop.is_set():
                with urlopen(f"{self.url}/fpv/{session}/{after}", timeout=3) as response:
                    data = response.read()
                if not data:
                    continue
                t, after = struct.unpack_from("<dI", data)
                image = cv2.imdecode(np.frombuffer(data, np.uint8, offset=12), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(cv2.remap(image, mx, my, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                with self.frame_lock:
                    self.received += 1
                    if first_capture is None:
                        first_capture = t
                    elif t > first_capture:
                        self.sim_rx_fps = (self.received - 1) / (t - first_capture)
                    self.frames.append((t, image, after))
        except Exception as exc:
            if not self.camera_stop.is_set():
                self.failure = exc

    def step(self, rc):
        if self.failure:
            raise RuntimeError("Camera stream failed") from self.failure
        # Request a fresh command at 60 Hz when there is headroom; expensive policy
        # steps use their actual elapsed time. Never queue overdue commands.
        now = time.monotonic()
        self.work_period = 0.85 * self.work_period + 0.15 * (now - self.sent)
        period = np.clip(self.work_period * 1.15, 1 / 60, 1 / 40)
        time.sleep(max(0.0, self.sent + period - now))
        self.sent = time.monotonic()
        data = json.loads(self._post("/step", {"rc": rc, "control_only": True}))
        with self.frame_lock:
            frame = next((f for f in reversed(self.frames) if f[0] <= data["sim_time"]), None)
        if frame is not None:
            self.current_frame = frame
        t, rgb, count = self.current_frame
        return Sample(
            rgb,
            np.asarray(data["xyz"]),
            data["sim_time"],
            data["armed"],
            np.asarray(data["quaternion"]),
            data["applied"],
            t,
            count,
            self.sent,
        )

    def _stop_camera(self):
        if self.reader is not None:
            self.camera_stop.set()
            self.reader.join(4)
            self.reader = None

    def disarm(self):
        try:
            super().disarm()
        finally:
            self._stop_camera()


@lru_cache(maxsize=1)
def sensor_maps():
    """Sample Gazebo's wide pinhole image using the calibrated lens projection."""
    y, x = np.indices((HEIGHT, WIDTH), np.float32)
    dx, dy = x - CX, y - CY
    rd = np.hypot(dx, dy) / FOCAL
    ru = rd.copy()
    for _ in range(8):
        ru -= (ru * (1 + K1 * ru**2) - rd) / (1 + 3 * K1 * ru**2)
    scale = EDGE_SCALE / (1 + K1 * ru**2)
    return CX + dx * scale, CY + dy * scale
