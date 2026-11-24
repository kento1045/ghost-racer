"""Run an isolated Gazebo + Betaflight instance and pair FPV frames with model poses."""

import base64
import io
from PIL import Image as PILImage
from collections import deque
from pathlib import Path
import socket
import struct
import threading
import time

from gz.msgs10.image_pb2 import Image, RGB_INT8
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

from actuators import ActuatorLink
from processes import ProcessManager
from circuit import settings
from optics import lens_image
from radio import NEUTRAL, Radio

SIM = Path(__file__).resolve().parents[1]


def stamp(message):
    t = message.header.stamp
    return t.sec + t.nsec * 1e-9


class Simulation:
    def __init__(self):
        # Gazebo's native worker threads and fork can deadlock under Rosetta.
        # Start the isolated process launcher before creating the transport node.
        self.manager = ProcessManager()
        self.manager.start()
        self.children = self.manager.Children()
        self.operation = threading.Lock()
        self.cv = threading.Condition()
        self.poses = deque(maxlen=48)
        self.image_count = 0
        self.images = deque(maxlen=6)
        self.sample = None
        self.session = 0
        self.pwm_count = 0
        self.last_pwm = 0
        self.motors = ()
        self.armed = False
        self.cfg = settings()
        self.used = False
        self.ready = False
        self.link = ActuatorLink()
        self.radio = Radio()
        self.node = Node()
        self.node.subscribe(Image, "/whoop/camera", self._image)
        self.node.subscribe(Pose_V, "/world/race/dynamic_pose/info", self._poses)
        self.pwm = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pwm.bind(("127.0.0.2", 9001))
        threading.Thread(target=self._monitor_motors, daemon=True).start()

    def _monitor_motors(self):
        # SITL emits this raw mirror alongside each output to Gazebo on 9002.
        while True:
            pkt = self.pwm.recv(256)
            if len(pkt) == 68 and struct.unpack_from("<H", pkt)[0] == 4:
                with self.cv:
                    self.motors = struct.unpack_from("<4f", pkt, 4)
                    self.pwm_count += 1
                    self.last_pwm = time.monotonic()

    def _image(self, msg):
        if msg.pixel_format_type != RGB_INT8 or msg.step != 3 * msg.width:
            return
        with self.cv:
            self.image_count += 1
            self.images.append((stamp(msg), msg.width, msg.height, msg.data))
            self._pair()

    def _poses(self, msg):
        for pose in msg.pose:
            if pose.name == "whoop":
                with self.cv:
                    p = pose.position
                    q = pose.orientation
                    self.poses.append((stamp(msg), (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
                    self.cv.notify_all()
                    self._pair()
                return

    def _pair(self):
        if not self.images or not self.poses:
            return
        for t, width, height, rgb in reversed(self.images):
            if self.sample and t <= self.sample[0]:
                break
            pt, xyz, quat = min(self.poses, key=lambda item: abs(item[0] - t))
            if abs(pt - t) <= 0.04:
                self.sample = (t, xyz, width, height, rgb, quat)
                self.cv.notify_all()
                break

    def _check_processes(self):
        error = self.children.error()
        if error:
            raise RuntimeError(error)

    def _frame(self, after, timeout=15):
        end = time.monotonic() + timeout
        with self.cv:
            while self.sample is None or self.sample[0] < after:
                self._check_processes()
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for a fresh RGB image paired with a Gazebo pose"
                    )
                self.cv.wait(min(0.5, remaining))
            return self.sample

    def _stop(self):
        self.ready = self.armed = False
        self.link.enabled = False
        self.radio.disarm()
        self.children.stop()

    def launch(self):
        self._stop()
        # Full process reset clears physics velocities, rotor PIDs and FC integrators.
        with self.cv:
            self.poses.clear()
            self.images.clear()
            self.sample = None
            self.motors = ()
            self.pwm_count = 0
            self.image_count = 0
            self.session += 1
        print("Waiting for the Gazebo camera and pose", flush=True)
        self.children.start_world(self.cfg)
        self._frame(0, timeout=75)
        # Require a live FC and its raw PWM mirror before accepting any actions.
        for attempt in range(3):
            if self._connect_fc():
                self.ready = True
                return
            print(f"Retrying SITL startup after attempt {attempt + 1}", flush=True)
            self.children.stop_fc()
        raise RuntimeError(
            "SITL did not complete calibration and the physics handshake. See simulator logs."
        )

    def _connect_fc(self):
        print("Connecting the Betaflight flight controller", flush=True)
        self.pwm_count = 0
        self.link.last_motor = 0.0
        self.link.enabled = True
        self.children.start_fc()
        started = time.monotonic()
        while time.monotonic() - started < 12:
            self._check_processes()
            if self.pwm_count >= 10 and time.monotonic() - self.last_pwm < 0.25:
                return True
            time.sleep(0.05)
        print(f"SITL startup: received {self.pwm_count} PWM packets", flush=True)
        return False

    def _motor_active(self):
        return bool(self.motors and min(self.motors) > 1010 and time.monotonic() - self.last_pwm < 0.5)

    def reset(self, cfg=None):
        self.radio.disarm()
        cfg = settings(cfg)
        changed = self.cfg != cfg
        self.cfg = cfg
        if changed or self.used or not self.ready:
            self.launch()
        self.used = True
        # Disarmed outputs are 1000 us. With MOTOR_STOP off, all four motors
        # switch to the configured idle output only after the FC accepts ARM.
        for _ in range(10):
            self.radio.disarm()
            time.sleep(0.35)
            self.radio.command(NEUTRAL)
            deadline = time.monotonic() + 0.55
            while time.monotonic() < deadline:
                if self._motor_active():
                    self.armed = True
                    return self._frame(self.sample[0] + 0.04)
                time.sleep(0.025)
        raise RuntimeError(f"ARM did not produce four idle motor outputs: {self.motors}")

    def camera_packet(self, session, after):
        # The transport callback always takes the newest frame, even during PPO updates.
        with self.cv:
            available = self.cv.wait_for(
                lambda: self.session != session or (self.images and self.image_count > after), timeout=1
            )
            if not available or self.session != session:
                return None
            count = self.image_count
            t, width, height, rgb = self.images[-1]
        stream = io.BytesIO()
        stream.write(struct.pack("<dI", t, count))
        PILImage.frombytes("RGB", (width, height), rgb).save(stream, format="JPEG", quality=90)
        return stream.getvalue()

    def control_step(self, rc):
        if not self.radio.state()[1] or not self.armed:
            raise RuntimeError("Reset before sending a control step")
        with self.cv:
            applied = self.poses[-1][0]
            self.radio.command(rc)
            if not self.cv.wait_for(lambda: self.poses and self.poses[-1][0] > applied, timeout=2):
                raise TimeoutError("Gazebo pose clock stalled")
            stamp, xyz, quat = self.poses[-1]
        if time.monotonic() - self.last_pwm > 0.5:
            raise RuntimeError("SITL motor telemetry timed out")
        if self.motors and max(self.motors) <= 1010:
            self.disarm()
        return dict(
            sim_time=stamp,
            xyz=xyz,
            quaternion=quat,
            applied=applied,
            armed=self.armed,
            session=self.session,
        )

    def step(self, rc):
        if not self.radio.state()[1] or not self.armed:
            raise RuntimeError("Episode inactive. Reset before sending a step.")
        self.radio.command(rc)
        sample = self._frame(self.sample[0] + 0.055)
        if time.monotonic() - self.last_pwm > 0.5:
            raise RuntimeError("SITL motor telemetry timed out")
        if self.motors and max(self.motors) <= 1010:
            self.disarm()
        return sample

    def disarm(self):
        self.radio.disarm()
        self.armed = False

    def close(self):
        self._stop()
        self.radio.stop.set()
        self.link.stop.set()
        self.manager.shutdown()

    def snapshot(self, sample=None):
        sample = self.sample if sample is None else sample
        if sample is None:
            raise RuntimeError("Gazebo camera is starting")
        t, xyz, width, height, rgb, quat = sample
        rc, arm_switch = self.radio.state()
        return dict(
            sim_time=t,
            xyz=xyz,
            quaternion=quat,
            width=width,
            height=height,
            rgb=base64.b64encode(lens_image(rgb)).decode("ascii"),
            rc=rc,
            armed=self.armed and arm_switch,
            motors=self.motors,
            session=self.session,
        )
