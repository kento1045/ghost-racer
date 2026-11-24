"""Gymnasium environment with FPV RGB observations and normalized AETR actions."""

import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.optics import HEIGHT, WIDTH
from .client import FlightBridge
from sim.circuit import course_for, settings


def gate_crossing(start, end, gate, opening, radius):
    """Return (intersection, cleared) for a forward plane crossing."""
    center = np.asarray(gate["center"])
    yaw = gate["yaw"]
    normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    before = (start - center) @ normal
    after = (end - center) @ normal
    if not before < 0 <= after:
        return None, False
    point = start + (end - start) * (-before / (after - before))
    offset = point - center
    lateral = offset @ np.array([-normal[1], normal[0], 0.0])
    cleared = abs(lateral) + radius < opening[0] / 2 and abs(offset[2]) + radius < opening[1] / 2
    return point, bool(cleared)


def touches_gate_frame(xyz, gate, opening, rail, radius):
    """Distance from the bounding sphere to the four rectangular rails in gate coordinates."""
    dx, dy, z = xyz - np.asarray(gate["center"])
    c, s = np.cos(gate["yaw"]), np.sin(gate["yaw"])
    depth, lateral = abs(dx * c + dy * s), abs(-dx * s + dy * c)
    width, height = np.asarray(opening) / 2
    r = rail / 2
    # Symmetry reduces the four rails to two nearest boxes.
    front = max(0.0, depth - r)
    post = (front, max(0.0, abs(lateral - width - r) - r), max(0.0, abs(z) - height - rail))
    bar = (front, max(0.0, lateral - width), max(0.0, abs(abs(z) - height - r) - r))
    return min(np.dot(post, post), np.dot(bar, bar)) <= radius**2


class GateEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        url="http://127.0.0.1:8765",
        *,
        bridge=None,
        config=None,
        max_seconds=None,
        render_mode=None,
    ):
        super().__init__()
        self.cfg = settings(config)
        max_seconds = self.cfg["seconds"] if max_seconds is None else max_seconds
        if render_mode not in (None, "rgb_array") or not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("Use render_mode=None or 'rgb_array' and a positive episode duration")
        self.render_mode = render_mode
        self.bridge = FlightBridge(url) if bridge is None else bridge
        self.course = course_for(self.cfg)
        self.bounds = np.asarray(self.course["bounds"])
        self.max_seconds = max_seconds
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.sample = None
        self.active = False

    def _observation(self):
        return self.sample.rgb

    def _info(self, reason):
        return {
            "gates_passed": self.gate_index,
            "laps": max(0, self.gate_index - 1) // len(self.course["gates"]),
            "elapsed_seconds": self.sample.sim_time - self.started,
            "reason": reason,
            "xyz": self.sample.xyz.tolist(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.active:
            self.close()
        self.active = True
        try:
            self.sample = self.bridge.reset(self.cfg)
        except BaseException:
            self._disarm_after_error()
            raise
        self.started = self.sample.sim_time
        self.still_since = self.started
        self.anchor = self.sample.xyz.copy()
        self.last_gate_time = self.started
        self.last_gate_center = self.sample.xyz.copy()
        self.gate_index = 0
        self.airborne = False
        return self._observation(), self._info("reset")

    def step(self, action):
        if not self.active:
            raise gym.error.ResetNeeded("Call reset() before the next step")
        aetr = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(aetr):
            raise ValueError("Expected four finite AETR stick positions in [-1, 1]")
        rc = np.rint(1500 + 500 * aetr).astype(int).tolist()
        previous = self.sample
        try:
            self.sample = self.bridge.step(rc)
            dt = self.sample.sim_time - previous.sim_time
            if dt <= 0:
                raise RuntimeError("The bridge returned a stale frame")
        except BaseException:
            self._disarm_after_error()
            raise
        xyz = self.sample.xyz
        gates = self.course["gates"]
        # Split progress at each crossed gate so that switching targets cannot cost reward.
        start = previous.xyz
        reward = -0.08 * dt
        reason = "flying"
        finish = 5 * len(gates) + 1
        while self.gate_index < finish:
            gate = gates[self.gate_index % len(gates)]
            center = np.asarray(gate["center"])
            point, cleared = gate_crossing(
                start, xyz, gate, gate["opening"], self.course["vehicle_radius"]
            )
            endpoint = xyz if point is None else point
            reward += np.linalg.norm(start - center) - np.linalg.norm(endpoint - center)
            if point is None:
                if touches_gate_frame(
                    xyz,
                    gate,
                    gate["opening"],
                    self.course["rail_thickness"],
                    self.course["vehicle_radius"],
                ):
                    reason = "gate_contact"
                break
            if not cleared:
                reason = "missed_gate"
                break
            fraction = np.linalg.norm(point - previous.xyz) / max(
                1e-8, np.linalg.norm(xyz - previous.xyz)
            )
            crossing_time = previous.sim_time + dt * fraction
            section_speed = np.linalg.norm(center - self.last_gate_center) / max(
                0.001, crossing_time - self.last_gate_time
            )
            reward += 5.0 + section_speed
            self.last_gate_time, self.last_gate_center = crossing_time, center
            start = point
            self.gate_index += 1
        if np.linalg.norm(xyz - self.anchor) > 0.12:
            self.still_since, self.anchor = self.sample.sim_time, xyz.copy()
        if self.gate_index == finish:
            reason = "finished"
        elif not self.sample.armed:
            reason = "fc_disarmed"
        elif (xyz[:2] < self.bounds[:2, 0]).any() or (xyz[:2] > self.bounds[:2, 1]).any():
            reason = "outside_course"
        elif self.airborne and xyz[2] <= self.course["spawn"][2] + 0.015:
            reason = "ground_contact"
        elif reason == "flying" and any(
            touches_gate_frame(
                xyz,
                other,
                other["opening"],
                self.course["rail_thickness"],
                self.course["vehicle_radius"],
            )
            for other in gates
        ):
            reason = "gate_contact"
        elif reason == "flying" and self.sample.sim_time - self.still_since > (
            3.0 if self.airborne else 6.0
        ):
            reason = "stalled"
        self.airborne |= xyz[2] > self.course["spawn"][2] + 0.10
        terminated = reason != "flying"
        truncated = not terminated and self.sample.sim_time - self.started >= self.max_seconds
        if truncated:
            reason = "time_limit"
        if reason in (
            "missed_gate",
            "gate_contact",
            "fc_disarmed",
            "outside_course",
            "ground_contact",
            "stalled",
        ):
            reward -= 3.0
        if terminated or truncated:
            self.close()
        return self._observation(), float(reward), terminated, truncated, self._info(reason)

    def render(self):
        return None if self.sample is None else self.sample.rgb.copy()

    def _disarm_after_error(self):
        try:
            self.close()
        except Exception as error:
            warnings.warn(f"Could not confirm disarm after bridge error: {error}", RuntimeWarning)

    def close(self):
        if self.active:
            try:
                self.bridge.disarm()
            finally:
                self.active = False
