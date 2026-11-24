"""FLU rigid-body prediction and course-local guidance for a closed racing circuit."""
from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np
from .env import gate_crossing

GRAVITY = 9.80665
RATE = np.deg2rad(300.0)
# Measured from true pose and applied AETR during a 12 s, 26 g SITL flight.
HOVER = 0.548
RATE_LAG = np.array([0.08, 0.08, 0.18])
THRUST_LAG = 0.04


def rotation(vector):
    return cv2.Rodrigues(np.asarray(vector, dtype=float))[0]


def rotation_vector(matrix):
    return cv2.Rodrigues(matrix)[0].ravel()


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q) / np.linalg.norm(q)
    v = np.array([x, y, z])
    return np.eye(3) + 2 * (
        np.outer(v, v) - v @ v * np.eye(3) + w * np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    )


@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray
    body: np.ndarray
    omega: np.ndarray
    thrust: float


def integrate(s, cmd, dt, mass):
    # Body angular rates and rotor thrust settle rather than changing instantly.
    target_rate = cmd[[0, 1, 3]] * [RATE, RATE, -RATE]
    omega = s.omega + (-np.expm1(-dt / RATE_LAG)) * (target_rate - s.omega)
    body = s.body @ rotation((omega + s.omega) * (0.5 * dt))
    target_thrust = GRAVITY * 26 / mass * ((cmd[2] + 1) / (2 * HOVER)) ** 2
    thrust = s.thrust + (-np.expm1(-dt / THRUST_LAG)) * (target_thrust - s.thrust)
    acc = (
        0.5 * (body[:, 2] * thrust + s.body[:, 2] * s.thrust)
        - [0, 0, GRAVITY]
        - (1.5 / mass) * np.linalg.norm(s.vel) * s.vel
    )
    pos = s.pos + s.vel * dt + 0.5 * acc * dt**2
    vel = s.vel + acc * dt
    if pos[2] < 0.014 and vel[2] < 0:
        pos[2], vel[2] = 0.014, 0.0
    return State(pos, vel, body, omega, thrust)


class Predictor:
    """Keep command intervals so a delayed pose update can be applied at exposure time."""

    def __init__(self, course, mass, t):
        self.mass = mass
        self.state = State(
            np.array(course["spawn"]),
            np.zeros(3),
            rotation([0, 0, course["spawn_yaw"]]),
            np.zeros(3),
            0.0,
        )
        self.t = t
        self.accepted = None
        self.start = t
        self.history = deque()

    def advance(self, end, cmd):
        while end - self.t > 1e-8:
            stop = min(end, self.t + 0.025)
            self.history.append((self.t, stop, cmd, self.state))
            self.state = integrate(self.state, cmd, stop - self.t, self.mass)
            self.t = stop
        while self.history and self.history[0][1] < end - 1.0:
            self.history.popleft()

    def state_at(self, t):
        if abs(t - self.t) <= 1e-7:
            return self.state
        for start, end, cmd, prior in reversed(self.history):
            if start <= t <= end:
                return integrate(prior, cmd, t - start, self.mass)
        return None

    def correct(self, t, xyz, body):
        if self.accepted and t <= self.accepted[0]:
            return False
        prior = self.state_at(t)
        if prior is None:
            return False
        # Near-frontal planar observations are noisy in depth. A correction
        # is weighted by elapsed time so faster detection does not amplify jitter.
        xyz = np.array(xyz, dtype=float, copy=True)
        xyz[2] = max(0.014, xyz[2])
        delta = xyz - prior.pos
        gap = t - (self.accepted[0] if self.accepted else self.start)
        angular = rotation_vector(prior.body.T @ body)
        if np.linalg.norm(delta) > 0.45 + 2.5 * gap or np.linalg.norm(angular) > 0.55 + gap:
            return False
        decay = np.exp(-gap / 0.18)
        position_gain = 1 - decay**2
        velocity_gain = (1 - decay) ** 2 / max(0.001, gap)
        vel = prior.vel + velocity_gain * delta
        attitude_gain = -np.expm1(-gap / 0.10)
        events = [event for event in self.history if event[1] > t]
        self.history.clear()
        self.t = t
        self.state = State(
            prior.pos + position_gain * delta,
            vel,
            prior.body @ rotation(attitude_gain * angular),
            prior.omega,
            prior.thrust,
        )
        self.accepted = (t, xyz.copy())
        for _, end, cmd, _ in events:
            self.advance(end, cmd)
        return True


class RacingLine:
    def __init__(self, course, crossing_tolerance=0.15):
        self.course = course
        self.crossing_tolerance = crossing_tolerance
        self.approached = False
        self.path = np.asarray(course["path"])
        self.gates = course["gates"]
        self.samples = len(self.path) // len(self.gates)
        self.index = 0
        self.confirmed = 0
        self.previous = np.asarray(course["spawn"])

    def update(self, pos, observed):
        gate = self.gates[self.index % len(self.gates)]
        _, cleared = gate_crossing(
            self.previous,
            pos,
            gate,
            np.asarray(gate["opening"]) + 2 * self.crossing_tolerance,
            self.course["vehicle_radius"],
        )
        if self.crossing_tolerance:
            center = np.asarray(gate["center"])
            normal = np.array([np.cos(gate["yaw"]), np.sin(gate["yaw"]), 0.0])
            offset = pos - center
            depth = offset @ normal
            side = abs(offset[:2] @ [-normal[1], normal[0]])
            if (
                -0.9 < depth < 0
                and side < gate["opening"][0] / 2
                and abs(offset[2]) < gate["opening"][1] / 2
            ):
                self.approached = True
            # A delayed correction can put the estimated plane intersection
            # outside the opening. Reacquire progress only on the mapped exit
            # of the gate just approached. Physical scoring remains independent.
            if observed and self.approached and depth > 0.2 and not cleared:
                start = self.index * self.samples
                exit_path = self.path[np.arange(start, start + self.samples + 1) % len(self.path)]
                cleared = np.min(np.linalg.norm(exit_path - pos, axis=1)) < 0.45
        if cleared:
            self.index += 1
            self.approached = False
        if observed and self.index > self.confirmed:
            last = self.gates[(self.index - 1) % len(self.gates)]
            n = np.array([np.cos(last["yaw"]), np.sin(last["yaw"]), 0.0])
            depth = (pos - last["center"]) @ n
            if depth < -0.12:
                self.index = self.confirmed
            elif depth > 0.05:
                self.confirmed = self.index
        self.previous = pos.copy()

    def reference(self, state):
        gate = self.gates[self.index % len(self.gates)]
        center = np.array(gate["center"])
        if self.index == 0:
            tangent = np.array([np.cos(gate["yaw"]), np.sin(gate["yaw"]), 0.0])
            nearest = center + min(0.0, (state.pos - center) @ tangent) * tangent
            return nearest, nearest + 0.55 * tangent, tangent
        # Search only the current segment, even where the hairpin doubles back.
        start = (self.index - 1) * self.samples
        points = self.path[np.arange(start, start + 2 * self.samples + 1) % len(self.path)]
        a = points[: self.samples]
        edges = points[1 : self.samples + 1] - a
        u = np.clip(np.sum((state.pos - a) * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
        projections = a + u[:, None] * edges
        j = np.argmin(np.linalg.norm(state.pos - projections, axis=1))
        nearest = projections[j]
        ahead = np.vstack([nearest, points[j + 1 :]])
        length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ahead, axis=0), axis=1))]
        look = 0.45 + 0.13 * np.linalg.norm(state.vel)
        lead = np.array([np.interp(look, length, col) for col in ahead.T])
        tangent = (lead - nearest) / max(1e-5, np.linalg.norm(lead - nearest))
        return nearest, lead, tangent
