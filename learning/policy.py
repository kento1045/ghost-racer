"""Camera features and nominal ACRO inputs for the PPO residual policy."""
import cv2
import numpy as np
from gymnasium import Wrapper, spaces
from .localization import VisualPose
from concurrent.futures import ThreadPoolExecutor
import time
from .client import FlightBridge
from .navigation import (
    GRAVITY,
    HOVER,
    RATE,
    State,
    Predictor,
    RacingLine,
    quaternion_matrix,
    rotation_vector,
)


def save_detection_frame(path, rgb, course):
    pose = VisualPose(course)
    ok = pose.update(rgb)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i, corners in pose.detected.items():
        points = corners.round().astype(np.int32)
        cv2.polylines(bgr, [points], True, (40, 255, 70), 1)
        for x, y in points:
            cv2.circle(bgr, (x, y), 3, (40, 255, 70), -1)
        cv2.putText(bgr, str(i + 1), tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 70), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(path)
    return ok


class RaceObservation(Wrapper):
    """Teacher and camera policy share state features and exactly the same controller."""

    def __init__(self, env, camera=None):
        super().__init__(env)
        self.camera = camera
        self.course = env.course
        self.teacher = False
        self.pace = 1.5
        self.reference = None
        self.augment = False
        self.dropout = 0.0
        self.executor = None
        self.observation_space = spaces.Box(-np.inf, np.inf, (25,), np.float32)

    def use_reference(self, reference):
        self.reference = reference
        self.observation_space = spaces.Box(-np.inf, np.inf, (38,), np.float32)

    def reset(self, **kwargs):
        self._finish_worker()
        rgb, info = self.env.reset(**kwargs)
        self.pose = VisualPose(self.course)
        self.line = RacingLine(self.course, crossing_tolerance=0.0 if self.teacher else 0.15)
        self.start = self.env.sample.sim_time
        self.predictor = Predictor(self.course, self.env.cfg["mass"], self.start)
        self.pose_bias = np.zeros(3)
        if self.augment:
            rng = self.env.np_random
            self.dropout = float(rng.choice([0.0, 0.20, 0.35, 0.50]))
            self.pose_bias = rng.uniform(-0.10, 0.10, 3)
            self.predictor.state.pos += self.pose_bias
        self.truth = self.predictor.state
        self.previous_action = np.array([0.0, 0.0, -1.0, 0.0])
        self.last_action = self.previous_action.copy()
        self.previous_time = self.start
        self.last_submitted = -np.inf
        self.pending = None
        self.executor = (
            ThreadPoolExecutor(max_workers=1) if isinstance(self.env.bridge, FlightBridge) else None
        )
        self.frames = self.visible = 0
        self.mask_start = None
        self.masked_gate = -1
        self.last_view = (self.start, rgb, False, {})
        self.mask_active = False
        self.first_wall = time.monotonic()
        self.commands = self.masked_frames = self.late_commands = 0
        self.first_sent = self.last_sent = None
        self.first_camera = None
        self.fresh_camera = 0
        self.latencies = []
        return self.observe(rgb, reset=True), info

    def _finish_worker(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def close(self):
        try:
            self.env.close()
        finally:
            self._finish_worker()

    def _detect(self, t, rgb, masked, position, orientation):
        # A deliberately missing frame must not seed the next PnP solve.
        if masked:
            return t, rgb, False, {}, position, orientation, True
        self.pose.xyz = position
        self.pose.body = orientation
        valid = self.pose.update(rgb, predicted_body=orientation)
        return (
            t,
            rgb,
            valid,
            self.pose.detected.copy(),
            self.pose.xyz.copy(),
            self.pose.body.copy(),
            False,
        )

    def _mask_image(self, exposure):
        target = self.line.gates[self.line.index % len(self.line.gates)]
        active = self.mask_start is not None and exposure - self.mask_start < self.dropout
        if self.dropout and not active and self.masked_gate != self.line.index:
            if np.linalg.norm(self.predictor.state.pos - target["center"]) < 0.65:
                self.mask_start = exposure
                self.masked_gate = self.line.index
                active = True
        return active

    def _course_stick(self, state, error, tangent):
        pace = self.pace
        if self.line.index == 0:
            pace *= np.clip((state.pos[2] - 0.12) / 0.5, 0, 1)
        acceleration = 4.5 * error + 3.0 * (pace * tangent - state.vel)
        acceleration[2] += GRAVITY
        direction = state.body.T @ acceleration
        yaw = np.arctan2(state.body[1, 0], state.body[0, 0])
        delta = np.arctan2(tangent[1], tangent[0]) - yaw
        desired = np.array(
            [
                -np.arctan2(direction[1], direction[2]),
                np.arctan2(direction[0], direction[2]),
                np.arctan2(np.sin(delta), np.cos(delta)),
            ]
        )
        stick = (5.0 * desired - 0.10 * state.omega) / RATE
        thrust = max(0.0, acceleration[2]) / max(0.35, state.body[2, 2])
        throttle = 2 * HOVER * np.sqrt(thrust * self.env.cfg["mass"] / (26 * GRAVITY)) - 1
        return np.clip([stick[0], stick[1], throttle, -stick[2]], -0.98, 0.98)

    def observe(self, rgb, reset=False):
        sample = self.env.sample
        now = sample.sim_time
        t = sample.image_time if sample.image_time is not None else now
        dt = now - self.previous_time
        if not reset:
            applied = np.clip(
                sample.applied if sample.applied is not None else self.previous_time,
                self.previous_time,
                now,
            )
            self.predictor.advance(applied, self.previous_action)
            self.predictor.advance(now, self.last_action)
        if self.teacher:
            if sample.quaternion is None:
                raise ValueError("Teacher requires simulator attitude")
            body = quaternion_matrix(sample.quaternion)
            vel = np.zeros(3) if reset else (sample.xyz - self.truth.pos) / dt
            rate = np.zeros(3) if reset else rotation_vector(self.truth.body.T @ body) / dt
            self.truth = State(sample.xyz.copy(), vel, body, rate, 0.0)
        result = None
        if self.pending is not None and self.pending.done():
            result = self.pending.result()
            self.pending = None
        if t > self.last_submitted and self.pending is None:
            self.last_submitted = t
            prior = self.predictor.state_at(t) or self.predictor.state
            args = (t, rgb, self._mask_image(t), prior.pos.copy(), prior.body.copy())
            if self.executor and not reset:
                self.pending = self.executor.submit(self._detect, *args)
            else:
                result = self._detect(*args)
        if result is not None:
            exposure, image, valid, gates, pos, body, masked = result
            valid = valid and self.predictor.correct(exposure, pos + self.pose_bias, body)
            self.mask_active = masked
            self.masked_frames += int(masked)
            self.frames += 1
            self.visible += int(valid)
            self.last_view = (exposure, image, valid, {} if masked else gates)
        exposure, image, valid, gates = self.last_view
        state = self.truth if self.teacher else self.predictor.state
        self.line.update(state.pos, self.teacher or (result is not None and valid))
        nearest, lead, tangent = self.line.reference(state)
        local = state.body.T
        gate = self.line.gates[self.line.index % len(self.line.gates)]
        error = nearest - state.pos
        target = lead - state.pos
        obs = np.r_[
            local @ (gate["center"] - state.pos),
            local @ target,
            local @ error,
            local @ tangent,
            local @ state.vel,
            state.body[2],
            state.omega / RATE,
            self.last_action,
        ]
        if self.reference is not None:
            point = self.reference.point(self.line.index, state.pos)
            pos_error = local @ (point.position - state.pos)
            vel_error = local @ (point.velocity - state.vel)
            angle_error = rotation_vector(local @ point.attitude)
            # Feed forward the recorded stick. Correct lateral drift by banking,
            # vertical drift with thrust, and attitude error with body rates.
            acc = 4.5 * pos_error + 3.0 * vel_error
            rate = 5.0 * (angle_error + np.array([-acc[1], acc[0], 0]) / GRAVITY) / RATE
            feedback = np.array([rate[0], rate[1], HOVER * acc[2] / GRAVITY, -rate[2]])
            self.nominal = np.clip(point.stick + feedback, -0.98, 0.98)
            obs = np.r_[obs, pos_error, vel_error, angle_error, point.stick]
        else:
            self.nominal = self._course_stick(state, error, tangent)
        if self.camera is not None:
            accepted = self.predictor.accepted
            self.camera.capture(
                image,
                dict(
                    t=now - self.start,
                    exposure=exposure - self.start,
                    masked=self.mask_active,
                    target=self.line.index % len(self.line.gates) + 1,
                    pnp=valid,
                    age=now - (accepted[0] if accepted else self.start),
                    position=self.predictor.state.pos.tolist(),
                    confirmed_gates=self.line.confirmed,
                    predicted_gates=self.line.index,
                    aetr=self.last_action.tolist(),
                    gates={str(i + 1): uv.tolist() for i, uv in gates.items()},
                ),
                reset=reset,
            )
        self.previous_time = now
        return obs.astype(np.float32)

    def step(self, action):
        self.previous_action = self.last_action
        self.last_action = np.tanh(
            np.arctanh(self.nominal) + np.arctanh(np.clip(action, -0.9999, 0.9999))
        )
        self.last_action = np.rint((1 + self.last_action) * 500) / 500 - 1
        used_exposure = self.last_view[0]
        prev_time = self.previous_time
        rgb, r, term, trunc, info = self.env.step(self.last_action.astype(np.float32))
        if self.reference is not None:
            distance = self.reference.distance(info["gates_passed"], self.env.sample.xyz)
            r -= distance * (self.env.sample.sim_time - prev_time)
            info["tracking_error_m"] = distance
        obs = self.observe(rgb)
        self.commands += 1
        sample = self.env.sample
        if sample.image_count:
            if self.first_camera is None:
                self.first_camera = (sample.image_time, sample.image_count)
            self.fresh_camera = sample.image_count - self.first_camera[1]
            self.latencies.append(
                max(
                    0.0,
                    (sample.applied if sample.applied is not None else sample.sim_time) - used_exposure,
                )
            )
        sent = sample.sent_wall or time.monotonic()
        if self.first_sent is None:
            self.first_sent = sent
        if self.last_sent is not None:
            self.late_commands += int(sent - self.last_sent > 1 / 40)
        self.last_sent = sent
        wall = sent - self.first_sent
        bridge = self.env.bridge
        info.update(
            pose_valid_fraction=self.visible / max(1, self.frames),
            control_hz=round((self.commands - 1) / max(0.001, wall), 2),
            state_source="teacher" if self.teacher else "camera",
            dropout_seconds=self.dropout,
            pose_bias_m=self.pose_bias.tolist(),
            masked_frames=self.masked_frames,
            mask_active=self.mask_active,
            estimate_error_m=float(np.linalg.norm(self.predictor.state.pos - sample.xyz)),
            late_commands=self.late_commands,
            received_sim_fps=round(getattr(bridge, "sim_rx_fps", 0.0), 2),
            received_fps=round(
                getattr(bridge, "received", 0)
                / max(0.001, time.monotonic() - getattr(bridge, "receiver_start", self.first_wall)),
                2,
            ),
            camera_fps=(
                self.fresh_camera / max(0.001, sample.image_time - self.first_camera[0])
                if self.first_camera
                else 0.0
            ),
        )
        if term or trunc:
            info["vision_age_p95_ms"] = (
                float(1000 * np.percentile(self.latencies, 95)) if self.latencies else 0.0
            )
            self._finish_worker()
        return obs, r, term, trunc, info
