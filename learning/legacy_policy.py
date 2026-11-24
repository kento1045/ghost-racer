"""Camera features and nominal ACRO inputs for the PPO residual policy."""
import cv2
import numpy as np
from gymnasium import Wrapper, spaces
from .localization import VisualPose
from sim.optics import CX, CY, WIDTH, HEIGHT


class LegacyRaceObservation(Wrapper):
    def __init__(self, env, camera=None):
        super().__init__(env)
        self.camera = camera
        self.course = env.course
        self.observation_space = spaces.Box(-np.inf, np.inf, (31,), np.float32)

    def reset(self, **kwargs):
        rgb, info = self.env.reset(**kwargs)
        self.pose = VisualPose(self.course)
        self.velocity = np.zeros(3)
        self.last_action = np.zeros(4)
        self.index = 0
        self.age = 0.0
        self.visible = 0
        self.frames = 0
        self.elapsed = 0.0
        return self.observe(rgb, 0.067), info

    def observe(self, rgb, dt):
        prev = self.pose.xyz.copy()
        valid = self.pose.update(rgb)
        self.frames += 1
        if valid:
            interval = self.age + dt
            self.velocity += (interval / (interval + 0.24)) * (
                (self.pose.xyz - prev) / interval - self.velocity
            )
            self.visible += 1
            self.age = 0.0
        else:
            self.age += dt
        pos = self.pose.xyz + self.velocity * min(self.age, 0.35)
        body = self.pose.body
        gates = self.course["gates"]
        g = gates[self.index % len(gates)]
        n = np.array([np.cos(g["yaw"]), np.sin(g["yaw"])])
        if valid and (pos[:2] - g["center"][:2]) @ n >= 0:
            self.index += 1
        i = self.index % len(gates)
        target = np.asarray(gates[i]["center"]) - pos
        following = np.asarray(gates[(i + 1) % len(gates)]["center"]) - pos
        vel = body.T @ self.velocity
        uv = self.pose.detected.get(i)
        corners = np.zeros(8) if uv is None else ((uv - [CX, CY]) / [WIDTH / 2, HEIGHT / 2]).ravel()
        local = body.T @ target
        features = np.r_[
            local / 3,
            body.T @ following / 3,
            vel / 3,
            body[2],
            corners,
            float(valid),
            float(uv is not None),
            np.log1p(self.age),
            self.last_action,
            np.sin(gates[i]["yaw"]),
            np.cos(gates[i]["yaw"]),
            body[0, 0],
            body[1, 0],
        ]
        # Apply forward acceleration on approach. Lateral correction aligns the
        # craft with the opening, and PPO can override any nominal stick input.
        heading = np.arctan2(target[1], target[0])
        yaw = np.arctan2(body[1, 0], body[0, 0])
        err = np.arctan2(np.sin(heading - yaw), np.cos(heading - yaw))
        roll = np.arctan2(body[2, 1], body[2, 2])
        pitch = np.arcsin(np.clip(-body[2, 0], -1, 1))
        bank = np.arctan2(-1.8 * local[1] + 1.6 * vel[1], 9.81)
        tilt = 0.28 * np.cos(err) * np.clip((pos[2] - 0.10) / 0.50, 0, 1)
        self.nominal = np.tanh(
            [
                0.65 * (bank - roll),
                0.65 * (tilt - pitch),
                0.1 + 0.55 * target[2] - 0.30 * self.velocity[2],
                -0.65 * err,
            ]
        )
        if self.camera is not None:
            self.camera.capture(
                rgb,
                {
                    "t": self.elapsed,
                    "target": i + 1,
                    "pnp": bool(valid),
                    "age": self.age,
                    "position": pos.tolist(),
                    "aetr": self.last_action.tolist(),
                    "gates": {str(g + 1): corners.tolist() for g, corners in self.pose.detected.items()},
                },
                reset=self.frames == 1,
            )
        return features.astype(np.float32)

    def step(self, action):
        self.last_action = np.tanh(
            np.arctanh(self.nominal * 0.98) + np.arctanh(np.clip(action, -0.9999, 0.9999))
        )
        rgb, r, term, trunc, info = self.env.step(self.last_action.astype(np.float32))
        dt = info["elapsed_seconds"] - self.elapsed
        self.elapsed = info["elapsed_seconds"]
        obs = self.observe(rgb, dt)
        info["pose_valid_fraction"] = self.visible / self.frames
        return obs, r, term, trunc, info


