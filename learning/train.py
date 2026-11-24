"""On-policy PPO training and deterministic evaluation against Gazebo + Betaflight."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .client import Bridge
from .legacy_policy import LegacyRaceObservation
from .env import GateEnv
from .policy import RaceObservation, save_detection_frame
from .reference import ReferenceTrajectory


class Flight:
    def __init__(self, spawn):
        self.trace = [[0.0, *spawn, 0]]
        self.score = self.distance = self.peak = 0.0
        self.start_line = None
        self.lap_times = []
        self.position_errors = []
        self.masked_errors = []
        self.tracking_errors = []

    def append(self, reward, info):
        if "tracking_error_m" in info:
            self.tracking_errors.append(info["tracking_error_m"])
        if "estimate_error_m" in info:
            self.position_errors.append(info["estimate_error_m"])
            if info["mask_active"]:
                self.masked_errors.append(info["estimate_error_m"])
        t = info["elapsed_seconds"]
        xyz = info["xyz"]
        last = self.trace[-1]
        d = float(np.linalg.norm(np.subtract(xyz[:2], last[1:3])))
        self.distance += d
        self.peak = max(self.peak, d / max(0.001, t - last[0]))
        self.score += float(reward)
        self.trace.append([t, *xyz, info["gates_passed"]])
        if info["gates_passed"] and self.start_line is None:
            self.start_line = t
        if info["laps"] > len(self.lap_times):
            self.lap_times.append(round(t - self.start_line, 3))
            self.start_line = t

    def result(self, info):
        t = info["elapsed_seconds"]
        return dict(
            score=round(self.score, 4),
            trajectory=self.trace,
            laps=info["laps"],
            gates=info["gates_passed"],
            mean_speed=round(self.distance / t, 3),
            peak_speed=round(self.peak, 3),
            sim_seconds=round(t, 3),
            reason=info["reason"],
            lap_times=self.lap_times,
            position_error_p95_m=round(float(np.percentile(self.position_errors, 95)), 3)
            if self.position_errors
            else None,
            masked_error_p95_m=round(float(np.percentile(self.masked_errors, 95)), 3)
            if self.masked_errors
            else None,
            tracking_error_p95_m=round(float(np.percentile(self.tracking_errors, 95)), 3)
            if self.tracking_errors else None,
            pose_valid_fraction=round(info["pose_valid_fraction"], 3),
            **{
                key: info[key]
                for key in (
                    "state_source",
                    "dropout_seconds",
                    "pose_bias_m",
                    "masked_frames",
                    "camera_fps",
                    "received_fps",
                    "received_sim_fps",
                    "control_hz",
                    "late_commands",
                    "vision_age_p95_ms",
                )
                if key in info
            },
        )


class RecordEpisodes(BaseCallback):
    def __init__(self, spawn, record, progress):
        super().__init__()
        self.progress = progress
        self.partial_reason = "training_budget"
        self.spawn = spawn
        self.flight = Flight(spawn)
        self.record = record
        self.episode = 1
        self.last_info = None

    def _on_step(self):
        info = self.locals["infos"][0]
        self.progress("training", self.num_timesteps)
        self.last_info = info
        self.flight.append(self.locals["rewards"][0], info)
        if self.locals["dones"][0]:
            self.record(dict(episode=self.episode, step=self.num_timesteps, **self.flight.result(info)))
            self.flight = Flight(self.spawn)
            self.episode += 1
        return True

    def _on_training_end(self):
        if len(self.flight.trace) > 1:
            info = dict(self.last_info, reason=self.partial_reason)
            self.record(dict(episode=self.episode, step=self.num_timesteps, **self.flight.result(info)))
            self.flight = Flight(self.spawn)
            self.episode += 1


def rollout(env, model, frame=None, reference_rows=None):
    obs, info = env.reset()
    flight = Flight(info["xyz"])
    if frame:
        save_detection_frame(frame, env.env.render(), env.course)
    while True:
        if reference_rows is not None:
            start = env.env.sample
            velocity = env.truth.vel
            segment = env.env.gate_index
        action = model.predict(obs, deterministic=True)[0] if model else np.zeros(4, np.float32)
        obs, r, term, trunc, info = env.step(action)
        if reference_rows is not None:
            reference_rows.append([
                segment, start.sim_time - env.start, *start.xyz, *info["xyz"], *velocity,
                *start.quaternion, *env.last_action,
            ])
        flight.append(r, info)
        if term or trunc:
            return flight.result(info)


def select_reference(env, record, progress):
    """Keep the fastest mean five-lap time with two successful flights at that pace."""
    env.teacher = True
    best = None
    # The slower pace is a recovery candidate when none of the racing paces qualify.
    paces = (1.5, 1.85, 2.2, 1.2)
    for index, pace in enumerate(paces):
        if index == 3 and best is not None:
            break
        env.pace = pace
        runs = []
        for attempt in (1, 2):
            progress("reference", 0, candidate=index + 1, pace=pace, attempt=attempt)
            rows = []
            flight = rollout(env, None, reference_rows=rows)
            record(dict(evaluation=True, reference_pace=pace, attempt=attempt, **flight))
            if flight["reason"] != "finished" or flight["laps"] != 5:
                break
            runs.append((sum(flight["lap_times"]), rows))
        if len(runs) == 2:
            mean = (runs[0][0] + runs[1][0]) / 2
            if best is None or mean < best[0]["mean_seconds"]:
                fastest, rows = min(runs, key=lambda run: run[0])
                best = (dict(pace=pace, mean_seconds=mean, five_lap_seconds=fastest),
                        ReferenceTrajectory(rows, len(env.course["gates"])))
    if best is None:
        raise ValueError("真値飛行で5周を2回完走した候補がなく、基準軌道を作成できませんでした")
    return best


def restore_reference(env, model):
    if getattr(model, "race_version", None) == 3:
        env.use_reference(ReferenceTrajectory(model.reference_rows, len(env.course["gates"])))


def train(env, output, record, progress=lambda *args, **kwargs: None):
    cfg = env.env.cfg
    torch.set_num_threads(1)
    steps = cfg["rollout_steps"]
    reference_info, reference = select_reference(env, record, progress)
    env.use_reference(reference)
    env.pace = 1.5
    env.teacher = False
    batch = next((i for i in range(min(64, steps), 1, -1) if steps % i == 0), steps)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=steps,
        batch_size=batch,
        n_epochs=6,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",
        seed=cfg["seed"],
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-2.5),
    )
    model.race_version = 3
    model.reference_rows = reference.rows
    model.reference_info = reference_info
    model.race_settings = cfg
    callback = RecordEpisodes(env.course["spawn"], record, progress)
    maximum = cfg["steps"]
    interval = maximum if cfg["goal"] == "off" else steps * cfg["check_rollouts"]
    stop_reason = "step_limit"
    last_check = None
    while model.num_timesteps < maximum:
        budget = min(interval, maximum - model.num_timesteps)
        callback.partial_reason = (
            "training_budget" if model.num_timesteps + budget == maximum else "goal_check"
        )
        env.augment = True
        model.learn(total_timesteps=budget, callback=callback, reset_num_timesteps=False)
        env.augment = False
        env.dropout = 0.0
        if cfg["goal"] == "off":
            break
        successes = 0
        total = cfg["eval_flights"]
        for i in range(total):
            progress("evaluation", model.num_timesteps, completed=i, total=total, successes=successes)
            flight = rollout(env, model)
            passed = flight["gates"] >= 1 if cfg["goal"] == "gate" else flight["laps"] >= 5
            successes += int(passed)
            record(
                dict(
                    evaluation=True, goal_check=i + 1, step=model.num_timesteps, success=passed, **flight
                )
            )
        last_check = dict(step=model.num_timesteps, successes=successes, total=total)
        progress("evaluation", model.num_timesteps, completed=total, total=total, successes=successes)
        if successes >= cfg["required_successes"]:
            stop_reason = "goal_reached"
            break
        # Evaluation flew the same Gazebo instance. Discard the old training observation.
        if model.num_timesteps < maximum:
            model.set_env(env, force_reset=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.zip")
    model.save(temporary)
    temporary.replace(output)
    return dict(
        steps=model.num_timesteps,
        reference=reference_info,
        epochs=model._n_updates,
        stop_reason=stop_reason,
        last_check=last_check,
        approximate_kl=float(model.logger.name_to_value["train/approx_kl"]),
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("baseline", "train", "evaluate"))
    cli.add_argument("--url", default="http://127.0.0.1:8765")
    cli.add_argument("--steps", type=int, default=8192)
    cli.add_argument("--rollout-steps", type=int, default=128)
    cli.add_argument("--episodes", type=int, default=1)
    cli.add_argument("--model", "--output", type=Path, default=Path("runs/ppo.zip"))
    cli.add_argument("--frame", type=Path)
    cli.add_argument("--state-source", choices=("camera", "teacher"), default="camera")
    cli.add_argument("--dropout-seconds", type=float, choices=(0, 0.2, 0.35, 0.5), default=0.0)
    args = cli.parse_args()
    model = PPO.load(args.model, device="cpu") if args.command == "evaluate" else None
    config = model.race_settings if model else dict(steps=args.steps, rollout_steps=args.rollout_steps)
    if model and model.observation_space.shape == (31,):
        if args.state_source != "camera" or args.dropout_seconds:
            cli.error("Legacy models support the original camera observation only")
        env = LegacyRaceObservation(GateEnv(bridge=Bridge(args.url), config=config))
    else:
        env = RaceObservation(GateEnv(args.url, config=config))
        if model:
            restore_reference(env, model)
        env.teacher = args.state_source == "teacher"
        env.dropout = args.dropout_seconds

    def report(result):
        print(json.dumps({k: v for k, v in result.items() if k != "trajectory"}), flush=True)

    try:
        if args.command == "train":
            report(train(env, args.model, report))
        else:
            for _ in range(args.episodes):
                report(rollout(env, model, args.frame))
    finally:
        env.close()


if __name__ == "__main__":
    main()
