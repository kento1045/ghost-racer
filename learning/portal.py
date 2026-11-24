"""Local training service. Run records survive browser disconnects and container restarts."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
from stable_baselines3 import PPO
from .policy import RaceObservation
from uuid import uuid4

from sim.circuit import DEFAULTS, LIMITS, course_for, settings
from .camera import FlightCamera, read_recording
from .env import GateEnv
from .train import restore_reference, rollout, train

ROOT = Path("runs/sessions")
UI = Path(__file__).with_name("portal.html")
BRIDGE = os.getenv("BRIDGE_URL", "http://127.0.0.1:8765")
LOCK = threading.Lock()


def save(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def save_trial(directory, record, camera, result):
    result = result.copy()
    index = len(record["trials"])
    save(directory / f"trace-{index}.json", result.pop("trajectory"))
    result["camera_frames"] = camera.finish(index)
    record["trials"].append(result)
    save(directory / "run.json", record)


def run_job(directory, record):
    cfg = record["settings"]
    env = None
    camera = FlightCamera(directory)

    last_write = 0.0

    def progress(phase, steps, **details):
        nonlocal last_write
        now = time.monotonic()
        previous = record.get("progress", {})
        if (
            phase == previous.get("phase") == "training"
            and steps < cfg["steps"]
            and now - last_write < 1
        ):
            return
        record["progress"] = dict(phase=phase, steps=steps, maximum=cfg["steps"], **details)
        save(directory / "progress.json", record["progress"])
        last_write = now

    def add_trial(result):
        save_trial(directory, record, camera, result)

    try:
        env = RaceObservation(GateEnv(BRIDGE, config=cfg), camera=camera)
        output = directory / "policy.zip"
        record["ppo"] = train(env, output, add_trial, progress)
        progress("final", record["ppo"]["steps"])
        model = PPO.load(output, device="cpu")
        restore_reference(env, model)
        conditions = [(True, 0.0), (False, 0.0), (False, 0.2), (False, 0.35), (False, 0.5)]
        for i, (env.teacher, env.dropout) in enumerate(conditions):
            progress("final", record["ppo"]["steps"], completed=i, total=len(conditions))
            add_trial(dict(evaluation=True, **rollout(env, model)))
        progress("final", record["ppo"]["steps"], completed=len(conditions), total=len(conditions))
        record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=str(error))
    finally:
        try:
            camera.close()
            if env:
                env.close()
        finally:
            try:
                save(directory / "run.json", record)
            finally:
                LOCK.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_data(self, status, obj, mime="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            return self.send_data(200, UI.read_bytes(), "text/html")
        if self.path == "/replay.js":
            return self.send_data(200, UI.with_name("replay.js").read_bytes(), "text/javascript")
        if self.path == "/api/settings":
            return self.send_data(
                200, dict(defaults=DEFAULTS, limits=LIMITS, course=course_for(settings()))
            )
        if self.path == "/api/runs":
            runs = []
            for f in sorted(ROOT.glob("*/run.json"), reverse=True):
                rec = read(f)
                if rec.get("algorithm") != "PPO":
                    continue
                summary = {k: rec[k] for k in ("id", "started", "status")}
                if rec["status"] == "running":
                    path = f.with_name("progress.json")
                    summary["progress"] = read(path) if path.exists() else rec.get("progress")
                runs.append(summary)
            return self.send_data(200, runs)
        camera = re.fullmatch(
            r"/api/runs/([0-9T-]+-[0-9a-f]{8})/camera-(\d{1,8})(?:/frame-(\d{1,8}))?", self.path
        )
        if camera:
            path = (ROOT / camera[1] / f"camera-{camera[2]}.sqlite").resolve()
            frame = int(camera[3]) if camera[3] is not None else None
            data = read_recording(path, frame)
            if data is None:
                return self.send_data(404, dict(error="この飛行のカメラ記録がありません"))
            return self.send_data(200, data, "image/jpeg" if frame is not None else "application/json")
        match = re.fullmatch(r"/api/runs/([0-9T-]+-[0-9a-f]{8})(?:/(policy|trace-\d+))?", self.path)
        if match:
            if match[2] == "policy":
                path = ROOT / match[1] / "policy.zip"
                if path.is_file():
                    return self.send_data(200, path.read_bytes(), "application/zip")
            path = ROOT / match[1] / ((match[2] or "run") + ".json")
            if path.is_file():
                if match[2] is None:
                    run = read(path)
                    progress_path = path.with_name("progress.json")
                    if run["status"] == "running" and progress_path.exists():
                        run["progress"] = read(progress_path)
                    return self.send_data(200, run)
                return self.send_data(200, path.read_bytes())
        self.send_data(404, dict(error="結果がありません"))

    def do_POST(self):
        if self.path != "/api/runs":
            return self.send_data(404, dict(error="Unknown endpoint"))
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if self.headers.get_content_type() != "application/json" or not 0 < size <= 32768:
                raise ValueError("JSON の設定が必要です")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict) or "course" in payload:
                raise ValueError("学習設定を指定してください")
            cfg = settings(payload)
        except (ValueError, UnicodeError) as error:
            return self.send_data(400, dict(error=str(error)))
        if not LOCK.acquire(blocking=False):
            return self.send_data(409, dict(error="学習中のジョブがあります"))
        try:
            now = datetime.now(timezone.utc)
            key = now.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
            directory = ROOT / key
            directory.mkdir(parents=True)
            rec = dict(
                id=key,
                started=now.isoformat(),
                status="running",
                settings=cfg,
                course=course_for(cfg),
                algorithm="PPO",
                progress=dict(phase="training", steps=0, maximum=cfg["steps"]),
                trials=[],
            )
            save(directory / "run.json", rec)
            threading.Thread(target=run_job, args=(directory, rec), daemon=True).start()
        except Exception:
            LOCK.release()
            raise
        self.send_data(202, dict(id=key))


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in ROOT.glob("*/run.json"):
        rec = read(path)
        if rec["status"] == "running":
            rec.update(status="interrupted", error="学習サービスが停止しました")
            save(path, rec)
    ThreadingHTTPServer(("0.0.0.0", 8766), Handler).serve_forever()
