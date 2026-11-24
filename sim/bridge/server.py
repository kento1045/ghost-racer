"""Local simulation API. One controller drives one Gazebo world at a time."""

import json
import re
import faulthandler
import signal
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from simulation import Simulation
from circuit import course_for, settings


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # A frame request every 100 ms would obscure the simulator logs.

    def reply(self, status, body, mime="application/json"):
        payload = json.dumps(body, allow_nan=False).encode() if mime == "application/json" else body
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A health probe or viewer may have already disconnected.

    def do_GET(self):
        sim = self.server.sim
        camera = re.fullmatch(r"/fpv/(\d{1,10})/(\d{1,12})", self.path)
        if camera:
            frame = sim.camera_packet(int(camera[1]), int(camera[2]))
            return self.reply(200, frame, "application/octet-stream") if frame else self.reply(204, b"", "application/octet-stream")
        if self.path == "/":
            self.reply(200, Path(__file__).with_name("viewer.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/course":
            self.reply(200, course_for(sim.cfg))
        elif self.path == "/health":
            ready = sim.ready and sim.children.error() is None
            self.reply(200 if ready else 503, {"ready": ready, "session": sim.session})
        elif self.path == "/sample" and sim.sample:
            self.reply(200, sim.snapshot())
        else:
            self.reply(503 if self.path == "/sample" else 404, {"error": "Unavailable"})

    def do_POST(self):
        sim = self.server.sim
        if self.path not in ("/step", "/reset", "/disarm"):
            self.reply(404, {"error": "Unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON object of at most 32768 bytes")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if self.path == "/reset":
                cfg = settings(body.get("settings"))
            if self.path == "/step":
                rc = body.get("rc")
                if not isinstance(rc, list) or len(rc) != 4 or not all(
                        type(v) is int and 1000 <= v <= 2000 for v in rc):
                    raise ValueError("rc requires four integer AETR channels from 1000 to 2000")
        except (ValueError, UnicodeError) as error:
            self.reply(400, {"error": str(error)})
            return
        if not sim.operation.acquire(blocking=False):
            self.reply(409, {"error": "The simulator is processing another command"})
            return
        try:
            owner = self.headers.get("X-Controller", "")
            if sim.radio.state()[1] and sim.armed and getattr(sim, "owner", "") != owner:
                self.reply(409, {"error": "Another controller is flying this episode"})
                return
            if self.path == "/reset":
                sim.owner = owner
            if self.path == "/disarm":
                sim.disarm()
                result = {"armed": False}
            else:
                if self.path == "/step" and body.get("control_only"):
                    return self.reply(200, sim.control_step(rc))
                sample = sim.reset(cfg) if self.path == "/reset" else sim.step(rc)
                result = sim.snapshot(sample)
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
            sim.disarm()
            print(f"{self.path} failed: {error}", flush=True)
            self.reply(503, {"error": str(error)})
            return
        finally:
            sim.operation.release()
        self.reply(200, result)


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)
    print("Opening Gazebo transport and the SITL radio", flush=True)
    sim = Simulation()
    server = ThreadingHTTPServer(("0.0.0.0", 8765), Handler)
    server.sim = sim
    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        print("Starting the simulation", flush=True)
        sim.launch()
        print("FPV viewer and learning API ready on port 8765", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sim.close()
