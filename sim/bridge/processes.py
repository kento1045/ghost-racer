"""Keep subprocess creation outside the threaded Gazebo Transport process."""

from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import sys

SIM = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(SIM))
from circuit import course_for
from build_course import build_world
from marker_mesh import marker_mesh


class Children:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ghost-racer-")
        self.binary = os.environ.get("SITL_BINARY", "betaflight-sitl")
        self.processes = []

    def start_world(self, cfg):
        if not Path(self.directory.name, "eeprom.bin").exists():
            config = subprocess.run(
                [self.binary, "--config", str(SIM / "bridge/sitl.cli")],
                cwd=self.directory.name,
                timeout=45,
                capture_output=True,
                text=True,
            )
            if config.returncode or "[CONFIG] Config file processed, EEPROM saved" not in config.stdout:
                raise RuntimeError(
                    f"SITL configuration failed: {config.stdout[-1500:]} {config.stderr[-500:]}"
                )
        world = Path(self.directory.name, "race.sdf")
        course = course_for(cfg)
        (world.parent / "white.obj").write_text(marker_mesh(course))
        world.write_text(build_world(course, cfg["mass"], world.parent))
        self._spawn(["gz", "sim", "-s", "-r", "--headless-rendering", str(world)])

    def start_fc(self):
        # SITL shares its virtual flight clock across the sensor and scheduler threads.
        # Keep these on one core. Gazebo and image rendering use the remaining VM cores.
        cpu = min(os.sched_getaffinity(0))
        command = shlex.join(["taskset", "-c", str(cpu), self.binary, "--ip", "127.0.0.2"])
        self._spawn(["script", "-qefc", "exec " + command, "/dev/null"])

    def stop_fc(self):
        self._terminate(self.processes.pop())

    def _spawn(self, args):
        self.processes.append(subprocess.Popen(args, cwd=self.directory.name, start_new_session=True))

    def error(self):
        for p in self.processes:
            if p.poll() is not None:
                return f"{p.args[0]} exited with code {p.returncode}. See docker compose logs."
        return None

    @staticmethod
    def _terminate(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def stop(self):
        for proc in reversed(self.processes):
            self._terminate(proc)
        self.processes.clear()


class ProcessManager(BaseManager):
    pass


ProcessManager.register("Children", Children)
