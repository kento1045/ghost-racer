"""Exercise the real local camera → PC → SITL → motors → physics loop."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.client import Bridge
from learning.policy import save_detection_frame
from sim.circuit import settings, course_for


def main():
    bridge = Bridge()
    try:
        first = bridge.reset()
        if not save_detection_frame(Path("runs/smoke-gate.png"), first.rgb, course_for(settings())):
            raise RuntimeError("Could not recover gate corners and camera pose")
        print(f"Confirmed gate PnP in FPV {first.rgb.shape}", flush=True)
        sample = first
        while sample.sim_time - first.sim_time < 3:
            previous_time = sample.sim_time
            sample = bridge.step([1500, 1500, 1750, 1500])
            if sample.sim_time <= previous_time:
                raise RuntimeError("The camera stopped advancing")
            lift = sample.xyz[2] - first.xyz[2]
            if lift > 0.08:
                print(f"Confirmed RC-driven lift of {lift:.3f} m at t={sample.sim_time:.3f} s.")
                break
        else:
            raise RuntimeError(f"RC lift test failed, final z={sample.xyz[2]:.3f} m")
    finally:
        bridge.disarm()


if __name__ == "__main__":
    main()
