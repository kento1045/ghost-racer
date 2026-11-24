"""Regression checks for gate clearance, camera-only control and RC episode boundaries."""

import sys
import xml.etree.ElementTree as ET
from unittest.mock import patch
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import cv2
import numpy as np

from sim.optics import WIDTH, HEIGHT
from .client import Sample
from .env import GateEnv, gate_crossing
from sim.circuit import course_for, settings
from .policy import RaceObservation
from .localization import VisualPose
from .reference import ReferenceTrajectory


def camera_frame():
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[40:160, 160:280] = [255, 96, 6]
    rgb[46:154, 166:274] = 0
    return rgb


class Replay:
    def __init__(self, positions):
        self.positions = positions
        self.disarms = 0
        self.rc = []

    def _sample(self):
        return Sample(
            camera_frame(),
            np.asarray(self.positions[self.index]),
            self.index * 0.1,
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        )

    def reset(self, cfg):
        self.index = 0
        return self._sample()

    def step(self, rc):
        self.rc.append(rc)
        self.index += 1
        return self._sample()

    def disarm(self):
        self.disarms += 1


class RacingTests(unittest.TestCase):
    def test_bridge_retries_only_rejected_busy_requests(self):
        from io import BytesIO
        from urllib.error import HTTPError
        from .client import Bridge

        bridge = Bridge()
        def conflict(message):
            return HTTPError(bridge.url, 409, "Conflict", {}, BytesIO(message.encode()))

        with patch("learning.client.urlopen", side_effect=[
            conflict("The simulator is processing another command"), BytesIO(b"ok")
        ]) as send:
            self.assertEqual(bridge._post("/step", {"rc": [1500] * 4}), b"ok")
            self.assertEqual(send.call_count, 2)
        with patch("learning.client.urlopen", side_effect=conflict("Another controller is flying")) as send:
            with self.assertRaisesRegex(RuntimeError, "Another controller"):
                bridge._post("/reset", {})
            self.assertEqual(send.call_count, 1)

    def sample_reference(self):
        course = course_for(settings())
        start = np.asarray(course["spawn"])
        rows = [[i, i * 0.1, *start, *(start + [1, 0, 0]), 1, 0, 0, 0, 0, 0, 1,
                 i * 0.01, 0, 0.1, 0] for i in range(5 * len(course["gates"]) + 1)]
        return ReferenceTrajectory(rows, len(course["gates"]))

    def test_reference_uses_measured_progress_and_completed_segment(self):
        ref = self.sample_reference()
        pos = ref.rows[0, 2:5] + [0.34, 0.1, 0.0]
        for segment in (1, 9):
            point = ref.point(segment, pos)
            np.testing.assert_allclose(point.position, pos - [0, 0.1, 0])
            self.assertAlmostEqual(point.stick[0], segment * 0.01)
            self.assertAlmostEqual(ref.distance(segment, pos), 0.1)
        with self.assertRaises(ValueError):
            ReferenceTrajectory(ref.rows[:-1], 8)

    def test_reference_requires_repeat_finishes_and_ranks_timed_laps(self):
        from .train import select_reference

        env = RaceObservation(GateEnv(bridge=Replay([])))
        rows = self.sample_reference().rows
        def fly(env, model, reference_rows):
            elapsed = next(times)
            complete = elapsed is not None
            if complete:
                reference_rows.extend(rows)
            return dict(reason="finished" if complete else "gate_contact",
                        laps=5 if complete else 0, lap_times=[elapsed / 5] * 5 if complete else [])

        with patch("learning.train.rollout", side_effect=fly):
            times = iter([80, 82, 79, 80, 50, None])
            summary, _ = select_reference(env, lambda _: None, lambda *a, **kw: None)
            self.assertEqual(summary, dict(pace=1.85, mean_seconds=79.5, five_lap_seconds=79))
            times = iter([None] * 4)
            with self.assertRaisesRegex(ValueError, "基準軌道"):
                select_reference(env, lambda _: None, lambda *a, **kw: None)

    def test_reference_camera_control_and_augmented_reset(self):
        ref = self.sample_reference()
        outputs = []
        for xyz in ([0., 0., 0.02], [0.4, 0.2, 0.8]):
            env = RaceObservation(GateEnv(bridge=Replay([[0., 0., 0.022], xyz])))
            env.use_reference(ref)
            env.augment = True
            obs, _ = env.reset(seed=7)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertGreater(np.linalg.norm(env.pose_bias), 0)
            self.assertGreater(env.dropout, 0)
            obs, _, _, _, info = env.step(np.zeros(4, np.float32))
            outputs.append((obs, env.last_action, info["tracking_error_m"]))
            env.augment = False
            env.dropout = 0.0
            env.reset()
            np.testing.assert_array_equal(env.pose_bias, np.zeros(3))
            env.close()
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        self.assertNotEqual(outputs[0][2], outputs[1][2])

    def test_fc_disarm_terminates_episode(self):
        bridge = Replay([[0, 0, 0.02]])
        env = GateEnv(bridge=bridge)
        env.reset()
        bridge.step = lambda rc: Sample(camera_frame(), np.array([0, 0, 0.02]), 0.1, armed=False)
        _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "fc_disarmed")
        self.assertLess(reward, -3)

    def test_lower_rail_stops_episode_before_plane(self):
        course = course_for(settings())
        x, y, z = course["gates"][0]["center"]
        bridge = Replay([course["spawn"], [x - 0.04, y, z - course["gates"][0]["opening"][1] / 2]])
        env = GateEnv(bridge=bridge)
        env.reset()
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "gate_contact")
        self.assertEqual(bridge.disarms, 1)

    def test_rotated_plane_and_frame_clearance(self):
        gate = {"center": [0, 0, 1], "yaw": np.pi / 2}
        start = np.array([0.1, -1, 1])
        end = np.array([0.1, 1, 1])
        point, cleared = gate_crossing(start, end, gate, [0.4, 0.4], 0.05)
        self.assertTrue(cleared)
        np.testing.assert_allclose(point, [0.1, 0, 1], atol=1e-10)
        self.assertIsNone(gate_crossing(end, start, gate, [0.4, 0.4], 0.05)[0])
        self.assertFalse(gate_crossing(start, end, gate, [0.24, 0.4], 0.05)[1])

    def test_camera_profile_and_distorted_corner_rays(self):
        from sim.optics import CX, CY, DIAGONAL_FOV, FOCAL, K1, RENDER_FOCAL, RENDER_HFOV, lens_mesh

        root = Path(__file__).resolve().parents[1]
        sensor = ET.parse(root / "sim/models/whoop/model.sdf").find(".//sensor[@name='fpv']")
        mask = int(sensor.findtext("camera/visibility_mask"))
        model = ET.parse(root / "sim/models/whoop/model.sdf")
        self.assertTrue(
            all(not (int(v.findtext("visibility_flags")) & mask) for v in model.findall(".//visual"))
        )
        self.assertAlmostEqual(float(sensor.findtext("update_rate")), 60, places=6)
        self.assertAlmostEqual(float(sensor.findtext("camera/horizontal_fov")), RENDER_HFOV, places=8)
        self.assertEqual(
            [int(sensor.findtext("camera/image/" + axis)) for axis in ("width", "height")],
            [WIDTH, HEIGHT],
        )
        self.assertTrue(all(g["opening"] == [1.0, 1.0] for g in settings()["course"]["gates"]))
        # Invert the renderer sampling mesh, then project through the PnP camera.
        origin = np.array([CX + 0.5, CY + 0.5])
        for (left, top, _, _), quad in lens_mesh()[::97]:
            ray = (np.array(quad[:2]) - origin) / RENDER_FOCAL
            pixel = origin + FOCAL * ray * (1 + K1 * np.dot(ray, ray))
            np.testing.assert_allclose(pixel, [left, top], atol=1e-5)
        ray_to_corner = np.linalg.norm(np.array(lens_mesh()[0][1][:2]) - origin) / RENDER_FOCAL
        self.assertAlmostEqual(2 * np.rad2deg(np.arctan(ray_to_corner)), DIAGONAL_FOV)

    def test_rectified_detector_contains_distorted_image_corners(self):
        from .localization import DIST, K, RECT_K, RECT_SIZE
        from sim.optics import CX, CY, EDGE_SCALE

        pixels = np.float64([[[0.0, 0.0]], [[2 * CX, 2 * CY]]])
        corners = cv2.undistortPointsIter(
            pixels, K, DIST, None, RECT_K, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
        ).reshape(2, 2)
        self.assertTrue(np.all(corners >= 0) and np.all(corners < RECT_SIZE))
        self.assertGreater(corners[1, 0] - corners[0, 0], WIDTH * (1 + (1 - EDGE_SCALE) / 2))

    def test_gate_corner_pnp_recovers_displaced_camera(self):
        for angle in (40., 22.5):
            with self.subTest(angle=angle):
                self.check_gate_pose(angle)

    def check_gate_pose(self, angle):
        from .localization import K, OFFSET, DIST
        from sim.circuit import fiducials

        course = course_for(settings(dict(camera_angle_deg=angle)))
        position = np.array([0.12, -0.11, 0.22])
        body = cv2.Rodrigues(np.array([0.03, 0.06, 0.05]))[0]
        horizontal_camera = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        mount = cv2.Rodrigues(np.array([0., -np.deg2rad(angle), 0.]))[0]
        rotation = (body @ mount @ horizontal_camera).T
        translation = -rotation @ (position + body @ OFFSET)
        detector = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        # Supersample to simulate continuous subpixel alignment of the printed tags.
        rgb = np.full((2 * HEIGHT, 2 * WIDTH), 255, np.uint8)
        src = np.float32([[0, 0], [119, 0], [119, 119], [0, 119]])
        for tag_id, points in enumerate(fiducials(course)[:4]):
            uv, _ = cv2.projectPoints(np.array(points), cv2.Rodrigues(rotation)[0], translation, K, DIST)
            H = cv2.getPerspectiveTransform(src, (2 * uv.reshape(4, 2)).astype(np.float32))
            texture = cv2.aruco.generateImageMarker(detector, tag_id, 120)
            layer = cv2.warpPerspective(texture, H, (2 * WIDTH, 2 * HEIGHT), borderValue=255)
            rgb = np.minimum(rgb, layer)
        rgb = cv2.cvtColor(
            cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB
        )
        pose = VisualPose(course)
        self.assertTrue(pose.update(rgb))
        self.assertEqual(pose.detected[0].shape, (4, 2))
        np.testing.assert_allclose(pose.xyz, position, atol=0.05)
        np.testing.assert_allclose(pose.body, body, atol=0.04)
        pose = VisualPose(course)
        self.assertTrue(pose.update(rgb, predicted_body=body))
        np.testing.assert_allclose(pose.xyz, position, atol=0.05)
        np.testing.assert_allclose(pose.body, body, atol=0.04)

    def test_faster_valid_section_earns_more_reward(self):
        course = course_for(settings())
        first = np.array(course["gates"][0]["center"])
        scores = []
        for duration in (0.1, 0.6):
            bridge = Replay([course["spawn"]])
            env = GateEnv(bridge=bridge)
            env.reset()
            bridge.step = lambda rc: Sample(camera_frame(), first, duration)
            _, reward, _, _, info = env.step(np.zeros(4, np.float32))
            self.assertEqual(info["gates_passed"], 1)
            scores.append(reward)
            env.close()
        self.assertGreater(scores[0], scores[1])

    def test_camera_loss_and_no_altitude_ceiling(self):
        bridge = Replay([[0, 0, 0.02], [0, 0, 12.0]])
        env = RaceObservation(GateEnv(bridge=bridge))
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        obs, _, term, trunc, info = env.step(np.zeros(4, np.float32))
        self.assertFalse(term or trunc)
        self.assertEqual(info["pose_valid_fraction"], 0.0)
        env.close()

    def test_delayed_pose_replays_applied_inputs(self):
        from .navigation import Predictor, rotation

        course = course_for(settings())
        immediate, delayed = [Predictor(course, 26.0, 0.0) for _ in range(2)]
        cmd = np.array([0.12, 0.2, 0.4, 0.1])
        next_cmd = np.array([-0.2, 0.0, -0.2, -0.1])
        t = 0.05
        for predictor in (immediate, delayed):
            predictor.advance(t, cmd)
        observed = immediate.state.pos + np.array([0.02, 0.01, 0.03])
        body = immediate.state.body @ rotation([0.01, -0.03, 0.0])
        self.assertTrue(immediate.correct(t, observed, body))
        for predictor in (immediate, delayed):
            predictor.advance(0.10, cmd)
            predictor.advance(0.24, next_cmd)
        self.assertTrue(delayed.correct(t, observed, body))
        for attr in ("pos", "vel", "body", "omega"):
            np.testing.assert_allclose(
                getattr(immediate.state, attr), getattr(delayed.state, attr), atol=1e-9
            )
        self.assertFalse(delayed.correct(t, observed, body))

    def test_predicted_gate_can_be_retracted_or_confirmed(self):
        from .navigation import RacingLine

        course = course_for(settings())
        center = np.asarray(course["gates"][0]["center"])
        n = np.array([np.cos(course["gates"][0]["yaw"]), np.sin(course["gates"][0]["yaw"]), 0.0])
        line = RacingLine(course)
        line.update(center - 0.2 * n, True)
        line.update(center + 0.1 * n, False)
        self.assertEqual((line.index, line.confirmed), (1, 0))
        line.update(center - 0.2 * n, True)
        self.assertEqual((line.index, line.confirmed), (0, 0))
        line.update(center + 0.1 * n, True)
        self.assertEqual((line.index, line.confirmed), (1, 1))

    def test_observation_on_exit_recovers_a_missed_navigation_crossing(self):
        from .navigation import RacingLine

        course = course_for(settings())
        gate = np.array(course["gates"][0]["center"])
        line = RacingLine(course)
        line.update(gate + [-0.5, 0, 0], True)
        line.update(gate + [-0.01, 0.8, 0], False)
        line.update(gate + [0.1, 0.8, 0], False)
        self.assertEqual(line.index, 0)
        line.update(gate + [0.55, 0.03, 0], False)
        self.assertEqual(line.index, 0)
        line.update(gate + [0.62, 0.03, 0], True)
        self.assertEqual((line.index, line.confirmed), (1, 1))

    def test_dropouts_use_capture_time_and_do_not_update_pose(self):
        course = course_for(settings())
        env = RaceObservation(GateEnv(bridge=Replay([course["spawn"]])))
        env.dropout = 0.35
        env.reset()
        try:
            env.predictor.state.pos = np.asarray(course["gates"][0]["center"])
            self.assertTrue(env._mask_image(5.0))
            self.assertTrue(env._mask_image(5.34))
            self.assertFalse(env._mask_image(5.36))
            with patch.object(env.pose, "update", side_effect=AssertionError("used hidden image")):
                _, _, valid, gates, _, _, masked = env._detect(
                    5.1, camera_frame(), True, env.predictor.state.pos, env.predictor.state.body
                )
            self.assertTrue(masked)
            self.assertFalse(valid or gates)
        finally:
            env.close()

    def test_camera_controller_is_independent_of_scoring_pose(self):
        outputs = []
        for xyz in ([0.0, 0.0, 0.02], [0.5, 0.1, 3.0]):
            env = RaceObservation(GateEnv(bridge=Replay([[0.0, 0.0, 0.02], xyz])))
            env.reset()
            obs, *_ = env.step(np.zeros(4, np.float32))
            outputs.append((obs, env.nominal))
            env.close()
        for a, b in zip(*outputs):
            np.testing.assert_array_equal(a, b)

    def test_five_complete_laps_and_episode_boundaries(self):
        course = course_for(settings())
        # Cross gate 1, then five full circuits returning to gate 1.
        positions = [course["spawn"]]
        count = len(course["gates"])
        for i in range(5 * count + 1):
            gate = course["gates"][i % count]
            center = np.array(gate["center"])
            delta = 0.08 * np.array([np.cos(gate["yaw"]), np.sin(gate["yaw"]), 0.0])
            positions.extend([center - delta, center + delta])
        bridge = Replay(positions)
        env = GateEnv(bridge=bridge)
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        for i in range(1, len(positions)):
            _, _, terminated, truncated, info = env.step([0, 0.5, -0.5, 1])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["gates_passed"], 1 + 5 * len(course["gates"]))
        self.assertEqual(info["laps"], 5)
        self.assertEqual(info["reason"], "finished")
        self.assertEqual(bridge.rc[0], [1500, 1750, 1250, 2000])
        env.close()
        self.assertEqual(bridge.disarms, 1)
        env.reset()
        env.close()
        self.assertEqual(bridge.disarms, 2)

    def test_early_stop_counts_only_current_evaluation_batch(self):
        from .train import train

        # Each entry is one deterministic evaluation: (passed gates, completed laps).
        cases = [
            ("gate", [(1, 0), (1, 0)], 32, "goal_reached"),
            ("gate", [(1, 0), (0, 0), (0, 0), (1, 0)], 64, "step_limit"),
            ("laps", [(1, 0), (41, 5), (41, 5), (41, 5)], 64, "goal_reached"),
            ("off", [], 64, "step_limit"),
        ]
        for goal, flights, expected_steps, reason in cases:
            with self.subTest(goal=goal, reason=reason), TemporaryDirectory() as tmp:
                config = dict(
                    steps=64,
                    rollout_steps=32,
                    goal=goal,
                    eval_flights=2,
                    required_successes=2,
                    check_rollouts=1,
                )
                env = RaceObservation(
                    GateEnv(bridge=Replay([[0.0, 0.0, 0.022]] * 2), max_seconds=0.1, config=config)
                )
                output = Path(tmp) / "policy.zip"
                progress = []
                with patch("learning.train.select_reference", return_value=({}, self.sample_reference())), patch(
                    "learning.train.rollout", side_effect=[dict(gates=g, laps=l) for g, l in flights]
                ) as evaluate:
                    result = train(
                        env,
                        output,
                        lambda trial: None,
                        lambda phase, n, **kw: progress.append((phase, n)),
                    )
                env.close()
                self.assertEqual(evaluate.call_count, len(flights))
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["steps"], expected_steps)
                self.assertEqual(
                    [n for phase, n in progress if phase == "training"],
                    list(range(1, expected_steps + 1)),
                )
                self.assertTrue(output.is_file())
                from stable_baselines3 import PPO
                from .train import restore_reference
                loaded = PPO.load(output, device="cpu")
                other = RaceObservation(GateEnv(bridge=Replay([[0., 0., 0.022]])))
                restore_reference(other, loaded)
                obs, _ = other.reset()
                self.assertEqual(obs.shape, (38,))
                self.assertEqual(loaded.predict(obs, deterministic=True)[0].shape, (4,))
                other.close()

    def test_training_settings(self):
        for invalid in ({"speed": 1}, {"steps": 33, "rollout_steps": 32}, {"mass": True}):
            with self.assertRaises(ValueError):
                settings(invalid)

    def test_camera_mount_in_world(self):
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "sim"), *sys.path]):
            from build_course import build_world
        for angle in (-15., 0., 32.5):
            course = course_for(settings(dict(camera_angle_deg=angle)))
            world = ET.fromstring(build_world(course))
            body = world.find(".//model[@name='whoop']/link[@name='body']")
            for element in ("sensor[@name='fpv']", "visual[@name='camera_housing']"):
                self.assertAlmostEqual(float(body.findtext(element + "/pose").split()[4]), -np.deg2rad(angle))
        for invalid in (91, -91, True):
            with self.assertRaises(ValueError):
                settings(dict(camera_angle_deg=invalid))

    def test_custom_json_survives_world_scoring_and_heading(self):
        from sim.circuit import course_for, settings

        custom = settings()["course"]
        custom["gates"][1].update(center=[4.8, 0.3, 1.55], yaw_deg=85, opening=[0.63, 1.35])
        actual = course_for(settings(dict(course=custom)))
        gate = actual["gates"][1]
        self.assertEqual(gate["opening"], [0.63, 1.35])
        self.assertEqual(gate["center"], [4.8, 0.3, 1.55])
        self.assertAlmostEqual(gate["yaw"], np.deg2rad(85))
        normal = np.array([np.cos(gate["yaw"]), np.sin(gate["yaw"]), 0.0])
        start = np.array(gate["center"]) - normal
        _, cleared = gate_crossing(start, start + 2 * normal, gate, gate["opening"], 0.051)
        self.assertTrue(cleared)
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "sim"), *sys.path]):
            from build_course import build_world

            world = ET.fromstring(build_world(actual, 29))
        model = world.find(".//model[@name='gate_2']")
        pose = np.fromstring(model.findtext("pose"), sep=" ")
        np.testing.assert_allclose(pose[:3], gate["center"])
        self.assertAlmostEqual(pose[5], gate["yaw"], places=5)
        top = model.find(".//collision[@name='top_collision']")
        size = np.fromstring(top.findtext("geometry/box/size"), sep=" ")
        self.assertAlmostEqual(size[1], 0.63)
        z = np.fromstring(top.findtext("pose"), sep=" ")[2]
        self.assertAlmostEqual(2 * z - actual["rail_thickness"], 1.35)
        mass = world.findtext(".//model[@name='whoop']/link[@name='body']/inertial/mass")
        self.assertAlmostEqual(float(mass) + 0.0004, 0.029)
        custom["gates"][1]["center"][2] = 0.1
        with self.assertRaises(ValueError):
            settings(dict(course=custom))

    def test_time_limit(self):
        bridge = Replay([[0, 0, 0.022]] * 2)
        env = GateEnv(bridge=bridge, max_seconds=0.1)
        env.reset()
        _, _, terminated, truncated, info = env.step([0] * 4)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "time_limit")
        self.assertEqual(bridge.disarms, 1)


if __name__ == "__main__":
    unittest.main()
