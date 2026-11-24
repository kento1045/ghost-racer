"""Recover physical gate-opening corners and estimate the FPV camera pose."""
import cv2
import numpy as np
from sim.circuit import DEFAULTS, fiducials, gate_corners

from sim.optics import CX, CY, FOCAL, WIDTH, HEIGHT, K1, EDGE_SCALE

DIST = np.array([K1, 0.0, 0.0, 0.0, 0.0])
K = np.array([[FOCAL, 0, CX], [0, FOCAL, CY], [0, 0, 1.0]])
# Detect straight gate edges in rectified space; replay uses original frame coordinates.
# Keep peripheral gates in the detection image after undoing barrel distortion.
RECT_SIZE = tuple(int(np.ceil(n / EDGE_SCALE)) for n in (WIDTH, HEIGHT))
RECT_CENTER = (np.array(RECT_SIZE) - 1) / 2
RECT_K = K.copy()
RECT_K[:2, 2] = RECT_CENTER
RECTIFY_MAP = cv2.initUndistortRectifyMap(K, DIST, None, RECT_K, RECT_SIZE, cv2.CV_16SC2)

OFFSET = np.array([0.022, 0, 0.013])


def distorted_pixels(rectified):
    rays = (rectified - RECT_CENTER) / FOCAL
    scale = 1 + K1 * np.sum(rays**2, axis=1, keepdims=True)
    return rays * scale * FOCAL + [CX, CY]


class VisualPose:
    def __init__(self, course):
        angle = np.deg2rad(course.get("camera_angle_deg", DEFAULTS["camera_angle_deg"]))
        c, s = np.cos(angle), np.sin(angle)
        self.optical = np.array([[0, s, c], [-1, 0, 0], [0, -c, s]])
        self.tags = np.array(fiducials(course))
        self.corners = np.array([gate_corners(g) for g in course["gates"]])
        self.xyz = np.array(course["spawn"], dtype=float)
        self.body = cv2.Rodrigues(np.array([0.0, 0.0, course["spawn_yaw"]]))[0]
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000), params
        )
        self.detected = {}

    def update(self, rgb, *, predicted_body=None):
        # The inner contour gives the four physical opening corners when all
        # rails are visible. Coded corner placards identify the gate.
        rgb = cv2.remap(rgb, *RECTIFY_MAP, cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        orange = cv2.inRange(hsv, (3, 90, 45), (28, 255, 255))
        contours, hierarchy = cv2.findContours(orange, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        if hierarchy is not None:
            for contour, node in zip(contours, hierarchy[0]):
                if node[3] < 0 or cv2.contourArea(contour) < 30:
                    continue
                poly = cv2.approxPolyDP(contour, 0.025 * cv2.arcLength(contour, True), True)
                if len(poly) == 4 and cv2.isContourConvex(poly):
                    holes.append(poly.reshape(4, 2))
        depths = {}
        single_tags = set()
        obj, img = [], []
        markers, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        self.detected = {}
        if ids is None:
            return False
        ids = ids.ravel()
        for gate in np.unique(ids // 4):
            if gate >= len(self.corners):
                continue
            corners = self.corners[gate]
            # Use gate-local metre coordinates on the plane of the markings.
            right = corners[1] - corners[0]
            right /= np.linalg.norm(right)

            def plane(points):
                offset = points - corners[0]
                return np.stack([offset @ right, -offset[..., 2]], axis=-1)

            selected = np.where(ids // 4 == gate)[0]
            world = self.tags[ids[selected]].reshape(-1, 3)
            pixels = np.array(markers)[selected].reshape(-1, 2)
            H, mask = cv2.findHomography(plane(world), pixels, cv2.RANSAC, 1.8)
            if H is None or mask.sum() < 4:
                continue

            uv = cv2.perspectiveTransform(plane(corners)[None], H)[0]
            if not np.isfinite(uv).all() or not cv2.isContourConvex(uv.astype(np.float32)):
                continue
            best = None
            threshold = max(3.0, 0.14 * np.linalg.norm(uv[0] - uv[2]))
            for hole in holes:
                indices = np.linalg.norm(uv[:, None] - hole, axis=2).argmin(axis=1)
                if len(set(indices)) < 4:
                    continue
                error = np.linalg.norm(hole[indices] - uv, axis=1).max()
                if error < threshold:
                    best, threshold = hole[indices], error
            if best is None and (len(selected) < 2 or mask.sum() < 6):
                # A nearby, well-resolved tag can maintain pose as the gate
                # leaves view. Extrapolated vertices are not independent data.
                sides = np.linalg.norm(pixels[:4] - np.roll(pixels[:4], 1, axis=0), axis=1)
                if len(selected) != 1 or min(sides) < 12:
                    continue
                single_tags.add(int(gate))
            rectified = uv if best is None else best.astype(float)
            self.detected[int(gate)] = distorted_pixels(rectified)
            # The measured tag points also constrain planar pose when an
            # opening corner lies outside the image and must be extrapolated.
            inliers = mask.ravel().astype(bool)
            obj.extend(world[inliers])
            img.extend(distorted_pixels(pixels[inliers]))
            depths[int(gate)] = 0.041 if best is None else 0.0325
        if not self.detected:
            return False
        # Use the measured rail face or the projected tag plane at its own depth.
        for gate, pixels in self.detected.items():
            if gate in single_tags:
                continue
            corners = self.corners[gate]
            normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            normal /= np.linalg.norm(normal)
            obj.extend(corners - depths[gate] * normal)
            img.extend(pixels)
        obj, img = np.array(obj), np.array(img)
        prior = self.optical.T @ self.body.T
        rv = cv2.Rodrigues(prior)[0]
        tv = -prior @ (self.xyz + self.body @ OFFSET)
        # During a long visual gap the predicted pose can fall outside the
        # iterative solver's basin. Reacquire directly from map correspondences.
        for use_prior in (True, False):
            mode = cv2.SOLVEPNP_ITERATIVE if use_prior else cv2.SOLVEPNP_SQPNP
            try:
                ok, rv, tv = cv2.solvePnP(obj, img, K, DIST, rv, tv, use_prior, mode)
            except cv2.error:
                # A foreshortened tag can yield a numerically singular planar problem.
                continue
            if not ok:
                continue
            rot = cv2.Rodrigues(rv)[0]
            if np.any(obj @ rot[2] + tv.ravel()[2] <= 0):
                continue
            projected = cv2.projectPoints(obj, rv, tv, K, DIST)[0].reshape(-1, 2)
            if (
                not np.isfinite(projected).all()
                or np.median(np.linalg.norm(projected - img, axis=1)) > 3
            ):
                continue
            if predicted_body is not None:
                # A nearly frontal planar gate can fit its pixels well at two
                # tilts. Use the ACRO attitude prediction to constrain this weak
                # direction, then refit translation on the observed rays.
                measured = rot.T @ self.optical.T
                innovation = cv2.Rodrigues(predicted_body.T @ measured)[0]
                body = predicted_body @ cv2.Rodrigues(0.25 * innovation)[0]
                rot = self.optical.T @ body.T
                rv = cv2.Rodrigues(rot)[0]
                rays = cv2.undistortPointsIter(
                    img.reshape(-1, 1, 2),
                    K,
                    DIST,
                    None,
                    None,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-9),
                ).reshape(-1, 2)
                points = obj @ rot.T
                design = np.tile(np.eye(2, 3), (len(obj), 1))
                design[:, 2] = -rays.ravel()
                rhs = (rays * points[:, 2:3] - points[:, :2]).ravel()
                weights = np.repeat(1 / np.maximum(0.05, points[:, 2] + tv.ravel()[2]), 2)
                tv = np.linalg.lstsq(design * weights[:, None], rhs * weights, rcond=None)[0]
                if np.any(points[:, 2] + tv[2] <= 0.01):
                    continue
            for gate in single_tags:
                vertices = self.corners[gate]
                if np.all(vertices @ rot[2] + tv.ravel()[2] > 0.01):
                    self.detected[gate] = cv2.projectPoints(vertices, rv, tv, K, DIST)[0].reshape(4, 2)
                else:
                    self.detected.pop(gate, None)
            self.body = rot.T @ self.optical.T
            self.xyz = (-rot.T @ tv).ravel() - self.body @ OFFSET
            return True
        return False
