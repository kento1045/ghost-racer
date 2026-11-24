"""Spatially indexed five-lap Gazebo demonstration, including the launch segment."""
from dataclasses import dataclass

import numpy as np

from .navigation import quaternion_matrix


@dataclass
class ReferencePoint:
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    stick: np.ndarray


class ReferenceTrajectory:
    # Each row describes one command interval: cleared gates, start time, start/end
    # positions, start velocity, start quaternion (xyzw), and the four transmitted sticks.
    def __init__(self, rows, gate_count):
        data = np.asarray(rows, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 19 or not np.isfinite(data).all():
            raise ValueError("基準軌道の記録が不正です")
        count = 5 * gate_count + 1
        ids = data[:, 0]
        if not np.array_equal(np.unique(ids), np.arange(count)) or np.any(np.diff(ids) < 0):
            raise ValueError("5周分のゲート区間が揃っていません")
        self.rows = data
        self.segments = np.split(data, np.flatnonzero(np.diff(ids)) + 1)
        self.edges = [s[:, 5:8] - s[:, 2:5] for s in self.segments]
        self.length2 = [np.maximum(np.einsum("ij,ij->i", e, e), 1e-16) for e in self.edges]

    def _project(self, cleared, position):
        # The terminal observation still needs a valid reference after the finish gate.
        i = min(cleared, len(self.segments) - 1)
        segment, edges = self.segments[i], self.edges[i]
        offsets = position - segment[:, 2:5]
        fraction = np.clip(np.einsum("ij,ij->i", offsets, edges) / self.length2[i], 0, 1)
        residual = offsets - fraction[:, None] * edges
        best = np.argmin(np.einsum("ij,ij->i", residual, residual))
        return segment[best], residual[best]

    def point(self, cleared, position):
        row, offset = self._project(cleared, position)
        return ReferencePoint(position - offset, row[8:11], quaternion_matrix(row[11:15]), row[15:19])

    def distance(self, cleared, position):
        _, offset = self._project(cleared, position)
        return float(np.linalg.norm(offset))
