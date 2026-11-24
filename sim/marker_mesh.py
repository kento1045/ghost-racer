"""Flat, unlit ArUco board geometry, rendered in Gazebo's real FPV camera."""
from pathlib import Path
from circuit import fiducials

BITS = Path(__file__).with_name("marker_bits.txt").read_text().splitlines()


def marker_mesh(course):
    quads = []
    for tag, bits in zip(fiducials(course), BITS):
        a, b, _, d = tag

        def point(i, j):
            return tuple(a[k] + (b[k] - a[k]) * i / 6 + (d[k] - a[k]) * j / 6 for k in range(3))

        for j in range(-1, 7):
            # Merge each consecutive same-color run into one coplanar quad.
            row = [
                1
                if i in (-1, 6) or j in (-1, 6)
                else (int(bits[(j - 1) * 4 + i - 1]) if 1 <= i <= 4 and 1 <= j <= 4 else 0)
                for i in range(-1, 7)
            ]
            start = -1
            for end in range(0, 8):
                if end < 7 and row[end + 1] == row[start + 1]:
                    continue
                if row[start + 1]:
                    quads.append(
                        [point(start, j), point(start, j + 1), point(end, j + 1), point(end, j)]
                    )
                start = end
    lines = []
    for n, quad in enumerate(quads):
        lines.extend("v %.5f %.5f %.5f" % p for p in quad)
        i = 4 * n + 1
        lines.append(f"f {i} {i+1} {i+2}")
        lines.append(f"f {i} {i+2} {i+3}")
    return "\n".join(lines) + "\n"
