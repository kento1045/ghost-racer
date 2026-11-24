"""Decoded NTSC 4:3 FPV profile. The nominal 160° lens specification is diagonal."""
from functools import lru_cache
from math import atan, hypot, radians, tan

WIDTH, HEIGHT = 640, 480
CX, CY = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
DIAGONAL_FOV = 160
EDGE_RADIUS = tan(radians(DIAGONAL_FOV / 2))
# Brown's radial polynomial in normalized camera coordinates. The outermost
# ray is compressed by 25% relative to ideal pinhole projection.
EDGE_SCALE = 0.75
K1 = (EDGE_SCALE - 1) / EDGE_RADIUS**2
RENDER_FOCAL = hypot(WIDTH, HEIGHT) / (2 * EDGE_RADIUS)
FOCAL = RENDER_FOCAL / EDGE_SCALE
RENDER_HFOV = 2 * atan(WIDTH / (2 * RENDER_FOCAL))


@lru_cache(maxsize=1)
def lens_mesh():
    # Pillow uses pixel-edge coordinates for its inverse sampling mesh.
    def source(x, y):
        dx, dy = x - WIDTH / 2, y - HEIGHT / 2
        rd = hypot(dx, dy) / FOCAL
        ru = rd
        for _ in range(6):
            ru -= (ru * (1 + K1 * ru**2) - rd) / (1 + 3 * K1 * ru**2)
        scale = EDGE_SCALE / (1 + K1 * ru**2)
        return (WIDTH / 2 + dx * scale, HEIGHT / 2 + dy * scale)

    mesh = []
    for top in range(0, HEIGHT, 16):
        for left in range(0, WIDTH, 16):
            right, bottom = left + 16, top + 16
            corners = (
                source(left, top),
                source(left, bottom),
                source(right, bottom),
                source(right, top),
            )
            mesh.append(((left, top, right, bottom), tuple(v for point in corners for v in point)))
    return mesh


def lens_image(rgb):
    from PIL import Image

    frame = Image.frombytes("RGB", (WIDTH, HEIGHT), rgb)
    return frame.transform(
        frame.size, Image.Transform.MESH, lens_mesh(), Image.Resampling.BILINEAR
    ).tobytes()
