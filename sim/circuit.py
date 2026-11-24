"""Shared race geometry and validated settings for the UI, scorer and simulator."""
from math import cos, dist, hypot, inf, isfinite, radians, sin
import json
from pathlib import Path

DEFAULTS = dict(
    course=None,
    mass=26.0,
    camera_angle_deg=40.0,
    steps=8192,
    rollout_steps=128,
    seed=7,
    seconds=180,
    goal="off",
    check_rollouts=8,
    eval_flights=5,
    required_successes=4,
)
LIMITS = dict(
    mass=(20, 35),
    camera_angle_deg=(-90, 90),
    steps=(32, 10000000),
    rollout_steps=(32, 2048),
    seed=(0, 2147483647),
    seconds=(3, 3600),
    check_rollouts=(1, 10000),
    eval_flights=(1, 100),
    required_successes=(1, 100),
)


def settings(values=None):
    values = {} if values is None else values
    if not isinstance(values, dict) or values.keys() - DEFAULTS.keys():
        raise ValueError("設定項目が不明です")
    cfg = DEFAULTS | values
    for key, (low, high) in LIMITS.items():
        v = cfg[key]
        if type(v) not in (int, float) or not isfinite(v) or not low <= v <= high:
            raise ValueError(f"{key} は {low} ～ {high} の範囲です")
        if type(DEFAULTS[key]) is int and type(v) is not int:
            raise ValueError(f"{key} は整数で指定してください")
    if cfg["goal"] not in ("off", "gate", "laps"):
        raise ValueError("早期終了条件を選んでください")
    if cfg["required_successes"] > cfg["eval_flights"]:
        raise ValueError("必要成功回数は評価飛行数以下にしてください")
    cfg["course"] = validate_layout(
        cfg["course"]
        if cfg["course"] is not None
        else json.loads((Path(__file__).parent / "course.json").read_text())
    )
    if cfg["steps"] % cfg["rollout_steps"]:
        raise ValueError("総ステップ数はロールアウト長の倍数にしてください")
    return cfg


def vector(v, length, low, high):
    return (
        isinstance(v, list)
        and len(v) == length
        and all(type(x) in (float, int) and isfinite(x) and low <= x <= high for x in v)
    )


def validate_layout(doc):
    if (
        not isinstance(doc, dict)
        or set(doc) != {"version", "gates"}
        or type(doc["version"]) is not int
        or doc["version"] != 1
    ):
        raise ValueError("コース JSON は version=1 と gates 配列を指定してください")
    gates = doc["gates"]
    if not isinstance(gates, list) or not 6 <= len(gates) <= 16:
        raise ValueError("ゲート数は6〜16です")
    for i, g in enumerate(gates):
        if (
            not isinstance(g, dict)
            or set(g) != {"center", "yaw_deg", "opening"}
            or not vector(g["center"], 3, -inf, inf)
            or any(abs(v) > 20 for v in g["center"][:2])
            or not vector(g["opening"], 2, 0.35, 2.5)
            or not vector([g["yaw_deg"]], 1, -360, 360)
            or g["center"][2] <= g["opening"][1] / 2 + 0.065
        ):
            raise ValueError(f"{i+1}番の位置・向き・開口寸法が範囲外です。下枠は床より上に配置してください")
    if any(dist(g["center"][:2], gates[i - 1]["center"][:2]) < 0.40 for i, g in enumerate(gates)):
        raise ValueError("連続するゲートの水平間隔は0.4 m以上必要です")
    return doc


def segment(first, last, t):
    """Cubic Hermite centerline, tangent to the specified direction at both gates."""
    p, q = first["center"], last["center"]
    size = hypot(q[0] - p[0], q[1] - p[1])
    tangents = [(size * cos(g["yaw"]), size * sin(g["yaw"]), 0.0) for g in (first, last)]
    a, b = tangents
    h = (2 * t**3 - 3 * t * t + 1, t**3 - 2 * t * t + t, -2 * t**3 + 3 * t * t, t**3 - t * t)
    dh = (6 * t * t - 6 * t, 3 * t * t - 4 * t + 1, -6 * t * t + 6 * t, 3 * t * t - 2 * t)
    cols = list(zip(p, a, q, b))
    return (
        [sum(k * v for k, v in zip(h, col)) for col in cols],
        [sum(k * v for k, v in zip(dh, col)) for col in cols],
    )


def course_for(cfg):
    doc = cfg["course"]
    gates = [
        dict(name=f"gate_{i+1}", center=g["center"], yaw=radians(g["yaw_deg"]), opening=g["opening"])
        for i, g in enumerate(doc["gates"])
    ]
    start = gates[0]
    x, y, _ = start["center"]
    spawn = [x - 1.6 * cos(start["yaw"]), y - 1.6 * sin(start["yaw"]), 0.022]
    path = [
        segment(g, gates[(i + 1) % len(gates)], j / 24)[0]
        for i, g in enumerate(gates)
        for j in range(24)
    ]
    points = path + [spawn]
    bounds = [[min(p[i] for p in points) - 1.65, max(p[i] for p in points) + 1.65] for i in (0, 1)]
    ceiling = max(3.4, max(g["center"][2] + g["opening"][1] / 2 + 0.6 for g in gates))
    return dict(
        gates=gates,
        camera_angle_deg=cfg.get("camera_angle_deg", DEFAULTS["camera_angle_deg"]),
        rail_thickness=0.065,
        vehicle_radius=0.051,
        spawn=spawn,
        spawn_yaw=start["yaw"],
        bounds=bounds + [[0, ceiling]],
        path=path,
    )


def gate_corners(gate):
    """Opening corners seen from the approach side: TL, TR, BR, BL."""
    x, y, z = gate["center"]
    yaw = gate["yaw"]
    w, h = gate["opening"]
    return [
        (x - s * sin(yaw) * w / 2, y + s * cos(yaw) * w / 2, z + t * h / 2)
        for s, t in ((1, 1), (-1, 1), (-1, -1), (1, -1))
    ]


def fiducials(course):
    """Four coded placards per gate, coplanar with its front face."""
    tags = []
    for gate in course["gates"]:
        yaw = gate["yaw"]
        for (x, y, z), (side, up) in zip(gate_corners(gate), ((1, 1), (-1, 1), (-1, -1), (1, -1))):
            # Placards sit outside the opening and identify their gate and corner.
            x -= 0.041 * cos(yaw) + 0.19 * side * sin(yaw)
            y += 0.19 * side * cos(yaw) - 0.041 * sin(yaw)
            z += 0.19 * up
            tags.append(
                [
                    (x - l * 0.14 * sin(yaw), y + l * 0.14 * cos(yaw), z + v * 0.14)
                    for l, v in ((1, 1), (-1, 1), (-1, -1), (1, -1))
                ]
            )
    return tags
