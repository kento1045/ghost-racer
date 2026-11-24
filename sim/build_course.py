"""Build the checked-in Gazebo world from the geometry used to score episodes."""
import argparse
from math import radians
from circuit import DEFAULTS, course_for, settings
from marker_mesh import marker_mesh
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
WORLD = ROOT / "worlds/race.sdf"
ORANGE = "1.0 0.30 0.015 1"
DARK = "0.10 0.14 0.18 1"


def numbers(values):
    return " ".join(f"{v:.6g}" for v in values)


def box(link, name, size, position, color, collision=True):
    for kind in ("collision", "visual") if collision else ("visual",):
        element = ET.SubElement(link, kind, name=name + "_" + kind)
        ET.SubElement(element, "pose").text = numbers((*position, 0, 0, 0))
        shape = ET.SubElement(ET.SubElement(element, "geometry"), "box")
        ET.SubElement(shape, "size").text = numbers(size)
        if kind == "visual":
            material = ET.SubElement(element, "material")
            ET.SubElement(material, "ambient").text = color
            ET.SubElement(material, "diffuse").text = color
            if color == ORANGE:
                ET.SubElement(material, "emissive").text = "0.12 0.022 0.002 1"


def static_link(world, name, pose):
    model = ET.SubElement(world, "model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = numbers(pose)
    return ET.SubElement(model, "link", name="frame")


def build_world(course, mass=26, marker_dir=None):
    sdf = ET.fromstring(
        """<sdf version="1.9"><world name="race">
      <physics name="flight" type="ignored">
        <max_step_size>0.002</max_step_size><real_time_factor>1</real_time_factor>
      </physics>
      <gravity>0 0 -9.80665</gravity>
      <spherical_coordinates>
        <surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation>
        <latitude_deg>35</latitude_deg><longitude_deg>139</longitude_deg>
        <elevation>0</elevation><heading_deg>0</heading_deg>
      </spherical_coordinates>
      <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
      <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
      <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"><dynamic_pose_hertz>200</dynamic_pose_hertz></plugin>
      <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
      <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
        <render_engine>ogre2</render_engine>
      </plugin>
      <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.65 0.72 0.80 1</background>
        <shadows>false</shadows></scene>
      <light name="ceiling_light" type="directional">
        <pose>0 0 6 0 0 0</pose><diffuse>0.8 0.8 0.8 1</diffuse>
        <specular>0.1 0.1 0.1 1</specular><direction>0.35 -0.2 -1</direction>
      </light></world></sdf>"""
    )
    world = sdf.find("world")
    (xmin, xmax), (ymin, ymax), (floor, ceiling) = course["bounds"]
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    length, width = xmax - xmin, ymax - ymin
    room = static_link(world, "room", (0, 0, 0, 0, 0, 0))
    box(room, "floor", (length + 0.2, width + 0.2, 0.1), (cx, cy, floor - 0.05), "0.23 0.28 0.32 1")
    for name, y in (("left", ymax + 0.05), ("right", ymin - 0.05)):
        box(
            room,
            name,
            (length, 0.1, ceiling - floor),
            (cx, y, (ceiling + floor) / 2),
            "0.025 0.03 0.035 1",
        )
    for name, x in (("far", xmax + 0.05), ("near", xmin - 0.05)):
        box(
            room,
            name,
            (0.1, width + 0.2, ceiling - floor),
            (x, cy, (ceiling + floor) / 2),
            "0.025 0.03 0.035 1",
        )
    # Floor markings provide optical flow during a level approach.
    for i in range(int(length / 0.5)):
        box(
            room,
            f"floor_stripe_{i}",
            (0.018, width, 0.001),
            (xmin + (i + 1) * 0.5, cy, floor + 0.001),
            DARK,
            collision=False,
        )

    visual = ET.SubElement(room, "visual", name="gate_corner_markings")
    mesh = ET.SubElement(ET.SubElement(visual, "geometry"), "mesh")
    uri = (marker_dir / "white.obj").as_uri() if marker_dir else "model://course_markers/white.obj"
    ET.SubElement(mesh, "uri").text = uri
    material = ET.SubElement(visual, "material")
    for prop in ("ambient", "diffuse", "emissive"):
        ET.SubElement(material, prop).text = "1 1 1 1"
    rail = course["rail_thickness"]
    for gate in course["gates"]:
        w, h = gate["opening"]
        x, y, z = gate["center"]
        link = static_link(world, gate["name"], (x, y, z, 0, 0, gate["yaw"]))
        for i, (side, up) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            box(
                link,
                f"corner_{i}",
                (0.008, 0.375, 0.375),
                (-0.035, side * (w / 2 + 0.19), up * (h / 2 + 0.19)),
                "0.005 0.005 0.005 1",
                collision=False,
            )
        for sign, side in ((1, "left"), (-1, "right")):
            offset = sign * (w + rail) / 2
            box(link, side, (rail, rail, h + 2 * rail), (0, offset, 0), ORANGE)
            foot_height = z - h / 2 - rail - floor
            if foot_height > 0:
                box(
                    link,
                    side + "_leg",
                    (0.025, 0.025, foot_height),
                    (0, offset, floor + foot_height / 2 - z),
                    DARK,
                )
        for sign, name in ((1, "top"), (-1, "bottom")):
            box(link, name, (rail, w, rail), (0, 0, sign * (h + rail) / 2), ORANGE)

    vehicle = ET.SubElement(world, "include")
    ET.SubElement(vehicle, "uri").text = "model://whoop"
    ET.SubElement(vehicle, "name").text = "whoop"
    ET.SubElement(vehicle, "pose").text = numbers((*course["spawn"], 0, 0, course["spawn_yaw"]))
    camera_angle = course.get("camera_angle_deg", DEFAULTS["camera_angle_deg"])
    if mass != 26 or camera_angle != DEFAULTS["camera_angle_deg"]:
        model = ET.parse(ROOT / "models/whoop/model.sdf").getroot().find("model")
        for name in ("sensor[@name='fpv']", "visual[@name='camera_housing']"):
            pose = model.find(f"link[@name='body']/{name}/pose")
            fields = pose.text.split()
            fields[4] = str(-radians(camera_angle))  # Positive mount angle points the FLU camera up.
            pose.text = " ".join(fields)
        body = model.find("link/inertial")
        ratio = (mass / 1000 - 0.0004) / float(body.findtext("mass"))
        body.find("mass").text = str(mass / 1000 - 0.0004)
        for axis in body.find("inertia"):
            axis.text = str(float(axis.text) * ratio)
        ET.SubElement(model, "pose").text = vehicle.findtext("pose")
        world.remove(vehicle)
        world.append(model)
    ET.indent(sdf, space="  ")
    return ET.tostring(sdf, encoding="unicode", xml_declaration=True) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Exit nonzero if the world needs regeneration."
    )
    args = parser.parse_args()
    course = course_for(settings())
    result = build_world(course)
    mesh_file = ROOT / "models/course_markers/white.obj"
    mesh_data = marker_mesh(course)
    if args.check:
        if result != WORLD.read_text() or mesh_data != mesh_file.read_text():
            raise SystemExit("Regenerate sim/worlds/race.sdf with python3 sim/build_course.py")
    else:
        WORLD.write_text(result)
        mesh_file.write_text(mesh_data)
