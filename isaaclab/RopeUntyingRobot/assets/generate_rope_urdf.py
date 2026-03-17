"""Generate a parameterized articulated-chain URDF for the rope.

Each segment is a rigid cylinder connected by 2-DOF joints (pitch + yaw)
via massless hinge links. Colors interpolate across Green->Yellow->Red->Blue
bands to match the real rope's appearance.

Run as ``__main__`` for standalone URDF generation.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

# Color bands matching the real rope: Green -> Yellow -> Red -> Blue
_COLOR_BANDS: list[tuple[float, float, float, float]] = [
    (0.0, 0.50, 0.0, 1.0),  # Green
    (1.0, 1.00, 0.0, 1.0),  # Yellow
    (1.0, 0.00, 0.0, 1.0),  # Red
    (0.0, 0.00, 1.0, 1.0),  # Blue
]

# 90-deg rotation around Y to align URDF Z-axis cylinder with link X-axis
_CYL_RPY = f"0 {math.pi / 2:.10f} 0"


def _interpolate_color(t: float) -> tuple[float, float, float, float]:
    """Interpolate across color bands for parameter *t* in [0, 1]."""
    n = len(_COLOR_BANDS)
    scaled = t * (n - 1)
    idx = min(int(scaled), n - 2)
    frac = scaled - idx
    c0, c1 = _COLOR_BANDS[idx], _COLOR_BANDS[idx + 1]
    return (
        c0[0] + (c1[0] - c0[0]) * frac,
        c0[1] + (c1[1] - c0[1]) * frac,
        c0[2] + (c1[2] - c0[2]) * frac,
        c0[3] + (c1[3] - c0[3]) * frac,
    )


def _cylinder_inertia(
    mass: float, radius: float, length: float,
) -> tuple[float, float, float]:
    """Principal inertias (Ixx, Iyy, Izz) for a cylinder along the X axis."""
    ixx = 0.5 * mass * radius ** 2
    iyy = (1.0 / 12.0) * mass * (3.0 * radius ** 2 + length ** 2)
    return ixx, iyy, iyy


def generate_rope_urdf(
    num_segments: int = 16,
    segment_radius: float = 0.005,
    total_length: float = 0.45,
    density: float = 5000.0,
    joint_limit_deg: float = 30.0,
    output_path: Path | None = None,
) -> Path:
    """Generate an articulated rope chain URDF.

    Args:
        num_segments: Number of rigid cylinder segments.
        segment_radius: Radius of each cylinder in metres.
        total_length: Total rope length in metres.
        density: Material density in kg/m^3 (inflated for PhysX stability).
        joint_limit_deg: Joint limit per revolute axis in degrees.
        output_path: Where to write the URDF. Defaults to
            ``rope_chain.urdf`` beside this script.

    Returns:
        Path to the generated URDF file.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "rope_chain.urdf"

    seg_len = total_length / num_segments
    seg_vol = math.pi * segment_radius ** 2 * seg_len
    seg_mass = density * seg_vol
    ixx, iyy, izz = _cylinder_inertia(seg_mass, segment_radius, seg_len)
    joint_limit_rad = math.radians(joint_limit_deg)

    robot = Element("robot", name="rope_chain")

    for i in range(num_segments):
        seg_name = f"rope_seg_{i}"
        link = SubElement(robot, "link", name=seg_name)

        # -- inertial (at cylinder centre of mass) --
        inertial = SubElement(link, "inertial")
        SubElement(
            inertial, "origin",
            xyz=f"{seg_len / 2:.6f} 0 0", rpy="0 0 0",
        )
        SubElement(inertial, "mass", value=f"{seg_mass:.8f}")
        SubElement(
            inertial, "inertia",
            ixx=f"{ixx:.10f}", ixy="0", ixz="0",
            iyy=f"{iyy:.10f}", iyz="0", izz=f"{izz:.10f}",
        )

        # -- color --
        t = i / max(num_segments - 1, 1)
        r, g, b, a = _interpolate_color(t)

        # -- visual --
        visual = SubElement(link, "visual")
        SubElement(
            visual, "origin",
            xyz=f"{seg_len / 2:.6f} 0 0", rpy=_CYL_RPY,
        )
        geom_v = SubElement(visual, "geometry")
        SubElement(
            geom_v, "cylinder",
            radius=f"{segment_radius:.6f}", length=f"{seg_len:.6f}",
        )
        mat = SubElement(visual, "material", name=f"rope_color_{i}")
        SubElement(mat, "color", rgba=f"{r:.4f} {g:.4f} {b:.4f} {a:.4f}")

        # -- collision --
        collision = SubElement(link, "collision")
        SubElement(
            collision, "origin",
            xyz=f"{seg_len / 2:.6f} 0 0", rpy=_CYL_RPY,
        )
        geom_c = SubElement(collision, "geometry")
        SubElement(
            geom_c, "cylinder",
            radius=f"{segment_radius:.6f}", length=f"{seg_len:.6f}",
        )

        # -- hinge link + joints to next segment --
        if i < num_segments - 1:
            hinge_name = f"rope_hinge_{i}"
            hinge_link = SubElement(robot, "link", name=hinge_name)
            h_inertial = SubElement(hinge_link, "inertial")
            SubElement(h_inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            SubElement(h_inertial, "mass", value="1e-6")
            SubElement(
                h_inertial, "inertia",
                ixx="1e-12", ixy="0", ixz="0",
                iyy="1e-12", iyz="0", izz="1e-12",
            )
            # Visual sphere at the hinge joint so the USD converter
            # creates valid visual prims (prevents unresolved references).
            h_visual = SubElement(hinge_link, "visual")
            SubElement(h_visual, "origin", xyz="0 0 0", rpy="0 0 0")
            h_geom = SubElement(h_visual, "geometry")
            SubElement(h_geom, "sphere", radius=f"{segment_radius:.6f}")
            h_mat = SubElement(h_visual, "material", name=f"hinge_color_{i}")
            SubElement(h_mat, "color", rgba=f"{r:.4f} {g:.4f} {b:.4f} {a:.4f}")

            # Pitch joint: seg_i -> hinge_i (rotate around Y)
            pitch = SubElement(
                robot, "joint",
                name=f"rope_pitch_{i}", type="revolute",
            )
            SubElement(pitch, "parent", link=seg_name)
            SubElement(pitch, "child", link=hinge_name)
            SubElement(
                pitch, "origin",
                xyz=f"{seg_len:.6f} 0 0", rpy="0 0 0",
            )
            SubElement(pitch, "axis", xyz="0 1 0")
            SubElement(
                pitch, "limit",
                lower=f"{-joint_limit_rad:.6f}",
                upper=f"{joint_limit_rad:.6f}",
                effort="0.1", velocity="10.0",
            )
            SubElement(pitch, "dynamics", damping="0.5", friction="0.0")

            # Yaw joint: hinge_i -> seg_{i+1} (rotate around Z)
            yaw = SubElement(
                robot, "joint",
                name=f"rope_yaw_{i}", type="revolute",
            )
            SubElement(yaw, "parent", link=hinge_name)
            SubElement(yaw, "child", link=f"rope_seg_{i + 1}")
            SubElement(yaw, "origin", xyz="0 0 0", rpy="0 0 0")
            SubElement(yaw, "axis", xyz="0 0 1")
            SubElement(
                yaw, "limit",
                lower=f"{-joint_limit_rad:.6f}",
                upper=f"{joint_limit_rad:.6f}",
                effort="0.1", velocity="10.0",
            )
            SubElement(yaw, "dynamics", damping="0.5", friction="0.0")

    # Pretty-print
    raw_xml = tostring(robot, encoding="unicode")
    pretty = parseString(raw_xml).toprettyxml(indent="  ")
    lines = pretty.split("\n")
    if lines and lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" encoding="utf-8"?>'
    urdf_str = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(urdf_str, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    path = generate_rope_urdf()
    n_seg = 16
    n_hinge = n_seg - 1
    print(f"Generated rope URDF: {path}")
    print(f"  Segments: {n_seg}, Hinges: {n_hinge}")
    print(f"  Total links: {n_seg + n_hinge}, Total joints: {2 * n_hinge}")
