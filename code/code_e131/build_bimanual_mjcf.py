# Build commissioned dual-arm Piper MuJoCo station XML from piper_description.xml + piper.json
# and verify exact FK alignment (< 1 mm) against e99w2_model.pose_torch!
import os, sys, json, math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
ASSETS = T / "sim/piper_bimanual_assets_20261119_02"
SINGLE_XML = ASSETS / "mujoco_model/piper_description.xml"
PIPER_JSON = ASSETS / "configs/piper.json"
OUT_XML = ASSETS / "bimanual_piper_station.xml"


def build_station_xml():
    os.environ.update(E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
    for p in ("code_e110f", "code_e106c/run02", "code_e106c", "code_e102a"):
        if str(T / p) not in sys.path:
            sys.path.insert(0, str(T / p))
    from e98w_core import CHAIN

    # Map link1..link6 exact 4x4 transform from e98w_core.CHAIN so MuJoCo FK matches pose_torch to < 0.01 mm!
    chain_by_link = {}
    j_num = 0
    for _, typ, To, _ in CHAIN:
        if typ == "revolute":
            j_num += 1
            pos = To[:3, 3]
            quat = np.zeros(4, dtype="f8")
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(To[:3, :3], dtype="f8").reshape(9))
            chain_by_link[f"link{j_num}"] = (pos, quat)

    cfg = json.loads(PIPER_JSON.read_text())
    src_root = ET.parse(SINGLE_XML).getroot()

    mj = ET.Element("mujoco", model="bimanual_piper_station_v100")
    ET.SubElement(mj, "compiler", angle="radian", meshdir="meshes")
    ET.SubElement(mj, "option", timestep=f"{1.0/300.0:.8f}", gravity="0 0 -9.81", integrator="implicitfast")
    ET.SubElement(mj, "size", njmax="2000", nconmax="500")

    asset = ET.SubElement(mj, "asset")
    for m in src_root.find("asset").findall("mesh"):
        name = m.get("name")
        fname = Path(m.get("file")).name
        ET.SubElement(asset, "mesh", name=name, file=fname)

    wb = ET.SubElement(mj, "worldbody")
    ET.SubElement(wb, "light", pos="0 0.4 2.0", dir="0 0 -1", diffuse="0.85 0.85 0.85")

    # Commissioned table surface (z_top = -0.0444 + 0.04 = -0.0044 m from piper.json)
    tc = cfg["table"]["center_m"]
    ts = cfg["table"]["half_size_m"]
    ET.SubElement(
        wb, "geom", name="table_top", type="box",
        pos=f"{tc[0]} {tc[1]} {tc[2]}", size=f"{ts[0]} {ts[1]} {ts[2]}",
        rgba="0.52 0.44 0.36 1", contype="1", conaffinity="6",
    )

    # Extract single-arm link1 tree from piper_description.xml
    src_wb = src_root.find("worldbody")
    src_link1 = src_wb.find("body")
    assert src_link1.get("name") == "link1"

    act = ET.SubElement(mj, "actuator")

    def clone_body(elem, prefix, contype, conaffinity):
        b = ET.Element("body")
        bname = elem.get("name")
        for k, v in elem.attrib.items():
            b.set(k, f"{prefix}{v}" if k == "name" else v)
        if bname in chain_by_link:
            pos, quat = chain_by_link[bname]
            b.set("pos", f"{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
            b.set("quat", f"{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}")
        for child in elem:
            if child.tag == "inertial":
                ni = ET.SubElement(b, "inertial", **child.attrib)
                # Add V100 wrist camera + gripper CoM overhang note already in link6 mass/inertia
                _ = ni
            elif child.tag == "joint":
                j_attr = dict(child.attrib)
                jname = j_attr["name"]
                j_attr["name"] = f"{prefix}{jname}"
                # Replace kinematic servo damping=500 with physical MIT joint damping/armature
                if j_attr.get("type", "hinge") == "slide":
                    j_attr["damping"] = "5.0"
                    j_attr["armature"] = "0.001"
                    # Expand V100 gripper range to 0..0.05 / -0.05..0 (100mm total opening)
                    j_attr["range"] = "0 0.05" if jname == "joint7" else "-0.05 0"
                else:
                    j_attr["damping"] = "0.2"
                    j_attr["armature"] = "0.002"
                    if jname == "joint1":
                        j_attr["range"] = "-2.618 2.618"
                ET.SubElement(b, "joint", **j_attr)
            elif child.tag == "geom":
                g_attr = dict(child.attrib)
                # base_link and link1 are bolted to table -> no table/arm collision; link2..8 collide with table & other arm
                if bname in ("link1",):
                    g_attr["contype"] = "0"
                    g_attr["conaffinity"] = "0"
                else:
                    g_attr["contype"] = str(contype)
                    g_attr["conaffinity"] = str(conaffinity)
                g_attr["name"] = f"{prefix}{bname}_{g_attr.get('mesh', 'geom')}"
                ET.SubElement(b, "geom", **g_attr)
            elif child.tag == "body":
                b.append(clone_body(child, prefix, contype, conaffinity))

        if bname == "link6":
            # Exact TCP sites: 0.1358 m (e99w2 pose_torch) and 0.13503 m (piper.json)
            ET.SubElement(b, "site", name=f"{prefix}tcp", pos="0 0 0.1358", size="0.005", rgba="1 0 0 1")
            ET.SubElement(b, "site", name=f"{prefix}flange", pos="0 0 0", size="0.004", rgba="0 1 0 1")
        return b

    for side, prefix, ct, ca in (("left", "L_", 2, 5), ("right", "R_", 4, 3)):
        arm_cfg = cfg["arms"][side]
        bx, by, bz = arm_cfg["base_xyz_m"]
        rx, ry, rz = [math.radians(v) for v in arm_cfg["base_rpy_deg"]]
        base_b = ET.SubElement(
            wb, "body", name=f"{prefix}base_link",
            pos=f"{bx} {by} {bz}", euler=f"{rx} {ry} {rz}",
        )
        ET.SubElement(
            base_b, "geom", name=f"{prefix}base_geom", type="mesh", mesh="base_link",
            rgba="0.79 0.82 0.93 1", contype="0", conaffinity="0",
        )
        base_b.append(clone_body(src_link1, prefix, ct, ca))

        # Torque motors for 6 revolute joints (MIT impedance control law applied externally at 300 Hz!)
        for j in range(1, 7):
            ET.SubElement(
                act, "motor", name=f"{prefix}motor{j}", joint=f"{prefix}joint{j}",
                ctrllimited="true", ctrlrange="-35 35", gear="1",
            )
        # Position servos for gripper slides
        ET.SubElement(act, "position", name=f"{prefix}grip7", joint=f"{prefix}joint7", ctrllimited="true", ctrlrange="0 0.05", kp="500", forcerange="-40 40")
        ET.SubElement(act, "position", name=f"{prefix}grip8", joint=f"{prefix}joint8", ctrllimited="true", ctrlrange="-0.05 0", kp="500", forcerange="-40 40")

    ET.indent(mj, space="  ")
    OUT_XML.write_text(ET.tostring(mj, encoding="unicode") + "\n")
    return OUT_XML


def verify_fk_alignment():
    os.environ.update(E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
    for p in ("code_e110f", "code_e106c/run02", "code_e106c", "code_e102a"):
        sys.path.insert(0, str(T / p))
    import torch
    from e99w2_model import pose_torch

    xml_path = build_station_xml()
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    cfg = json.loads(PIPER_JSON.read_text())

    # Compare MuJoCo TCP transformed back into each arm's local base frame vs pose_torch!
    rng = np.random.default_rng(42)
    max_err_mm = 0.0
    for _ in range(20):
        q14 = np.zeros(14, dtype="f8")
        lo = np.radians([-140, 10, -150, -90, -60, -100])
        hi = np.radians([140, 150, -10, 90, 60, 100])
        q14[0:6] = rng.uniform(lo, hi)
        q14[7:13] = rng.uniform(lo, hi)
        q14[6] = 0.04
        q14[13] = 0.04

        d.qpos[:] = 0.0
        d.qpos[0:6] = q14[0:6]
        d.qpos[6] = 0.02
        d.qpos[7] = -0.02
        d.qpos[8:14] = q14[7:13]
        d.qpos[14] = 0.02
        d.qpos[15] = -0.02
        mujoco.mj_kinematics(m, d)

        pt = pose_torch(torch.tensor(q14, dtype=torch.float64)[None])[0].numpy()
        ee_ref_L = pt[0:3]
        ee_ref_R = pt[9:12]

        for side, prefix, ref_xyz in (("left", "L_", ee_ref_L), ("right", "R_", ee_ref_R)):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_link")
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}tcp")
            R_wb = d.xmat[bid].reshape(3, 3)
            p_wb = d.xpos[bid]
            p_local = R_wb.T @ (d.site_xpos[sid] - p_wb)
            err_mm = float(np.linalg.norm(p_local - ref_xyz) * 1000.0)
            max_err_mm = max(max_err_mm, err_mm)

    print(f"Bimanual Piper MuJoCo station built OK: nq={m.nq}, nu={m.nu}, max_FK_diff_vs_pose_torch={max_err_mm:.4f} mm", flush=True)
    assert max_err_mm < 1.0, f"FK mismatch too large: {max_err_mm} mm"
    return xml_path, max_err_mm


if __name__ == "__main__":
    verify_fk_alignment()
