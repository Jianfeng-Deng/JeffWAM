# Render crisp 3-panel side-by-side MuJoCo comparison video (GT_demo vs WAM_v1_s90 vs blind_t1_s90 under M3 MIT control)
# across 3 representative held-out dev windows (Release event, Grasp event, and 15mm Perturbation Recovery).
import os, sys, json, math, subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for sub in ("code_e132", "code_e131", "code_e130", "code_e129"):
    sys.path.insert(0, str(T / sub))

from build_bimanual_mjcf import ASSETS, PIPER_JSON
from run_e131_sim import build_sim_proprio_batch, fit_blind_t1_model, predict_blind_t1, JI
from run_e132_fix_and_ori import BimanualPiperSimV2, pose18_to_pos_and_R, geodesic_deg
from e129_common import LOCK_PATH, GR, UpgradedWAM, build_or_load_extra_stats
from e99w2_model import pose_torch
from e99w2_data import pose_np
from e102x_setup import datasets
from e102a_contract import contract
from e110f_core import infer_d
from e110f_eval import float_batch
from e86.data import move, collate

OUT_DIR = T / "results/e132_video_20261119_02"
VIS_XML = ASSETS / "bimanual_piper_visual_render.xml"


def build_clean_visual_xml():
    """
    Build a visually crisp MuJoCo station with exact e98w_core.CHAIN kinematics, commissioned piper.json bases & table,
    clean articulated cylinder/capsule link geometry + dark motor joints + parallel gripper fingers + mocap GT target spheres!
    """
    os.environ.update(E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
    for p in ("code_e110f", "code_e106c/run02", "code_e106c", "code_e102a"):
        if str(T / p) not in sys.path:
            sys.path.insert(0, str(T / p))
    from e98w_core import CHAIN

    cfg = json.loads(PIPER_JSON.read_text())
    chain_by_link = {}
    j_num = 0
    for _, typ, To, _ in CHAIN:
        if typ == "revolute":
            j_num += 1
            pos = To[:3, 3]
            quat = np.zeros(4, dtype="f8")
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(To[:3, :3], dtype="f8").reshape(9))
            chain_by_link[f"link{j_num}"] = (pos, quat)

    src_root = ET.parse(ASSETS / "bimanual_piper_station.xml").getroot()
    mj = ET.Element("mujoco", model="bimanual_piper_visual")
    ET.SubElement(mj, "compiler", angle="radian")
    ET.SubElement(mj, "option", timestep=f"{1.0/300.0:.8f}", gravity="0 0 -9.81", integrator="implicitfast")
    ET.SubElement(mj, "size", njmax="2000", nconmax="500")

    vis = ET.SubElement(mj, "visual")
    ET.SubElement(vis, "global", offwidth="960", offheight="720")
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45", diffuse="0.55 0.55 0.55", specular="0.2 0.2 0.2")
    ET.SubElement(vis, "rgba", haze="0.15 0.20 0.28 1")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.88 0.91 0.96", rgb2="0.68 0.74 0.84", width="512", height="512")
    ET.SubElement(asset, "texture", name="grid_tex", type="2d", builtin="checker", rgb1="0.74 0.72 0.68", rgb2="0.66 0.64 0.60", width="512", height="512", mark="edge", markrgb="0.52 0.50 0.46")
    ET.SubElement(asset, "material", name="table_mat", texture="grid_tex", texrepeat="8 7", reflectance="0.05")
    ET.SubElement(asset, "material", name="arm_L", rgba="0.22 0.46 0.78 1", specular="0.4", shininess="0.5")
    ET.SubElement(asset, "material", name="arm_R", rgba="0.20 0.58 0.52 1", specular="0.4", shininess="0.5")
    ET.SubElement(asset, "material", name="joint_mat", rgba="0.18 0.20 0.24 1", specular="0.3", shininess="0.4")
    ET.SubElement(asset, "material", name="finger_mat", rgba="0.92 0.55 0.16 1", specular="0.3", shininess="0.4")

    wb = ET.SubElement(mj, "worldbody")
    ET.SubElement(wb, "light", pos="0.0 -0.6 1.8", dir="0.0 0.4 -1.0", diffuse="0.70 0.70 0.70", castshadow="true")
    ET.SubElement(wb, "light", pos="0.0 0.9 1.6", dir="0.0 -0.3 -1.0", diffuse="0.40 0.40 0.40", castshadow="false")

    # Wide 3/4 overview camera framing both Piper arms from bases to gripper tips + green GT spheres
    ET.SubElement(wb, "camera", name="overview", pos="0.0 -0.82 0.88", xyaxes="1 0 0 0 0.52 0.85", fovy="52")

    tc = cfg["table"]["center_m"]
    ts = cfg["table"]["half_size_m"]
    ET.SubElement(wb, "geom", name="table_top", type="box", pos=f"{tc[0]} {tc[1]} {tc[2]}", size=f"{ts[0]} {ts[1]} {ts[2]}", material="table_mat", contype="1", conaffinity="6")

    # Mocap spheres showing real-time Ground-Truth Left/Right TCP target positions (translucent bright green!)
    for side in ("L", "R"):
        mb = ET.SubElement(wb, "body", name=f"gt_target_{side}", mocap="true", pos="0 0.3 0.2")
        ET.SubElement(mb, "geom", name=f"gt_sphere_{side}", type="sphere", size="0.015", rgba="0.15 0.88 0.25 0.60", contype="0", conaffinity="0")

    act = ET.SubElement(mj, "actuator")

    # Copy inertial/joint structure from bimanual_piper_station.xml but replace blocky 48-tri STLs with crisp articulated link geoms!
    def add_arm(side, prefix, mat_name, ct, ca):
        arm_cfg = cfg["arms"][side]
        bx, by, bz = arm_cfg["base_xyz_m"]
        rx, ry, rz = [math.radians(v) for v in arm_cfg["base_rpy_deg"]]
        base_b = ET.SubElement(wb, "body", name=f"{prefix}base_link", pos=f"{bx} {by} {bz}", euler=f"{rx} {ry} {rz}")
        ET.SubElement(base_b, "geom", name=f"{prefix}base_cyl", type="cylinder", pos="0 0 0.045", size="0.055 0.045", material="joint_mat", contype="0", conaffinity="0")

        # Recursively clone links 1..6 + 7,8 from bimanual_piper_station.xml
        src_base = [b for b in src_root.find("worldbody").findall("body") if b.get("name") == f"{prefix}base_link"][0]
        src_l1 = src_base.find("body")

        def build_link(elem):
            bname = elem.get("name")
            short = bname[len(prefix):]
            b = ET.Element("body", name=bname, pos=elem.get("pos", "0 0 0"))
            if elem.get("quat"):
                b.set("quat", elem.get("quat"))
            for ch in elem:
                if ch.tag in ("inertial", "joint", "site"):
                    ET.SubElement(b, ch.tag, **ch.attrib)
            # Clean parametric visual+collision geometry matched to exact Piper link dimensions
            if short == "link1":
                ET.SubElement(b, "geom", type="cylinder", pos="0 0 -0.02", size="0.042 0.04", material=mat_name, contype="0", conaffinity="0")
            elif short == "link2":
                ET.SubElement(b, "geom", type="cylinder", pos="0 0 0", size="0.036 0.042", material="joint_mat", contype=str(ct), conaffinity=str(ca))
                ET.SubElement(b, "geom", type="capsule", fromto="0 0 0 0.285 0 0", size="0.028", material=mat_name, contype=str(ct), conaffinity=str(ca))
            elif short == "link3":
                ET.SubElement(b, "geom", type="cylinder", pos="0 0 0", size="0.032 0.038", material="joint_mat", contype=str(ct), conaffinity=str(ca))
                ET.SubElement(b, "geom", type="capsule", fromto="0 0 0 -0.022 -0.250 0", size="0.023", material=mat_name, contype=str(ct), conaffinity=str(ca))
            elif short == "link4":
                ET.SubElement(b, "geom", type="cylinder", pos="0 0 0", size="0.026 0.030", material="joint_mat", contype=str(ct), conaffinity=str(ca))
            elif short == "link5":
                ET.SubElement(b, "geom", type="cylinder", pos="0 0 0", size="0.024 0.028", material="joint_mat", contype=str(ct), conaffinity=str(ca))
                ET.SubElement(b, "geom", type="capsule", fromto="0 0 0 0 -0.091 0", size="0.020", material=mat_name, contype=str(ct), conaffinity=str(ca))
            elif short == "link6":
                ET.SubElement(b, "geom", type="cylinder", pos="0 0 0.022", size="0.026 0.024", material="joint_mat", contype=str(ct), conaffinity=str(ca))
                ET.SubElement(b, "geom", type="box", pos="0 0 0.062", size="0.018 0.042 0.020", material="joint_mat", contype=str(ct), conaffinity=str(ca))
                # Wrist camera housing visual cue
                ET.SubElement(b, "geom", type="box", pos="0 0.055 0.035", size="0.016 0.012 0.014", rgba="0.12 0.12 0.14 1", contype="0", conaffinity="0")
                # Red live TCP indicator dot
                ET.SubElement(b, "geom", type="sphere", pos="0 0 0.1358", size="0.008", rgba="0.95 0.18 0.18 0.9", contype="0", conaffinity="0")
            elif short in ("link7", "link8"):
                # Parallel gripper finger pad extending to tip TCP
                ET.SubElement(b, "geom", type="box", pos="0 -0.028 0.006", size="0.008 0.030 0.005", material="finger_mat", contype=str(ct), conaffinity=str(ca))

            for ch in elem.findall("body"):
                b.append(build_link(ch))
            return b

        base_b.append(build_link(src_l1))
        for j in range(1, 7):
            ET.SubElement(act, "motor", name=f"{prefix}motor{j}", joint=f"{prefix}joint{j}", ctrllimited="true", ctrlrange="-35 35", gear="1")
        ET.SubElement(act, "position", name=f"{prefix}grip7", joint=f"{prefix}joint7", ctrllimited="true", ctrlrange="0 0.05", kp="500", forcerange="-40 40")
        ET.SubElement(act, "position", name=f"{prefix}grip8", joint=f"{prefix}joint8", ctrllimited="true", ctrlrange="-0.05 0", kp="500", forcerange="-40 40")

    add_arm("left", "L_", "arm_L", 2, 5)
    add_arm("right", "R_", "arm_R", 4, 3)

    ET.indent(mj, space="  ")
    VIS_XML.write_text(ET.tostring(mj, encoding="unicode") + "\n")
    return VIS_XML


def draw_hud(frame_np, title, subtitle, metrics_line, color_rgb=(35, 110, 210)):
    im = Image.fromarray(frame_np)
    dr = ImageDraw.Draw(im)
    w, h = im.size
    # Clean semi-opaque dark top & bottom bars
    dr.rectangle([0, 0, w, 46], fill=(22, 26, 34))
    dr.rectangle([0, 0, w, 4], fill=color_rgb)
    dr.rectangle([0, h - 30, w, h], fill=(22, 26, 34))
    font = ImageFont.load_default()
    dr.text((10, 8), title, fill=(255, 255, 255), font=font)
    dr.text((10, 26), subtitle, fill=(195, 210, 230), font=font)
    dr.text((10, h - 21), metrics_line, fill=(245, 245, 245), font=font)
    return np.asarray(im)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vis_xml = build_clean_visual_xml()
    piper_cfg = json.loads(PIPER_JSON.read_text())

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    _, _, tr, dev = datasets()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    build_or_load_extra_stats(tr, dev, entries)
    st_json = json.loads((T / "results/e129_control_20261119_01/extra_proprio_stats.json").read_text())
    extra_stats = dict(mean66=np.asarray(st_json["mean66"], "f4"), std66=np.asarray(st_json["std66"], "f4"))
    c_dict = contract()

    # Load WAM_v1_s90 and blind_t1_s90
    cp = torch.load(T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt", map_location="cpu", weights_only=False)
    wam_v1 = UpgradedWAM(tr.norm, variant="v1").widen_from_164()
    wam_v1.load_state_dict(cp["model"], strict=True)
    wam_v1 = wam_v1.cuda().eval()
    blind_t1 = fit_blind_t1_model(tr)

    # Pick 3 informative held-out windows:
    #   1) A gripper release (open) event window (with large motion)
    #   2) A gripper grasp (close) event window (with large motion)
    #   3) A 15mm initial-perturbation recovery window
    open_cands, close_cands = [], []
    for wi, e in enumerate(entries):
        b0 = float_batch(dev.batch([e["index"]], (10,), "short"))
        gl = b0["grip_label"][0].numpy()  # (50, 2)
        gc = (b0["grip_current"][0].numpy() > 0.5)
        gt_a = b0["action"][0].numpy()
        disp = float(np.abs(gt_a[-1, JI] - b0["q_current"][0].numpy()[JI]).mean())
        for side in (0, 1):
            if (not gc[side]) and (gl[:, side] > 0.5).any():
                open_cands.append((disp, wi))
            if gc[side] and (gl[:, side] < 0.5).any():
                close_cands.append((disp, wi))
    open_cands.sort(reverse=True)
    close_cands.sort(reverse=True)
    wi_open = open_cands[0][1]
    wi_close = close_cands[0][1]
    wi_pert = close_cands[min(2, len(close_cands) - 1)][1]

    clips = [
        dict(name="Clip 1/3: S2 Closed-Loop (Gripper Release Window)", wi=wi_open, perturb=False),
        dict(name="Clip 2/3: S2 Closed-Loop (Gripper Grasp Window)", wi=wi_close, perturb=False),
        dict(name="Clip 3/3: S4 Generalization (+15mm Initial Offset Recovery)", wi=wi_pert, perturb=True),
    ]

    rng_pert = np.random.default_rng(131)
    PERT_Q14 = np.zeros((len(entries), 14), dtype="f4")
    for wi in range(len(entries)):
        dq = np.zeros(14, dtype="f4")
        dq[[0, 1, 2, 7, 8, 9]] = rng_pert.choice([-0.03, 0.03], size=6)
        dq[[3, 4, 5, 10, 11, 12]] = rng_pert.choice([-0.02, 0.02], size=6)
        PERT_Q14[wi] = dq

    # Build 3 sims (or reuse 1 sim + 1 EGL renderer cleanly!)
    sim = BimanualPiperSimV2(vis_xml, piper_cfg)
    renderer = mujoco.Renderer(sim.m, height=360, width=480)

    def rollout_and_render(wi, model_kind, perturb):
        e = entries[wi]
        z, q_ep, p_ep, good = dev.states.get(dev, e["cid"])
        a0 = e["anchor"]
        b0 = float_batch(dev.batch([e["index"]], (10,), "short"))
        vis_b = {0: b0}
        for off in (10, 25):
            sh = dev.shifted(e["index"], off, "short")
            if sh is not None:
                vis_b[off] = float_batch(collate([sh]))

        gt_act = b0["action"][0].numpy()
        with torch.no_grad():
            p18_gt = pose_torch(torch.from_numpy(gt_act)[None].cuda().double()).cpu().numpy()[0]
        gt_pos, gt_R = pose18_to_pos_and_R(p18_gt)  # (50, 2, 3), (50, 2, 3, 3)

        # World transforms of L/R bases for placing green GT target mocap spheres
        q_hist = q_ep.copy()
        p_hist = p_ep.copy()
        if perturb:
            q_init = np.clip(q_ep[a0] + PERT_Q14[wi], -3.0, 3.0)
            q_init[JI] = np.clip(q_init[JI], sim.lo12, sim.hi12)
            q_init[GR] = q_ep[a0, GR]
            q_hist[max(0, a0 - 24) : a0 + 1] = q_init[None]
            p_hist[max(0, a0 - 24) : a0 + 1] = pose_np(q_init[None])
        else:
            q_init = q_ep[a0].copy()

        sim.reset(q_init)
        R_wb_L = sim.d.xmat[sim.base_l].reshape(3, 3).copy()
        p_wb_L = sim.d.xpos[sim.base_l].copy()
        R_wb_R = sim.d.xmat[sim.base_r].reshape(3, 3).copy()
        p_wb_R = sim.d.xpos[sim.base_r].copy()

        replan_set = {0, 10, 25}
        cur_chunk = None
        chunk_t0 = 0
        q_ref = q_init.copy()

        frames = []
        for step in range(50):
            if step in replan_set:
                if model_kind == "gt":
                    cur_chunk = gt_act[step:]
                else:
                    if model_kind == "blind":
                        raw_chunk = predict_blind_t1(blind_t1, q_hist, a0 + step)
                    else:
                        vis_off = 25 if (step >= 25 and 25 in vis_b) else (10 if (step >= 10 and 10 in vis_b) else 0)
                        b_live = build_sim_proprio_batch(vis_b[vis_off], q_hist, p_hist, a0 + step, z, good, c_dict, extra_stats, dev.base.stats, True)
                        with torch.inference_mode():
                            raw_chunk = infer_d(wam_v1, move(b_live, "cuda"), 4, 90)[0].float().cpu().numpy()
                    if step > 0:
                        q_sim_cur = q_hist[a0 + step]
                        q_anchor = q_sim_cur.copy()
                        q_anchor[JI] = q_sim_cur[JI] + np.clip(q_ref[JI] - q_sim_cur[JI], -0.08, 0.08)
                        cur_chunk = raw_chunk.copy()
                        cur_chunk[:, JI] = q_anchor[None, JI] + (raw_chunk[:, JI] - q_sim_cur[None, JI])
                    else:
                        cur_chunk = raw_chunk
                chunk_t0 = step

            cmd = cur_chunk[min(step - chunk_t0, len(cur_chunk) - 1)]
            q_ref = cmd.copy()
            # Update mocap GT target spheres in world coordinates
            sim.d.mocap_pos[0] = p_wb_L + R_wb_L @ gt_pos[step, 0]
            sim.d.mocap_pos[1] = p_wb_R + R_wb_R @ gt_pos[step, 1]

            st_out = sim.step_frame_v2(cmd, use_vff=True)
            if a0 + step + 1 < len(q_hist):
                q_hist[a0 + step + 1] = st_out["q14"]
                p_hist[a0 + step + 1] = pose_np(st_out["q14"][None])[0]

            pos_err_mm = float(np.linalg.norm(st_out["pos"] - gt_pos[step], axis=-1).mean() * 1000.0)
            rot_err_deg = float(geodesic_deg(st_out["R"], gt_R[step]).mean())

            renderer.update_scene(sim.d, camera="overview")
            rgb = renderer.render()
            frames.append((rgb, pos_err_mm, rot_err_deg, st_out["peak_tau"]))
        return frames

    all_video_frames = []
    preview_frames = []

    panel_specs = [
        ("gt", "1. GT Demo (MIT Floor)", (40, 165, 85)),
        ("wam", "2. WAM_v1_s90 (M3 Closed-Loop)", (35, 115, 220)),
        ("blind", "3. blind_t1_s90 (No Vision)", (215, 80, 55)),
    ]

    for clip in clips:
        print("Rendering", clip["name"], "win=", clip["wi"], flush=True)
        panel_runs = [rollout_and_render(clip["wi"], mk, clip["perturb"]) for mk, _, _ in panel_specs]
        for step in range(50):
            row_imgs = []
            for (mk, label, col), run_f in zip(panel_specs, panel_runs):
                rgb, err_mm, rot_d, tau = run_f[step]
                sub = f"{clip['name']}  |  t={step+1:02d}/50"
                met = f"TCP Err: {err_mm:5.1f} mm  |  Rot: {rot_d:4.1f} deg  |  Tau: {tau:4.1f} Nm"
                row_imgs.append(draw_hud(rgb, label, sub, met, color_rgb=col))
            comp = np.concatenate(row_imgs, axis=1)  # (360, 1440, 3)
            all_video_frames.append(comp)
            if step in (0, 24, 49):
                preview_frames.append(comp)
        # Hold final frame for 12 frames (~0.6s at 20fps) for readability
        for _ in range(12):
            all_video_frames.append(all_video_frames[-1])

    # Save contact sheet PNG (3 key frames: end of Clip 1, Clip 2, Clip 3)
    contact = np.concatenate([preview_frames[2], preview_frames[5], preview_frames[8]], axis=0)
    Image.fromarray(contact).save(OUT_DIR / "preview_contact_sheet.png")

    # Encode crisp compact H.264 MP4 via ffmpeg (20 fps, yuv420p for QuickTime/macOS/browser)
    mp4_path = OUT_DIR / "e132_bimanual_piper_mujoco_demo.mp4"
    h, w, _ = all_video_frames[0].shape
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", "20", "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium",
        "-movflags", "+faststart", str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for fr in all_video_frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="ignore")
    rc = proc.wait()
    assert rc == 0, f"ffmpeg failed: {err[-1000:]}"
    renderer.close()
    print(f"Rendered MP4 OK: {mp4_path} ({mp4_path.stat().st_size / 1024:.1f} KB, {len(all_video_frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
