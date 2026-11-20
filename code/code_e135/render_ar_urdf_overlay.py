# E135: Real-Video AR URDF Ghost-Arm & 3D Trajectory Overlay Verification
# Projects and renders calibrated 3D Piper URDF meshes + 50-step predicted 3D TCP trajectories
# directly onto the real RealSense dataset video (episode_62 / piper_01a709ceeae49b530c694814),
# comparing GT URDF registration, E134_FM_res (Flow Matching), E134_D_dec (Deterministic), and blind_t1!
import os, sys, re, json, math, time, subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import cv2
import torch
import mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for sub in ("code_e134", "code_e133", "code_e132", "code_e131", "code_e130", "code_e129"):
    if str(T / sub) not in sys.path:
        sys.path.insert(0, str(T / sub))

from e134_common import E134WAM, LOCK_PATH, brt, build_or_load_extra_stats
from train_e134 import ARM_CONFIGS
from build_bimanual_mjcf import PIPER_JSON
from render_e132_video import build_clean_visual_xml
from run_e131_sim import build_sim_proprio_batch, fit_blind_t1_model, predict_blind_t1, JI
from run_e132_fix_and_ori import BimanualPiperSimV2, pose18_to_pos_and_R, geodesic_deg
from e129_common import GR
from e99w2_model import pose_torch
from e99w2_data import pose_np
from e102x_setup import datasets
from e102a_contract import contract
from e110f_eval import float_batch
from e86.data import move, collate

OUT_DIR = T / "results/e135_ar_urdf_overlay_20261119_02"
CID = "piper_01a709ceeae49b530c694814"  # Held-out dev episode_62 (verified success, 71 cached SigLIP2 frames)


def load_mp4_rgb(mp4_path, max_frames=800):
    pr = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", mp4_path,
        ],
        text=True,
    )
    st = json.loads(pr)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    raw = subprocess.check_output(
        [
            "ffmpeg", "-v", "error", "-i", mp4_path,
            "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)


def build_pure_arm_ar_xml(fovy_init):
    vis_xml_path = build_clean_visual_xml()
    txt = Path(vis_xml_path).read_text()
    txt = re.sub(
        r'<texture[^>]*type="skybox"[^>]*/>',
        '<texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="32" height="32"/>',
        txt,
    )
    for pref in ("L_", "R_"):
        pos_m = re.search(rf'(<body name="{pref}link6"[^>]*>)', txt)
        if pos_m:
            cam_geom = (
                f'{pos_m.group(1)}\n'
                f'                <geom name="{pref}wrist_cam_mount" type="box" '
                f'pos="0 0.055 0.025" size="0.018 0.028 0.018" rgba="0.82 0.85 0.90 1" contype="0" conaffinity="0"/>'
            )
            txt = txt.replace(pos_m.group(1), cam_geom, 1)
    txt = txt.replace(
        '<camera name="overview"',
        f'<camera name="cam_ar" pos="0 0 0.6" quat="1 0 0 0" fovy="{fovy_init:.6f}"/>\n    <camera name="overview"',
    )
    out_p = Path("/tmp/e135_bimanual_ar_pure.xml")
    out_p.write_text(txt)
    return out_p


def configure_pure_arm_appearance(sim):
    """Hide all world/table/floor/mocap geoms (alpha=0, group=4) and make Piper links bright neutral 3D-shaded white."""
    scene_opt = mujoco.MjvOption()
    scene_opt.geomgroup[4] = 0
    for gi in range(sim.m.ngeom):
        bi = int(sim.m.geom_bodyid[gi])
        bname = mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, bi) or ""
        sim.m.geom_matid[gi] = -1
        if bname.startswith(("L_", "R_")):
            if "link7" in bname or "link8" in bname:
                sim.m.geom_rgba[gi] = [0.95, 0.96, 0.98, 1.0]
            else:
                sim.m.geom_rgba[gi] = [0.78, 0.82, 0.88, 1.0]
        else:
            sim.m.geom_rgba[gi] = [0.0, 0.0, 0.0, 0.0]
            sim.m.geom_group[gi] = 4
    return scene_opt


def calibrate_cam_high(sim, scene_opt, q_ep, vid_high, cam_offset):
    fx = 914.8648681640625 * 0.5
    fy = 913.3756713867188 * 0.5
    cx0 = 638.3804931640625 * 0.5
    cy0 = 376.7340087890625 * 0.5 + 60.0
    K0 = np.array([[fx, 0.0, cx0], [0.0, fy, cy0], [0.0, 0.0, 1.0]], dtype=np.float64)

    pts_3d = np.array([
        [-0.08519, 0.25385, 0.43253],
        [-0.19479, 0.17373, 0.43569],
        [-0.06636, 0.23501, 0.37708],
        [ 0.06268, 0.25296, 0.41539],
        [ 0.14914, 0.26507, 0.44108],
        [ 0.27422, 0.42315, 0.21069],
        [ 0.28563, 0.29483, 0.25366],
        [-0.07361, 0.22736, 0.46052],
        [-0.01026, 0.25794, 0.41646],
        [-0.01552, 0.20224, 0.43039],
        [ 0.11819, 0.22655, 0.45873],
    ], dtype=np.float64)
    pts_2d = np.array([
        [252.0, 246.0], [115.0, 280.0], [250.0, 294.0], [368.0, 282.0],
        [545.0, 195.0], [541.0, 252.0], [592.0, 290.0],
        [252.0, 232.0], [315.0, 248.0], [306.0, 282.0], [440.0, 265.0],
    ], dtype=np.float64)

    _, rvec0, tvec0 = cv2.solvePnP(pts_3d, pts_2d, K0, np.zeros(5), flags=cv2.SOLVEPNP_EPNP)
    _, rvec0, tvec0 = cv2.solvePnP(pts_3d, pts_2d, K0, np.zeros(5), rvec0, tvec0, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    bg_f = vid_high[60].astype(np.float32)
    calib_qs = [215, 242, 315, 385, 462, 525, 615, 705]
    real_masks_h = []
    for qa in calib_qs:
        vf = int(np.clip(qa + cam_offset, 0, len(vid_high) - 1))
        fr = vid_high[vf].astype(np.float32)
        diff_dark = (bg_f.mean(-1) - fr.mean(-1)) > 28.0
        is_dark = fr.mean(-1) < 102.0
        not_orange = (fr[:, :, 0] - fr[:, :, 2]) < 42.0
        m = diff_dark & is_dark & not_orange
        m[:62, :] = False
        m[418:, :] = False
        real_masks_h.append(cv2.resize(m.astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST) > 0)

    cam_id = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_ar")
    fovy0 = math.degrees(2.0 * math.atan(240.0 / fy))
    ren_fast = mujoco.Renderer(sim.m, height=240, width=320)

    def apply_cam(d9):
        rv = rvec0[:, 0] + d9[:3]
        tv = tvec0[:, 0] + d9[3:6]
        fv = float(fovy0 + d9[6])
        cx_c = float(cx0 + d9[7])
        cy_c = float(cy0 + d9[8])
        R_cw, _ = cv2.Rodrigues(rv)
        R_wc_cv = R_cw.T
        pos_w = -R_wc_cv @ tv
        R_wc_mj = R_wc_cv @ np.diag([1.0, -1.0, -1.0])
        q_xyzw = Rotation.from_matrix(R_wc_mj).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        sim.m.cam_pos[cam_id] = pos_w
        sim.m.cam_quat[cam_id] = q_wxyz
        sim.m.cam_fovy[cam_id] = fv
        M_f = np.float32([[1.0, 0.0, cx_c - 320.0], [0.0, 1.0, cy_c - 240.0]])
        M_h = np.float32([[1.0, 0.0, (cx_c - 320.0) * 0.5], [0.0, 1.0, (cy_c - 240.0) * 0.5]])
        fy_c = 240.0 / math.tan(math.radians(fv) * 0.5)
        K_c = np.array([[fy_c, 0.0, cx_c], [0.0, fy_c, cy_c], [0.0, 0.0, 1.0]], dtype=np.float64)
        return rv, tv, fv, cx_c, cy_c, pos_w, M_f, M_h, K_c

    def neg_iou(d9):
        _, _, _, _, _, _, _, M_h, _ = apply_cam(d9)
        inter = 0
        union = 0
        for qa, rm in zip(calib_qs, real_masks_h):
            sim.reset(q_ep[qa])
            ren_fast.update_scene(sim.d, camera="cam_ar", scene_option=scene_opt)
            rgb_h = ren_fast.render()
            m_mj = (rgb_h.max(axis=-1) > 10).astype(np.uint8)
            m_s = cv2.warpAffine(m_mj, M_h, (320, 240), flags=cv2.INTER_NEAREST) > 0
            m_s[:31, :] = False
            m_s[209:, :] = False
            inter += int(np.logical_and(m_s, rm).sum())
            union += int(np.logical_or(m_s, rm).sum())
        return -float(inter) / max(union, 1)

    iou_init = -neg_iou(np.zeros(9))
    res = minimize(
        neg_iou,
        np.zeros(9),
        method="Powell",
        bounds=[(-0.25, 0.25)] * 3 + [(-0.12, 0.12)] * 3 + [(-8.0, 8.0), (-20.0, 20.0), (-20.0, 20.0)],
        options={"maxiter": 350, "ftol": 1e-4},
    )
    ren_fast.close()
    rv, tv, fv, cx_c, cy_c, pos_w, M_f, _, K_c = apply_cam(res.x)
    iou_opt = -float(res.fun)
    print(
        f"Pure-arm Silhouette Calibration: IoU {iou_init:.4f} -> {iou_opt:.4f} | "
        f"fovy={fv:.2f}deg cx={cx_c:.1f} cy={cy_c:.1f} cam_pos={[round(x, 4) for x in pos_w]}",
        flush=True,
    )
    return dict(rvec=rv, tvec=tv, fovy=fv, cx=cx_c, cy=cy_c, cam_pos=pos_w.tolist(), M_full=M_f, K=K_c, iou_init=iou_init, iou_opt=iou_opt)


def build_dense_causal_vis_getter(dev, cid):
    """Build a 100% causal visual batch builder using all 71 immutable cached SigLIP2 rows of episode_62."""
    cid_rows = [(i, r) for i, r in enumerate(dev.rows) if r["canonical_id"] == cid]
    assert len(cid_rows) > 0
    tmpl_i, _ = cid_rows[0]
    tmpl_sample = dev.get(tmpl_i, (10,), "short")
    tmpl_batch = float_batch(collate([tmpl_sample]))

    cam_offsets = [int(np.round(np.median([np.asarray(r["obs_rows"])[4, c] - r["anchor"] for _, r in cid_rows]))) for c in range(3)]
    cached_by_cam = []
    for c in range(3):
        dev.base.features(cid, c, [])
        rows_c = np.array(sorted(set(int(x) for r_arr, _ in dev.base.feature_maps[(cid, c)] for x in r_arr)), dtype=np.int64)
        cached_by_cam.append(rows_c)

    def get_vis_at(a_cur):
        obs_cams = []
        for c in range(3):
            rows_c = cached_by_cam[c]
            src_cur = a_cur + cam_offsets[c]
            chosen_ids = []
            for slot_off in (-24, -18, -12, -6, 0):
                tgt = src_cur + slot_off
                idx = int(np.searchsorted(rows_c, tgt, side="right")) - 1
                chosen_ids.append(int(rows_c[max(0, idx)]))
            obs_cams.append(dev.base.features(cid, c, chosen_ids))  # (5, 8, 768)
        obs_np = np.stack(obs_cams, axis=1)  # (5, 3, 8, 768)
        b_out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in tmpl_batch.items()}
        b_out["obs"] = torch.from_numpy(obs_np)[None].float()
        b_out["current_valid"] = torch.ones((1, 3), dtype=torch.bool)
        b_out["history_valid"] = torch.ones((1, 4), dtype=torch.bool)
        return b_out, cam_offsets[0]

    return get_vis_at, cam_offsets[0], len(cached_by_cam[0])


def load_font(sz, bold=False):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def overlay_ghost_urdf_and_traj(
    base_rgb_480, ren_full, sim, scene_opt, q_draw, calib,
    color_mult, color_add, contour_bgr,
    pred_tcp_w=None, gt_tcp_w=None, wrist_insets=None
):
    sim.reset(q_draw)
    ren_full.update_scene(sim.d, camera="cam_ar", scene_option=scene_opt)
    rgb_raw = ren_full.render()
    M_f = calib["M_full"]
    rgb_mj = cv2.warpAffine(rgb_raw, M_f, (640, 480), flags=cv2.INTER_LINEAR)
    mask_s = cv2.warpAffine((rgb_raw.max(axis=-1) > 10).astype(np.uint8), M_f, (640, 480), flags=cv2.INTER_NEAREST) > 0

    canvas = base_rgb_480.copy()
    tint = np.clip(rgb_mj.astype(np.float32) * np.asarray(color_mult) + np.asarray(color_add), 0, 255).astype(np.uint8)
    # 46% ghost URDF + 54% real video so you see BOTH the real black arm/orange zip-tie AND the 3D Ghost Arm clearly!
    canvas[mask_s] = (0.54 * canvas[mask_s].astype(np.float32) + 0.46 * tint[mask_s].astype(np.float32)).astype(np.uint8)

    contours, _ = cv2.findContours(mask_s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Note: canvas is RGB, so contour color is RGB tuple!
    cv2.drawContours(canvas, contours, -1, contour_bgr, 2, lineType=cv2.LINE_AA)

    # Project 50-step 3D TCP trajectories onto the real camera image (with Z_cam > 0.08m safety guard)!
    rvec, tvec, K = calib["rvec"], calib["tvec"], calib["K"]
    R_cw_m, _ = cv2.Rodrigues(rvec)
    t_cw_v = np.asarray(tvec, dtype=np.float64).reshape(1, 3)

    def safe_proj(pts_w):
        pts_w = np.asarray(pts_w, dtype=np.float64)
        z_cam = (pts_w @ R_cw_m.T + t_cw_v)[:, 2]
        pts_ok = pts_w[z_cam > 0.08]
        if len(pts_ok) < 2:
            return None
        pts2, _ = cv2.projectPoints(pts_ok, rvec, tvec, K, np.zeros(5))
        return np.round(np.clip(pts2[:, 0, :], -2000, 2000)).astype(np.int32)

    if gt_tcp_w is not None and len(gt_tcp_w) >= 2:
        for side in range(2):
            p_int = safe_proj(gt_tcp_w[:, side, :])
            if p_int is not None:
                for k_i in range(0, len(p_int) - 1, 2):
                    cv2.line(canvas, tuple(p_int[k_i]), tuple(p_int[k_i + 1]), (80, 245, 120), 1, cv2.LINE_AA)

    if pred_tcp_w is not None and len(pred_tcp_w) >= 2:
        traj_cols = [(50, 235, 255), (255, 215, 45)]  # Left TCP: bright cyan; Right TCP: bright gold
        for side in range(2):
            p_int = safe_proj(pred_tcp_w[:, side, :])
            if p_int is not None:
                cv2.polylines(canvas, [p_int], False, traj_cols[side], 2, cv2.LINE_AA)
                cv2.circle(canvas, tuple(p_int[0]), 4, traj_cols[side], -1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(p_int[-1]), 4, (255, 255, 255), 1, cv2.LINE_AA)

    crop = canvas[60:420, :].copy()  # (360, 640, 3)

    if wrist_insets is not None:
        wL, wR = wrist_insets
        in_w, in_h = 124, 93
        imL = cv2.resize(wL, (in_w, in_h), interpolation=cv2.INTER_AREA)
        imR = cv2.resize(wR, (in_w, in_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(imL, (0, 0), (in_w - 1, in_h - 1), (80, 235, 125), 2)
        cv2.rectangle(imR, (0, 0), (in_w - 1, in_h - 1), (80, 235, 125), 2)
        cv2.putText(imL, "L-Wrist", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(imR, "R-Wrist", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        crop[8 : 8 + in_h, 8 : 8 + in_w] = imL
        crop[8 : 8 + in_h, 640 - 8 - in_w : 640 - 8] = imR

    return crop


def wrap_tile_with_hud(crop_360, title, sub_line, met_line, grip_mismatch, bar_rgb):
    # Returns (416, 640, 3): 28px top bar + 360px video + 28px bottom bar
    tile = np.zeros((416, 640, 3), dtype=np.uint8)
    tile[:28, :] = (20, 24, 32)
    tile[28:388, :] = crop_360
    tile[388:, :] = (55, 22, 24) if grip_mismatch else (20, 24, 32)
    tile[:3, :] = bar_rgb

    pil = Image.fromarray(tile)
    dr = ImageDraw.Draw(pil)
    f_b = load_font(14, bold=True)
    f_r = load_font(12, bold=False)
    dr.text((10, 6), title, fill=(255, 255, 255), font=f_b)
    dr.text((370, 8), sub_line, fill=(185, 200, 220), font=f_r)
    dr.text((10, 395), met_line, fill=(255, 210, 140) if grip_mismatch else (140, 245, 185), font=f_r)
    return np.asarray(pil)


def main():
    assert not OUT_DIR.exists(), f"{OUT_DIR} already exists"
    OUT_DIR.mkdir(parents=False)
    t_start = time.monotonic()

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    _, _, tr, dev = datasets()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    build_or_load_extra_stats(tr, dev, entries)

    st_json = json.loads((T / "results/e129_control_20261119_01/extra_proprio_stats.json").read_text())
    extra_stats = dict(mean66=np.asarray(st_json["mean66"], "f4"), std66=np.asarray(st_json["std66"], "f4"))
    c_dict = contract()

    get_vis_at, cam0_off, n_cached = build_dense_causal_vis_getter(dev, CID)
    print(f"Loaded {CID}: {n_cached} cached SigLIP2 frames, cam_high offset={cam0_off}", flush=True)

    b3_src = json.loads((T / "results/e99b3_20261115_01/sources.json").read_text())
    rec = next(s for s in b3_src if s["cid"] == CID)
    vids_p = {
        k: v.replace("/mnt/workspace/<colleague>/Datasets/piper/", "/mnt/workspace/dataset/piper/")
        for k, v in rec["videos"].items()
    }
    vid_high = load_mp4_rgb(vids_p["cam_high"], max_frames=820)
    vid_wL = load_mp4_rgb(vids_p["cam_left_wrist"], max_frames=820)
    vid_wR = load_mp4_rgb(vids_p["cam_right_wrist"], max_frames=820)

    z, q_ep, p_ep, good = dev.states.get(dev, CID)

    fy_init = 913.3756713867188 * 0.5
    fovy_init = math.degrees(2.0 * math.atan(240.0 / fy_init))
    ar_xml = build_pure_arm_ar_xml(fovy_init)
    piper_cfg = json.loads(PIPER_JSON.read_text())
    sim = BimanualPiperSimV2(ar_xml, piper_cfg)
    scene_opt = configure_pure_arm_appearance(sim)

    calib = calibrate_cam_high(sim, scene_opt, q_ep, vid_high, cam0_off)

    # Base transforms to map pose_torch base-local TCP -> MuJoCo world frame
    sim.reset(q_ep[200])
    R_wb_L = sim.d.xmat[sim.base_l].reshape(3, 3).copy()
    p_wb_L = sim.d.xpos[sim.base_l].copy()
    R_wb_R = sim.d.xmat[sim.base_r].reshape(3, 3).copy()
    p_wb_R = sim.d.xpos[sim.base_r].copy()

    def chunk_q_to_world_tcp(q_seq):
        with torch.no_grad():
            p18 = pose_torch(torch.from_numpy(np.asarray(q_seq))[None].cuda().double()).cpu().numpy()[0]
        pos_local, _ = pose18_to_pos_and_R(p18)  # (T, 2, 3)
        pos_w = np.zeros_like(pos_local)
        pos_w[:, 0, :] = (pos_local[:, 0, :] @ R_wb_L.T) + p_wb_L[None, :]
        pos_w[:, 1, :] = (pos_local[:, 1, :] @ R_wb_R.T) + p_wb_R[None, :]
        return pos_w

    # Load models: E134_FM_res, E134_D_dec, blind_t1
    def load_e134(arm):
        p = T / f"results/e134_wam_{arm.lower()}_s90_20261119_01/ckpt_04000.pt"
        cp = torch.load(p, map_location="cpu", weights_only=False)
        m = E134WAM(tr.norm, arm_mode=ARM_CONFIGS[arm]["arm_mode"]).widen_from_164()
        m.load_state_dict(cp["model"], strict=True)
        return m.cuda().eval()

    m_fm = load_e134("FM_res")
    m_det = load_e134("D_dec")
    blind_t1 = fit_blind_t1_model(tr)

    # Simulate 11 consecutive 50-frame (1.67s) closed-loop M3 windows across frames [200..750] (18.3s total!)
    s0 = 200
    win_len = 50
    n_wins = 11
    total_steps = win_len * n_wins  # 550 frames = 18.33s

    gt_q_all = q_ep[s0 + 1 : s0 + 1 + total_steps]
    with torch.no_grad():
        p18_gt_all = pose_torch(torch.from_numpy(gt_q_all)[None].cuda().double()).cpu().numpy()[0]
    gt_pos_local, gt_R_local = pose18_to_pos_and_R(p18_gt_all)

    def simulate_policy_rolling(kind):
        out_records = []
        q_hist = q_ep.copy()
        p_hist = p_ep.copy()
        cur_chunk = None
        chunk_t0 = 0
        q_ref = q_ep[s0].copy()

        for step in range(total_steps):
            a_cur = s0 + step
            in_win = step % win_len

            if kind == "gt":
                q_draw = gt_q_all[step].copy()
                rem_gt = q_ep[a_cur + 1 : min(a_cur + 51, len(q_ep))]
                fut_w = chunk_q_to_world_tcp(rem_gt)
                out_records.append(dict(q14=q_draw, pos_err=0.0, rot_err=0.0, pred_tcp_w=None, gt_tcp_w=fut_w))
                continue

            # Re-anchor cleanly to real robot state at the start of each 50-frame (1.67s) horizon window!
            if in_win == 0:
                q_hist = q_ep.copy()
                p_hist = p_ep.copy()
                sim.reset(q_ep[a_cur])
                q_ref = q_ep[a_cur].copy()

            if in_win in (0, 10, 25):
                if kind == "blind":
                    raw_chunk = predict_blind_t1(blind_t1, q_hist, a_cur)
                else:
                    vis_b, _ = get_vis_at(a_cur)
                    b_live = build_sim_proprio_batch(vis_b, q_hist, p_hist, a_cur, z, good, c_dict, extra_stats, dev.base.stats, True)
                    b_cu = move(b_live, "cuda")
                    with torch.inference_mode():
                        if kind == "fm":
                            raw_chunk = m_fm.infer_e134(b_cu, steps=8, sample_mode="mean8", n_samples=8, seed=700 + step)[0].float().cpu().numpy()
                        else:
                            raw_chunk = m_det.infer_e134(b_cu, steps=8, sample_mode="zero")[0].float().cpu().numpy()

                if in_win > 0:
                    q_sim_cur = q_hist[a_cur]
                    q_anchor = q_sim_cur.copy()
                    q_anchor[JI] = q_sim_cur[JI] + np.clip(q_ref[JI] - q_sim_cur[JI], -0.08, 0.08)
                    cur_chunk = raw_chunk.copy()
                    cur_chunk[:, JI] = q_anchor[None, JI] + (raw_chunk[:, JI] - q_sim_cur[None, JI])
                else:
                    cur_chunk = raw_chunk
                chunk_t0 = step

            idx_c = min(step - chunk_t0, len(cur_chunk) - 1)
            cmd = cur_chunk[idx_c]
            q_ref = cmd.copy()
            st_out = sim.step_frame_v2(cmd, use_vff=True)
            if a_cur + 1 < len(q_hist):
                q_hist[a_cur + 1] = st_out["q14"]
                p_hist[a_cur + 1] = pose_np(st_out["q14"][None])[0]

            pos_err = float(np.linalg.norm(st_out["pos"] - gt_pos_local[step], axis=-1).mean() * 1000.0)
            rot_err = float(geodesic_deg(st_out["R"], gt_R_local[step]).mean())

            pred_rem = cur_chunk[idx_c:]
            pred_w = chunk_q_to_world_tcp(pred_rem) if len(pred_rem) >= 2 else None
            gt_rem = q_ep[a_cur + 1 : min(a_cur + 1 + len(pred_rem), len(q_ep))]
            gt_w = chunk_q_to_world_tcp(gt_rem) if len(gt_rem) >= 2 else None
            out_records.append(dict(q14=st_out["q14"].copy(), pos_err=pos_err, rot_err=rot_err, pred_tcp_w=pred_w, gt_tcp_w=gt_w))

        return out_records

    panel_specs = [
        ("gt",    "1. Real Video + GT URDF (Calibration Check)", [0.35, 1.35, 0.65], [0, 40, 15],  (60, 255, 115),  (45, 185, 95)),
        ("fm",    "2. Flow-Matching WAM (E134_FM_res)",          [1.25, 0.45, 1.45], [35, 0, 45],  (235, 95, 255),  (165, 70, 235)),
        ("det",   "3. Deterministic WAM (E134_D_dec)",           [0.35, 1.05, 1.50], [0, 25, 50],  (65, 210, 255),  (40, 130, 235)),
        ("blind", "4. Blind Baseline (blind_t1 - No Vision)",    [1.50, 0.50, 0.35], [50, 10, 0],  (255, 95, 65),   (225, 75, 55)),
    ]

    sim_runs = {}
    summary_metrics = {}
    for pk, title, *_ in panel_specs:
        print(f"Simulating rolling 50-step M3 AR trajectories for {title}...", flush=True)
        recs = simulate_policy_rolling(pk)
        sim_runs[pk] = recs
        if pk != "gt":
            end_errs = [recs[w * win_len + (win_len - 1)]["pos_err"] for w in range(n_wins)]
            summary_metrics[pk] = dict(
                mean_all_mm=round(float(np.mean([r["pos_err"] for r in recs])), 2),
                mean_end50_mm=round(float(np.mean(end_errs)), 2),
                mean_rot_deg=round(float(np.mean([r["rot_err"] for r in recs])), 2),
            )

    print("Rendering 550-frame 2x2 Real-Video AR URDF Overlay MP4...", flush=True)
    ren_full = mujoco.Renderer(sim.m, height=480, width=640)
    mp4_path = OUT_DIR / "e135_real_video_ar_urdf_quad.mp4"
    cmd_ff = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", "1280x832", "-pix_fmt", "rgb24", "-r", "30", "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "25", "-preset", "medium",
        "-movflags", "+faststart", str(mp4_path),
    ]
    proc = subprocess.Popen(cmd_ff, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    preview_indices = {
        40:  "ar_preview_1_bottle_pickup.png",
        260: "ar_preview_2_center_align.png",
        435: "ar_preview_3_ziptie_thread.png",
    }

    thr_L, thr_R = c_dict["threshold_m"][0], c_dict["threshold_m"][1]

    for step in range(total_steps):
        a_cur = s0 + step + 1
        vf = int(np.clip(a_cur + cam0_off, 0, len(vid_high) - 1))
        base_fr = vid_high[vf]
        w_insets = (vid_wL[min(vf, len(vid_wL) - 1)], vid_wR[min(vf, len(vid_wR) - 1)])

        gt_q = gt_q_all[step]
        gt_gL = "OPEN" if gt_q[6] > thr_L else "GRIP"
        gt_gR = "OPEN" if gt_q[13] > thr_R else "GRIP"

        win_i = step // win_len + 1
        in_w = step % win_len + 1
        sub_txt = f"Stage {win_i:02d}/11 (Step {in_w:02d}/50) | t={step/30.0:4.1f}s/18.3s"

        tiles = []
        for pk, title, cmul, cadd, cbgr, bar_c in panel_specs:
            rc = sim_runs[pk][step]
            crop = overlay_ghost_urdf_and_traj(
                base_fr, ren_full, sim, scene_opt, rc["q14"], calib,
                cmul, cadd, cbgr,
                pred_tcp_w=rc["pred_tcp_w"],
                gt_tcp_w=rc["gt_tcp_w"],
                wrist_insets=(w_insets if pk == "gt" else None),
            )
            p_gL = "OPEN" if rc["q14"][6] > thr_L else "GRIP"
            p_gR = "OPEN" if rc["q14"][13] > thr_R else "GRIP"
            mismatch = (pk != "gt") and ((p_gL != gt_gL) or (p_gR != gt_gR))

            if pk == "gt":
                met_txt = f"Calibration IoU={calib['iou_opt']:.3f} | Real Grippers: L={gt_gL} R={gt_gR} | Dotted=Next 50f GT"
            else:
                mm = summary_metrics[pk]["mean_all_mm"]
                g_tag = " [GRIP DIFF]" if mismatch else ""
                met_txt = (
                    f"3D TCP Err: {rc['pos_err']:5.1f}mm (Avg {mm:4.1f}mm) | "
                    f"Rot: {rc['rot_err']:4.1f}deg | L:{p_gL} R:{p_gR}{g_tag}"
                )
            tiles.append(wrap_tile_with_hud(crop, title, sub_txt, met_txt, mismatch, bar_c))

        quad = np.concatenate([
            np.concatenate(tiles[:2], axis=1),
            np.concatenate(tiles[2:], axis=1),
        ], axis=0)  # (832, 1280, 3)

        if step in preview_indices:
            Image.fromarray(quad).save(OUT_DIR / preview_indices[step])

        proc.stdin.write(quad.tobytes())

    proc.stdin.close()
    err_ff = proc.stderr.read().decode("utf-8", errors="ignore")
    rc_ff = proc.wait()
    ren_full.close()
    assert rc_ff == 0, f"ffmpeg failed: {err_ff[-1000:]}"

    meta_out = dict(
        experiment="E135_real_video_ar_urdf_overlay",
        brt=brt(),
        episode_cid=CID,
        frame_span=[s0, s0 + total_steps],
        duration_s=round(total_steps / 30.0, 2),
        calibration={k: v for k, v in calib.items() if k not in ("M_full", "K", "rvec", "tvec")},
        summary_metrics=summary_metrics,
        mp4_mb=round(mp4_path.stat().st_size / (1024.0 * 1024.0), 2),
        elapsed_s=round(time.monotonic() - t_start, 1),
    )
    (OUT_DIR / "ar_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print("DONE E135 AR URDF video:", json.dumps(meta_out), flush=True)


if __name__ == "__main__":
    main()
