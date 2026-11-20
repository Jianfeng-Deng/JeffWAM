# Render a COMPLETE continuous multi-stage bimanual Piper episode (360~450 frames = 12~15s start-to-finish!)
# in a 2x2 Quad-View MuJoCo closed-loop dashboard:
#   [Top-Left:  GT Demo (MIT Floor)]               [Top-Right:    E134 Flow-Matching WAM (M3 Closed-Loop)]
#   [Bot-Left:  E134 Decoupled Deterministic WAM]  [Bot-Right:    blind_t1 (No Vision - Full-Episode Drift)]
import os, sys, copy, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for sub in ("code_e134", "code_e133", "code_e132", "code_e131", "code_e130", "code_e129"):
    if str(T / sub) not in sys.path:
        sys.path.insert(0, str(T / sub))

from e134_common import E134WAM, INIT, LOCK_PATH, build_or_load_extra_stats
from train_e134 import ARM_CONFIGS
from build_bimanual_mjcf import PIPER_JSON
from render_e132_video import build_clean_visual_xml, draw_hud
from run_e131_sim import build_sim_proprio_batch, fit_blind_t1_model, predict_blind_t1, JI
from run_e132_fix_and_ori import BimanualPiperSimV2, pose18_to_pos_and_R, geodesic_deg
from e129_common import GR
from e99w2_model import pose_torch
from e99w2_data import pose_np
from e102x_setup import datasets
from e102a_contract import contract
from e110f_eval import float_batch
from e86.data import move, collate
from e90.data import ImageDataset

OUT_DIR = T / "results/e134_full_ep_video_20261119_02"


def probe_dense_episode_and_vis_getter(dev, entries):
    """
    Build a helper that queries cached V-JEPA visual observations at any frame `a` of a held-out dev episode,
    falling back smoothly to the nearest causal cached frame <= `a` if a frame wasn't cached.
    Picks a held-out dev episode with a rich, continuous multi-event manipulation span (360~450 frames)!
    """
    by_cid = {}
    for e in entries:
        by_cid.setdefault(e["cid"], []).append(e)

    best_score = -1.0
    best_cid = None
    best_span = (0, 360)

    c_dict = contract()
    for cid, es in by_cid.items():
        z, q, _, good = dev.states.get(dev, cid)
        seg = z["continuous_segment"]
        # Find longest contiguous valid segment containing the dev anchors
        ancs = sorted(e["anchor"] for e in es)
        s0 = max(24, ancs[0] - 20)
        # Walk forward while segment stays continuous and within 420 frames (~14s full task cycle!)
        s1 = s0
        while s1 + 51 < len(q) and good[s1] and seg[s1] == seg[s0] and (s1 - s0) < 420:
            s1 += 1
        span_len = s1 - s0
        if span_len < 180:
            continue
        # Count gripper transitions + joint motion inside [s0, s1]
        g_bin = (q[s0:s1, GR] > np.asarray(c_dict["threshold_m"])).astype(int)
        n_trans = int(np.abs(np.diff(g_bin, axis=0)).sum())
        motion = float(np.abs(np.diff(q[s0:s1, JI], axis=0)).sum())
        score = n_trans * 20.0 + motion + 0.1 * len(es) * 10.0
        if score > best_score:
            best_score = score
            best_cid = cid
            best_span = (s0, s1)

    assert best_cid is not None, "Could not find a suitable multi-hundred-frame dev episode"

    # Build a probe dataset clone with candidate rows every 10 frames across [s0, s1] to see which frames have cached V-JEPA features!
    tmpl_idx = next(e["index"] for e in by_cid[best_cid])
    tmpl_row = copy.deepcopy(dev.rows[tmpl_idx])
    z, q, _, good = dev.states.get(dev, best_cid)
    seg = z["continuous_segment"]

    cached_vis = {}
    # First seed with all existing dev.rows for best_cid (and their +10, +25 shifted views!)
    for i_r, r in enumerate(dev.rows):
        if r["canonical_id"] == best_cid:
            try:
                cached_vis[r["anchor"]] = float_batch(collate([dev.get(i_r, (10,), "short")]))
                for off in (10, 25):
                    sh = dev.shifted(i_r, off, "short")
                    if sh is not None:
                        cached_vis[r["anchor"] + off] = float_batch(collate([sh]))
            except Exception:
                pass

    # Probe every 5 frames across [best_span[0], best_span[1]] via dev.shifted first!
    for f_idx in range(best_span[0], best_span[1], 5):
        if f_idx in cached_vis or f_idx + 51 >= len(q):
            continue
        try:
            sh = dev.shifted(tmpl_idx, int(f_idx - tmpl_row["anchor"]), "short")
            if sh is not None:
                b_sh = float_batch(collate([sh]))
                if torch.isfinite(b_sh["obs"]).all() and bool(b_sh["current_valid"].any()):
                    cached_vis[int(f_idx)] = b_sh
        except Exception:
            pass

    avail_frames = sorted(cached_vis.keys())
    assert len(avail_frames) > 0, "No cached visual frames found for episode"

    def get_causal_vis_batch(frame_t):
        # Strictly causal: pick largest cached visual frame <= frame_t (or earliest if before first)
        cands = [f for f in avail_frames if f <= frame_t]
        chosen = cands[-1] if cands else avail_frames[0]
        return cached_vis[chosen], chosen

    return best_cid, best_span, len(avail_frames), get_causal_vis_batch


def main(smoke_only=False):
    vis_xml = build_clean_visual_xml()
    piper_cfg = json.loads(PIPER_JSON.read_text())

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    _, _, tr, dev = datasets()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    build_or_load_extra_stats(tr, dev, entries)

    cid, (s0, s1), n_vis, get_vis = probe_dense_episode_and_vis_getter(dev, entries)
    print(f"Selected full dev episode: {cid}, frames=[{s0}..{s1}] (len={s1-s0}, {((s1-s0)/30.0):.1f}s), cached_vis_frames={n_vis}", flush=True)
    if smoke_only:
        sim_t = BimanualPiperSimV2(vis_xml, piper_cfg)
        r_t = mujoco.Renderer(sim_t.m, height=320, width=480)
        sim_t.reset(dev.states.get(dev, cid)[1][s0])
        r_t.update_scene(sim_t.d, camera="overview")
        im_t = r_t.render()
        r_t.close()
        assert im_t.shape == (320, 480, 3)
        print("render_full_episode smoke check passed!", flush=True)
        return

    assert not OUT_DIR.exists(), f"{OUT_DIR} already exists"
    OUT_DIR.mkdir(parents=False)

    st_json = json.loads((T / "results/e129_control_20261119_01/extra_proprio_stats.json").read_text())
    extra_stats = dict(mean66=np.asarray(st_json["mean66"], "f4"), std66=np.asarray(st_json["std66"], "f4"))
    c_dict = contract()

    # Pick the best trained Flow-Matching arm from E134 summary.json (fallback to FM_res or FM_dit)
    eval_sum_path = T / "results/e134_eval_20261119_01/summary.json"
    best_fm_arm = "FM_res"
    best_fm_mode = "stochastic"
    if eval_sum_path.exists():
        esum = json.loads(eval_sum_path.read_text())
        mj_s2 = esum.get("mujoco_m3", {}).get("S2_replan_vis", {})
        fm_scores = []
        for arm_c in ("FM_nat", "FM_dit", "FM_res"):
            for md_lbl, md_arg in (("s1", "stochastic"), ("mean8", "mean8")):
                k_full = f"E134_{arm_c}_{md_lbl}"
                if k_full in mj_s2:
                    fm_scores.append((mj_s2[k_full]["e50"][0], arm_c, md_arg, k_full))
        if fm_scores:
            fm_scores.sort()
            _, best_fm_arm, best_fm_mode, best_fm_label = fm_scores[0]
        else:
            best_fm_label = "E134_FM_res"
    else:
        best_fm_label = "E134_FM_res"

    def load_e134(arm):
        p = T / f"results/e134_wam_{arm.lower()}_s90_20261119_01/ckpt_04000.pt"
        cp = torch.load(p, map_location="cpu", weights_only=False)
        m = E134WAM(tr.norm, arm_mode=ARM_CONFIGS[arm]["arm_mode"]).widen_from_164()
        m.load_state_dict(cp["model"], strict=True)
        return m.cuda().eval()

    m_fm = load_e134(best_fm_arm)
    m_det = load_e134("D_dec")
    blind_t1 = fit_blind_t1_model(tr)

    z, q_ep, p_ep, good = dev.states.get(dev, cid)
    total_steps = s1 - s0
    gt_seq = q_ep[s0 + 1 : s0 + 1 + total_steps]
    with torch.no_grad():
        p18_gt = pose_torch(torch.from_numpy(gt_seq)[None].cuda().double()).cpu().numpy()[0]
    gt_pos, gt_R = pose18_to_pos_and_R(p18_gt)

    sim = BimanualPiperSimV2(vis_xml, piper_cfg)
    renderer = mujoco.Renderer(sim.m, height=340, width=560)

    def run_full_episode(runner_kind):
        q_hist = q_ep.copy()
        p_hist = p_ep.copy()
        q_init = q_ep[s0].copy()
        sim.reset(q_init)

        R_wb_L = sim.d.xmat[sim.base_l].reshape(3, 3).copy()
        p_wb_L = sim.d.xpos[sim.base_l].copy()
        R_wb_R = sim.d.xmat[sim.base_r].reshape(3, 3).copy()
        p_wb_R = sim.d.xpos[sim.base_r].copy()

        cur_chunk = None
        chunk_t0 = 0
        q_ref = q_init.copy()
        frames = []

        # Replan every 10 frames (3 Hz visual-motor closed-loop) across the entire 12~15s episode!
        for step in range(total_steps):
            a_cur = s0 + step
            if step % 10 == 0:
                if runner_kind == "gt":
                    cur_chunk = gt_seq[step : min(step + 50, total_steps)]
                else:
                    if runner_kind == "blind":
                        raw_chunk = predict_blind_t1(blind_t1, q_hist, a_cur)
                    else:
                        vis_b, _ = get_vis(a_cur)
                        b_live = build_sim_proprio_batch(vis_b, q_hist, p_hist, a_cur, z, good, c_dict, extra_stats, dev.base.stats, True)
                        b_cu = move(b_live, "cuda")
                        with torch.inference_mode():
                            if runner_kind == "fm":
                                raw_chunk = m_fm.infer_e134(b_cu, steps=8, sample_mode=best_fm_mode, n_samples=8, seed=1000 + step)[0].float().cpu().numpy()
                            else:
                                raw_chunk = m_det.infer_e134(b_cu, steps=8, sample_mode="zero")[0].float().cpu().numpy()
                    if step > 0:
                        q_sim_cur = q_hist[a_cur]
                        q_anchor = q_sim_cur.copy()
                        q_anchor[JI] = q_sim_cur[JI] + np.clip(q_ref[JI] - q_sim_cur[JI], -0.08, 0.08)
                        cur_chunk = raw_chunk.copy()
                        cur_chunk[:, JI] = q_anchor[None, JI] + (raw_chunk[:, JI] - q_sim_cur[None, JI])
                    else:
                        cur_chunk = raw_chunk
                chunk_t0 = step

            cmd = cur_chunk[min(step - chunk_t0, len(cur_chunk) - 1)]
            q_ref = cmd.copy()
            sim.d.mocap_pos[0] = p_wb_L + R_wb_L @ gt_pos[step, 0]
            sim.d.mocap_pos[1] = p_wb_R + R_wb_R @ gt_pos[step, 1]

            st_out = sim.step_frame_v2(cmd, use_vff=True)
            if a_cur + 1 < len(q_hist):
                q_hist[a_cur + 1] = st_out["q14"]
                p_hist[a_cur + 1] = pose_np(st_out["q14"][None])[0]

            pos_err = float(np.linalg.norm(st_out["pos"] - gt_pos[step], axis=-1).mean() * 1000.0)
            rot_err = float(geodesic_deg(st_out["R"], gt_R[step]).mean())
            gL = "OPEN" if st_out["q14"][6] > c_dict["threshold_m"][0] else "GRIP"
            gR = "OPEN" if st_out["q14"][13] > c_dict["threshold_m"][1] else "GRIP"

            renderer.update_scene(sim.d, camera="overview")
            rgb = renderer.render()
            frames.append((rgb, pos_err, rot_err, gL, gR))
        return frames

    panels = [
        ("gt",    "1. GT Demonstration (MIT Reference)",          (40, 165, 85)),
        ("fm",    f"2. Flow-Matching WAM ({best_fm_label})",      (145, 70, 225)),
        ("det",   "3. Decoupled Deterministic WAM (E134_D_dec)",  (35, 115, 220)),
        ("blind", "4. Blind Policy (blind_t1 - No Vision)",       (215, 80, 55)),
    ]

    runs = {}
    mean_errs = {}
    for pk, title, _ in panels:
        print(f"Simulating & rendering full episode [{s0}..{s1}] for {title}...", flush=True)
        rf = run_full_episode(pk)
        runs[pk] = rf
        mean_errs[pk] = float(np.mean([x[1] for x in rf]))

    quad_frames = []
    for step in range(total_steps):
        sec = step / 30.0
        tot_sec = total_steps / 30.0
        rendered_tiles = []
        for pk, title, col in panels:
            rgb, perr, rerr, gL, gR = runs[pk][step]
            sub = f"Full Episode {cid[-8:]}  |  Time: {sec:4.1f}s / {tot_sec:4.1f}s (Frame {step+1}/{total_steps})"
            met = f"TCP Err: {perr:5.1f}mm (Avg {mean_errs[pk]:4.1f}mm) | Rot: {rerr:4.1f}deg | L:{gL} R:{gR}"
            rendered_tiles.append(draw_hud(rgb, title, sub, met, color_rgb=col))
        top_row = np.concatenate([rendered_tiles[0], rendered_tiles[1]], axis=1)
        bot_row = np.concatenate([rendered_tiles[2], rendered_tiles[3]], axis=1)
        quad_frames.append(np.concatenate([top_row, bot_row], axis=0))  # (680, 1120, 3)

    # Save 3-keyframe contact sheet (start 20%, middle 55%, end 90%)
    kf_idx = [int(total_steps * 0.20), int(total_steps * 0.55), int(total_steps * 0.90)]
    Image.fromarray(quad_frames[kf_idx[1]]).save(OUT_DIR / "full_ep_mid_preview.png")
    Image.fromarray(quad_frames[kf_idx[2]]).save(OUT_DIR / "full_ep_end_preview.png")

    mp4_path = OUT_DIR / "e134_full_episode_mujoco_quad.mp4"
    h, w, _ = quad_frames[0].shape
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", "30", "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "25", "-preset", "medium",
        "-movflags", "+faststart", str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for fr in quad_frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="ignore")
    rc = proc.wait()
    assert rc == 0, f"ffmpeg failed: {err[-1000:]}"
    renderer.close()

    meta = dict(
        episode_cid=cid,
        frame_span=[int(s0), int(s1)],
        total_frames=int(total_steps),
        duration_s=round(total_steps / 30.0, 2),
        cached_vis_frames=int(n_vis),
        best_fm_model=best_fm_label,
        mean_tcp_err_mm=mean_errs,
        mp4_kb=round(mp4_path.stat().st_size / 1024.0, 1),
    )
    (OUT_DIR / "video_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Full-episode MP4 rendered OK:", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main(smoke_only=(len(sys.argv) > 1 and sys.argv[1] == "--smoke"))
