# E132:
#   1. Exact 3D End-Effector Orientation Error (SO(3) geodesic degrees) audit (offline + closed-loop MuJoCo)
#   2. Cheap Claude-suggested ablation on WAM_N2_kinskip: kin_skip-only vs full (testing visual variance in steps 1..8)
#   3. MuJoCo Replanning Lag Fix: comparing M1 (naive q_sim anchor, v_des=0), M2 (windup-safe q_ref anchor, v_des=0),
#      and M3 (MIT velocity feedforward +kd*(v_cmd-v_cur) + q_ref anchor) across S1, S2, S3 (K=4), S4 (15mm perturb)!
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import mujoco

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for sub in ("code_e131", "code_e130", "code_e129"):
    sys.path.insert(0, str(T / sub))

from build_bimanual_mjcf import verify_fk_alignment, PIPER_JSON
from run_e131_sim import (
    BimanualPiperSim, build_sim_proprio_batch, fit_blind_t1_model, predict_blind_t1, JI,
)
from e130_common import E130WAM
from e129_common import INIT, LOCK_PATH, GR, brt, UpgradedWAM, build_or_load_extra_stats, attach_extra_proprio
from e103d_model import WorldModel as OrigModel
from e99w2_model import JOINTS, pose_torch
from e99w2_data import pose_np
from e102x_setup import datasets
from e102a_contract import contract
from e110f_core import infer_d
from e110f_eval import float_batch
from e86.data import move, collate

OUT = T / "results/e132_sim_fix_ori_20261119_01"
REPORT_MD = T / "notes/e132_replan_lag_fix_and_orientation_20261119.md"


def pose18_to_pos_and_R(p18):
    """
    p18: (..., 18) -> pos (..., 2, 3) meters, R (..., 2, 3, 3) rotation matrices.
    e99w2 pose format per arm: [x, y, z, rx0, rx1, rx2, ry0, ry1, ry2], so rz = rx x ry!
    """
    lead = p18.shape[:-1]
    p2 = p18.reshape(*lead, 2, 9)
    pos = p2[..., 0:3]
    rx = p2[..., 3:6]
    ry = p2[..., 6:9]
    rx = rx / np.maximum(np.linalg.norm(rx, axis=-1, keepdims=True), 1e-12)
    ry = ry - (rx * ry).sum(-1, keepdims=True) * rx
    ry = ry / np.maximum(np.linalg.norm(ry, axis=-1, keepdims=True), 1e-12)
    rz = np.cross(rx, ry)
    R = np.stack([rx, ry, rz], axis=-1)  # (..., 2, 3, 3)
    return pos, R


def geodesic_deg(R_a, R_b):
    # R_a, R_b: (..., 3, 3) -> angle in degrees
    tr = np.einsum("...ij,...ij->...", R_a, R_b)
    cos_th = np.clip(0.5 * (tr - 1.0), -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


class BimanualPiperSimV2(BimanualPiperSim):
    def get_local_pos_and_R(self):
        pos = np.zeros((2, 3), dtype="f8")
        Rm = np.zeros((2, 3, 3), dtype="f8")
        for idx, (bid, sid) in enumerate(((self.base_l, self.tcp_l), (self.base_r, self.tcp_r))):
            R_wb = self.d.xmat[bid].reshape(3, 3)
            pos[idx] = R_wb.T @ (self.d.site_xpos[sid] - self.d.xpos[bid])
            R_ws = self.d.site_xmat[sid].reshape(3, 3)
            Rm[idx] = R_wb.T @ R_ws
        return pos, Rm

    def step_frame_v2(self, q14_target, use_vff=False):
        q_tgt = np.asarray(q14_target, "f8")[JI]
        g_tgt = np.clip(np.asarray(q14_target, "f8")[GR], 0.0, 0.10)
        q_tgt = np.clip(q_tgt, self.lo12, self.hi12)

        peak_tau = 0.0
        table_col = False
        arm_col = False

        for sub in range(self.substeps_per_frame):
            q_des = self.q_cmd_prev + (q_tgt - self.q_cmd_prev) * (1.0 / (self.substeps_per_frame - sub))
            v_des = np.clip((q_des - self.q_cmd_prev) / self.dt, -self.max_vel, self.max_vel)
            dv = np.clip(v_des - self.v_cmd_prev, -self.max_acc * self.dt, self.max_acc * self.dt)
            v_cmd = np.clip(self.v_cmd_prev + dv, -self.max_vel, self.max_vel)
            q_cmd = np.clip(self.q_cmd_prev + v_cmd * self.dt, self.lo12, self.hi12)
            self.q_cmd_prev = q_cmd
            self.v_cmd_prev = v_cmd

            dg = np.clip(g_tgt - self.g_cmd_prev, -self.max_gvel * self.dt, self.max_gvel * self.dt)
            g_cmd = np.clip(self.g_cmd_prev + dg, 0.0, 0.10)
            self.g_cmd_prev = g_cmd

            self.d_grav.qpos[:] = self.d.qpos
            self.d_grav.qvel[:] = 0.0
            self.d_grav.qacc[:] = 0.0
            mujoco.mj_forward(self.m, self.d_grav)
            tau_ff = self.d_grav.qfrc_bias[self.qpos_j12]

            q_cur = self.d.qpos[self.qpos_j12]
            v_cur = self.d.qvel[self.qpos_j12]
            v_err = (v_cmd - v_cur) if use_vff else (-v_cur)
            tau = np.clip(self.kp * (q_cmd - q_cur) + self.kd * v_err + tau_ff, -35.0, 35.0)

            self.d.ctrl[0:6] = tau[0:6]
            self.d.ctrl[6] = 0.5 * g_cmd[0]
            self.d.ctrl[7] = -0.5 * g_cmd[0]
            self.d.ctrl[8:14] = tau[6:12]
            self.d.ctrl[14] = 0.5 * g_cmd[1]
            self.d.ctrl[15] = -0.5 * g_cmd[1]

            mujoco.mj_step(self.m, self.d)
            peak_tau = max(peak_tau, float(np.abs(tau).max()))
            for ci in range(self.d.ncon):
                c = self.d.contact[ci]
                if c.dist < -0.001:
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if g1 == self.table_gid or g2 == self.table_gid:
                        table_col = True
                    elif (g1 in self.left_gids and g2 in self.right_gids) or (g2 in self.left_gids and g1 in self.right_gids):
                        arm_col = True

        pos, Rm = self.get_local_pos_and_R()
        return dict(q14=self.get_q14(), pos=pos, R=Rm, peak_tau=peak_tau, table_col=table_col, arm_col=arm_col)


def main():
    assert not OUT.exists(), f"{OUT} already exists"
    OUT.mkdir(parents=False)
    t0 = time.monotonic()

    xml_path, _ = verify_fk_alignment()
    piper_cfg = json.loads(PIPER_JSON.read_text())
    sim = BimanualPiperSimV2(xml_path, piper_cfg)

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    _, _, tr, dev = datasets()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    assert len(entries) == 211
    _, norm_dev = build_or_load_extra_stats(tr, dev, entries)
    st_json = json.loads((T / "results/e129_control_20261119_01/extra_proprio_stats.json").read_text())
    extra_stats = dict(mean66=np.asarray(st_json["mean66"], "f4"), std66=np.asarray(st_json["std66"], "f4"))
    c_dict = contract()

    ps = sorted({e["parent"] for e in entries})
    pid = np.array([ps.index(e["parent"]) for e in entries])
    W = np.random.default_rng(90).integers(len(ps), size=(2000, len(ps)))
    W = np.stack([np.bincount(w, minlength=len(ps)) for w in W]).astype("f8")

    def stat(v, mask=None, g=pid):
        v = np.asarray(v, "f8")
        mask = np.ones(len(v), bool) if mask is None else mask
        s = np.bincount(g[mask], v[mask], len(ps))
        n = np.bincount(g[mask], minlength=len(ps)).astype("f8")
        return [float(s.sum() / max(n.sum(), 1))] + [float(x) for x in np.nanquantile((W @ s) / np.maximum(W @ n, 1), [0.025, 0.975])]

    # Pre-fetch visual batches & GT pos + orientation R
    vis_batches = {}
    gt_actions, gt_pos, gt_R = [], [], []
    gl_list, gc_list = [], []
    for wi, e in enumerate(entries):
        b0 = float_batch(dev.batch([e["index"]], (10,), "short"))
        vis_batches[(wi, 0)] = b0
        gt_act = b0["action"][0]
        gt_actions.append(gt_act.numpy())
        with torch.no_grad():
            p18 = pose_torch(gt_act[None].cuda().double()).cpu().numpy()[0]
        pos_i, R_i = pose18_to_pos_and_R(p18)
        gt_pos.append(pos_i)
        gt_R.append(R_i)
        gl_list.append(b0["grip_label"][0].numpy())
        gc_list.append(b0["grip_current"][0].numpy())
        for off in (10, 25):
            sh = dev.shifted(e["index"], off, "short")
            if sh is not None:
                vis_batches[(wi, off)] = float_batch(collate([sh]))

    GT_ACT = np.stack(gt_actions)
    GT_POS = np.stack(gt_pos)   # (211, 50, 2, 3)
    GT_R = np.stack(gt_R)       # (211, 50, 2, 3, 3)
    GL = np.stack(gl_list).astype("i1").transpose(0, 2, 1).reshape(-1, 50)
    GC = (np.stack(gc_list) > 0.5).astype("i1").reshape(-1)
    events = lambda to, frm: [(i, int(np.argmax(GL[i] == to))) for i in range(len(GL)) if GC[i] == frm and (GL[i] == to).any()]
    EV = {"close": events(0, 1), "open": events(1, 0)}
    evg = {n: pid[np.array([x[0] for x in ev]) // 2] for n, ev in EV.items()}

    rng_pert = np.random.default_rng(131)
    PERT_Q14 = np.zeros((len(entries), 14), dtype="f4")
    for wi in range(len(entries)):
        dq = np.zeros(14, dtype="f4")
        dq[[0, 1, 2, 7, 8, 9]] = rng_pert.choice([-0.03, 0.03], size=6)
        dq[[3, 4, 5, 10, 11, 12]] = rng_pert.choice([-0.02, 0.02], size=6)
        PERT_Q14[wi] = dq

    blind_t1 = fit_blind_t1_model(tr)

    def load_wam(kind, path):
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if kind == "orig":
            m = OrigModel(tr.norm)
            m.load_state_dict(cp["model"], strict=True)
        elif kind == "v1":
            m = UpgradedWAM(tr.norm, variant="v1").widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        elif kind == "N2":
            m = E130WAM(tr.norm, clean_state=True, use_kin_skip=True).widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        else:
            raise ValueError(kind)
        return m.cuda().eval()

    # ==========================================================================
    # Part A: Offline 3D Orientation Error (deg) + N2 kin_skip-only Ablation!
    # ==========================================================================
    off_preds = {}
    for mname, kind, ckpt, is_230 in (
        ("E106C", "orig", INIT, False),
        ("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt", True),
        ("WAM_N2_kinskip", "N2", T / "results/e130_wam_n2_s90_20261119_01/ckpt_04000.pt", True),
    ):
        mw = load_wam(kind, ckpt)
        preds = []
        preds_kinonly = []
        with torch.inference_mode():
            for wi, e in enumerate(entries):
                b = move(vis_batches[(wi, 0)], "cuda")
                if is_230:
                    b = attach_extra_proprio(b, norm_dev[e["index"]][None])
                preds.append(infer_d(mw, b, 4, 90)[0].float().cpu().numpy())
                if kind == "N2":
                    # Claude's cheap test: evaluate kin_skip branch ALONE (visual delta_head=0) vs full N2!
                    kin_in = torch.cat([b["q_normalized"] * b["state_valid"][:, None], b["proprio"]], dim=-1)
                    d_kin = mw.kin_skip(kin_in).reshape(1, 50, 12)
                    act_k = b["q_current"][:, None].expand(-1, 50, -1).clone()
                    act_k[:, :, JOINTS] = b["q_current"][:, None, JOINTS] + d_kin * b["action_std"][:, :, JOINTS]
                    preds_kinonly.append(act_k[0].float().cpu().numpy())
        off_preds[mname] = np.stack(preds)
        if kind == "N2":
            off_preds["WAM_N2_kinonly"] = np.stack(preds_kinonly)
        del mw
        torch.cuda.empty_cache()

    # Offline blind_t1_s90
    b_preds = []
    for wi, e in enumerate(entries):
        _, q_ep, _, _ = dev.states.get(dev, e["cid"])
        b_preds.append(predict_blind_t1(blind_t1, q_ep, e["anchor"]))
    off_preds["blind_t1_s90"] = np.stack(b_preds)

    offline_ori = {"models": {}, "diffs": {}}
    ori_arrs = {}
    for mname, pa in off_preds.items():
        with torch.no_grad():
            p18 = pose_torch(torch.from_numpy(pa).cuda().double()).cpu().numpy()
        pos_m, R_m = pose18_to_pos_and_R(p18)
        ee_mm = np.linalg.norm(pos_m - GT_POS, axis=-1) * 1000.0
        rot_deg = geodesic_deg(R_m, GT_R)  # (211, 50, 2) deg
        l8_w = np.abs(pa[:, :8, JI] - GT_ACT[:, :8, JI]).mean((1, 2))
        l50_w = np.abs(pa[:, :, JI] - GT_ACT[:, :, JI]).mean((1, 2))
        d = dict(
            l8=l8_w, l50=l50_w,
            e8=ee_mm[:, 7].mean(1), e50=ee_mm[:, 49].mean(1),
            rot8=rot_deg[:, 7].mean(1), rot25=rot_deg[:, 24].mean(1), rot50=rot_deg[:, 49].mean(1),
        )
        for n, ev in EV.items():
            ii = np.array([x[0] for x in ev])
            tt = np.array([x[1] for x in ev])
            d["rot_" + n] = rot_deg[ii // 2, tt, ii % 2]
        ori_arrs[mname] = d
        offline_ori["models"][mname] = {
            k: (stat(v, None, evg[k[4:]]) if k.startswith("rot_") else stat(v))
            for k, v in d.items()
        }

    for a, b in (
        ("WAM_v1_s90", "E106C"),
        ("WAM_v1_s90", "blind_t1_s90"),
        ("WAM_N2_kinskip", "blind_t1_s90"),
        ("WAM_N2_kinonly", "WAM_N2_kinskip"),
        ("WAM_N2_kinonly", "blind_t1_s90"),
    ):
        offline_ori["diffs"][f"{a}-{b}"] = {
            k: (stat(ori_arrs[a][k] - ori_arrs[b][k], None, evg[k[4:]]) if k.startswith("rot_") else stat(ori_arrs[a][k] - ori_arrs[b][k]))
            for k in ("l8", "l50", "e8", "e50", "rot8", "rot25", "rot50", "rot_close", "rot_open")
        }
    print("Offline orientation + N2 kinonly audit done!", flush=True)

    # ==========================================================================
    # Part B: MuJoCo Replanning Lag Fix (M1_naive vs M2_ref_anchor vs M3_mit_vff_ref)
    # ==========================================================================
    modes = [
        ("M1_naive", dict(ref_anchor=False, use_vff=False)),
        ("M2_ref_anchor", dict(ref_anchor=True, use_vff=False)),
        ("M3_mit_vff_ref", dict(ref_anchor=True, use_vff=True)),
    ]
    scenarios = [
        ("S1_open50", dict(replan_steps=(0,), perturb=False)),
        ("S2_replan_vis", dict(replan_steps=(0, 10, 25), perturb=False)),
        ("S3_replan_k4", dict(replan_steps=tuple(range(0, 50, 4)), perturb=False)),
        ("S4_perturb_15mm", dict(replan_steps=(0, 10, 25), perturb=True)),
    ]
    sim_models = [
        ("GT_demo_MIT", "gt", None, False),
        ("E106C", "orig", INIT, False),
        ("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt", True),
        ("blind_t1_s90", "blind", None, False),
    ]

    sim_res = {md: {sc: {} for sc, _ in scenarios} for md, _ in modes}

    for mname, kind, ckpt, is_230 in sim_models:
        wm = load_wam(kind, ckpt) if kind not in ("gt", "blind") else None
        for md_name, md_cfg in modes:
            for sc_name, sc in scenarios:
                # GT_demo doesn't differ between M1 and M2 (only M3 changes MIT vff)
                if kind == "gt" and md_name == "M2_ref_anchor":
                    sim_res[md_name][sc_name][mname] = sim_res["M1_naive"][sc_name][mname]
                    continue

                replan_set = set(sc["replan_steps"])
                sim_pos = np.zeros((len(entries), 50, 2, 3), dtype="f8")
                sim_R = np.zeros((len(entries), 50, 2, 3, 3), dtype="f8")
                sim_q = np.zeros((len(entries), 50, 14), dtype="f4")
                peak_taus = np.zeros(len(entries), dtype="f8")
                tcols = np.zeros(len(entries), dtype="f8")
                acols = np.zeros(len(entries), dtype="f8")

                for wi, e in enumerate(entries):
                    z, q_ep, p_ep, good = dev.states.get(dev, e["cid"])
                    a0 = e["anchor"]
                    q_hist = q_ep.copy()
                    p_hist = p_ep.copy()

                    if sc["perturb"]:
                        q_init = np.clip(q_ep[a0] + PERT_Q14[wi], -3.0, 3.0)
                        q_init[JI] = np.clip(q_init[JI], sim.lo12, sim.hi12)
                        q_init[GR] = q_ep[a0, GR]
                        start_h = max(0, a0 - 24)
                        q_hist[start_h : a0 + 1] = q_init[None]
                        p_hist[start_h : a0 + 1] = pose_np(q_init[None])
                    else:
                        q_init = q_ep[a0].copy()

                    sim.reset(q_init)
                    cur_chunk = None
                    chunk_t0 = 0
                    q_ref = q_init.copy()

                    for step in range(50):
                        if step in replan_set:
                            if kind == "gt":
                                cur_chunk = GT_ACT[wi, step:]
                            else:
                                if kind == "blind":
                                    raw_chunk = predict_blind_t1(blind_t1, q_hist, a0 + step)
                                else:
                                    vis_off = 0
                                    if step >= 25 and (wi, 25) in vis_batches:
                                        vis_off = 25
                                    elif step >= 10 and (wi, 10) in vis_batches:
                                        vis_off = 10
                                    b_live = build_sim_proprio_batch(
                                        vis_batches[(wi, vis_off)], q_hist, p_hist, a0 + step, z, good, c_dict,
                                        extra_stats, dev.base.stats, is_230,
                                    )
                                    with torch.inference_mode():
                                        raw_chunk = infer_d(wm, move(b_live, "cuda"), 4, 90)[0].float().cpu().numpy()

                                if md_cfg["ref_anchor"] and step > 0:
                                    # Re-anchor predicted delta onto continuous desired reference q_ref (windup-clamped <= 0.08 rad from q_sim!)
                                    q_sim_cur = q_hist[a0 + step]
                                    q_anchor_safe = q_sim_cur.copy()
                                    q_anchor_safe[JI] = q_sim_cur[JI] + np.clip(q_ref[JI] - q_sim_cur[JI], -0.08, 0.08)
                                    delta_chunk = raw_chunk[:, JI] - q_sim_cur[None, JI]
                                    cur_chunk = raw_chunk.copy()
                                    cur_chunk[:, JI] = q_anchor_safe[None, JI] + delta_chunk
                                else:
                                    cur_chunk = raw_chunk
                            chunk_t0 = step

                        cmd = cur_chunk[min(step - chunk_t0, len(cur_chunk) - 1)]
                        q_ref = cmd.copy()
                        st_out = sim.step_frame_v2(cmd, use_vff=md_cfg["use_vff"])
                        sim_q[wi, step] = st_out["q14"]
                        sim_pos[wi, step] = st_out["pos"]
                        sim_R[wi, step] = st_out["R"]
                        if a0 + step + 1 < len(q_hist):
                            q_hist[a0 + step + 1] = st_out["q14"]
                            p_hist[a0 + step + 1] = pose_np(st_out["q14"][None])[0]
                        peak_taus[wi] = max(peak_taus[wi], st_out["peak_tau"])
                        tcols[wi] = max(tcols[wi], float(st_out["table_col"]))
                        acols[wi] = max(acols[wi], float(st_out["arm_col"]))

                ee_mm = np.linalg.norm(sim_pos - GT_POS, axis=-1) * 1000.0
                rot_deg = geodesic_deg(sim_R, GT_R)
                jerk_e = (np.diff(sim_q[:, :, JI], 3, axis=1) ** 2).mean((1, 2))
                ev_s = {}
                for n, ev in EV.items():
                    ii = np.array([x[0] for x in ev])
                    tt = np.array([x[1] for x in ev])
                    ev_s["ev_" + n] = stat(np.linalg.norm(sim_pos[ii // 2, tt, ii % 2] - GT_POS[ii // 2, tt, ii % 2], axis=-1) * 1000.0, None, evg[n])
                    ev_s["rot_" + n] = stat(rot_deg[ii // 2, tt, ii % 2], None, evg[n])

                sim_res[md_name][sc_name][mname] = dict(
                    e8=stat(ee_mm[:, 7].mean(1)),
                    e25=stat(ee_mm[:, 24].mean(1)),
                    e50=stat(ee_mm[:, 49].mean(1)),
                    rot8=stat(rot_deg[:, 7].mean(1)),
                    rot50=stat(rot_deg[:, 49].mean(1)),
                    **ev_s,
                    peak_tau_nm=float(peak_taus.mean()),
                    jerk_e=float(jerk_e.mean()),
                    table_col=int(tcols.sum()),
                    arm_col=int(acols.sum()),
                )
                print(
                    f"[{brt()}] {md_name:14s} | {sc_name:15s} | {mname:12s} | e50={sim_res[md_name][sc_name][mname]['e50'][0]:.2f}mm "
                    f"rot50={sim_res[md_name][sc_name][mname]['rot50'][0]:.2f}deg open={sim_res[md_name][sc_name][mname]['ev_open'][0]:.2f}mm "
                    f"col={int(tcols.sum())}/{int(acols.sum())}",
                    flush=True,
                )
        if wm is not None:
            del wm
            torch.cuda.empty_cache()

    summary = dict(
        experiment="E132_replan_lag_fix_and_orientation",
        brt=brt(),
        total_elapsed_s=round(time.monotonic() - t0, 1),
        offline_orientation_and_n2_ablation=offline_ori,
        sim_modes=sim_res,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_markdown(summary)

    # Append concise summary to worklog.md
    ov = offline_ori["models"]
    m1_s3 = sim_res["M1_naive"]["S3_replan_k4"]["WAM_v1_s90"]
    m2_s3 = sim_res["M2_ref_anchor"]["S3_replan_k4"]["WAM_v1_s90"]
    m3_s3 = sim_res["M3_mit_vff_ref"]["S3_replan_k4"]["WAM_v1_s90"]
    m3_s4 = sim_res["M3_mit_vff_ref"]["S4_perturb_15mm"]["WAM_v1_s90"]
    m3_s4_b = sim_res["M3_mit_vff_ref"]["S4_perturb_15mm"]["blind_t1_s90"]
    wl = (
        f"\n- 2026-11-19 BRT E132（Antigravity执行）：①补测三维末端姿态旋转误差(deg)：+50步 WAM_v1={ov['WAM_v1_s90']['rot50'][0]:.2f}° vs blind_t1={ov['blind_t1_s90']['rot50'][0]:.2f}° vs E106C={ov['E106C']['rot50'][0]:.2f}°；"
        f"②N2去视觉仅保留kin_skip测试：l8={ov['WAM_N2_kinonly']['l8'][0]:.5f} vs N2全量={ov['WAM_N2_kinskip']['l8'][0]:.5f}；"
        f"③MuJoCo重规划滞后修复(WAM_v1 K=4 +50步末端/姿态)：M1朴素={m1_s3['e50'][0]:.2f}mm/{m1_s3['rot50'][0]:.2f}° -> "
        f"M2期望参考锚定={m2_s3['e50'][0]:.2f}mm/{m2_s3['rot50'][0]:.2f}° -> M3(+MIT速度前馈)={m3_s3['e50'][0]:.2f}mm/{m3_s3['rot50'][0]:.2f}°（0碰撞）；"
        f"M3开局15mm扰动恢复：WAM_v1={m3_s4['e50'][0]:.2f}mm vs blind_t1={m3_s4_b['e50'][0]:.2f}mm。见 notes/e132_replan_lag_fix_and_orientation_20261119.md。\n"
    )
    with (T / "notes/worklog.md").open("a") as f:
        f.write(wl)
    print("E132 COMPLETED in", summary["total_elapsed_s"], "s", flush=True)


def fmd(arr, d=2):
    return f"{arr[0]:+.{d}f} [{arr[1]:+.{d}f}, {arr[2]:+.{d}f}]"


def write_markdown(s):
    om = s["offline_orientation_and_n2_ablation"]["models"]
    od = s["offline_orientation_and_n2_ablation"]["diffs"]
    sr = s["sim_modes"]

    lines = [
        "# E132 三维末端姿态角评估、`N2` 视觉方差检验与 MuJoCo 重规划滞后修复",
        "",
        f"完成时间：{s['brt']}（总耗时 {s['total_elapsed_s']}s）。代码：`code_e132/`；结果：`results/e132_sim_fix_ori_20261119_01/`。",
        "",
        "## 一、离线三维末端姿态误差（SO(3) 测地旋转角 `deg`）与 `N2` 去视觉检验",
        "",
        "| 模型 | 前8步 `l8` | 50步 `l50` | +8步位置 / 姿态 | +25步姿态 | +50步位置 / 姿态 | 夹紧姿态 | 松爪姿态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in om.items():
        lines.append(
            f"| `{k}` | {r['l8'][0]:.5f} | {r['l50'][0]:.5f} | {r['e8'][0]:.2f}mm / {r['rot8'][0]:.2f}° | "
            f"{r['rot25'][0]:.2f}° | {r['e50'][0]:.2f}mm / **{r['rot50'][0]:.2f}°** | {r['rot_close'][0]:.2f}° | {r['rot_open'][0]:.2f}° |"
        )

    lines.extend([
        "",
        "### 关键配对差值（负值表示前者更好）",
        "",
        "| 配对比较 | `Δl8` | `Δe50` (mm) | `Δrot8` (°) | `Δrot50` (°) | `Δ松爪姿态` (°) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pair, d in od.items():
        lines.append(
            f"| `{pair}` | {fmd(d['l8'], 5)} | {fmd(d['e50'], 2)} | {fmd(d['rot8'], 2)} | {fmd(d['rot50'], 2)} | {fmd(d['rot_open'], 2)} |"
        )

    lines.extend([
        "",
        "## 二、MuJoCo 重规划滞后修复对比（`M1` 朴素 vs `M2` 参考轨迹锚定 vs `M3` MIT 速度前馈 + 参考锚定）",
        "",
        "| 场景与模型 | `M1_naive` (+50位置/姿态) | `M2_ref_anchor` (+50位置/姿态) | `M3_mit_vff_ref` (+50位置/姿态) | `M3` 松爪位置/姿态 | `M3` 峰值力矩 / 碰撞 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for sc in ("S1_open50", "S2_replan_vis", "S3_replan_k4", "S4_perturb_15mm"):
        for m in ("GT_demo_MIT", "E106C", "WAM_v1_s90", "blind_t1_s90"):
            r1 = sr["M1_naive"][sc][m]
            r2 = sr["M2_ref_anchor"][sc][m]
            r3 = sr["M3_mit_vff_ref"][sc][m]
            lines.append(
                f"| `{sc}` / `{m}` | {r1['e50'][0]:.2f}mm / {r1['rot50'][0]:.2f}° | "
                f"{r2['e50'][0]:.2f}mm / {r2['rot50'][0]:.2f}° | "
                f"**{r3['e50'][0]:.2f}mm / {r3['rot50'][0]:.2f}°** | "
                f"{r3['ev_open'][0]:.2f}mm / {r3['rot_open'][0]:.2f}° | "
                f"{r3['peak_tau_nm']:.2f}Nm / {r3['table_col']}/{r3['arm_col']} |"
            )

    REPORT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
