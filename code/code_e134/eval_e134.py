# E134 comprehensive evaluator:
#   - Compares Decoupled Deterministic (D_dec) vs 3 Flow-Matching WAM variants (FM_nat, FM_dit, FM_res)
#     across zero-noise ODE, single stochastic sample (N=1), 8-sample mean (mean8), AND Best-of-8 multimodal coverage!
#   - Evaluates arm pos (mm), 3D orientation (deg), amplitude RMS, gripper switch F1@5/10 & open/close recall,
#     World Head W(N1) retention, and MuJoCo M3 closed-loop S2 & S4 + win=166 large lift!
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for sub in ("code_e133", "code_e132", "code_e131", "code_e130", "code_e129"):
    if str(T / sub) not in sys.path:
        sys.path.insert(0, str(T / sub))

from e134_common import E134WAM, INIT, G1_CKPT, LOCK_PATH, CTRL134, brt, build_or_load_extra_stats, attach_extra_proprio
from train_e134 import ARM_CONFIGS
from e133_common import E133WAM
from build_bimanual_mjcf import verify_fk_alignment, PIPER_JSON
from run_e131_sim import build_sim_proprio_batch, fit_blind_t1_model, predict_blind_t1, JI
from run_e132_fix_and_ori import BimanualPiperSimV2, pose18_to_pos_and_R, geodesic_deg
from e129_common import GR, UpgradedWAM
from e99w2_model import JOINTS, pose_torch, trajectory
from e99w2_data import pose_np
from e102x_setup import datasets
from e102a_contract import contract, onset_labels
from e110f_core import infer_d
from e110f_eval import float_batch, transitions, match
from e86.data import move, collate

OUT = T / "results/e134_eval_20261119_01"
REPORT_MD = T / "notes/e134_flow_matching_and_decoupled_grip_20261119.md"


def main():
    assert not OUT.exists(), f"{OUT} already exists"
    OUT.mkdir(parents=False)
    t0 = time.monotonic()

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    _, _, tr, dev = datasets()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    assert len(entries) == 211
    _, norm_dev = build_or_load_extra_stats(tr, dev, entries)
    st_json = json.loads((T / "results/e129_control_20261119_01/extra_proprio_stats.json").read_text())
    extra_stats = dict(mean66=np.asarray(st_json["mean66"], "f4"), std66=np.asarray(st_json["std66"], "f4"))
    c_dict = contract()
    thr = np.asarray(c_dict["threshold_m"])

    ps = sorted({e["parent"] for e in entries})
    pid = np.array([ps.index(e["parent"]) for e in entries])
    g2 = np.repeat(pid, 2)
    W = np.random.default_rng(90).integers(len(ps), size=(2000, len(ps)))
    W = np.stack([np.bincount(w, minlength=len(ps)) for w in W]).astype("f8")

    def stat(v, mask=None, g=pid):
        v = np.asarray(v, "f8")
        mask = np.ones(len(v), bool) if mask is None else mask
        s = np.bincount(g[mask], v[mask], len(ps))
        n = np.bincount(g[mask], minlength=len(ps)).astype("f8")
        return [float(s.sum() / max(n.sum(), 1))] + [float(x) for x in np.nanquantile((W @ s) / np.maximum(W @ n, 1), [0.025, 0.975])]

    Bt = [float_batch(dev.batch([e["index"] for e in entries[s : s + 4]], (10,), "short")) for s in range(0, len(entries), 4)]
    cat = lambda k: torch.cat([b[k] for b in Bt])
    OBS, Q0, TR, GL_T, GC_T = cat("obs"), cat("q_current"), cat("action"), cat("grip_label"), cat("grip_current")
    q0_j = Q0[:, JI].numpy()

    with torch.no_grad():
        p18_gt = pose_torch(TR.cuda().double()).cpu().numpy()
    GT_POS, GT_R = pose18_to_pos_and_R(p18_gt)
    GT_ACT = TR.numpy()
    tj = GT_ACT[:, :, JI]

    ts = GL_T.numpy().astype("i1").transpose(0, 2, 1).reshape(-1, 50)
    cur_g = (GC_T.numpy() > 0.5).astype("i1").reshape(-1)
    events = lambda to, frm: [(i, int(np.argmax(ts[i] == to))) for i in range(len(ts)) if cur_g[i] == frm and (ts[i] == to).any()]
    EV = {"close": events(0, 1), "open": events(1, 0)}
    evg = {n: pid[np.array([x[0] for x in ev]) // 2] for n, ev in EV.items()}

    def load_model(kind, path):
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if kind == "v1":
            m = UpgradedWAM(tr.norm, variant="v1").widen_from_164()
        elif kind == "G1":
            m = E133WAM(tr.norm, use_switch_head=False, vis_token_drop=0.0, use_kin_skip=False).widen_from_164()
        elif kind in ARM_CONFIGS:
            m = E134WAM(tr.norm, arm_mode=ARM_CONFIGS[kind]["arm_mode"]).widen_from_164()
        else:
            raise ValueError(kind)
        m.load_state_dict(cp["model"], strict=True)
        return m.cuda().eval()

    P = {}
    best8_stats = {}

    # Baselines: WAM_v1_s90 and E133_G1_posevel
    for bname, bkind, bpath in (
        ("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt"),
        ("E133_G1_posevel", "G1", G1_CKPT),
    ):
        mw = load_model(bkind, bpath)
        out_l = []
        with torch.inference_mode():
            for idx_b, b in enumerate(Bt):
                es = entries[idx_b * 4 : idx_b * 4 + 4]
                b2 = attach_extra_proprio(move(b, "cuda"), np.stack([norm_dev[e["index"]] for e in es]))
                out_l.append(infer_d(mw, b2, 4, 90).float().cpu())
        P[bname] = torch.cat(out_l)
        del mw
        torch.cuda.empty_cache()

    # E134 models
    e134_defs = [
        ("D_dec",  T / "results/e134_wam_d_dec_s90_20261119_01/ckpt_04000.pt"),
        ("FM_nat", T / "results/e134_wam_fm_nat_s90_20261119_01/ckpt_04000.pt"),
        ("FM_dit", T / "results/e134_wam_fm_dit_s90_20261119_01/ckpt_04000.pt"),
        ("FM_res", T / "results/e134_wam_fm_res_s90_20261119_01/ckpt_04000.pt"),
    ]
    for arm, path in e134_defs:
        if not path.exists():
            continue
        mw = load_model(arm, path)
        if arm == "D_dec":
            out_l = []
            with torch.inference_mode():
                for idx_b, b in enumerate(Bt):
                    es = entries[idx_b * 4 : idx_b * 4 + 4]
                    b2 = attach_extra_proprio(move(b, "cuda"), np.stack([norm_dev[e["index"]] for e in es]))
                    out_l.append(mw.infer_e134(b2, steps=8, sample_mode="zero").float().cpu())
            P["E134_D_dec"] = torch.cat(out_l)
        else:
            # Evaluate zero-noise ODE, single stochastic sample (s1), mean of 8 samples (m8), AND Best-of-8 multimodal coverage!
            z_l, k_l = [], []
            with torch.inference_mode():
                for idx_b, b in enumerate(Bt):
                    es = entries[idx_b * 4 : idx_b * 4 + 4]
                    b2 = attach_extra_proprio(move(b, "cuda"), np.stack([norm_dev[e["index"]] for e in es]))
                    z_l.append(mw.infer_e134(b2, steps=8, sample_mode="zero", seed=90 + idx_b).float().cpu())
                    k_l.append(mw.infer_e134(b2, steps=8, sample_mode="all_k", n_samples=8, seed=900 + idx_b).float().cpu())
            z_act = torch.cat(z_l, dim=0)     # (211, 50, 14)
            k_act = torch.cat(k_l, dim=1)     # (8, 211, 50, 14)
            P[f"E134_{arm}_zero"] = z_act
            P[f"E134_{arm}_s1"] = k_act[0]
            m8_act = k_act[0].clone()
            m8_act[:, :, JI] = k_act[:, :, :, JI].mean(0)
            P[f"E134_{arm}_mean8"] = m8_act

            # Compute Best-of-8 (oracle min over 8 samples) on e50, ev_open, and win=166 large lift!
            ee_8 = []
            for ki in range(8):
                with torch.no_grad():
                    p18_k = pose_torch(k_act[ki].cuda().double()).cpu().numpy()
                pos_k, _ = pose18_to_pos_and_R(p18_k)
                ee_8.append(np.linalg.norm(pos_k - GT_POS, axis=-1) * 1000.0)  # (211, 50, 2)
            ee_8 = np.stack(ee_8, axis=0)  # (8, 211, 50, 2)
            min_e50_w = ee_8[:, :, 49, :].mean(-1).min(axis=0)  # best sample per window
            ii_op = np.array([x[0] for x in EV["open"]])
            tt_op = np.array([x[1] for x in EV["open"]])
            min_open = ee_8[:, ii_op // 2, tt_op, ii_op % 2].min(axis=0)
            # Sample diversity std in mm at +50
            div_mm = float(np.std(ee_8[:, :, 49, :].mean(-1), axis=0).mean())
            best8_stats[f"E134_{arm}"] = dict(
                best8_e50_mm=stat(min_e50_w),
                best8_open_mm=stat(min_open, None, evg["open"]),
                best8_win166_mm=float(ee_8[:, 166, 49, :].mean(-1).min()),
                s1_win166_mm=float(ee_8[0, 166, 49, :].mean()),
                sample_spread_mm=div_mm,
            )
        del mw
        torch.cuda.empty_cache()

    blind_t1 = fit_blind_t1_model(tr)
    P["blind_t1_s90"] = torch.from_numpy(np.stack([predict_blind_t1(blind_t1, dev.states.get(dev, e["cid"])[1], e["anchor"]) for e in entries])).float()

    # ==========================================================================
    # 1. Offline Metrics + Gripper Metrics
    # ==========================================================================
    M = {}
    for k, v in P.items():
        pa = v.numpy()
        pj = pa[:, :, JI]
        with torch.no_grad():
            p18 = pose_torch(v.cuda().double()).cpu().numpy()
        pos_m, R_m = pose18_to_pos_and_R(p18)
        ee_mm = np.linalg.norm(pos_m - GT_POS, axis=-1) * 1000.0
        rot_deg = geodesic_deg(R_m, GT_R)
        d = dict(
            l8=np.abs(pj[:, :8] - tj[:, :8]).mean((1, 2)),
            l50=np.abs(pj - tj).mean((1, 2)),
            e8=ee_mm[:, 7].mean(1),
            e25=ee_mm[:, 24].mean(1),
            e50=ee_mm[:, 49].mean(1),
            rot8=rot_deg[:, 7].mean(1),
            rot50=rot_deg[:, 49].mean(1),
            rms_disp=float(np.sqrt(((pj - q0_j[:, None, :]) ** 2).mean())),
            win166_mm=float(ee_mm[166, 49].mean()),
        )
        for n, ev in EV.items():
            ii = np.array([x[0] for x in ev])
            tt = np.array([x[1] for x in ev])
            d["ev_" + n] = ee_mm[ii // 2, tt, ii % 2]
            d["rot_" + n] = rot_deg[ii // 2, tt, ii % 2]
        M[k] = d

    act_res = {"models": {}, "gripper": {}, "best8_multimodal": best8_stats}
    for k in M:
        act_res["models"][k] = {n: stat(M[k][n]) for n in ("l8", "l50", "e8", "e25", "e50", "rot8", "rot50")}
        act_res["models"][k]["rms_disp_rad"] = M[k]["rms_disp"]
        act_res["models"][k]["win166_mm"] = M[k]["win166_mm"]
        for n in EV:
            act_res["models"][k]["ev_" + n] = stat(M[k]["ev_" + n], None, evg[n])
            act_res["models"][k]["rot_" + n] = stat(M[k]["rot_" + n], None, evg[n])

    onset_cache = {}
    grip_keys = [k for k in ("WAM_v1_s90", "E133_G1_posevel", "E134_D_dec", "E134_FM_nat_zero", "E134_FM_dit_zero", "E134_FM_res_zero") if k in P]
    for k in grip_keys:
        pg = (P[k][:, :, [6, 13]].numpy() > thr).astype("i1").transpose(0, 2, 1).reshape(-1, 50)
        acc = (pg == ts).mean(1)
        hold = (cur_g == 0) & (ts == 0).all(1)
        hit10 = {n: [float((pg[i, max(t - 10, 0) : min(t + 11, 50)] == (0 if n == "close" else 1)).any()) for i, t in ev] for n, ev in EV.items()}
        hit5 = {n: [float((pg[i, max(t - 5, 0) : min(t + 6, 50)] == (0 if n == "close" else 1)).any()) for i, t in ev] for n, ev in EV.items()}
        by_p5, by_p10 = {}, {}
        for wi, e in enumerate(entries):
            r5 = by_p5.setdefault(e["parent"], [0, 0, 0])
            r10 = by_p10.setdefault(e["parent"], [0, 0, 0])
            z, qq, _, good = dev.states.get(dev, e["cid"])
            if e["cid"] not in onset_cache:
                onset_cache[e["cid"]] = onset_labels(qq, z["continuous_segment"], good, c_dict)[0]
            init_g = onset_cache[e["cid"]][e["anchor"]]
            ps_w = (P[k][wi, :, [6, 13]].numpy() > thr).astype("i1")
            ts_w = GL_T[wi].numpy().astype("i1")
            cur_w = (GC_T[wi].numpy() > 0.5).astype("i1")
            for side in range(2):
                pe = transitions(ps_w[:, side], cur_w[side])
                te = transitions(ts_w[:, side], init_g[side])
                r5[0] += len(match(pe, te, 5)); r5[1] += len(pe); r5[2] += len(te)
                r10[0] += len(match(pe, te, 10)); r10[1] += len(pe); r10[2] += len(te)
        f1_5_p = np.array([2 * by_p5[p][0] / max(1, by_p5[p][1] + by_p5[p][2]) for p in ps])
        f1_10_p = np.array([2 * by_p10[p][0] / max(1, by_p10[p][1] + by_p10[p][2]) for p in ps])
        act_res["gripper"][k] = dict(
            state_acc=float(stat(acc, None, g2)[0]),
            drop25=float(stat(pg[:, :25].max(1), hold, g2)[0]),
            close_hit10=float(np.mean(hit10["close"])),
            open_hit10=float(np.mean(hit10["open"])),
            close_hit5=float(np.mean(hit5["close"])),
            open_hit5=float(np.mean(hit5["open"])),
            f1_5=float(f1_5_p.mean()),
            f1_10=float(f1_10_p.mean()),
        )

    # ==========================================================================
    # 2. World Head W(N1) Retention on Single Sample (s1) vs Deterministic!
    # ==========================================================================
    wm_ref = load_model("v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt")
    w_cands = [k for k in ("WAM_v1_s90", "E134_D_dec", "E134_FM_nat_s1", "E134_FM_dit_s1", "E134_FM_dit_mean8", "E134_FM_res_s1", "E134_FM_res_mean8") if k in P]
    world_res = {}
    for h in (25, 50, 100):
        Eh = [e for e in entries if h in e["horizons"]]
        psh = sorted({e["parent"] for e in Eh})
        by_p = {p: {"cp": [], "gt": [], "no": [], **{ck: [] for ck in w_cands}} for p in psh}
        with torch.inference_mode():
            for s in range(0, len(Eh), 4):
                es = Eh[s : s + 4]
                bh = move(float_batch(dev.batch([e["index"] for e in es], (h,), "short")), "cuda")
                Y = bh["world"].double()
                cp_m = Y.square().mean((1, 2)).cpu().numpy()
                gt_m = (wm_ref.forward_mean(bh)["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()
                no_m = (wm_ref.forward_mean(dict(bh, robot=torch.zeros_like(bh["robot"]), robot_valid=torch.zeros_like(bh["robot_valid"])))["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()
                e_pos = [entries.index(e) for e in es]
                c_m = {}
                for ck in w_cands:
                    a_b = P[ck][e_pos].to("cuda")
                    c_m[ck] = (wm_ref.forward_mean(dict(bh, robot=trajectory(bh["context"], a_b)))["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()
                for k_i, e in enumerate(es):
                    rec = by_p[e["parent"]]
                    rec["cp"].append(float(cp_m[k_i])); rec["gt"].append(float(gt_m[k_i])); rec["no"].append(float(no_m[k_i]))
                    for ck in w_cands:
                        rec[ck].append(float(c_m[ck][k_i]))
        sk = lambda key: np.array([100.0 * (1.0 - np.mean(by_p[p][key]) / max(np.mean(by_p[p]["cp"]), 1e-12)) for p in psh])
        s_gt, s_no = sk("gt"), sk("no")
        denom = float((s_gt - s_no).mean())
        world_res[str(h)] = {
            ck: dict(
                n1_minus_no_pp=float((sk(ck) - s_no).mean()),
                retention_pct=float(100.0 * (sk(ck) - s_no).mean() / denom) if abs(denom) > 1e-9 else 0.0,
            )
            for ck in w_cands
        }
    del wm_ref
    torch.cuda.empty_cache()

    # ==========================================================================
    # 3. MuJoCo M3 Closed-Loop Verification (S2_replan_vis & S4_perturb_15mm)
    # ==========================================================================
    xml_path, _ = verify_fk_alignment()
    sim = BimanualPiperSimV2(xml_path, json.loads(PIPER_JSON.read_text()))
    vis_batches = {}
    for wi, e in enumerate(entries):
        vis_batches[(wi, 0)] = float_batch(dev.batch([e["index"]], (10,), "short"))
        for off in (10, 25):
            sh = dev.shifted(e["index"], off, "short")
            if sh is not None:
                vis_batches[(wi, off)] = float_batch(collate([sh]))

    rng_pert = np.random.default_rng(131)
    PERT_Q14 = np.zeros((len(entries), 14), dtype="f4")
    for wi in range(len(entries)):
        dq = np.zeros(14, dtype="f4")
        dq[[0, 1, 2, 7, 8, 9]] = rng_pert.choice([-0.03, 0.03], size=6)
        dq[[3, 4, 5, 10, 11, 12]] = rng_pert.choice([-0.02, 0.02], size=6)
        PERT_Q14[wi] = dq

    mj_cands = [
        ("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt", "det"),
        ("E134_D_dec", "D_dec", T / "results/e134_wam_d_dec_s90_20261119_01/ckpt_04000.pt", "zero"),
        ("E134_FM_nat_mean8", "FM_nat", T / "results/e134_wam_fm_nat_s90_20261119_01/ckpt_04000.pt", "mean8"),
        ("E134_FM_dit_s1", "FM_dit", T / "results/e134_wam_fm_dit_s90_20261119_01/ckpt_04000.pt", "stochastic"),
        ("E134_FM_dit_mean8", "FM_dit", T / "results/e134_wam_fm_dit_s90_20261119_01/ckpt_04000.pt", "mean8"),
        ("E134_FM_res_s1", "FM_res", T / "results/e134_wam_fm_res_s90_20261119_01/ckpt_04000.pt", "stochastic"),
        ("E134_FM_res_mean8", "FM_res", T / "results/e134_wam_fm_res_s90_20261119_01/ckpt_04000.pt", "mean8"),
    ]
    mujoco_res = {"S2_replan_vis": {}, "S4_perturb_15mm": {}}
    loaded_cache = {}
    for label, kind, path, smode in mj_cands:
        if not path.exists():
            continue
        if kind not in loaded_cache:
            for old_m in loaded_cache.values():
                del old_m
            loaded_cache.clear()
            torch.cuda.empty_cache()
            loaded_cache[kind] = load_model(kind, path)
        wm = loaded_cache[kind]

        for sc_n, pert in (("S2_replan_vis", False), ("S4_perturb_15mm", True)):
            sim_pos = np.zeros((len(entries), 50, 2, 3), dtype="f8")
            sim_R = np.zeros((len(entries), 50, 2, 3, 3), dtype="f8")
            tcols, acols = np.zeros(len(entries)), np.zeros(len(entries))

            for wi, e in enumerate(entries):
                z, q_ep, p_ep, good = dev.states.get(dev, e["cid"])
                a0 = e["anchor"]
                q_hist, p_hist = q_ep.copy(), p_ep.copy()
                if pert:
                    q_init = np.clip(q_ep[a0] + PERT_Q14[wi], -3.0, 3.0)
                    q_init[JI] = np.clip(q_init[JI], sim.lo12, sim.hi12)
                    q_init[GR] = q_ep[a0, GR]
                    q_hist[max(0, a0 - 24) : a0 + 1] = q_init[None]
                    p_hist[max(0, a0 - 24) : a0 + 1] = pose_np(q_init[None])
                else:
                    q_init = q_ep[a0].copy()

                sim.reset(q_init)
                cur_chunk, chunk_t0, q_ref = None, 0, q_init.copy()
                for step in range(50):
                    if step in (0, 10, 25):
                        vis_off = 25 if (step >= 25 and (wi, 25) in vis_batches) else (10 if (step >= 10 and (wi, 10) in vis_batches) else 0)
                        b_live = build_sim_proprio_batch(vis_batches[(wi, vis_off)], q_hist, p_hist, a0 + step, z, good, c_dict, extra_stats, dev.base.stats, True)
                        b_cu = move(b_live, "cuda")
                        with torch.inference_mode():
                            if smode == "det":
                                raw_chunk = infer_d(wm, b_cu, 4, 90)[0].float().cpu().numpy()
                            else:
                                raw_chunk = wm.infer_e134(b_cu, steps=8, sample_mode=smode, n_samples=8, seed=90 + wi * 10 + step)[0].float().cpu().numpy()
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
                    st_out = sim.step_frame_v2(cmd, use_vff=True)
                    sim_pos[wi, step] = st_out["pos"]
                    sim_R[wi, step] = st_out["R"]
                    if a0 + step + 1 < len(q_hist):
                        q_hist[a0 + step + 1] = st_out["q14"]
                        p_hist[a0 + step + 1] = pose_np(st_out["q14"][None])[0]
                    tcols[wi] = max(tcols[wi], float(st_out["table_col"]))
                    acols[wi] = max(acols[wi], float(st_out["arm_col"]))

            ee_mm = np.linalg.norm(sim_pos - GT_POS, axis=-1) * 1000.0
            rot_deg = geodesic_deg(sim_R, GT_R)
            ii_op = np.array([x[0] for x in EV["open"]])
            tt_op = np.array([x[1] for x in EV["open"]])
            mujoco_res[sc_n][label] = dict(
                e50=stat(ee_mm[:, 49].mean(1)),
                rot50=stat(rot_deg[:, 49].mean(1)),
                ev_open=stat(ee_mm[ii_op // 2, tt_op, ii_op % 2], None, evg["open"]),
                win166_end_mm=float(ee_mm[166, 49].mean()),
                table_col=int(tcols.sum()),
                arm_col=int(acols.sum()),
            )
    loaded_cache.clear()
    torch.cuda.empty_cache()

    full_summary = dict(
        experiment="E134_flow_matching_and_decoupled_grip",
        brt=brt(),
        elapsed_s=round(time.monotonic() - t0, 1),
        action=act_res,
        world=world_res,
        mujoco_m3=mujoco_res,
    )
    (OUT / "summary.json").write_text(json.dumps(full_summary, indent=2, ensure_ascii=False) + "\n")
    write_markdown(full_summary)
    print("E134 evaluation completed in", full_summary["elapsed_s"], "s!", flush=True)


def write_markdown(s):
    am = s["action"]["models"]
    gr = s["action"]["gripper"]
    b8 = s["action"]["best8_multimodal"]
    wr = s["world"]
    mj = s["mujoco_m3"]

    lines = [
        "# E134 流匹配（Flow Matching）生成动作头与梯度解耦夹爪跳变头对照",
        "",
        f"完成时间：{s['brt']}。代码：`code_e134/`；结果：`results/e134_*_20261119_01/`。",
        "",
        "## 1. 离线动作、三维位置/姿态、动作位移 `RMS` 与 `win=166` 大抬臂（211 窗 / 27 parent）",
        "",
        "| 模型与推理方式 | 前8步 `l8` | 50步 `l50` | +50步位置 / 姿态 | 松爪位置 / 姿态 | 位移 `RMS` (GT=0.1566) | `win=166` 大抬臂 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in am.items():
        lines.append(
            f"| `{k}` | {r['l8'][0]:.5f} | {r['l50'][0]:.5f} | **{r['e50'][0]:.2f}mm / {r['rot50'][0]:.2f}°** | "
            f"{r['ev_open'][0]:.2f}mm / {r['rot_open'][0]:.2f}° | **{r['rms_disp_rad']:.4f}** | {r['win166_mm']:.1f}mm |"
        )

    lines.extend([
        "",
        "### 多模态候选覆盖度（`Best-of-8`：8 条采样中最优一条 vs 单样多模态离散度）",
        "",
        "| 流匹配模型 | 8采样标准差 (mm) | `Best-of-8` +50步末端 | `Best-of-8` 松爪位置 | `Best-of-8` `win=166` 大抬臂 |",
        "|---|---:|---:|---:|---:|",
    ])
    for k, v in b8.items():
        lines.append(
            f"| `{k}` | {v['sample_spread_mm']:.2f} mm | **{v['best8_e50_mm'][0]:.2f} mm** | "
            f"**{v['best8_open_mm'][0]:.2f} mm** | **{v['best8_win166_mm']:.1f} mm** |"
        )

    lines.extend([
        "",
        "## 2. 梯度解耦夹爪跳变头表现（检验保住手臂精度的同时吃下夹爪收益）",
        "",
        "| 模型 | 状态准确率 | `F1@±5` | `F1@±10` | 闭合命中(±10/±5) | **松开命中(±10/±5)** | 误松开率(25步) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for k, g in gr.items():
        lines.append(
            f"| `{k}` | {g['state_acc']:.4f} | **{g['f1_5']:.3f}** | **{g['f1_10']:.3f}** | "
            f"{g['close_hit10']*100:.1f}% / {g['close_hit5']*100:.1f}% | "
            f"**{g['open_hit10']*100:.1f}% / {g['open_hit5']*100:.1f}%** | {g['drop25']*100:.2f}% |"
        )

    lines.extend([
        "",
        "## 3. 驱动 `WAM_v1` 世界头的闭环 `W(N1)` 保留率（+25 / +50 / +100）",
        "",
        "| 动作来源 | +25 保留率 | +50 `N1−no` (pp) | **+50 保留率** | **+100 保留率** |",
        "|---|---:|---:|---:|---:|",
    ])
    for ck in wr["50"]:
        lines.append(
            f"| `{ck}` | {wr['25'][ck]['retention_pct']:.1f}% | {wr['50'][ck]['n1_minus_no_pp']:+.2f} pp | "
            f"**{wr['50'][ck]['retention_pct']:.1f}%** | **{wr['100'][ck]['retention_pct']:.1f}%** |"
        )

    lines.extend([
        "",
        "## 4. MuJoCo 双臂 `M3` 闭环验证（`S2` 正常闭环与 `S4` 15mm 偏置恢复）",
        "",
        "| 模型 | `S2` +50位置/姿态 | `S2` 松爪位置 | `S2` `win=166` 大抬臂 | `S4` 15mm偏置 +50位置/姿态 | 碰撞 (S2 / S4) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for k in mj["S2_replan_vis"]:
        s2, s4 = mj["S2_replan_vis"][k], mj["S4_perturb_15mm"][k]
        lines.append(
            f"| `{k}` | **{s2['e50'][0]:.2f}mm / {s2['rot50'][0]:.2f}°** | {s2['ev_open'][0]:.2f}mm | "
            f"{s2['win166_end_mm']:.1f}mm | **{s4['e50'][0]:.2f}mm / {s4['rot50'][0]:.2f}°** | "
            f"{s2['table_col']}/{s2['arm_col']} , {s4['table_col']}/{s4['arm_col']} |"
        )

    REPORT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        _, _, tr, dev = datasets()
        entries = json.loads(LOCK_PATH.read_text())["entries"]
        _, norm_dev = build_or_load_extra_stats(tr, dev, entries[:4])
        cp = torch.load(INIT, map_location="cpu", weights_only=False)
        mw = E134WAM(tr.norm, arm_mode="fm_dit")
        mw.load_state_dict(cp["model"], strict=False)
        mw.widen_from_164().cuda().eval()
        b = attach_extra_proprio(move(float_batch(dev.batch([e["index"] for e in entries[:4]], (25,), "short")), "cuda"), np.stack([norm_dev[e["index"]] for e in entries[:4]]))
        act = mw.infer_e134(b, steps=8, sample_mode="mean8")
        assert act.shape == (4, 50, 14) and torch.isfinite(act).all()
        print("eval_e134 smoke check passed!", flush=True)
    else:
        main()
