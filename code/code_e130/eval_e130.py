# E130 comprehensive diagnostic evaluator:
#   Q1: Per-joint (j1..j6, proximal vs wrist) x per-step (t=1..50) error breakdown, FK lever arms, surgical FK counterfactuals & window quantiles
#   Q2: Clean-state WAM ablations (N0, N1, N2) vs WAM_v1_s90 & blind_t1_s90
#   Q3: Matched-error perturbed GT action driving World Head (+25, +50, +100) to adjudicate action-error vs information-ceiling
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from e130_common import (
    T, INIT, LOCK_PATH, CTRL130, brt,
    E130WAM, UpgradedWAM, build_or_load_extra_stats, attach_extra_proprio,
)
sys.path.insert(0, str(T / "code_e129"))
from eval_e129 import fit_blind_baselines

from e103d_model import WorldModel as OrigModel
from e99w2_model import JOINTS, pose_torch, trajectory
from e102x_setup import datasets
from e102a_contract import contract, onset_labels
from e110f_core import infer_d
from e110f_eval import float_batch, transitions, match
from e86.data import move

OUT = T / "results/e130_eval_20261119_01"
REPORT_MD = T / "notes/e130_diagnostic_breakdown_and_clean_state_20261119.md"

PROX_LOCAL = [0, 1, 2, 6, 7, 8]   # L_j1..j3, R_j1..j3 in 12-joint array
WRIST_LOCAL = [3, 4, 5, 9, 10, 11] # L_j4..j6, R_j4..j6 in 12-joint array
JOINT_NAMES_6 = ["j1(腰转)", "j2(肩俯仰)", "j3(肘俯仰)", "j4(前臂滚转)", "j5(腕俯仰)", "j6(腕滚转)"]


def ee_from_act(a):
    with torch.no_grad():
        p = pose_torch(a.cuda().double()).cpu().numpy()
    return p[..., [0, 1, 2, 9, 10, 11]].reshape(*a.shape[:2], 2, 3)


def compute_empirical_jacobian_mm_per_rad(q0_tensor, ji_np, eps=1e-4):
    # q0_tensor: (N, 14) -> finite-difference fingertip displacement (mm) per 1 rad perturbation on each joint
    base_ee = ee_from_act(q0_tensor[:, None, :])[:, 0]  # (N, 2, 3) m
    sens = np.zeros((len(q0_tensor), 12), dtype="f8")
    for k_loc, j_full in enumerate(ji_np):
        qp = q0_tensor.clone()
        qp[:, j_full] += eps
        ee_p = ee_from_act(qp[:, None, :])[:, 0]
        arm_side = 0 if k_loc < 6 else 1
        disp_m = np.linalg.norm(ee_p[:, arm_side] - base_ee[:, arm_side], axis=-1)
        sens[:, k_loc] = (disp_m / eps) * 1000.0  # mm / rad
    return sens  # (N, 12)


def main():
    assert not OUT.exists(), f"{OUT} already exists"
    OUT.mkdir(parents=False)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)

    _, _, tr, dev = datasets()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    assert len(entries) == 211
    _, norm_dev = build_or_load_extra_stats(tr, dev, entries)
    c_dict = contract()
    thr = np.asarray(c_dict["threshold_m"])
    JI = np.asarray(JOINTS)

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
    OBS, Q0, TR, GL, GC = cat("obs"), cat("q_current"), cat("action"), cat("grip_label"), cat("grip_current")
    N = len(OBS)
    q0_j = Q0[:, JI].numpy()
    near = np.array([(lambda o: o[np.argmin(np.abs(q0_j[o] - q0_j[i]).sum(1))])(np.where(pid != pid[i])[0]) for i in range(N)])

    def abs_from_delta(d_np, grip_ref):
        x = grip_ref.clone()
        x[:, :, JI] = Q0[:, None, JI] + torch.from_numpy(d_np).float()
        return x

    def load_any_wam(kind, path):
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if kind == "orig":
            m = OrigModel(tr.norm)
            m.load_state_dict(cp["model"], strict=True)
        elif kind == "v1":
            m = UpgradedWAM(tr.norm, variant="v1").widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        elif kind in ("N0", "N1"):
            m = E130WAM(tr.norm, clean_state=True, use_kin_skip=False).widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        elif kind == "N2":
            m = E130WAM(tr.norm, clean_state=True, use_kin_skip=True).widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        else:
            raise ValueError(kind)
        return m.cuda().eval()

    def infer_wam_actions(m, is_230=False, swap=False):
        out = []
        k = 0
        with torch.inference_mode():
            for idx_b, b in enumerate(Bt):
                es = entries[idx_b * 4 : idx_b * 4 + 4]
                n = len(b["obs"])
                sl = np.arange(k, k + n)
                k += n
                b2 = dict(b)
                if swap:
                    b2["obs"] = OBS[near[sl]]
                b2 = move(b2, "cuda")
                if is_230:
                    b2 = attach_extra_proprio(b2, np.stack([norm_dev[e["index"]] for e in es]))
                out.append(infer_d(m, b2, 4, 90).float().cpu())
        return torch.cat(out)

    P = {}
    wam_ckpts = [
        ("E106C", "orig", INIT, False),
        ("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt", True),
        ("WAM_N0_clean", "N0", T / "results/e130_wam_n0_s90_20261119_01/ckpt_04000.pt", True),
        ("WAM_N1_fastp", "N1", T / "results/e130_wam_n1_s90_20261119_01/ckpt_04000.pt", True),
        ("WAM_N2_kinskip", "N2", T / "results/e130_wam_n2_s90_20261119_01/ckpt_04000.pt", True),
    ]
    for name, kind, path, is_230 in wam_ckpts:
        if path.exists():
            mw = load_any_wam(kind, path)
            P[name] = infer_wam_actions(mw, is_230=is_230, swap=False)
            if is_230:
                P[name + "_swap"] = infer_wam_actions(mw, is_230=is_230, swap=True)
            del mw
            torch.cuda.empty_cache()

    blind_deltas = fit_blind_baselines(tr, dev, entries)
    for bname, d_arr in blind_deltas.items():
        P[bname] = abs_from_delta(d_arr, P["E106C"])
    P["copy"] = Q0[:, None].expand(-1, 50, -1).clone()

    EE = {k: ee_from_act(v) for k, v in P.items()}
    EEt = ee_from_act(TR)
    tj = TR[:, :, JI].numpy()  # (211, 50, 12)

    # ==========================================================================
    # Q1: Per-Joint x Per-Step Breakdown & Wrist-Rotation Hypothesis Audit
    # ==========================================================================
    jac_sens = compute_empirical_jacobian_mm_per_rad(Q0, JI)  # (211, 12)
    jac_by_6 = [float(0.5 * (jac_sens[:, j].mean() + jac_sens[:, j + 6].mean())) for j in range(6)]

    step_probe = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 25, 35, 50]
    q1_models = [k for k in ("E106C", "WAM_v1_s90", "WAM_N0_clean", "WAM_N1_fastp", "WAM_N2_kinskip", "blind_t3_s90", "blind_t1_s90", "blind_t1_ens3") if k in P]

    per_step_l1 = {}
    per_step_ee = {}
    per_joint_l8 = {}
    per_joint_l50 = {}
    prox_wrist_summary = {}
    fk_surgical = {}
    window_quantiles = {}

    for k in q1_models:
        pj = P[k][:, :, JI].numpy()
        abs_err = np.abs(pj - tj)  # (211, 50, 12)
        ee_err = np.linalg.norm(EE[k] - EEt, axis=-1).mean(-1) * 1000.0  # (211, 50) mm

        per_step_l1[k] = {str(s): float(abs_err[:, s - 1, :].mean()) for s in step_probe}
        per_step_ee[k] = {str(s): float(ee_err[:, s - 1].mean()) for s in step_probe}

        # Average Left & Right for j1..j6, plus keep full 12
        j8_12 = abs_err[:, :8, :].mean((0, 1))
        j50_12 = abs_err.mean((0, 1))
        per_joint_l8[k] = {
            "j1_to_j6_avg_lr": [float(0.5 * (j8_12[j] + j8_12[j + 6])) for j in range(6)],
            "left_6": [float(j8_12[j]) for j in range(6)],
            "right_6": [float(j8_12[j + 6]) for j in range(6)],
        }
        per_joint_l50[k] = {
            "j1_to_j6_avg_lr": [float(0.5 * (j50_12[j] + j50_12[j + 6])) for j in range(6)],
            "left_6": [float(j50_12[j]) for j in range(6)],
            "right_6": [float(j50_12[j + 6]) for j in range(6)],
        }

        prox_wrist_summary[k] = dict(
            l8_prox=float(abs_err[:, :8, PROX_LOCAL].mean()),
            l8_wrist=float(abs_err[:, :8, WRIST_LOCAL].mean()),
            l50_prox=float(abs_err[:, :, PROX_LOCAL].mean()),
            l50_wrist=float(abs_err[:, :, WRIST_LOCAL].mean()),
        )

        # Surgical FK counterfactuals: isolate EE error from proximal (j1..j3) vs wrist (j4..j6)
        prox_cols = JI[PROX_LOCAL].tolist()
        wrist_cols = JI[WRIST_LOCAL].tolist()
        act_prox_only = TR.clone()
        act_prox_only[:, :, prox_cols] = P[k][:, :, prox_cols]
        ee_prox = np.linalg.norm(ee_from_act(act_prox_only) - EEt, axis=-1).mean(-1) * 1000.0

        act_wrist_only = TR.clone()
        act_wrist_only[:, :, wrist_cols] = P[k][:, :, wrist_cols]
        ee_wrist = np.linalg.norm(ee_from_act(act_wrist_only) - EEt, axis=-1).mean(-1) * 1000.0

        fk_surgical[k] = dict(
            e8_full=float(ee_err[:, 7].mean()),
            e8_prox_only=float(ee_prox[:, 7].mean()),
            e8_wrist_only=float(ee_wrist[:, 7].mean()),
            e50_full=float(ee_err[:, 49].mean()),
            e50_prox_only=float(ee_prox[:, 49].mean()),
            e50_wrist_only=float(ee_wrist[:, 49].mean()),
        )

        w_l50 = abs_err.mean((1, 2))
        w_e50 = ee_err[:, 49]
        window_quantiles[k] = dict(
            l50_p50_p75_p90_p95_mean=[float(np.quantile(w_l50, q)) for q in (0.50, 0.75, 0.90, 0.95)] + [float(w_l50.mean())],
            e50_p50_p75_p90_p95_mean=[float(np.quantile(w_e50, q)) for q in (0.50, 0.75, 0.90, 0.95)] + [float(w_e50.mean())],
        )

    q1_res = dict(
        joint_names_6=JOINT_NAMES_6,
        jacobian_mm_per_rad_j1_to_j6=jac_by_6,
        per_step_l1=per_step_l1,
        per_step_ee_mm=per_step_ee,
        per_joint_l8=per_joint_l8,
        per_joint_l50=per_joint_l50,
        prox_wrist_summary=prox_wrist_summary,
        fk_surgical_mm=fk_surgical,
        window_quantiles=window_quantiles,
    )

    # ==========================================================================
    # Q2: Action / EE / Event / Gripper / Swap Evaluation for Clean-State Models
    # ==========================================================================
    ts = GL.numpy().astype("i1").transpose(0, 2, 1).reshape(-1, 50)
    cur_g = (GC.numpy() > 0.5).astype("i1").reshape(-1)
    events = lambda to, frm: [(i, int(np.argmax(ts[i] == to))) for i in range(len(ts)) if cur_g[i] == frm and (ts[i] == to).any()]
    EV = {"close": events(0, 1), "open": events(1, 0)}
    evg = {n: pid[np.array([x[0] for x in ev]) // 2] for n, ev in EV.items()}

    M = {}
    for k, v in P.items():
        pj = v[:, :, JI].numpy()
        e = np.linalg.norm(EE[k] - EEt, axis=-1) * 1000.0
        d = dict(
            l8=np.abs(pj[:, :8] - tj[:, :8]).mean((1, 2)),
            l50=np.abs(pj - tj).mean((1, 2)),
            e8=e[:, 7].mean(1),
            e25=e[:, 24].mean(1),
            e50=e[:, 49].mean(1),
        )
        for n, ev in EV.items():
            ii = np.array([x[0] for x in ev])
            tt = np.array([x[1] for x in ev])
            d["ev_" + n] = np.linalg.norm(EE[k][ii // 2, tt, ii % 2] - EEt[ii // 2, tt, ii % 2], axis=-1) * 1000.0
        M[k] = d

    q2_res = {"models": {}, "diffs": {}, "vision_swap_cost": {}, "gripper": {}}
    for k in q1_models:
        q2_res["models"][k] = {n: stat(M[k][n]) for n in ("l8", "l50", "e8", "e25", "e50")}
        for n in EV:
            q2_res["models"][k]["ev_" + n] = stat(M[k]["ev_" + n], None, evg[n])

    onset_cache = {}
    for k in [x for x in ("E106C", "WAM_v1_s90", "WAM_N0_clean", "WAM_N1_fastp", "WAM_N2_kinskip") if x in P]:
        if (k + "_swap") in M:
            q2_res["vision_swap_cost"][k] = {n: stat(M[k + "_swap"][n] - M[k][n]) for n in ("l8", "l50", "e8", "e25", "e50")}
        pg = (P[k][:, :, [6, 13]].numpy() > thr).astype("i1").transpose(0, 2, 1).reshape(-1, 50)
        acc = (pg == ts).mean(1)
        by_p = {}
        for wi, e in enumerate(entries):
            p_rec = by_p.setdefault(e["parent"], [0, 0, 0])
            z, qq, _, good = dev.states.get(dev, e["cid"])
            if e["cid"] not in onset_cache:
                onset_cache[e["cid"]] = onset_labels(qq, z["continuous_segment"], good, c_dict)[0]
            init_g = onset_cache[e["cid"]][e["anchor"]]
            ps_w = (P[k][wi, :, [6, 13]].numpy() > thr).astype("i1")
            ts_w = GL[wi].numpy().astype("i1")
            cur_w = (GC[wi].numpy() > 0.5).astype("i1")
            for side in range(2):
                pe = transitions(ps_w[:, side], cur_w[side])
                te = transitions(ts_w[:, side], init_g[side])
                p_rec[0] += len(match(pe, te, 10))
                p_rec[1] += len(pe)
                p_rec[2] += len(te)
        f1_p = np.array([2 * by_p[p][0] / max(1, by_p[p][1] + by_p[p][2]) for p in ps])
        q2_res["gripper"][k] = dict(state_acc=float(stat(acc, None, g2)[0]), f1_10=float(f1_p.mean()))

    comp_pairs = [
        ("WAM_N0_clean", "WAM_v1_s90"),
        ("WAM_N1_fastp", "WAM_N0_clean"),
        ("WAM_N1_fastp", "WAM_v1_s90"),
        ("WAM_N2_kinskip", "WAM_N1_fastp"),
        ("WAM_N2_kinskip", "WAM_v1_s90"),
        ("WAM_N0_clean", "blind_t3_s90"),
        ("WAM_N0_clean", "blind_t1_s90"),
        ("WAM_N1_fastp", "blind_t1_s90"),
        ("WAM_N2_kinskip", "blind_t1_s90"),
    ]
    for a, b in comp_pairs:
        if a in M and b in M:
            q2_res["diffs"][f"{a}-{b}"] = {n: stat(M[a][n] - M[b][n]) for n in ("l8", "l50", "e8", "e25", "e50")}
            for n in EV:
                q2_res["diffs"][f"{a}-{b}"]["ev_" + n] = stat(M[a]["ev_" + n] - M[b]["ev_" + n], None, evg[n])

    # ==========================================================================
    # Q3: Matched-Error Perturbed GT Action Driving World Head (+25, +50, +100)
    # ==========================================================================
    a_gt = TR.clone()  # (211, 50, 14)
    a_n1 = P["WAM_v1_s90"].clone()
    err_n1 = a_n1[:, :, JI] - a_gt[:, :, JI]  # (211, 50, 12)

    # 1. Cross-parent empirical error permutation (exact marginal l8, l50, jerk/acc spectrum!)
    rng_p = np.random.default_rng(130)
    perm_idx = np.zeros(N, dtype=int)
    for i in range(N):
        cands = np.where(pid != pid[i])[0]
        perm_idx[i] = int(rng_p.choice(cands))

    a_perm = a_n1.clone()  # carries same gripper predictions or perm gripper error
    a_perm[:, :, [6, 13]] = a_gt[:, :, [6, 13]]
    # Inject cross-parent gripper flip at exact empirical rate
    g_mismatch = (a_n1[:, :, [6, 13]] - a_gt[:, :, [6, 13]])[perm_idx]
    a_perm[:, :, [6, 13]] = torch.clamp(a_gt[:, :, [6, 13]] + g_mismatch, 0.0, 0.08)
    a_perm[:, :, JI] = a_gt[:, :, JI] + err_n1[perm_idx]

    # 2. Per-window exact L1-scaled smooth perturbation (every window i has ||a_ws[i]-a_gt[i]||_1 == ||a_n1[i]-a_gt[i]||_1!)
    a_winscaled = a_n1.clone()  # exact same grippers as a_n1!
    raw_r = err_n1[perm_idx]
    scale_w = err_n1.abs().mean((1, 2), keepdim=True) / raw_r.abs().mean((1, 2), keepdim=True).clamp_min(1e-8)
    a_winscaled[:, :, JI] = a_gt[:, :, JI] + raw_r * scale_w

    # 3. Pointwise mirror error: a_gt - (a_n1 - a_gt) = 2*a_gt - a_n1 (pointwise identical |error| at every i, t, j!)
    a_mirror = a_n1.clone()  # exact same grippers as a_n1!
    a_mirror[:, :, JI] = 2.0 * a_gt[:, :, JI] - a_n1[:, :, JI]

    # 4. Causal smooth AR(1) Gaussian perturbation matched per-(step, joint) to E|err_n1[t, j]|
    gen_g = np.random.default_rng(2026)
    z_ar = np.zeros((N, 50, 12), dtype="f4")
    rho = 0.96
    for t_i in range(50):
        eps_t = gen_g.standard_normal((N, 12)).astype("f4")
        z_ar[:, t_i, :] = eps_t if t_i == 0 else (rho * z_ar[:, t_i - 1, :] + np.sqrt(1.0 - rho ** 2) * eps_t)
    z_t = torch.from_numpy(z_ar)
    target_step_joint_l1 = err_n1.abs().mean(0, keepdim=True)  # (1, 50, 12)
    cur_z_l1 = z_t.abs().mean(0, keepdim=True).clamp_min(1e-8)
    a_ar1 = a_n1.clone()
    a_ar1[:, :, JI] = a_gt[:, :, JI] + z_t * (target_step_joint_l1 / cur_z_l1)

    # 5. Amplitude-damped GT toward copy: (1 - alpha)*a_gt + alpha*a_copy matched to exact l50 of WAM_v1_s90!
    a_copy = P["copy"]
    copy_l50 = float((a_copy[:, :, JI] - a_gt[:, :, JI]).abs().mean())
    target_l50 = float(err_n1.abs().mean())
    alpha_damp = min(1.0, target_l50 / max(copy_l50, 1e-8))
    a_damped = a_n1.clone()
    a_damped[:, :, JI] = (1.0 - alpha_damp) * a_gt[:, :, JI] + alpha_damp * a_copy[:, :, JI]

    cand_actions = {
        "gt": a_gt,
        "pred_N1 (WAM_v1)": a_n1,
        "gt_plus_perm_err": a_perm,
        "gt_plus_winscaled_err": a_winscaled,
        "gt_mirror_err (2*gt-N1)": a_mirror,
        "gt_smooth_ar1_err": a_ar1,
        f"gt_damped_to_copy (a={alpha_damp:.3f})": a_damped,
        "blind_t1_ens3": P["blind_t1_ens3"],
    }
    for extra_k in ("WAM_N0_clean", "WAM_N1_fastp", "WAM_N2_kinskip"):
        if extra_k in P:
            cand_actions[f"pred_N1 ({extra_k})"] = P[extra_k]

    cand_action_stats = {}
    for cname, ca in cand_actions.items():
        cj = ca[:, :, JI].numpy()
        cee = np.linalg.norm(ee_from_act(ca) - EEt, axis=-1) * 1000.0
        cand_action_stats[cname] = dict(
            l8=float(np.abs(cj[:, :8] - tj[:, :8]).mean()),
            l50=float(np.abs(cj - tj).mean()),
            e25_mm=float(cee[:, 24].mean()),
            e50_mm=float(cee[:, 49].mean()),
            traj_rms_disp_rad=float(np.sqrt(((cj - q0_j[:, None, :]) ** 2).mean())),
        )

    w_models_to_test = [
        ("WAM_v1_s90_W", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt"),
        ("E106C_W", "orig", INIT),
    ]
    if (T / "results/e130_wam_n0_s90_20261119_01/ckpt_04000.pt").exists():
        w_models_to_test.append(("WAM_N0_clean_W", "N0", T / "results/e130_wam_n0_s90_20261119_01/ckpt_04000.pt"))

    q3_world = {}
    for w_label, w_kind, w_path in w_models_to_test:
        wm = load_any_wam(w_kind, w_path)
        q3_world[w_label] = {}
        for h in (25, 50, 100):
            Eh = [e for e in entries if h in e["horizons"]]
            psh = sorted({e["parent"] for e in Eh})
            W_h = np.random.default_rng(90).integers(len(psh), size=(2000, len(psh)))
            by_p = {p: {"cp": [], "no": [], **{cn: [] for cn in cand_actions}} for p in psh}

            with torch.inference_mode():
                for s in range(0, len(Eh), 4):
                    es = Eh[s : s + 4]
                    bh = move(float_batch(dev.batch([e["index"] for e in es], (h,), "short")), "cuda")
                    Y = bh["world"].double()
                    cp_m = Y.square().mean((1, 2)).cpu().numpy()
                    no_m = (wm.forward_mean(dict(bh, robot=torch.zeros_like(bh["robot"]), robot_valid=torch.zeros_like(bh["robot_valid"])))["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()
                    e_pos = [entries.index(e) for e in es]

                    c_mses = {}
                    for cname, ca in cand_actions.items():
                        a_b = ca[e_pos].to("cuda")
                        c_mses[cname] = (wm.forward_mean(dict(bh, robot=trajectory(bh["context"], a_b)))["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()

                    for k_i, e in enumerate(es):
                        rec = by_p[e["parent"]]
                        rec["cp"].append(float(cp_m[k_i]))
                        rec["no"].append(float(no_m[k_i]))
                        for cname in cand_actions:
                            rec[cname].append(float(c_mses[cname][k_i]))

            sk = lambda key: np.array([100.0 * (1.0 - np.mean(by_p[p][key]) / max(np.mean(by_p[p]["cp"]), 1e-12)) for p in psh])
            s_no = sk("no")
            s_gt = sk("gt")
            denom = float((s_gt - s_no).mean())
            h_dict = {"no_traj_skill": float(s_no.mean()), "candidates": {}}
            for cname in cand_actions:
                s_c = sk(cname)
                d_no = s_c - s_no
                bs_d = d_no[W_h].mean(1)
                h_dict["candidates"][cname] = dict(
                    skill=float(s_c.mean()),
                    minus_no_pp=[float(d_no.mean()), float(np.quantile(bs_d, 0.025)), float(np.quantile(bs_d, 0.975))],
                    retention_pct=float(100.0 * d_no.mean() / denom) if abs(denom) > 1e-9 else 0.0,
                )
            q3_world[w_label][str(h)] = h_dict
        del wm
        torch.cuda.empty_cache()

    full_summary = dict(
        experiment="E130_diagnostic_breakdown_clean_state_and_perturbed_gt_world",
        brt=brt(),
        q1_breakdown=q1_res,
        q2_clean_state=q2_res,
        q3_perturbed_gt=dict(action_stats=cand_action_stats, world=q3_world),
    )
    (OUT / "summary.json").write_text(json.dumps(full_summary, indent=2, ensure_ascii=False) + "\n")
    write_markdown(full_summary)
    print("Wrote E130 summary & report!", flush=True)


def fmd(arr, d=5):
    return f"{arr[0]:+.{d}f} [{arr[1]:+.{d}f}, {arr[2]:+.{d}f}]"


def write_markdown(s):
    q1 = s["q1_breakdown"]
    q2 = s["q2_clean_state"]
    q3 = s["q3_perturbed_gt"]

    lines = [
        "# E130 三项机制核验：按关节/步数误差拆解、去状态噪声重训、同幅扰动真值驱动世界头",
        "",
        f"完成时间：{s['brt']}。代码：`code_e130/`；结果：`results/e130_*_20261119_01/`。",
        "",
        "## 一、问题 1：按关节、按步数拆分误差（直接检验“腕部旋转”解释与前 8 步差距来源）",
        "",
        "### 1.1 前 8 步逐帧 + 中长时域逐帧误差（关节 L1 rad / 末端物理误差 mm）",
        "",
        "| 步数 `t` | `WAM_v1` 关节/末端 | `blind_t1` 关节/末端 | `WAM_N0(去噪)` 关节/末端 | `WAM_N1(快LR)` 关节/末端 | `WAM_N2(直连skip)` 关节/末端 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for st_k in ("1", "2", "3", "4", "5", "6", "7", "8", "12", "25", "50"):
        row = [f"| `+{st_k}` |"]
        for m in ("WAM_v1_s90", "blind_t1_s90", "WAM_N0_clean", "WAM_N1_fastp", "WAM_N2_kinskip"):
            if m in q1["per_step_l1"]:
                row.append(f" {q1['per_step_l1'][m][st_k]:.5f} / {q1['per_step_ee_mm'][m][st_k]:.2f}mm |")
            else:
                row.append(" — |")
        lines.append("".join(row))

    lines.extend([
        "",
        "### 1.2 按 6 个关节自由度拆分（左右臂均值）及末端物理杠杆率 `||∂EE/∂q||`",
        "",
        "| 关节 | 末端杠杆 (mm/rad) | 前8步 `WAM_v1` | 前8步 `blind_t1` | 前8步差值(占总差%) | 50步 `WAM_v1` | 50步 `blind_t1` | 50步差值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    w8 = q1["per_joint_l8"]["WAM_v1_s90"]["j1_to_j6_avg_lr"]
    b8 = q1["per_joint_l8"]["blind_t1_s90"]["j1_to_j6_avg_lr"]
    w50 = q1["per_joint_l50"]["WAM_v1_s90"]["j1_to_j6_avg_lr"]
    b50 = q1["per_joint_l50"]["blind_t1_s90"]["j1_to_j6_avg_lr"]
    tot_d8 = sum(w8[i] - b8[i] for i in range(6))
    for i, jn in enumerate(q1["joint_names_6"]):
        d8 = w8[i] - b8[i]
        d50 = w50[i] - b50[i]
        pct8 = 100.0 * d8 / tot_d8 if abs(tot_d8) > 1e-9 else 0.0
        lines.append(
            f"| `{jn}` | {q1['jacobian_mm_per_rad_j1_to_j6'][i]:.1f} | {w8[i]:.5f} | {b8[i]:.5f} | "
            f"{d8:+.5f} ({pct8:.1f}%) | {w50[i]:.5f} | {b50[i]:.5f} | {d50:+.5f} |"
        )

    lines.extend([
        "",
        "### 1.3 正运动学手术反事实（单独保留近端 `j1..j3` 误差 vs 单独保留腕部 `j4..j6` 误差）与分位数分布",
        "",
        "| 模型 | `+8`全误差 / 仅近端 / 仅腕部 (mm) | `+50`全误差 / 仅近端 / 仅腕部 (mm) | `+50`末端分位数 `P50 / P75 / P90 / P95 / 均值` (mm) |",
        "|---|---:|---:|---:|",
    ])
    for m in ("E106C", "WAM_v1_s90", "blind_t3_s90", "blind_t1_s90", "WAM_N0_clean", "WAM_N1_fastp", "WAM_N2_kinskip"):
        if m in q1["fk_surgical_mm"]:
            fk = q1["fk_surgical_mm"][m]
            qt = q1["window_quantiles"][m]["e50_p50_p75_p90_p95_mean"]
            lines.append(
                f"| `{m}` | {fk['e8_full']:.2f} / {fk['e8_prox_only']:.2f} / {fk['e8_wrist_only']:.2f} | "
                f"{fk['e50_full']:.2f} / {fk['e50_prox_only']:.2f} / {fk['e50_wrist_only']:.2f} | "
                f"{qt[0]:.1f} / {qt[1]:.1f} / {qt[2]:.1f} / {qt[3]:.1f} / **{qt[4]:.2f}** |"
            )

    lines.extend([
        "",
        "## 二、问题 2：去掉状态噪声与状态丢弃重训（`N0` / `N1` / `N2` 对照）",
        "",
        "| 模型 | 前8步 `l8` | 50步 `l50` | +8步末端 | +25步末端 | +50步末端 | 夹紧37次 | 松爪16次 | 换图代价(+8/+25/+50) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for k, r in q2["models"].items():
        sw = q2["vision_swap_cost"].get(k)
        sw_s = f"+{sw['e8'][0]:.2f}/+{sw['e25'][0]:.2f}/+{sw['e50'][0]:.2f}mm" if sw else "—"
        lines.append(
            f"| `{k}` | {r['l8'][0]:.5f} | {r['l50'][0]:.5f} | {r['e8'][0]:.2f} mm | {r['e25'][0]:.2f} mm | "
            f"{r['e50'][0]:.2f} mm | {r['ev_close'][0]:.2f} mm | {r['ev_open'][0]:.2f} mm | {sw_s} |"
        )

    lines.extend([
        "",
        "### 关键配对差值",
        "",
        "| 配对比较 | `Δl8` | `Δl50` | `Δe8` (mm) | `Δe50` (mm) |",
        "|---|---:|---:|---:|---:|",
    ])
    for pair, d in q2["diffs"].items():
        lines.append(f"| `{pair}` | {fmd(d['l8'], 5)} | {fmd(d['l50'], 5)} | {fmd(d['e8'], 2)} | {fmd(d['e50'], 2)} |")

    lines.extend([
        "",
        "## 三、问题 3：同幅扰动真值动作驱动世界头（检验“动作误差幅度” vs “信息上限”）",
        "",
        "| 驱动动作条件 | 实测 `l8 / l50` (vs GT) | `+50`末端误差 | 动作RMS位移 | `+25` 保留率 | `+50` `skill−no` (pp) | `+50` 保留率 | `+100` 保留率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    w_main = q3["world"]["WAM_v1_s90_W"]
    for cname, ast in q3["action_stats"].items():
        c25 = w_main["25"]["candidates"][cname]
        c50 = w_main["50"]["candidates"][cname]
        c100 = w_main["100"]["candidates"][cname]
        lines.append(
            f"| `{cname}` | {ast['l8']:.5f} / {ast['l50']:.5f} | {ast['e50_mm']:.2f} mm | {ast['traj_rms_disp_rad']:.4f} | "
            f"{c25['retention_pct']:.1f}% | {fmd(c50['minus_no_pp'], 2)} | **{c50['retention_pct']:.1f}%** | **{c100['retention_pct']:.1f}%** |"
        )

    REPORT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        _, _, tr, dev = datasets()
        entries = json.loads(LOCK_PATH.read_text())["entries"]
        _, norm_dev = build_or_load_extra_stats(tr, dev, entries[:4])
        cp = torch.load(INIT, map_location="cpu", weights_only=False)
        mw = E130WAM(tr.norm, clean_state=True, use_kin_skip=True)
        mw.load_state_dict(cp["model"], strict=False)
        mw.widen_from_164().cuda().eval()
        b = move(float_batch(dev.batch([e["index"] for e in entries[:4]], (25,), "short")), "cuda")
        b = attach_extra_proprio(b, np.stack([norm_dev[e["index"]] for e in entries[:4]]))
        with torch.inference_mode():
            act = infer_d(mw, b, 4, 90)
        sens = compute_empirical_jacobian_mm_per_rad(b["q_current"].cpu(), np.asarray(JOINTS))
        assert act.shape == (4, 50, 14) and sens.shape == (4, 12) and np.isfinite(sens).all()
        print("eval_e130 smoke check passed!", flush=True)
    else:
        main()
