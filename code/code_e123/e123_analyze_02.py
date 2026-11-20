# E123 (V1) Step B (CPU): Analyze world-head local action sensitivity, wrist-camera gripper counterfactuals, and candidate reranking.
import os, sys, json, pickle
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

os.environ["ARM"] = "CF"
T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
sys.path[:0] = [str(T / "code_e102s2"), str(T / "code_e86")]
from s2 import predict, calibrate, emission, truths, PI, A

D = T / "results/e123_action_sens_20261118_02"
CF = T / "results/e102s2_cf_20261116_01"
FEATS = T / "results/e115p1_stage_feats"

ACT_CONDS = [
    "gt", "s15", "s12", "s08", "s05", "freeze", "rev05",
    "lag15", "lead15", "g_hold", "g_flip",
    "n1", "n1_s13", "n1_s07", "n1_gflip",
    "no", "sh",
]
ALL_CONDS = ["real", "cur"] + ACT_CONDS

pooled = np.load(D / "pooled.npz")
lmse = np.load(D / "latent_mse.npz")
rows = json.loads((D / "rows.json").read_text())

with (CF / "H0_full.pkl").open("rb") as f:
    head = pickle.load(f)
with (CF / "final_aux.pkl").open("rb") as f:
    aux = pickle.load(f)["H0"]

# Load causal stage controller posteriors at anchor for stage160 parents
ctrl_cache = {}
def get_ctrl_stage(parent, anchor):
    if parent not in ctrl_cache:
        ctrl_cache[parent] = np.load(FEATS / (parent + ".npz"))["ctrl"].astype("f4")
    arr = ctrl_cache[parent]
    vec = arr[min(max(int(anchor), 0), len(arr) - 1)]
    return vec[:10]  # 10-D causal filtered soft posterior


prob = {}
cal_prob = {}
post_stage = {}
stg_argmax = {}
for c in ALL_CONDS:
    pr, _ = predict(head, pooled[c])
    prob[c] = pr
    x = calibrate(pr, aux["cal"])
    x[:, PI["P073"]] = 0.5
    logits = np.c_[x, np.ones(len(x))] @ aux["w"]
    log_post = logits - logsumexp(logits, axis=1, keepdims=True)
    cal_prob[c] = x
    post_stage[c] = np.exp(log_post)
    stg_argmax[c] = emission(pr, aux).argmax(1)

N = len(rows)
par = np.array([r["parent"] for r in rows])
H = np.array([r["h"] for r in rows])
in_s160 = np.array([bool(r["in_stage160"]) for r in rows])
disp_h = np.array([r["joint_disp_h"] for r in rows])
sw_L = np.array([bool(r["grip_sw_L_h"]) for r in rows])
sw_R = np.array([bool(r["grip_sw_R_h"]) for r in rows])
sw_any = sw_L | sw_R

# Retrospective model stage labels (for stage160 reference only; explicitly non-human)
tf = np.array([
    truths[r["parent"]][r["anchor"] + r["h"]]
    if (r["in_stage160"] and r["parent"] in truths and r["anchor"] + r["h"] < len(truths[r["parent"]]))
    else -1
    for r in rows
])
ta = np.array([
    truths[r["parent"]][r["anchor"]]
    if (r["in_stage160"] and r["parent"] in truths and r["anchor"] < len(truths[r["parent"]]))
    else -1
    for r in rows
])

# Causal controller stage at anchor (strictly causal, zero future leakage)
s_ctrl_cur = np.full(N, -1, dtype=int)
s_ctrl_post = np.zeros((N, 10), dtype="f4")
for i, r in enumerate(rows):
    if r["in_stage160"]:
        p10 = get_ctrl_stage(r["parent"], r["anchor"])
        s_ctrl_post[i] = p10
        s_ctrl_cur[i] = int(np.argmax(p10))


def boot_draws(parents_sorted):
    return np.random.default_rng(90).integers(0, len(parents_sorted), size=(2000, len(parents_sorted)))


def fmt_ci(mean_val, ci_pair, digits=3):
    return f"{mean_val:.{digits}f} [{ci_pair[0]:.{digits}f}, {ci_pair[1]:.{digits}f}]"


# ==============================================================================
# PART 1: Latent World Skill (all 211 windows / 27 parents, ZERO model labels)
# Exact parent-equal skill formula matching E106-C / E110-F
# ==============================================================================
copy_all = lmse["copy_all"]
copy_cam = lmse["copy_cam"]  # (N, 3): cam_high, cam_left_wrist, cam_right_wrist


def parent_skill_dict(cond, h, mask=None, cam=None):
    sel = (H == h) if mask is None else ((H == h) & mask)
    mse_arr = lmse[f"all_{cond}"] if cam is None else lmse[f"cam_{cond}"][:, cam]
    cp_arr = copy_all if cam is None else copy_cam[:, cam]
    out = {}
    for p in sorted(set(par[sel])):
        ix = np.where(sel & (par == p))[0]
        if len(ix) == 0:
            continue
        out[p] = 100.0 * (1.0 - float(mse_arr[ix].mean()) / max(float(cp_arr[ix].mean()), 1e-12))
    return out


def eval_skill(cond, h, ref=None, mask=None, cam=None):
    d1 = parent_skill_dict(cond, h, mask=mask, cam=cam)
    if ref is None:
        ps = sorted(d1)
        vals = np.array([d1[p] for p in ps])
    else:
        d0 = parent_skill_dict(ref, h, mask=mask, cam=cam)
        ps = sorted(set(d1) & set(d0))
        vals = np.array([d1[p] - d0[p] for p in ps])
    draws = boot_draws(ps)
    bs = vals[draws].mean(1)
    return float(vals.mean()), [float(x) for x in np.quantile(bs, [0.025, 0.975])], len(ps)


part1 = {}
print("=== PART 1: Latent World Skill (%) on 211 windows / 27 parents (zero model labels) ===")
for h in (10, 25, 50, 100):
    part1[str(h)] = {}
    print(f"\n--- Horizon +{h} (n={int((H==h).sum())}) ---")
    for c in ACT_CONDS:
        sk = eval_skill(c, h)
        d_gt = eval_skill(c, h, ref="gt") if c != "gt" else (0.0, [0.0, 0.0], sk[2])
        d_no = eval_skill(c, h, ref="no") if c != "no" else (0.0, [0.0, 0.0], sk[2])
        part1[str(h)][c] = dict(
            skill_pct=sk[0], skill_ci95=sk[1],
            delta_vs_gt_pp=d_gt[0], delta_vs_gt_ci95=d_gt[1],
            delta_vs_no_pp=d_no[0], delta_vs_no_ci95=d_no[1],
            parents=sk[2],
        )
        print(f"  {c:8s}: skill={fmt_ci(sk[0], sk[1], 2)}% | vs_gt={fmt_ci(d_gt[0], d_gt[1], 2)} pp | vs_no={fmt_ci(d_no[0], d_no[1], 2)} pp")

# High-motion half (top 50% joint_disp_h at each horizon) to avoid stationary dilution
part1_moving = {}
print("\n=== PART 1B: Latent Skill Diff vs GT on Active-Motion Windows (top 50% joint_disp_h) ===")
for h in (25, 50, 100):
    med = float(np.median(disp_h[H == h]))
    m_mov = disp_h >= med
    part1_moving[str(h)] = {"median_disp_rad": med}
    line = []
    for c in ("s12", "s08", "s15", "s05", "lag15", "lead15", "g_hold", "g_flip", "n1", "freeze", "no", "rev05"):
        r = eval_skill(c, h, ref="gt", mask=m_mov)
        part1_moving[str(h)][c] = dict(delta_vs_gt_pp=r[0], ci95=r[1], parents=r[2])
        line.append(f"{c}:{fmt_ci(r[0], r[1], 2)}")
    print(f"  +{h} (disp>={med:.4f} rad): " + " | ".join(line))

# Also check n1 local perturbations vs n1
part1_around_n1 = {}
print("\n=== PART 1C: Perturbations Around Model Action n1 (vs n1, all 211 windows) ===")
for h in (25, 50, 100):
    part1_around_n1[str(h)] = {}
    line = []
    for c in ("gt", "n1_s13", "n1_s07", "n1_gflip", "no"):
        r = eval_skill(c, h, ref="n1")
        part1_around_n1[str(h)][c] = dict(delta_vs_n1_pp=r[0], ci95=r[1], parents=r[2])
        line.append(f"{c}-n1:{fmt_ci(r[0], r[1], 2)}")
    print(f"  +{h}: " + " | ".join(line))


# ==============================================================================
# PART 2: Wrist-Camera Gripper Counterfactual Test (zero model labels + predicates)
# ==============================================================================
part2 = {}
print("\n=== PART 2: Gripper Counterfactuals (Per-Camera Skill Diff vs GT: g_hold - gt, g_flip - gt) ===")
CAM_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
for h in (25, 50, 100):
    part2[str(h)] = {}
    for sub_name, mask in (
        ("all_windows", np.ones(N, dtype=bool)),
        ("any_grip_switch", sw_any),
        ("no_grip_switch", ~sw_any),
        ("left_grip_switch", sw_L),
        ("right_grip_switch", sw_R),
    ):
        n_win = int(((H == h) & mask).sum())
        if n_win < 3:
            continue
        part2[str(h)][sub_name] = {"windows": n_win}
        for c in ("g_hold", "g_flip"):
            r_all = eval_skill(c, h, ref="gt", mask=mask, cam=None)
            cams = {}
            for ci_idx, cname in enumerate(CAM_NAMES):
                rc = eval_skill(c, h, ref="gt", mask=mask, cam=ci_idx)
                cams[cname] = dict(delta_pp=rc[0], ci95=rc[1], parents=rc[2])
            part2[str(h)][sub_name][c] = dict(overall_pp=r_all[0], overall_ci95=r_all[1], parents=r_all[2], by_cam=cams)
        if sub_name in ("any_grip_switch", "no_grip_switch", "left_grip_switch", "right_grip_switch"):
            gh = part2[str(h)][sub_name]["g_hold"]
            gf = part2[str(h)][sub_name]["g_flip"]
            print(
                f"  +{h} {sub_name:18s} (n={n_win:3d}, p={gh['parents']:2d}): "
                f"g_hold-gt all={fmt_ci(gh['overall_pp'], gh['overall_ci95'], 2)} "
                f"[L_wrist={fmt_ci(gh['by_cam']['cam_left_wrist']['delta_pp'], gh['by_cam']['cam_left_wrist']['ci95'], 2)}, "
                f"R_wrist={fmt_ci(gh['by_cam']['cam_right_wrist']['delta_pp'], gh['by_cam']['cam_right_wrist']['ci95'], 2)}] | "
                f"g_flip-gt all={fmt_ci(gf['overall_pp'], gf['overall_ci95'], 2)} "
                f"[L_wrist={fmt_ci(gf['by_cam']['cam_left_wrist']['delta_pp'], gf['by_cam']['cam_left_wrist']['ci95'], 2)}, "
                f"R_wrist={fmt_ci(gf['by_cam']['cam_right_wrist']['delta_pp'], gf['by_cam']['cam_right_wrist']['ci95'], 2)}]"
            )


# ==============================================================================
# PART 3: P2 Predicate & Stage Readout Across Local Perturbations (160 stage-dev windows / 20 parents)
# Exact P2 bootstrap protocol (multiplicity-preserving parent draws, seed 90)
# ==============================================================================
s160_rows = np.where(in_s160)[0]
ps160 = sorted(set(par[s160_rows]))
draws160 = boot_draws(ps160)
pidx160 = {p: np.where(in_s160 & (par == p))[0] for p in ps160}


def p2_stat(h, kind, c):
    def f(idx):
        idx = idx[H[idx] == h]
        if kind == "pcorr":
            a = prob[c][idx]
            b = prob["real"][idx]
            o = []
            for j in range(21):
                if j == PI["P073"] or a[:, j].std() < 1e-6 or b[:, j].std() < 1e-6:
                    continue
                o.append(np.corrcoef(a[:, j], b[:, j])[0, 1])
            return float(np.mean(o)) if o else np.nan
        if kind == "pbrier":
            # Mean squared error against real future token's calibrated predicate probabilities (lower is better; negate so higher is better)
            valid_j = [j for j in range(21) if j != PI["P073"]]
            return -float(((cal_prob[c][idx][:, valid_j] - cal_prob["real"][idx][:, valid_j]) ** 2).mean())
        if kind == "fid":
            return float((stg_argmax[c][idx] == stg_argmax["real"][idx]).mean())
        if kind == "acc":
            m = tf[idx] >= 0
            return float((stg_argmax[c][idx][m] == tf[idx][m]).mean()) if m.any() else np.nan
    return f


def p2_run(h, kind, c, base=None):
    f = p2_stat(h, kind, c)
    g = p2_stat(h, kind, base) if base else None
    fn = (lambda idx: f(idx) - g(idx)) if g else f
    pt = fn(s160_rows)
    bs = np.array([fn(np.concatenate([pidx160[ps160[i]] for i in d])) for d in draws160])
    return float(pt), [float(x) for x in np.nanquantile(bs, [0.025, 0.975])]


part3 = {}
print("\n=== PART 3: Predicate/Stage Readout on 160 Stage-Dev Windows (20 parents; dev stage truth = model-generated) ===")
for h in (25, 50, 100):
    part3[str(h)] = {}
    print(f"\n--- Stage-Dev Horizon +{h} ---")
    for c in ("cur", "gt", "s12", "s08", "n1", "n1_s13", "n1_s07", "lag15", "lead15", "s05", "g_hold", "g_flip", "no", "freeze", "rev05", "sh"):
        pc = p2_run(h, "pcorr", c)
        pc_d = p2_run(h, "pcorr", c, "gt") if c != "gt" else (0.0, [0.0, 0.0])
        fd = p2_run(h, "fid", c)
        fd_d = p2_run(h, "fid", c, "gt") if c != "gt" else (0.0, [0.0, 0.0])
        ac = p2_run(h, "acc", c)
        part3[str(h)][c] = dict(
            pcorr=pc[0], pcorr_ci95=pc[1],
            pcorr_vs_gt=pc_d[0], pcorr_vs_gt_ci95=pc_d[1],
            fid=fd[0], fid_ci95=fd[1],
            fid_vs_gt=fd_d[0], fid_vs_gt_ci95=fd_d[1],
            acc=ac[0], acc_ci95=ac[1],
        )
        print(
            f"  {c:8s}: pcorr={fmt_ci(pc[0], pc[1])} (vs_gt {fmt_ci(pc_d[0], pc_d[1])}) | "
            f"fid={fmt_ci(fd[0], fd[1])} | acc={fmt_ci(ac[0], ac[1])}"
        )


# ==============================================================================
# PART 4: Candidate Action Reranking Test (Can World Head + Critic Rank Actions?)
# Compare 3 Critics:
#   (A) Oracle Latent Critic: -MSE(W(a), z_real)  [World-head physical upper bound]
#   (B) Oracle Predicate Critic: -MSE(cal_prob(W(a)), cal_prob(z_real))  [Survives pooling + 21 probes]
#   (C) Causal Progress Critic (ZERO future leakage!):
#       Using causal S1 posterior p_ctrl(t) at anchor and legal transition matrix A:
#       score(a) = sum_{s0} p_ctrl(s0) * sum_{s1 != s0, A[s0, s1]} P_post(s1 | W(a))
#       (predicted probability of advancing to a legal next stage!)
# ==============================================================================
valid_pred_j = [j for j in range(21) if j != PI["P073"]]

# Precompute per-row critic scores for each ACT_COND
score_latent_oracle = {c: -lmse[f"all_{c}"] for c in ACT_CONDS}
score_pred_oracle = {
    c: -((cal_prob[c][:, valid_pred_j] - cal_prob["real"][:, valid_pred_j]) ** 2).mean(1)
    for c in ACT_CONDS
}
# Strictly causal forward-transition matrix F_mat[s0, s1] = 1 iff s1 != s0 and A[s0, s1]
F_mat = A.copy().astype("f4")
np.fill_diagonal(F_mat, 0.0)
# Expected legal next-stage probability weighted by causal S1 posterior at t: (N, 10) @ (10, 10) -> target weights (N, 10)
target_next_weights = s_ctrl_post @ F_mat  # (N, 10), strictly causal!
score_causal_prog = {
    c: (post_stage[c] * target_next_weights).sum(1)
    for c in ACT_CONDS
}


def pairwise_win_rate(better, worse, h, score_dict, mask):
    sel = (H == h) & mask
    s_b = score_dict[better]
    s_w = score_dict[worse]
    # Tie = 0.5
    win = np.where(s_b > s_w, 1.0, np.where(s_b < s_w, 0.0, 0.5))
    ps = sorted(set(par[sel]))
    p_means = np.array([win[sel & (par == p)].mean() for p in ps])
    draws = boot_draws(ps)
    bs = p_means[draws].mean(1)
    return float(p_means.mean()), [float(x) for x in np.quantile(bs, [0.025, 0.975])], len(ps), int(sel.sum())


part4 = {"pairwise": {}, "pool_rerank": {}}
print("\n=== PART 4A: Pairwise Action Ranking Win-Rate of GT over Perturbed Actions (chance = 0.500) ===")
PAIRS = [
    ("gt", "sh", "all_stage160", in_s160),
    ("gt", "rev05", "all_stage160", in_s160),
    ("gt", "freeze", "all_stage160", in_s160),
    ("gt", "s05", "all_stage160", in_s160),
    ("gt", "lag15", "all_stage160", in_s160),
    ("gt", "s08", "all_stage160", in_s160),
    ("gt", "s12", "all_stage160", in_s160),
    ("gt", "n1", "all_stage160", in_s160),
    ("gt", "g_flip", "grip_switch_stage160", in_s160 & sw_any),
    ("gt", "g_hold", "grip_switch_stage160", in_s160 & sw_any),
    # Also on stage-change windows where true advancement actually happens within +h!
    ("gt", "freeze", "stage_change_windows", in_s160 & (tf >= 0) & (ta >= 0) & (tf != ta)),
    ("gt", "s05", "stage_change_windows", in_s160 & (tf >= 0) & (ta >= 0) & (tf != ta)),
    ("gt", "lag15", "stage_change_windows", in_s160 & (tf >= 0) & (ta >= 0) & (tf != ta)),
    ("gt", "s08", "stage_change_windows", in_s160 & (tf >= 0) & (ta >= 0) & (tf != ta)),
    ("gt", "n1", "stage_change_windows", in_s160 & (tf >= 0) & (ta >= 0) & (tf != ta)),
]

for h in (50, 100):
    part4["pairwise"][str(h)] = {}
    print(f"\n--- Pairwise Win-Rates at Horizon +{h} ---")
    for b_c, w_c, m_name, mask in PAIRS:
        if int(((H == h) & mask).sum()) < 5:
            continue
        r_lat = pairwise_win_rate(b_c, w_c, h, score_latent_oracle, mask)
        r_prd = pairwise_win_rate(b_c, w_c, h, score_pred_oracle, mask)
        r_cau = pairwise_win_rate(b_c, w_c, h, score_causal_prog, mask)
        key = f"{b_c}_vs_{w_c}|{m_name}"
        part4["pairwise"][str(h)][key] = dict(
            windows=r_lat[3], parents=r_lat[2],
            oracle_latent=dict(win=r_lat[0], ci95=r_lat[1]),
            oracle_pred=dict(win=r_prd[0], ci95=r_prd[1]),
            causal_prog=dict(win=r_cau[0], ci95=r_cau[1]),
        )
        print(
            f"  {b_c} > {w_c:7s} [{m_name:20s}, n={r_lat[3]:3d}]: "
            f"OracleLat={fmt_ci(r_lat[0], r_lat[1])} | "
            f"OraclePred={fmt_ci(r_prd[0], r_prd[1])} | "
            f"CausalProg={fmt_ci(r_cau[0], r_cau[1])}"
        )

# Candidate pool reranking: pick argmax_c score(c) at h=50 and evaluate resulting 50-step joint L1 error!
print("\n=== PART 4B: Candidate Pool Reranking at h=+50 (160 stage-dev windows, 50-step Joint L1 Error) ===")
POOLS = {
    "broad_5cand": ["gt", "s05", "lag15", "freeze", "rev05"],
    "fine_gt_ladder": ["gt", "s12", "s08", "s05", "lag15"],
    "deploy_around_n1": ["n1_s07", "n1", "n1_s13"],
}
sel50 = (H == 50) & in_s160
ps50 = sorted(set(par[sel50]))
draws50 = boot_draws(ps50)

for pool_name, cands in POOLS.items():
    l1_mat = np.column_stack([np.array([r["l1_50"][c] for r in rows])[sel50] for c in cands])  # (M, K)
    unif_l1 = l1_mat.mean(1)
    best_l1 = l1_mat.min(1)
    res_pool = {}
    for critic_name, s_dict in (
        ("causal_prog", score_causal_prog),
        ("oracle_pred", score_pred_oracle),
        ("oracle_latent", score_latent_oracle),
    ):
        s_mat = np.column_stack([s_dict[c][sel50] for c in cands])
        pick = np.argmax(s_mat, axis=1)
        picked_l1 = l1_mat[np.arange(len(pick)), pick]
        # Parent-equal mean L1 and improvement over uniform random candidate (or over n1 for deploy_around_n1)
        ref_l1 = np.array([r["l1_50"]["n1"] for r in rows])[sel50] if pool_name == "deploy_around_n1" else unif_l1
        par_s = par[sel50]
        p_pick = np.array([picked_l1[par_s == p].mean() for p in ps50])
        p_diff = np.array([(picked_l1[par_s == p] - ref_l1[par_s == p]).mean() for p in ps50])
        bs_pick = p_pick[draws50].mean(1)
        bs_diff = p_diff[draws50].mean(1)
        res_pool[critic_name] = dict(
            picked_l1_50=float(p_pick.mean()),
            picked_ci95=[float(x) for x in np.quantile(bs_pick, [0.025, 0.975])],
            diff_vs_ref=float(p_diff.mean()),
            diff_ci95=[float(x) for x in np.quantile(bs_diff, [0.025, 0.975])],
            pick_counts={cands[k]: int((pick == k).sum()) for k in range(len(cands))},
        )
    p_ref = np.array([ (np.array([r["l1_50"]["n1"] for r in rows])[sel50] if pool_name == "deploy_around_n1" else unif_l1)[par[sel50] == p].mean() for p in ps50 ])
    p_oracle_best = np.array([best_l1[par[sel50] == p].mean() for p in ps50])
    res_pool["ref_l1_50"] = float(p_ref.mean())
    res_pool["oracle_best_action_l1_50"] = float(p_oracle_best.mean())
    part4["pool_rerank"][pool_name] = res_pool
    print(f"  Pool {pool_name} (ref L1={p_ref.mean():.5f}, best-in-pool={p_oracle_best.mean():.5f}):")
    for cn in ("causal_prog", "oracle_pred", "oracle_latent"):
        rc = res_pool[cn]
        print(
            f"    {cn:14s}: L1={fmt_ci(rc['picked_l1_50'], rc['picked_ci95'], 5)} "
            f"(Δ={fmt_ci(rc['diff_vs_ref'], rc['diff_ci95'], 5)}) | picks={rc['pick_counts']}"
        )

summary = dict(
    experiment="E123_V1_local_action_sensitivity",
    windows_total=211,
    parents_total=27,
    windows_stage160=160,
    parents_stage160=20,
    dev_stage_truth_note="model-generated retrospective labels (non-human), used only in Part 3 acc & stage_change stratify",
    part1_latent_skill_27p=part1,
    part1b_moving_vs_gt_27p=part1_moving,
    part1c_around_n1_27p=part1_around_n1,
    part2_gripper_counterfactual_27p=part2,
    part3_predicate_stage_160w=part3,
    part4_reranking_160w=part4,
)

# Print summary check before saving (handover rule)
print("\nSaving summary.json to", D / "summary.json", "keys:", list(summary.keys()))
(D / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print("ALL DONE.")
