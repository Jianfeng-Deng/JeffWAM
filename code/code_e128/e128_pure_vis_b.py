# E128-B: True Pure-Vision Residual Head (unfrozen visual layers + strictly zero proprioception) & 5-fold Parent-OOF Shrinkage Diagnosis.
import os, sys, json, time, hashlib, datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.update(E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4")
T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
sys.path.insert(0, str(T / "code_e86"))
for p in ("code_e110f", "code_e106c", "code_e102a"):
    sys.path.insert(0, str(T / p))

from e99w2_model import WorldModel, JOINTS, pose_torch
from e102x_setup import datasets
from e102a_contract import contract
from e90.sampling import MultiSourceSampler
from e90.data import sample_horizons
from e86.common import seed_for
from e86.data import move
from e110f_core import infer_d, tensor_sha, target_delta
from e110f_eval import float_batch

BL = T / "results/e127r_blind_20261118_01"
INIT = T / "results/e106c_f2_s90_20261115_02/ckpt_20000.pt"
D_CKPT = T / "results/e110f_d_20261116_05/ckpt_04000.pt"
LOCK = T / "results/e106c_control_20261115_02/PRETRAIN_LOCK.json"
OUT = T / "results/e128_ablation_20261119_01"

PURE_VIS_PREFIXES = (
    "current.", "pool.", "camera", "obs_age.", "hist_time", "history_camera",
    "text_proj.", "null_text", "kind", "a_pos", "ta.",
    "blocks.", "norm.", "delta_head.", "grip_head.",
)


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def strip_proprio(b):
    out = dict(b)
    out["q_normalized"] = torch.zeros_like(b["q_normalized"])
    out["proprio"] = torch.zeros_like(b["proprio"])
    return out


class SpatialPureVisNet(nn.Module):
    def __init__(self, d_cell=2304, d_tok=48, hidden=512):
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(d_cell, d_tok), nn.LayerNorm(d_tok), nn.GELU())
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(24 * d_tok, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, 600),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        # x: (B, 24, 2304)
        z = self.cell(x).flatten(1)
        return self.head(z).reshape(-1, 50, 12)


def extract_vis_cells(obs_np, hv_np):
    # obs_np: (5, 3, 8, 768), hv_np: (4,) for (-24, -18, -12, -6) -> idx 0,1,2,3
    cur = obs_np[4]  # (3, 8, 768)
    d6 = (cur - obs_np[3]) if hv_np[3] else np.zeros_like(cur)
    d18 = (cur - obs_np[1]) if hv_np[1] else np.zeros_like(cur)
    # concat along feature dim -> (3, 8, 2304) -> flatten cells (24, 2304) float16
    return np.concatenate([cur, d6, d18], axis=-1).reshape(24, 2304).astype("f2")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)

    _, _, tr, dev = datasets()
    entries = json.loads(LOCK.read_text())["entries"]
    assert len(entries) == 211
    JI = np.asarray(JOINTS)
    J_cuda = torch.as_tensor(JOINTS, device="cuda")
    thr = np.asarray(contract()["threshold_m"])

    B_oof = np.load(BL / "oof_train.npy")  # (34777, 50, 12)
    fold5 = torch.from_numpy(np.load(BL / "dev_fold5.npy"))  # (211, 50, 12)
    full3 = torch.from_numpy(np.load(BL / "dev_full3.npy"))  # (211, 50, 12)
    assert len(B_oof) == len(tr.rows)

    # ==========================================================================
    # 1. Train WAM_PV: Fix E127-R by UNFREEZING visual backbone & ZEROING proprio!
    # ==========================================================================
    print(f"[{brt()}] Starting WAM_PV (unfrozen visual backbone, zero proprio, unweighted L1 on y - B_oof)...", flush=True)
    torch.manual_seed(90)
    np.random.seed(90)
    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    want_sha = torch.load(D_CKPT, map_location="cpu", weights_only=False)["sample_stream_sha256"]
    m = WorldModel(tr.norm)
    m.load_state_dict(cp["model"], strict=True)
    m = m.cuda()
    del cp

    nn.init.zeros_(m.delta_head.weight)
    nn.init.zeros_(m.delta_head.bias)

    backbone_params, head_params, trainable_names = [], [], []
    for n, p in m.named_parameters():
        ok = any(n == pref or n.startswith(pref) for pref in PURE_VIS_PREFIXES)
        p.requires_grad_(ok)
        if ok:
            trainable_names.append(n)
            if n.startswith(("delta_head.", "grip_head.")):
                head_params.append(p)
            else:
                backbone_params.append(p)
    frozen_names = tuple(n for n, _ in m.named_parameters() if n not in trainable_names)
    assert "proprio_proj.0.weight" in frozen_names and "state_proj.weight" in frozen_names
    fs = tensor_sha(m, frozen_names)

    STEPS = 4000
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 1e-4}, {"params": head_params, "lr": 3e-4}],
        weight_decay=1e-3,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    B_oof_t = torch.from_numpy(B_oof)
    sampler = MultiSourceSampler(tr.rows, seed=90, weights={"piper": 1.0}, caps={}, demo3_cap=0.45)
    stream = hashlib.sha256()

    early_losses, late_losses = [], []
    for step in range(1, STEPS + 1):
        seed = seed_for(90, "E110F", step)
        torch.manual_seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        ids = sampler.sample(32)
        hs = sample_horizons(22000 + step, 90)
        stream.update(np.asarray(ids, dtype="<i8").tobytes())
        stream.update(np.asarray(hs, dtype="<i8").tobytes())
        b = strip_proprio(move(tr.batch(ids, hs), "cuda"))

        opt.zero_grad(set_to_none=True)
        m.train()
        bsz = len(b["obs"])
        tz = torch.zeros(bsz, device="cuda")
        out = m(b, torch.zeros_like(b["action"]), None, tz, tz)
        y_res = target_delta(b) - B_oof_t[torch.as_tensor(np.asarray(ids))].cuda().to(out["delta"].dtype) / b["action_std"][:, :, J_cuda]
        delta_l1 = (out["delta"] - y_res).abs().mean()
        grip_bce = F.binary_cross_entropy_with_logits(out["grip_logits"], b["grip_label"])
        loss = delta_l1 + grip_bce
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone_params + head_params, 1.0)
        opt.step()
        sch.step()

        if step <= 100:
            early_losses.append(float(delta_l1.detach()))
        if step > STEPS - 200:
            late_losses.append(float(delta_l1.detach()))
        if step == 1 or step % 500 == 0:
            print(f"  WAM_PV step {step}/{STEPS} res_l1={float(delta_l1):.4f} grip={float(grip_bce):.4f}", flush=True)

    assert tensor_sha(m, frozen_names) == fs
    assert stream.hexdigest() == want_sha, "Stream differs from D!"
    wam_pv_ckpt = OUT / "wam_pv_ckpt_04000.pt"
    torch.save(dict(model={k: v.detach().cpu() for k, v in m.state_dict().items()}, early_res_l1=float(np.mean(early_losses)), late_res_l1=float(np.mean(late_losses))), wam_pv_ckpt)
    print(f"  WAM_PV train residual L1: first100={np.mean(early_losses):.4f} -> last200={np.mean(late_losses):.4f}", flush=True)

    # Collect dev batches & evaluate WAM_PV (and swapped vision)
    Bt = [float_batch(dev.batch([e["index"] for e in entries[s : s + 4]], (10,), "short")) for s in range(0, len(entries), 4)]
    cat = lambda k: torch.cat([b[k] for b in Bt])
    OBS, Q0, TR, GL, GC = cat("obs"), cat("q_current"), cat("action"), cat("grip_label"), cat("grip_current")
    N = len(OBS)
    ps_dev = sorted({e["parent"] for e in entries})
    pid = np.array([ps_dev.index(e["parent"]) for e in entries])
    g2 = np.repeat(pid, 2)
    q0_j = Q0[:, JI].numpy()
    near = np.array([(lambda o: o[np.argmin(np.abs(q0_j[o] - q0_j[i]).sum(1))])(np.where(pid != pid[i])[0]) for i in range(N)])

    def infer_wam_pv(model_obj, swap=False):
        model_obj.eval()
        out = []
        k = 0
        with torch.inference_mode():
            for b in Bt:
                n = len(b["obs"])
                idx = np.arange(k, k + n)
                k += n
                b2 = dict(b)
                if swap:
                    b2["obs"] = OBS[near[idx]]
                b2 = strip_proprio(move(b2, "cuda"))
                out.append(infer_d(model_obj, b2, 4, 90).float().cpu())
        x = torch.cat(out)
        # infer_d sets absolute[:,:,JI] = q_current + heads['delta']*action_std; add blind_fold5 displacement!
        x[:, :, JI] = x[:, :, JI] + fold5.to(x.dtype)
        return x

    P = {}
    P["WAM_PV"] = infer_wam_pv(m, swap=False)
    P["WAM_PV_swap"] = infer_wam_pv(m, swap=True)
    del m
    torch.cuda.empty_cache()

    # Also load E106C for the 0.75*blind_fold5 + 0.25*E106C reference check
    cp0 = torch.load(INIT, map_location="cpu", weights_only=False)
    m0 = WorldModel(tr.norm)
    m0.load_state_dict(cp0["model"], strict=True)
    m0 = m0.cuda().eval()
    del cp0
    out0 = []
    with torch.inference_mode():
        for b in Bt:
            out0.append(infer_d(m0, move(b, "cuda"), 4, 90).float().cpu())
    P["E106C"] = torch.cat(out0)
    del m0
    torch.cuda.empty_cache()

    # ==========================================================================
    # 2. SpatialPV 5-fold Parent-OOF Pure-Vision Residual + OOF Shrinkage Curve α_t
    # ==========================================================================
    print(f"[{brt()}] Loading 34,777 train visual grids via mmap...", flush=True)
    V_tr = np.empty((len(tr.rows), 24, 2304), dtype="f2")
    Y_res = np.empty((len(tr.rows), 50, 12), dtype="f4")
    P_tr = []
    for i, r in enumerate(tr.rows):
        cid = r["canonical_id"]
        obs_i = np.stack([tr.base.features(cid, k, np.array(r["obs_rows"])[:, k]) for k in range(3)], axis=1)
        V_tr[i] = extract_vis_cells(obs_i, np.asarray(r["history_valid"], dtype=bool))
        z, q, _, _ = tr.states.get(tr, cid)
        a = r["anchor"]
        Y_res[i] = (q[a + 1 : a + 51, JI] - q[a, JI]) - B_oof[i]
        P_tr.append(r["parent_component"])
    P_tr = np.asarray(P_tr)

    V_dev = np.stack([
        extract_vis_cells(OBS[i].numpy(), cat("history_valid")[i].numpy())
        for i in range(N)
    ])
    V_dev_swap = V_dev[near]

    # Exact same 5 parent folds as E127 blind.py!
    ps_tr = sorted(set(P_tr))
    order = np.random.default_rng(90).permutation(len(ps_tr))
    fold_map = {ps_tr[j]: int(k % 5) for k, j in enumerate(order)}
    F_idx = np.array([fold_map[p] for p in P_tr])

    # Standardize per feature dim across all cells using global train stats (slight approx across folds? No, compute per fold cleanly!)
    r_std = Y_res.std((0, 1), keepdims=True) + 1e-6  # (1, 1, 12)
    oof_vis = np.zeros_like(Y_res)
    dev_vis_folds = []
    dev_vis_swap_folds = []
    train_fold_l1_before, train_fold_l1_after = [], []

    print(f"[{brt()}] Training 5-fold SpatialPureVisNet (zero proprio, Dropout=0.5, wd=1e-2)...", flush=True)
    V_gpu = torch.from_numpy(V_tr).cuda()  # ~3.8 GiB fp16 in VRAM!
    Yr_gpu = torch.from_numpy(Y_res).cuda()
    Vd_gpu = torch.from_numpy(V_dev).cuda()
    Vds_gpu = torch.from_numpy(V_dev_swap).cuda()

    for k in range(5):
        tri = torch.from_numpy(np.where(F_idx != k)[0]).cuda()
        tei = torch.from_numpy(np.where(F_idx == k)[0]).cuda()
        # Subsample 4096 train rows to compute clean fold mean/std in fp32
        sub_s = tri[:: max(1, len(tri) // 4096)]
        vs = V_gpu[sub_s].float()
        v_mean = vs.mean((0, 1), keepdim=True)
        v_std = vs.std((0, 1), keepdim=True) + 1e-5
        ys_std = Yr_gpu[tri].std(0, keepdim=True) + 1e-6  # (1, 50, 12)

        torch.manual_seed(200 + k)
        net = SpatialPureVisNet().cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=2e-2)
        S_fold = 3000
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, S_fold)
        net.train()
        for st in range(1, S_fold + 1):
            b_ix = tri[torch.randint(len(tri), (512,), device="cuda")]
            xb = (V_gpu[b_ix].float() - v_mean) / v_std
            yb = Yr_gpu[b_ix] / ys_std
            pred_b = net(xb)
            loss = F.l1_loss(pred_b, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()

        net.eval()
        with torch.no_grad():
            # Train-fold fit diagnostic (on 4096 train samples)
            tr_pred = net((V_gpu[sub_s].float() - v_mean) / v_std) * ys_std
            tr_b = float(Yr_gpu[sub_s].abs().mean().item())
            tr_a = float((Yr_gpu[sub_s] - tr_pred).abs().mean().item())
            train_fold_l1_before.append(tr_b)
            train_fold_l1_after.append(tr_a)

            # Held-out parent OOF prediction
            te_preds = []
            for s0 in range(0, len(tei), 512):
                sl = tei[s0 : s0 + 512]
                te_preds.append((net((V_gpu[sl].float() - v_mean) / v_std) * ys_std).cpu().numpy())
            oof_k = np.concatenate(te_preds, axis=0)
            oof_vis[tei.cpu().numpy()] = oof_k

            dv = (net((Vd_gpu.float() - v_mean) / v_std) * ys_std).cpu().numpy()
            dvs = (net((Vds_gpu.float() - v_mean) / v_std) * ys_std).cpu().numpy()
            dev_vis_folds.append(dv)
            dev_vis_swap_folds.append(dvs)
            print(f"  fold {k}: train_res_L1 {tr_b:.5f}->{tr_a:.5f} | heldout_oof_L1 {np.abs(Y_res[tei.cpu().numpy()]).mean():.5f}->{np.abs(Y_res[tei.cpu().numpy()] - oof_k).mean():.5f}", flush=True)

    del V_gpu, Yr_gpu
    torch.cuda.empty_cache()

    # Fit per-step shrinkage α_t ∈ [0, 1] STRICTLY on held-out train-parent OOF!
    alphas = np.zeros(50, dtype="f4")
    oof_corrs = np.zeros(50, dtype="f4")
    grid = np.linspace(0.0, 1.0, 51)
    for t_step in range(50):
        yt = Y_res[:, t_step, :].ravel()
        pt = oof_vis[:, t_step, :].ravel()
        oof_corrs[t_step] = float(np.corrcoef(yt, pt)[0, 1]) if pt.std() > 1e-8 else 0.0
        # Subsample 200k scalars if needed? 34777*12 = 417k scalars, fast!
        best_a, best_l = 0.0, float(np.abs(yt).mean())
        for a_val in grid[1:]:
            val = float(np.abs(yt - a_val * pt).mean())
            if val < best_l:
                best_l = val
                best_a = float(a_val)
        alphas[t_step] = best_a

    print("  Train-OOF correlation with blind residual (t=1,8,25,50):", [round(float(oof_corrs[i]), 4) for i in (0, 7, 24, 49)], flush=True)
    print("  Train-OOF optimal shrinkage alpha_t (t=1,8,25,50):", [round(float(alphas[i]), 3) for i in (0, 7, 24, 49)], flush=True)
    print("  Train-OOF mean alpha (1..8 / 9..25 / 26..50):", round(float(alphas[:8].mean()), 3), round(float(alphas[8:25].mean()), 3), round(float(alphas[25:].mean()), 3), flush=True)

    dev_vis_mean = np.mean(dev_vis_folds, axis=0)  # (211, 50, 12)
    dev_vis_swap_mean = np.mean(dev_vis_swap_folds, axis=0)

    def abs_from_delta(d_t):
        x = Q0[:, None].expand(-1, 50, -1).clone()
        x[:, :, JI] = Q0[:, None, JI] + torch.as_tensor(d_t, dtype=x.dtype)
        return x

    P["blind_fold5"] = abs_from_delta(fold5)
    P["blind_full3"] = abs_from_delta(full3)
    P["SpatialPV_raw"] = abs_from_delta(fold5.numpy() + dev_vis_mean)
    P["SpatialPV_raw_swap"] = abs_from_delta(fold5.numpy() + dev_vis_swap_mean)
    P["SpatialPV_shrunk"] = abs_from_delta(fold5.numpy() + dev_vis_mean * alphas[None, :, None])
    # Also apply same OOF-learned horizon curve to WAM_PV residual to see if WAM_PV had good late-step signal!
    wam_res = P["WAM_PV"][:, :, JI] - P["blind_fold5"][:, :, JI]
    x_ws = P["WAM_PV"].clone()
    x_ws[:, :, JI] = P["blind_fold5"][:, :, JI] + wam_res * torch.from_numpy(alphas[None, :, None])
    P["WAM_PV_shrunk"] = x_ws
    # Reference blend 0.75*blind_fold5 + 0.25*E106C
    x_bl = P["E106C"].clone()
    x_bl[:, :, JI] = 0.75 * P["blind_fold5"][:, :, JI] + 0.25 * P["E106C"][:, :, JI]
    P["blend25_E106C"] = x_bl

    # ==========================================================================
    # 3. Evaluate All Arms on Locked 211 Dev Windows (Exact E127 Bootstrap)
    # ==========================================================================
    def ee_fn(a):
        with torch.no_grad():
            p = pose_torch(a.cuda().double()).cpu().numpy()
        return p[..., [0, 1, 2, 9, 10, 11]].reshape(*a.shape[:2], 2, 3)

    EE = {k: ee_fn(v) for k, v in P.items()}
    EEt = ee_fn(TR)
    tj = TR[:, :, JI].numpy()
    W = np.random.default_rng(90).integers(len(ps_dev), size=(2000, len(ps_dev)))
    W = np.stack([np.bincount(w, minlength=len(ps_dev)) for w in W]).astype("f8")

    def stat(v, mask=None, g=pid):
        v = np.asarray(v, "f8")
        mask = np.ones(len(v), bool) if mask is None else mask
        s = np.bincount(g[mask], v[mask], len(ps_dev))
        n = np.bincount(g[mask], minlength=len(ps_dev)).astype("f8")
        return [float(s.sum() / max(n.sum(), 1))] + [float(x) for x in np.nanquantile((W @ s) / np.maximum(W @ n, 1), [0.025, 0.975])]

    ts = GL.numpy().astype("i1").transpose(0, 2, 1).reshape(-1, 50)
    cur_g = (GC.numpy() > 0.5).astype("i1").reshape(-1)
    events = lambda to, frm: [(i, int(np.argmax(ts[i] == to))) for i in range(len(ts)) if cur_g[i] == frm and (ts[i] == to).any()]
    EV = {"close": events(0, 1), "open": events(1, 0)}
    evg = {n: pid[np.array([x[0] for x in ev]) // 2] for n, ev in EV.items()}

    M = {}
    for k, v in P.items():
        pj = v[:, :, JI].numpy()
        e = np.linalg.norm(EE[k] - EEt, axis=-1) * 1000.0
        d = dict(l8=np.abs(pj[:, :8] - tj[:, :8]).mean((1, 2)), l50=np.abs(pj - tj).mean((1, 2)), e8=e[:, 7].mean(1), e25=e[:, 24].mean(1), e50=e[:, 49].mean(1))
        for n, ev in EV.items():
            ii = np.array([x[0] for x in ev])
            tt = np.array([x[1] for x in ev])
            d["ev_" + n] = np.linalg.norm(EE[k][ii // 2, tt, ii % 2] - EEt[ii // 2, tt, ii % 2], axis=-1) * 1000.0
        M[k] = d

    keys = ("l8", "l50", "e8", "e25", "e50")
    res = {
        "brt": brt(),
        "train_diagnostics": {
            "wam_pv_train_res_l1_first100": float(np.mean(early_losses)),
            "wam_pv_train_res_l1_last200": float(np.mean(late_losses)),
            "spatial_pv_train_res_l1_before": float(np.mean(train_fold_l1_before)),
            "spatial_pv_train_res_l1_after": float(np.mean(train_fold_l1_after)),
            "oof_heldout_parent_res_l1_before": float(np.abs(Y_res).mean()),
            "oof_heldout_parent_res_l1_raw": float(np.abs(Y_res - oof_vis).mean()),
            "oof_heldout_parent_res_l1_shrunk": float(np.abs(Y_res - oof_vis * alphas[None, :, None]).mean()),
            "oof_corrs_by_step": [float(x) for x in oof_corrs],
            "oof_alphas_by_step": [float(x) for x in alphas],
        },
        "models": {},
        "diffs_vs_blind_fold5": {},
        "diffs_vs_blind_full3": {},
        "vision_swap_costs": {},
        "gripper": {},
    }

    eval_models = ("blind_fold5", "blind_full3", "E106C", "WAM_PV", "WAM_PV_shrunk", "SpatialPV_raw", "SpatialPV_shrunk", "blend25_E106C")
    print("\n=== E128-B Dev Results (211 windows / 27 parents) ===")
    for k in eval_models:
        res["models"][k] = {n: stat(M[k][n]) for n in keys}
        for n in EV:
            res["models"][k]["ev_" + n] = stat(M[k]["ev_" + n], None, evg[n])
        r = res["models"][k]
        print(
            f"  {k:18s}: l8={r['l8'][0]:.5f} | l50={r['l50'][0]:.5f} | e50={r['e50'][0]:.2f}mm | "
            f"ev_close={r['ev_close'][0]:.2f}mm | ev_open={r['ev_open'][0]:.2f}mm",
            flush=True,
        )

    print("\n=== Paired Differences vs Same Base (blind_fold5) ===")
    for k in ("WAM_PV", "WAM_PV_shrunk", "SpatialPV_raw", "SpatialPV_shrunk", "blend25_E106C"):
        res["diffs_vs_blind_fold5"][k] = {n: stat(M[k][n] - M["blind_fold5"][n]) for n in keys}
        for n in EV:
            res["diffs_vs_blind_fold5"][k]["ev_" + n] = stat(M[k]["ev_" + n] - M["blind_fold5"]["ev_" + n], None, evg[n])
        res["diffs_vs_blind_full3"][k] = {n: stat(M[k][n] - M["blind_full3"][n]) for n in keys}
        d = res["diffs_vs_blind_fold5"][k]
        print(
            f"  {k:18s} - blind_fold5: Δl8={d['l8'][0]:+.5f} [{d['l8'][1]:+.5f}, {d['l8'][2]:+.5f}] | "
            f"Δl50={d['l50'][0]:+.5f} [{d['l50'][1]:+.5f}, {d['l50'][2]:+.5f}] | "
            f"Δe50={d['e50'][0]:+.2f} [{d['e50'][1]:+.2f}, {d['e50'][2]:+.2f}]mm | "
            f"Δev_open={d['ev_open'][0]:+.2f} [{d['ev_open'][1]:+.2f}, {d['ev_open'][2]:+.2f}]mm",
            flush=True,
        )

    for k, sw in (("WAM_PV", "WAM_PV_swap"), ("SpatialPV_raw", "SpatialPV_raw_swap")):
        res["vision_swap_costs"][k] = {n: stat(M[sw][n] - M[k][n]) for n in keys}
        sc = res["vision_swap_costs"][k]["e50"]
        print(f"  Vision swap cost {k:14s}: +50 EE = {sc[0]:+.2f} [{sc[1]:+.2f}, {sc[2]:+.2f}] mm", flush=True)

    # Pure-vision gripper check on WAM_PV vs E106C
    for k in ("E106C", "WAM_PV"):
        pg = (P[k][:, :, [6, 13]].numpy() > thr).astype("i1").transpose(0, 2, 1).reshape(-1, 50)
        acc = (pg == ts).mean(1)
        hold = (cur_g == 0) & (ts == 0).all(1)
        hit = {n: [float((pg[i, max(t - 10, 0) : min(t + 11, 50)] == (0 if n == "close" else 1)).any()) for i, t in ev] for n, ev in EV.items()}
        res["gripper"][k] = dict(state_acc=stat(acc, None, g2), drop25=stat(pg[:, :25].max(1), hold, g2), close_hit=float(np.mean(hit["close"])), open_hit=float(np.mean(hit["open"])))
        print(f"  Gripper {k:8s}: state_acc={res['gripper'][k]['state_acc'][0]:.4f} | drop25={res['gripper'][k]['drop25'][0]:.4f} | close_hit={res['gripper'][k]['close_hit']:.3f} | open_hit={res['gripper'][k]['open_hit']:.3f}", flush=True)

    res["elapsed_s"] = time.monotonic() - t0
    (OUT / "part_b_pure_vision_residual.json").write_text(json.dumps(res, indent=2) + "\n")
    print("E128-B DONE in %.1fs" % res["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
