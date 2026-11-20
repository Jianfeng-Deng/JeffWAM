# E128-A: Exact waterfall decomposition of why Blind MLP beat WAM (history offsets t-1/t-3 vs loss vs schedule/draws vs ensemble).
import os, sys, json, time, datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

os.environ.update(E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4")
T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for p in ("code_e110f", "code_e106c", "code_e102a"):
    sys.path.insert(0, str(T / p))

from e99w2_model import JOINTS, pose_torch
from e102x_setup import datasets

LOCK = T / "results/e106c_control_20261115_02/PRETRAIN_LOCK.json"
OUT = T / "results/e128_ablation_20261119_01"


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def build_xy(rows_or_entries, state_getter, offsets, is_dev=False):
    JI = np.asarray(JOINTS)
    X, Y = [], []
    for r in rows_or_entries:
        cid = r["cid"] if is_dev else r["canonical_id"]
        a = r["anchor"]
        z, q, pose, good = state_getter.states.get(state_getter, cid)
        feat = np.concatenate([q[a]] + [q[a] - q[max(a - k, 0)] for k in offsets])
        X.append(feat)
        Y.append((q[a + 1 : a + 51] - q[a])[:, JI])
    return np.asarray(X, "f4"), np.asarray(Y, "f4")


def train_mlp(Xn, Yn, seed, steps=6000, batch=1024, lr=1e-3, loss_mode="l1_unweighted", cosine=True):
    torch.manual_seed(seed)
    d_in = Xn.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d_in, 1024), torch.nn.GELU(),
        torch.nn.Linear(1024, 1024), torch.nn.GELU(),
        torch.nn.Linear(1024, 1024), torch.nn.GELU(),
        torch.nn.Linear(1024, 600),
    ).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cosine else None
    wt = torch.tensor([4.0] * 10 + [2.0] * 15 + [1.0] * 25, device="cuda")[None, :, None]

    for _ in range(steps):
        idx = torch.randint(len(Xn), (batch,), device="cuda")
        out = net(Xn[idx]).reshape(-1, 50, 12)
        tgt = Yn[idx]
        if loss_mode == "l1_unweighted":
            loss = F.l1_loss(out, tgt)
        elif loss_mode == "mse_wam_weighted":
            loss = ((out - tgt).square() * wt).sum() / (wt.sum() * 12 * len(idx))
        elif loss_mode == "l1_wam_weighted":
            loss = ((out - tgt).abs() * wt).sum() / (wt.sum() * 12 * len(idx))
        else:
            raise ValueError(loss_mode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sch is not None:
            sch.step()
    return net.eval()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    _, _, tr, dev = datasets()
    entries = json.loads(LOCK.read_text())["entries"]
    assert len(entries) == 211
    ps = sorted({e["parent"] for e in entries})
    pid = np.array([ps.index(e["parent"]) for e in entries])
    JI = np.asarray(JOINTS)

    # Ground-truth dev tensors for EE & joint L1
    Q0 = torch.stack([torch.tensor(dev.states.get(dev, e["cid"])[1][e["anchor"]], dtype=torch.float32) for e in entries])
    TR = torch.stack([torch.tensor(dev.states.get(dev, e["cid"])[1][e["anchor"] + 1 : e["anchor"] + 51], dtype=torch.float32) for e in entries])

    def ee_np(a_t):
        with torch.no_grad():
            p = pose_torch(a_t.cuda().double()).cpu().numpy()
        return p[..., [0, 1, 2, 9, 10, 11]].reshape(*a_t.shape[:2], 2, 3)

    EE_tr = ee_np(TR)
    tj = TR[:, :, JI].numpy()
    W = np.random.default_rng(90).integers(len(ps), size=(2000, len(ps)))
    W = np.stack([np.bincount(w, minlength=len(ps)) for w in W]).astype("f8")

    def stat(v):
        v = np.asarray(v, "f8")
        s = np.bincount(pid, v, len(ps))
        n = np.bincount(pid, minlength=len(ps)).astype("f8")
        return [float(s.sum() / max(n.sum(), 1))] + [float(x) for x in np.nanquantile((W @ s) / np.maximum(W @ n, 1), [0.025, 0.975])]

    # Precompute datasets for 3 offset families: WAM (6,12,18,24), E127 (3,6,12,24), Dense (1,3,6,12,24)
    fam_data = {}
    for name, offs in (("off_6_12_18_24", (6, 12, 18, 24)), ("off_3_6_12_24", (3, 6, 12, 24)), ("off_1_3_6_12_24", (1, 3, 6, 12, 24))):
        X, Y = build_xy(tr.rows, tr, offs, is_dev=False)
        Xd, _ = build_xy(entries, dev, offs, is_dev=True)
        mx, sx = X.mean(0), X.std(0) + 1e-6
        my, sy = Y.mean(0, keepdims=True), Y.std(0, keepdims=True) + 1e-6
        fam_data[name] = dict(
            Xn=torch.from_numpy((X - mx) / sx).cuda(),
            Yn=torch.from_numpy((Y - my) / sy).cuda(),
            Xdn=torch.from_numpy((Xd - mx) / sx).cuda(),
            my=my, sy=sy,
        )

    def eval_pred_delta(delta_np):
        a = Q0[:, None].expand(-1, 50, -1).clone()
        a[:, :, JI] = Q0[:, None, JI] + torch.from_numpy(delta_np).float()
        ee = ee_np(a)
        err_mm = np.linalg.norm(ee - EE_tr, axis=-1) * 1000.0
        pj = a[:, :, JI].numpy()
        return dict(
            l8=np.abs(pj[:, :8] - tj[:, :8]).mean((1, 2)),
            l50=np.abs(pj - tj).mean((1, 2)),
            e8=err_mm[:, 7].mean(1),
            e25=err_mm[:, 24].mean(1),
            e50=err_mm[:, 49].mean(1),
        )

    configs = [
        # (tag, fam, seeds, steps, batch, lr, loss_mode, cosine)
        ("1_wam_sched_6_12_18_24_wmse_1s", "off_6_12_18_24", (90,), 4000, 32, 3e-4, "mse_wam_weighted", False),
        ("2_long_cos_6_12_18_24_wmse_1s", "off_6_12_18_24", (90,), 6000, 1024, 1e-3, "mse_wam_weighted", True),
        ("3_long_cos_6_12_18_24_l1_1s", "off_6_12_18_24", (90,), 6000, 1024, 1e-3, "l1_unweighted", True),
        ("4_add_t3_3_6_12_24_l1_1s", "off_3_6_12_24", (90,), 6000, 1024, 1e-3, "l1_unweighted", True),
        ("5_add_t1_1_3_6_12_24_l1_1s", "off_1_3_6_12_24", (90,), 6000, 1024, 1e-3, "l1_unweighted", True),
        ("6_ens3_6_12_18_24_l1_3s", "off_6_12_18_24", (90, 91, 92), 6000, 1024, 1e-3, "l1_unweighted", True),
        ("7_ens3_3_6_12_24_l1_3s", "off_3_6_12_24", (90, 91, 92), 6000, 1024, 1e-3, "l1_unweighted", True),
        ("8_ens3_1_3_6_12_24_l1_3s", "off_1_3_6_12_24", (90, 91, 92), 6000, 1024, 1e-3, "l1_unweighted", True),
    ]

    per_cfg = {}
    summary = {"brt": brt(), "models": {}, "step_diffs": {}}
    for tag, fam, seeds, steps, batch, lr, lmode, cos in configs:
        fd = fam_data[fam]
        ps_list = []
        for s in seeds:
            net = train_mlp(fd["Xn"], fd["Yn"], s, steps=steps, batch=batch, lr=lr, loss_mode=lmode, cosine=cos)
            with torch.no_grad():
                out = net(fd["Xdn"]).reshape(-1, 50, 12).cpu().numpy() * fd["sy"] + fd["my"]
            ps_list.append(out)
        pred_mean = np.mean(ps_list, axis=0)
        m = eval_pred_delta(pred_mean)
        per_cfg[tag] = m
        summary["models"][tag] = {k: stat(m[k]) for k in ("l8", "l50", "e8", "e25", "e50")}
        r = summary["models"][tag]
        print(f"{tag:34s} | l8={r['l8'][0]:.5f} | l50={r['l50'][0]:.5f} | e8={r['e8'][0]:.2f}mm | e50={r['e50'][0]:.2f}mm", flush=True)

    # Incremental factor attribution pairs
    pairs = [
        ("effect_schedule_and_draws", "2_long_cos_6_12_18_24_wmse_1s", "1_wam_sched_6_12_18_24_wmse_1s"),
        ("effect_unweighted_L1_vs_wMSE", "3_long_cos_6_12_18_24_l1_1s", "2_long_cos_6_12_18_24_wmse_1s"),
        ("effect_add_t3_velocity_1s", "4_add_t3_3_6_12_24_l1_1s", "3_long_cos_6_12_18_24_l1_1s"),
        ("effect_add_t1_velocity_1s", "5_add_t1_1_3_6_12_24_l1_1s", "4_add_t3_3_6_12_24_l1_1s"),
        ("effect_3seed_ensemble_on_3_6_12_24", "7_ens3_3_6_12_24_l1_3s", "4_add_t3_3_6_12_24_l1_1s"),
        ("effect_t3_within_3seed_ensemble", "7_ens3_3_6_12_24_l1_3s", "6_ens3_6_12_18_24_l1_3s"),
        ("effect_t1_within_3seed_ensemble", "8_ens3_1_3_6_12_24_l1_3s", "7_ens3_3_6_12_24_l1_3s"),
    ]
    print("\n=== Factor Attribution (Paired 27-Parent Bootstrap) ===")
    for pname, a, b in pairs:
        summary["step_diffs"][pname] = {k: stat(per_cfg[a][k] - per_cfg[b][k]) for k in ("l8", "l50", "e8", "e50")}
        d8 = summary["step_diffs"][pname]["l8"]
        d50 = summary["step_diffs"][pname]["l50"]
        print(f"  {pname:36s}: Δl8={d8[0]:+.5f} [{d8[1]:+.5f}, {d8[2]:+.5f}] | Δl50={d50[0]:+.5f} [{d50[1]:+.5f}, {d50[2]:+.5f}]", flush=True)

    summary["elapsed_s"] = time.monotonic() - t0
    (OUT / "part_a_blind_decomposition.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("E128-A DONE in %.1fs" % summary["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
