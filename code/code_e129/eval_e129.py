# E129 comprehensive evaluator & Markdown report generator (211 windows / 27 parents).
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn.functional as F

from e129_common import (
    T, INIT, LOCK_PATH, CTRL, brt,
    UpgradedWAM, build_or_load_extra_stats, attach_extra_proprio,
)
from e103d_model import WorldModel as OrigModel
from e99w2_model import JOINTS, pose_torch, trajectory
from e102x_setup import datasets
from e102a_contract import contract, onset_labels
from e110f_core import infer_d
from e110f_eval import float_batch, transitions, match
from e86.data import move

BL = T / "results/e127r_blind_20261118_01"
OUT = T / "results/e129_eval_20261119_01"
REPORT_MD = T / "notes/e129_upgraded_end2end_wam_20261119.md"


def fit_blind_baselines(tr, dev, entries):
    # Reproduce exact single-seed & 3-seed blind baselines for (3,6,12,24) and (1,3,6,12,24)
    JI = np.asarray(JOINTS)
    out = {}
    for name, offs in (("t3", (3, 6, 12, 24)), ("t1", (1, 3, 6, 12, 24))):
        X, Y = [], []
        for r in tr.rows:
            z, q, _, _ = tr.states.get(tr, r["canonical_id"])
            a = r["anchor"]
            X.append(np.concatenate([q[a]] + [q[a] - q[max(a - k, 0)] for k in offs]))
            Y.append((q[a + 1 : a + 51] - q[a])[:, JI])
        X, Y = np.asarray(X, "f4"), np.asarray(Y, "f4")
        Xd = np.asarray([
            np.concatenate([dev.states.get(dev, e["cid"])[1][e["anchor"]]] + [
                dev.states.get(dev, e["cid"])[1][e["anchor"]] - dev.states.get(dev, e["cid"])[1][max(e["anchor"] - k, 0)]
                for k in offs
            ])
            for e in entries
        ], "f4")
        mx, sx = X.mean(0), X.std(0) + 1e-6
        my, sy = Y.mean(0, keepdims=True), Y.std(0, keepdims=True) + 1e-6
        Xn = torch.from_numpy((X - mx) / sx).cuda()
        Yn = torch.from_numpy((Y - my) / sy).cuda()
        Xdn = torch.from_numpy((Xd - mx) / sx).cuda()

        preds = []
        for seed in (90, 91, 92):
            torch.manual_seed(seed)
            f = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], 1024), torch.nn.GELU(),
                torch.nn.Linear(1024, 1024), torch.nn.GELU(),
                torch.nn.Linear(1024, 1024), torch.nn.GELU(),
                torch.nn.Linear(1024, 600),
            ).cuda()
            opt = torch.optim.AdamW(f.parameters(), 1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 6000)
            for _ in range(6000):
                ix = torch.randint(len(Xn), (1024,), device="cuda")
                loss = F.l1_loss(f(Xn[ix]).reshape(-1, 50, 12), Yn[ix])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()
            f.eval()
            with torch.no_grad():
                preds.append(f(Xdn).reshape(-1, 50, 12).cpu().numpy() * sy + my)
        out[f"blind_{name}_s90"] = preds[0]
        out[f"blind_{name}_ens3"] = np.mean(preds, axis=0)
    return out


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

    def load_wam(kind, path):
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if kind == "orig":
            m = OrigModel(tr.norm)
            m.load_state_dict(cp["model"], strict=True)
        else:
            m = UpgradedWAM(tr.norm, variant=kind).widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        return m.cuda().eval()

    def infer_wam_actions(m, is_upgraded=False, swap=False):
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
                if is_upgraded:
                    b2 = attach_extra_proprio(b2, np.stack([norm_dev[e["index"]] for e in es]))
                out.append(infer_d(m, b2, 4, 90).float().cpu())
        return torch.cat(out)

    P = {}
    # Reference WAMs
    m_e106 = load_wam("orig", INIT)
    P["E106C"] = infer_wam_actions(m_e106, is_upgraded=False)
    del m_e106

    j_clean_path = T / "results/e125_j_clean_20261118_01/ckpt_04000.pt"
    if j_clean_path.exists():
        m_jc = load_wam("orig", j_clean_path)
        P["J_clean"] = infer_wam_actions(m_jc, is_upgraded=False)
        del m_jc

    # E129 Upgraded WAMs
    wam_defs = [
        ("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt"),
        ("WAM_v2_s90", "v2", T / "results/e129_wam_v2_s90_20261119_01/ckpt_04000.pt"),
        ("WAM_v2_s91", "v2", T / "results/e129_wam_v2_s91_20261119_01/ckpt_04000.pt"),
        ("WAM_v2_s92", "v2", T / "results/e129_wam_v2_s92_20261119_01/ckpt_04000.pt"),
    ]
    v2_seeds = []
    v2_swap_seeds = []
    for name, var, path in wam_defs:
        if path.exists():
            mw = load_wam(var, path)
            P[name] = infer_wam_actions(mw, is_upgraded=True, swap=False)
            P[name + "_swap"] = infer_wam_actions(mw, is_upgraded=True, swap=True)
            if name.startswith("WAM_v2_s"):
                v2_seeds.append(P[name])
                v2_swap_seeds.append(P[name + "_swap"])
            del mw
            torch.cuda.empty_cache()

    if len(v2_seeds) == 3:
        ens = v2_seeds[0].clone()
        ens[:, :, JI] = torch.stack([x[:, :, JI] for x in v2_seeds]).mean(0)
        # Majority vote on binary grippers across 3 seeds
        g_votes = sum((x[:, :, [6, 13]] > torch.tensor(thr)).long() for x in v2_seeds) >= 2
        ens[:, :, [6, 13]] = torch.where(g_votes, ens.new_tensor(c_dict["open_m"]), ens.new_tensor(c_dict["closed_m"]))
        P["WAM_v2_ens3"] = ens

        ens_sw = v2_swap_seeds[0].clone()
        ens_sw[:, :, JI] = torch.stack([x[:, :, JI] for x in v2_swap_seeds]).mean(0)
        P["WAM_v2_ens3_swap"] = ens_sw

    # Blind baselines (single-seed & 3-seed for both t-3 and t-1)
    blind_deltas = fit_blind_baselines(tr, dev, entries)
    for bname, d_arr in blind_deltas.items():
        P[bname] = abs_from_delta(d_arr, P["E106C"])
    P["copy"] = Q0[:, None].expand(-1, 50, -1).clone()

    # ==========================================================================
    # Action + EE + Event + Gripper Metrics (Exact E127 Bootstrap)
    # ==========================================================================
    def ee_fn(a):
        with torch.no_grad():
            p = pose_torch(a.cuda().double()).cpu().numpy()
        return p[..., [0, 1, 2, 9, 10, 11]].reshape(*a.shape[:2], 2, 3)

    EE = {k: ee_fn(v) for k, v in P.items()}
    EEt = ee_fn(TR)
    tj = TR[:, :, JI].numpy()
    W = np.random.default_rng(90).integers(len(ps), size=(2000, len(ps)))
    W = np.stack([np.bincount(w, minlength=len(ps)) for w in W]).astype("f8")

    def stat(v, mask=None, g=pid):
        v = np.asarray(v, "f8")
        mask = np.ones(len(v), bool) if mask is None else mask
        s = np.bincount(g[mask], v[mask], len(ps))
        n = np.bincount(g[mask], minlength=len(ps)).astype("f8")
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
        d = dict(
            l8=np.abs(pj[:, :8] - tj[:, :8]).mean((1, 2)),
            l50=np.abs(pj - tj).mean((1, 2)),
            e8=e[:, 7].mean(1),
            e25=e[:, 24].mean(1),
            e50=e[:, 49].mean(1),
            acc_e=(np.diff(pj, 2, axis=1) ** 2).mean((1, 2)),
        )
        for n, ev in EV.items():
            ii = np.array([x[0] for x in ev])
            tt = np.array([x[1] for x in ev])
            d["ev_" + n] = np.linalg.norm(EE[k][ii // 2, tt, ii % 2] - EEt[ii // 2, tt, ii % 2], axis=-1) * 1000.0
        M[k] = d

    main_models = [k for k in (
        "E106C", "J_clean",
        "blind_t3_s90", "blind_t1_s90",
        "WAM_v1_s90", "WAM_v2_s90",
        "blind_t3_ens3", "blind_t1_ens3",
        "WAM_v2_ens3",
    ) if k in M]

    act_res = {"models": {}, "parent_equal": {}, "diffs": {}, "vision_swap_cost": {}, "gripper": {}}
    keys = ("l8", "l50", "e8", "e25", "e50", "acc_e")
    W_pe = np.random.default_rng(90).integers(0, len(ps), size=(2000, len(ps)))
    for k in main_models:
        act_res["models"][k] = {n: stat(M[k][n]) for n in keys}
        for n in EV:
            act_res["models"][k]["ev_" + n] = stat(M[k]["ev_" + n], None, evg[n])
        # Also record exact E110-F / E125 parent-equal l8 / l50 for seamless historical comparison
        pe_rec = {}
        for n in ("l8", "l50"):
            p_means = np.array([M[k][n][pid == gi].mean() for gi in range(len(ps))])
            bs = p_means[W_pe].mean(1)
            pe_rec[n] = [float(p_means.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        act_res["parent_equal"][k] = pe_rec

    # Gripper stats on WAM models
    onset_cache = {}
    for k in [x for x in ("E106C", "J_clean", "WAM_v1_s90", "WAM_v2_s90", "WAM_v2_ens3") if x in P]:
        pg = (P[k][:, :, [6, 13]].numpy() > thr).astype("i1").transpose(0, 2, 1).reshape(-1, 50)
        acc = (pg == ts).mean(1)
        hold = (cur_g == 0) & (ts == 0).all(1)
        hit = {n: [float((pg[i, max(t - 10, 0) : min(t + 11, 50)] == (0 if n == "close" else 1)).any()) for i, t in ev] for n, ev in EV.items()}
        # Also parent-level F1@±10
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
        f1_ci = [float(f1_p.mean())] + [float(x) for x in np.quantile((W @ f1_p) / np.maximum(W.sum(1), 1), [0.025, 0.975])]
        act_res["gripper"][k] = dict(
            state_acc=stat(acc, None, g2),
            drop25=stat(pg[:, :25].max(1), hold, g2),
            close_hit=float(np.mean(hit["close"])),
            open_hit=float(np.mean(hit["open"])),
            f1_10=f1_ci,
        )

    # Paired comparisons
    comp_pairs = [
        ("WAM_v1_s90", "E106C"),
        ("WAM_v1_s90", "J_clean"),
        ("WAM_v1_s90", "blind_t3_s90"),
        ("WAM_v1_s90", "blind_t1_s90"),
        ("WAM_v2_s90", "WAM_v1_s90"),
        ("WAM_v2_s90", "blind_t3_s90"),
        ("WAM_v2_s90", "blind_t1_s90"),
        ("WAM_v2_ens3", "blind_t3_ens3"),
        ("WAM_v2_ens3", "blind_t1_ens3"),
        ("WAM_v2_ens3", "E106C"),
    ]
    for a, b in comp_pairs:
        if a in M and b in M:
            act_res["diffs"][f"{a}-{b}"] = {n: stat(M[a][n] - M[b][n]) for n in ("l8", "l50", "e8", "e25", "e50")}
            for n in EV:
                act_res["diffs"][f"{a}-{b}"]["ev_" + n] = stat(M[a]["ev_" + n] - M[b]["ev_" + n], None, evg[n])

    for k in ("WAM_v1_s90", "WAM_v2_s90", "WAM_v2_ens3"):
        if k in M and (k + "_swap") in M:
            act_res["vision_swap_cost"][k] = {n: stat(M[k + "_swap"][n] - M[k][n]) for n in ("l8", "l50", "e8", "e25", "e50")}

    # ==========================================================================
    # World Head Evaluation: gt, no_traj, and Closed-Loop N1 Retention (+25, +50, +100)
    # ==========================================================================
    w_defs = [("E106C", "orig", INIT, "E106C")]
    if j_clean_path.exists():
        w_defs.append(("J_clean", "orig", j_clean_path, "J_clean"))
    if (T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt").exists():
        w_defs.append(("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt", "WAM_v1_s90"))
    if (T / "results/e129_wam_v2_s90_20261119_01/ckpt_04000.pt").exists():
        w_defs.append(("WAM_v2_s90", "v2", T / "results/e129_wam_v2_s90_20261119_01/ckpt_04000.pt", "WAM_v2_s90"))
        if "WAM_v2_ens3" in P:
            w_defs.append(("WAM_v2_s90+ens3A", "v2", T / "results/e129_wam_v2_s90_20261119_01/ckpt_04000.pt", "WAM_v2_ens3"))

    world_res = {str(h): {} for h in (25, 50, 100)}
    for w_label, w_kind, w_path, act_key in w_defs:
        wm = load_wam(w_kind, w_path)
        for h in (25, 50, 100):
            Eh = [e for e in entries if h in e["horizons"]]
            psh = sorted({e["parent"] for e in Eh})
            W_h = np.random.default_rng(90).integers(len(psh), size=(2000, len(psh)))
            by_p = {p: {"cp": [], "gt": [], "no": [], "n1": []} for p in psh}
            with torch.inference_mode():
                for s in range(0, len(Eh), 4):
                    es = Eh[s : s + 4]
                    bh = move(float_batch(dev.batch([e["index"] for e in es], (h,), "short")), "cuda")
                    Y = bh["world"].double()
                    cp_m = Y.square().mean((1, 2)).cpu().numpy()
                    gt_m = (wm.forward_mean(bh)["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()
                    no_m = (wm.forward_mean(dict(bh, robot=torch.zeros_like(bh["robot"]), robot_valid=torch.zeros_like(bh["robot_valid"])))["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()
                    # Map dev entry index -> row in P[act_key]
                    e_pos = [entries.index(e) for e in es]
                    a_b = P[act_key][e_pos].to("cuda")
                    n1_m = (wm.forward_mean(dict(bh, robot=trajectory(bh["context"], a_b)))["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()
                    for k_i, e in enumerate(es):
                        rec = by_p[e["parent"]]
                        rec["cp"].append(float(cp_m[k_i]))
                        rec["gt"].append(float(gt_m[k_i]))
                        rec["no"].append(float(no_m[k_i]))
                        rec["n1"].append(float(n1_m[k_i]))
            sk = lambda key: np.array([100.0 * (1.0 - np.mean(by_p[p][key]) / max(np.mean(by_p[p]["cp"]), 1e-12)) for p in psh])
            s_gt, s_no, s_n1 = sk("gt"), sk("no"), sk("n1")
            d_n1_no = s_n1 - s_no
            bs_d = d_n1_no[W_h].mean(1)
            denom = float((s_gt - s_no).mean())
            world_res[str(h)][w_label] = dict(
                gt_skill=float(s_gt.mean()),
                no_skill=float(s_no.mean()),
                n1_skill=float(s_n1.mean()),
                n1_minus_no_pp=[float(d_n1_no.mean()), float(np.quantile(bs_d, 0.025)), float(np.quantile(bs_d, 0.975))],
                retention_pct=float(100.0 * d_n1_no.mean() / denom) if abs(denom) > 1e-9 else 0.0,
            )
        del wm
        torch.cuda.empty_cache()

    full_summary = dict(experiment="E129_upgraded_end2end_wam", brt=brt(), action=act_res, world=world_res)
    print("Saving E129 summary to", OUT / "summary.json", flush=True)
    (OUT / "summary.json").write_text(json.dumps(full_summary, indent=2, ensure_ascii=False) + "\n")
    write_markdown(full_summary)


def fm(arr, d=5):
    return f"{arr[0]:.{d}f} [{arr[1]:.{d}f}, {arr[2]:.{d}f}]"


def fmd(arr, d=5):
    return f"{arr[0]:+.{d}f} [{arr[1]:+.{d}f}, {arr[2]:+.{d}f}]"


def write_markdown(s):
    am = s["action"]["models"]
    ad = s["action"]["diffs"]
    sw = s["action"]["vision_swap_cost"]
    gr = s["action"]["gripper"]
    wr = s["world"]

    lines = [
        "# E129 端到端升级版联合 WAM：补齐 `(t-1, t-3)` 近期速度 + 等权 L1 + 消除暴露偏差的世界头",
        "",
        f"完成时间：{s['brt']}。代码：`code_e129/`；结果：`results/e129_*_20261119_01/`、`results/e129_eval_20261119_01/`。",
        "锁定评估：211 窗 / 27 parent（parent bootstrap 2000，seed 90）。",
        "",
        "## 1. 升级内容",
        "",
        "1. **补齐因果近期速度**：在原 164 维本体特征（`-6, -12, -18, -24`）基础上追加严格因果校验的 **`t-3`（100 ms）与 `t-1`（33 ms）关节+末端位姿+二值夹爪差分**（共 230 维，仅用训练集统计标准化，新增列权重零初始化，第 0 步与 `E106-C` 逐位一致）。",
        "2. **两种结构变体**：",
        "   - `WAM_v1`：230 维经 `proprio_proj` 进入 `state` token，与 3 相机 `2×4` 视觉 token 经 6 层 Transformer 端到端耦合；",
        "   - `WAM_v2`：在 `v1` 基础上增加零初始化的 13.7 万参数秩-256 运动学时间基（`kin_mlp(proprio) * kin_time_basis[50, 256]`），在进入第 0 层前为 50 个动作查询 token 提供逐时刻初始运动学偏置，再做 6 层跨模态视觉注意力。",
        "3. **干净联合损失 + 消除世界头暴露偏差**：动作项改用等权标准化 L1 + 夹爪 BCE + 余弦退火；世界头四模式改为 `25% gt + 25% detached_pred_N1 + 25% no_traj + 25% action_only`（零额外动作前向开销）。",
        "",
        "## 2. 动作与物理末端误差对比（211 窗 / 27 parent）",
        "",
        "| 模型 | 种子 | 前8步 L1 (`l8`) | 50步 L1 (`l50`) | +8步末端 | +25步末端 | +50步末端 | 夹紧37次 | 松爪16次 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    seed_lbl = {
        "E106C": "1s", "J_clean": "1s", "blind_t3_s90": "1s", "blind_t1_s90": "1s",
        "WAM_v1_s90": "1s", "WAM_v2_s90": "1s",
        "blind_t3_ens3": "3s", "blind_t1_ens3": "3s", "WAM_v2_ens3": "3s",
    }
    for k in am:
        r = am[k]
        lines.append(
            f"| `{k}` | {seed_lbl.get(k,'-')} | {r['l8'][0]:.5f} | {r['l50'][0]:.5f} | "
            f"{r['e8'][0]:.2f} mm | {r['e25'][0]:.2f} mm | {r['e50'][0]:.2f} mm | "
            f"{r['ev_close'][0]:.2f} mm | {r['ev_open'][0]:.2f} mm |"
        )

    lines.extend([
        "",
        "### 关键配对差值（负值表示前者误差更低）",
        "",
        "| 配对比较 | `Δl8` | `Δl50` | `Δe25` (mm) | `Δe50` (mm) | `Δ松爪(ev_open)` (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pair, d in ad.items():
        lines.append(
            f"| `{pair}` | {fmd(d['l8'], 5)} | {fmd(d['l50'], 5)} | "
            f"{fmd(d['e25'], 2)} | {fmd(d['e50'], 2)} | {fmd(d['ev_open'], 2)} |"
        )

    lines.extend([
        "",
        "### 视觉贡献（换同构型其他场景图像的末端误差增量）与夹爪指标",
        "",
        "| 模型 | 换图代价 +8/+25/+50 (mm) | 夹爪状态准确率 | 事件 F1@±10 | 闭合命中 | 张开命中 | 夹持误张开(25步) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for k in gr:
        g = gr[k]
        if k in sw:
            s_str = f"+{sw[k]['e8'][0]:.2f} / +{sw[k]['e25'][0]:.2f} / +{sw[k]['e50'][0]:.2f}"
        else:
            s_str = "—"
        lines.append(
            f"| `{k}` | {s_str} | {g['state_acc'][0]:.4f} | {g['f1_10'][0]:.3f} | "
            f"{g['close_hit']*100:.1f}% | {g['open_hit']*100:.1f}% | {g['drop25'][0]*100:.2f}% |"
        )

    lines.extend([
        "",
        "## 3. 世界头表现与闭环 `N1` 保留率（检验 `sync = detached N1` 消除暴露偏差）",
        "",
        "| 模型 | +25 `gt / no / N1` (%) | +25 保留率 | +50 `gt / no / N1` (%) | +50 `N1−no` (pp) | +50 保留率 | +100 保留率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for wk in wr["50"]:
        r25, r50, r100 = wr["25"][wk], wr["50"][wk], wr["100"][wk]
        lines.append(
            f"| `{wk}` | {r25['gt_skill']:.2f} / {r25['no_skill']:.2f} / {r25['n1_skill']:.2f} | {r25['retention_pct']:.1f}% | "
            f"{r50['gt_skill']:.2f} / {r50['no_skill']:.2f} / {r50['n1_skill']:.2f} | "
            f"{fmd(r50['n1_minus_no_pp'], 2)} | {r50['retention_pct']:.1f}% | {r100['retention_pct']:.1f}% |"
        )

    REPORT_MD.write_text("\n".join(lines) + "\n")
    print("Wrote report to", REPORT_MD, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        _, _, tr, dev = datasets()
        entries = json.loads(LOCK_PATH.read_text())["entries"]
        _, norm_dev = build_or_load_extra_stats(tr, dev, entries[:4])
        cp = torch.load(INIT, map_location="cpu", weights_only=False)
        mw = UpgradedWAM(tr.norm, variant="v2")
        mw.load_state_dict(cp["model"], strict=False)
        mw.widen_from_164().cuda().eval()
        b = move(float_batch(dev.batch([e["index"] for e in entries[:4]], (25,), "short")), "cuda")
        b = attach_extra_proprio(b, np.stack([norm_dev[e["index"]] for e in entries[:4]]))
        with torch.inference_mode():
            act = infer_d(mw, b, 4, 90)
            wm_out = mw.forward_mean(b)["mean"]
        assert act.shape == (4, 50, 14) and torch.isfinite(act).all()
        assert torch.isfinite(wm_out).all()
        print("eval_e129 smoke check passed!", flush=True)
    else:
        main()
