# E125 evaluator & report generator: Full 211-window / 27-parent evaluation of E106C, D, U_full, J_legacy, J_clean.
import os, sys, json, time, hashlib, datetime
from pathlib import Path
import numpy as np
import torch

_saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "7")
os.environ.update(
    E102_GROUP="A2",
    ARM="CF",
    PYTHONDONTWRITEBYTECODE="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    OMP_NUM_THREADS="4",
)

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
sys.path[:0] = [str(T / "code_e102s2"), str(T / "code_e86")]
from s2 import truths
os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cvd
for _m in ("s2", "train", "common", "linear"):
    sys.modules.pop(_m, None)

for p in ("code_e110f", "code_e106c", "code_e102a"):
    sys.path.insert(0, str(T / p))

from e99w2_model import WorldModel, JOINTS, trajectory
from e102x_setup import datasets
from e102a_contract import contract, onset_labels
from e110f_core import infer_d
from e110f_eval import float_batch, transitions, match
from e86.data import move

INIT = T / "results/e106c_f2_s90_20261115_02/ckpt_20000.pt"
FEATS = T / "results/e115p1_stage_feats"
OUT = T / "results/e125_eval_20261118_01"
REPORT_MD = T / "notes/e125_fullpool_unfrozen_and_clean_joint_20261118.md"


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def has_boundary(parent, anchor, span=50):
    if parent not in truths:
        return False
    y = truths[parent][anchor : anchor + span + 1]
    y = y[y >= 0]
    return bool(len(y) > 1 and (y != y[0]).any())


def boot_idx(n):
    return np.random.default_rng(90).integers(0, n, size=(2000, n))


def ci95(arr):
    return [float(v) for v in np.quantile(np.asarray(arr), [0.025, 0.975])]


def main():
    assert not OUT.exists(), f"{OUT} already exists"
    OUT.mkdir(parents=False)

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)

    _, _, tr, dev = datasets()
    entries = json.loads((T / "results/e106c_control_20261115_02/PRETRAIN_LOCK.json").read_text())["entries"]
    assert len(entries) == 211 and len({e["parent"] for e in entries}) == 27
    avail_stage = {p.stem for p in FEATS.glob("pc_*.npz")}

    ck_map = {
        "E106C": INIT,
        "D": T / "results/e110f_d_20261116_05/ckpt_04000.pt",
        "U_full": T / "results/e125_u_full_20261118_01/ckpt_04000.pt",
        "J_legacy": T / "results/e125_j_legacy_20261118_01/ckpt_04000.pt",
        "J_clean": T / "results/e125_j_clean_20261118_01/ckpt_04000.pt",
    }
    for k, p in ck_map.items():
        assert p.exists(), f"Missing checkpoint {k}: {p}"

    c = contract()
    ji_np = np.asarray(JOINTS)
    ji_cuda = torch.as_tensor(JOINTS, device="cuda")
    onset = {}

    action_rows = []
    pred_actions = {}  # (arm, index) -> (50, 14) f4

    # 1. Evaluate Action + Gripper + Smoothness for all 5 models across all 211 windows
    for arm, path in ck_map.items():
        cp = torch.load(path, map_location="cpu", weights_only=False)
        m = WorldModel(tr.norm)
        m.load_state_dict(cp["model"], strict=True)
        m = m.cuda().eval()
        del cp

        with torch.inference_mode():
            for s in range(0, len(entries), 4):
                es = entries[s : s + 4]
                ids = [e["index"] for e in es]
                b = move(float_batch(dev.batch(ids, (10,), "short")), "cuda")
                truth = b["action"]
                q = b["q_current"]
                pred = infer_d(m, b, 4, 90)
                for k_i, e in enumerate(es):
                    pr_np = pred[k_i].float().cpu().numpy()
                    tr_np = truth[k_i].float().cpu().numpy()
                    pred_actions[(arm, e["index"])] = pr_np

                    p_j = pr_np[:, ji_np]
                    t_j = tr_np[:, ji_np]
                    q_j = q[k_i, ji_cuda].float().cpu().numpy()

                    l1_8 = float(np.abs(p_j[:8] - t_j[:8]).mean())
                    l1_50 = float(np.abs(p_j[:50] - t_j[:50]).mean())
                    cp8 = float(np.abs(q_j[None] - t_j[:8]).mean())
                    cp50 = float(np.abs(q_j[None] - t_j[:50]).mean())

                    acc_e = float((np.diff(p_j, 2, axis=0) ** 2).mean())
                    jerk_e = float((np.diff(p_j, 3, axis=0) ** 2).mean())
                    tacc_e = float((np.diff(t_j, 2, axis=0) ** 2).mean())
                    tjerk_e = float((np.diff(t_j, 3, axis=0) ** 2).mean())

                    ps_g = (pr_np[:, [6, 13]] > np.asarray(c["threshold_m"])).astype("i1")
                    ts_g = b["grip_label"][k_i].cpu().numpy().astype("i1")
                    cur_g = (b["grip_current"][k_i].cpu().numpy() > 0.5).astype("i1")
                    z, qq, pose, good = dev.states.get(dev, e["cid"])
                    if e["cid"] not in onset:
                        onset[e["cid"]] = onset_labels(qq, z["continuous_segment"], good, c)[0]
                    init_g = onset[e["cid"]][e["anchor"]]
                    ev = {str(t): dict(tp=0, pred=0, true=0, minutes=50.0 / 1800.0) for t in (5, 10, 20)}
                    for side in range(2):
                        pe = transitions(ps_g[:, side], cur_g[side])
                        te = transitions(ts_g[:, side], init_g[side])
                        for t in (5, 10, 20):
                            tp = len(match(pe, te, t))
                            ev[str(t)]["tp"] += tp
                            ev[str(t)]["pred"] += len(pe)
                            ev[str(t)]["true"] += len(te)

                    in_s160 = e["parent"] in avail_stage
                    action_rows.append(
                        dict(
                            arm=arm,
                            parent=e["parent"],
                            index=e["index"],
                            original48=bool(e["original48"]),
                            in_s160=in_s160,
                            bd=(in_s160 and has_boundary(e["parent"], e["anchor"])),
                            l1_8=l1_8,
                            l1_50=l1_50,
                            copy8=cp8,
                            copy50=cp50,
                            state_acc=float((ps_g == ts_g).mean()),
                            events=ev,
                            acc_e=acc_e,
                            jerk_e=jerk_e,
                            tacc_e=tacc_e,
                            tjerk_e=tjerk_e,
                        )
                    )
        del m
        torch.cuda.empty_cache()

    # 2. Evaluate World Heads (E106C, J_legacy, J_clean) across all 211 windows (+10, +25, +50, +100)
    # Pairings tested:
    #   - Own gt_action & no_traj for each world head (E106C, J_legacy, J_clean)
    #   - Action N1 pairings:
    #       E106C_W + E106C_A, E106C_W + D_A, E106C_W + U_full_A,
    #       J_legacy_W + J_legacy_A, J_clean_W + J_clean_A, J_clean_W + U_full_A
    world_models = ("E106C", "J_legacy", "J_clean")
    w_pairings = {
        "E106C": ["E106C", "D", "U_full"],
        "J_legacy": ["J_legacy"],
        "J_clean": ["J_clean", "U_full"],
    }
    world_rows = []
    for w_name in world_models:
        cp = torch.load(ck_map[w_name], map_location="cpu", weights_only=False)
        wm = WorldModel(tr.norm)
        wm.load_state_dict(cp["model"], strict=True)
        wm = wm.cuda().eval()
        del cp

        with torch.inference_mode():
            for h in (10, 25, 50, 100):
                Eh = [e for e in entries if h in e["horizons"]]
                for s in range(0, len(Eh), 4):
                    es = Eh[s : s + 4]
                    ids = [e["index"] for e in es]
                    bh = move(float_batch(dev.batch(ids, (h,), "short")), "cuda")
                    Y = bh["world"].double()
                    cp_mse = Y.square().mean((1, 2)).cpu().numpy()
                    gt_mse = (wm.forward_mean(bh)["mean"].double() - Y).square().mean((1, 2)).cpu().numpy()
                    no_mse = (
                        wm.forward_mean(dict(bh, robot=torch.zeros_like(bh["robot"]), robot_valid=torch.zeros_like(bh["robot_valid"])))["mean"].double()
                        - Y
                    ).square().mean((1, 2)).cpu().numpy()

                    n1_mses = {}
                    for a_name in w_pairings[w_name]:
                        a_batch = torch.from_numpy(np.stack([pred_actions[(a_name, e["index"])] for e in es])).to("cuda")
                        P_n1 = wm.forward_mean(dict(bh, robot=trajectory(bh["context"], a_batch)))["mean"].double()
                        n1_mses[a_name] = (P_n1 - Y).square().mean((1, 2)).cpu().numpy()

                    for k_i, e in enumerate(es):
                        world_rows.append(
                            dict(
                                w_model=w_name,
                                parent=e["parent"],
                                index=e["index"],
                                h=h,
                                copy=float(cp_mse[k_i]),
                                gt=float(gt_mse[k_i]),
                                no=float(no_mse[k_i]),
                                n1={a_name: float(n1_mses[a_name][k_i]) for a_name in w_pairings[w_name]},
                            )
                        )
        del wm
        torch.cuda.empty_cache()

    # 3. Parent-equal Bootstrap Aggregation (exact E110-F protocol on 27 parents + boundary 30w subset)
    arms = list(ck_map.keys())

    def parent_act_dict(arm, subset="all211"):
        rr = [
            r for r in action_rows
            if r["arm"] == arm
            and (
                subset == "all211"
                or (subset == "orig48" and r["original48"])
                or (subset == "bd30" and r["bd"])
            )
        ]
        by_p = {}
        for r in rr:
            by_p.setdefault(r["parent"], []).append(r)
        out = {}
        for p, v in by_p.items():
            tp = sum(x["events"]["10"]["tp"] for x in v)
            pr = sum(x["events"]["10"]["pred"] for x in v)
            tt = sum(x["events"]["10"]["true"] for x in v)
            mins = sum(x["events"]["10"]["minutes"] for x in v)
            out[p] = dict(
                l1_8=float(np.mean([x["l1_8"] for x in v])),
                l1_50=float(np.mean([x["l1_50"] for x in v])),
                copy8=float(np.mean([x["copy8"] for x in v])),
                copy50=float(np.mean([x["copy50"] for x in v])),
                state_acc=float(np.mean([x["state_acc"] for x in v])),
                f1_10=float(2 * tp / max(1, pr + tt)),
                false_sw=float((pr - tp) / max(mins, 1e-9)),
                acc_e=float(np.mean([x["acc_e"] for x in v])),
                jerk_e=float(np.mean([x["jerk_e"] for x in v])),
                tacc_e=float(np.mean([x["tacc_e"] for x in v])),
                tjerk_e=float(np.mean([x["tjerk_e"] for x in v])),
            )
        return out

    act_summary = {}
    for subset in ("all211", "bd30"):
        p_dicts = {a: parent_act_dict(a, subset) for a in arms}
        ps = sorted(p_dicts["E106C"])
        idx = boot_idx(len(ps))
        sub_res = {"parents": len(ps), "arms": {}, "diffs": {}}
        for a in arms:
            pd = p_dicts[a]
            m_out = {}
            for m in ("l1_8", "l1_50", "state_acc", "f1_10", "false_sw", "acc_e", "jerk_e"):
                arr = np.array([pd[p][m] for p in ps])
                m_out[m] = dict(mean=float(arr.mean()), ci95=ci95(arr[idx].mean(1)))
            cp8 = np.array([pd[p]["copy8"] for p in ps])
            cp50 = np.array([pd[p]["copy50"] for p in ps])
            l8 = np.array([pd[p]["l1_8"] for p in ps])
            l50 = np.array([pd[p]["l1_50"] for p in ps])
            m_out["rel_copy_8_pct"] = float(100.0 * (cp8.mean() - l8.mean()) / cp8.mean())
            m_out["rel_copy_50_pct"] = float(100.0 * (cp50.mean() - l50.mean()) / cp50.mean())
            sub_res["arms"][a] = m_out
        sub_res["truth_acc_e"] = float(np.mean([p_dicts["E106C"][p]["tacc_e"] for p in ps]))
        sub_res["truth_jerk_e"] = float(np.mean([p_dicts["E106C"][p]["tjerk_e"] for p in ps]))

        for a, b in (
            ("U_full", "D"),
            ("U_full", "E106C"),
            ("J_clean", "J_legacy"),
            ("J_clean", "E106C"),
            ("U_full", "J_clean"),
            ("J_legacy", "E106C"),
        ):
            pa, pb = p_dicts[a], p_dicts[b]
            d_out = {}
            for m in ("l1_8", "l1_50", "state_acc", "f1_10", "false_sw", "acc_e", "jerk_e"):
                da = np.array([pa[p][m] - pb[p][m] for p in ps])
                d_out[m] = dict(mean=float(da.mean()), ci95=ci95(da[idx].mean(1)))
            # Also relative % L1 diff like E110-F
            for h_k in ("l1_8", "l1_50"):
                va = np.array([pa[p][h_k] for p in ps])
                vb = np.array([pb[p][h_k] for p in ps])
                rel = 100.0 * (va - vb) / vb
                d_out[f"{h_k}_rel_pct"] = dict(mean=float(rel.mean()), ci95=ci95(rel[idx].mean(1)))
            sub_res["diffs"][f"{a}-{b}"] = d_out
        act_summary[subset] = sub_res

    # World summary on 211 windows / 27 parents
    world_summary = {}
    for h in (10, 25, 50, 100):
        wh_e = [r for r in world_rows if r["h"] == h and r["w_model"] == "E106C"]
        psh = sorted({r["parent"] for r in wh_e})
        idx = boot_idx(len(psh))

        def p_w_skill(w_mod, mode_key, a_mod=None):
            wr = [r for r in world_rows if r["h"] == h and r["w_model"] == w_mod]
            out = []
            for p in psh:
                rp = [r for r in wr if r["parent"] == p]
                cp_m = np.mean([r["copy"] for r in rp])
                val_m = np.mean([r[mode_key] if mode_key in ("gt", "no") else r["n1"][a_mod] for r in rp])
                out.append(100.0 * (1.0 - val_m / max(cp_m, 1e-12)))
            return np.array(out)

        h_res = {"parents": len(psh), "own_world": {}, "n1_pairings": {}, "diffs": {}}
        w_vecs = {}
        for wm_name in world_models:
            v_gt = p_w_skill(wm_name, "gt")
            v_no = p_w_skill(wm_name, "no")
            w_vecs[(wm_name, "gt")] = v_gt
            w_vecs[(wm_name, "no")] = v_no
            h_res["own_world"][wm_name] = dict(
                gt_skill=dict(mean=float(v_gt.mean()), ci95=ci95(v_gt[idx].mean(1))),
                no_skill=dict(mean=float(v_no.mean()), ci95=ci95(v_no[idx].mean(1))),
            )
            for a_name in w_pairings[wm_name]:
                v_n1 = p_w_skill(wm_name, "n1", a_name)
                w_vecs[(wm_name, f"n1:{a_name}")] = v_n1
                d_no = v_n1 - v_no
                denom = float((v_gt - v_no).mean())
                h_res["n1_pairings"][f"{wm_name}_W+{a_name}_A"] = dict(
                    n1_skill=dict(mean=float(v_n1.mean()), ci95=ci95(v_n1[idx].mean(1))),
                    vs_own_no_pp=dict(mean=float(d_no.mean()), ci95=ci95(d_no[idx].mean(1))),
                    retention_pct=float(100.0 * d_no.mean() / denom) if abs(denom) > 1e-9 else np.nan,
                )

        # Key paired world comparisons (pp)
        w_comp_list = [
            ("J_clean_gt_vs_J_legacy_gt", w_vecs[("J_clean", "gt")] - w_vecs[("J_legacy", "gt")]),
            ("J_clean_no_vs_J_legacy_no", w_vecs[("J_clean", "no")] - w_vecs[("J_legacy", "no")]),
            ("J_clean_n1_vs_J_legacy_n1", w_vecs[("J_clean", "n1:J_clean")] - w_vecs[("J_legacy", "n1:J_legacy")]),
            ("E106CW_Ufull_vs_D", w_vecs[("E106C", "n1:U_full")] - w_vecs[("E106C", "n1:D")]),
            ("E106CW_Ufull_vs_E106C", w_vecs[("E106C", "n1:U_full")] - w_vecs[("E106C", "n1:E106C")]),
            ("JcleanW_Ufull_vs_E106CW_E106C", w_vecs[("J_clean", "n1:U_full")] - w_vecs[("E106C", "n1:E106C")]),
        ]
        for cname, dv in w_comp_list:
            h_res["diffs"][cname] = dict(mean_pp=float(dv.mean()), ci95_pp=ci95(dv[idx].mean(1)))
        world_summary[str(h)] = h_res

    # Collect training speed ledgers
    train_speed = {}
    for arm in ("U_full", "J_legacy", "J_clean"):
        lp = T / f"results/e125_{arm.lower()}_20261118_01/summary.json"
        if lp.exists():
            d = json.loads(lp.read_text())
            train_speed[arm] = dict(
                elapsed_s=d["elapsed_s"],
                sec_per_step=d["sec_per_step"],
                peak_mib=d["peak_allocated_mib"],
                matched_d_stream=d["matched_d_stream"],
            )

    summary = dict(
        experiment="E125_fullpool_unfrozen_and_clean_joint",
        evaluated_brt=brt(),
        windows=211,
        parents=27,
        boundary_windows=30,
        train_speed=train_speed,
        action_summary=act_summary,
        world_summary=world_summary,
    )

    print("Saving E125 summary; train_speed:", json.dumps(train_speed), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_report(summary)


def fv(d, digits=4):
    return f"{d['mean']:.{digits}f} [{d['ci95'][0]:.{digits}f}, {d['ci95'][1]:.{digits}f}]"


def fd(d, digits=5):
    return f"{d['mean']:+.{digits}f} [{d['ci95'][0]:+.{digits}f}, {d['ci95'][1]:+.{digits}f}]"


def write_report(s):
    a211 = s["action_summary"]["all211"]
    abd = s["action_summary"]["bd30"]
    sp = s["train_speed"]

    lines = [
        "# E125 全量 145 Parent 池：解冻动作主干 (`U_full`) 与去流匹配干净联合训练 (`J_clean` vs `J_legacy`) 对照",
        "",
        f"完成时间：{s['evaluated_brt']}。代码：`code_e125/`；结果：`results/e125_*_20261118_01/`、`results/e125_eval_20261118_01/`。",
        "起点：统一从默认 `E106-C` 第 20000 步检查点及原 AdamW 状态续训 4000 步，P3 样本流 SHA 与 `E110-F` 的 `D` 组 100% 一致（全量 145 parent / 34,777 窗，排除 `episode142`）。",
        "",
        "## 1. 训练耗时与显存对比（4000 步，H20 单卡）",
        "",
        "| 组别 | 含义 | 总耗时 (s) | 每步耗时 (s/step) | 峰值显存 (MiB) | 与 `D` 样本流一致 |",
        "|---|---|---:|---:|---:|---|",
    ]
    desc_map = {
        "U_full": "解冻 6 层主干·纯确定性动作头（世界头冻结保留 E106C）",
        "J_legacy": "原联合目标续训（保留旧流匹配 `action_head` + 高斯噪声轨迹 R）",
        "J_clean": "干净联合目标（去掉流匹配死分支，世界头仅看真实 gt 轨迹或 no_traj）",
    }
    for k in ("U_full", "J_clean", "J_legacy"):
        if k in sp:
            x = sp[k]
            lines.append(f"| `{k}` | {desc_map[k]} | {x['elapsed_s']:.1f} | {x['sec_per_step']:.3f} | {x['peak_mib']:.1f} | `{x['matched_d_stream']}` |")

    lines.extend([
        "",
        "## 2. 动作、夹爪与平滑度（全量 211 窗 / 27 parent，parent bootstrap 2000）",
        "",
        "| 模型 | 前8步 L1 | 50步 L1 | 相对复制改善(8/50) | 夹爪准确率 | 夹爪事件 F1@±10 | 假切换/分 | 加速度能量(211窗 / 切换30窗) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ("E106C", "D", "U_full", "J_legacy", "J_clean"):
        m = a211["arms"][arm]
        mb = abd["arms"][arm]
        lines.append(
            f"| `{arm}` | {fv(m['l1_8'], 4)} | {fv(m['l1_50'], 4)} | {m['rel_copy_8_pct']:.1f}% / {m['rel_copy_50_pct']:.1f}% | "
            f"{fv(m['state_acc'], 3)} | {fv(m['f1_10'], 3)} | {m['false_sw']['mean']:.2f} | "
            f"{m['acc_e']['mean']:.2e} / {mb['acc_e']['mean']:.2e} |"
        )
    lines.append(f"| *真值参照* | — | — | — | — | — | — | *{a211['truth_acc_e']:.2e} / {abd['truth_acc_e']:.2e}* |")

    lines.extend([
        "",
        "### 动作关键配对差值（负的 ΔL1 为变好，正的 ΔF1 为变好）",
        "",
        "| 配对比较 | 全211窗 ΔL1_8 | 全211窗 ΔL1_50 | 切换30窗 ΔL1_50 | 全211窗 ΔF1@±10 | 全211窗 Δ加速度能量 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pair in ("U_full-D", "U_full-E106C", "J_clean-J_legacy", "J_clean-E106C", "U_full-J_clean", "J_legacy-E106C"):
        d = a211["diffs"][pair]
        db = abd["diffs"][pair]
        lines.append(
            f"| `{pair}` | {fd(d['l1_8'], 5)} | {fd(d['l1_50'], 5)} | {fd(db['l1_50'], 5)} | "
            f"{fd(d['f1_10'], 3)} | {d['acc_e']['mean']:+.2e} [{d['acc_e']['ci95'][0]:+.2e}, {d['acc_e']['ci95'][1]:+.2e}] |"
        )

    lines.extend([
        "",
        "## 3. 世界头表现与闭环 `N1` 保留率（211 窗 / 27 parent）",
        "",
        "### 3.1 世界头自身 `gt_action` / `no_traj` skill (%)",
        "",
        "| 世界头 | +25 `gt` / `no` | +50 `gt` / `no` | +100 `gt` / `no` |",
        "|---|---:|---:|---:|",
    ])
    w25, w50, w100 = s["world_summary"]["25"], s["world_summary"]["50"], s["world_summary"]["100"]
    for wm in ("E106C", "J_legacy", "J_clean"):
        lines.append(
            f"| `{wm}` | {w25['own_world'][wm]['gt_skill']['mean']:.2f} / {w25['own_world'][wm]['no_skill']['mean']:.2f} | "
            f"{w50['own_world'][wm]['gt_skill']['mean']:.2f} / {w50['own_world'][wm]['no_skill']['mean']:.2f} | "
            f"{w100['own_world'][wm]['gt_skill']['mean']:.2f} / {w100['own_world'][wm]['no_skill']['mean']:.2f} |"
        )

    lines.extend([
        "",
        "### 3.2 动作预测 `N1` 驱动世界头的 skill (%) 与保留率 `(N1−no)/(gt−no)`",
        "",
        "| 组合 (`世界头 + 动作头`) | +25 `N1−no` (pp) | +25 保留率 | +50 `N1−no` (pp) | +50 保留率 | +100 `N1−no` (pp) | +100 保留率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for pk in ("E106C_W+E106C_A", "E106C_W+D_A", "E106C_W+U_full_A", "J_legacy_W+J_legacy_A", "J_clean_W+J_clean_A", "J_clean_W+U_full_A"):
        r25, r50, r100 = w25["n1_pairings"][pk], w50["n1_pairings"][pk], w100["n1_pairings"][pk]
        lines.append(
            f"| `{pk}` | {fd(r25['vs_own_no_pp'], 2)} | {r25['retention_pct']:.1f}% | "
            f"{fd(r50['vs_own_no_pp'], 2)} | {r50['retention_pct']:.1f}% | "
            f"{fd(r100['vs_own_no_pp'], 2)} | {r100['retention_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "### 3.3 世界头关键配对差值 (pp)",
        "",
        "| 对比项 | +25 Δskill (pp) | +50 Δskill (pp) | +100 Δskill (pp) |",
        "|---|---:|---:|---:|",
    ])
    for cname in (
        "J_clean_gt_vs_J_legacy_gt",
        "J_clean_no_vs_J_legacy_no",
        "J_clean_n1_vs_J_legacy_n1",
        "E106CW_Ufull_vs_D",
        "E106CW_Ufull_vs_E106C",
        "JcleanW_Ufull_vs_E106CW_E106C",
    ):
        d25, d50, d100 = w25["diffs"][cname], w50["diffs"][cname], w100["diffs"][cname]
        lines.append(
            f"| `{cname}` | {d25['mean_pp']:+.2f} [{d25['ci95_pp'][0]:+.2f}, {d25['ci95_pp'][1]:+.2f}] | "
            f"{d50['mean_pp']:+.2f} [{d50['ci95_pp'][0]:+.2f}, {d50['ci95_pp'][1]:+.2f}] | "
            f"{d100['mean_pp']:+.2f} [{d100['ci95_pp'][0]:+.2f}, {d100['ci95_pp'][1]:+.2f}] |"
        )

    REPORT_MD.write_text("\n".join(lines) + "\n")
    print("Wrote E125 report to", REPORT_MD, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        _, _, tr, dev = datasets()
        x = torch.as_tensor(JOINTS, device="cuda")
        assert len(dev.rows) == 211 and len(truths) > 0 and x.shape[0] == 12
        print("eval_e125 smoke check passed (including CUDA tensor init)!", flush=True)
    else:
        main()
