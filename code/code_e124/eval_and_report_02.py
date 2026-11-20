# E124 (P1-v2) comprehensive evaluator & Markdown report generator.
import os, sys, json, time, datetime
from pathlib import Path
import numpy as np
import torch

os.environ.update(
    E102_GROUP="A2",
    ARM="CF",
    PYTHONDONTWRITEBYTECODE="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    OMP_NUM_THREADS="4",
)

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
# Import s2 truths cleanly first, then clear top-level module cache so zero collision is possible
_saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "2")
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
_stage_cache = {}


def stage_feat(parent, frame, kind):
    if kind == "zero":
        return np.zeros(20, "f4")
    k = (parent, kind)
    if k not in _stage_cache:
        _stage_cache[k] = np.load(FEATS / (parent + ".npz"))["ctrl" if kind == "ctrl" else "oracle"].astype("f4")
    a = _stage_cache[k]
    return a[min(max(int(frame), 0), len(a) - 1)]


def widen(model):
    old = model.proprio_proj[0]
    new = torch.nn.Linear(184, old.out_features)
    with torch.no_grad():
        new.weight[:, :164] = old.weight
        new.weight[:, 164:] = 0
        new.bias.copy_(old.bias)
    model.proprio_proj[0] = new
    return model


def add_stage(b, rows_meta, kind):
    S_arr = np.stack([stage_feat(r["parent_component"], r["anchor"], kind) for r in rows_meta])
    s_t = torch.from_numpy(S_arr).to(b["proprio"].device, b["proprio"].dtype)
    b = dict(b)
    b["proprio"] = torch.cat([b["proprio"], s_t], 1)
    return b

OUT_DIR = T / "results/e124_p1v2_eval_20261118_02"
REPORT_MD = T / "notes/e124_p1v2_unfrozen_dropout_20261118.md"


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def has_boundary(parent, anchor, span=50):
    y = truths[parent][anchor : anchor + span + 1]
    y = y[y >= 0]
    return bool(len(y) > 1 and (y != y[0]).any())


def near_boundary(parent, anchor, span=50, pad=30):
    lo = max(0, anchor - pad)
    y = truths[parent][lo : anchor + span + 1 + pad]
    y = y[y >= 0]
    return bool(len(y) > 1 and (y != y[0]).any())


def main(phase="core4"):
    OUT_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)

    _, _, tr, dev = datasets()
    ent_list = json.loads((T / "results/e106c_control_20261115_02/PRETRAIN_LOCK.json").read_text())["entries"]
    avail = {p.stem for p in FEATS.glob("pc_*.npz")}
    E = [e for e in ent_list if e["parent"] in avail]
    assert len(E) == 160 and len({e["parent"] for e in E}) == 20

    # Candidate checkpoints (reference + E124 arms)
    ck_defs = [
        ("E106C", INIT, None),
        ("D", T / "results/e110f_d_20261116_05/ckpt_04000.pt", None),
        ("S0", T / "results/e115p1_s0_20261117_01/ckpt_04000.pt", "zero"),
        ("S1", T / "results/e115p1_s1_20261117_01/ckpt_04000.pt", "ctrl"),
        ("S2", T / "results/e115p1_s2_20261117_01/ckpt_04000.pt", "oracle"),
        ("U0", T / "results/e124_p1v2_u0_20261118_01/ckpt_04000.pt", "zero"),
        ("U1", T / "results/e124_p1v2_u1_20261118_01/ckpt_04000.pt", "ctrl"),
        ("U2", T / "results/e124_p1v2_u2_20261118_01/ckpt_04000.pt", "oracle"),
        ("D0", T / "results/e124_p1v2_d0_20261118_01/ckpt_04000.pt", "zero"),
        ("D1", T / "results/e124_p1v2_d1_20261118_01/ckpt_04000.pt", "ctrl"),
        ("D2", T / "results/e124_p1v2_d2_20261118_01/ckpt_04000.pt", "oracle"),
    ]
    active_cks = [(a, p, k) for (a, p, k) in ck_defs if p.exists()]
    print("Evaluating arms:", [a for a, _, _ in active_cks], flush=True)

    c = contract()
    ji_np = np.asarray(JOINTS)
    ji_cuda = torch.as_tensor(JOINTS, device="cuda")
    onset = {}

    rows = []
    pred_actions = {}  # (arm, index) -> (50, 14) numpy f4

    # 1. Action + Gripper + Jerk extraction per arm
    for arm, path, kind in active_cks:
        cp = torch.load(path, map_location="cpu", weights_only=False)
        m = WorldModel(tr.norm)
        if kind is not None:
            m = widen(m)
        m.load_state_dict(cp["model"], strict=True)
        m = m.cuda().eval()
        del cp

        with torch.inference_mode():
            for s in range(0, len(E), 4):
                es = E[s : s + 4]
                ids = [e["index"] for e in es]
                b = move(float_batch(dev.batch(ids, (10,), "short")), "cuda")
                if kind is not None:
                    b = add_stage(b, [dict(parent_component=e["parent"], anchor=e["anchor"]) for e in es], kind)
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
                    ev = {str(t): [0, 0, 0] for t in (5, 10, 20)}
                    for side in range(2):
                        pe = transitions(ps_g[:, side], cur_g[side])
                        te = transitions(ts_g[:, side], init_g[side])
                        for t in (5, 10, 20):
                            ev[str(t)][0] += len(match(pe, te, t))
                            ev[str(t)][1] += len(pe)
                            ev[str(t)][2] += len(te)

                    rows.append(
                        dict(
                            arm=arm,
                            parent=e["parent"],
                            index=e["index"],
                            anchor=e["anchor"],
                            bd=has_boundary(e["parent"], e["anchor"]),
                            near=near_boundary(e["parent"], e["anchor"]),
                            l1_8=l1_8,
                            l1_50=l1_50,
                            copy8=cp8,
                            copy50=cp50,
                            acc=float((ps_g == ts_g).mean()),
                            ev=ev,
                            acc_e=acc_e,
                            jerk_e=jerk_e,
                            tacc_e=tacc_e,
                            tjerk_e=tjerk_e,
                        )
                    )
        del m
        torch.cuda.empty_cache()

    # 2. Evaluate predicted actions inside untouched frozen E106-C world head (+25, +50, +100)
    w_cp = torch.load(INIT, map_location="cpu", weights_only=False)
    wm = WorldModel(tr.norm)
    wm.load_state_dict(w_cp["model"], strict=True)
    wm = wm.cuda().eval()
    del w_cp

    world_rows = []
    arm_names = [a for a, _, _ in active_cks]
    with torch.inference_mode():
        for h in (25, 50, 100):
            Eh = [e for e in E if h in e["horizons"]]
            for s in range(0, len(Eh), 4):
                es = Eh[s : s + 4]
                ids = [e["index"] for e in es]
                bh = move(float_batch(dev.batch(ids, (h,), "short")), "cuda")
                Y = bh["world"].double()
                copy_mse = Y.square().mean((1, 2)).cpu().numpy()
                P_gt = wm.forward_mean(bh)["mean"].double()
                P_no = wm.forward_mean(
                    dict(bh, robot=torch.zeros_like(bh["robot"]), robot_valid=torch.zeros_like(bh["robot_valid"]))
                )["mean"].double()
                gt_mse = (P_gt - Y).square().mean((1, 2)).cpu().numpy()
                no_mse = (P_no - Y).square().mean((1, 2)).cpu().numpy()

                arm_mse = {}
                for arm in arm_names:
                    a_batch = torch.from_numpy(np.stack([pred_actions[(arm, e["index"])] for e in es])).to("cuda")
                    P_a = wm.forward_mean(dict(bh, robot=trajectory(bh["context"], a_batch)))["mean"].double()
                    arm_mse[arm] = (P_a - Y).square().mean((1, 2)).cpu().numpy()

                for k_i, e in enumerate(es):
                    world_rows.append(
                        dict(
                            parent=e["parent"],
                            index=e["index"],
                            h=h,
                            copy=float(copy_mse[k_i]),
                            gt=float(gt_mse[k_i]),
                            no=float(no_mse[k_i]),
                            arms={arm: float(arm_mse[arm][k_i]) for arm in arm_names},
                        )
                    )
    del wm
    torch.cuda.empty_cache()

    # 3. Bootstrap statistics (2000 replicates, seed 90, exact E115-P1 protocol)
    ps = sorted({r["parent"] for r in rows})
    draws = np.random.default_rng(90).integers(len(ps), size=(2000, len(ps)))

    def sel_rows(arm, wkind):
        return [
            r for r in rows
            if r["arm"] == arm
            and (
                wkind == "all"
                or (wkind == "boundary" and r["bd"])
                or (wkind == "interior" and not r["bd"])
                or (wkind == "near" and r["near"])
            )
        ]

    def calc_metric(rs, mname):
        if not rs:
            return np.nan
        if mname in ("l1_8", "l1_50", "acc", "acc_e", "jerk_e", "tacc_e", "tjerk_e"):
            return float(np.mean([r[mname] for r in rs]))
        if mname == "rel8":
            return float(1.0 - np.mean([r["l1_8"] for r in rs]) / np.mean([r["copy8"] for r in rs]))
        if mname == "rel50":
            return float(1.0 - np.mean([r["l1_50"] for r in rs]) / np.mean([r["copy50"] for r in rs]))
        if mname == "f1_10":
            tp = sum(r["ev"]["10"][0] for r in rs)
            pr = sum(r["ev"]["10"][1] for r in rs)
            tt = sum(r["ev"]["10"][2] for r in rs)
            return float(2 * tp / max(pr + tt, 1))
        raise ValueError(mname)

    def boot_single(arm, wkind, mname):
        A_rows = sel_rows(arm, wkind)
        pa = {p: [r for r in A_rows if r["parent"] == p] for p in ps}
        fn = lambda idx: calc_metric([r for i in idx for r in pa[ps[i]]], mname)
        pt = fn(np.arange(len(ps)))
        bs = np.array([fn(d) for d in draws])
        return dict(value=float(pt), ci=[float(x) for x in np.nanquantile(bs, [0.025, 0.975])])

    def boot_diff(a, b, wkind, mname):
        A_rows = sel_rows(a, wkind)
        B_rows = sel_rows(b, wkind)
        pa = {p: [r for r in A_rows if r["parent"] == p] for p in ps}
        pb = {p: [r for r in B_rows if r["parent"] == p] for p in ps}
        fn = lambda idx: calc_metric([r for i in idx for r in pa[ps[i]]], mname) - calc_metric([r for i in idx for r in pb[ps[i]]], mname)
        pt = fn(np.arange(len(ps)))
        bs = np.array([fn(d) for d in draws])
        return dict(value=float(pt), ci=[float(x) for x in np.nanquantile(bs, [0.025, 0.975])])

    action_stats = {}
    for wkind in ("all", "boundary", "interior", "near"):
        action_stats[wkind] = {"arms": {}, "diffs": {}}
        for arm in arm_names:
            action_stats[wkind]["arms"][arm] = {
                m: boot_single(arm, wkind, m)
                for m in ("l1_8", "l1_50", "rel8", "rel50", "acc", "f1_10", "acc_e", "jerk_e")
            }
        action_stats[wkind]["truth_smoothness"] = {
            "tacc_e": float(calc_metric(sel_rows(arm_names[0], wkind), "tacc_e")),
            "tjerk_e": float(calc_metric(sel_rows(arm_names[0], wkind), "tjerk_e")),
        }
        pairs = [
            ("U2", "U0"), ("D2", "D0"), ("U1", "U0"), ("D1", "D0"),
            ("U0", "S0"), ("D0", "U0"), ("D0", "S0"), ("S2", "S0"), ("U0", "D"),
        ]
        for a, b in pairs:
            if a in arm_names and b in arm_names:
                action_stats[wkind]["diffs"][f"{a}-{b}"] = {
                    m: boot_diff(a, b, wkind, m)
                    for m in ("l1_8", "l1_50", "rel50", "acc", "f1_10", "acc_e", "jerk_e")
                }

    # World-head n1 skill & retention summary (parent-equal skill matching E110-F / E123)
    world_stats = {}
    for h in (25, 50, 100):
        wh = [r for r in world_rows if r["h"] == h]
        psh = sorted({r["parent"] for r in wh})
        dh = np.random.default_rng(90).integers(len(psh), size=(2000, len(psh)))

        def p_skill(key):
            out = []
            for p in psh:
                rp = [r for r in wh if r["parent"] == p]
                cp_m = np.mean([r["copy"] for r in rp])
                val_m = np.mean([r[key] if key in ("gt", "no") else r["arms"][key] for r in rp])
                out.append(100.0 * (1.0 - val_m / max(cp_m, 1e-12)))
            return np.array(out)

        sk_gt = p_skill("gt")
        sk_no = p_skill("no")
        denom = float((sk_gt - sk_no).mean())
        world_stats[str(h)] = {
            "gt_skill_pct": float(sk_gt.mean()),
            "no_skill_pct": float(sk_no.mean()),
            "gt_minus_no_pp": denom,
            "arms": {},
        }
        for arm in arm_names:
            sk_a = p_skill(arm)
            diff_no = sk_a - sk_no
            bs_no = diff_no[dh].mean(1)
            ret = float(diff_no.mean() / denom) if abs(denom) > 1e-9 else np.nan
            world_stats[str(h)]["arms"][arm] = dict(
                skill_pct=float(sk_a.mean()),
                vs_no_pp=float(diff_no.mean()),
                vs_no_ci95=[float(x) for x in np.quantile(bs_no, [0.025, 0.975])],
                retention_pct=100.0 * ret,
            )

    summary = dict(
        experiment="E124_P1v2_unfrozen_dropout",
        phase=phase,
        evaluated_brt=brt(),
        windows_all=160,
        windows_boundary=int(sum(r["bd"] for r in sel_rows(arm_names[0], "all"))),
        windows_interior=int(sum(not r["bd"] for r in sel_rows(arm_names[0], "all"))),
        parents=len(ps),
        arms_evaluated=arm_names,
        dev_stage_truth_note="model-generated retrospective labels (non-human), used as S2 upper-bound input and boundary stratify",
        action_and_smoothness=action_stats,
        world_n1_retention=world_stats,
    )

    print("Summary computed for phase:", phase, "arms:", arm_names, flush=True)
    (OUT_DIR / f"summary_{phase}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    # 4. Render clean Chinese Markdown report
    write_markdown_report(summary)


def fmt_v(d, digits=4, sci=False):
    if sci:
        return f"{d['value']:.2e} [{d['ci'][0]:.2e}, {d['ci'][1]:.2e}]"
    return f"{d['value']:.{digits}f} [{d['ci'][0]:.{digits}f}, {d['ci'][1]:.{digits}f}]"


def fmt_d(d, digits=5, sci=False):
    if sci:
        return f"{d['value']:+.2e} [{d['ci'][0]:+.2e}, {d['ci'][1]:+.2e}]"
    return f"{d['value']:+.{digits}f} [{d['ci'][0]:+.{digits}f}, {d['ci'][1]:+.{digits}f}]"


def write_markdown_report(s):
    arms = s["arms_evaluated"]
    all_s = s["action_and_smoothness"]["all"]
    bd_s = s["action_and_smoothness"]["boundary"]
    in_s = s["action_and_smoothness"]["interior"]

    lines = [
        "# E124（P1-v2）解冻主干 + 强制依赖阶段条件动作头复验",
        "",
        f"更新时间：{s['evaluated_brt']}（阶段：`{s['phase']}`）。代码：`code_e124/`；训练目录：`results/e124_p1v2_*_20261118_01/`；评估目录：`results/e124_p1v2_eval_20261118_02/`。",
        "基线：`results/e106c_f2_s90_20261115_02/ckpt_20000.pt`（E106-C 20000 步）；训练池与样本流与原 `E115-P1` 严格配对（45 parent / 13,172 行，4000 步，样本流 sha `f4c1a7c2ae74...` 完全一致，排除 `episode142`）。",
        "**注：阶段参考标签为模型回顾式生成（非人工），`U2`/`D2` 为真实阶段上界对照。**",
        "",
        "## 1. 实验组设置",
        "",
        "- 原 `E115-P1`（冻结 6 层主干）：`S0`（零阶段）、`S1`（因果控制器）、`S2`（回顾式阶段上界）。",
        "- 本轮 `E124`（**解冻 6 层 Transformer 主干及全部观测/动作投影**，冻结世界头专用参数并校验 SHA 未变）：",
        "  - `U0` / `U2`（及触发时 `U1`）：无额外丢弃（保留原 15% 历史丢弃），零阶段 vs 阶段上界（vs 因果阶段）。",
        "  - `D0` / `D2`（及触发时 `D1`）：训练时施加 **50% 快捷通道整块丢弃**（逐相机当前视觉 50%、历史视觉 50%、164 维本体+关节状态 50% 随机遮挡，阶段向量 100% 保留；配对步级随机掩码完全相同；推理时不丢弃）。",
        "",
        "## 2. 主评估结果（160 窗 / 20 parent，parent bootstrap 2000，seed 90）",
        "",
        "| 组别 | 前8步 L1 | 50步 L1 | 相对复制改善(H50) | 夹爪准确率 | 夹爪事件 F1@±10 | 预测加速度能量(all) | 预测加速度能量(切换30窗) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in arms:
        ma = all_s["arms"][a]
        mb = bd_s["arms"][a]
        lines.append(
            f"| `{a}` | {fmt_v(ma['l1_8'], 4)} | {fmt_v(ma['l1_50'], 4)} | {ma['rel50']['value']*100:.2f}% | "
            f"{fmt_v(ma['acc'], 3)} | {fmt_v(ma['f1_10'], 3)} | {ma['acc_e']['value']:.2e} | {mb['acc_e']['value']:.2e} |"
        )
    lines.append(
        f"| *真值参照* | — | — | — | — | — | *{all_s['truth_smoothness']['tacc_e']:.2e}* | *{bd_s['truth_smoothness']['tacc_e']:.2e}* |"
    )

    lines.extend([
        "",
        "## 3. 关键配对差值（全部 160 窗 vs 含阶段切换 30 窗）",
        "",
        "| 对比项 | 子集 | Δ 前8步 L1（负为优） | Δ 50步 L1（负为优） | Δ 夹爪 F1@±10（正为优） | Δ 加速度能量 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for pair in ("U2-U0", "D2-D0", "U1-U0", "D1-D0", "U0-S0", "D0-U0", "S2-S0"):
        for wk, wlabel in (("all", "全部 160 窗"), ("boundary", "切换 30 窗"), ("interior", "非切换 130 窗")):
            ds = s["action_and_smoothness"][wk]["diffs"]
            if pair in ds:
                d = ds[pair]
                lines.append(
                    f"| `{pair}` | {wlabel} | {fmt_d(d['l1_8'], 5)} | {fmt_d(d['l1_50'], 5)} | "
                    f"{fmt_d(d['f1_10'], 3)} | {fmt_d(d['acc_e'], sci=True)} |"
                )

    lines.extend([
        "",
        "## 4. 喂入冻结默认世界头（E106-C 20000 步）的 `n1` 想象保留率（160 窗 / 20 parent）",
        "",
        "| 动作来源 | +25 skill (%) | +25 `n1−no` (pp) | +25 保留率 | +50 skill (%) | +50 `n1−no` (pp) | +50 保留率 | +100 保留率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    w25, w50, w100 = s["world_n1_retention"]["25"], s["world_n1_retention"]["50"], s["world_n1_retention"]["100"]
    lines.append(
        f"| `gt`（上限） / `no`（下限） | {w25['gt_skill_pct']:.2f} / {w25['no_skill_pct']:.2f} | +{w25['gt_minus_no_pp']:.2f} | 100% / 0% | "
        f"{w50['gt_skill_pct']:.2f} / {w50['no_skill_pct']:.2f} | +{w50['gt_minus_no_pp']:.2f} | 100% / 0% | 100% / 0% |"
    )
    for a in arms:
        r25, r50, r100 = w25["arms"][a], w50["arms"][a], w100["arms"][a]
        lines.append(
            f"| `{a}` | {r25['skill_pct']:.2f} | {r25['vs_no_pp']:+.2f} [{r25['vs_no_ci95'][0]:+.2f}, {r25['vs_no_ci95'][1]:+.2f}] | {r25['retention_pct']:.1f}% | "
            f"{r50['skill_pct']:.2f} | {r50['vs_no_pp']:+.2f} [{r50['vs_no_ci95'][0]:+.2f}, {r50['vs_no_ci95'][1]:+.2f}] | {r50['retention_pct']:.1f}% | {r100['retention_pct']:.1f}% |"
        )

    # Automated objective verdict based on pre-registered criteria
    verdict_lines = ["", "## 5. 自动预注册判定小结", ""]
    for pair, desc in (("U2-U0", "解冻主干、无额外丢弃下真实阶段效应"), ("D2-D0", "解冻主干 + 50% 强制丢弃下真实阶段效应")):
        if pair in all_s["diffs"]:
            da = all_s["diffs"][pair]["l1_50"]
            db = bd_s["diffs"][pair]["l1_50"]
            df1 = all_s["diffs"][pair]["f1_10"]
            dacc_bd = bd_s["diffs"][pair]["acc_e"]
            pass_l1 = (da["ci"][1] < 0) or (db["ci"][1] < 0)
            pass_f1 = df1["ci"][0] > 0
            sharper = dacc_bd["ci"][0] > 0
            verdict_lines.append(
                f"- **`{pair}`（{desc}）**：全窗 ΔL1_50 = `{fmt_d(da, 5)}`，切换窗 ΔL1_50 = `{fmt_d(db, 5)}`，"
                f"ΔF1@±10 = `{fmt_d(df1, 3)}`，切换窗 Δ加速度能量 = `{fmt_d(dacc_bd, sci=True)}`"
                f" → 精度提升显著：`{pass_l1 or pass_f1}`；切换转折更尖锐：`{sharper}`。"
            )

    lines.extend(verdict_lines)
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print("Wrote report to", REPORT_MD, flush=True)


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "core4"
    if ph == "--smoke":
        _, _, tr, dev = datasets()
        assert len(truths) > 0 and INIT.exists() and FEATS.exists() and len(dev.rows) == 211
        print("eval_and_report smoke check passed; truths parents:", len(truths), "dev rows:", len(dev.rows), flush=True)
    else:
        main(ph)
