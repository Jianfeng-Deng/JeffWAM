# E133 comprehensive evaluator (offline 211w/27p + gripper F1@5/10 & open/close recall + World N1 retention + MuJoCo M3 closed-loop!)
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for sub in ("code_e132", "code_e131", "code_e130", "code_e129"):
    if str(T / sub) not in sys.path:
        sys.path.insert(0, str(T / sub))

from e133_common import E133WAM, INIT, LOCK_PATH, CTRL133, brt, build_or_load_extra_stats, attach_extra_proprio
from train_e133 import ARM_SPECS
from build_bimanual_mjcf import verify_fk_alignment, PIPER_JSON
from run_e131_sim import build_sim_proprio_batch, fit_blind_t1_model, predict_blind_t1, JI
from run_e132_fix_and_ori import BimanualPiperSimV2, pose18_to_pos_and_R, geodesic_deg
from e129_common import GR, UpgradedWAM
from e103d_model import WorldModel as OrigModel
from e99w2_model import JOINTS, pose_torch, trajectory
from e99w2_data import pose_np
from e102x_setup import datasets
from e102a_contract import contract, onset_labels
from e110f_core import infer_d
from e110f_eval import float_batch, transitions, match
from e86.data import move, collate

OUT = T / "results/e133_eval_20261119_01"
REPORT_MD = T / "notes/e133_pose_vel_grip_tokendrop_20261119.md"


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
    N = len(OBS)
    q0_j = Q0[:, JI].numpy()
    near = np.array([(lambda o: o[np.argmin(np.abs(q0_j[o] - q0_j[i]).sum(1))])(np.where(pid != pid[i])[0]) for i in range(N)])

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

    def load_any_wam(kind, path):
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if kind == "orig":
            m = OrigModel(tr.norm)
            m.load_state_dict(cp["model"], strict=True)
        elif kind == "v1":
            m = UpgradedWAM(tr.norm, variant="v1").widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        elif kind in ARM_SPECS:
            sp = ARM_SPECS[kind]
            m = E133WAM(tr.norm, use_switch_head=sp["grip_switch_loss"], vis_token_drop=sp["vis_token_drop"], use_kin_skip=sp["use_kin_skip"]).widen_from_164()
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

    model_defs = [
        ("E106C", "orig", INIT, False),
        ("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt", True),
        ("E133_G1_posevel", "G1", T / "results/e133_wam_g1_s90_20261119_01/ckpt_04000.pt", True),
        ("E133_G2_grip", "G2", T / "results/e133_wam_g2_s90_20261119_01/ckpt_04000.pt", True),
        ("E133_G3_drop15", "G3", T / "results/e133_wam_g3_s90_20261119_01/ckpt_04000.pt", True),
        ("E133_G4_drop20_skip", "G4", T / "results/e133_wam_g4_s90_20261119_01/ckpt_04000.pt", True),
    ]

    P = {}
    for name, kind, path, is_230 in model_defs:
        if path.exists():
            mw = load_any_wam(kind, path)
            P[name] = infer_wam_actions(mw, is_230=is_230, swap=False)
            if is_230:
                P[name + "_swap"] = infer_wam_actions(mw, is_230=is_230, swap=True)
            del mw
            torch.cuda.empty_cache()

    blind_t1 = fit_blind_t1_model(tr)
    b_preds = []
    for wi, e in enumerate(entries):
        _, q_ep, _, _ = dev.states.get(dev, e["cid"])
        b_preds.append(predict_blind_t1(blind_t1, q_ep, e["anchor"]))
    P["blind_t1_s90"] = torch.from_numpy(np.stack(b_preds)).float()

    main_models = [k for k in ("E106C", "WAM_v1_s90", "E133_G1_posevel", "E133_G2_grip", "E133_G3_drop15", "E133_G4_drop20_skip", "blind_t1_s90") if k in P]

    # ==========================================================================
    # 1. Action + Position + 3D Orientation + Amplitude + Gripper Metrics
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
            rot25=rot_deg[:, 24].mean(1),
            rot50=rot_deg[:, 49].mean(1),
            rms_disp=float(np.sqrt(((pj - q0_j[:, None, :]) ** 2).mean())),
        )
        for n, ev in EV.items():
            ii = np.array([x[0] for x in ev])
            tt = np.array([x[1] for x in ev])
            d["ev_" + n] = ee_mm[ii // 2, tt, ii % 2]
            d["rot_" + n] = rot_deg[ii // 2, tt, ii % 2]
        M[k] = d

    act_res = {"models": {}, "diffs": {}, "vision_swap_cost": {}, "gripper": {}}
    for k in main_models:
        act_res["models"][k] = {
            n: stat(M[k][n]) for n in ("l8", "l50", "e8", "e25", "e50", "rot8", "rot25", "rot50")
        }
        act_res["models"][k]["rms_disp_rad"] = M[k]["rms_disp"]
        for n in EV:
            act_res["models"][k]["ev_" + n] = stat(M[k]["ev_" + n], None, evg[n])
            act_res["models"][k]["rot_" + n] = stat(M[k]["rot_" + n], None, evg[n])
        if (k + "_swap") in M:
            act_res["vision_swap_cost"][k] = {n: stat(M[k + "_swap"][n] - M[k][n]) for n in ("e8", "e25", "e50")}

    onset_cache = {}
    for k in [x for x in main_models if x != "blind_t1_s90"]:
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
        f1_5_ci = [float(f1_5_p.mean())] + [float(x) for x in np.quantile((W @ f1_5_p) / np.maximum(W.sum(1), 1), [0.025, 0.975])]
        f1_10_ci = [float(f1_10_p.mean())] + [float(x) for x in np.quantile((W @ f1_10_p) / np.maximum(W.sum(1), 1), [0.025, 0.975])]

        act_res["gripper"][k] = dict(
            state_acc=stat(acc, None, g2),
            drop25=stat(pg[:, :25].max(1), hold, g2),
            close_hit10=float(np.mean(hit10["close"])),
            open_hit10=float(np.mean(hit10["open"])),
            close_hit5=float(np.mean(hit5["close"])),
            open_hit5=float(np.mean(hit5["open"])),
            f1_5=f1_5_ci,
            f1_10=f1_10_ci,
        )

    comp_pairs = [
        ("E133_G1_posevel", "WAM_v1_s90"),
        ("E133_G2_grip", "E133_G1_posevel"),
        ("E133_G2_grip", "WAM_v1_s90"),
        ("E133_G3_drop15", "E133_G2_grip"),
        ("E133_G3_drop15", "WAM_v1_s90"),
        ("E133_G4_drop20_skip", "WAM_v1_s90"),
        ("E133_G3_drop15", "E106C"),
        ("E133_G4_drop20_skip", "E106C"),
        ("E133_G3_drop15", "blind_t1_s90"),
        ("E133_G4_drop20_skip", "blind_t1_s90"),
    ]
    for a, b in comp_pairs:
        if a in M and b in M:
            act_res["diffs"][f"{a}-{b}"] = {n: stat(M[a][n] - M[b][n]) for n in ("l8", "l50", "e8", "e25", "e50", "rot8", "rot50")}
            for n in EV:
                act_res["diffs"][f"{a}-{b}"]["ev_" + n] = stat(M[a]["ev_" + n] - M[b]["ev_" + n], None, evg[n])
                act_res["diffs"][f"{a}-{b}"]["rot_" + n] = stat(M[a]["rot_" + n] - M[b]["rot_" + n], None, evg[n])

    # ==========================================================================
    # 2. World Head Closed-Loop W(N1) Retention (+25, +50, +100)
    # ==========================================================================
    world_res = {str(h): {} for h in (25, 50, 100)}
    for name, kind, path, _ in model_defs:
        if not path.exists():
            continue
        wm = load_any_wam(kind, path)
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
                    e_pos = [entries.index(e) for e in es]
                    a_b = P[name][e_pos].to("cuda")
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
            world_res[str(h)][name] = dict(
                gt_skill=float(s_gt.mean()),
                no_skill=float(s_no.mean()),
                n1_skill=float(s_n1.mean()),
                n1_minus_no_pp=[float(d_n1_no.mean()), float(np.quantile(bs_d, 0.025)), float(np.quantile(bs_d, 0.975))],
                retention_pct=float(100.0 * d_n1_no.mean() / denom) if abs(denom) > 1e-9 else 0.0,
            )
        del wm
        torch.cuda.empty_cache()

    # ==========================================================================
    # 3. MuJoCo M3 Closed-Loop Verification (S2_replan_vis & S4_perturb_15mm + Clip 2 win=166 check!)
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

    sim_scenarios = [
        ("S2_replan_vis", False),
        ("S4_perturb_15mm", True),
    ]
    mujoco_res = {sc_n: {} for sc_n, _ in sim_scenarios}

    for name, kind, path, is_230 in model_defs + [("blind_t1_s90", "blind", None, False)]:
        if kind != "blind" and not path.exists():
            continue
        wm = load_any_wam(kind, path) if kind != "blind" else None
        for sc_n, pert in sim_scenarios:
            sim_pos = np.zeros((len(entries), 50, 2, 3), dtype="f8")
            sim_R = np.zeros((len(entries), 50, 2, 3, 3), dtype="f8")
            tcols = np.zeros(len(entries), dtype="f8")
            acols = np.zeros(len(entries), dtype="f8")

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
                        if kind == "blind":
                            raw_chunk = predict_blind_t1(blind_t1, q_hist, a0 + step)
                        else:
                            vis_off = 25 if (step >= 25 and (wi, 25) in vis_batches) else (10 if (step >= 10 and (wi, 10) in vis_batches) else 0)
                            b_live = build_sim_proprio_batch(vis_batches[(wi, vis_off)], q_hist, p_hist, a0 + step, z, good, c_dict, extra_stats, dev.base.stats, is_230)
                            with torch.inference_mode():
                                raw_chunk = infer_d(wm, move(b_live, "cuda"), 4, 90)[0].float().cpu().numpy()
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
            ev_s = {}
            for n, ev in EV.items():
                ii = np.array([x[0] for x in ev])
                tt = np.array([x[1] for x in ev])
                ev_s["ev_" + n] = stat(np.linalg.norm(sim_pos[ii // 2, tt, ii % 2] - GT_POS[ii // 2, tt, ii % 2], axis=-1) * 1000.0, None, evg[n])
            mujoco_res[sc_n][name] = dict(
                e25=stat(ee_mm[:, 24].mean(1)),
                e50=stat(ee_mm[:, 49].mean(1)),
                rot50=stat(rot_deg[:, 49].mean(1)),
                **ev_s,
                win166_end_mm=float(ee_mm[166, 49].mean()),
                table_col=int(tcols.sum()),
                arm_col=int(acols.sum()),
            )
        if wm is not None:
            del wm
            torch.cuda.empty_cache()

    full_summary = dict(
        experiment="E133_pose_vel_grip_tokendrop",
        brt=brt(),
        elapsed_s=round(time.monotonic() - t0, 1),
        action=act_res,
        world=world_res,
        mujoco_m3=mujoco_res,
    )
    (OUT / "summary.json").write_text(json.dumps(full_summary, indent=2, ensure_ascii=False) + "\n")
    write_markdown(full_summary)
    print("E133 evaluation completed in", full_summary["elapsed_s"], "s!", flush=True)


def fmd(arr, d=2):
    return f"{arr[0]:+.{d}f} [{arr[1]:+.{d}f}, {arr[2]:+.{d}f}]"


def write_markdown(s):
    am = s["action"]["models"]
    ad = s["action"]["diffs"]
    gr = s["action"]["gripper"]
    sw = s["action"]["vision_swap_cost"]
    wr = s["world"]
    mj = s["mujoco_m3"]

    lines = [
        "# E133 第一梯队三项改进验证：三维指尖位姿+速度斜率损失、夹爪切换加权与跳变头、视觉随机遮挡",
        "",
        f"完成时间：{s['brt']}。代码：`code_e133/`；结果：`results/e133_*_20261119_01/`。",
        "四组阶梯对照（统一 4000 步，`seed=90` 与 `D` 样本流 SHA 完全一致）：",
        "- `E133_G1_posevel`：在 `WAM_v1` 基础上加入 **三维指尖位置+朝向损失（`pose_torch`）+ 帧间速度斜率 L1**；",
        "- `E133_G2_grip`：在 `G1` 基础上加入 **夹爪切换窗口加权 BCE（闭合 6× / 松开 10×）+ 独立切换窗口与跳变帧预测头**；",
        "- `E133_G3_drop15`：在 `G2` 基础上加入 **`15%` 视觉空间 Token 随机遮挡**；",
        "- `E133_G4_drop20_skip`：在 `G2` 基础上加入 **`20%` 视觉空间 Token 随机遮挡 + 直连运动学分支**。",
        "",
        "## 1. 离线动作、三维位置 (`mm`)、三维姿态角 (`deg`) 与动作幅度 `RMS`（211 窗 / 27 parent）",
        "",
        "| 模型 | 前8步 `l8` | 50步 `l50` | +8步位置/姿态 | +50步位置/姿态 | 夹紧位置/姿态 | 松爪位置/姿态 | 位移 `RMS` | 换图代价(+50) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in am.items():
        sw_s = f"+{sw[k]['e50'][0]:.2f}mm" if k in sw else "—"
        lines.append(
            f"| `{k}` | {r['l8'][0]:.5f} | {r['l50'][0]:.5f} | {r['e8'][0]:.2f}mm / {r['rot8'][0]:.2f}° | "
            f"**{r['e50'][0]:.2f}mm / {r['rot50'][0]:.2f}°** | {r['ev_close'][0]:.2f}mm / {r['rot_close'][0]:.2f}° | "
            f"**{r['ev_open'][0]:.2f}mm / {r['rot_open'][0]:.2f}°** | {r['rms_disp_rad']:.4f} | {sw_s} |"
        )

    lines.extend([
        "",
        "### 关键配对差值（负值表示前者误差更低）",
        "",
        "| 配对比较 | `Δl8` | `Δl50` | `Δe50` (mm) | `Δrot50` (°) | `Δ松爪位置` (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pair, d in ad.items():
        lines.append(
            f"| `{pair}` | {fmd(d['l8'], 5)} | {fmd(d['l50'], 5)} | {fmd(d['e50'], 2)} | {fmd(d['rot50'], 2)} | {fmd(d['ev_open'], 2)} |"
        )

    lines.extend([
        "",
        "## 2. 夹爪开关事件专项指标（检验 `G2` 切换加权 + 独立跳变帧头）",
        "",
        "| 模型 | 状态准确率 | 事件 `F1@±5` | 事件 `F1@±10` | 闭合命中(±10/±5) | **松开命中(±10/±5)** | 夹持误张开(25步) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for k, g in gr.items():
        lines.append(
            f"| `{k}` | {g['state_acc'][0]:.4f} | **{g['f1_5'][0]:.3f}** | **{g['f1_10'][0]:.3f}** | "
            f"{g['close_hit10']*100:.1f}% / {g['close_hit5']*100:.1f}% | "
            f"**{g['open_hit10']*100:.1f}% / {g['open_hit5']*100:.1f}%** | {g['drop25'][0]*100:.2f}% |"
        )

    lines.extend([
        "",
        "## 3. 世界头闭环 `W(N1)` 保留率（检验速度斜率损失修复时间/幅度后对世界头的增益）",
        "",
        "| 模型 | +25 `gt / no / N1` (%) | +25 保留率 | +50 `gt / no / N1` (%) | +50 `N1−no` (pp) | **+50 保留率** | **+100 保留率** |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for wk in wr["50"]:
        r25, r50, r100 = wr["25"][wk], wr["50"][wk], wr["100"][wk]
        lines.append(
            f"| `{wk}` | {r25['gt_skill']:.2f}/{r25['no_skill']:.2f}/{r25['n1_skill']:.2f} | {r25['retention_pct']:.1f}% | "
            f"{r50['gt_skill']:.2f}/{r50['no_skill']:.2f}/{r50['n1_skill']:.2f} | "
            f"{fmd(r50['n1_minus_no_pp'], 2)} | **{r50['retention_pct']:.1f}%** | **{r100['retention_pct']:.1f}%** |"
        )

    lines.extend([
        "",
        "## 4. MuJoCo 双臂 `M3` 闭环验证（含视频 Clip 2 `win=166` 大幅抬臂窗口检验）",
        "",
        "| 模型 | `S2` 闭环 +50位置/姿态 | `S2` 松爪位置 | `S2` `win=166` 大抬臂末帧误差 | `S4` 拨偏15mm +50位置/姿态 | 撞桌/互碰 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for k in mj["S2_replan_vis"]:
        s2 = mj["S2_replan_vis"][k]
        s4 = mj["S4_perturb_15mm"][k]
        lines.append(
            f"| `{k}` | **{s2['e50'][0]:.2f}mm / {s2['rot50'][0]:.2f}°** | {s2['ev_open'][0]:.2f}mm | "
            f"{s2['win166_end_mm']:.1f}mm | **{s4['e50'][0]:.2f}mm / {s4['rot50'][0]:.2f}°** | "
            f"{s2['table_col']}/{s2['arm_col']} (S4: {s4['table_col']}/{s4['arm_col']}) |"
        )

    REPORT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        _, _, tr, dev = datasets()
        entries = json.loads(LOCK_PATH.read_text())["entries"]
        _, norm_dev = build_or_load_extra_stats(tr, dev, entries[:4])
        cp = torch.load(INIT, map_location="cpu", weights_only=False)
        mw = E133WAM(tr.norm, use_switch_head=True, vis_token_drop=0.15, use_kin_skip=True)
        mw.load_state_dict(cp["model"], strict=False)
        mw.widen_from_164().cuda().eval()
        b = move(float_batch(dev.batch([e["index"] for e in entries[:4]], (25,), "short")), "cuda")
        b = attach_extra_proprio(b, np.stack([norm_dev[e["index"]] for e in entries[:4]]))
        with torch.inference_mode():
            act = infer_d(mw, b, 4, 90)
        assert act.shape == (4, 50, 14) and torch.isfinite(act).all()
        print("eval_e133 smoke check passed!", flush=True)
    else:
        main()
