# E131: Hardware-Matched Bimanual Piper MuJoCo Closed-Loop Verification
# Tests E106C, WAM_v1_s90, WAM_N0_clean, WAM_N2_kinskip, blind_t1_s90 (+ GT_demo MIT floor)
# across 211 held-out windows / 27 unseen parents under commissioned piper.json 200Hz/300Hz MIT control
# (kp=10.0, kd=0.8 + gravity comp, 1.8 rad/s speed limit, 4.5 rad/s^2 accel limit, table & inter-arm collision).
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import mujoco

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
sys.path.insert(0, str(T / "code_e130"))
sys.path.insert(0, str(T / "code_e129"))

from build_bimanual_mjcf import verify_fk_alignment, ASSETS, PIPER_JSON, OUT_XML
from e130_common import E130WAM
from e129_common import (
    INIT, LOCK_PATH, GR, SHORT_OFFSETS, brt,
    UpgradedWAM, build_or_load_extra_stats, attach_extra_proprio, extra_causal_66,
)
from eval_e129 import fit_blind_baselines
from e103d_model import WorldModel as OrigModel
from e99w2_model import JOINTS, pose_torch
from e99w2_data import pose_np
from e102x_setup import datasets
from e102a_contract import contract, causal_features, fixed_np, states as grip_states
from e110f_core import infer_d
from e110f_eval import float_batch
from e86.data import move

OUT = T / "results/e131_sim_20261119_01"
REPORT_MD = T / "notes/e131_mujoco_bimanual_piper_sim_20261119.md"

JI = np.asarray(JOINTS)
JOINT_LIMITS_RAD = np.radians(
    np.array([[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-120, 120]], dtype="f8")
)


class BimanualPiperSim:
    def __init__(self, xml_path, piper_cfg):
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        self.d_grav = mujoco.MjData(self.m)
        self.dt = float(self.m.opt.timestep)  # 1/300 s
        self.substeps_per_frame = int(round((1.0 / 30.0) / self.dt))  # 10 substeps = 33.33 ms

        mit = piper_cfg["mit"]
        ctrl = piper_cfg["control"]
        self.kp = np.array(mit["axes"]["left"]["kp"] + mit["axes"]["right"]["kp"], dtype="f8")  # 12x 10.0
        self.kd = np.array(mit["axes"]["left"]["kd"] + mit["axes"]["right"]["kd"], dtype="f8")  # 12x 0.8
        self.max_vel = float(ctrl["max_joint_speed_rad_s"])   # 1.8 rad/s
        self.max_acc = float(ctrl["max_joint_accel_rad_s2"])  # 4.5 rad/s^2
        self.max_gvel = float(ctrl["max_gripper_speed_m_s"])  # 0.15 m/s

        self.lo12 = np.concatenate([JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 0]])
        self.hi12 = np.concatenate([JOINT_LIMITS_RAD[:, 1], JOINT_LIMITS_RAD[:, 1]])

        # Identify geom groups for collision auditing
        self.table_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.left_gids = {i for i in range(self.m.ngeom) if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("L_")}
        self.right_gids = {i for i in range(self.m.ngeom) if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("R_")}
        self.tcp_l = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "L_tcp")
        self.tcp_r = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "R_tcp")
        self.base_l = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "L_base_link")
        self.base_r = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "R_base_link")

        # qpos layout: [L_j1..6 (0..5), L_g7, L_g8 (6..7), R_j1..6 (8..13), R_g7, R_g8 (14..15)]
        self.qpos_j12 = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13], dtype=int)
        self.q_cmd_prev = np.zeros(12, dtype="f8")
        self.v_cmd_prev = np.zeros(12, dtype="f8")
        self.g_cmd_prev = np.zeros(2, dtype="f8")

    def reset(self, q14_init):
        mujoco.mj_resetData(self.m, self.d)
        q12 = np.clip(np.asarray(q14_init, "f8")[JI], self.lo12, self.hi12)
        g2 = np.clip(np.asarray(q14_init, "f8")[GR], 0.0, 0.10)
        self.d.qpos[self.qpos_j12] = q12
        self.d.qpos[6] = 0.5 * g2[0]
        self.d.qpos[7] = -0.5 * g2[0]
        self.d.qpos[14] = 0.5 * g2[1]
        self.d.qpos[15] = -0.5 * g2[1]
        self.d.qvel[:] = 0.0
        self.q_cmd_prev = q12.copy()
        self.v_cmd_prev = np.zeros(12, dtype="f8")
        self.g_cmd_prev = g2.copy()
        mujoco.mj_forward(self.m, self.d)

    def get_q14(self):
        q14 = np.zeros(14, dtype="f4")
        q14[JI] = self.d.qpos[self.qpos_j12]
        q14[6] = float(self.d.qpos[6] - self.d.qpos[7])
        q14[13] = float(self.d.qpos[14] - self.d.qpos[15])
        return q14

    def get_local_ee(self):
        out = np.zeros((2, 3), dtype="f8")
        for idx, (bid, sid) in enumerate(((self.base_l, self.tcp_l), (self.base_r, self.tcp_r))):
            R_wb = self.d.xmat[bid].reshape(3, 3)
            out[idx] = R_wb.T @ (self.d.site_xpos[sid] - self.d.xpos[bid])
        return out  # (2, 3) m in each arm's local base frame (matches pose_torch!)

    def step_frame(self, q14_target):
        q_tgt = np.asarray(q14_target, "f8")[JI]
        g_tgt = np.clip(np.asarray(q14_target, "f8")[GR], 0.0, 0.10)
        limit_hit = bool(((q_tgt < self.lo12) | (q_tgt > self.hi12)).any())
        q_tgt = np.clip(q_tgt, self.lo12, self.hi12)

        peak_tau = 0.0
        tau_sq_sum = 0.0
        table_col = False
        arm_col = False

        # Smoothly interpolate command across the 10 substeps with hardware speed/accel limits
        for sub in range(self.substeps_per_frame):
            frac = (sub + 1) / self.substeps_per_frame
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

            # Gravity-only RNEA feedforward (exact piper.json gravity_only mode)
            self.d_grav.qpos[:] = self.d.qpos
            self.d_grav.qvel[:] = 0.0
            self.d_grav.qacc[:] = 0.0
            mujoco.mj_forward(self.m, self.d_grav)
            tau_ff = self.d_grav.qfrc_bias[self.qpos_j12]

            q_cur = self.d.qpos[self.qpos_j12]
            v_cur = self.d.qvel[self.qpos_j12]
            tau = np.clip(self.kp * (q_cmd - q_cur) - self.kd * v_cur + tau_ff, -35.0, 35.0)

            # Actuators: [L_m1..6, L_g7, L_g8, R_m1..6, R_g7, R_g8]
            self.d.ctrl[0:6] = tau[0:6]
            self.d.ctrl[6] = 0.5 * g_cmd[0]
            self.d.ctrl[7] = -0.5 * g_cmd[0]
            self.d.ctrl[8:14] = tau[6:12]
            self.d.ctrl[14] = 0.5 * g_cmd[1]
            self.d.ctrl[15] = -0.5 * g_cmd[1]

            mujoco.mj_step(self.m, self.d)

            peak_tau = max(peak_tau, float(np.abs(tau).max()))
            tau_sq_sum += float((tau ** 2).mean())

            for ci in range(self.d.ncon):
                c = self.d.contact[ci]
                if c.dist < -0.001:  # >1mm real penetration
                    g1, g2 = int(c.geom1), int(c.geom2)
                    if g1 == self.table_gid or g2 == self.table_gid:
                        table_col = True
                    elif (g1 in self.left_gids and g2 in self.right_gids) or (g2 in self.left_gids and g1 in self.right_gids):
                        arm_col = True

        return dict(
            q14=self.get_q14(),
            ee=self.get_local_ee(),
            peak_tau=peak_tau,
            rms_tau=float(np.sqrt(tau_sq_sum / self.substeps_per_frame)),
            table_col=table_col,
            arm_col=arm_col,
            limit_hit=limit_hit,
        )


def build_sim_proprio_batch(base_b, q_hist, p_hist, a_idx, z, good, c_dict, extra_stats, dev_stats, is_230):
    """
    Replace ONLY proprioception / current state in batch `base_b` using live MuJoCo physical history `q_hist, p_hist`
    (preserving visual observations `obs` untouched).
    """
    b = dict(base_b)
    q_cur = q_hist[a_idx].astype("f4")
    q_mean = np.asarray(dev_stats["q_mean"], dtype="f4")
    q_std = np.asarray(dev_stats["q_std"], dtype="f4")
    b["q_current"] = torch.from_numpy(q_cur)[None]
    b["q_normalized"] = torch.from_numpy((q_cur - q_mean) / q_std)[None]
    b["grip_current"] = torch.from_numpy((q_cur[GR] > np.asarray(c_dict["threshold_m"])).astype("f4"))[None]

    # Recompute 164-D causal proprio from live MuJoCo trajectory
    hv = [bool(a_idx + off >= 0 and good[a_idx + off] and z["continuous_segment"][a_idx + off] == z["continuous_segment"][a_idx]) for off in (-24, -18, -12, -6)]
    hist164 = causal_features(q_hist, p_hist, a_idx, hv, z["continuous_segment"], good, c_dict)
    p164 = ((hist164 - np.asarray(c_dict["history_mean"], "f4")) / np.asarray(c_dict["history_std"], "f4")).astype("f4")
    if is_230:
        v66 = extra_causal_66(q_hist, p_hist, a_idx, z, good, c_dict)
        p66 = ((v66 - extra_stats["mean66"]) / extra_stats["std66"]).astype("f4")
        b["proprio"] = torch.from_numpy(np.concatenate([p164, p66])[None])
    else:
        b["proprio"] = torch.from_numpy(p164[None])
    return b


def fit_blind_t1_model(tr):
    # Train exact blind_t1_s90 PyTorch module so it can also run closed-loop inside MuJoCo!
    offs = (1, 3, 6, 12, 24)
    X, Y = [], []
    for r in tr.rows:
        _, q, _, _ = tr.states.get(tr, r["canonical_id"])
        a = r["anchor"]
        X.append(np.concatenate([q[a]] + [q[a] - q[max(a - k, 0)] for k in offs]))
        Y.append((q[a + 1 : a + 51] - q[a])[:, JI])
    X, Y = np.asarray(X, "f4"), np.asarray(Y, "f4")
    mx, sx = X.mean(0), X.std(0) + 1e-6
    my, sy = Y.mean(0, keepdims=True), Y.std(0, keepdims=True) + 1e-6
    Xn = torch.from_numpy((X - mx) / sx).cuda()
    Yn = torch.from_numpy((Y - my) / sy).cuda()

    torch.manual_seed(90)
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
        loss = torch.nn.functional.l1_loss(f(Xn[ix]).reshape(-1, 50, 12), Yn[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
    f.eval()
    return dict(net=f, mx=mx, sx=sx, my=my[0], sy=sy[0], offs=offs)


def predict_blind_t1(bmodel, q_hist, a_idx):
    x = np.concatenate([q_hist[a_idx]] + [q_hist[a_idx] - q_hist[max(a_idx - k, 0)] for k in bmodel["offs"]]).astype("f4")
    xn = torch.from_numpy((x - bmodel["mx"]) / bmodel["sx"])[None].cuda()
    with torch.inference_mode():
        d = bmodel["net"](xn).reshape(50, 12).cpu().numpy() * bmodel["sy"] + bmodel["my"]
    act = np.repeat(q_hist[a_idx][None], 50, axis=0).astype("f4")
    act[:, JI] = q_hist[a_idx, JI][None] + d
    return act


def main():
    assert not OUT.exists(), f"{OUT} already exists"
    OUT.mkdir(parents=False)
    t0 = time.monotonic()

    xml_path, fk_diff_mm = verify_fk_alignment()
    piper_cfg = json.loads(PIPER_JSON.read_text())
    sim = BimanualPiperSim(xml_path, piper_cfg)

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    _, _, tr, dev = datasets()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    assert len(entries) == 211
    build_or_load_extra_stats(tr, dev, entries)
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

    # Pre-fetch visual batches at t=0, t=+10, t=+25 for every dev window (zero visual domain gap!)
    vis_batches = {}  # (wi, offset) -> single-item float batch on CPU
    gt_actions = []
    gt_ees = []
    gl_list, gc_list = [], []
    for wi, e in enumerate(entries):
        b0 = float_batch(dev.batch([e["index"]], (10,), "short"))
        vis_batches[(wi, 0)] = b0
        gt_act = b0["action"][0]  # (50, 14)
        gt_actions.append(gt_act.numpy())
        with torch.no_grad():
            p_gt = pose_torch(gt_act[None].cuda().double()).cpu().numpy()[0]
        gt_ees.append(p_gt[:, [0, 1, 2, 9, 10, 11]].reshape(50, 2, 3))
        gl_list.append(b0["grip_label"][0].numpy())
        gc_list.append(b0["grip_current"][0].numpy())
        for off in (10, 25):
            sh = dev.shifted(e["index"], off, "short")
            if sh is not None:
                from e86.data import collate
                vis_batches[(wi, off)] = float_batch(collate([sh]))

    GT_ACT = np.stack(gt_actions)  # (211, 50, 14)
    GT_EE = np.stack(gt_ees)       # (211, 50, 2, 3) m
    GL = np.stack(gl_list).astype("i1").transpose(0, 2, 1).reshape(-1, 50)
    GC = (np.stack(gc_list) > 0.5).astype("i1").reshape(-1)
    events = lambda to, frm: [(i, int(np.argmax(GL[i] == to))) for i in range(len(GL)) if GC[i] == frm and (GL[i] == to).any()]
    EV = {"close": events(0, 1), "open": events(1, 0)}
    evg = {n: pid[np.array([x[0] for x in ev]) // 2] for n, ev in EV.items()}

    # Fixed deterministic 15mm (~0.03 rad on j1..j3, ~0.02 rad on j4..j6) initial perturbation per window
    rng_pert = np.random.default_rng(131)
    PERT_Q14 = np.zeros((len(entries), 14), dtype="f4")
    for wi in range(len(entries)):
        dq = np.zeros(14, dtype="f4")
        dq[[0, 1, 2, 7, 8, 9]] = rng_pert.choice([-0.03, 0.03], size=6)
        dq[[3, 4, 5, 10, 11, 12]] = rng_pert.choice([-0.02, 0.02], size=6)
        PERT_Q14[wi] = dq

    # Fit blind_t1_s90 once
    blind_t1 = fit_blind_t1_model(tr)

    def load_wam(kind, path):
        cp = torch.load(path, map_location="cpu", weights_only=False)
        if kind == "orig":
            m = OrigModel(tr.norm)
            m.load_state_dict(cp["model"], strict=True)
        elif kind == "v1":
            m = UpgradedWAM(tr.norm, variant="v1").widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        elif kind == "N0":
            m = E130WAM(tr.norm, clean_state=True, use_kin_skip=False).widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        elif kind == "N2":
            m = E130WAM(tr.norm, clean_state=True, use_kin_skip=True).widen_from_164()
            m.load_state_dict(cp["model"], strict=True)
        else:
            raise ValueError(kind)
        return m.cuda().eval()

    model_defs = [
        ("GT_demo_MIT", "gt", None, False),
        ("E106C", "orig", INIT, False),
        ("WAM_v1_s90", "v1", T / "results/e129_wam_v1_s90_20261119_01/ckpt_04000.pt", True),
        ("WAM_N0_clean", "N0", T / "results/e130_wam_n0_s90_20261119_01/ckpt_04000.pt", True),
        ("WAM_N2_kinskip", "N2", T / "results/e130_wam_n2_s90_20261119_01/ckpt_04000.pt", True),
        ("blind_t1_s90", "blind", None, False),
    ]

    scenarios = [
        ("S1_open50", dict(replan_steps=(0,), perturb=False)),
        ("S2_replan_vis", dict(replan_steps=(0, 10, 25), perturb=False)),
        ("S3_replan_k4", dict(replan_steps=tuple(range(0, 50, 4)), perturb=False)),
        ("S4_perturb_15mm", dict(replan_steps=(0, 10, 25), perturb=True)),
    ]

    results = {sc_name: {} for sc_name, _ in scenarios}

    for mname, kind, ckpt_path, is_230 in model_defs:
        wm = load_wam(kind, ckpt_path) if kind not in ("gt", "blind") else None
        t_m = time.monotonic()

        for sc_name, sc in scenarios:
            replan_set = set(sc["replan_steps"])
            sim_q_all = np.zeros((len(entries), 50, 14), dtype="f4")
            sim_ee_all = np.zeros((len(entries), 50, 2, 3), dtype="f8")
            init_ee_err = np.zeros(len(entries), dtype="f8")
            peak_taus = np.zeros(len(entries), dtype="f8")
            rms_taus = np.zeros(len(entries), dtype="f8")
            table_cols = np.zeros(len(entries), dtype="f8")
            arm_cols = np.zeros(len(entries), dtype="f8")
            limit_hits = np.zeros(len(entries), dtype="f8")
            replan_jumps = []

            for wi, e in enumerate(entries):
                z, q_ep, p_ep, good = dev.states.get(dev, e["cid"])
                a0 = e["anchor"]
                q_hist = q_ep.copy()
                p_hist = p_ep.copy()

                if sc["perturb"]:
                    # Shift initial pose at anchor (and pre-anchor history consistently with zero initial velocity!)
                    q_init = np.clip(q_ep[a0] + PERT_Q14[wi], -3.0, 3.0)
                    q_init[JI] = np.clip(q_init[JI], sim.lo12, sim.hi12)
                    q_init[GR] = q_ep[a0, GR]
                    # Zero-velocity start at perturbed pose so policy sees genuine stationary offset
                    start_h = max(0, a0 - 24)
                    q_hist[start_h : a0 + 1] = q_init[None]
                    p_hist[start_h : a0 + 1] = pose_np(q_init[None])
                else:
                    q_init = q_ep[a0].copy()

                sim.reset(q_init)
                ee0 = sim.get_local_ee()
                ee_gt0 = pose_np(q_ep[a0][None])[0, [0, 1, 2, 9, 10, 11]].reshape(2, 3)
                init_ee_err[wi] = float(np.linalg.norm(ee0 - ee_gt0, axis=-1).mean() * 1000.0)

                cur_chunk = None
                chunk_t0 = 0
                prev_cmd = q_init.copy()

                for step in range(50):
                    if step in replan_set:
                        if kind == "gt":
                            cur_chunk = GT_ACT[wi, step:]
                        elif kind == "blind":
                            cur_chunk = predict_blind_t1(blind_t1, q_hist, a0 + step)
                        else:
                            # Pick latest available real visual observation at <= step (0, 10, or 25)
                            vis_off = 0
                            if step >= 25 and (wi, 25) in vis_batches:
                                vis_off = 25
                            elif step >= 10 and (wi, 10) in vis_batches:
                                vis_off = 10
                            base_b = vis_batches[(wi, vis_off)]
                            b_live = build_sim_proprio_batch(
                                base_b, q_hist, p_hist, a0 + step, z, good, c_dict,
                                extra_stats, dev.base.stats, is_230,
                            )
                            b_cuda = move(b_live, "cuda")
                            with torch.inference_mode():
                                cur_chunk = infer_d(wm, b_cuda, 4, 90)[0].float().cpu().numpy()
                        chunk_t0 = step
                        if step > 0:
                            replan_jumps.append(float(np.abs(cur_chunk[0, JI] - prev_cmd[JI]).mean()))

                    cmd = cur_chunk[min(step - chunk_t0, len(cur_chunk) - 1)]
                    prev_cmd = cmd
                    st_out = sim.step_frame(cmd)

                    sim_q_all[wi, step] = st_out["q14"]
                    sim_ee_all[wi, step] = st_out["ee"]
                    # Update live history with MuJoCo physical state (NOT raw prediction!)
                    q_hist[a0 + step + 1] = st_out["q14"]
                    p_hist[a0 + step + 1] = pose_np(st_out["q14"][None])[0]

                    peak_taus[wi] = max(peak_taus[wi], st_out["peak_tau"])
                    rms_taus[wi] += st_out["rms_tau"] / 50.0
                    table_cols[wi] = max(table_cols[wi], float(st_out["table_col"]))
                    arm_cols[wi] = max(arm_cols[wi], float(st_out["arm_col"]))
                    limit_hits[wi] = max(limit_hits[wi], float(st_out["limit_hit"]))

            # Compute physical tracking & smoothness metrics
            ee_err_mm = np.linalg.norm(sim_ee_all - GT_EE, axis=-1) * 1000.0  # (211, 50, 2)
            q_err = np.abs(sim_q_all[:, :, JI] - GT_ACT[:, :, JI])            # (211, 50, 12)
            jerk_e = (np.diff(sim_q_all[:, :, JI], 3, axis=1) ** 2).mean((1, 2))

            ev_stats = {}
            for n, ev in EV.items():
                ii = np.array([x[0] for x in ev])
                tt = np.array([x[1] for x in ev])
                ev_v = np.linalg.norm(sim_ee_all[ii // 2, tt, ii % 2] - GT_EE[ii // 2, tt, ii % 2], axis=-1) * 1000.0
                ev_stats["ev_" + n] = stat(ev_v, None, evg[n])

            results[sc_name][mname] = dict(
                init_ee_mm=float(init_ee_err.mean()),
                l8=stat(q_err[:, :8].mean((1, 2))),
                l50=stat(q_err.mean((1, 2))),
                e8=stat(ee_err_mm[:, 7].mean(1)),
                e25=stat(ee_err_mm[:, 24].mean(1)),
                e50=stat(ee_err_mm[:, 49].mean(1)),
                **ev_stats,
                peak_tau_nm=float(peak_taus.mean()),
                max_peak_tau_nm=float(peak_taus.max()),
                rms_tau_nm=float(rms_taus.mean()),
                jerk_e=float(jerk_e.mean()),
                replan_jump_rad=float(np.mean(replan_jumps)) if replan_jumps else 0.0,
                table_col_windows=int(table_cols.sum()),
                arm_col_windows=int(arm_cols.sum()),
                limit_hit_windows=int(limit_hits.sum()),
            )
            print(
                f"[{brt()}] {sc_name:16s} | {mname:15s} | e8={results[sc_name][mname]['e8'][0]:.2f} "
                f"e25={results[sc_name][mname]['e25'][0]:.2f} e50={results[sc_name][mname]['e50'][0]:.2f}mm "
                f"open={results[sc_name][mname]['ev_open'][0]:.2f}mm tau_pk={results[sc_name][mname]['peak_tau_nm']:.2f}Nm "
                f"col(tbl/arm)={results[sc_name][mname]['table_col_windows']}/{results[sc_name][mname]['arm_col_windows']}",
                flush=True,
            )

        if wm is not None:
            del wm
            torch.cuda.empty_cache()
        print(f"Finished {mname} in {time.monotonic() - t_m:.1f}s", flush=True)

    summary = dict(
        experiment="E131_mujoco_bimanual_piper_sim",
        brt=brt(),
        fk_alignment_max_diff_mm=fk_diff_mm,
        mit_kp=10.0,
        mit_kd=0.8,
        total_elapsed_s=round(time.monotonic() - t0, 1),
        results=results,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_markdown(summary)

    # Append concise summary to worklog.md
    s2 = results["S2_replan_vis"]
    s3 = results["S3_replan_k4"]
    s4 = results["S4_perturb_15mm"]
    wl = (
        f"\n- 2026-11-19 BRT E131（Antigravity执行）：MuJoCo 3.8.1 双臂Piper真机工作台闭环验证（完全对齐piper.json：±0.3027m基座、z=-0.0044m台面、kp=10/kd=0.8+重力前馈，FK对齐<{fk_diff_mm:.2f}mm，211窗/27parent）："
        f"S2视觉+物理重规划(+50末端/松爪)：GT底座={s2['GT_demo_MIT']['e50'][0]:.2f}/{s2['GT_demo_MIT']['ev_open'][0]:.2f}mm，"
        f"E106C={s2['E106C']['e50'][0]:.2f}/{s2['E106C']['ev_open'][0]:.2f}mm，WAM_v1={s2['WAM_v1_s90']['e50'][0]:.2f}/{s2['WAM_v1_s90']['ev_open'][0]:.2f}mm，"
        f"WAM_N2={s2['WAM_N2_kinskip']['e50'][0]:.2f}/{s2['WAM_N2_kinskip']['ev_open'][0]:.2f}mm，blind_t1={s2['blind_t1_s90']['e50'][0]:.2f}/{s2['blind_t1_s90']['ev_open'][0]:.2f}mm；"
        f"S3高频K=4反馈(WAM_v1 e50={s3['WAM_v1_s90']['e50'][0]:.2f}mm, jerk={s3['WAM_v1_s90']['jerk_e']:.2e})；"
        f"S4开局15mm扰动恢复(+50末端)：WAM_v1={s4['WAM_v1_s90']['e50'][0]:.2f}mm vs blind_t1={s4['blind_t1_s90']['e50'][0]:.2f}mm。见 notes/e131_mujoco_bimanual_piper_sim_20261119.md。\n"
    )
    with (T / "notes/worklog.md").open("a") as f:
        f.write(wl)
    print("E131 COMPLETED in", summary["total_elapsed_s"], "s", flush=True)


def write_markdown(s):
    lines = [
        "# E131 双臂 Piper 真机工作台 MuJoCo 闭环动力学与泛化验证",
        "",
        f"完成时间：{s['brt']}（总耗时 {s['total_elapsed_s']}s）。代码：`code_e131/`；模型资产：`sim/piper_bimanual_assets_20261119_02/`；结果：`results/e131_sim_20261119_01/`。",
        f"物理建模校验：双臂基座 `±0.3027 m, yaw=+90°`，台面 `z=-0.0044 m`，300 Hz 物理步进 + 真机 `piper.json` 锁定 MIT 控制律（`kp=10.0, kd=0.8` + 重力前馈 + `1.8 rad/s` 限速 + `4.5 rad/s²` 限加速度）；正运动学与 `pose_torch` 对齐最大误差 **`{s['fk_alignment_max_diff_mm']:.4f} mm`**。",
        "",
    ]
    sc_titles = {
        "S1_open50": "## 1. S1 开环 50 步执行（经真机 MIT 柔顺控制器滤波，基准物理跟踪与安全）",
        "S2_replan_vis": "## 2. S2 视觉+物理闭环重规划（`t=0, +10, +25` 刷新真实视觉与 MuJoCo 物理状态）",
        "S3_replan_k4": "## 3. S3 高频 `K=4` 步（7.5 Hz）物理速度反馈重规划（专项检验 `t-1, t-3` 速度反馈稳定性）",
        "S4_perturb_15mm": "## 4. S4 泛化抗扰测试：开局强行拨偏 `±15 mm`（`±0.03 rad`，零初速）+ 闭环重规划恢复",
    }
    for sc_key, title in sc_titles.items():
        lines.extend([
            title,
            "",
            "| 模型 | +8步末端 | +25步末端 | +50步末端 | 夹紧37次 | 松爪16次 | 峰值力矩 (N·m) | 物理 Jerk | 撞桌 / 双臂碰撞窗数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for mname, r in s["results"][sc_key].items():
            lines.append(
                f"| `{mname}` | {r['e8'][0]:.2f} mm | {r['e25'][0]:.2f} mm | **{r['e50'][0]:.2f} mm** | "
                f"{r['ev_close'][0]:.2f} mm | {r['ev_open'][0]:.2f} mm | {r['peak_tau_nm']:.2f} (max {r['max_peak_tau_nm']:.1f}) | "
                f"{r['jerk_e']:.2e} | {r['table_col_windows']} / {r['arm_col_windows']} |"
            )
        lines.append("")

    REPORT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        xml_path, fk_mm = verify_fk_alignment()
        sim = BimanualPiperSim(xml_path, json.loads(PIPER_JSON.read_text()))
        q0 = np.zeros(14, dtype="f4")
        q0[[1, 2, 4, 8, 9, 11]] = [0.5, -1.0, 0.5, 0.5, -1.0, 0.5]
        q0[[6, 13]] = 0.05
        sim.reset(q0)
        out = sim.step_frame(q0)
        assert np.isfinite(out["q14"]).all() and out["peak_tau"] < 35.0
        print(f"E131 MuJoCo smoke passed! fk_diff={fk_mm:.4f}mm peak_tau={out['peak_tau']:.2f}Nm", flush=True)
    else:
        main()
