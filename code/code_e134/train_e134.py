# E134 trainer: 4-arm Flow-Matching vs Decoupled-Deterministic study on exact D sample stream (4000 steps):
#   - D_dec:  Decoupled Deterministic (G1 arm + gradient-detached G2 gripper switch head with 20% gripper dropout)
#   - FM_nat: Upgraded Native 6-layer Transformer Flow Matching (fixing all legacy e99w2 bugs + pose/vel consistency)
#   - FM_dit: Cached-backbone 4-block TemporalFlowHead (K=4 flow draws/step = 16k flow updates + pose/vel consistency)
#   - FM_res: Residual TemporalFlowHead warm-started on E133_G1 (mean + stochastic timing/amplitude residual flow)
import os, sys, time, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from e134_common import (
    T, INIT, G1_CKPT, D_CKPT, LOCK_PATH, CTRL134, POS_IDX, ROT_IDX, brt,
    UpgradedWAM, E134WAM, sample_smooth_noise, build_or_load_extra_stats, attach_extra_proprio,
)
sys.path.insert(0, str(T / "code_e133"))
sys.path.insert(0, str(T / "code_e129"))
from train_e133 import compute_switch_grip_loss
from train_e129 import PEAK_MIB, H_TUP, query_gpu_free, p1_loss

from e103d_data import p1_only
from e102x_setup import datasets
from e102a_contract import contract
from e99w2_model import JOINTS, trajectory, pose_torch
from e90.data import load_data, sample_horizons
from e90.sampling import MultiSourceSampler
from e90.objective import weighted_mean, chain_weight
from e86.common import seed_for
from e86.data import move
from e86.objective import mode_for, contrastive
from e96a import train as base
from e99w2_train import make_chain
from e110f_core import target_delta

ARM_CONFIGS = {
    "D_dec":  dict(arm_mode="det_decoupled", init_ckpt="E106C"),
    "FM_nat": dict(arm_mode="fm_native",     init_ckpt="E106C"),
    "FM_dit": dict(arm_mode="fm_dit",        init_ckpt="E106C"),
    "FM_res": dict(arm_mode="fm_residual",   init_ckpt="G1"),
}


def pose_and_vel_loss_from_delta(delta_pred, b, j_idx, fk_gt, sample_weights=None):
    """
    Differentiable 3D fingertip position + orientation + step-velocity slope loss.
    Supports optional per-sample weight `sample_weights` (e.g., t^2 weighting for Flow Matching 1-step denoised x1_hat!).
    """
    pred_q = b["q_current"][:, None].expand(-1, 50, -1).clone()
    pred_q[:, :, j_idx] = b["q_current"][:, None, j_idx] + delta_pred * b["action_std"][:, :, j_idx]
    fk_pred = pose_torch(pred_q)

    pos_err = (fk_pred[..., POS_IDX] - fk_gt[..., POS_IDX]).abs().mean(dim=(1, 2))
    rot_err = (fk_pred[..., ROT_IDX] - fk_gt[..., ROT_IDX]).abs().mean(dim=(1, 2))

    q0_j = b["q_current"][:, None, j_idx]
    v_pred = torch.cat([q0_j, pred_q[:, :, j_idx]], dim=1).diff(dim=1)
    v_gt = torch.cat([q0_j, b["action"][:, :, j_idx]], dim=1).diff(dim=1)
    v_scale = b["action_std"][:, 0:1, j_idx].clamp_min(1e-4)
    vel_err = ((v_pred - v_gt) / v_scale).abs().mean(dim=(1, 2))

    per_sample = 8.0 * pos_err + 1.5 * rot_err + 0.5 * vel_err
    if sample_weights is not None:
        loss = (per_sample * sample_weights).mean()
    else:
        loss = per_sample.mean()
    return loss, float(pos_err.mean().detach() * 1000.0), float(vel_err.mean().detach())


def build_detached_action_from_delta(b, delta_det, grip_det, c_dict, j_idx):
    act = b["q_current"][:, None].expand(-1, 50, -1).clone()
    act[:, :, j_idx] = b["q_current"][:, None, j_idx] + delta_det.detach() * b["action_std"][:, :, j_idx]
    act[:, :, [6, 13]] = torch.where(
        grip_det.detach() > 0,
        act.new_tensor(c_dict["open_m"]),
        act.new_tensor(c_dict["closed_m"]),
    )
    return act


def e134_joint_loss(model, b, globalstep, seed, c_dict, j_idx, chain=None):
    device = b["obs"].device
    bsz = len(b["obs"])
    t0 = torch.zeros(bsz, device=device)
    a0 = torch.zeros_like(b["action"])

    tgt_y1 = target_delta(b)  # (B, 50, 12)
    with torch.no_grad():
        fk_gt = pose_torch(b["action"])

    # Pass 1: Clean backbone pass (provides st_out, ah, base delta, and gradient-detached gripper switch predictions)
    heads = model(b, a0, None, t0, t0)
    grip_loss, grip_bce_val, sw_val = compute_switch_grip_loss(heads, b)
    align = contrastive(model, heads, b)

    if model.arm_mode == "det_decoupled":
        delta_l1 = (heads["delta"] - tgt_y1).abs().mean()
        pv_loss, pos_mm, vel_val = pose_and_vel_loss_from_delta(heads["delta"], b, j_idx, fk_gt)
        arm_loss = delta_l1 + pv_loss
        pred_delta_for_world = heads["delta"].detach()
        fm_l1_val = float(delta_l1.detach())

    elif model.arm_mode == "fm_native":
        # Sample flow time t ~ LogitNormal(0, 1) clamped to [0.02, 0.98], smooth AR(1) noise x0
        t_f = torch.sigmoid(torch.randn(bsz, device=device)).clamp(0.02, 0.98)
        x0 = sample_smooth_noise((bsz, 50, 12), device, dtype=tgt_y1.dtype, rho=0.85)
        xt = (1.0 - t_f[:, None, None]) * x0 + t_f[:, None, None] * tgt_y1
        v_target = tgt_y1 - x0

        a_in = torch.zeros_like(b["action"])
        a_in[:, :, j_idx] = xt
        out_fm = model(b, a_in, None, t_f, t_f)
        v_pred = out_fm["action"][:, :, j_idx]
        fm_l1 = (v_pred - v_target).abs().mean()

        # 1-step denoised target x1_hat = xt + (1 - t) * v_pred weighted by t^2 (strong near t->1, gentle near t->0)
        x1_hat = xt + (1.0 - t_f[:, None, None]) * v_pred
        pv_loss, pos_mm, vel_val = pose_and_vel_loss_from_delta(x1_hat, b, j_idx, fk_gt, sample_weights=(t_f ** 2))
        arm_loss = fm_l1 + 0.5 * pv_loss
        pred_delta_for_world = x1_hat.detach()
        fm_l1_val = float(fm_l1.detach())

    elif model.arm_mode in ("fm_dit", "fm_residual"):
        is_res = (model.arm_mode == "fm_residual")
        base_mu = heads["delta"].detach() if is_res else torch.zeros_like(tgt_y1)
        flow_target = (tgt_y1 - base_mu) if is_res else tgt_y1

        # For fm_residual, keep gentle G1 supervision on base delta_head so backbone stays crisp while flow_head learns multimodal residual!
        if is_res:
            base_dl1 = (heads["delta"] - tgt_y1).abs().mean()
            base_pv, _, _ = pose_and_vel_loss_from_delta(heads["delta"], b, j_idx, fk_gt)
            base_loss = 0.25 * (base_dl1 + base_pv)
            ah_for_flow = heads["ah"].detach()
            st_for_flow = heads["st_out"].detach()
        else:
            base_loss = 0.0
            ah_for_flow = heads["ah"]
            st_for_flow = heads["st_out"]

        # Draw K=4 independent (x0, t) flow samples per Transformer forward pass -> 16,000 flow updates in 4000 steps!
        K = 4
        y_rep = flow_target.repeat(K, 1, 1)
        ah_rep = ah_for_flow.repeat(K, 1, 1)
        st_rep = st_for_flow.repeat(K, 1)
        t_f = torch.sigmoid(torch.randn(K * bsz, device=device)).clamp(0.02, 0.98)
        x0 = sample_smooth_noise((K * bsz, 50, 12), device, dtype=tgt_y1.dtype, rho=0.85)
        xt = (1.0 - t_f[:, None, None]) * x0 + t_f[:, None, None] * y_rep
        v_target = y_rep - x0

        v_pred = model.flow_head(xt, t_f, ah_rep, st_rep)
        fm_l1 = (v_pred - v_target).abs().mean()

        # 1-step denoised full delta on first slice (0..bsz) for lightweight FK pose & velocity consistency
        x1_hat_0 = base_mu + xt[:bsz] + (1.0 - t_f[:bsz, None, None]) * v_pred[:bsz]
        pv_loss, pos_mm, vel_val = pose_and_vel_loss_from_delta(x1_hat_0, b, j_idx, fk_gt, sample_weights=(t_f[:bsz] ** 2))
        arm_loss = base_loss + fm_l1 + 0.5 * pv_loss
        pred_delta_for_world = x1_hat_0.detach()
        fm_l1_val = float(fm_l1.detach())
    else:
        raise ValueError(model.arm_mode)

    # Pass 2: World Head co-supervision
    mode = mode_for(globalstep, seed, (0.25, 0.25, 0.25, 0.25))
    zero = heads["observation"].sum() * 0.0
    world = zero
    cos = zero
    ad = zero
    cl = zero
    cw = 0.0

    if mode != "action_only":
        if mode == "clean":
            bw = b
        elif mode == "sync":
            pred_act = build_detached_action_from_delta(b, pred_delta_for_world, heads["grip_logits"], c_dict, j_idx)
            bw = dict(b, robot=trajectory(b["context"], pred_act))
        elif mode == "independent":
            bw = dict(b, robot=torch.zeros_like(b["robot"]), robot_valid=torch.zeros_like(b["robot_valid"]))
        else:
            raise ValueError(mode)

        out_w = model.forward_mean(bw)
        world = weighted_mean((out_w["mean"] - b["world"]).square().mean(-1), b["world_valid"], b["world_horizons"])
        cos = weighted_mean(1 - F.cosine_similarity(out_w["world_cos"], b["world"], dim=-1), b["world_valid"], b["world_horizons"], True)
        ad, _ = model.adapter_loss(b["obs"][:, 4])
        cw = chain_weight("P1", 5000 + globalstep, bool(model.adapter_ready) and chain is not None, 3000, 1000, 0.1)
        if cw:
            cl, _ = chain(model)

    total = arm_loss + grip_loss + 0.1 * (world + 0.1 * cos + 0.1 * ad + cw * cl) + 0.01 * align
    metrics = dict(
        mode=mode,
        arm_l1=fm_l1_val,
        pos_mm=pos_mm,
        vel_l1=vel_val,
        grip_bce=grip_bce_val,
        sw_aux=sw_val,
        world=float(world.detach()),
    )
    return total, metrics


def make_e134_model_and_opt(norm, arm, cp_e106, cp_g1=None):
    cfg = ARM_CONFIGS[arm]
    m = E134WAM(norm, arm_mode=cfg["arm_mode"])

    if cfg["init_ckpt"] == "G1":
        assert cp_g1 is not None
        m.widen_from_164()
        missing, unexpected = m.load_state_dict(cp_g1["model"], strict=False)
        assert not unexpected, f"Unexpected keys from G1: {unexpected}"
        assert all(k.startswith(("dec_grip.", "flow_head.")) for k in missing), f"Unexpected missing: {missing}"
    else:
        missing, unexpected = m.load_state_dict(cp_e106["model"], strict=False)
        assert not unexpected, f"Unexpected keys from E106C: {unexpected}"
        assert all(k.startswith(("dec_grip.", "flow_head.")) for k in missing), f"Unexpected missing: {missing}"
        m.widen_from_164()

    m = m.cuda()

    # Freeze unused heads cleanly so AdamW weight decay never touches dead params
    for n, p in m.named_parameters():
        if cfg["arm_mode"] != "fm_native" and n.startswith("action_head."):
            p.requires_grad_(False)
        elif cfg["arm_mode"] in ("fm_native", "fm_dit") and n.startswith("delta_head."):
            p.requires_grad_(False)
        elif n.startswith("grip_head."):
            # Base grip_head is kept as frozen anchor while dec_grip learns the decoupled refinement + switch CDF!
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    fast_prefixes = ("proprio_proj.", "dec_grip.", "flow_head.", "action_head.", "action_proj.", "ta.")
    fast_params, base_params, trainable_named = [], [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        trainable_named.append((n, p))
        if n.startswith(fast_prefixes):
            fast_params.append(p)
        else:
            base_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": base_params, "lr": 3e-4},
            {"params": fast_params, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    restored = 0
    if cfg["init_ckpt"] == "E106C":
        ref_m = UpgradedWAM(norm, variant="v1")
        old_names = [n for n, _ in ref_m.named_parameters()]
        del ref_m
        old_opt_state = cp_e106["optimizer"]["state"]
        name_to_old_state = {old_names[i]: old_opt_state[i] for i in range(len(old_names)) if i in old_opt_state}
        for n, p in trainable_named:
            if n in name_to_old_state:
                st = name_to_old_state[n]
                if "exp_avg" in st and st["exp_avg"].shape == p.shape:
                    opt.state[p] = {
                        k: (v.cpu() if (k == "step" and torch.is_tensor(v)) else (v.to(p.device) if torch.is_tensor(v) else v))
                        for k, v in st.items()
                    }
                    restored += 1
    return m, opt, restored


def run_smoke():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"VRAM low: {ginfo}"
    base.configure()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)

    _, _, tr, dev = datasets()
    _, _, p1, _, _ = load_data()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    norm_tr, _ = build_or_load_extra_stats(tr, dev, entries)
    cp_e106 = torch.load(INIT, map_location="cpu", weights_only=False)
    cp_g1 = torch.load(G1_CKPT, map_location="cpu", weights_only=False)
    c_dict = contract()
    j_idx = torch.as_tensor(JOINTS, device="cuda")

    hs = sample_horizons(22001, 90)
    b_230 = attach_extra_proprio(move(tr.batch([0, 1, 2, 3], hs), "cuda"), norm_tr[[0, 1, 2, 3]])
    pb = move(p1_only(p1.batch([0, 1, 2, 3], H_TUP, "instruction", "P1")), "cuda")

    for arm in ("D_dec", "FM_nat", "FM_dit", "FM_res"):
        m, opt, _ = make_e134_model_and_opt(tr.norm, arm, cp_e106, cp_g1)
        m.eval()
        for sm in ("zero", "stochastic", "mean8"):
            act = m.infer_e134(b_230, steps=8, sample_mode=sm, n_samples=8, seed=90)
            assert act.shape == (4, 50, 14) and torch.isfinite(act).all()

        m.train()
        sch_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
        for step_t in (22001, 22002):
            opt.zero_grad(set_to_none=True)
            v, _ = e134_joint_loss(m, b_230, step_t, 90, c_dict, j_idx, None)
            v.backward()
            pv = p1_loss(m, pb)
            (0.1 * pv).backward()
            assert torch.isfinite(v) and torch.isfinite(pv)
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            opt.step()
            sch_t.step()
        del m, opt

    peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    print(json.dumps(dict(smoke="passed", peak_mib=peak, gpu=ginfo)), flush=True)


def train(arm, seed=90, steps=4000):
    assert arm in ARM_CONFIGS
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"Insufficient VRAM: {ginfo}"

    tag = f"wam_{arm.lower()}_s{seed}"
    run = T / f"results/e134_{tag}_20261119_01"
    assert not run.exists(), f"{run} already exists"
    run.mkdir(parents=False)

    t0 = time.monotonic()
    ledger = dict(tag=tag, arm=arm, cfg=ARM_CONFIGS[arm], seed=seed, steps=steps, gpu=ginfo, status="running", started_brt=brt())
    (run / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    base.configure()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    np.random.seed(seed)

    _, _, tr, dev = datasets()
    _, _, p1, _, _ = load_data()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    norm_tr, _ = build_or_load_extra_stats(tr, dev, entries)
    c_dict = contract()
    j_idx = torch.as_tensor(JOINTS, device="cuda")

    d_expected_sha = None
    if seed == 90 and steps == 4000 and D_CKPT.exists():
        d_expected_sha = torch.load(D_CKPT, map_location="cpu", weights_only=False)["sample_stream_sha256"]

    cp_e106 = torch.load(INIT, map_location="cpu", weights_only=False)
    cp_g1 = torch.load(G1_CKPT, map_location="cpu", weights_only=False) if ARM_CONFIGS[arm]["init_ckpt"] == "G1" else None
    m, opt, restored = make_e134_model_and_opt(tr.norm, arm, cp_e106, cp_g1)
    del cp_e106, cp_g1
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    raws, _ = base.calibration_raws(tr)
    with torch.no_grad():
        av, ac = m.adapter_loss(raws[64:])
        m.adapter_ready.fill_(base.adapter_pass(av, ac))

    p1sampler = MultiSourceSampler(p1.rows, seed=10300 + seed, weights={"piper": 1.0}, caps={}, demo3_cap=0.45)

    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler = MultiSourceSampler(tr.rows, seed=seed, weights={"piper": 1.0}, caps={}, demo3_cap=0.45)
    stream = hashlib.sha256()

    with (run / "updates.jsonl").open("x") as log:
        for step in range(1, steps + 1):
            if step % 100 == 1:
                gf = query_gpu_free(vis)
                assert gf["free_mib"] >= 4096, f"Shared GPU VRAM pressure during {tag}: {gf}"

            globalstep = 22000 + step
            if step % 16 == 0:
                with torch.no_grad():
                    av, ac = m.adapter_loss(raws[64:])
                    m.adapter_ready.fill_(base.adapter_pass(av, ac))

            s_step = seed_for(seed, "E110F", step)
            torch.manual_seed(s_step)
            np.random.seed(s_step & 0xFFFFFFFF)
            ids = sampler.sample(32)
            hs = sample_horizons(globalstep, seed)
            stream.update(np.asarray(ids, dtype="<i8").tobytes())
            stream.update(np.asarray(hs, dtype="<i8").tobytes())
            b = attach_extra_proprio(move(tr.batch(ids, hs), "cuda"), norm_tr[ids])

            opt.zero_grad(set_to_none=True)
            m.train()

            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                ch = make_chain(tr, ids, (25, 50)[globalstep % 2], globalstep) if bool(m.adapter_ready) else None
                v, mt = e134_joint_loss(m, b, globalstep, seed, c_dict, j_idx, ch)
                assert torch.isfinite(v)
                v.backward()

                torch.manual_seed(seed_for(seed, "e129_p1", step))
                pids = p1sampler.sample(32)
                pb = p1_only(p1.batch(pids, H_TUP, "instruction", "P1"))
                pv = p1_loss(m, move(pb, "cuda"))
                assert torch.isfinite(pv)
                (0.1 * pv).backward()

            gn = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            assert torch.isfinite(gn)
            opt.step()
            sch.step()

            tot_l = float(v.detach() + 0.1 * pv.detach())
            if step == 1 or step % 50 == 0:
                rec = dict(step=step, loss=tot_l, lr=float(sch.get_last_lr()[0]), grad_norm=float(gn), elapsed_s=time.monotonic() - t0, p1_loss=float(pv.detach()), **mt)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % 500 == 0:
                print(
                    f"[{brt()}] {tag} step {step}/{steps} elapsed={time.monotonic()-t0:.1f}s "
                    f"arm={mt['arm_l1']:.4f} pos={mt['pos_mm']:.1f}mm grip={mt['grip_bce']:.4f}",
                    flush=True,
                )

    stream_hex = stream.hexdigest()
    if d_expected_sha is not None:
        assert stream_hex == d_expected_sha, f"Seed 90 stream mismatch vs D: {stream_hex} != {d_expected_sha}"

    ckpt_path = run / f"ckpt_{steps:05d}.pt"
    torch.save(
        dict(
            model={k: v.detach().cpu() for k, v in m.state_dict().items()},
            tag=tag,
            arm=arm,
            cfg=ARM_CONFIGS[arm],
            seed=seed,
            steps=steps,
            restored_opt_tensors=restored,
            sample_stream_sha256=stream_hex,
        ),
        ckpt_path,
    )
    elapsed = time.monotonic() - t0
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    ledger.update(status="completed", elapsed_s=elapsed, sec_per_step=elapsed / steps, peak_allocated_mib=peak_mib, sample_stream_sha256=stream_hex, finished_brt=brt())
    (run / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (run / "summary.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"[{brt()}] DONE {tag} in {elapsed:.1f}s (peak {peak_mib:.1f} MiB)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--smoke":
        run_smoke()
    else:
        arm = sys.argv[1]
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        st = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
        train(arm, seed=sd, steps=st)
