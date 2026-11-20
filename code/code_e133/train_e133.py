# E133 trainer: 4-arm controlled incremental ladder on exact D sample stream (seed 90, 4000 steps):
#   - G1: WAM_v1 (fast proprio lr=1e-3) + 3D fingertip pos & rot FK L1 + step-velocity slope L1
#   - G2: G1 + transition-weighted gripper BCE (6x close / 10x open +/-5f) + dedicated switch-frame CDF head
#   - G3: G2 + 15% current visual spatial token dropout
#   - G4: G2 + 20% current visual spatial token dropout + direct kin_skip residual highway
import os, sys, time, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from e133_common import (
    T, INIT, D_CKPT, LOCK_PATH, CTRL133, POS_IDX, ROT_IDX, brt,
    E133WAM, build_or_load_extra_stats, attach_extra_proprio,
)
sys.path.insert(0, str(T / "code_e129"))
from train_e129 import PEAK_MIB, H_TUP, query_gpu_free, p1_loss, build_detached_pred_action

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

ARM_SPECS = {
    "G1": dict(pose_vel_loss=True, grip_switch_loss=False, vis_token_drop=0.0, use_kin_skip=False),
    "G2": dict(pose_vel_loss=True, grip_switch_loss=True,  vis_token_drop=0.0, use_kin_skip=False),
    "G3": dict(pose_vel_loss=True, grip_switch_loss=True,  vis_token_drop=0.15, use_kin_skip=False),
    "G4": dict(pose_vel_loss=True, grip_switch_loss=True,  vis_token_drop=0.20, use_kin_skip=True),
}


def compute_switch_grip_loss(heads, b):
    g_cur = (b["grip_current"] > 0.5).float()        # (B, 2)
    g_lbl = b["grip_label"].float()                  # (B, 50, 2)
    g_seq = torch.cat([g_cur[:, None, :], g_lbl], dim=1)  # (B, 51, 2)
    diff_g = g_seq[:, 1:, :] - g_seq[:, :-1, :]      # (B, 50, 2)

    is_close = (diff_g < -0.5).float()               # 1 -> 0
    is_open = (diff_g > 0.5).float()                 # 0 -> 1

    # +/- 5 frame halo around true transitions
    close_halo = F.max_pool1d(is_close.permute(0, 2, 1), kernel_size=11, stride=1, padding=5).permute(0, 2, 1)
    open_halo = F.max_pool1d(is_open.permute(0, 2, 1), kernel_size=11, stride=1, padding=5).permute(0, 2, 1)
    hold_closed = (g_seq < 0.5).all(dim=1, keepdim=True).float().expand_as(g_lbl)

    w_grip = 1.0 + 0.5 * hold_closed + 5.0 * close_halo + 9.0 * open_halo
    bce_raw = F.binary_cross_entropy_with_logits(heads["grip_logits"], g_lbl, reduction="none")
    grip_w_bce = (bce_raw * w_grip).mean()

    # Dedicated switch-window & switch-frame supervision
    target_ev = torch.cat([is_close, is_open], dim=-1)  # (B, 50, 4)
    has_ev = (target_ev.sum(dim=1) > 0.5).float()       # (B, 4)
    win_logits = heads["switch_win_logits"]             # (B, 4)
    step_logits = heads["switch_step_logits"]           # (B, 50, 4)

    pos_w = win_logits.new_tensor([6.0, 6.0, 12.0, 12.0])
    loss_win = F.binary_cross_entropy_with_logits(win_logits, has_ev, pos_weight=pos_w)

    mask_pos = has_ev > 0.5
    if mask_pos.any():
        # Extract positive (batch, channel) tracks -> (M, 50)
        ev_tracks = target_ev.permute(0, 2, 1)[mask_pos]       # (M, 50)
        logit_tracks = step_logits.permute(0, 2, 1)[mask_pos]  # (M, 50)
        t_star = ev_tracks.argmax(dim=-1).float()              # (M,)
        grid_t = torch.arange(50, device=win_logits.device, dtype=torch.float32)[None, :]
        soft_tgt = torch.softmax(-0.5 * ((grid_t - t_star[:, None]) / 1.5) ** 2, dim=-1)
        loss_frame = -(soft_tgt * F.log_softmax(logit_tracks, dim=-1)).sum(dim=-1).mean()
    else:
        loss_frame = win_logits.sum() * 0.0

    sw_aux = 0.5 * loss_win + 0.25 * loss_frame
    return grip_w_bce + sw_aux, float(grip_w_bce.detach()), float(sw_aux.detach())


def e133_joint_loss(model, b, globalstep, seed, c_dict, j_idx, spec, chain=None):
    device = b["obs"].device
    n = len(b["obs"])
    t0 = torch.zeros(n, device=device)
    a0 = torch.zeros_like(b["action"])

    heads = model(b, a0, None, t0, t0)
    tgt = target_delta(b)
    delta_l1 = (heads["delta"] - tgt).abs().mean()

    # 1. Differentiable 3D fingertip position + orientation + step-velocity slope loss
    pred_q = b["q_current"][:, None].expand(-1, 50, -1).clone()
    pred_q[:, :, j_idx] = b["q_current"][:, None, j_idx] + heads["delta"] * b["action_std"][:, :, j_idx]
    fk_pred = pose_torch(pred_q)
    with torch.no_grad():
        fk_gt = pose_torch(b["action"])
    pos_l1_m = (fk_pred[..., POS_IDX] - fk_gt[..., POS_IDX]).abs().mean()
    rot_l1 = (fk_pred[..., ROT_IDX] - fk_gt[..., ROT_IDX]).abs().mean()
    pose_loss = 8.0 * pos_l1_m + 1.5 * rot_l1

    q0_j = b["q_current"][:, None, j_idx]
    v_pred = torch.cat([q0_j, pred_q[:, :, j_idx]], dim=1).diff(dim=1)
    v_gt = torch.cat([q0_j, b["action"][:, :, j_idx]], dim=1).diff(dim=1)
    v_scale = b["action_std"][:, 0:1, j_idx].clamp_min(1e-4)
    vel_l1 = ((v_pred - v_gt) / v_scale).abs().mean()

    # 2. Gripper loss (standard vs transition-weighted + dedicated switch-frame head)
    if spec["grip_switch_loss"]:
        grip_loss, grip_bce_val, sw_val = compute_switch_grip_loss(heads, b)
    else:
        grip_loss = F.binary_cross_entropy_with_logits(heads["grip_logits"], b["grip_label"])
        grip_bce_val = float(grip_loss.detach())
        sw_val = 0.0

    act_total = delta_l1 + pose_loss + 0.5 * vel_l1 + grip_loss
    align = contrastive(model, heads, b)

    # 3. Exposure-bias-free World Head (Pass 2 uses unmasked visual tokens!)
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
            pred_act = build_detached_pred_action(b, heads, c_dict, j_idx)
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

    total = act_total + 0.1 * (world + 0.1 * cos + 0.1 * ad + cw * cl) + 0.01 * align
    metrics = dict(
        mode=mode,
        delta_l1=float(delta_l1.detach()),
        pos_mm=float(pos_l1_m.detach() * 1000.0),
        rot_l1=float(rot_l1.detach()),
        vel_l1=float(vel_l1.detach()),
        grip_bce=grip_bce_val,
        sw_aux=sw_val,
        world=float(world.detach()),
    )
    return total, metrics


def make_e133_model_and_opt(norm, arm, cp):
    spec = ARM_SPECS[arm]
    m = E133WAM(
        norm,
        use_switch_head=spec["grip_switch_loss"],
        vis_token_drop=spec["vis_token_drop"],
        use_kin_skip=spec["use_kin_skip"],
    )
    missing, unexpected = m.load_state_dict(cp["model"], strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"
    assert all(k.startswith(("switch_", "kin_skip.")) for k in missing), f"Unexpected missing: {missing}"
    m.widen_from_164()
    m = m.cuda()

    for n, p in m.named_parameters():
        p.requires_grad_(not n.startswith("action_head."))

    ref_m = E133WAM(norm, use_switch_head=False, vis_token_drop=0.0, use_kin_skip=False)
    old_names = [n for n, _ in ref_m.named_parameters()]
    del ref_m
    old_opt_state = cp["optimizer"]["state"]
    name_to_old_state = {old_names[i]: old_opt_state[i] for i in range(len(old_names)) if i in old_opt_state}

    fast_prefixes = ("proprio_proj.", "switch_", "kin_skip.")
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
    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    c_dict = contract()
    j_idx = torch.as_tensor(JOINTS, device="cuda")

    hs = sample_horizons(22001, 90)
    b_orig = move(tr.batch([0, 1, 2, 3], hs), "cuda")
    b_230 = attach_extra_proprio(b_orig, norm_tr[[0, 1, 2, 3]])
    pb = move(p1_only(p1.batch([0, 1, 2, 3], H_TUP, "instruction", "P1")), "cuda")

    from e110f_core import infer_d
    from e103d_model import WorldModel as OrigModel
    m_orig = OrigModel(tr.norm)
    m_orig.load_state_dict(cp["model"], strict=True)
    m_orig = m_orig.cuda().eval()
    with torch.no_grad():
        a_ref = infer_d(m_orig, b_orig, 4, 90)
    del m_orig

    for arm in ("G1", "G2", "G3", "G4"):
        m, opt, _ = make_e133_model_and_opt(tr.norm, arm, cp)
        m.eval()
        with torch.no_grad():
            a_new = infer_d(m, b_230, 4, 90)
        max_diff = float((a_new - a_ref).abs().max().item())
        assert max_diff < 1e-6, f"Step-0 equivalence failed on {arm}: {max_diff}"

        m.train()
        sch_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
        for step_t in (22001, 22002, 22003, 22004):
            opt.zero_grad(set_to_none=True)
            v, _ = e133_joint_loss(m, b_230, step_t, 90, c_dict, j_idx, ARM_SPECS[arm], None)
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
    assert arm in ARM_SPECS
    spec = ARM_SPECS[arm]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"Insufficient VRAM: {ginfo}"

    tag = f"wam_{arm.lower()}_s{seed}"
    run = T / f"results/e133_{tag}_20261119_01"
    assert not run.exists(), f"{run} already exists"
    run.mkdir(parents=False)

    t0 = time.monotonic()
    ledger = dict(tag=tag, arm=arm, spec=spec, seed=seed, steps=steps, gpu=ginfo, status="running", started_brt=brt())
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

    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    m, opt, restored = make_e133_model_and_opt(tr.norm, arm, cp)
    del cp
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
                v, mt = e133_joint_loss(m, b, globalstep, seed, c_dict, j_idx, spec, ch)
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
                    f"dl1={mt['delta_l1']:.4f} pos={mt['pos_mm']:.1f}mm vel={mt['vel_l1']:.4f} grip={mt['grip_bce']:.4f}",
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
            spec=spec,
            seed=seed,
            steps=steps,
            restored_opt_tensors=restored,
            sample_stream_sha256=stream_hex,
            source=str(INIT),
        ),
        ckpt_path,
    )
    elapsed = time.monotonic() - t0
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    ledger.update(status="completed", elapsed_s=elapsed, sec_per_step=elapsed / steps, peak_allocated_mib=peak_mib, sample_stream_sha256=stream_hex, finished_brt=brt())
    (run / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (run / "summary.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"[{brt()}] DONE {tag} in {elapsed:.1f}s (peak {peak_mib:.1f} MiB, stream={stream_hex[:12]})", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--smoke":
        run_smoke()
    else:
        arm = sys.argv[1]
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        st = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
        train(arm, seed=sd, steps=st)
