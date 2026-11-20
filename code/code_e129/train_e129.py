# E129 single-arm trainer: Upgraded end-to-end WAM with (t-1, t-3) velocity, unweighted L1 action loss,
# Cosine LR, and exposure-bias-free World Head (25% gt + 25% detached pred N1 + 25% no_traj + 25% action_only).
import os, sys, time, json, hashlib, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn.functional as F

from e129_common import (
    T, INIT, D_CKPT, LOCK_PATH, CTRL, brt,
    UpgradedWAM, build_or_load_extra_stats, attach_extra_proprio,
)
from e103d_data import p1_only
from e102x_setup import datasets
from e102a_contract import contract
from e99w2_model import JOINTS, trajectory
from e90.data import load_data, sample_horizons
from e90.sampling import MultiSourceSampler
from e90.objective import weighted_mean, chain_weight
from e86.common import seed_for
from e86.data import move
from e86.objective import mode_for, contrastive
from e96a import train as base
from e99w2_train import make_chain
from e110f_core import target_delta

PEAK_MIB = 6144
H_TUP = (10, 25, 50, 100)


def query_gpu_free(vis):
    s = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in s.strip().splitlines():
        idx, uuid, free, util = [x.strip() for x in line.split(",")]
        if idx == vis or uuid == vis:
            return dict(index=idx, uuid=uuid, free_mib=int(free), util=int(util))
    raise RuntimeError(f"GPU {vis} not found")


def p1_loss(m, b):
    o = m.forward_mean(b)
    w = weighted_mean((o["mean"] - b["world"]).square().mean(-1), b["world_valid"], b["world_horizons"])
    c = weighted_mean(1 - F.cosine_similarity(o["world_cos"], b["world"], dim=-1), b["world_valid"], b["world_horizons"], True)
    ad, _ = m.adapter_loss(b["obs"][:, 4])
    return w + 0.1 * c + 0.1 * ad


def build_detached_pred_action(b, heads, c_dict, j_idx):
    # Reconstruct 50x14 predicted action from Pass 1 heads without extra forward pass (strictly detached)
    delta_det = heads["delta"].detach()
    grip_det = heads["grip_logits"].detach()
    act = b["q_current"][:, None].expand(-1, 50, -1).clone()
    act[:, :, j_idx] = b["q_current"][:, None, j_idx] + delta_det * b["action_std"][:, :, j_idx]
    act[:, :, [6, 13]] = torch.where(
        grip_det > 0,
        act.new_tensor(c_dict["open_m"]),
        act.new_tensor(c_dict["closed_m"]),
    )
    return act


def e129_joint_loss(model, b, globalstep, seed, c_dict, j_idx, chain=None):
    device = b["obs"].device
    n = len(b["obs"])
    t0 = torch.zeros(n, device=device)
    a0 = torch.zeros_like(b["action"])

    # Pass 1: Deterministic action prediction with unweighted standardized L1 + gripper BCE
    heads = model(b, a0, None, t0, t0)
    tgt = target_delta(b)
    delta_l1 = (heads["delta"] - tgt).abs().mean()
    grip_bce = F.binary_cross_entropy_with_logits(heads["grip_logits"], b["grip_label"])
    act_total = delta_l1 + grip_bce
    align = contrastive(model, heads, b)

    # Pass 2: Exposure-bias-free World Head (25% clean gt, 25% sync=detached pred N1, 25% independent=no_traj, 25% action_only)
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
        cos = weighted_mean(
            1 - F.cosine_similarity(out_w["world_cos"], b["world"], dim=-1),
            b["world_valid"],
            b["world_horizons"],
            True,
        )
        ad, _ = model.adapter_loss(b["obs"][:, 4])
        cw = chain_weight("P1", 5000 + globalstep, bool(model.adapter_ready) and chain is not None, 3000, 1000, 0.1)
        if cw:
            cl, _ = chain(model)

    total = act_total + 0.1 * (world + 0.1 * cos + 0.1 * ad + cw * cl) + 0.01 * align
    metrics = dict(
        mode=mode,
        delta_l1=float(delta_l1.detach()),
        grip_bce=float(grip_bce.detach()),
        world=float(world.detach()),
        cosine=float(cos.detach()),
        adapter=float(ad.detach()),
        chain=float(cl.detach()),
    )
    return total, metrics


def make_model_and_opt(norm, variant, cp, lr_base=3e-4):
    m = UpgradedWAM(norm, variant=variant)
    # Load E106-C 164-D weights first (ignore missing kin_* keys on v2)
    missing, unexpected = m.load_state_dict(cp["model"], strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"
    assert all(k.startswith("kin_") for k in missing), f"Unexpected missing keys: {missing}"
    # Now widen proprio_proj[0] from 164 -> 230 (new 66 cols zero-initialized -> step-0 output bitwise identical to E106-C!)
    m.widen_from_164()
    m = m.cuda()

    #Freeze unused legacy flow-matching action_head
    for n, p in m.named_parameters():
        p.requires_grad_(not n.startswith("action_head."))

    # Build AdamW: restore momentum/variance from cp['optimizer'] for all unchanged-shape tensors; fresh state for widened/new tensors
    # Map old parameter index in E106-C to parameter name
    ref_m = UpgradedWAM(norm, variant="v1")  # before widen, exact E106-C parameter order!
    old_names = [n for n, _ in ref_m.named_parameters()]
    del ref_m
    old_opt_state = cp["optimizer"]["state"]
    name_to_old_state = {old_names[i]: old_opt_state[i] for i in range(len(old_names)) if i in old_opt_state}

    trainable_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _, p in trainable_params], lr=lr_base, weight_decay=1e-4)
    restored = 0
    for n, p in trainable_params:
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
    norm_tr, norm_dev = build_or_load_extra_stats(tr, dev, entries)
    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    c_dict = contract()
    j_idx = torch.as_tensor(JOINTS, device="cuda")

    hs = sample_horizons(22001, 90)
    b_orig = move(tr.batch([0, 1, 2, 3], hs), "cuda")
    b_230 = attach_extra_proprio(b_orig, norm_tr[[0, 1, 2, 3]])
    pb = move(p1_only(p1.batch([0, 1, 2, 3], H_TUP, "instruction", "P1")), "cuda")

    # Verify step-0 bitwise equivalence with E106-C and test all 4 modes!
    from e110f_core import infer_d
    from e103d_model import WorldModel as OrigModel
    m_orig = OrigModel(tr.norm)
    m_orig.load_state_dict(cp["model"], strict=True)
    m_orig = m_orig.cuda().eval()
    with torch.no_grad():
        a_ref = infer_d(m_orig, b_orig, 4, 90)
    del m_orig

    for var in ("v1", "v2"):
        m, opt, restored = make_model_and_opt(tr.norm, var, cp)
        m.eval()
        with torch.no_grad():
            a_new = infer_d(m, b_230, 4, 90)
        max_diff = float((a_new - a_ref).abs().max().item())
        assert max_diff < 1e-6, f"Step-0 equivalence failed on {var}: {max_diff}"

        m.train()
        sch_t = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
        for step_t in (22001, 22002, 22003, 22004):
            opt.zero_grad(set_to_none=True)
            v, _ = e129_joint_loss(m, b_230, step_t, 90, c_dict, j_idx, None)
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


def train(variant, seed=90, steps=4000):
    assert variant in ("v1", "v2")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"Insufficient VRAM: {ginfo}"

    tag = f"wam_{variant}_s{seed}"
    run = T / f"results/e129_{tag}_20261119_01"
    assert not run.exists(), f"{run} already exists"
    run.mkdir(parents=False)

    t0 = time.monotonic()
    ledger = dict(tag=tag, variant=variant, seed=seed, steps=steps, gpu=ginfo, status="running", started_brt=brt())
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
    m, opt, restored = make_model_and_opt(tr.norm, variant, cp, lr_base=3e-4)
    del cp
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    raws, _ = base.calibration_raws(tr)
    with torch.no_grad():
        av, ac = m.adapter_loss(raws[64:])
        m.adapter_ready.fill_(base.adapter_pass(av, ac))

    p1sampler = MultiSourceSampler(p1.rows, seed=10300 + seed, weights={"piper": 1.0}, caps={}, demo3_cap=0.45)

    # Reset RNG before P3 sampler so seed=90 matches D's exact P3 sample stream!
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
                v, mt = e129_joint_loss(m, b, globalstep, seed, c_dict, j_idx, ch)
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
                print(f"[{brt()}] {tag} step {step}/{steps} elapsed={time.monotonic()-t0:.1f}s dl1={mt['delta_l1']:.4f} grip={mt['grip_bce']:.4f} w={mt['world']:.4f}", flush=True)

    stream_hex = stream.hexdigest()
    if d_expected_sha is not None:
        assert stream_hex == d_expected_sha, f"Seed 90 stream mismatch vs D: {stream_hex} != {d_expected_sha}"

    ckpt_path = run / f"ckpt_{steps:05d}.pt"
    torch.save(
        dict(
            model={k: v.detach().cpu() for k, v in m.state_dict().items()},
            tag=tag,
            variant=variant,
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
        var = sys.argv[1]
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        st = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
        train(var, seed=sd, steps=st)
