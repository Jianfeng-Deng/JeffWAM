# E125 trainer: Full 145-parent pool paired comparison (U_full vs J_legacy vs J_clean, exact E110-F D sample stream).
import os, sys, time, json, hashlib, datetime, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

os.environ.update(
    E102_GROUP="A2",
    PYTHONDONTWRITEBYTECODE="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    OMP_NUM_THREADS="4",
)

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for p in ("code_e110f", "code_e106c/run02", "code_e106c", "code_e102a"):
    sys.path.insert(0, str(T / p))

from e99w2_common import GROUP
assert GROUP == "A2"
from e103d_model import WorldModel
from e103d_data import p1_only
from e99w2_data import load
from e90.data import load_data, sample_horizons
from e90.sampling import MultiSourceSampler
from e90.objective import weighted_mean, chain_weight
from e86.common import seed_for
from e86.data import move
from e86.objective import mode_for, contrastive
from e96a import train as base
from e99w2_train import make_chain
from e99w2_objective import loss as legacy_joint_loss
from e110f_core import tensor_sha, target_delta, weights as time_weights, deterministic_loss

INIT = T / "results/e106c_f2_s90_20261115_02/ckpt_20000.pt"
D_CKPT = T / "results/e110f_d_20261116_05/ckpt_04000.pt"
PEAK_MIB = 8192
H_TUP = (10, 25, 50, 100)

UNFROZEN_ACTION_PREFIXES = (
    "current.", "pool.", "camera", "obs_age.", "hist_time", "history_camera",
    "text_proj.", "null_text", "kind",
    "state_proj.", "action_proj.", "a_pos", "ta.", "proprio_proj.",
    "blocks.", "norm.", "delta_head.", "grip_head.",
)


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


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


def clean_joint_loss(model, b, globalstep, chain=None):
    device = b["obs"].device
    n = len(b["obs"])
    t0 = torch.zeros(n, device=device)
    a0 = torch.zeros_like(b["action"])

    # Pass 1: Clean deterministic action head (no Flow-Matching branch!)
    heads = model(b, a0, None, t0, t0)
    tgt = target_delta(b)
    wt = time_weights(tgt.device, tgt.dtype)
    delta = ((heads["delta"] - tgt).square() * wt).sum() / (wt.sum() * tgt.shape[-1] * n)
    grip = F.binary_cross_entropy_with_logits(heads["grip_logits"], b["grip_label"])
    act_total = delta + grip
    align = contrastive(model, heads, b)

    # Pass 2: Clean world head (physical gt trajectory or no_traj; zero Gaussian-noisy FK trajectories!)
    mode = mode_for(globalstep, 90, (0.25, 0.25, 0.25, 0.25))
    zero = heads["observation"].sum() * 0.0
    world = zero
    cos = zero
    ad = zero
    cl = zero
    cw = 0.0

    if mode != "action_only":
        if mode == "independent":
            bw = dict(b, robot=torch.zeros_like(b["robot"]), robot_valid=torch.zeros_like(b["robot_valid"]))
        else:
            # 'clean' and 'sync' both use clean physical gt trajectory
            bw = b
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
        delta_mse=float(delta.detach()),
        grip_bce=float(grip.detach()),
        world=float(world.detach()),
        cosine=float(cos.detach()),
        adapter=float(ad.detach()),
        chain=float(cl.detach()),
        chain_weight=float(cw),
        alignment=float(align.detach()),
    )
    return total, metrics


def set_u_full_trainables(model):
    names = []
    for name, p in model.named_parameters():
        ok = any(name == pref or name.startswith(pref) for pref in UNFROZEN_ACTION_PREFIXES)
        p.requires_grad_(ok)
        if ok:
            names.append(name)
    assert any(n.startswith("blocks.0.") for n in names) and any(n.startswith("blocks.5.") for n in names)
    return tuple(names)


def run_smoke():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"Insufficient free VRAM: {ginfo}"
    base.configure()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    _, _, tr, _ = load()
    _, _, p1, _, _ = load_data()
    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    d_meta = torch.load(D_CKPT, map_location="cpu", weights_only=False)
    d_sha = d_meta["sample_stream_sha256"]
    del d_meta

    m = WorldModel(tr.norm)
    m.load_state_dict(cp["model"], strict=True)
    m = m.cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    opt.load_state_dict(cp["optimizer"])
    del cp

    hs = sample_horizons(22001, 90)
    b = move(tr.batch([0, 1, 2, 3], hs), "cuda")
    pb = move(p1_only(p1.batch([0, 1, 2, 3], H_TUP, "instruction", "P1")), "cuda")

    # Test all 3 loss paths (U_full, J_legacy, J_clean)
    for arm_test in ("U_full", "J_legacy", "J_clean"):
        for _, p in m.named_parameters():
            p.requires_grad_(True)
        if arm_test == "U_full":
            set_u_full_trainables(m)
        m.zero_grad(set_to_none=True)
        m.train()
        if arm_test == "U_full":
            v, _ = deterministic_loss(m, b)
            v.backward()
        elif arm_test == "J_legacy":
            v, _ = legacy_joint_loss(m, b, 22001, None)
            v.backward()
            pv = p1_loss(m, pb)
            (0.1 * pv).backward()
        else:
            v, _ = clean_joint_loss(m, b, 22001, None)
            v.backward()
            pv = p1_loss(m, pb)
            (0.1 * pv).backward()
        assert torch.isfinite(v)
    peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    print(json.dumps(dict(smoke="passed", d_stream_sha=d_sha[:16], peak_mib=peak, gpu=ginfo)), flush=True)


def train(arm, steps=4000):
    assert arm in ("U_full", "J_legacy", "J_clean")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"Insufficient VRAM before {arm}: {ginfo}"

    run = T / f"results/e125_{arm.lower()}_20261118_01"
    assert not run.exists(), f"{run} already exists"
    run.mkdir(parents=False)

    t0 = time.monotonic()
    ledger = dict(arm=arm, steps=steps, gpu=ginfo, peak_estimate_mib=PEAK_MIB, status="running", started_brt=brt())
    (run / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    base.configure()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(90)
    np.random.seed(90)

    _, _, tr, _ = load()
    assert "piper_4dc107335389f5711a0f65e0" not in {r["canonical_id"] for r in tr.rows}
    is_joint = arm in ("J_legacy", "J_clean")
    p1 = None
    p1sampler = None
    raws = None
    if is_joint:
        _, _, p1, _, _ = load_data()
        p1sampler = MultiSourceSampler(p1.rows, seed=10390, weights={"piper": 1.0}, caps={}, demo3_cap=0.45)
        raws, _ = base.calibration_raws(tr)

    d_expected_sha = None
    if steps == 4000 and D_CKPT.exists():
        d_expected_sha = torch.load(D_CKPT, map_location="cpu", weights_only=False)["sample_stream_sha256"]

    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    m = WorldModel(tr.norm)
    m.load_state_dict(cp["model"], strict=True)
    m = m.cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    opt.load_state_dict(cp["optimizer"])
    del cp

    frozen = ()
    frozen_sha = None
    if arm == "U_full":
        trainable = set_u_full_trainables(m)
        frozen = tuple(n for n, _ in m.named_parameters() if n not in trainable)
        frozen_sha = tensor_sha(m, frozen)
    else:
        for _, p in m.named_parameters():
            p.requires_grad_(True)
        if arm == "J_clean":
            # Dead legacy flow-matching head is unused in J_clean; freeze explicitly so AdamW weight_decay doesn't drift it
            for n, p in m.named_parameters():
                if n.startswith("action_head."):
                    p.requires_grad_(False)
        trainable = tuple(n for n, p in m.named_parameters() if p.requires_grad)
        with torch.no_grad():
            av, ac = m.adapter_loss(raws[64:])
            m.adapter_ready.fill_(base.adapter_pass(av, ac))

    # Reset RNG right before sampler creation to match E110-F D exactly
    torch.manual_seed(90)
    np.random.seed(90)
    sampler = MultiSourceSampler(tr.rows, seed=90, weights={"piper": 1.0}, caps={}, demo3_cap=0.45)
    stream = hashlib.sha256()

    with (run / "updates.jsonl").open("x") as log:
        for step in range(1, steps + 1):
            if step % 100 == 1:
                gf = query_gpu_free(vis)
                assert gf["free_mib"] >= 4096, f"Shared VRAM pressure during {arm}: {gf}"

            globalstep = 22000 + step
            if is_joint and step % 16 == 0:
                with torch.no_grad():
                    av, ac = m.adapter_loss(raws[64:])
                    m.adapter_ready.fill_(base.adapter_pass(av, ac))

            # Exact E110-F D per-step RNG and sample stream!
            seed = seed_for(90, "E110F", step)
            torch.manual_seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            ids = sampler.sample(32)
            hs = sample_horizons(globalstep, 90)
            stream.update(np.asarray(ids, dtype="<i8").tobytes())
            stream.update(np.asarray(hs, dtype="<i8").tobytes())
            b = move(tr.batch(ids, hs), "cuda")

            opt.zero_grad(set_to_none=True)
            m.train()

            if arm == "U_full":
                v, parts = deterministic_loss(m, b)
                assert torch.isfinite(v)
                v.backward()
                assert all(p.grad is None for n, p in m.named_parameters() if n in frozen)
                rec_loss = float(v.detach())
                rec_parts = {k: float(val) for k, val in parts.items()}
            else:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    ch = make_chain(tr, ids, (25, 50)[globalstep % 2], globalstep) if bool(m.adapter_ready) else None
                    if arm == "J_legacy":
                        v, mt = legacy_joint_loss(m, b, globalstep, ch)
                    else:
                        v, mt = clean_joint_loss(m, b, globalstep, ch)
                    assert torch.isfinite(v)
                    v.backward()

                    torch.manual_seed(seed_for(90, "e125_p1", step))
                    pids = p1sampler.sample(32)
                    pb = p1_only(p1.batch(pids, H_TUP, "instruction", "P1"))
                    pv = p1_loss(m, move(pb, "cuda"))
                    assert torch.isfinite(pv)
                    (0.1 * pv).backward()

                rec_loss = float(v.detach() + 0.1 * pv.detach())
                rec_parts = dict(p1_loss=float(pv.detach()), **mt)

            gn = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            assert torch.isfinite(gn)
            opt.step()

            if step == 1 or step % 50 == 0:
                rec = dict(step=step, loss=rec_loss, grad_norm=float(gn), elapsed_s=time.monotonic() - t0, **rec_parts)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % 500 == 0:
                print(f"[{brt()}] {arm} step {step}/{steps} elapsed={time.monotonic()-t0:.1f}s loss={rec_loss:.4f}", flush=True)

    if arm == "U_full":
        assert tensor_sha(m, frozen) == frozen_sha, "Frozen W-only params changed in U_full!"

    stream_hex = stream.hexdigest()
    if d_expected_sha is not None:
        assert stream_hex == d_expected_sha, f"Stream SHA mismatch vs D: {stream_hex} != {d_expected_sha}"

    ckpt_path = run / f"ckpt_{steps:05d}.pt"
    torch.save(
        dict(
            model={k: v.detach().cpu() for k, v in m.state_dict().items()},
            arm=arm,
            steps=steps,
            trainable_count=len(trainable),
            sample_stream_sha256=stream_hex,
            matched_d_stream=(stream_hex == d_expected_sha),
            source=str(INIT),
        ),
        ckpt_path,
    )
    elapsed = time.monotonic() - t0
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    ledger.update(
        status="completed",
        elapsed_s=elapsed,
        sec_per_step=elapsed / steps,
        gpu_hours=elapsed / 3600.0,
        peak_allocated_mib=peak_mib,
        sample_stream_sha256=stream_hex,
        matched_d_stream=(stream_hex == d_expected_sha),
        finished_brt=brt(),
    )
    (run / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (run / "summary.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"[{brt()}] DONE {arm} in {elapsed:.1f}s ({elapsed/steps:.3f}s/step, peak {peak_mib:.1f} MiB, stream={stream_hex[:12]})", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "--smoke":
        run_smoke()
    else:
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
        train(arg, steps=steps)
