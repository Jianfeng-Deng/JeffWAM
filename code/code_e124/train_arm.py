# E124 (P1-v2) single-arm trainer: Unfrozen backbone +/- 50% shortcut block dropout.
import os, sys, time, json, hashlib, datetime, threading, subprocess
from pathlib import Path
import numpy as np
import torch

os.environ.update(
    E102_GROUP="A2",
    PYTHONDONTWRITEBYTECODE="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    OMP_NUM_THREADS="4",
)

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
for p in ("code_e115p", "code_e110f", "code_e106c", "code_e102a"):
    sys.path.insert(0, str(T / p))

from e99w2_common import GROUP
assert GROUP == "A2"
from common import INIT, FEATS, widen, add_stage
from e99w2_data import load
from e99w2_model import WorldModel
from e90.sampling import MultiSourceSampler
from e90.data import sample_horizons
from e86.common import seed_for
from e86.data import move
from e110f_core import tensor_sha, deterministic_loss

EXPECTED_P1_STREAM_SHA = "f4c1a7c2ae74e8c4fa221cdb884179a33b71c79e5ca8eb31b9d49b4773805946"
EXCLUDE_CID = "piper_4dc107335389f5711a0f65e0"
PEAK_MIB = 8192

UNFROZEN_PREFIXES = (
    "current.", "pool.", "camera", "obs_age.", "hist_time", "history_camera",
    "text_proj.", "null_text", "kind",
    "state_proj.", "action_proj.", "a_pos", "ta.", "proprio_proj.",
    "blocks.", "norm.", "delta_head.", "grip_head.",
)

ARM_SPECS = {
    "U0": dict(kind="zero", drop=0.0),
    "U1": dict(kind="ctrl", drop=0.0),
    "U2": dict(kind="oracle", drop=0.0),
    "D0": dict(kind="zero", drop=0.5),
    "D1": dict(kind="ctrl", drop=0.5),
    "D2": dict(kind="oracle", drop=0.5),
}


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def set_unfrozen_action_trainables(model):
    names = []
    for name, p in model.named_parameters():
        ok = any(name == pref or name.startswith(pref) for pref in UNFROZEN_PREFIXES)
        p.requires_grad_(ok)
        if ok:
            names.append(name)
    assert any(n.startswith("blocks.0.") for n in names) and any(n.startswith("blocks.5.") for n in names)
    assert any(n.startswith("proprio_proj.") for n in names) and any(n.startswith("delta_head.") for n in names)
    return tuple(names)


def apply_forced_dropout(b, step, p_drop):
    if p_drop <= 0.0:
        return b
    dev = b["obs"].device
    g = torch.Generator(device=dev).manual_seed(seed_for(90, "E124_DROP", step))
    B = len(b["obs"])
    keep_cam = torch.rand((B, 3), device=dev, generator=g) >= p_drop
    keep_hist = torch.rand((B, 1), device=dev, generator=g) >= p_drop
    keep_prop = (torch.rand((B, 1), device=dev, generator=g) >= p_drop).to(b["proprio"].dtype)

    out = dict(b)
    out["current_valid"] = b["current_valid"] & keep_cam
    out["history_valid"] = b["history_valid"] & keep_hist

    obs = b["obs"].clone()
    obs[:, 4] = obs[:, 4] * keep_cam[:, :, None, None]
    obs[:, :4] = obs[:, :4] * keep_hist[:, :, None, None, None]
    out["obs"] = obs

    prop = b["proprio"].clone()
    prop[:, :164] = prop[:, :164] * keep_prop
    out["proprio"] = prop
    out["q_normalized"] = b["q_normalized"] * keep_prop.to(b["q_normalized"].dtype)
    return out


def query_gpu_free(vis):
    s = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in s.strip().splitlines():
        idx, uuid, free, util = [x.strip() for x in line.split(",")]
        if idx == vis or uuid == vis:
            return dict(index=idx, uuid=uuid, free_mib=int(free), util=int(util))
    raise RuntimeError(f"GPU {vis} not found in nvidia-smi")


def run_smoke_test():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"VRAM insufficient: {ginfo}"
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(90)
    np.random.seed(90)
    _, _, tr, _ = load()
    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    model = widen(WorldModel(tr.norm)).cuda()
    cp_mod = dict(cp["model"])
    # widen already loaded AFTER load_state_dict in actual run; do exact order here:
    model = WorldModel(tr.norm)
    model.load_state_dict(cp_mod, strict=True)
    model = widen(model).cuda()
    trainable = set_unfrozen_action_trainables(model)
    avail = {p.stem for p in FEATS.glob("pc_*.npz")}
    sub = [i for i, r in enumerate(tr.rows) if r["parent_component"] in avail and r["canonical_id"] != EXCLUDE_CID]
    assert len(sub) == 13172 and len({tr.rows[i]["parent_component"] for i in sub}) == 45
    hs = sample_horizons(22001, 90)
    b = move(tr.batch(sub[:8], hs), "cuda")
    b = add_stage(b, [tr.rows[i] for i in sub[:8]], "oracle")
    # Test extreme p_drop=1.0 (all cameras + history + proprio dropped!) for zero NaNs!
    for p_test in (0.0, 0.5, 1.0):
        model.zero_grad(set_to_none=True)
        model.train()
        bd = apply_forced_dropout(b, 1, p_test)
        loss, parts = deterministic_loss(model, bd)
        assert torch.isfinite(loss), f"Non-finite loss at p_drop={p_test}"
        loss.backward()
        grads_ok = all(
            (p.grad is None or torch.isfinite(p.grad).all())
            for _, p in model.named_parameters() if p.requires_grad
        )
        assert grads_ok, f"Non-finite grad at p_drop={p_test}"
    peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    print(json.dumps(dict(smoke="passed", trainable_tensors=len(trainable), peak_mib=peak, gpu=ginfo)), flush=True)


def train_arm(arm, steps=4000):
    spec = ARM_SPECS[arm]
    kind = spec["kind"]
    p_drop = spec["drop"]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"VRAM insufficient before {arm}: {ginfo}"

    run = T / f"results/e124_p1v2_{arm.lower()}_20261118_01"
    assert not run.exists(), f"{run} already exists"
    run.mkdir(parents=False)

    t0 = time.monotonic()
    ledger = dict(arm=arm, kind=kind, p_drop=p_drop, steps=steps, gpu=ginfo, peak_estimate_mib=PEAK_MIB, status="running", started_brt=brt())
    (run / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(90)
    np.random.seed(90)

    _, _, tr, _ = load()
    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    model = WorldModel(tr.norm)
    model.load_state_dict(cp["model"], strict=True)
    model = widen(model).cuda()
    del cp

    avail = {p.stem for p in FEATS.glob("pc_*.npz")}
    sub = [i for i, r in enumerate(tr.rows) if r["parent_component"] in avail and r["canonical_id"] != EXCLUDE_CID]
    assert len(sub) == 13172

    trainable = set_unfrozen_action_trainables(model)
    frozen = tuple(n for n, _ in model.named_parameters() if n not in trainable)
    fsha = tensor_sha(model, frozen)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    sampler = MultiSourceSampler([tr.rows[i] for i in sub], seed=90, weights={"piper": 1.0}, caps={}, demo3_cap=0.45)
    stream = hashlib.sha256()

    with (run / "updates.jsonl").open("x") as log:
        for step in range(1, steps + 1):
            if step % 100 == 1:
                gf = query_gpu_free(vis)
                assert gf["free_mib"] >= 4096, f"Shared GPU memory pressure during {arm}: {gf}"
            seed = seed_for(90, "E115P1", step)
            torch.manual_seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            ids = sampler.sample(32)
            orig = [sub[i] for i in ids]
            hs = sample_horizons(22000 + step, 90)
            stream.update(np.asarray(orig, dtype="<i8").tobytes())

            b = move(tr.batch(orig, hs), "cuda")
            b = add_stage(b, [tr.rows[i] for i in orig], kind)
            b = apply_forced_dropout(b, step, p_drop)

            opt.zero_grad(set_to_none=True)
            model.train()
            loss, parts = deterministic_loss(model, b)
            assert torch.isfinite(loss), f"Nonfinite loss on {arm} step {step}"
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            assert torch.isfinite(gn), f"Nonfinite grad norm on {arm} step {step}"
            opt.step()

            if step == 1 or step % 50 == 0:
                rec = dict(
                    step=step,
                    loss=float(loss.detach()),
                    delta=float(parts["delta"]),
                    grip=float(parts["grip"]),
                    grad_norm=float(gn),
                    elapsed_s=time.monotonic() - t0,
                )
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % 500 == 0:
                print(f"[{brt()}] {arm} step {step}/{steps} elapsed={time.monotonic()-t0:.1f}s delta={float(parts['delta']):.4f} grip={float(parts['grip']):.4f}", flush=True)

    assert tensor_sha(model, frozen) == fsha, "Frozen W parameters modified!"
    stream_hex = stream.hexdigest()
    if steps == 4000:
        assert stream_hex == EXPECTED_P1_STREAM_SHA, f"Sample stream mismatch: {stream_hex} != {EXPECTED_P1_STREAM_SHA}"

    ckpt_path = run / f"ckpt_{steps:05d}.pt"
    torch.save(
        dict(
            model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            arm=arm,
            kind=kind,
            p_drop=p_drop,
            steps=steps,
            trainable_count=len(trainable),
            sample_stream_sha256=stream_hex,
            subset_rows=len(sub),
            source=str(INIT),
        ),
        ckpt_path,
    )
    elapsed = time.monotonic() - t0
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    ledger.update(status="completed", elapsed_s=elapsed, gpu_hours=elapsed / 3600.0, peak_allocated_mib=peak_mib, sample_stream_sha256=stream_hex, finished_brt=brt())
    (run / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (run / "summary.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"[{brt()}] DONE {arm} in {elapsed:.1f}s (peak {peak_mib:.1f} MiB, stream={stream_hex[:12]})", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "--smoke":
        run_smoke_test()
    else:
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
        train_arm(arg, steps=steps)
