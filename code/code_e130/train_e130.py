# E130 trainer: Clean-state WAM ablations on exact D sample stream (seed 90, 4000 steps):
#   - N0: exact WAM_v1_s90 + clean_state (state_noise=0, state_drop=0)
#   - N1: N0 + fast LR (1e-3) on proprio_proj & state_proj (testing convergence of new t-1/t-3 velocity weights)
#   - N2: N1 + zero-init direct kinematic skip [q_normalized, proprio_230] -> 50x12 co-trained end-to-end with WAM
import os, sys, time, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from e130_common import (
    T, INIT, D_CKPT, LOCK_PATH, CTRL130, brt,
    E130WAM, build_or_load_extra_stats, attach_extra_proprio,
)
sys.path.insert(0, str(T / "code_e129"))
from train_e129 import PEAK_MIB, H_TUP, query_gpu_free, p1_loss, e129_joint_loss

from e103d_data import p1_only
from e102x_setup import datasets
from e102a_contract import contract
from e99w2_model import JOINTS
from e90.data import load_data, sample_horizons
from e90.sampling import MultiSourceSampler
from e86.common import seed_for
from e86.data import move
from e96a import train as base
from e99w2_train import make_chain

ARM_SPECS = {
    "N0": dict(clean_state=True, fast_proprio_lr=False, use_kin_skip=False),
    "N1": dict(clean_state=True, fast_proprio_lr=True, use_kin_skip=False),
    "N2": dict(clean_state=True, fast_proprio_lr=True, use_kin_skip=True),
}


def make_e130_model_and_opt(norm, arm, cp):
    spec = ARM_SPECS[arm]
    m = E130WAM(norm, clean_state=spec["clean_state"], use_kin_skip=spec["use_kin_skip"])
    missing, unexpected = m.load_state_dict(cp["model"], strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"
    assert all(k.startswith("kin_skip.") for k in missing), f"Unexpected missing: {missing}"
    m.widen_from_164()
    m = m.cuda()

    for n, p in m.named_parameters():
        p.requires_grad_(not n.startswith("action_head."))

    ref_m = E130WAM(norm, clean_state=True, use_kin_skip=False)
    old_names = [n for n, _ in ref_m.named_parameters()]
    del ref_m
    old_opt_state = cp["optimizer"]["state"]
    name_to_old_state = {old_names[i]: old_opt_state[i] for i in range(len(old_names)) if i in old_opt_state}

    fast_prefixes = ("proprio_proj.", "kin_skip.")
    fast_params = []
    base_params = []
    trainable_named = []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        trainable_named.append((n, p))
        if spec["fast_proprio_lr"] and n.startswith(fast_prefixes):
            fast_params.append(p)
        else:
            base_params.append(p)

    if fast_params:
        opt = torch.optim.AdamW(
            [
                {"params": base_params, "lr": 3e-4},
                {"params": fast_params, "lr": 1e-3},
            ],
            weight_decay=1e-4,
        )
    else:
        opt = torch.optim.AdamW(base_params, lr=3e-4, weight_decay=1e-4)

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

    # Verify whether tr.batch has any stochastic noise on proprio or q_normalized
    b_chk1 = tr.batch([0, 1, 2, 3], (10, 25, 50, 100))
    b_chk2 = tr.batch([0, 1, 2, 3], (10, 25, 50, 100))
    proprio_det = bool(torch.equal(b_chk1["proprio"], b_chk2["proprio"]))
    qnorm_det = bool(torch.equal(b_chk1["q_normalized"], b_chk2["q_normalized"]))
    assert proprio_det and qnorm_det, f"Data loader non-deterministic on proprio/q_norm: {proprio_det}, {qnorm_det}"

    hs = sample_horizons(22001, 90)
    b_orig = move(tr.batch([0, 1, 2, 3], hs), "cuda")
    b_230 = attach_extra_proprio(b_orig, norm_tr[[0, 1, 2, 3]])
    pb = move(p1_only(p1.batch([0, 1, 2, 3], H_TUP, "instruction", "P1")), "cuda")

    from e110f_core import infer_d
    from e103d_model import WorldModel as OrigModel
    m_orig = OrigModel(tr.norm)
    m_orig.load_state_dict(cp["model"], strict=True)
    cfg_dump = {k: getattr(m_orig.c, k) for k in dir(m_orig.c) if not k.startswith("_") and isinstance(getattr(m_orig.c, k), (int, float, bool, str))}
    m_orig = m_orig.cuda().eval()
    with torch.no_grad():
        a_ref = infer_d(m_orig, b_orig, 4, 90)
    del m_orig

    for arm in ("N0", "N1", "N2"):
        m, opt, restored = make_e130_model_and_opt(tr.norm, arm, cp)
        m.eval()
        with torch.no_grad():
            a_new = infer_d(m, b_230, 4, 90)
        max_diff = float((a_new - a_ref).abs().max().item())
        assert max_diff < 1e-6, f"Step-0 equivalence failed on {arm}: {max_diff}"

        # Also check that in train() mode, clean_state makes two forward passes with identical RNG-independent state deterministic if block dropout is 0
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

    CTRL130.mkdir(parents=True, exist_ok=True)
    (CTRL130 / "wam_cfg_audit.json").write_text(json.dumps(dict(cfg=cfg_dump, proprio_loader_deterministic=proprio_det), indent=2) + "\n")
    peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    print(json.dumps(dict(smoke="passed", peak_mib=peak, state_noise=cfg_dump.get("state_noise"), state_drop=cfg_dump.get("state_drop"), dropout=cfg_dump.get("dropout"))), flush=True)


def train(arm, seed=90, steps=4000):
    assert arm in ARM_SPECS
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ginfo = query_gpu_free(vis)
    assert ginfo["free_mib"] >= int(PEAK_MIB * 1.5), f"Insufficient VRAM: {ginfo}"

    tag = f"wam_{arm.lower()}_s{seed}"
    run = T / f"results/e130_{tag}_20261119_01"
    assert not run.exists(), f"{run} already exists"
    run.mkdir(parents=False)

    t0 = time.monotonic()
    ledger = dict(tag=tag, arm=arm, spec=ARM_SPECS[arm], seed=seed, steps=steps, gpu=ginfo, status="running", started_brt=brt())
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
    m, opt, restored = make_e130_model_and_opt(tr.norm, arm, cp)
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
            arm=arm,
            spec=ARM_SPECS[arm],
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
