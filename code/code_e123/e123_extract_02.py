# E123 (V1) Step A (GPU, zero training, run 02): Local action sensitivity of E106-C world head across 211 dev windows (27 parents) & 160 stage-dev windows (20 parents).
import os, sys, json, time, hashlib, datetime, subprocess
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
for p in ("code_e110f", "code_e106c", "code_e102a"):
    sys.path.insert(0, str(T / p))

from e99w2_common import GROUP
assert GROUP == "A2", f"Expected GROUP=A2, got {GROUP}"
from e99w2_model import WorldModel, JOINTS, trajectory
from e102x_setup import datasets
from e102a_contract import contract
from e110f_core import infer_d
from e110f_eval import float_batch
from e86.data import move

INIT = T / "results/e106c_f2_s90_20261115_02/ckpt_20000.pt"
FEATS = T / "results/e115p1_stage_feats"
LOCK_PATH = T / "results/e106c_control_20261115_02/PRETRAIN_LOCK.json"
P2_REF = T / "results/e115p2_pooled_20261117_02"
OUT = T / "results/e123_action_sens_20261118_02"
PEAK_MIB = 6144
EXCLUDE_CID = "piper_4dc107335389f5711a0f65e0"

ACT_CONDS = (
    "gt", "s15", "s12", "s08", "s05", "freeze", "rev05",
    "lag15", "lead15", "g_hold", "g_flip",
    "n1", "n1_s13", "n1_s07", "n1_gflip",
    "no", "sh",
)
ALL_POOLED = ("real", "cur") + ACT_CONDS


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_gpu():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert vis != "", "CUDA_VISIBLE_DEVICES must be explicitly set"
    s = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in s.strip().splitlines():
        idx, uuid, free, util = [x.strip() for x in line.split(",")]
        rows.append(dict(index=idx, uuid=uuid, free_mib=int(free), util=int(util)))
    chosen = [r for r in rows if r["index"] == vis or r["uuid"] == vis]
    assert len(chosen) == 1, f"Could not match CUDA_VISIBLE_DEVICES={vis} in {rows}"
    g = chosen[0]
    assert g["free_mib"] >= int(PEAK_MIB * 1.5), f"Insufficient free VRAM: {g} < {PEAK_MIB * 1.5}"
    return g


def scale_joints(a_base, q0, alpha, j_idx):
    out = a_base.clone()
    out[:, :, j_idx] = q0[:, None, j_idx] + alpha * (a_base[:, :, j_idx] - q0[:, None, j_idx])
    return out


def hold_grip(a_base, q0):
    out = a_base.clone()
    out[:, :, [6, 13]] = q0[:, None, [6, 13]].expand(-1, 50, -1)
    return out


def flip_grip(a_base, c):
    out = a_base.clone()
    th = out.new_tensor(c["threshold_m"])
    lo = out.new_tensor(c["closed_m"])
    hi = out.new_tensor(c["open_m"])
    out[:, :, [6, 13]] = torch.where(out[:, :, [6, 13]] > th, lo, hi)
    return out


def pool_cam(x):
    assert x.shape[1] % 3 == 0 and x.shape[2] == 768
    return x.reshape(len(x), 3, x.shape[1] // 3, 768).mean(2).reshape(len(x), 2304).cpu().numpy().astype("f4")


def main():
    assert not OUT.exists(), f"{OUT} already exists; refusing to overwrite"
    gpu_info = check_gpu()
    OUT.mkdir(parents=False)
    t0 = time.monotonic()

    ckpt_sha = sha256_file(INIT)
    assert ckpt_sha.startswith("dd27fe05c034"), f"Unexpected ckpt sha: {ckpt_sha}"

    ledger = dict(
        experiment="E123_V1_local_action_sensitivity",
        run="02",
        status="running",
        gpu=gpu_info,
        peak_estimate_mib=PEAK_MIB,
        ckpt=str(INIT),
        ckpt_sha256=ckpt_sha,
        started_brt=brt(),
    )
    (OUT / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.reset_peak_memory_stats()

    _, _, tr, dev = datasets()
    entries = json.loads(LOCK_PATH.read_text())["entries"]
    assert len(entries) == 211 and len({e["parent"] for e in entries}) == 27
    assert all(e["cid"] != EXCLUDE_CID for e in entries)

    avail_stage = {p.stem for p in FEATS.glob("pc_*.npz")}
    E_stage = [e for e in entries if e["parent"] in avail_stage]
    E_rest = [e for e in entries if e["parent"] not in avail_stage]
    assert len(E_stage) == 160 and len(E_rest) == 51

    c = contract()
    cp = torch.load(INIT, map_location="cpu", weights_only=False)
    m = WorldModel(tr.norm)
    m.load_state_dict(cp["model"], strict=True)
    m = m.cuda().eval()
    del cp

    j_idx = torch.as_tensor(JOINTS, device="cuda")
    th_t = torch.tensor(c["threshold_m"], device="cuda", dtype=torch.float32)

    rows = []
    pooled = {k: [] for k in ALL_POOLED}
    mse_all = {k: [] for k in ACT_CONDS}
    mse_cam = {k: [] for k in ACT_CONDS}
    copy_mse_all = []
    copy_mse_cam = []
    max_gt_traj_diff = 0.0

    with torch.inference_mode():
        for h in (10, 25, 50, 100):
            Eh_stage = [e for e in E_stage if h in e["horizons"]]
            Eh_stage = [Eh_stage[i] for i in np.random.default_rng(5).permutation(len(Eh_stage))]
            Eh_rest = [e for e in E_rest if h in e["horizons"]]
            Eh_rest = [Eh_rest[i] for i in np.random.default_rng(5).permutation(len(Eh_rest))]

            for subset_name, Eh in (("stage160", Eh_stage), ("rest51", Eh_rest)):
                for s in range(0, len(Eh), 4):
                    es = Eh[s : s + 4]
                    b = move(float_batch(dev.batch([e["index"] for e in es], (h,), "short")), "cuda")
                    assert b["world_valid"].all()
                    Y = b["world"].double()
                    assert torch.isfinite(Y).all()
                    B, nw, D = Y.shape
                    assert nw % 3 == 0 and D == 768
                    Y_cam = Y.reshape(B, 3, nw // 3, 768)

                    copy_mse_all.append(Y.square().mean((1, 2)).cpu().numpy().astype("f8"))
                    copy_mse_cam.append(Y_cam.square().mean((2, 3)).cpu().numpy().astype("f8"))

                    scale = b["world_scale"]
                    cur = b["world_current"]
                    to_raw = lambda P: cur + P * scale

                    ctx = b["context"]
                    q0 = ctx[:, :14]
                    a_gt = b["plan_action"]

                    r_gt_rebuilt = trajectory(ctx, a_gt)
                    diff = float((r_gt_rebuilt - b["robot"]).abs().max().item())
                    if diff > max_gt_traj_diff:
                        max_gt_traj_diff = diff
                    assert diff < 1e-4, f"b['robot'] vs trajectory(ctx, plan_action) mismatch: {diff}"

                    a_n1 = infer_d(m, b, 4, 90)

                    act_map = {
                        "gt": a_gt,
                        "s15": scale_joints(a_gt, q0, 1.5, j_idx),
                        "s12": scale_joints(a_gt, q0, 1.2, j_idx),
                        "s08": scale_joints(a_gt, q0, 0.8, j_idx),
                        "s05": scale_joints(a_gt, q0, 0.5, j_idx),
                        "freeze": hold_grip(scale_joints(a_gt, q0, 0.0, j_idx), q0),
                        "rev05": scale_joints(a_gt, q0, -0.5, j_idx),
                        "lag15": torch.cat([q0[:, None].expand(-1, 15, -1), a_gt[:, :35]], dim=1),
                        "lead15": torch.cat([a_gt[:, 15:], a_gt[:, -1:].expand(-1, 15, -1)], dim=1),
                        "g_hold": hold_grip(a_gt, q0),
                        "g_flip": flip_grip(a_gt, c),
                        "n1": a_n1,
                        "n1_s13": scale_joints(a_n1, q0, 1.3, j_idx),
                        "n1_s07": scale_joints(a_n1, q0, 0.7, j_idx),
                        "n1_gflip": flip_grip(a_n1, c),
                    }

                    l1_50_map = {}
                    l1_h_map = {}
                    steps_h = min(h, 50)
                    for k_act, a_cand in act_map.items():
                        l1_50_map[k_act] = (a_cand[:, :, j_idx] - a_gt[:, :, j_idx]).abs().mean((1, 2)).cpu().numpy()
                        l1_h_map[k_act] = (a_cand[:, :steps_h, j_idx] - a_gt[:, :steps_h, j_idx]).abs().mean((1, 2)).cpu().numpy()

                    g0 = (q0[:, [6, 13]] > th_t).long()
                    gt_g = (a_gt[:, :, [6, 13]] > th_t).long()
                    n1_g = (a_n1[:, :, [6, 13]] > th_t).long()
                    sw_h = (gt_g[:, :steps_h] != g0[:, None]).any(1).cpu().numpy()
                    sw_50 = (gt_g != g0[:, None]).any(1).cpu().numpy()
                    n1_g_err_h = (n1_g[:, :steps_h] != gt_g[:, :steps_h]).float().mean(1).cpu().numpy()
                    joint_disp_h = (a_gt[:, :steps_h, j_idx] - q0[:, None, j_idx]).abs().mean((1, 2)).cpu().numpy()
                    joint_disp_50 = (a_gt[:, :, j_idx] - q0[:, None, j_idx]).abs().mean((1, 2)).cpu().numpy()

                    pooled["real"].append(pool_cam(b["world_raw"]))
                    pooled["cur"].append(pool_cam(cur))

                    for k_cond in ACT_CONDS:
                        if k_cond == "gt":
                            P = m.forward_mean(b)["mean"]
                        elif k_cond == "no":
                            P = m.forward_mean(
                                dict(b, robot=torch.zeros_like(b["robot"]), robot_valid=torch.zeros_like(b["robot_valid"]))
                            )["mean"]
                        elif k_cond == "sh":
                            if subset_name == "stage160" or B > 1:
                                r_sh = b["robot"].roll(1, 0)
                                rv_sh = b["robot_valid"].roll(1, 0)
                            else:
                                donor = move(float_batch(dev.batch([Eh[0]["index"]], (h,), "short")), "cuda")
                                r_sh = donor["robot"]
                                rv_sh = donor["robot_valid"]
                            P = m.forward_mean(dict(b, robot=r_sh, robot_valid=rv_sh))["mean"]
                        else:
                            P = m.forward_mean(dict(b, robot=trajectory(ctx, act_map[k_cond])))["mean"]

                        assert torch.isfinite(P).all(), f"Nonfinite prediction on {k_cond}"
                        Pd = P.double()
                        err_all = (Pd - Y).square().mean((1, 2)).cpu().numpy().astype("f8")
                        err_cam = (Pd.reshape(B, 3, nw // 3, 768) - Y_cam).square().mean((2, 3)).cpu().numpy().astype("f8")
                        mse_all[k_cond].append(err_all)
                        mse_cam[k_cond].append(err_cam)
                        pooled[k_cond].append(pool_cam(to_raw(P)))

                    for k_i, e in enumerate(es):
                        rows.append(
                            dict(
                                parent=e["parent"],
                                cid=e["cid"],
                                index=e["index"],
                                anchor=e["anchor"],
                                h=h,
                                in_stage160=(subset_name == "stage160"),
                                original48=bool(e["original48"]),
                                stratum=e["stratum"],
                                joint_disp_h=float(joint_disp_h[k_i]),
                                joint_disp_50=float(joint_disp_50[k_i]),
                                grip_sw_L_h=bool(sw_h[k_i, 0]),
                                grip_sw_R_h=bool(sw_h[k_i, 1]),
                                grip_sw_L_50=bool(sw_50[k_i, 0]),
                                grip_sw_R_50=bool(sw_50[k_i, 1]),
                                n1_g_err_L_h=float(n1_g_err_h[k_i, 0]),
                                n1_g_err_R_h=float(n1_g_err_h[k_i, 1]),
                                l1_50={ka: float(l1_50_map[ka][k_i]) for ka in act_map},
                                l1_h={ka: float(l1_h_map[ka][k_i]) for ka in act_map},
                            )
                        )

    pooled_np = {k: np.concatenate(v, axis=0) for k, v in pooled.items()}
    mse_all_np = {k: np.concatenate(v, axis=0) for k, v in mse_all.items()}
    mse_cam_np = {k: np.concatenate(v, axis=0) for k, v in mse_cam.items()}
    copy_all_np = np.concatenate(copy_mse_all, axis=0)
    copy_cam_np = np.concatenate(copy_mse_cam, axis=0)

    p2_check = {}
    if (P2_REF / "pooled.npz").exists():
        ref_z = np.load(P2_REF / "pooled.npz")
        ref_rows = json.loads((P2_REF / "rows.json").read_text())
        s160_idx = [i for i, r in enumerate(rows) if r["in_stage160"]]
        assert len(s160_idx) == len(ref_rows), f"{len(s160_idx)} vs {len(ref_rows)}"
        for i_loc, r_ref in zip(s160_idx, ref_rows):
            r_loc = rows[i_loc]
            assert (r_loc["parent"], r_loc["index"], r_loc["anchor"], r_loc["h"]) == (
                r_ref["parent"], r_ref["index"], r_ref["anchor"], r_ref["h"]
            )
        for k_ref in ("real", "cur", "gt", "no", "n1", "sh"):
            max_abs = float(np.abs(pooled_np[k_ref][s160_idx] - ref_z[k_ref]).max())
            p2_check[k_ref] = max_abs
            assert max_abs < 1e-4, f"P2 equivalence failed on {k_ref}: {max_abs}"

    np.savez_compressed(OUT / "pooled.npz", **pooled_np)
    np.savez_compressed(
        OUT / "latent_mse.npz",
        copy_all=copy_all_np,
        copy_cam=copy_cam_np,
        **{f"all_{k}": v for k, v in mse_all_np.items()},
        **{f"cam_{k}": v for k, v in mse_cam_np.items()},
    )
    (OUT / "rows.json").write_text(json.dumps(rows))

    elapsed = time.monotonic() - t0
    peak_alloc_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    ledger.update(
        status="completed",
        elapsed_s=elapsed,
        gpu_hours=elapsed / 3600.0,
        peak_allocated_mib=peak_alloc_mib,
        max_gt_traj_diff=max_gt_traj_diff,
        p2_equivalence_max_abs=p2_check,
        total_rows=len(rows),
        finished_brt=brt(),
    )
    (OUT / "gpu_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2), flush=True)


if __name__ == "__main__":
    main()
