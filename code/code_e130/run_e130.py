# E130 parallel multi-GPU runner:
#   1. Audits simulation assets inside T (read-only) + WAM state_noise/state_drop config
#   2. Runs smoke tests for eval_e130.py and train_e130.py
#   3. Trains N0, N1, N2 concurrently across 3 idle GPUs (~20 min wall-clock)
#   4. Runs full Q1 + Q2 + Q3 evaluation (eval_e130.py) and appends to notes/worklog.md
import os, sys, json, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e130_common import T, LOCK_PATH, CTRL130, brt, build_or_load_extra_stats
from e102x_setup import datasets

PY = "/mnt/workspace/dengjianfeng/venvs/e29/bin/python"
CODE = T / "code_e130"
EVAL_DIR = T / "results/e130_eval_20261119_01"


def write_status(state, **kw):
    CTRL130.mkdir(parents=True, exist_ok=True)
    d = dict(status=state, updated_brt=brt(), **kw)
    tmp = CTRL130 / "status.json.tmp"
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, CTRL130 / "status.json")
    print(f"[{brt()}] E130 STATUS: {state} {kw}", flush=True)


def get_free_gpus(min_free_mib=18000):
    s = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    cand = []
    for line in s.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        if int(free) >= min_free_mib and int(util) < 50:
            cand.append((int(free), idx))
    cand.sort(reverse=True)
    return [idx for _, idx in cand]


def audit_sim_environment():
    sim_info = {"mujoco_installed": False, "mujoco_version": None, "xml_urdf_files": [], "sim_dirs": []}
    try:
        import mujoco
        sim_info["mujoco_installed"] = True
        sim_info["mujoco_version"] = getattr(mujoco, "__version__", "unknown")
    except Exception as e:
        sim_info["mujoco_import_error"] = repr(e)

    for sub in ("sim", "assets", "models", "urdf"):
        p = T / sub
        if p.exists():
            sim_info["sim_dirs"].append(str(p))
    # Search up to depth 4 inside T for xml / urdf / mjcf files
    found = []
    for root, dirs, files in os.walk(T):
        rel = Path(root).relative_to(T)
        if len(rel.parts) > 3:
            dirs[:] = []
            continue
        # skip large result/data folders
        dirs[:] = [d for d in dirs if d not in ("results", "runs", "data", ".git", "__pycache__")]
        for fn in files:
            if fn.endswith((".xml", ".urdf")) or ("piper" in fn.lower() and fn.endswith((".py", ".md", ".yaml", ".json"))):
                found.append(str(rel / fn))
    sim_info["xml_urdf_and_piper_files"] = sorted(found)[:80]
    CTRL130.mkdir(parents=True, exist_ok=True)
    (CTRL130 / "sim_audit.json").write_text(json.dumps(sim_info, indent=2, ensure_ascii=False) + "\n")


def main():
    t0 = time.monotonic()
    try:
        write_status("auditing_sim_and_stats")
        audit_sim_environment()
        _, _, tr, dev = datasets()
        entries = json.loads(LOCK_PATH.read_text())["entries"]
        build_or_load_extra_stats(tr, dev, entries)
        del tr, dev

        gpus = get_free_gpus(min_free_mib=18000)
        assert len(gpus) >= 1, "No GPU with >=18 GiB free"

        write_status("smoke_test", gpu=gpus[0])
        env0 = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "eval_e130.py"), "--smoke"], env=env0, check=True)
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "train_e130.py"), "--smoke"], env=env0, check=True)

        jobs = ["N0", "N1", "N2"]
        gpus = get_free_gpus(min_free_mib=18000)
        max_parallel = min(len(gpus), len(jobs))
        write_status("training_parallel", gpus=gpus[:max_parallel], jobs=jobs)

        pending = list(jobs)
        running = {}
        completed = []

        while pending or running:
            for g in gpus[:max_parallel]:
                if g not in running and pending:
                    arm = pending.pop(0)
                    log_path = CTRL130 / f"train_{arm}.log"
                    lf = log_path.open("w")
                    env_g = dict(os.environ, CUDA_VISIBLE_DEVICES=g, E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
                    proc = subprocess.Popen(
                        [PY, "-I", "-B", "-u", str(CODE / "train_e130.py"), arm, "90", "4000"],
                        env=env_g, stdout=lf, stderr=subprocess.STDOUT,
                    )
                    running[g] = (proc, lf, arm)
                    print(f"[{brt()}] Launched {arm} on GPU {g} (pid={proc.pid})", flush=True)

            time.sleep(5)
            done_gpus = []
            for g, (proc, lf, arm) in running.items():
                rc = proc.poll()
                if rc is not None:
                    lf.close()
                    if rc != 0:
                        tail = (CTRL130 / f"train_{arm}.log").read_text()[-2000:]
                        raise RuntimeError(f"Job {arm} failed on GPU {g} with rc={rc}:\n{tail}")
                    completed.append(arm)
                    done_gpus.append(g)
                    print(f"[{brt()}] Finished {arm} on GPU {g}", flush=True)
            for g in done_gpus:
                del running[g]
            if done_gpus:
                write_status("training_parallel", completed=completed, active=[v[2] for v in running.values()], elapsed_s=round(time.monotonic() - t0, 1))

        eval_gpu = get_free_gpus(min_free_mib=15000)[0]
        write_status("evaluating", gpu=eval_gpu, elapsed_s=round(time.monotonic() - t0, 1))
        env_ev = dict(os.environ, CUDA_VISIBLE_DEVICES=eval_gpu, E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "eval_e130.py")], env=env_ev, check=True)

        s = json.loads((EVAL_DIR / "summary.json").read_text())
        q2m = s["q2_clean_state"]["models"]
        n0 = q2m["WAM_N0_clean"]
        n1 = q2m["WAM_N1_fastp"]
        n2 = q2m["WAM_N2_kinskip"]
        q3c = s["q3_perturbed_gt"]["world"]["WAM_v1_s90_W"]["50"]["candidates"]
        wl_line = (
            f"\n- 2026-11-19 BRT E130（Antigravity执行）：三项机制核验（211窗/27parent）："
            f"①逐关节/逐帧拆解+正运动学反事实；"
            f"②去状态噪声重训：N0(仅去state_noise/drop) l8/l50={n0['l8'][0]:.5f}/{n0['l50'][0]:.5f}，"
            f"N1(+本体快LR)={n1['l8'][0]:.5f}/{n1['l50'][0]:.5f}，N2(+直连运动学skip)={n2['l8'][0]:.5f}/{n2['l50'][0]:.5f}（e50={n2['e50'][0]:.2f}mm）；"
            f"③同幅扰动真值驱动WAM_v1世界头(+50保留率)：pred_N1={q3c['pred_N1 (WAM_v1)']['retention_pct']:.1f}%，"
            f"gt+跨窗同幅误差={q3c['gt_plus_perm_err']['retention_pct']:.1f}%，gt+逐窗等L1误差={q3c['gt_plus_winscaled_err']['retention_pct']:.1f}%，"
            f"镜像误差(2gt-N1)={q3c['gt_mirror_err (2*gt-N1)']['retention_pct']:.1f}%。见 notes/e130_diagnostic_breakdown_and_clean_state_20261119.md。\n"
        )
        with (T / "notes/worklog.md").open("a") as f:
            f.write(wl_line)

        write_status("completed", total_elapsed_s=round(time.monotonic() - t0, 1))
        (CTRL130 / "COMPLETE").write_text(brt() + "\n")

    except BaseException as e:
        import traceback
        write_status("failed", error=repr(e), traceback=traceback.format_exc(), elapsed_s=round(time.monotonic() - t0, 1))
        raise


if __name__ == "__main__":
    main()
