# E129 parallel multi-GPU autonomous runner: runs WAM_v1_s90, WAM_v2_s90, WAM_v2_s91, WAM_v2_s92 across free GPUs, then evaluates!
import os, sys, json, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e129_common import T, INIT, LOCK_PATH, CTRL, brt, build_or_load_extra_stats
from e102x_setup import datasets

PY = "/mnt/workspace/dengjianfeng/venvs/e29/bin/python"
CODE = T / "code_e129"
EVAL_DIR = T / "results/e129_eval_20261119_01"


def write_status(state, **kw):
    CTRL.mkdir(parents=True, exist_ok=True)
    d = dict(status=state, updated_brt=brt(), **kw)
    tmp = CTRL / "status.json.tmp"
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, CTRL / "status.json")
    print(f"[{brt()}] E129 STATUS: {state} {kw}", flush=True)


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


def main():
    t0 = time.monotonic()
    try:
        # 1. Pre-build extra_proprio_stats.json once cleanly so parallel workers never race!
        write_status("building_extra_proprio_stats")
        _, _, tr, dev = datasets()
        entries = json.loads(LOCK_PATH.read_text())["entries"]
        build_or_load_extra_stats(tr, dev, entries)
        del tr, dev

        gpus = get_free_gpus(min_free_mib=18000)
        assert len(gpus) >= 1, "No GPU with >=18 GiB free"
        print("Available GPUs (>=18 GiB free):", gpus, flush=True)

        # 2. Smoke test on best GPU (both evaluator and trainer!)
        write_status("smoke_test", gpu=gpus[0])
        env0 = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "eval_e129.py"), "--smoke"], env=env0, check=True)
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "train_e129.py"), "--smoke"], env=env0, check=True)

        # 3. Schedule 4 jobs across up to 4 GPUs concurrently (1 job per GPU at a time, polite VRAM < 2.5 GiB each!)
        jobs = [
            ("v1", 90),
            ("v2", 90),
            ("v2", 91),
            ("v2", 92),
        ]
        gpus = get_free_gpus(min_free_mib=18000)
        max_parallel = min(len(gpus), len(jobs))
        write_status("training_parallel", gpus=gpus[:max_parallel], jobs=[f"{v}_s{s}" for v, s in jobs])

        pending = list(jobs)
        running = {}  # gpu_id -> (Popen, log_file, job_tuple)
        completed_jobs = []

        while pending or running:
            # Fill free GPU slots
            for g in gpus[:max_parallel]:
                if g not in running and pending:
                    var, sd = pending.pop(0)
                    log_path = CTRL / f"train_{var}_s{sd}.log"
                    lf = log_path.open("w")
                    env_g = dict(os.environ, CUDA_VISIBLE_DEVICES=g, E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
                    proc = subprocess.Popen(
                        [PY, "-I", "-B", "-u", str(CODE / "train_e129.py"), var, str(sd), "4000"],
                        env=env_g, stdout=lf, stderr=subprocess.STDOUT,
                    )
                    running[g] = (proc, lf, (var, sd))
                    print(f"[{brt()}] Launched {var}_s{sd} on GPU {g} (pid={proc.pid})", flush=True)

            time.sleep(5)
            done_gpus = []
            for g, (proc, lf, jb) in running.items():
                rc = proc.poll()
                if rc is not None:
                    lf.close()
                    if rc != 0:
                        tail = (CTRL / f"train_{jb[0]}_s{jb[1]}.log").read_text()[-2000:]
                        raise RuntimeError(f"Job {jb} failed on GPU {g} with rc={rc}:\n{tail}")
                    completed_jobs.append(f"{jb[0]}_s{jb[1]}")
                    done_gpus.append(g)
                    print(f"[{brt()}] Finished {jb[0]}_s{jb[1]} on GPU {g}", flush=True)
            for g in done_gpus:
                del running[g]
            if done_gpus:
                write_status("training_parallel", completed=completed_jobs, active=[f"{v[2][0]}_s{v[2][1]}" for v in running.values()], elapsed_s=round(time.monotonic() - t0, 1))

        # 4. Run full evaluation on best GPU
        eval_gpu = get_free_gpus(min_free_mib=15000)[0]
        write_status("evaluating", gpu=eval_gpu, elapsed_s=round(time.monotonic() - t0, 1))
        env_ev = dict(os.environ, CUDA_VISIBLE_DEVICES=eval_gpu, E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "eval_e129.py")], env=env_ev, check=True)

        # 5. Append concise summary to notes/worklog.md
        s = json.loads((EVAL_DIR / "summary.json").read_text())
        am = s["action"]["models"]
        v1 = am["WAM_v1_s90"]
        v2 = am["WAM_v2_s90"]
        v2e = am["WAM_v2_ens3"]
        bt3 = am["blind_t3_ens3"]
        w50 = s["world"]["50"]["WAM_v2_s90"]
        wl_line = (
            f"\n- 2026-11-19 BRT E129（Antigravity执行）：端到端升级版联合WAM（补t-1/t-3速度230维+等权L1+余弦退火+世界头25% detached_N1消除暴露偏差，211窗/27parent）："
            f"单种子WAM_v1_s90 l8/l50={v1['l8'][0]:.5f}/{v1['l50'][0]:.5f}，WAM_v2_s90={v2['l8'][0]:.5f}/{v2['l50'][0]:.5f}；"
            f"3种子集成WAM_v2_ens3 l8/l50/e50={v2e['l8'][0]:.5f}/{v2e['l50'][0]:.5f}/{v2e['e50'][0]:.2f}mm（对比原盲策略3s {bt3['l8'][0]:.5f}/{bt3['l50'][0]:.5f}/{bt3['e50'][0]:.2f}mm）；"
            f"+50世界头N1保留率={w50['retention_pct']:.1f}%（N1−no={w50['n1_minus_no_pp'][0]:+.2f}pp）。见 notes/e129_upgraded_end2end_wam_20261119.md。\n"
        )
        with (T / "notes/worklog.md").open("a") as f:
            f.write(wl_line)

        write_status("completed", total_elapsed_s=round(time.monotonic() - t0, 1))
        (CTRL / "COMPLETE").write_text(brt() + "\n")

    except BaseException as e:
        import traceback
        write_status("failed", error=repr(e), traceback=traceback.format_exc(), elapsed_s=round(time.monotonic() - t0, 1))
        raise


if __name__ == "__main__":
    main()
