# E133 parallel 4-GPU runner: smoke checks -> parallel G1, G2, G3, G4 training -> full offline + MuJoCo evaluation!
import os, sys, json, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e133_common import T, LOCK_PATH, CTRL133, brt, build_or_load_extra_stats
from e102x_setup import datasets

PY = "/mnt/workspace/dengjianfeng/venvs/e29/bin/python"
CODE = T / "code_e133"
EVAL_DIR = T / "results/e133_eval_20261119_01"


def write_status(state, **kw):
    CTRL133.mkdir(parents=True, exist_ok=True)
    d = dict(status=state, updated_brt=brt(), **kw)
    tmp = CTRL133 / "status.json.tmp"
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, CTRL133 / "status.json")
    print(f"[{brt()}] E133 STATUS: {state} {kw}", flush=True)


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
        write_status("preparing_stats")
        _, _, tr, dev = datasets()
        entries = json.loads(LOCK_PATH.read_text())["entries"]
        build_or_load_extra_stats(tr, dev, entries)
        del tr, dev

        gpus = get_free_gpus(min_free_mib=18000)
        assert len(gpus) >= 1, "No GPU with >=18 GiB free"

        write_status("smoke_test", gpu=gpus[0])
        env0 = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "eval_e133.py"), "--smoke"], env=env0, check=True)
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "train_e133.py"), "--smoke"], env=env0, check=True)

        jobs = ["G1", "G2", "G3", "G4"]
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
                    log_path = CTRL133 / f"train_{arm}.log"
                    lf = log_path.open("w")
                    env_g = dict(os.environ, CUDA_VISIBLE_DEVICES=g, E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
                    proc = subprocess.Popen(
                        [PY, "-I", "-B", "-u", str(CODE / "train_e133.py"), arm, "90", "4000"],
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
                        tail = (CTRL133 / f"train_{arm}.log").read_text()[-2000:]
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
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "eval_e133.py")], env=env_ev, check=True)

        s = json.loads((EVAL_DIR / "summary.json").read_text())
        am = s["action"]["models"]
        gr = s["action"]["gripper"]
        w50 = s["world"]["50"]
        mj = s["mujoco_m3"]["S2_replan_vis"]
        g3 = am["E133_G3_drop15"]
        g4 = am["E133_G4_drop20_skip"]
        wl = (
            f"\n- 2026-11-19 BRT E133（Antigravity执行）：第一梯队三项改进4组阶梯对照（G1三维位姿+速度斜率损失 -> G2+夹爪切换加权与跳变帧头 -> G3+15%视觉遮挡 / G4+20%遮挡+直连skip，211窗/27parent）："
            f"G3 l8/l50/e50/rot50={g3['l8'][0]:.5f}/{g3['l50'][0]:.5f}/{g3['e50'][0]:.2f}mm/{g3['rot50'][0]:.2f}°，"
            f"G4={g4['l8'][0]:.5f}/{g4['l50'][0]:.5f}/{g4['e50'][0]:.2f}mm/{g4['rot50'][0]:.2f}°；"
            f"夹爪F1@±10/松爪命中：WAM_v1={gr['WAM_v1_s90']['f1_10'][0]:.3f}/{gr['WAM_v1_s90']['open_hit10']*100:.1f}% -> "
            f"G3={gr['E133_G3_drop15']['f1_10'][0]:.3f}/{gr['E133_G3_drop15']['open_hit10']*100:.1f}%；"
            f"+50世界头N1保留率：G3={w50['E133_G3_drop15']['retention_pct']:.1f}%；"
            f"MuJoCo M3 S2闭环(+50)：G3={mj['E133_G3_drop15']['e50'][0]:.2f}mm/{mj['E133_G3_drop15']['rot50'][0]:.2f}°。见 notes/e133_pose_vel_grip_tokendrop_20261119.md。\n"
        )
        with (T / "notes/worklog.md").open("a") as f:
            f.write(wl)

        write_status("completed", total_elapsed_s=round(time.monotonic() - t0, 1))
        (CTRL133 / "COMPLETE").write_text(brt() + "\n")

    except BaseException as e:
        import traceback
        write_status("failed", error=repr(e), traceback=traceback.format_exc(), elapsed_s=round(time.monotonic() - t0, 1))
        raise


if __name__ == "__main__":
    main()
