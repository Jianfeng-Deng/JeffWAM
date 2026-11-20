# E134 parallel 4-GPU runner:
#   1. Smoke-tests eval_e134, train_e134, and render_full_episode
#   2. Trains D_dec, FM_nat, FM_dit, FM_res concurrently across 4 idle H20 GPUs
#   3. Runs eval_e134.py (offline + Best-of-8 multimodal + gripper + World N1 + MuJoCo M3)
#   4. Renders full-episode 2x2 Quad-View MuJoCo simulation video (render_full_episode.py) and updates worklog.md!
import os, sys, json, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e134_common import T, LOCK_PATH, CTRL134, brt, build_or_load_extra_stats
from e102x_setup import datasets

PY = "/mnt/workspace/dengjianfeng/venvs/e29/bin/python"
CODE = T / "code_e134"
EVAL_DIR = T / "results/e134_eval_20261119_01"


def write_status(state, **kw):
    CTRL134.mkdir(parents=True, exist_ok=True)
    d = dict(status=state, updated_brt=brt(), **kw)
    tmp = CTRL134 / "status.json.tmp"
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, CTRL134 / "status.json")
    print(f"[{brt()}] E134 STATUS: {state} {kw}", flush=True)


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
        env0 = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], E102_GROUP="A2", MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "eval_e134.py"), "--smoke"], env=env0, check=True)
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "train_e134.py"), "--smoke"], env=env0, check=True)
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "render_full_episode.py"), "--smoke"], env=env0, check=True)

        jobs = ["D_dec", "FM_nat", "FM_dit", "FM_res"]
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
                    log_path = CTRL134 / f"train_{arm}.log"
                    lf = log_path.open("w")
                    env_g = dict(os.environ, CUDA_VISIBLE_DEVICES=g, E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")
                    proc = subprocess.Popen(
                        [PY, "-I", "-B", "-u", str(CODE / "train_e134.py"), arm, "90", "4000"],
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
                        tail = (CTRL134 / f"train_{arm}.log").read_text()[-2000:]
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
        env_ev = dict(os.environ, CUDA_VISIBLE_DEVICES=eval_gpu, E102_GROUP="A2", MUJOCO_GL="egl", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "eval_e134.py")], env=env_ev, check=True)

        write_status("rendering_full_episode", gpu=eval_gpu, elapsed_s=round(time.monotonic() - t0, 1))
        subprocess.run([PY, "-I", "-B", "-u", str(CODE / "render_full_episode.py")], env=env_ev, check=True)

        s = json.loads((EVAL_DIR / "summary.json").read_text())
        am = s["action"]["models"]
        gr = s["action"]["gripper"]
        b8 = s["action"]["best8_multimodal"]
        d_dec = am["E134_D_dec"]
        f_dit_m8 = am["E134_FM_dit_mean8"]
        f_dit_s1 = am["E134_FM_dit_s1"]
        f_res_m8 = am["E134_FM_res_mean8"]
        wl = (
            f"\n- 2026-11-19 BRT E134（Antigravity执行）：流匹配(Flow Matching)动作头与梯度解耦夹爪头4组对照（211窗/27parent）："
            f"①解耦确定性D_dec：e50/rot50={d_dec['e50'][0]:.2f}mm/{d_dec['rot50'][0]:.2f}°，夹爪F1@±10={gr['E134_D_dec']['f1_10']:.3f}，松爪命中={gr['E134_D_dec']['open_hit10']*100:.1f}%，误松开={gr['E134_D_dec']['drop25']*100:.2f}%；"
            f"②流匹配FM_dit：单样位移RMS={f_dit_s1['rms_disp_rad']:.4f}(GT=0.1566)，mean8 e50/rot50={f_dit_m8['e50'][0]:.2f}mm/{f_dit_m8['rot50'][0]:.2f}°，Best-of-8 e50={b8['E134_FM_dit']['best8_e50_mm'][0]:.2f}mm(win166={b8['E134_FM_dit']['best8_win166_mm']:.1f}mm)；"
            f"③残差流匹配FM_res mean8 e50/rot50={f_res_m8['e50'][0]:.2f}mm/{f_res_m8['rot50'][0]:.2f}°；并完成整集连续MuJoCo四分屏仿真视频渲染。见 notes/e134_flow_matching_and_decoupled_grip_20261119.md。\n"
        )
        with (T / "notes/worklog.md").open("a") as f:
            f.write(wl)

        write_status("completed", total_elapsed_s=round(time.monotonic() - t0, 1))
        (CTRL134 / "COMPLETE").write_text(brt() + "\n")

    except BaseException as e:
        import traceback
        write_status("failed", error=repr(e), traceback=traceback.format_exc(), elapsed_s=round(time.monotonic() - t0, 1))
        raise


if __name__ == "__main__":
    main()
