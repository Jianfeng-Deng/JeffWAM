# E125 autonomous server runner (U_full + J_clean + J_legacy + full 211-window eval & report).
import os, sys, json, time, datetime, subprocess
from pathlib import Path

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
PY = "/mnt/workspace/dengjianfeng/venvs/e29/bin/python"
CODE = T / "code_e125"
CTRL = T / "results/e125_control_20261118_01"
EVAL_DIR = T / "results/e125_eval_20261118_01"


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def write_status(state, **kw):
    d = dict(status=state, updated_brt=brt(), **kw)
    tmp = CTRL / "status.json.tmp"
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, CTRL / "status.json")
    print(f"[{brt()}] E125 STATUS: {state} {kw}", flush=True)


def run_cmd(args, env):
    print(f"[{brt()}] EXEC: {' '.join(map(str, args))}", flush=True)
    subprocess.run(args, env=env, check=True)


def main():
    assert not CTRL.exists(), f"{CTRL} already exists"
    CTRL.mkdir(parents=False)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "7")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id), E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")

    t0 = time.monotonic()
    try:
        write_status("smoke_test", gpu=gpu_id)
        run_cmd([PY, "-I", "-B", "-u", str(CODE / "eval_e125.py"), "--smoke"], env)
        run_cmd([PY, "-I", "-B", "-u", str(CODE / "train_e125.py"), "--smoke"], env)

        for arm in ("U_full", "J_clean", "J_legacy"):
            write_status("training", current_arm=arm, elapsed_s=round(time.monotonic() - t0, 1))
            run_cmd([PY, "-I", "-B", "-u", str(CODE / "train_e125.py"), arm, "4000"], env)

        write_status("evaluating", elapsed_s=round(time.monotonic() - t0, 1))
        run_cmd([PY, "-I", "-B", "-u", str(CODE / "eval_e125.py")], env)

        s = json.loads((EVAL_DIR / "summary.json").read_text())
        d_ud = s["action_summary"]["all211"]["diffs"]["U_full-D"]["l1_50"]
        d_jc = s["action_summary"]["all211"]["diffs"]["J_clean-J_legacy"]["l1_50"]
        wl_line = (
            f"\n- 2026-11-18 BRT E125（Antigravity执行）：全量145 parent池4000步同流对照（U_full解冻纯动作头 vs J_clean去流匹配干净联合 vs J_legacy原联合，对比冻结D与E106C，211窗/27parent）："
            f"L1_50 U_full−D={d_ud['mean']:+.5f} [{d_ud['ci95'][0]:+.5f},{d_ud['ci95'][1]:+.5f}]、"
            f"J_clean−J_legacy={d_jc['mean']:+.5f} [{d_jc['ci95'][0]:+.5f},{d_jc['ci95'][1]:+.5f}]；"
            f"总耗时{round((time.monotonic()-t0)/60,1)}min。见 notes/e125_fullpool_unfrozen_and_clean_joint_20261118.md。\n"
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
