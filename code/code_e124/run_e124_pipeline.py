# E124 (P1-v2) autonomous detached server pipeline (survives local client shutdown).
import os, sys, json, time, datetime, subprocess
from pathlib import Path

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
PY = "/mnt/workspace/dengjianfeng/venvs/e29/bin/python"
CODE = T / "code_e124"
CTRL = T / "results/e124_p1v2_control_20261118_01"
EVAL_DIR = T / "results/e124_p1v2_eval_20261118_01"


def brt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def write_status(state, **kw):
    d = dict(status=state, updated_brt=brt(), **kw)
    tmp = CTRL / "status.json.tmp"
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, CTRL / "status.json")
    print(f"[{brt()}] PIPELINE STATUS: {state} {kw}", flush=True)


def run_cmd(args, env):
    print(f"[{brt()}] EXEC: {' '.join(map(str, args))}", flush=True)
    subprocess.run(args, env=env, check=True)


def main():
    assert not CTRL.exists(), f"{CTRL} already exists"
    CTRL.mkdir(parents=False)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "2")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id), E102_GROUP="A2", PYTHONDONTWRITEBYTECODE="1")

    prereg = dict(
        experiment="E124_P1v2_unfrozen_dropout",
        started_brt=brt(),
        gpu=gpu_id,
        arms_core=["U0", "U2", "D0", "D2"],
        arms_conditional=["U1", "D1"],
        steps_per_arm=4000,
        subset="45 stage-feature train parents (13,172 rows, exact E115-P1 sample stream sha f4c1a7c2ae74...)",
        eval_set="160 windows / 20 stage-dev parents (30 boundary windows)",
        dropout_spec="D arms: 50% random block dropout on current cameras (B,3), history frames (B,1), and 164-D proprio + q_normalized (B,1); 20-D stage vector kept 100%; inference dropout=0",
        conditional_s1_rule="Run U1 if U2-U0 improves all/boundary l1_50 (CI upper < 0) or f1_10 (CI lower > 0); run D1 if D2-D0 meets same criterion.",
    )
    (CTRL / "PREREG_LOCK.json").write_text(json.dumps(prereg, indent=2, ensure_ascii=False) + "\n")

    t0 = time.monotonic()
    try:
        write_status("smoke_test")
        run_cmd([PY, "-I", "-B", "-u", str(CODE / "eval_and_report.py"), "--smoke"], env)
        run_cmd([PY, "-I", "-B", "-u", str(CODE / "train_arm.py"), "--smoke"], env)

        for arm in ("U0", "U2", "D0", "D2"):
            write_status("training_core", current_arm=arm, elapsed_s=round(time.monotonic() - t0, 1))
            run_cmd([PY, "-I", "-B", "-u", str(CODE / "train_arm.py"), arm, "4000"], env)

        write_status("evaluating_core4", elapsed_s=round(time.monotonic() - t0, 1))
        run_cmd([PY, "-I", "-B", "-u", str(CODE / "eval_and_report.py"), "core4"], env)

        # Check pre-registered trigger for deployable causal controller arms U1 / D1
        s4 = json.loads((EVAL_DIR / "summary_core4.json").read_text())
        all_d = s4["action_and_smoothness"]["all"]["diffs"]
        bd_d = s4["action_and_smoothness"]["boundary"]["diffs"]

        extra_arms = []
        for cand_arm, pair in (("U1", "U2-U0"), ("D1", "D2-D0")):
            if pair in all_d:
                win_l1 = (all_d[pair]["l1_50"]["ci"][1] < 0) or (bd_d[pair]["l1_50"]["ci"][1] < 0)
                win_f1 = all_d[pair]["f1_10"]["ci"][0] > 0
                if win_l1 or win_f1:
                    extra_arms.append(cand_arm)

        if extra_arms:
            for arm in extra_arms:
                write_status("training_conditional_ctrl", current_arm=arm, elapsed_s=round(time.monotonic() - t0, 1))
                run_cmd([PY, "-I", "-B", "-u", str(CODE / "train_arm.py"), arm, "4000"], env)
            write_status("evaluating_final", extra_arms=extra_arms, elapsed_s=round(time.monotonic() - t0, 1))
            run_cmd([PY, "-I", "-B", "-u", str(CODE / "eval_and_report.py"), "final"], env)

        # Append concise completion line to notes/worklog.md
        sf = json.loads((EVAL_DIR / "summary.json").read_text())
        ad = sf["action_and_smoothness"]["all"]["diffs"]
        bd = sf["action_and_smoothness"]["boundary"]["diffs"]
        u2u0 = ad["U2-U0"]["l1_50"]
        d2d0 = ad["D2-D0"]["l1_50"]
        d2d0_bd = bd["D2-D0"]["l1_50"]
        wl_line = (
            f"\n- 2026-11-18 BRT E124（P1-v2，Antigravity后台执行）：解冻6层主干+50%观测/本体整块丢弃复验阶段条件动作头（U0/U2/D0/D2，各4000步，同E115-P1流）："
            f"全160窗L1_50 U2−U0={u2u0['value']:+.5f} [{u2u0['ci'][0]:+.5f},{u2u0['ci'][1]:+.5f}]、"
            f"D2−D0={d2d0['value']:+.5f} [{d2d0['ci'][0]:+.5f},{d2d0['ci'][1]:+.5f}]（切换30窗D2−D0={d2d0_bd['value']:+.5f} [{d2d0_bd['ci'][0]:+.5f},{d2d0_bd['ci'][1]:+.5f}]）；"
            f"触发因果臂={extra_arms}；总耗时{round((time.monotonic()-t0)/60,1)}min。见 notes/e124_p1v2_unfrozen_dropout_20261118.md。\n"
        )
        with (T / "notes/worklog.md").open("a") as f:
            f.write(wl_line)

        write_status("completed", extra_arms=extra_arms, total_elapsed_s=round(time.monotonic() - t0, 1))
        (CTRL / "COMPLETE").write_text(brt() + "\n")

    except BaseException as e:
        import traceback
        write_status("failed", error=repr(e), traceback=traceback.format_exc(), elapsed_s=round(time.monotonic() - t0, 1))
        raise


if __name__ == "__main__":
    main()
