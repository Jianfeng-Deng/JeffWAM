# Task: Consolidated results analysis report, E21-E28 (profile: sol-deep = gpt-6.1-sol, high)

## Goal
Read the experiment log and result files and deliver ONE analysis report that summarizes every experiment so far, separates confirmed positive findings, confirmed negative findings and unresolved items, and recommends the remaining trials.

## Done criteria
- [ ] Report at runs/report_20261103/ANALYSIS_REPORT.md (<=250 lines) plus its tables as CSV in the same folder.
- [ ] Every number cites its source file (run dir + json/md). Mark numbers you could not verify.
- [ ] Section 1 answers: "What is the status and outcome of the experiment verifying 3D arm/hand pose from URDF FK?" (E22b calibration check, E24/E24r hand rescore, E25 hand vs FK, E28 S1 hand_geometry module, FK-vs-measured-vs-commanded). Give per-arm pixel and mm errors, which calibration version, which points, what passed or failed, and the known A-arm instability (~66 mm) vs B (~5.6 mm).
- [ ] Section 2: one master table of all experiments (ID, change, key metric, verdict) for E21..E28 incl. E27a arms A0-A4 and E28 full/nogate/nofk, and the earlier V-JEPA result in docs/experiment_handover_20261030/.
- [ ] Section 3: confirmed positives, confirmed negatives, each with evidence strength (single seed? n windows? CI?).
- [ ] Section 4: unresolved/at-risk claims and evaluation caveats (e.g. E28 native value gate is UNMEASURED because the value bridge strips E28 modules; static-shift 0.002 px may be structural; metric mismatches cosine vs raw L2; clean-label caveats from E26a).
- [ ] Section 5: ranked remaining trials, each with goal, minimal experiment, expected cost (GPU-hours), pass/fail criterion. Include at least: native E28 value test, FK-geometry check using measured vs commanded joints, action-diverse/policy-seeded data pilot, and any trial the evidence itself suggests. Do not run them.
- [ ] Add an E-report line to docs/experiment_log/EXPERIMENT_LOG.md (atomic, re-read before writing) and an entry in notes/worklog.md.

## Context
- Host files: ssh -o BatchMode=yes dengjianfeng-101 ; project /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu
- You MAY ssh -o BatchMode=yes <gpu-server> (GPU host) if you need to check a GPU or a file there; never touch other users' processes; no installs into system envs.
- Start from: docs/experiment_log/EXPERIMENT_LOG.md, runs/e28_chain_20261103/{ANALYSIS_INPUT.md,VERIFICATION.md,summary.md}, runs/e27a_chain_20261103/summary.md, runs/e27a_copycheck_20261103/report.md, runs/e26a_audit_20261102/summary.md, runs/e25_chain_20261102/, runs/e22b_calib_check_20261102_fix/summary.md, runs/e24r_hand_rescore_20261102/, runs/e28_geometry_20261103/report.md, docs/MODEL_DIRECTION_20261102.md, docs/INTERACTION_FACTORIZATION_ANALYSIS_20261102.md, runs/e25_capacity_audit_20261102.md.
- Joint order A7+B7+L20+R20 radians; A=image-left arm, B=image-right. Times in Brasilia (BRT = host CST - 11h).
- Do not touch trainer outputs or overwrite earlier runs; write only the new folder, log, worklog.

## Constraints
Evidence, not claims. If two files disagree, say so and which you trust. Do not hide a failed or unmeasured gate.

## Report back
Summary <=15 lines; the report path; list of numbers you could not verify; suggested next step.
