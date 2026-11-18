# SPEC E27a: loss balance and latent residual (Claude, 2026-11-03)

Why (docs/MODEL_DIRECTION_20261102.md):
- E25 audit: future-RGB losses are 55% of the weighted loss, the JEPA latent term 4%.
- E26a: E24's latent prediction with the true action is worse than copy for every group (global grid 1.28x copy at H1,
  rising to 2.4x at H4); an action-free MLP is ~copy.
- E25 still fails the action-sensitivity gate (H=2: other 56%/48%, hold 65%/66%).

## Arms (all fine-tune from runs/ffn_E25_histdrop_20261102/latest.pt EMA, 4 epochs, same data/split/seed,
## same E25 code otherwise; effective batch 32)
| Arm | future_img, fore_img | roll_latent | image weighting | global latent prediction |
|---|---|---|---|---|
| A0 control | 5, 5 | 0.5 | change-weighted (as E25) | absolute (as E25) |
| A1 rebalance | 1, 1 | 2.0 | change-weighted | absolute |
| A2 object-weighted | 1, 1 | 2.0 | clean masks: hands, bottles, bin 1.0; rest 0.05 (normalized per region) | absolute |
| A3 no image | 0, 0 (keep recon 1.25 on frame t only) | 2.0 | n/a | absolute |
| A4 latent residual | 1, 1 | 2.0 | change-weighted | z_pred = z_t + delta, delta layer zero-initialized |
Masks for A2 and all object losses: clean cache from E26a (data/object_masks_clean_20261102, persistent bottle IDs,
cleaned hands, bin slot 8 now filled). Use the same clean cache in all arms (so the bin slot is supervised in all).

## Engineering
- Encode only the future frames actually used by the sampled rollout length (E25 encodes all 6). Verify identical
  loss values on one batch vs the old path, then use it in all arms.
- Log per-term gradient L2 norm at (a) the last trainable DINO block and (b) the dynamics, every 50 steps, for every loss.
- GPUs on <gpu-server> (user allowed free GPUs, 2026-11-03): A0, A1, A4 on GPU 2; A2 on GPU 6; A3 on GPU 4.
  Check free memory before each launch (need ~22 GB per run with micro-batch 8); if short, use micro-batch 4 x 8.
  Never touch other users' processes.

## Evaluation (each arm, same windows as E25)
1. Policy: chunk RMSE, first-5, onset, medoid-6 (E25 is the reference).
2. Copy-normalized latent error per token group (global, bottle_target, bottle_other, hands, bin) at H1-H4, moving vs
   static, with trajectory-bootstrap CI: reuse tools/e26a_baselines_20261102.py logic.
3. Object prediction vs clean masks; hands also vs FK point (check only).
4. Counterfactual top-1; action-swap sensitivity per group (stage0 D2 style).
5. E23a set-b value test (real + P1..P4, 4 seeds): the main gate (H=2: demo vs other >= 70%, vs hold >= 80%).
6. Gradient-share table per arm.

## Decision rule
Primary: value gate and copy-normalized latent error (global and bottle_target, moving windows). Secondary: policy RMSE
(no worse than E25 + 3%). Report all arms in one table with E25 and E24 rows.

Outputs: runs/e27a_<arm>_20261103/, runs/e27a_chain_20261103/ (status/, summary.md comparing arms),
log entry E27a in docs/experiment_log/EXPERIMENT_LOG.md (atomic, re-read before writing).
