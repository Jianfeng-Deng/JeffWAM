# SPEC E24r: re-score E24 hand prediction with clean targets (Claude, 2026-11-02)

Why: E22b-fix showed raw SAM3 hand_left/hand_right masks include forearms and switch sides. E24's hand metrics
(centroid error 82-86 px, pred-beats-copy 41-56%) used those masks, so they are unreliable. Re-score with two
clean targets. Evaluation only: no training, no change to E24.

## Targets (both at 640x384 crop pixels; E24 predictions converted to the same frame)
T1 (label-free, main): calibrated FK hand point. Oct 15 head_world_robot calibration as used in E22b, offset-only
   variant (fits in runs/e22b_calib_check_20261102_fix/fits.json), A = image-left = hand_left,
   B = image-right = hand_right. Vendor MDH FK from dataset joints (radians -> degrees).
T2: cleaned SAM3 hand target exactly as in E22b-fix (hand-only component, side reassignment from image only).
Bottle is not re-scored (no FK); report the E24 bottle numbers unchanged for reference.

## What to compare, per hand, at t+10/20/30, validation windows (same 2109 windows as E24 eval)
- E24 predicted hand centroid (from its predicted mask / centroid head, exactly as E24 eval did).
- Copy baseline: the target value at time t (T1 at t for T1; T2 at t for T2).
- E24 own t+0 decode vs target at t (encoding error; splits encoding from dynamics).
Metrics: median and p90 centroid error (px); for T2 also IoU of predicted mask with the cleaned mask.
Moving windows: T1 hand displacement t -> t+k > 10 px (defined from FK, label-free).
Report pred-beats-copy % on moving windows, and error on static windows.
Also report a systematic-offset check: mean signed (dx, dy) of predicted minus T1, to detect that E24 learned the
"hand" as hand+forearm (a constant shift toward the arm).

## How
If the E24 evaluator saved per-window predictions, use them. Otherwise run a light forward pass of E24 (EMA weights,
runs/ffn_E24_objaware_20261102/latest.pt) on GPU 4 that saves per-window predicted object states only. GPU 4 is shared
with E25 setup: check free memory first, use small batches, do not disturb the E25 job.
Script: tools/e24r_hand_rescore_20261102.py. Output: runs/e24r_hand_rescore_20261102/ (results.json, summary.md,
8 overlay images: frame t+k with predicted hand point, T1 point, T2 outline). Add log entry E24r to
docs/experiment_log/EXPERIMENT_LOG.md (atomic). Detached host chain if longer than a few minutes.
