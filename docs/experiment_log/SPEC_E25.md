# SPEC E25: break the history shortcut (Claude, 2026-11-02)

## Why
E24 distinguishes actions in latent space (counterfactual top-1 0.93) but the value heads see no difference in task
progress (demo vs other 52-54%, vs hold 61-63% at H=2), and object prediction is worse than "nothing moves".
Hypotheses:
- H1 (main): the 1 s joint history already tells the dynamics where the arm is going, so it can predict the
  future while mostly ignoring the action input.
- H2: 0.67 s targets are close to "continue past motion"; the action matters more at longer horizons
  (E24 sensitivity was higher at 3-4 chunks than at 1-2).
- H3: the object head predicts absolute state and so starts worse than copy; predicting change fixes the baseline.
- H4: the action effect E24 learned lives on the hand part of the latent, not on the bottle.
No robot-specific inputs (no FK or calibration in the model), per the user's generality requirement.

## Stage 0: diagnostics on E24, no training (run first, ~1 h)
D1. Object decode at t+0 (current frame through the same object head): centroid error and IoU vs SAM3 for
    bottle / hands. Splits encoding error from dynamics error.
D2. Where the action effect lives: for 512 validation windows, roll out demo vs hold / reversed / other actions
    (same alternatives as E23a) for H=1..4 chunks; report mean latent distance separately for global tokens,
    bottle tokens, hand tokens, bin token, each normalized by the same distance between two seeds of the demo.
D3. History reliance: E24 dynamics with the joint history replaced by the current q repeated (no motion): change in
    rollout error. Large change = H1 supported.

## Model changes (E25 = E24 code, initialized from E24 latest.pt EMA weights)
C1. History dropout in the dynamics only. With p=0.5 per sample, the dynamics sees the joint history replaced by
    the current q repeated (zero velocity), otherwise the true history plus Gaussian noise sigma=0.02 rad.
    The policy (flow head) always keeps the full clean history. If history reaches the dynamics only through tokens
    shared with the policy, build a separate dynamics context with the masked history; document how.
C2. Random-length rollout. Each batch samples an unroll length L uniformly in {1..6} chunks (10..60 steps); losses at
    every chunk end (latent + object), weight 1/L each. Gradient through at most 3 chunks (stop-grad on the carried
    state before chunk 4). Valid only where the episode has enough future; otherwise shorten L.
C3. Residual object head. Predicted object state = current-frame object state + delta. Centroid/box: delta in pixels;
    mask: logits_t + delta_logits; presence/held: logit residual. Initialize the delta layers to zero, so the
    starting model equals copy. Current-frame state at training = SAM3 teacher (schedule as E24: teacher -> own
    t+0 prediction 30%->70%); at evaluation = own t+0 prediction only.
C4. Inverse-dynamics auxiliary: from (z_t, z_{t+10}) predict the 10-step action chunk (MSE on normalized actions),
    weight 0.1, small MLP on pooled global + object tokens.
Keep everything else as E24 (L_cf weight 0.5 with its masking, 336x560 input, effective batch 32, lr 3e-4 for new
parts and heads, dino_lr 2e-5). Train 12 epochs from E24.

## Runs
- E25 (C1-C4). Then, only if E25 improves the H=2 sensitivity over E24 by >= 5 points on either alternative:
- E25-noHD (C2-C4 without C1), same budget, to measure what history dropout contributes.
Before training: check free memory on GPU 4 (shared; others use ~67 GB now) and pick the micro-batch so effective batch
is 32 with accumulation. Report time per optimizer step after 100 steps; stop and report if 12 epochs > 14 h.

## Evaluation (same evaluator and windows as E24, plus D1-D2 repeated on E25)
1. Policy: chunk RMSE, first-5, onset, medoid-6; E24 and E with the same code.
2. Object prediction vs SAM3 at t+10/20/30: error and IoU, copy baseline, pred-beats-copy on moving windows,
   and error on static windows.
3. Counterfactual top-1.
4. E23a set-b value test on E25 latents (real + P1..P4, 4 seeds, V_pool and V_box), beside E23a and E24.
5. D2 token-group sensitivity for E25 vs E24.

## Pass criteria
- Main gate (unchanged): demo vs other >= 70% and demo vs hold >= 80% at H=2.
- Object: moving-window pred-beats-copy >= 65% for bottle and hands at t+10; static-window error <= copy + 3 px.
- Policy: chunk RMSE no worse than E24 + 3%.
Partial result worth reporting: H=2 sensitivity up >= 5 points, or the D2 share of action effect on bottle tokens up.

## Engineering rules (lessons from E24)
- Smoke test the FULL evaluator (all metrics, 64 windows) on the initial checkpoint BEFORE training starts.
- One detached host chain (setsid nohup) with status files; it writes the E25 log entry itself (atomic) and saves
  every result JSON before writing summaries/figures.
- Use GPU 4 only (CUDA_VISIBLE_DEVICES=4) on <gpu-server>; no installs into system Python; never use wrist
  cameras.
Outputs: runs/ffn_E25_histdrop_20261102/, runs/e25_chain_20261102/, tools/e25_*_20261102.py.

## AMENDMENT 1 (Claude, 2026-11-02): C1 corrected
Finding: E24's dynamics never sees the 15-step joint history (only the policy does). It sees two image frames
(previous + current) plus the current joint position and joint velocity. Motion information therefore reaches the
dynamics through the velocity input and the previous frame, not through a history input.
C1 (replaces the original): with p=0.5 per sample, the dynamics gets (a) joint velocity set to zero and (b) the
previous frame replaced by the current frame (a static pair). Otherwise the true inputs with Gaussian noise
sigma=0.02 rad on the joint position and velocity. Do NOT add a history input to the dynamics. The policy keeps its
inputs unchanged. Stage 0 D3 becomes: E24 rollout error with velocity zeroed and previous frame = current frame.
E25-noHD = C2-C4 without this C1.
Micro-batch: GPU 4 now has ~90 GB free; re-check at launch and use the largest micro-batch that fits with 15% margin,
effective batch 32.

## AMENDMENT 2 (Claude, 2026-11-02)
1. D2 normalization (E24 dynamics is deterministic, so seed-to-seed distance is 0): report, per token group and H,
   (a) the raw mean distance between the demo-action rollout and the alternative-action rollout, and
   (b) that distance divided by the real change of the same token group, ||z_real(t+H) - z_real(t)|| from the
   encoder on real frames ("fraction of real change that the action alters"). Also (c) the demo-rollout error
   ||z_demo(t+H) - z_real(t+H)|| for scale.
2. Hand targets. E24r showed E24's hand head learned the raw SAM3 hand_left/hand_right labels, which include forearms
   and switch sides (t+0 hand error ~140 px vs the calibrated FK hand point; predictions sit on forearms).
   For E25 training AND evaluation, replace the hand targets with the E22b-fix cleaned hand-only components with
   image-only side assignment (same code as runs/e22b_calib_check_20261102_fix; label-free, no FK used to choose).
   Bottle and bin targets unchanged. Precompute the cleaned hand masks once for all 93 episodes.
   Evaluation additionally scores hands against the calibrated FK hand point (T1 in SPEC_E24r) as a check only;
   FK is never a model input or training target.
3. Pass criteria for hands use the cleaned targets; E24 hand numbers to compare against are the E24r T2 rows.
