# Spec E24: object-aware, action-sensitive world model (design by Claude, 2026-11-02)
Rules as SPEC_E18_E19.md (GPU 4 only, borrowed python / workspace venvs, no system installs, wrist cameras never used, atomic log edits).
SAM3 allowed (training labels). IMPORTANT robustness rule: every long job runs as a detached chain on the host (setsid nohup) that does
not depend on the Codex session staying alive; write a small status file per stage.

Why (findings): E20 — bottle detail is lost at the 12x20 encoder input (8-px bottle AP 0.46 vs 0.92 at 24x40).
E23a — a value head learns time-to-success (Spearman 0.93) but predicted futures barely depend on the action
(demo vs another window's chunk = chance). So the dynamics must carry what the action does to the key objects.

## Model (start from run E code: tools/train_ffn_cd_20261031.py NetCD/DinoEncoder; init from E best.pt where shapes match)
1. One DINOv2-B pass at 336x560 -> 24x40 patch features (last 4 blocks fine-tuned, anchor 0.5 as in E).
   Global tokens = 2x2 avg-pool -> 12x20 (same grid as E, so E's dynamics/policy weights load).
2. Object tokens (K_max = 9: up to 6 bottles, left hand, right hand, bin; padded + presence flag):
   masked mean of 24x40 features inside the object mask + attention pooling, + sincos(centroid, box size) + class embedding.
   Masks: SAM3 masks from E21 (data/object_masks_sam3_20261101, downsampled to 24x40) during training. To avoid needing SAM3 at
   inference, add a mask head on the 24x40 features (per class, instance-agnostic for bottles via 6 query slots, DETR-style matching)
   trained on SAM3 masks; schedule: teacher masks for the first 30% of steps, then linearly switch to predicted masks (100% predicted
   by 70%). If E22 confidence exists (data/object_masks_sam2_20261101/confidence.npz), weight object losses by it; else weight 1.
3. Dynamics: same AdaBlock transformer over [240 global + K object] tokens, action-conditioned as in E. Policy head unchanged
   (flow head + foresight), reading global + object tokens.

## Losses (on top of E's: recon/future pixel L1 at 12x20 decoder, anchor, policy flow, past-token, foresight)
L_obj  For predicted object tokens at t+10, t+20, t+30 (multi-step, see L_roll): heads predict per object centroid (px), box (px),
       24x40 mask (BCE+dice for bottles/hands), presence, and held flag (bottle held = its track moves rigidly with a hand; derive
       from E21 tracks + E22 FK if available, else from mask contact + hand closure). Targets from SAM3 at the real future frames.
L_roll Unroll dynamics 3 chunks on its own predicted latents (chunk 2,3 inputs = predicted, gradients flow; detach every other step
       if memory needs). Latent loss vs encoder target tokens (stop-grad, cosine) + L_obj at each step. Pixel loss only at chunk 1.
L_cf   Counterfactual contrast (the key new term): per window roll out 1 chunk under the demo action and under M=7 alternatives
       (4 other-window chunks, hold, reversed, demo shifted +3). Score each by -L_obj(predicted object state, REAL future object state).
       InfoNCE over the 8 with temperature 0.1: the demo action must best explain the real object outcome. Weight 0.5.
Weights to start: L_obj 1.0, L_roll latent 0.5, L_cf 0.5; keep E's other weights. Tune only if smoke shows a term dominating.

## Training
Same split, 25 epochs (E's best was ep 20), AdamW as E, bf16, batch to fit < 60 GB. Smoke test 200 steps, report s/step and
memory; launch full run automatically if the projected time is <= 36 h, else halve epochs and report. Run dir runs/ffn_E24_objaware_20261102.
Ablation (after the full run, only if time allows): same config without L_cf -> runs/ffn_E24_nocf_20261102.

## Evaluation (pass/fail vs E; validation episodes)
1. Policy: chunk RMSE, first-5 RMSE, onset error / missed / false onsets (single sample and medoid of 6). Pass: no worse than E by >3%.
2. Object prediction (SAM3 labels): centroid error (px) and mask IoU at t+10/20/30 for target bottle and hands vs copy-current and
   vs E (E via region-JiT render + SAM3 as in the E21 re-score). Moving windows separately.
3. Counterfactual identification: given the real future object state, top-1 accuracy of picking the demo among the 8 actions.
4. Repeat E23a with the new latents (retrain the shift-robust value heads, set b): P(demo better) vs hold / reversed / other
   for H = 1..4. Pass: demo vs other >= 70% and demo vs hold >= 80% at H = 2.
Log entry E24 before launch; results after; Conclusion "pending (Claude)". Copy summaries/figures to the scratchpad with prefix e24_.
