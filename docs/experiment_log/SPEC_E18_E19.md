# Spec E18 + E19 (design by Claude, 2026-11-01). Implement exactly; ask nothing; record everything.

Project root (shared): /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu. Files via `ssh dengjianfeng-101`; GPU runs on
`<gpu-server>` with CUDA_VISIBLE_DEVICES=4 ONLY, python /mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python.
Never pip install into system env (any install goes under /mnt/workspace/dengjianfeng). No VLM. Never touch <ip-redacted>.
Launch long jobs with `setsid nohup ... > run_dir/train.log 2>&1 &`. Use `[p]ython` patterns in pgrep/pkill. Run E18 first, then E19 (one GPU).
Reuse existing tools: train_ffn_cd_20261031.py (WindowsCD, unpack), train_foresight_flow_20261031.py (world),
train_progress_scorer_20261101.py (load_e), train_region_jit_decoder_v2_20261101.py (RegionJiT v2, select, gather, latent_to_grid, sample),
render_action_contrast_20261101.py (how to get z^(t+10) under any action chunk: wm.dyn(z_hist, pad, prop, act, zeros)[0][:,1]),
cluster_dinov2_regions_20261101.py + data/dinov2_regions_20261101/kmeans.pt (how clusters were assigned).
E run: runs/ffn_phaseE_dinob_qhist15_20261031 ; region-JiT v2: runs/region_jit_v2_E_latentonly_20261101/region_jit.pt (ema weights);
boxes: data/dinov2_regions_20261101/boxes_v2_24x40.npy ; split file = the one used by earlier runs (see E17 / E16 commands).

## E18: does the predicted future put objects in the right place? (eval only, ~minutes)
256 evenly spaced validation windows. For each: real t, real t+10, JiT v2 render of predicted t+10 (demo action), JiT v2 render of current t
(recon upper bound). Renders: inside union boxes, outside filled with real t+10 (so outside pixels cannot help), 50 Euler steps, 2 seeds.
Run DINOv2-B on each image exactly as in cluster_dinov2_regions, assign k-means clusters on the 24x40 grid.
For cluster 15 (bottles/cups) and cluster 11 (hands/dark bottles): mask centroid (patch units) and mask IoU against real t+10.
Report for: copy (real t used as the prediction), predicted render, current-recon render (vs real t, as upper bound).
Report on all windows AND on "moving" windows (real centroid shift t->t+10 > 1 patch for that cluster); skip a window if the cluster is absent.
Key numbers: centroid error (patches), IoU, and % of moving windows where predicted beats copy. Save json + a figure of 6 worst moving windows
(real t | real t+10 | pred render, with centroid dots). Script: tools/eval_future_object_position_20261101.py, out runs/future_pos_E_20261101.

## E19: action scorer v3 (counterfactual, local outcome)
Goal: score a candidate action chunk; must pick good E samples. Avoid the old failures: (v1) real-vs-generated shortcut, (v2) "prefer low noise", (progress) global phase.
1. Candidates (precompute, tools/gen_scorer_cands_E_20261101.py -> data/scorer_cands_E_20261101, train + val windows, normalized action space):
   demo; 6 E flow samples (3 at noise 1.0, 3 at noise 0.5, fixed seeds); hold; reversed; other-window chunk (random other window, same split);
   demo shifted +3 and -3 steps (pad by repeating edge). K = 12.
2. Target quality per candidate: q = -RMSE(cand, demo) in normalized action units over the 10x54 chunk, steps 0-4 weighted x2. Demo q = 0.
   Graded target => generated samples are not "fake", they are ranked by closeness, which removes the real-vs-generated shortcut.
3. Three scorer variants, same data, same loss (this is the key ablation: does the world model add value?):
   A  action-only: input pooled current latent z_t + 1 s joint history + candidate chunk (MLP over flattened chunk).
   W  world-only: input box tokens of z_t and dz = z^(t+10)(cand) - z_t on the 12x20 grid, only tokens inside union boxes of t
      (boxes pooled 24x40 -> 12x20 with max), + joint history; transformer pooling (2 layers, d 256, 4 learned queries). No action input.
   AW both (W + action embedding as extra tokens).
   E frozen; z^(t+10)(cand) computed on the fly under no_grad bf16 (or cache if memory allows). Train 15 epochs, AdamW lr 3e-4, wd 0.05, bs 32 windows.
4. Loss: listwise KL( softmax(q/tau) || softmax(score) ) per window, tau = 0.5 * median within-window std of q over train;
   + 0.5 * pairwise hinge margin 0.1 on demo > {hold, reversed, other, shift+3, shift-3}.
5. Validation (all 2,289 val windows), per variant:
   a) selection among the 6 E samples only (realistic case): chunk RMSE (rad) and first-5 RMSE of scorer pick vs random vs medoid
      vs best-at-noise-0.5-medoid vs oracle; also grasp onset error / missed / false onsets of the pick (reuse onset from train_foresight_flow).
   b) counterfactual accuracy: P(score demo > c) for c in hold, reversed, other, shift+3, shift-3.
   c) mean within-window Spearman(score, q) over all 12 and over the 6 E samples.
Script tools/train_action_scorer_v3_20261101.py --variant {A,W,AW}; runs runs/scorer_v3_{A,W,AW}_20261101. Smoke test 50 steps first.

## Logging (mandatory)
Before each launch add an entry (E18, E19) to docs/experiment_log/EXPERIMENT_LOG.md using the "How to add an entry" template
(goal, change, command, run dir, status running). After finishing fill Results with tables; Conclusion = "pending (Claude)".
Also list new scripts under "Code changes". Copy the final log to the local scratchpad as EXPERIMENT_LOG.md.
Final stdout report: what ran, key numbers (tables), anything that deviated from this spec and why.
