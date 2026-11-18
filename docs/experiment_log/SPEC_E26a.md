# SPEC E26a: object-token audit and motion-stratified baselines (Claude, 2026-11-02)

Why: the gated factored world model (E26, docs/INTERACTION_FACTORIZATION_ANALYSIS_20261102.md) only works if
object slots are correct. Known problems: E25 stage0 D1 - E24 misses 62% of active bottles at t+0 (median IoU 0);
E21 - 1,816 suspected bottle identity switches, 5.3% frames with no bottle; D2 - bin "real change" exactly 0.0000;
raw SAM3 hand labels included forearms (already fixed by E22b-fix cleaned cache).

GPU: use GPU 0 (CUDA_VISIBLE_DEVICES=0) on <gpu-server> (user approved GPU 0 or 6 on 2026-11-02; GPU 0 has
~35 GB free and is idle). Check free memory first; if GPU 0 is full, use GPU 6. Never touch GPU 4 (E25 is training).

## Part A: bin token check (first, small)
Find how the bin token and its "real change" in tools/e25_stage0_20261102.py / e24 code are computed. Is the bin
mask fixed (one frame / constant), or the latent pooled from a frozen frame, or truly constant? Report the cause
and the real per-frame variation of (a) the bin mask area/centroid, (b) the encoder bin token.

## Part B: bottle-track audit
On the E21 SAM3 outputs (data/object_masks_sam3_20261101), per episode:
- count bottles visible, empty masks, tracks per frame, identity switches (centroid jump > 25 px at 640x384 between
  consecutive frames while IoU < 0.3), merges/splits, masks overlapping a hand mask (cleaned cache), masks in the bin.
- separate: frames where a bottle is visible but SAM3 has no mask (check by eye on 24 sampled frames: make a contact
  sheet) vs frames where SAM3 has a mask but E24's token pipeline drops it.
Build cleaned persistent bottle IDs: Hungarian matching over time on mask IoU + centroid distance with track
re-identification after short occlusion (<= 10 frames); keep a visibility flag. The grasped target bottle must keep its
ID while held (allow overlap with the hand mask). No manual labels; no FK.
If SAM3 truly misses bottles in > 5% of frames where one is visible (from the contact sheet), re-run SAM3 on those
episodes on GPU 0 with box/point prompts propagated from neighbouring frames; otherwise skip.
Output: data/object_masks_clean_20261102/ (bottles with persistent IDs + cleaned hands + bin), audit.json, contact
sheets before/after.

## Part C: motion-stratified baselines on E24 latents (GPU 0)
Re-encode validation windows with E24's encoder using the cleaned masks (object tokens from clean slots).
For token groups global / bottle (target bottle vs other bottles) / hands / bin, at H = 1..4 chunks, report latent
error normalized by copy error for:
- copy (=1 by definition), constant velocity from the last two frames, an action-free predictor (small MLP/transformer
  on token history, trained on train split, GPU 0, <= 1 h), and E24 dynamics with the true action.
Split by: moving vs static (cleaned mask centroid displacement > 10 px), task phase (approach / grasp / carry / release
from the held flag and hand-bottle overlap), visible vs occluded. Trajectory-level bootstrap 95% CI.
Also the action-swap sensitivity of each group (as stage0 D2) with cleaned tokens.

## Outputs
runs/e26a_audit_20261102/: bin_check.json, bottle_audit.json, contact sheets, baselines.json, summary.md.
Log entry E26a in docs/experiment_log/EXPERIMENT_LOG.md (atomic; another job updates E25 concurrently, so re-read
before writing). One detached host chain with status files. No system installs; python
/mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python; never use wrist cameras; never access <ip-redacted>.
