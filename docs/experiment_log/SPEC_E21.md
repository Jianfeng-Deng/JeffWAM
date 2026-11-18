# Spec E21: precise object masks with SAM3 (CHANGED 2026-11-01: user prefers SAM3; the Grounding DINO + SAM2 attempt was stopped at its 2-episode test) (design by Claude, 2026-11-01)
Rules as SPEC_E18_E19.md, except: SAM / VLM are now ALLOWED (user, 2026-11-01). Wrist cameras are removed from the robot: never use them.
User approved downloads: SAM 2.1 Hiera-Large (official Meta release) and Grounding DINO base (IDEA-Research, Hugging Face) into
/mnt/workspace/dengjianfeng/models/ (record URL + SHA256). Any python package goes into a venv under /mnt/workspace/dengjianfeng (e.g. clone the
borrowed venv or `--target` dir); never into the GPU host system env or the borrowed venv. GPU 4 is shared with E19/E20: stay < 16 GB, never kill them.
Log: atomic single-entry edits only (another job edits the log too).

1. Data: overhead frames data/real_workspace_192x320_20261031 (all 93 episodes). Also try the original higher-res overhead frames if available
   in the raw dataset (better for transparent bottles); masks must finally be stored on the 192x320 crop grid.
2. Per episode: Grounding DINO on the first frame (and re-detect every 30 frames to catch new/lost objects) with prompts
   "transparent plastic bottle . bottle . cup . robot hand . black glove . bin . basket"; tune box/text thresholds on a few episodes.
   Seed SAM 2 video predictor with the boxes, propagate forward through the episode; keep object ids; merge duplicate tracks (IoU > 0.7).
   Classes: bottle (with sub-type transparent / opaque by mean saturation+contrast), hand_left, hand_right (left/right by image side +
   continuity), bin. Store: data/object_masks_sam2_20261101/{masks_192x320 uint16 instance ids (npz per episode or one memmap), objects.json
   (id -> class, episode, frames present)}, plus 24x40 and 12x20 occupancy per class.
3. Quality check: contact sheet of 60 random frames (stratified by episode phase, include validation episodes) with mask overlays and ids
   -> runs/object_masks_qc_20261101/qc_sheet_*.png for the user to spot-check. Report: tracks per episode, frames with zero bottle, id switches
   (sudden centroid jumps > 3 cells), mean mask area per class. Compare with the k-means cluster masks (IoU per class) to show what changed.
4. Re-run E18 with these labels (bottle = SAM2 bottle mask; hand = hand masks) for the predicted render vs copy vs current rebuild: centroid
   error in px and IoU. For rendered images use the SAME detector+SAM2 per image (no propagation) as for the real frame of comparison, so both
   sides get the same labeller. Output runs/future_pos_E_sam2_20261101.
5. Log entry E21 (template) before launch; results after; Conclusion "pending (Claude)". Copy QC sheets + results json here with prefix e21_.
Final stdout: tables, QC stats, deviations + reasons.

## CHANGE (overrides steps 1-2 where they conflict): use SAM3, not Grounding DINO + SAM2
Weights: /mnt/workspace/dengjianfeng/eyerobot2-top-stereo/models/sam3/sam3.pt (record SHA256). Code: /mnt/workspace/dengjianfeng/eyerobot2-top-stereo/third_party/sam3
(also a copy at /mnt/workspace/dengjianfeng/pi05_v8_offline_env_20261027/snapshots/sam3). Find and reuse an existing python env that already runs SAM3
(look in eyerobot2-top-stereo and pi05_v8_offline_env_20261027, e.g. outputs/*sam3* runs show it was used); otherwise make a venv under
/mnt/workspace/dengjianfeng. Do not download new weights. Use SAM3's text-prompted video predictor (concept prompts: "transparent plastic bottle",
"bottle", "robot hand", "bin") for detection + tracking with ids; re-prompt/re-detect every 15 frames; merge duplicate tracks.
Lessons from the user's active-gaze project: tracking drifts and wrong identities hide behind good coverage -> keep ids by mask IoU/centroid,
bind a grasped bottle to the closing hand, report id switches. Only key objects: target bottle(s), left/right hand, bin.
Output dir: data/object_masks_sam3_20261101, and create a symlink data/object_masks_sam2_20261101 -> object_masks_sam3_20261101 (E22 expects that path).
Log entry E21: say the GDINO+SAM2 test was stopped and SAM3 used instead. Keep the stopped E21 partial files (tmp_e21) untouched.
