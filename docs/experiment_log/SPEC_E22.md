# Spec E22: label-free verification of object masks via robot kinematics (design by Claude, 2026-11-01)
Rules as SPEC_E18_E19.md / SPEC_E21.md (GPU 4 shared, no installs outside /mnt/workspace/dengjianfeng, atomic log edits, wrist cams never used).
Goal: no recurring manual labelling. Masks come from E21 (Grounding DINO + SAM2); robot kinematics + consistency checks give each mask a
confidence; humans only look at a handful of flagged frames once. Only key objects: target bottle(s), left/right hand, bin.
Assets (find the exact absolute roots; likely under /mnt/workspace/colleague-c or the project's third_party):
  URDF+STL: third_party/RoboDojo/third_party/curobo/robot/tianji_wujihand/urdf/
  arm/hand meshes: third_party/OpenWAM/tools/wuji_dataset_studio/robot_assets/meshes/
No camera calibration exists -> estimate it.

Stage A (can start now, CPU mostly): kinematics
 1. Parse the URDF (pure python / existing package in a workspace venv: urdfpy, yourdfpy, pytorch_kinematics or curobo if already present).
    Map our 54-D joint order A7+B7+L20+R20 (rad) to URDF joint names; verify on a few frames that limits/ranges match and the hands' 20 dofs map.
    Write the mapping to data/tianji_kinematics_20261101/joint_map.json with evidence.
 2. FK for all frames of all 93 episodes: 3D positions (robot base frame) of per-arm wrist, palm centre, 5 fingertips; save npz.
 3. Simple 3D sanity: palm distance between arms, workspace extents, fingertip spread vs hand-closure.
Stage B (after E21 masks exist): overhead camera self-calibration
 4. Fit a pinhole camera (R, t, f; principal point = image centre of the ORIGINAL uncropped camera image, then map to our 192x320 crop
    = rows 96..480 resized) by maximizing agreement between projected hand geometry and E21 hand masks: start with palm/fingertip points
    -> mask centroid / inside-mask fraction, then refine with rendered mesh silhouettes (pyrender/nvdiffrast if available, else point-sampled
    mesh vertices) vs mask IoU. Robust loss; use many frames from training episodes; hold out validation episodes.
    Report: projected-vs-mask IoU and centroid error (px) on held-out episodes; median/P90. Save camera params.
 5. Per-frame mask confidence (all episodes):
    hand: IoU(rendered silhouette, SAM2 hand mask), left/right identity agreement;
    bottle: (a) static unless a hand is within contact distance (projected fingertips touching the mask), (b) while held (hand closure high
    and in contact) the bottle mask moves with the projected palm (rigid-follow residual), (c) at episode end the bottle lies inside the bin mask,
    (d) forward/backward SAM2 propagation agreement (cycle IoU).
    Combine into confidence in [0,1]; output data/object_masks_sam2_20261101/confidence.npz; list the 30 lowest-confidence frames with a
    QC sheet (the only frames a human may look at, once).
 6. Re-score E18 using only frames with confidence >= 0.8 and report how many remain.
Log entry E22 before launch; results after; conclusion "pending (Claude)". Copy reports/figures here with prefix e22_.
Do Stage A now. Start Stage B only when E21 has finished (check data/object_masks_sam2_20261101 and the E21 log status); if E21 is not done
when Stage A finishes, wait for it (poll every 10 min, max 6 h).

NOTE 2026-11-01: E21 now uses SAM3; masks in data/object_masks_sam3_20261101 (symlinked as data/object_masks_sam2_20261101). Treat them as the E21 masks.
