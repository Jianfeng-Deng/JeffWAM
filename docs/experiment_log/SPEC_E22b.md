# SPEC E22b: does the 2026-10-15 calibration fit our dataset? (Claude, 2026-11-02)

Goal: replace E22 Stage B (failed camera self-calibration) with the measured calibration, and check it
against our 93-episode dataset with no manual labels. CPU job (no GPU needed). Detached host chain.

## Inputs
- Calibration: /mnt/workspace/dengjianfeng/data/camera_calibration_20261015/camera_calibration/
  - head_world_robot_20261015/calibration.json (top RGB intrinsics, T_world_from_base A/B, T_flange_from_rgb A/B,
    vendor MDH), robot_world.py (offline FK; input joints in DEGREES), quality_summary.json, CALIBRATION_STATUS.md
  - 20261014_rgb_factory_intrinsics.json (resolution the intrinsics refer to)
  - world = top RGB optical frame (X right, Y down, Z forward), metres, column vectors.
- Dataset: /mnt/workspace/colleague-e/pi05_lingjun_joint54_zero2_b256_20261018/datasets/lerobot_optical93_eef72_15hz_20261016
  joints A7+B7+L20+R20 in RADIANS; split = meta/training_split.json. Wrist cameras: never use.
- SAM3 masks: data/object_masks_sam3_20261101 (hand_left, hand_right tracks; 640x384 native crop as used in E21).
- E22 Stage A: data/tianji_kinematics_20261101 (URDF FK) for cross-check only.

## Step 0: image geometry (must be resolved before any number is reported)
Find the top camera resolution the intrinsics refer to and how dataset frames are produced from it
(crop offset, resize). Derive K for the 640x384 frames used by SAM3 and for 192x320. Write the derivation
to geometry.json. If the dataset frames are not a plain crop/resize of that camera, stop and report.

## Step 1: projection with calibration as-is
For every frame (or every 3rd frame) of every episode: MDH FK with the vendor model for both arms
(degrees conversion!), get flange origin and the 7 joint-frame origins in world; project with K (+ distortion).
Cross-check the MDH flange against E22 URDF FK (same arm, base frame): report max position difference.

## Step 2: arm-to-image mapping, label-free
Decide which of A/B is image-left / image-right by comparing projected flange tracks with SAM3 hand_left /
hand_right centroid tracks (correlation of image motion). Report the mapping and its margin.

## Step 3: hand point offset (3 parameters per arm)
SAM3 hand-mask centroid is not the flange. Fit a constant 3-D offset p_hand in the flange frame per arm
(least squares on reprojected centroid error, robust Huber loss) on TRAIN episodes only. Also fit a variant
with a 6-DoF base correction on top of the calibration (A arm especially), again train only.

## Step 4: metrics on VALIDATION episodes (held out), per arm, for: (a) flange as-is, (b) + offset,
(c) + offset + base correction
- centroid error px (median, p90) at 640x384
- fraction of frames where the projected hand point lies inside the SAM3 mask dilated by 5 px
- error vs image position and vs time (drift within an episode -> sync or mount change)
- frames where SAM3 hand is missing are skipped and counted.

## Pass criteria (per arm, variant b, validation)
median centroid error <= 12 px and in-mask >= 80%  -> usable for hand conditioning.
B arm is expected to pass; A arm may need variant c. If c fixes A on validation with a base correction
< 30 mm / 3 deg, use it and record it as a self-correction, not a calibration.

## Outputs
runs/e22b_calib_check_20261102/: geometry.json, mdh_vs_urdf.json, arm_mapping.json, fits.json, metrics.json,
summary.md, overlays/ (16 validation frames across stages: projected joint chain + hand point over the image,
SAM3 hand mask outline; filenames listed in summary), per-episode error plot.
Update docs/experiment_log/EXPERIMENT_LOG.md with entry E22b (status running -> done/failed) automatically.
Do not use or modify the robot. Do not install into system Python
(use /mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python or a venv under /mnt/workspace/dengjianfeng).
