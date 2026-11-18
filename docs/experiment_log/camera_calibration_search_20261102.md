# Tianji top-camera calibration search — 2026-11-02

## Finding

The strongest direct top-camera calibration is the 2026-10-17 capture 02 result at:

- Source: /home/tiancai/code/tianji_wuji_deploy_code/data/camera_calibration/project_20261015/tianji-camera-calibration/results/top_world_20261017/capture_02_confirmed/calibration.json
- Destination copy: /mnt/workspace/dengjianfeng/data/camera_calibration_20261015/camera_calibration/project_20261015/tianji-camera-calibration/results/top_world_20261017/capture_02_confirmed/calibration.json

It maps the top RGB optical frame to a board/table world frame. It does not estimate a robot-base extrinsic. The top RGB intrinsics are well identified for serial CP28563000JZ at 640×480. The board fit is repeatable in pixels, while independent table-wide metric accuracy remains unverified.

The Tianji calibration project's selected working default is a separate 2026-10-16 nominal-geometry result, nominal_geometry_single_group_A_00/calibration.json. The project's own configuration says this is the default, but marks it unaccepted for metric use. OpenWAM's raw54-to-SDK-flange converter records calibration_used=false. The offline keyframe geometry helper reads the 2026-10-17 table result plus derived A/B base estimates; it is not the OpenWAM conversion/training path.

## Search, copy, and inventory

The requested candidate directory exists. The search covered the parent camera_calibration directory and its sibling result folders, a max-depth-limited filename search under /home/tiancai for calib, extrinsic, intrinsic, hand_eye, handeye, camera_info, and cam2base names, and focused greps of tianji_wuji_deploy_code source. Other home-directory matches were unrelated session data; no additional Tianji top-camera candidate was identified outside data/camera_calibration.

Source inventory under data/camera_calibration: 1,418 files, 537,450,537 bytes (512.55 MiB). Ten packaged release ZIP snapshots under project_20261015 total 392,735,144 bytes (374.5 MiB). The working project folder and sibling calibration result folders were copied with their relative structure. The filtered archive SHA-256 matched between source and target: 91a7010fd617a166bb67cd5bc5f5b73124f0ad0653c397ae3414010ce0888662. The target manifest matches 1,408 copied source files totaling 144,715,393 bytes. There is one additional target-only file, head_world_robot_20261015/__pycache__/robot_world.cpython-311.pyc (5,397 bytes; mtime 2026-11-02 14:58:51.735338364), absent from the source manifest; it was left untouched. The ten duplicate ZIP snapshots are listed here and omitted. The remaining payload is 1,408 files / 144,715,393 bytes (138.0 MiB). Direct image/video payload in the working folders is about 69.78 MiB, below the 200 MiB threshold; all direct image/video files in the copied folders are included. No .npz files were found under the calibration tree, so there are no NPZ key/shape inventories.

The ten omitted source archives are:
- project_20261015/tianji-camera-calibration_0.1.0.zip
- project_20261015/tianji-camera-calibration_0.1.1.zip
- project_20261015/tianji-camera-calibration_0.1.2.zip
- project_20261015/tianji-camera-calibration_0.2.0.zip
- project_20261015/tianji-camera-calibration_0.2.1.zip
- project_20261015/tianji-camera-calibration_0.2.2.zip
- project_20261015/tianji-camera-calibration_0.2.3.zip
- project_20261015/tianji-camera-calibration_0.2.4.zip
- project_20261015/tianji-camera-calibration_0.2.5.zip
- project_20261015/tianji-camera-calibration_20261015_working.zip

Upper-computer access was read-only and key-only for SSH/SCP. No source files were changed. One read-only hostname query was issued during the first connection; subsequent source access used the requested read operations and scp-out.

Source modification times below are the values reported by find on <ip-redacted>.

## Top camera intrinsics

The same factory intrinsics occur in:
- data/camera_calibration/20261014_rgb_factory_intrinsics.json — 2026-10-14 14:59:27.493256951
- project_20261015/tianji-camera-calibration/data/intrinsics/20261014_rgb_factory_intrinsics.json — 2026-10-15 18:45:55.637816453
- project_20261015/tianji-camera-calibration/results/top_world_20261017/provided_intrinsics.json — 2026-10-17 20:24:52.207282487

The top camera is Orbbec Gemini 335L, role top, serial CP28563000JZ, RGB resolution 640×480. Intrinsics are:
- fx 366.53265380859375, fy 366.5903625488281
- cx 321.3335876464844, cy 238.58607482910156
- K = [[366.5326538, 0, 321.3335876], [0, 366.5903625, 238.5860748], [0, 0, 1]]

Distortion model is OBCameraDistortionModel.BROWN_CONRADY_K6. OpenCV order [k1,k2,p1,p2,k3,k4,k5,k6]:
[-0.028631508350372314, 0.03352833166718483, 0.00035273877438157797, -0.00015485245967283845, -0.011858249083161354, 0, 0, 0].
Camera optical axes: x right, y down, z forward. The intrinsics JSON says they were read from the live Orbbec COLOR_SENSOR factory profile without starting streams. The top-camera robot_extrinsics field is null in that source.

## Best top RGB to board/table result

The result set is:
- results/top_world_20261017/capture_02_confirmed/calibration.json — 2026-10-17 20:30:05.400757556
- results/top_world_20261017/capture_02_confirmed/T_world_from_top_rgb_m.json — 2026-10-17 20:30:05.401757580
- results/top_world_20261017/capture_02_confirmed/T_top_rgb_from_world_m.json — 2026-10-17 20:30:05.401757580

The calibration JSON contains K, D, camera serial/resolution, quality, limitations, and both 4×4 transforms. Convention is T_target_from_source with column vectors and translation in meters.

Board/table frame: origin at the printed board's upper-left inner corner; X along print-right, Y toward the printed TOP edge/far side of the table, Z out of the printed surface/up. The user confirmed four squares measure 100 mm, so square size is 25 mm. Table alignment assumes the board was flat and coplanar with the tabletop; board thickness was not compensated.

T_world_from_top_rgb maps top RGB camera coordinates to board/table coordinates:
~~~
[[ 0.999841846,  0.012167652, -0.012970401,  0.098588357],
 [ 0.017539902, -0.795172065,  0.606130133, -0.751572772],
 [-0.002938520, -0.606261771, -0.795259725,  0.699229639],
 [ 0,            0,            0,            1]]
~~~

T_top_rgb_from_world is the inverse, mapping board/table coordinates into the camera:
~~~
[[ 0.999841846,  0.017539902, -0.002938520, -0.083335551],
 [ 0.012167652, -0.795172065, -0.606261771, -0.174913063],
 [-0.012970401,  0.606130133, -0.795259725,  1.012898805],
 [ 0,            0,            0,            1]]
~~~

Method: fixed factory K/D; 9×6 inner-corner checkerboard; OpenCV solvePnPGeneric with SOLVEPNP_IPPE per frame; lowest-RMS positive-depth solution seeds a Huber robust multi-frame fit. Thirty unique 640×480 frames; every fifth frame held out for repeatability, then all frames refit. Reported all-frame RMS 0.1549035 px; held-out RMS 0.1541111 px. Camera position in board world is [0.098588, -0.751573, 0.699230] m. Across-frame position standard deviation is [0.193, 0.166, 0.248] mm and maximum rotation deviation is 0.0390°. The board covers only about 78.5×34.4 pixels. The JSON explicitly says independent metric accuracy is false. Low pixel reprojection error is not an independent mm-accuracy measurement.

The prior capture_01 result is superseded:
- results/top_world_20261017/capture_01/calibration.json — 2026-10-17 20:27:26.482906237
- result matrices: capture_01/T_world_from_top_rgb_m.json and T_top_rgb_from_world_m.json — 2026-10-17 20:26:34.335661276
- Same camera, board frame, fixed K/D and 30-frame method; all-frame RMS 0.1566461 px, held-out RMS 0.1570114 px; independent metric accuracy false.

## Robot-base candidates and frame conventions

No result is accepted as an independently measured top-camera-to-robot-base calibration. In the Oct 17 table result, robot extrinsics are explicitly absent. In the Oct 14 board result, T_robot_base_from_world is null.

The top-world matrix above is camera→board/table. The project-default and older arm-graph files instead use T_world_from_base_m: base→world, where their world is the top/head RGB optical frame. They define arm bases A and B, not Link7, SDK flange, or hand TCP. Translation units are meters.

The project README/config selects results/nominal_geometry_single_group_A_00/calibration.json as the working default (source mtime 2026-10-15 18:45:55.890822620). Its status is nominal_geometry_single_group_anchor_unmeasured_registration; accepted_for_metric_use=false. It composes declared nominal CAD mount geometry with one A_00 board observation (three images); no mount parameter fitting, multi-pose mean, or independent validation. Its K/D for the top camera are the factory values above. Its two base→top-RGB-world transforms are:
~~~
T_world_from_base_m[A] =
[[-0.035165592,  0.049639580, -0.998147932, -0.019774903],
 [-0.825564533,  0.561421006,  0.057005747,  0.194472552],
 [ 0.563210957,  0.826040172,  0.021237974,  0.179810517],
 [ 0, 0, 0, 1]]

T_world_from_base_m[B] =
[[ 0.024069053,  0.012649992,  0.999630261,  0.062503936],
 [-0.772780663, -0.634114195,  0.026631485,  0.176524925],
 [ 0.634216626, -0.773135930, -0.005486859,  0.175364707],
 [ 0, 0, 0, 1]]
~~~

The Oct 17 code path uses results/wrist_world_20261017/derived_bases/base_estimates_top_origin_table_axes.json (mtime 2026-10-17 21:36:34.135591946). It stores T_shifted_world_from_base for A/B: source is each arm base; target is a frame with origin at the top RGB optical center and axes parallel to board/table axes (not top optical axes); meters; accepted_for_metric_use=false. The bases were derived from wrist-camera calibration plus nominal mount geometry and saved FK/joint measurements, using one snapshot per arm. The result warns that the A/B base estimates differ by 19.53 mm in height and are not independently verified.

Composing that file with capture_02 gives the following derived camera→base transforms. These composed matrices are not stored as standalone files:
~~~
T_A_base_from_top_camera =
[[-0.019068588, -0.825860919,  0.563551357,  0.056975198],
 [ 0.032810166,  0.562833378,  0.825918931, -0.264185871],
 [-0.999279682,  0.034239321,  0.016364196, -0.032107818],
 [ 0, 0, 0, 1]]

T_B_base_from_top_camera =
[[ 0.010633212, -0.770483292,  0.637371502,  0.020220386],
 [ 0.009897442, -0.637295217, -0.770556193,  0.248318895],
 [ 0.999894482,  0.014501835,  0.000849319, -0.068850485],
 [ 0, 0, 0, 1]]
~~~
The target frames are the robot's A_base and B_base; they are not wrist-camera, Link7, SDK-flange, or hand frames. Translation units are meters. Treat both as unaccepted candidates.

Other top-camera/arm-graph results retained in the copied tree:
| Source-relative result | Source mtime | Transform and method | Reported validation |
|---|---|---|---|
| head_world_robot_20261015/calibration.json | 2026-10-15 10:50:16.661608114 | T_world_from_base_m and inverse T_base_from_world_m for A/B; top RGB optical world; multi-pose arm-camera graph | A held-out reprojection 2.325 px, position consistency 3.053 mm; pose-subset base sensitivity 65.714 mm / 6.568°. B 0.935 px, 0.487 mm; sensitivity 5.555 mm / 0.895°. Both unaccepted. |
| project_20261015/tianji-camera-calibration/results/full_body_head_world/calibration.json | 2026-10-15 18:45:55.848821597 | Same graph-style T_world_from_base_m A/B; alternative result | Same limited-validation family; A unstable, B candidate, neither accepted. |
| project_20261015/tianji-camera-calibration/results/cad_mount_fit/calibration.json | 2026-10-15 18:45:55.817820841 | T_world_from_base_m and inverse for A/B; nominal CAD chain with fitted flange clock and base-board relation; fixed train/holdout split | A holdout 4.931 px / 5.694 mm, base subset sensitivity 8.516 mm / 1.223°. B 1.366 px / 0.757 mm, sensitivity 1.844 mm / 0.365°. Both explicitly unaccepted. |
| project_20261015/tianji-camera-calibration/results/assumed_identity_mount/calibration.json | 2026-10-15 18:45:55.811820695 | T_world_from_base_m and inverse for A/B; user-assumed identity mount | Approximation only; not measured; unaccepted. |
| project_20261015/tianji-camera-calibration/results/photo_informed_geometry_single_group_A_00/calibration.json | 2026-10-15 19:06:46.639230176 | T_world_from_base_m and inverse for A/B; photo-informed nominal assembly assumptions with one A_00 anchor | Unvalidated; unaccepted. |
| assumed_identity_mount_20261015/calibration.json | 2026-10-15 11:05:21.299342177 | Duplicate of the assumed-identity alternative outside the project folder | Approximation only; unaccepted. |

The wrist calibrations supporting the derived-base candidate are not top-camera extrinsics:
- results/wrist_world_20261017/left/calibration.json — 2026-10-17 21:10:34.273882904; left Gemini 305, serial CV2F86100083, 640×480; fx 370.1646423, fy 370.1999512, cx 318.6178284, cy 236.7398682; Brown-Conrady K6. T_world_from_camera maps wrist RGB→board world; inverse is stored. Thirty frames; all-frame RMS 0.249255 px, held-out 0.248740 px; independent metric accuracy false.
- results/wrist_world_20261017/right/calibration.json — 2026-10-17 21:10:34.291883341; right Gemini 305, serial CV2F861000F5, 640×480; fx 370.3011780, fy 370.1721497, cx 317.3395386, cy 241.0218964; Brown-Conrady K6. T_world_from_camera maps wrist RGB→board world; inverse is stored. Thirty frames; all-frame RMS 0.419466 px, held-out 0.384675 px; independent metric accuracy false.
The full wrist distortion arrays, K matrices, and transforms are retained in those copied calibration.json files.

## Older top-camera board results

- world_20261014/top_extrinsics_PROVISIONAL.json — mtime 2026-10-14 15:40:05.794926959; provisional pending measured board size and physical-origin confirmation; per-frame PnP records; square-size assumption 25 mm; RMS range 0.1083–0.1181 px across 12 observations.
- world_20261014/top_world_extrinsics.json — mtime 2026-10-14 15:43:18.628642651; top serial CP28563000JZ, 640×480, 25 mm square; status says scale/orientation user-confirmed. Selected frame 7, RMS 0.1082973 px; transform pair is T_world_from_camera (camera→board) and T_camera_from_world (board→camera), meters. Independent metric accuracy false. The old convention has X right, Y down toward the operator, and Z into the board/table, so it differs from the Oct 17 frame.

For the Oct 14 result:
~~~
T_world_from_camera =
[[ 0.999716903,  0.018731966, -0.014670634,  0.094073316],
 [-0.023792546,  0.791441031, -0.610782293,  0.358476803],
 [ 0.000169789,  0.610958434,  0.791662658, -0.647490742],
 [ 0, 0, 0, 1]]

T_camera_from_world =
[[ 0.999716903, -0.023792546,  0.000169789, -0.085407672],
 [ 0.018731966,  0.791441031,  0.610958434,  0.110114501],
 [-0.014670634, -0.610782293,  0.791662658,  0.732925641],
 [ 0, 0, 0, 1]]
~~~
This result explicitly has T_robot_base_from_world=null and wrist_camera_extrinsics=null.

## Code consumers

Source grep showed:
- tools/keyframe_agent/geometry.py:62–79 loads capture_02_confirmed/calibration.json and derived_bases/base_estimates_top_origin_table_axes.json. It composes the table-world transform, reads top K/D, and supplies pixel_to_world, world_to_pixel, and arm FK helpers. Pixel-to-world requires an explicitly supplied plane height. Its calibration acceptance flag is false.
- tools/export_calibrated_coordinate_urdf.py reads the derived bases, nominal_geometry_single_group_A_00/calibration.json, and capture_02_confirmed/calibration.json.
- Project configs/project.json and README select nominal_geometry_single_group_A_00/calibration.json as the working default from 2026-10-16; metric acceptance remains false.
- training/convert_bottle_raw54_to_openwam_sdk_flange.py writes calibration_used=false. It converts recorded joint commands/feedback through SDK kinematics into flange-space labels.
- training/validate_openwam_sdk_flange.py reads nominal_geometry_single_group_A_00/calibration.json for offline validation. This does not make the OpenWAM converter/training data depend on that camera extrinsic.
No scanned OpenWAM training/conversion entry point loads the Oct 17 top-camera-to-table result as a camera calibration input.

## Dataset intrinsics for E21 crops

Dataset: /mnt/workspace/colleague-e/pi05_lingjun_joint54_zero2_b256_20261018/datasets/lerobot_optical93_eef72_15hz_20261016
- meta/info.json reports observation.images.cam_high shape [3,480,640] (channels,height,width); 93 episodes, 27,281 frames, 15 fps.
- meta/contract.json maps cam_high→camera_top. Neither file declares a crop or crop offset.
- tools/convert_real_workspace_crop_20261031.py confirms E21 uses source 640×480, crops (x0,y0)=(0,96) through row 479 (Pillow box [0,96,640,480], height 384), then LANCZOS-resizes to width 320, height 192. The resulting frame shape is [192,320,3].
- The resulting crop-aware intrinsics for E21's 320×192 frames use sx=sy=0.5:
  fx'=183.2663269, fy'=183.2951813, cx'=160.6667938, cy'=71.2930374.
  K' = [[183.266327,0,160.666794],[0,183.295181,71.293037],[0,0,1]].
  Distortion coefficients are unchanged for this uniform scale. The crop offset in the original 640×480 image is x0=0, y0=96. For the intermediate 640×384 crop before resize, cy becomes 142.5860748 and the other intrinsics are unchanged.

## Confidence ranking

1. Factory top RGB intrinsics — high confidence for this camera's 640×480 K/D: directly sourced from the Orbbec factory COLOR_SENSOR profile and repeated consistently in later result files.
2. Oct 17 capture_02 top RGB→board/table transform — strongest top-camera extrinsic evidence: 30-frame robust fit, 0.155 px all-frame and 0.154 px held-out RMS, and confirmed 25 mm squares. Confidence is high for the reported image fit and moderate for absolute table coordinates because board coplanarity/thickness and independent table measurements are not validated.
3. Oct 17 wrist-derived A/B base estimates — useful provisional candidate transforms; unaccepted for metric use. B has better held-out wrist/base consistency than A in the saved calibration summaries. Both rely on nominal mount geometry and lack independent robot-base survey.
4. Oct 15 CAD/head-world alternatives — B candidates show the best multi-pose consistency among these files, especially CAD B, while A is weaker. All have unresolved absolute accuracy and remain unaccepted.
5. Oct 16 selected nominal-geometry working default — operational project default, but a single A_00 visual anchor with declared unmeasured mount geometry. Treat it as a baseline assumption, not a verified robot-camera measurement.
6. Oct 14 board result — low pixel residual but older single-frame selected pose and a different board-axis convention. Keep separate from the Oct 17 result; do not combine their world coordinates without a frame transform.
7. Assumed-identity and photo-informed candidates — lowest confidence; their mount registration is assumed or unvalidated.


### Additional diagnostic result inventory

The project also contains results/head_world/head_world_extrinsics_by_pose.json (mtime 2026-10-15 18:45:55.889822596) and results/head_world/summary.json (same mtime). The per-pose file has status provisional_visual_per_pose_alignment_not_independently_metric_validated, 30 samples (24 with head visible), world=head RGB optical, axes x-right/y-down/z-forward, translation meters, column vectors. Its matrix fields are T_world_from_head_rgb and T_head_rgb_from_board_m; the reported maximum fixed head-board image consistency is 0.204 px, translation deviation 2.134 mm, rotation deviation 0.520 degrees. It is a historical visual alignment diagnostic and does not establish a surveyed robot-base transform.

The arm-graph candidate calibration files listed above also carry top-camera identity CP28563000JZ, 640×480, and the same top K/D. Their outer matrix convention is T_target_from_source with column vectors and meters. T_world_from_base_m maps an arm base to the head/top RGB optical world; T_base_from_world_m is the inverse; T_base_from_board_m maps board coordinates into the arm base. The camera fields in these legacy project files also show T_world_from_rgb_m as identity for the top camera and robot_extrinsics=null, because the calibration does not include an independently measured top-camera mount-to-base transform.

The source candidate file keys are preserved in the destination tree. Top-to-board files expose T_world_from_camera and T_camera_from_world (or the corresponding top_rgb names). The derived-base file exposes T_shifted_world_from_base for A/B and T_shifted_world_from_board_world. The Oct 14 provisional file stores per-frame PnP diagnostics rather than one final accepted robot-base matrix. No NPZ files exist in this source tree.

The project also has results/flange_hand_eye_A/ with audit.json, observations.json, source_sha256.json, and subset_sensitivity.json. This is an A-side wrist/flange hand-eye audit, not a finalized top-camera-to-robot-base calibration result. No calibration.json is present in that directory.
Wrist-camera distortion arrays (OpenCV order [k1,k2,p1,p2,k3,k4,k5,k6]) are:
- Left: [-1.1977955102920532, 0.7573210000991821, -6.805238808738068e-05, 0.00031652062898501754, -0.16277547180652618, -1.1791441440582275, 0.7325649261474609, -0.15326273441314697].
- Right: [-1.1976383924484253, 0.759762704372406, -3.07157461065799e-05, -0.0001079681096598506, -0.15495871007442474, -1.179025650024414, 0.7349571585655212, -0.14528700709342957].
These are support-camera intrinsics; their image reprojection residuals are not top-camera-to-base error estimates.
