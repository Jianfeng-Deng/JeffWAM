# E103-D ABC/yam 外部无动作世界头实验

开始：2026-11-15T04:01:33.710411-03:00 BRT。状态：RUNNING（最终状态见末尾）。

## 预注册与基线
待核对并锁定，结果前写入。C2 ckpt_05000.pt；seed90；D0/D1 同步数与学习率。

## 数据清单与哈希
待完成。源只读；ABC train S1 RS640；yam E87 allowlist 减 E92 exclusions；ep142 永久排除。

## 相机、特征复现与时间合同
待完成；先复现2个Piper parent，不通过不铺开。

## 缓存覆盖与资源估算
GPU 总预算21600秒，新增全部落盘≤40GB。

## 训练与验证
待完成。

## 全部评估表与判定
待完成；dev无模型语言；不以模型标注定阈值或选点。

## 命令、资源、未完成与用户决定
待完成。

## 数据与资源预估（提取前）
```json
{"abc_selected_episodes": 390, "abc_selected_hours": 9.998302923377771, "hours_by_task": {"zip_tie_the_wires": 2.3689758697244443, "untangle_the_cables": 2.354184486257779, "wrap_the_cables": 2.3669835324444444, "wrap_the_headphones": 1.8953555480177786, "connect_and_route_the_hose": 1.0128034869333333}, "yam_allowlist": 49, "yam_excluded_intersection": ["yam:20261016_130636_1c53f6e4", "yam:20261016_130723_e6d67ab7", "yam:20261016_130921_bf0ed819", "yam:20261016_131147_b1ad2c1b", "yam:20261016_131226_b454130a", "yam:20261016_131434_c0a49409"], "qualified_episodes": {"abc": 390, "yam": 43}, "windows": {"abc": 28801, "yam": 628}, "camera_frames": 720026, "payload_bytes": 8847679488, "estimated_cache_gpuh_218fps_margin25": 1.1468304026503566, "input_hashes": {"/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/abc_inventory_v1/proposed_subsets/S1.json": "d160876cc638250ce03c0f8ab91cf44ba2543eeb49334a31709b1baabfc93b7b", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/yam_inventory_v1/proposed_p1_yam_list.json": "aee694d6037aa529c564a50cfc69e96872af3a85b50594946b0aacf2d991b450", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e92_yam_lang_v1/exclusions_success_subpool_20261114.json": "420067819ac82daf01ed42feee9062cbcd7253c710997e73ff5ac547bbdc1321"}, "time_contract": "30Hz physical offsets; 1s anchor stride; causal O age<=100ms; 3cam skew<=50ms; native nearest W tolerance median/2<=50ms; no gap>100ms across [-24,+100]; full episode fps/gap gates; ABC retains E89 S1 six-endpoint screening; YAM raw ms receipt approximations, no shifts"}
```

## 两个 Piper parent 数值复现
```json
{"parents": 2, "records": [{"parent": "pc_debd1e36f85dcd1f9bae", "cid": "piper_06f720d7a3c9840bfbaa17da", "camera": "cam_high", "frames": 128, "max_abs": 0.0, "receipt_sha256": "d733765231dec22d141ec5484ebb41ae4e56e4ec6efd2ad6a2e9b80d1032109b", "read_path": "/mnt/workspace/dataset/piper/demo6_raw_20261030/raw/demo6_teleop_realsense_v2_hdf5/aloha_mobile_dummy/video/cam_high/episode_5.mp4", "source_sha256": "018fa2a56d26749f1f21844be7ffb3c14f9a638c33c16147513b632ea515f1df"}, {"parent": "pc_debd1e36f85dcd1f9bae", "cid": "piper_06f720d7a3c9840bfbaa17da", "camera": "cam_left_wrist", "frames": 128, "max_abs": 0.0, "receipt_sha256": "d733765231dec22d141ec5484ebb41ae4e56e4ec6efd2ad6a2e9b80d1032109b", "read_path": "/mnt/workspace/dataset/piper/demo6_raw_20261030/raw/demo6_teleop_realsense_v2_hdf5/aloha_mobile_dummy/video/cam_left_wrist/episode_5.mp4", "source_sha256": "878c0229bad0a5e36b534f915e9e2036405fadfd2ee827a4f3c74a4f991bcf5d"}, {"parent": "pc_debd1e36f85dcd1f9bae", "cid": "piper_06f720d7a3c9840bfbaa17da", "camera": "cam_right_wrist", "frames": 128, "max_abs": 0.0, "receipt_sha256": "d733765231dec22d141ec5484ebb41ae4e56e4ec6efd2ad6a2e9b80d1032109b", "read_path": "/mnt/workspace/dataset/piper/demo6_raw_20261030/raw/demo6_teleop_realsense_v2_hdf5/aloha_mobile_dummy/video/cam_right_wrist/episode_5.mp4", "source_sha256": "02ed6467da2328caa819a2696fbfc0238fbebe3af2debfd1986780969ade9345"}, {"parent": "pc_e74e67b81d2044834538", "cid": "piper_0cd776bb2c690cf2811ada64", "camera": "cam_high", "frames": 128, "max_abs": 0.0, "receipt_sha256": "1dd4fe320d48f74cb6e6ac198228826855de5a47a1f101c8b60432632a4c748e", "read_path": "/mnt/workspace/dataset/piper/demo6_raw_20261030/raw/demo6_teleop_realsense_v2_hdf5/aloha_mobile_dummy/video/cam_high/episode_12.mp4", "source_sha256": "8e345a18f094b8579c74ec8e656be19a57852acf0539dcb2c397930b2d0b3066"}, {"parent": "pc_e74e67b81d2044834538", "cid": "piper_0cd776bb2c690cf2811ada64", "camera": "cam_left_wrist", "frames": 128, "max_abs": 0.0, "receipt_sha256": "1dd4fe320d48f74cb6e6ac198228826855de5a47a1f101c8b60432632a4c748e", "read_path": "/mnt/workspace/dataset/piper/demo6_raw_20261030/raw/demo6_teleop_realsense_v2_hdf5/aloha_mobile_dummy/video/cam_left_wrist/episode_12.mp4", "source_sha256": "386a852e0d97afaaf974b05d005799cda1726179e5a1a9f2a2db31a7978cf295"}, {"parent": "pc_e74e67b81d2044834538", "cid": "piper_0cd776bb2c690cf2811ada64", "camera": "cam_right_wrist", "frames": 128, "max_abs": 0.0, "receipt_sha256": "1dd4fe320d48f74cb6e6ac198228826855de5a47a1f101c8b60432632a4c748e", "read_path": "/mnt/workspace/dataset/piper/demo6_raw_20261030/raw/demo6_teleop_realsense_v2_hdf5/aloha_mobile_dummy/video/cam_right_wrist/episode_12.mp4", "source_sha256": "2fa29069fdf51484f59c589f2b0f313fc566cf02cbaf4fd245180074e68e5b0c"}], "max_abs": 0.0, "preprocess": {"version": "e86_siglip2_e84_fov_grid2x4_v1", "resolution": 224, "rgb": true, "resize": "aspect_preserve_round_bilinear", "padding": "edge_center", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "crop": null, "patch_grid": [14, 14], "cache_grid": [2, 4], "cache_dtype": "float16", "target": "fixed_spatial_average_stop_grad", "official_difference": "E84 aspect padding replaces direct square warp"}}
```

## 初步 CPU 核验（真实 Piper+ABC，yam 待缓存齐备再检）
{"status": "passed", "P3_rows": 34777, "P1_rows": 107057, "P1_parents": 242, "no_traj_deletion_max_abs": 3.337860107421875e-06, "P1_mixed_Piper_ABC_forward_backward_finite": true, "no_action_gradient": true, "CUDA_initialized": false, "loss": 1.59743070602417, "brt": "2026-11-15T04:14:34.158051-03:00"}

## 数据来源与时间闸门审计
{"status": "passed", "timing_summary": {"abc": {"episodes": 390, "candidate_anchors": 34584, "valid_anchors": 28801, "age_or_target": 5578, "gap_overlap": 1880, "skew": 617}, "yam": {"episodes": 43, "candidate_anchors": 725, "valid_anchors": 628, "age_or_target": 75, "skew": 23}}, "refs_sha256": {"/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e89_abc_inventory_20261114.md": "d4b162e7e4fce26172f6e1cabd60df6163d742721f762c47d5c7ef7ed9563b29", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e87_yam_inventory_20261113.md": "fd487324a10aef89e0308bbe6c3c53382a1d76784ceceaa9eb49d99865288f13", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e90_full_v31_p1_20261114.md": "27816e1aa55da6590aede084fb6fd4a81243631cdbf570694a5cd1f32d2f7f91", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e100c_chain_fix_20261115.md": "c4b3e4e9125eaef9120d227f31b5d50650effc70d60707d59271829945b3c6e8", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e91_abc_lang_20261114.md": "11a618b2b5e01a4973a16557f8633ffcff12382d8d2497328784daac8ab50221", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e91b_abc_uncertain_20261114.md": "8ad2e2cf750fbc36f0ade36acdc9873000ee05091e10b9442d7f0e152efce2da", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e91c_abc_fill_20261114.md": "9545aa15f26ffe4177b7a8d2a87b95387886e90467f32fb10473db22289a8f79", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e92_yam_lang_20261114.md": "f34e329f3a088fde32bd5d276aa81fa332ae3fb00577ad92954729dfd85b2795", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e92b_yam_uncertain_20261114.md": "e13389f6d8657b26cfc84236b79548ee528342b9991adaf297f47a43b76b1a10", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e92c2_yam_review_20261114.md": "f62555f04ed4abe8a992eecf8f7ff9b6d474023f2e73b28c37a53a6b79733b7c", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e92c_yam_review_20261114.md": "8be71b5dba8fbf288b1440850805e960dbf86bc7c74ddecaa48c9c81cded64fd", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e91_abc_lang_v1/lang_abc_v3.jsonl": "28c233f41c6a0fab90739ecae8f843c089217c1b9ad2929337020a0ba0b68a16", "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e92_yam_lang_v1/lang_yam_v2.jsonl": "9cf9265c6be5ca2664d14cb20c3d8c862c8455e2fee9a0268e83ce2ae869dbdd"}, "model_text_use": "ABC/yam model narratives are not inputs, targets, truth, thresholds or checkpoint selection in this run; fixed generic instruction only; user-mandated six-clip training-pool exclusion applied", "clock_limitation": "ABC record timestamp has no independent receipt/exposure clock evidence; yam raw timestamps are host receive-time approximation. No manual shift or interpolation.", "camera_mapping": {"Piper": ["cam_high", "cam_left_wrist", "cam_right_wrist"], "ABC": ["/top-camera", "/left-wrist-camera", "/right-wrist-camera"], "yam": ["top", "left", "right"]}, "camera_evidence": "ABC named MCAP topics and E89 calibration/visual audit; yam E87 raw-to-cam_high/left/right conversion schema and wrist views; no geometric equivalence or extrinsics claim", "brt": "2026-11-15T04:18:02.030869-03:00"}

## 缓存阶段切换与失败记账
首次复现因Piper原片迁址失败30.543秒；只读发现原片由colleague-a/Datasets/piper迁至dataset/piper，大小相符、768帧输出逐位复现为0。首版CPU哈希审计少写head/tail间NUL，严格复用E89摘要算法后全部通过，非源漂移。cache02为三路串行重复读取MCAP，约87相机帧/s；保留完整receipt缓存，主动中断并计入全部时长，切换cache03单次三路读取+3个CPU解码worker/同一GPU。未完成episode缓存隔离保留，不覆盖。cache03对后续ABC全解码帧检查无B帧和packet/decoded帧数一致；早期cache02完整episode另补同样解码审计。

## cache02 早期帧序补审计
{"episodes": 14, "status": "passed", "all_B_frames_zero": true, "packet_decode_count_equal": true, "CPU_only": true}

## 评估覆盖 CPU 预检（无模型前向）
{"status": "passed", "fixed": {"10": {"windows": 48, "parents": 6}, "25": {"windows": 48, "parents": 6}, "50": {"windows": 48, "parents": 6}, "100": {"windows": 48, "parents": 6}}, "expanded": {"10": {"windows": 211, "parents": 27}, "25": {"windows": 211, "parents": 27}, "50": {"windows": 211, "parents": 27}, "100": {"windows": 208, "parents": 26}}, "no_model_forward": true, "brt": "2026-11-15T04:26:34.021223-03:00"}

## Piper/评估输入哈希固定（训练前）
见 results/e103d_control_20261115_01/INPUT_HASHES.json，包含 P1/P3 窗口、训练统计、固定48窗、E93窗口/相机/特征合同、外部窗口清单及通用文本向量的完整SHA256。源视频首尾摘要不冒称全文件哈希。

## 三路相机对应复核
43/43 yam 原始metadata相机name/role逐条一致：top/left/right→cam_high/cam_left_wrist/cam_right_wrist；640×480逐条通过，metadata全文件SHA与E87一致。ABC使用已盘点的/top-camera、/left-wrist-camera、/right-wrist-camera及RS640标定，未镜像或交换左右。详见CAMERA_CONTRACT.json；逐session物理序列号/外参证据仍缺，不声称跨机器几何等价。

## 新历史/四目标窗口的逐episode有效率复核
所有纳入episode在1秒稀疏锚点上的有效窗口率均≥50%；最低ABC=51.85%、yam=57.14%。既有S1的原E89完整六端点筛选仍保留。详见EPISODE_VALID_FRACTION_AUDIT.json。

## 缓存提取完成
{"status": "complete", "completed": 433, "camera_frames": 720026, "fresh_camera_frames": 668420, "elapsed_s": 3049.915001694113, "cache": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e103d_cache/features", "brt": "2026-11-15T05:11:20.007257-03:00"}

## 最终缓存覆盖核验
{"abc": {"episodes": 390, "windows": 28801, "camera_frames": 704444, "hours": 9.99830292337778, "hours_scope": "ABC common-camera recorded span", "payload_bytes": 8656207872}, "yam": {"episodes": 43, "windows": 628, "camera_frames": 15582, "hours": 0.24417592592592594, "hours_scope": "E87 nominal frames/30, not measured exposure duration", "payload_bytes": 191471616}}

## CPU 合同验证
{"status": "passed", "physical_robot_action_state_deletion": true, "no_action_gradient": true, "finite_abc_yam": true, "no_traj_equivalence_max_abs": 3.337860107421875e-06, "initial_sha256": "a432001ffeb7d99fa2980ef337379464948d8c22678bb37dbfea0ab6708da99b", "modules": {"e99w2_train": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e103d/e99w2_train.py", "e99w2_objective": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e103d/e99w2_objective.py", "e103d_model": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e103d/e103d_model.py", "e103d_data": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e103d/e103d_data.py", "e98w_core": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e103d/e98w_core.py"}, "cuda_initialized": false, "brt": "2026-11-15T05:12:17.066313-03:00"}

## 结果前预注册锁定
```json
{"brt": "2026-11-15T05:12:18.919931-03:00", "initial_checkpoint": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c2_s90_20261115_03/ckpt_05000.pt", "initial_sha256": "a432001ffeb7d99fa2980ef337379464948d8c22678bb37dbfea0ab6708da99b", "steps_each": 5000, "seed": 90, "lr": 0.0003, "schedule": "constant; restore identical C2 optimizer; P3 global steps5001..10000, full C2 chain .1, predicted reentry probability .8", "P3": "Both arms identical Piper P3 32-window update every optimizer step, four modes equal probability; original sampler seed90 and demo3 cap .45", "P1": "Both arms additional batch32 no-robot/no-action deterministic world auxiliary, coefficient .1; four fixed horizons. D0 all Piper; D1 20 Piper +6 yam +6 ABC; missing source slots back to Piper. P1 has independent per-step RNG; P3 RNG unaffected by source substitution.", "ratio_ambiguity_disclosure": "C2 training code is P3-only, no historical standalone P1 ratio exists. Added paired P1 auxiliary to implement explicit user P1+P3 request; disclosed before training; optional clarification had no answer at implementation time.", "P1_source_fractions_D1": {"piper": 0.625, "yam": 0.1875, "abc": 0.1875}, "P1_sampler": "Piper parent equal demo3 cap.45 seed10390; external task round robin, parent uniform, valid window uniform; source fixed slots. ABC 5 tasks, each20% external draws asymptotically; no single task >25% physical hours", "text": "Piper original train instruction/fallback; ABC/yam fixed generic Manipulate objects on a tabletop. No model narratives. dev fixed human default only.", "world_loss": "same normalized deterministic world MSE +.1 cosine +.1 reentry adapter; no external action/alignment/phase/chain loss", "eval": "Fixed Piper dev48 windows/6parent all4 horizons gt_action/no_traj/pred_N1/pred_N8; 4Euler N1/N8 seeds90..97; E93 extended cached valid dev no_traj; 2000 paired parent bootstrap seed90; fixed endpoints no checkpoint selection", "criterion": "D1-D0 no_traj +25/+50/+100 >=2 CI lower>0pp; all3 gt_action CI lower>-1pp; N1 joint_L1_rad and gripper_L1_mm at8/50 point relative degradation<=5% (all4). Otherwise 无增益.", "plan_sha256": "cb57feeaf3abf4e0901abdc2f8a3b20688c7ece588575d9b5f7c510635dd71c5", "cache_contract_sha256": "c30329b186f6424146037ad905bab7f011d27243ee507452966c4c9aa9ac2ba2", "data_audit_sha256": "f634bf286bcfbd95ce4896fc598a1a4e9097bbda0ec67c28b888935fd7188130", "CPU_tests_sha256": "514041e212b3f9b8245465349806af06644da42977046fd0a1948ed4910bca23", "GPU_seconds_cap": 21600, "disk_bytes_cap": 40000000000}
```

## 运行时优化后同预注册重启（无评分变更）
D0 run01同步nvidia-smi每次实测0.458s，300步后中断，完整391.358s计入；日志、代码快照、旧锁保留。新run02只将空闲显存检查移至2秒后台监控，每步仍检查预算/峰值，与C2相同监控方式。200次四线程CPU检查和超时拒绝通过，无CUDA初始化；模型/数据/损失/优化器/seed/N/判据不变。PRETRAIN_LOCK_02.json在新训练前固定；无任何dev模型评分。

## 运行时优化数学路径复现
run02与被中断run01的前100步全部逐步训练指标逐位相同，最大loss差0。监控后台化不改变模型/数据/噪声/优化器路径；旧run仅性能中断，不参与评估。详见RUNTIME_PREFIX_REPLAY_02.json。

## D0 训练完成
{"status": "completed", "steps": 5000, "checkpoint": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e103d_d0_s90_20261115_02/ckpt_05000.pt", "checkpoint_sha256": "1aa1e6e4a9033ba23e20a1985561665b3ba78534767c2ffbde394e022d7e8fd3", "elapsed_s": 1523.1880635600537, "peak_mib": 2748.0712890625, "P1_counts": {"piper": 160000}, "P3_sampler": {"count": 160000, "counts": {"piper": 160000}, "seed": 90, "weights": {"piper": 1.0}, "caps": {}, "children": {"piper": {"count": 160000, "demo3_count": 50423, "seed": 90, "cap": 0.45, "strategy": "parent_equal", "stratum_weights": {}}}}, "P3_samples_sha256": "d28d1a5e8a77648897f0d2f186df8fbc8e90448ab40c7b8bcfdd9ab4b09aa502", "modes": {"clean": 1221, "independent": 1286, "action_only": 1256, "sync": 1237}, "adapter_initial": true, "last": {"step": 5000, "global_step": 10000, "loss": 0.41326507925987244, "mode": "clean", "action": 0.146868035197258, "world": 1.1789289712905884, "cosine": 0.8629432916641235, "adapter": 0.04076875001192093, "alignment": 0.8879218101501465, "chain": 1.0382624864578247, "chain_weight": 0.1, "chain_supervised": 1.0382624864578247, "chain_gap": 0.0, "predicted_reentry": true, "chain_k": 25, "p1_world": 1.1093008518218994, "p1_cos": 0.8857395052909851, "p1_adapter": 0.04177073389291763}, "brt": "2026-11-15T05:47:16.359276-03:00"}

## 观测时间元数据格式核验
缓存启动前已把CPU初稿obs_meta统一为Piper的[相对锚点历史秒数, 因果帧age秒数]；plan_raw_cpu.json保留初稿，仅审计、不训练。此修正不改变任何帧ID、窗口、目标或时间戳。当前plan.json被输入哈希和两版预注册锁固定。全量441435个相机观测元数据格式/age检查通过，详见OBS_META_AUDIT.json。

## D1 训练完成
{"status": "completed", "steps": 5000, "checkpoint": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e103d_d1_s90_20261115_02/ckpt_05000.pt", "checkpoint_sha256": "0164555718988782ab3f7c27c532d9ba59e447f7e6a41ce58e80a8d95f15ef07", "elapsed_s": 1525.3034456009045, "peak_mib": 2748.0712890625, "P1_counts": {"piper": 100000, "yam": 30000, "abc": 30000}, "P3_sampler": {"count": 160000, "counts": {"piper": 160000}, "seed": 90, "weights": {"piper": 1.0}, "caps": {}, "children": {"piper": {"count": 160000, "demo3_count": 50423, "seed": 90, "cap": 0.45, "strategy": "parent_equal", "stratum_weights": {}}}}, "P3_samples_sha256": "d28d1a5e8a77648897f0d2f186df8fbc8e90448ab40c7b8bcfdd9ab4b09aa502", "modes": {"clean": 1221, "independent": 1286, "action_only": 1256, "sync": 1237}, "adapter_initial": true, "last": {"step": 5000, "global_step": 10000, "loss": 0.4148702025413513, "mode": "clean", "action": 0.14691384136676788, "world": 1.1859681606292725, "cosine": 0.8679311275482178, "adapter": 0.0378146693110466, "alignment": 0.9035560488700867, "chain": 1.0328258275985718, "chain_weight": 0.1, "chain_supervised": 1.0328258275985718, "chain_gap": 0.0, "predicted_reentry": true, "chain_k": 25, "p1_world": 1.1167272329330444, "p1_cos": 0.8874353170394897, "p1_adapter": 0.03911926597356796}, "brt": "2026-11-15T06:13:05.058468-03:00"}

## D0 评估完成
固定48窗/6parent，768世界记录；扩展841时域记录。

## D1 评估完成
固定48窗/6parent，768世界记录；扩展841时域记录。

## 固定48窗/6 parent
|h|条件|窗/parent|D0 skill% [CI]|D1 skill% [CI]|D1−D0 pp [CI]|
|---:|---|---|---|---|---|
|10|gt_action|48/6|14.7944 [11.4553,17.4484]|14.4166 [11.0794,17.0911]|-0.3777 [-0.4570,-0.2963]|
|10|no_traj|48/6|12.0632 [8.8443,14.5716]|10.8426 [7.7350,13.4183]|-1.2206 [-1.6923,-0.8024]|
|10|pred_N1|48/6|13.3613 [10.5294,15.6831]|13.0111 [10.1119,15.3141]|-0.3503 [-0.6071,-0.1519]|
|10|pred_N8|48/6|13.4130 [10.3307,15.9838]|13.0085 [9.7700,15.5505]|-0.4045 [-0.6914,-0.1833]|
|25|gt_action|48/6|23.4523 [20.9813,25.2265]|23.1754 [20.8804,24.8665]|-0.2769 [-0.4331,-0.0751]|
|25|no_traj|48/6|18.8489 [14.3424,21.9413]|17.4241 [13.2988,20.5431]|-1.4249 [-1.9021,-0.8865]|
|25|pred_N1|48/6|19.5219 [16.1582,21.9546]|19.1899 [15.9062,21.5636]|-0.3319 [-0.4319,-0.2151]|
|25|pred_N8|48/6|20.2108 [16.2840,23.0936]|19.9248 [16.1407,22.6995]|-0.2860 [-0.4189,-0.1240]|
|50|gt_action|48/6|30.9524 [25.0993,35.2313]|30.3572 [24.5300,34.7770]|-0.5952 [-1.0382,-0.2549]|
|50|no_traj|48/6|24.3368 [17.7439,28.1546]|23.2118 [16.6782,26.9651]|-1.1250 [-1.7133,-0.6187]|
|50|pred_N1|48/6|21.9397 [16.5348,25.6295]|20.7272 [15.7927,24.1601]|-1.2125 [-1.6499,-0.6560]|
|50|pred_N8|48/6|23.4023 [16.8532,27.4254]|22.6762 [16.4148,26.4489]|-0.7261 [-0.9912,-0.3493]|
|100|gt_action|48/6|44.7790 [36.6655,50.7880]|44.4385 [36.5336,50.1122]|-0.3405 [-0.8044,0.1449]|
|100|no_traj|48/6|41.8805 [32.6485,48.3428]|40.3039 [30.9276,46.9499]|-1.5765 [-2.2602,-0.9271]|
|100|pred_N1|48/6|38.6097 [30.3025,44.6852]|36.9875 [28.7013,43.2447]|-1.6222 [-2.5128,-0.7200]|
|100|pred_N8|48/6|40.6742 [32.2172,45.7993]|39.8200 [31.4506,44.7880]|-0.8542 [-1.1909,-0.3986]|
## E93扩展 no_traj
|h|条件|窗/parent|D0 skill% [CI]|D1 skill% [CI]|D1−D0 pp [CI]|
|---:|---|---|---|---|---|
|10|no_traj|211/27|12.4749 [10.6412,14.2648]|11.7165 [10.0083,13.4419]|-0.7584 [-1.0177,-0.4966]|
|25|no_traj|211/27|19.7768 [17.0182,22.7471]|18.6937 [16.0969,21.4996]|-1.0830 [-1.3697,-0.7884]|
|50|no_traj|211/27|25.5288 [22.1981,29.0223]|24.5832 [21.4280,27.9386]|-0.9456 [-1.3061,-0.5412]|
|100|no_traj|208/26|39.9311 [34.4139,45.1696]|39.1357 [34.1190,43.9226]|-0.7954 [-1.5616,-0.0031]|
## 动作误差（parent等权）
|采样|h|指标|D0 [CI]|D1 [CI]|相对变化% [CI]|
|---|---:|---|---:|---:|---:|
|N1|8|joint_L1_rad|0.010824 [0.009686,0.012048]|0.010772 [0.009681,0.011950]|-0.478 [-1.388,0.618]|
|N1|8|joint_RMSE_rad|0.019749 [0.018117,0.021533]|0.019709 [0.018063,0.021485]|-0.200 [-1.335,1.344]|
|N1|8|gripper_L1_mm|2.328638 [1.919512,2.797594]|2.302189 [1.871984,2.761970]|-1.136 [-2.718,0.420]|
|N1|8|gripper_RMSE_mm|2.839536 [2.399130,3.468931]|2.782883 [2.332589,3.416752]|-1.995 [-3.753,-0.424]|
|N1|50|joint_L1_rad|0.054641 [0.045868,0.063221]|0.054529 [0.045393,0.063343]|-0.205 [-1.383,0.917]|
|N1|50|joint_RMSE_rad|0.098805 [0.087381,0.110537]|0.098740 [0.086905,0.110721]|-0.066 [-1.483,1.068]|
|N1|50|gripper_L1_mm|5.036085 [3.819597,6.180982]|4.984701 [3.843030,6.128640]|-1.020 [-3.812,1.997]|
|N1|50|gripper_RMSE_mm|7.434662 [5.729395,9.051614]|7.299192 [5.696750,8.888096]|-1.822 [-5.061,1.241]|
|N8_mean|8|joint_L1_rad|0.010337 [0.009352,0.011426]|0.010282 [0.009293,0.011349]|-0.535 [-1.570,0.488]|
|N8_mean|8|joint_RMSE_rad|0.018995 [0.017643,0.020616]|0.018980 [0.017560,0.020578]|-0.075 [-1.360,1.397]|
|N8_mean|8|gripper_L1_mm|2.389948 [1.898050,3.018308]|2.369183 [1.913091,2.964759]|-0.869 [-2.437,1.302]|
|N8_mean|8|gripper_RMSE_mm|2.879179 [2.326197,3.685966]|2.830478 [2.311531,3.627583]|-1.691 [-3.127,-0.128]|
|N8_mean|50|joint_L1_rad|0.050380 [0.043168,0.057633]|0.051046 [0.043481,0.058311]|1.323 [-0.291,3.010]|
|N8_mean|50|joint_RMSE_rad|0.092905 [0.083588,0.104323]|0.094133 [0.084460,0.105307]|1.321 [-0.489,3.447]|
|N8_mean|50|gripper_L1_mm|4.286727 [3.335549,5.115682]|4.281475 [3.309568,5.204549]|-0.123 [-3.259,2.529]|
|N8_mean|50|gripper_RMSE_mm|5.635082 [4.507381,6.608824]|5.578442 [4.377478,6.702075]|-1.005 [-5.225,2.524]|
## 预注册判定
```json
{"external_data_effective": false, "verdict": "无增益", "no_traj_improved_horizons": 0, "gt_action_noninferiority": false, "action_within5pct": true, "bootstrap": 2000, "seed": 90, "single_seed_limitation": true}
```
指标先窗内聚合、再parent等权；parent配对bootstrap2000，seed90；单seed不估计seed间不确定性。所有checkpoint均固定终点，未按dev选型。

## 资源与最终状态
{"gpu_seconds": 7245.939688177779, "gpu_hours": 2.0127610244938277, "new_bytes": 9862679852, "new_decimal_GB": 9.862679852, "segments": [{"phase": "cache_01", "pid": 267488, "gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 67678.0, "util": 3.0}, "started_brt": "2026-11-15T04:07:12.846069-03:00", "status": "closed", "elapsed_s": 30.543273319955915, "peak_estimate_mib": 6000, "finished_brt": "2026-11-15T04:07:43.138863-03:00"}, {"phase": "cache_02", "pid": 288133, "gpu": {"index": 3, "uuid": "GPU-5f10ec89-e0b8-ec0e-f6ab-cc0903493b2f", "total": 97871.0, "free": 67839.0, "util": 5.0}, "started_brt": "2026-11-15T04:09:23.359059-03:00", "status": "closed_interrupted_for_pipeline_optimization", "elapsed_s": 660.5587747097015, "peak_estimate_mib": 6000, "finished_brt": "2026-11-15T04:20:23.917876-03:00"}, {"phase": "cache_03", "pid": 397509, "gpu": {"index": 7, "uuid": "GPU-0345da1d-abf6-b973-f49d-c1023ae9a97d", "total": 97871.0, "free": 43780.0, "util": 100.0}, "started_brt": "2026-11-15T04:20:29.989492-03:00", "status": "closed", "elapsed_s": 3050.2452694210224, "peak_estimate_mib": 6000, "finished_brt": "2026-11-15T05:11:20.016123-03:00"}, {"phase": "eval_D0_02", "pid": 1590800, "gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 97206.0, "util": 0.0}, "started_brt": "2026-11-15T06:13:26.899414-03:00", "status": "closed", "elapsed_s": 31.743658798746765, "peak_estimate_mib": 6000, "finished_brt": "2026-11-15T06:13:58.461157-03:00"}, {"phase": "eval_D1_02", "pid": 1597573, "gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 97206.0, "util": 0.0}, "started_brt": "2026-11-15T06:14:12.699581-03:00", "status": "closed", "elapsed_s": 32.87561624171212, "peak_estimate_mib": 6000, "finished_brt": "2026-11-15T06:14:45.278211-03:00"}, {"phase": "train_D0", "pid": 946677, "gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 42218.0, "util": 83.0}, "started_brt": "2026-11-15T05:12:43.127501-03:00", "status": "closed_interrupted_for_monitor_optimization", "elapsed_s": 391.3580651283264, "peak_estimate_mib": 6000, "finished_brt": "2026-11-15T05:19:14.485647-03:00"}, {"phase": "train_D0_02", "pid": 1028725, "gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 44936.0, "util": 80.0}, "started_brt": "2026-11-15T05:21:53.320469-03:00", "status": "closed", "elapsed_s": 1523.2066987943836, "peak_estimate_mib": 6000, "finished_brt": "2026-11-15T05:47:16.372907-03:00"}, {"phase": "train_D1_02", "pid": 1288228, "gpu": {"index": 5, "uuid": "GPU-5e231c95-8cec-b63f-a071-4acf439637ed", "total": 97871.0, "free": 97206.0, "util": 0.0}, "started_brt": "2026-11-15T05:47:39.927756-03:00", "status": "closed", "elapsed_s": 1525.4083317639306, "peak_estimate_mib": 6000, "finished_brt": "2026-11-15T06:13:05.158372-03:00"}]}

状态：COMPLETE。无增益；GPU 2.0128h（含失败/中断段），新增9.863GB；D0/D1各5000步。
未完成必需项：无。扩展+100仅208窗/26parent，其余扩展时域211窗/27parent；不补造缺帧。动作真值为既有Piper next-frame follower反馈，不声称发送命令或实机成功。单seed、固定主集6parent；无测试集选型。无需用户新增决定。
