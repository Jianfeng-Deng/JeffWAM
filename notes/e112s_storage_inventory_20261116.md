# E112-S 存储只读盘点

- 日期：2026-11-16 BRT（UTC-3）；服务器 mtime CST 减 11 小时换算。
- 范围：runs/results/data 每个顶层目录；notes/*.md、experiment log、worklog、code_* 引用扫描。
- 方法：串行 nice/ionice du -skx，单目录 12 秒超时；类别扫描仅目录项/stat，不读 checkpoint/特征内容。
- 初始盘点为只读；其后按用户明确指示清除了 E28 特征数组及指定六目录中的特征数组/旧 checkpoint。未清理其他目录、未移动、压缩或改权限。
- 详细清单：results/e112s_inventory_20261116/，目前约 5 MiB。

## 总览

| 区域 | 顶层目录 | du 完整 | du GiB | 失败/跳过 |
|---|---:|---:|---:|---:|
| runs/ | 563 | 563 | 458.87 | 0 |
| results/ | 92 | 92 | 19.54 | 0 |
| data/ | 39 | 36 | 93.44 | 3 |

分类 A 230 项/499.76 GiB；B 14 项；C 0 项/0.00 GiB；D 451 项。du 为空或失败项不计入面积。

du 为分配空间；文件类型为 apparent size，可能因硬链接/稀疏文件而不同。E112 自身输出不计入区域总量。

## A 必须保留

明确保留默认模型、用户指定的历史基线/评估目录/标注与缓存；任何 notes 或代码精确路径引用也保守纳入 A。

| 目录 | 大小 | 最后修改（BRT） | 属主 | 引用 | 建议/依据 |
|---|---:|---|---|---:|---|
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/action_contrast_E_20261101 | 0.001 GiB | 2026-11-01 01:14 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/action_scorer_v1_20261031 | 0.047 GiB | 2026-10-31 12:19 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/action_scorer_v2_pairwise_20261031 | 0.047 GiB | 2026-10-31 12:32 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_reader_semantics_20261110_v1 | 0.000 GiB | 2026-11-09 13:49 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_reader_semantics_20261110_v2 | 0.000 GiB | 2026-11-09 13:50 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_reader_semantics_20261110_v3 | 0.000 GiB | 2026-11-09 13:52 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_reader_training_20261110_v1 | 17.945 GiB | 2026-11-10 02:21 | dengjianfeng | 精确21 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_x0_20261109 | 0.001 GiB | 2026-11-09 13:23 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_x0_x1_x4_20261109_v1 | 0.000 GiB | 2026-11-09 13:30 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_x0_x1_x4_20261109_v3 | 0.001 GiB | 2026-11-09 13:40 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_x1_history_ablation_20261109_v4 | 0.000 GiB | 2026-11-09 13:35 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_x4_reviewed_missing_confidence_20261109_v5 | 0.000 GiB | 2026-11-09 13:38 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e19_launch_20261101 | 0.000 GiB | 2026-11-01 06:19 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e21_postprocess_20261102 | 0.005 GiB | 2026-11-01 14:59 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e22_camera_calibration_20261101 | 0.002 GiB | 2026-11-01 15:04 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e22b_calib_check_20261102 | 0.005 GiB | 2026-11-02 04:01 | dengjianfeng | 精确10 / 名称1 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e22b_calib_check_20261102_fix | 0.035 GiB | 2026-11-02 04:28 | dengjianfeng | 精确8 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e24r_hand_rescore_20261102 | 0.002 GiB | 2026-11-02 05:25 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e25_chain_20261102 | 16.767 GiB | 2026-11-02 13:09 | dengjianfeng | 精确8 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e26a_audit_20261102 | 6.745 GiB | 2026-11-02 13:18 | dengjianfeng | 精确8 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_A0_20261103 | 18.173 GiB | 2026-11-02 19:51 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_A4_20261103 | 18.173 GiB | 2026-11-02 16:30 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_chain_20261103 | 0.001 GiB | 2026-11-03 00:57 | dengjianfeng | 精确6 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_copycheck_20261103 | 0.013 GiB | 2026-11-02 23:42 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e28_chain_20261103 | 6.372 GiB | 2026-11-03 05:15 | dengjianfeng | 精确11 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e28_geometry_20261103 | 0.000 GiB | 2026-11-03 00:43 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e29_vjepa_ac_probe_20261103 | 8.654 GiB | 2026-11-03 07:35 | dengjianfeng | 精确16 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e30_colleague-e_scan_20261103 | 0.000 GiB | 2026-11-03 08:35 | dengjianfeng | 精确5 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e31_dit_wm_survey_20261103 | 0.000 GiB | 2026-11-03 10:08 | dengjianfeng | 精确11 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e32_jepa_distill_20261103 | 14.832 GiB | 2026-11-03 11:17 | dengjianfeng | 精确15 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e33_20261104 | 10.868 GiB | 2026-11-04 05:36 | dengjianfeng | 精确32 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e33sub_20261104 | 0.916 GiB | 2026-11-04 04:18 | dengjianfeng | 精确10 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e33sub_full_20261104 | 0.546 GiB | 2026-11-04 06:01 | dengjianfeng | 精确10 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e34_20261104 | 1.272 GiB | 2026-11-04 08:36 | dengjianfeng | 精确10 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e35vis_20261104 | 0.008 GiB | 2026-11-04 09:38 | dengjianfeng | 精确5 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e36_20261104 | 0.061 GiB | 2026-11-04 09:48 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e36b_20261104 | 0.440 GiB | 2026-11-04 14:26 | dengjianfeng | 精确12 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e37_20261104 | 0.001 GiB | 2026-11-04 09:31 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e44_20261104 | 0.000 GiB | 2026-11-04 10:42 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e45_20261104 | 0.019 GiB | 2026-11-04 11:17 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e46_recalib_20261104 | 0.150 GiB | 2026-11-04 13:31 | dengjianfeng | 精确11 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e47_p0p1_20261104 | 10.757 GiB | 2026-11-06 04:06 | dengjianfeng | 精确42 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e54_ablation_20261106 | 11.735 GiB | 2026-11-07 00:59 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e60_p0_20261107 | 41.820 GiB | 2026-11-07 07:33 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e60_p0b_20261107 | 1.654 GiB | 2026-11-07 13:43 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e61_endpoint_20261108 | 7.545 GiB | 2026-11-08 02:20 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e63_encoder_e2e_20261108 | 49.601 GiB | 2026-11-09 05:16 | dengjianfeng | 精确5 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e64_imagination_20261109 | 4.330 GiB | 2026-11-09 05:26 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e65_selection_20261110 | 11.883 GiB | 2026-11-09 09:44 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e66_e67_bc_wam_20261110 | 18.571 GiB | 2026-11-10 21:42 | dengjianfeng | 精确6 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e68_lang_policy_20261110 | 9.953 GiB | 2026-11-11 01:46 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e70_siglip_align_20261111 | 19.582 GiB | 2026-11-11 02:35 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e71_command_history_20261111 | 16.327 GiB | 2026-11-11 08:31 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e72_transition_jepa_20261111 | 16.857 GiB | 2026-11-11 08:52 | dengjianfeng | 精确12 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e73_causal_wam_20261111 | 0.502 GiB | 2026-11-11 12:28 | dengjianfeng | 精确8 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e74_joint_v2_20261112 | 0.017 GiB | 2026-11-11 14:32 | dengjianfeng | 精确12 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e75_openwam_teacher_20261112 | 0.002 GiB | 2026-11-12 00:01 | dengjianfeng | 精确5 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e75_openwam_teacher_20261112_foundation154000_v1 | 0.067 GiB | 2026-11-12 01:04 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/embed_analysis_E_20261101 | 0.001 GiB | 2026-11-01 08:08 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_E24_objaware_20261102 | 0.702 GiB | 2026-11-02 01:27 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_E25_histdrop_20261102 | 1.407 GiB | 2026-11-02 12:14 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_phaseB_A_noforesight_20261031 | 0.189 GiB | 2026-10-31 06:47 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_phaseB_B_foresight_20261031 | 0.189 GiB | 2026-10-31 06:48 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_phaseC_wrist_20261031 | 0.149 GiB | 2026-10-31 10:21 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_phaseD_qhist15_20261031 | 0.129 GiB | 2026-10-31 10:10 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_phaseE_dinob_qhist15_20261031 | 1.398 GiB | 2026-10-31 13:57 | dengjianfeng | 精确10 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_phaseF_semboxes_20261101 | 1.398 GiB | 2026-10-31 16:25 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/future_pos_E_20261101 | 0.003 GiB | 2026-11-01 05:55 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/future_pos_E_sam3_20261101 | 0.003 GiB | 2026-11-01 14:59 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/interaction_jepa_v2_event_mapping_20261110 | 0.000 GiB | 2026-11-10 04:03 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/interaction_jepa_v2_screen_20261110 | 2.532 GiB | 2026-11-10 04:15 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/interaction_jepa_v3_grounding_20261110 | 0.146 GiB | 2026-11-10 04:27 | dengjianfeng | 精确1 / 名称1 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftfrozen_f0_20261031 | 0.119 GiB | 2026-10-31 01:45 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftfrozen_f1_20261031 | 0.119 GiB | 2026-10-31 01:49 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftfrozen_f2_20261031 | 0.119 GiB | 2026-10-31 01:50 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftfrozen_f3_20261031 | 0.119 GiB | 2026-10-31 01:49 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftfrozen_f4_20261031 | 0.119 GiB | 2026-10-31 01:48 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftreal_f0_20261031 | 0.119 GiB | 2026-10-31 01:05 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftreal_f1_20261031 | 0.119 GiB | 2026-10-31 01:07 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftreal_f2_20261031 | 0.119 GiB | 2026-10-31 01:07 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftreal_f3_20261031 | 0.119 GiB | 2026-10-31 01:06 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_ftreal_f4_20261031 | 0.119 GiB | 2026-10-31 01:06 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_scratch_f0_20261031 | 0.119 GiB | 2026-10-31 00:40 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_scratch_f1_20261031 | 0.119 GiB | 2026-10-31 00:40 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_scratch_f2_20261031 | 0.119 GiB | 2026-10-31 00:40 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_scratch_f3_20261031 | 0.119 GiB | 2026-10-31 00:40 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_scratch_f4_20261031 | 0.119 GiB | 2026-10-31 00:40 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_upstream_R0_e2e_s42_20261031 | 0.119 GiB | 2026-10-31 00:25 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_upstream_R0_e2e_s7_20261031 | 0.119 GiB | 2026-10-31 00:25 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_upstream_R1_frozen_drawer_20261031 | 0.119 GiB | 2026-10-31 00:25 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_upstream_R2_e2e_idm_20261031 | 0.119 GiB | 2026-10-31 00:25 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_upstream_R3_e2e_noreg_20261031 | 0.119 GiB | 2026-10-31 00:25 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/progress_scorer_v1_20261101 | 0.007 GiB | 2026-10-31 13:58 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/region_jit_E_latentonly_20261101 | 0.086 GiB | 2026-10-31 15:57 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/region_jit_v2_E_latentonly_20261101 | 1.084 GiB | 2026-11-01 03:44 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/report_20261103 | 0.011 GiB | 2026-11-03 05:46 | dengjianfeng | 精确11 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_AW_20261101 | 0.016 GiB | 2026-11-01 07:28 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_A_20261101 | 0.007 GiB | 2026-11-01 06:20 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_W_20261101 | 0.015 GiB | 2026-11-01 06:34 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/value_ttg_E23a_20261101 | 15.507 GiB | 2026-11-01 14:07 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/value_ttg_E24_objaware_20261102 | 16.061 GiB | 2026-11-02 02:48 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/wan_phaseA_image_only_20261031 | 0.061 GiB | 2026-10-31 04:26 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/wan_phaseA_latent_image_20261031 | 0.061 GiB | 2026-10-31 04:26 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c0_s90_20261115_01 | 0.000 GiB | 2026-11-15 02:26 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c0_s90_20261115_02 | 0.000 GiB | 2026-11-15 02:30 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c0_s90_20261115_03 | 0.260 GiB | 2026-11-15 02:46 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c1_s90_20261115_01 | 0.000 GiB | 2026-11-15 02:26 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c1_s90_20261115_02 | 0.000 GiB | 2026-11-15 02:30 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c1_s90_20261115_03 | 0.260 GiB | 2026-11-15 02:48 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c2_s90_20261115_01 | 0.000 GiB | 2026-11-15 02:27 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c2_s90_20261115_02 | 0.000 GiB | 2026-11-15 02:30 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_c2_s90_20261115_03 | 0.260 GiB | 2026-11-15 02:48 | dengjianfeng | 精确13 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_control_20261115_01 | 0.000 GiB | 2026-11-15 02:28 | dengjianfeng | 精确7 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_control_20261115_02 | 0.000 GiB | 2026-11-15 02:30 | dengjianfeng | 精确7 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e100c_control_20261115_03 | 0.001 GiB | 2026-11-15 02:50 | dengjianfeng | 精确9 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e101a_diag_20261115_01 | 0.106 GiB | 2026-11-15 04:15 | dengjianfeng | 精确14 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e101s_20261115_01 | 0.014 GiB | 2026-11-15 04:16 | dengjianfeng | 精确17 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a0_s90_20261115_01 | 0.000 GiB | 2026-11-15 04:54 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a0_s90_20261115_02 | 0.000 GiB | 2026-11-15 04:57 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a0_s90_20261115_03 | 0.259 GiB | 2026-11-15 05:14 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a1_s90_20261115_01 | 0.000 GiB | 2026-11-15 04:54 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a1_s90_20261115_02 | 0.000 GiB | 2026-11-15 04:57 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a1_s90_20261115_03 | 0.259 GiB | 2026-11-15 05:19 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a2_s90_20261115_01 | 0.000 GiB | 2026-11-15 04:54 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a2_s90_20261115_02 | 0.000 GiB | 2026-11-15 04:57 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_a2_s90_20261115_03 | 0.260 GiB | 2026-11-15 05:19 | dengjianfeng | 精确11 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_01 | 0.001 GiB | 2026-11-15 04:55 | dengjianfeng | 精确2 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_02 | 0.001 GiB | 2026-11-15 04:58 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_03 | 0.004 GiB | 2026-11-15 05:21 | dengjianfeng | 精确3 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102s_20261115_01 | 1.001 GiB | 2026-11-15 05:34 | dengjianfeng | 精确8 / 名称1 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102x_extended_s90_20261115_01 | 0.009 GiB | 2026-11-15 05:44 | dengjianfeng | 精确32 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e103d_control_20261115_01 | 0.000 GiB | 2026-11-15 06:14 | dengjianfeng | 精确9 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e103d_d0_s90_20261115_01 | 0.000 GiB | 2026-11-15 05:19 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e103d_d0_s90_20261115_02 | 0.261 GiB | 2026-11-15 06:13 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e103d_d1_s90_20261115_02 | 0.261 GiB | 2026-11-15 06:14 | dengjianfeng | 精确1 / 名称2 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_control_20261115_01 | 0.001 GiB | 2026-11-15 13:01 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_control_20261115_02 | 0.000 GiB | 2026-11-15 13:01 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_control_20261115_03 | 0.014 GiB | 2026-11-15 14:14 | dengjianfeng | 精确19 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f1_s90_20261115_03 | 0.519 GiB | 2026-11-15 13:47 | dengjianfeng | 精确23 / 名称7 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f2_s90_20261115_01 | 0.000 GiB | 2026-11-15 13:00 | dengjianfeng | 精确1 / 名称1 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e104p_f2_s90_20261115_03 | 0.519 GiB | 2026-11-15 14:02 | dengjianfeng | 精确21 / 名称7 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e105y_audit_20261115_142944 | 0.000 GiB | 2026-11-15 14:40 | dengjianfeng | 精确5 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e106c_control_20261115_01 | 0.000 GiB | 2026-11-15 14:51 | dengjianfeng | 精确8 / 名称1 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e106c_control_20261115_02 | 0.031 GiB | 2026-11-15 15:58 | dengjianfeng | 精确29 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e106c_f2_s90_20261115_01 | 0.000 GiB | 2026-11-15 14:51 | dengjianfeng | 精确7 / 名称1 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e106c_f2_s90_20261115_02 | 0.519 GiB | 2026-11-15 15:52 | dengjianfeng | 精确23 / 名称1 | 当前默认模型；ckpt_20000.pt；记录 SHA 前缀 dd27fe05c034 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e107m_control_20261116_01 | 1.108 GiB | 2026-11-16 01:45 | dengjianfeng | 精确10 / 名称1 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e107m_m0_s90_20261116_01 | 0.259 GiB | 2026-11-16 01:16 | dengjianfeng | 精确8 / 名称1 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e107m_m1_s90_20261116_01 | 0.200 GiB | 2026-11-16 01:29 | dengjianfeng | 精确8 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e107p_diagnosis_20261116_01 | 0.000 GiB | 2026-11-16 00:23 | dengjianfeng | 精确2 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e107p_diagnosis_20261116_02 | 0.210 GiB | 2026-11-16 00:33 | dengjianfeng | 精确14 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e108g_diag_20261116_01 | 0.035 GiB | 2026-11-16 02:50 | dengjianfeng | 精确16 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e109g_generalization_20261116_01 | 0.059 GiB | 2026-11-16 04:34 | dengjianfeng | 精确24 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_d_20261116 | 0.000 GiB | 2026-11-16 05:01 | dengjianfeng | 精确4 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_d_20261116_02 | 0.000 GiB | 2026-11-16 05:15 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_d_20261116_03 | 0.000 GiB | 2026-11-16 05:15 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_d_20261116_04 | 0.000 GiB | 2026-11-16 05:19 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_d_20261116_05 | 0.197 GiB | 2026-11-16 05:31 | dengjianfeng | 精确3 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_eulerdiag_20261116_02 | 0.000 GiB | 2026-11-16 05:06 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_eulerdiag_20261116_03 | 0.000 GiB | 2026-11-16 05:09 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_eval_20261116_01 | 0.002 GiB | 2026-11-16 05:52 | dengjianfeng | 精确2 / 名称1 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_f_20261116_05 | 0.197 GiB | 2026-11-16 05:44 | dengjianfeng | 精确2 / 名称1 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_eval_add_04000 | 0.002 GiB | 2026-11-16 09:21 | dengjianfeng | 精确2 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_eval_add_08000 | 0.002 GiB | 2026-11-16 09:24 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_eval_add_12000 | 0.002 GiB | 2026-11-16 09:26 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_eval_add_16000 | 0.002 GiB | 2026-11-16 09:28 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_eval_driver_20261116_01 | 0.000 GiB | 2026-11-16 08:00 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_eval_driver_20261116_02 | 0.000 GiB | 2026-11-16 09:28 | dengjianfeng | 精确1 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_flow_20261116_01 | 0.000 GiB | 2026-11-16 07:23 | dengjianfeng | 精确2 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_flow_20261116_02 | 0.259 GiB | 2026-11-16 08:13 | dengjianfeng | 精确4 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_flow_20261116_03 | 0.000 GiB | 2026-11-16 08:17 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_flow_20261116_04 | 0.259 GiB | 2026-11-16 08:47 | dengjianfeng | 精确2 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e111f_flow_20261116_05 | 0.519 GiB | 2026-11-16 09:17 | dengjianfeng | 精确2 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e112s_inventory_20261116 | 0.000 GiB | 2026-11-16 09:38 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e90_p1_20261114_01 | 0.001 GiB | 2026-11-14 04:04 | dengjianfeng | 精确25 / 名称23 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e93_dev_diag_20261114_01 | 0.169 GiB | 2026-11-14 05:03 | dengjianfeng | 精确102 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e94a_control_20261114_01 | 0.002 GiB | 2026-11-14 06:32 | dengjianfeng | 精确61 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e94a_m1_p1_20261114_01 | 0.929 GiB | 2026-11-14 06:28 | dengjianfeng | 精确23 / 名称12 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e94a_m2_p1_20261114_01 | 1.030 GiB | 2026-11-14 06:30 | dengjianfeng | 精确23 / 名称12 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e94b_dinov3_20261114_01 | 1.860 GiB | 2026-11-14 07:31 | dengjianfeng | 精确8 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e95_worldhead_structure_astra_20261114_01 | 0.000 GiB | 2026-11-14 06:57 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e96a_control_20261114_01 | 0.027 GiB | 2026-11-14 07:33 | dengjianfeng | 精确43 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e96a_m1_p1_20261114_01 | 0.949 GiB | 2026-11-14 07:25 | dengjianfeng | 精确55 / 名称10 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e97a_robot_env_20261114_01 | 0.493 GiB | 2026-11-14 13:53 | dengjianfeng | 精确30 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e98w_a_20261114_02 | 0.000 GiB | 2026-11-14 14:11 | dengjianfeng | 精确9 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e98w_a_20261114_03 | 0.000 GiB | 2026-11-14 14:14 | dengjianfeng | 精确9 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e98w_a_20261114_04 | 0.200 GiB | 2026-11-14 14:33 | dengjianfeng | 精确35 / 名称9 | E98-W A 初始化 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e98w_b_20261114_04 | 0.200 GiB | 2026-11-14 15:17 | dengjianfeng | 精确28 / 名称9 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e98w_control_20261114_01 | 1.237 GiB | 2026-11-14 15:34 | dengjianfeng | 精确85 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e99b3_20261115_01 | 3.978 GiB | 2026-11-15 01:46 | dengjianfeng | 精确20 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e99w2_control_20261115_01 | 0.069 GiB | 2026-11-15 02:28 | dengjianfeng | 精确51 / 名称8 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e99w2_p3_20261115_01 | 0.000 GiB | 2026-11-15 00:58 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e99w2_p3_20261115_02 | 0.000 GiB | 2026-11-15 01:00 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e99w2_p3_20261115_03 | 0.000 GiB | 2026-11-15 01:02 | dengjianfeng | 精确0 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e99w2_p3_20261115_04 | 0.260 GiB | 2026-11-15 01:19 | dengjianfeng | 精确11 / 名称0 | 用户指定的实验检查点/评估摘要 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/abc_inventory_v1 | 0.205 GiB | 2026-11-14 00:56 | dengjianfeng | 精确23 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/dinov2_regions_20261101 | 0.052 GiB | 2026-10-31 14:21 | dengjianfeng | 精确7 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/dinov2b_feats_20261031 | 5.073 GiB | 2026-10-31 12:12 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e103d_cache | 8.663 GiB | 2026-11-15 04:20 | dengjianfeng | 精确24 / 名称3 | E103-D 外部特征缓存，E104-P 复用 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e29_probe_windows_20261103 | 0.008 GiB | 2026-11-03 06:24 | dengjianfeng | 精确6 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e29_state_action_20261103 | 0.022 GiB | 2026-11-03 06:27 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e32_teacher_feats_20261103 | 6.278 GiB | 2026-11-03 13:26 | dengjianfeng | 精确5 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e32_windows_20261103 | 0.000 GiB | 2026-11-03 11:19 | dengjianfeng | 精确8 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e34_state_20261104 | 0.004 GiB | 2026-11-04 05:39 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e38_subgoals_20261104 | 0.113 GiB | 2026-11-04 10:39 | dengjianfeng | 精确7 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e49_labels_20261105 | 1.155 GiB | 2026-11-06 07:32 | dengjianfeng | 精确5 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1 | 未测 | 2026-11-13 14:47 | dengjianfeng | 精确157 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1 | 24.255 GiB | 2026-11-13 01:39 | dengjianfeng | 精确122 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e91_abc_lang_v1 | 0.142 GiB | 2026-11-14 05:02 | dengjianfeng | 精确15 / 名称0 | E91 ABC 文本标注训练目标 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e92_yam_lang_v1 | 0.060 GiB | 2026-11-14 06:39 | dengjianfeng | 精确12 / 名称0 | E92 yam 文本标注训练目标 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e94b_dinov3_cache | 21.660 GiB | 2026-11-14 06:28 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e97b_stages_v1 | 0.000 GiB | 2026-11-14 13:47 | dengjianfeng | 精确9 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e98_generalization_v1 | 0.009 GiB | 2026-11-14 14:11 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e98_predicate_labels_dev | 1.097 GiB | 2026-11-14 15:42 | dengjianfeng | 精确11 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e98_predicate_labels_train | 0.884 GiB | 2026-11-14 16:35 | dengjianfeng | 精确14 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e98_predicate_labels_train_causal_v1 | 未测 | 2026-11-16 08:01 | dengjianfeng | 精确4 / 名称0 | E98-C 标签，活跃勿动 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/hand_geometry_20261103 | 0.018 GiB | 2026-11-03 00:41 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/lewam_upstream_h5_20261031 | 7.662 GiB | 2026-10-30 15:03 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/mujoco_lewam_h5_20261031 | 2.653 GiB | 2026-10-31 00:38 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/mujoco_transition_224_20261031 | 0.497 GiB | 2026-10-31 00:35 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/object_masks_clean_20261102 | 0.457 GiB | 2026-11-02 13:10 | dengjianfeng | 精确17 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/object_masks_sam3_20261101 | 0.467 GiB | 2026-11-02 05:47 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/piper_events_20261106 | 未测 | 2026-11-07 07:28 | dengjianfeng | 精确6 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/piper_events_codex_20261110 | 0.075 GiB | 2026-11-10 06:11 | dengjianfeng | 精确3 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/policy_samples_D_20261031 | 0.424 GiB | 2026-10-31 12:18 | dengjianfeng | 精确1 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/real_workspace_192x320_20261031 | 4.694 GiB | 2026-10-31 01:04 | dengjianfeng | 精确9 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/real_wrist_144x192_20261031 | 4.215 GiB | 2026-10-31 06:07 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/scorer_cands_E_20261101 | 0.638 GiB | 2026-11-01 05:59 | dengjianfeng | 精确2 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/tianji_annotation_codex_20261110 | 1.842 GiB | 2026-11-10 07:19 | dengjianfeng | 精确4 / 名称0 | 项目内存在精确路径引用，保守保留 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/yam_inventory_v1 | 0.040 GiB | 2026-11-13 15:14 | dengjianfeng | 精确6 / 名称0 | 项目内存在精确路径引用，保守保留 |

## B 可压缩候选

旧日志专属目录，按日志 apparent bytes 估算压缩可能释放 30–70%，未实际压缩；压缩后仍需保留归档并解压读取。

| 目录 | 大小 | 最后修改（BRT） | 属主 | 引用 | 建议/依据 |
|---|---:|---|---|---:|---|
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e20_launch_20261101 | 0.000 GiB | 2026-11-01 07:33 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000005-0.000011 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_sam_mask_ablation_sam18_20261028 | 0.000 GiB | 2026-10-27 18:38 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000000-0.000000 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_sam_mask_ablation_sam18_20261028_v2 | 0.000 GiB | 2026-10-27 18:38 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000000-0.000000 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/object_masks_test_E21_20261101 | 0.000 GiB | 2026-11-01 07:13 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000001-0.000001 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_policy_spatial_future_state_20261030_v2 | 0.000 GiB | 2026-10-29 05:40 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000000-0.000001 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_policy_spatial_future_state_20261030_v3 | 0.000 GiB | 2026-10-29 05:43 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000000-0.000000 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_spatial_goal_motion_fk_20261029_v1 | 0.000 GiB | 2026-10-29 06:53 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000000-0.000000 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_grounding_ep056_v1 | 0.000 GiB | 2026-10-23 03:16 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000005-0.000011 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_grounding_ep066_v1 | 0.000 GiB | 2026-10-23 03:17 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000001-0.000003 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_grounding_ep088_v1 | 0.000 GiB | 2026-10-23 03:18 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000001-0.000003 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_labels_nogoal_smoke_v1 | 0.000 GiB | 2026-10-23 03:59 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000010-0.000023 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_labels_nogoal_v1 | 0.001 GiB | 2026-10-23 04:52 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000116-0.000271 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_labels_smoke_v1 | 0.000 GiB | 2026-10-23 03:48 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000000-0.000001 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_labels_v1 | 0.001 GiB | 2026-10-23 04:50 | dengjianfeng | 精确0 / 名称0 | 旧日志专属；压缩预计释放 0.000125-0.000291 GiB（按日志 apparent bytes 30-70%）；日志不能等价重建，压缩后可解压，无需训练重建 |

B 日志现有 0.001 GiB，估计压缩释放 0.000–0.001 GiB。

## C 可删除候选（只建议）

本轮没有目录同时满足可重建性已验证这一条件，故 C 暂无合格项。D 中仍有旧 cache 命名目录且无 checkpoint/精确路径引用，合计大小见 D 清单；在核实脚本和源数据前不计入可释放值。

| 目录 | 大小 | 最后修改（BRT） | 属主 | 引用 | 建议/依据 |
|---|---:|---|---|---:|---|

C 按 du 合计 0.000 GiB；本轮未确认任何目录具备低成本重建条件。

## D 待确认

包括名称引用路径不清、重建能力不明和统计失败目录。E（最佳旧模型）实际目录/ckpt 未唯一定位，相关旧实验不能仅凭本表删除。

| 目录 | 大小 | 最后修改（BRT） | 属主 | 引用 | 建议/依据 |
|---|---:|---|---|---:|---|
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/action_supervision_stages_20261027_smoke | 0.088 GiB | 2026-10-26 13:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/action_supervision_stages_20261027_v1 | 0.105 GiB | 2026-10-26 13:40 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arch_x0_x1_x4_20261109_v2 | 0.000 GiB | 2026-11-09 13:32 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arm_cv_flow_hybrid_v1 | 0.001 GiB | 2026-10-22 14:05 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/arm_velocity_flow_hybrid_v2 | 0.002 GiB | 2026-10-22 14:06 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/bottle_xy_visual_probe_20261025_proprio_v1 | 0.000 GiB | 2026-10-25 11:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/bottle_xy_visual_probe_20261025_roi_v1 | 0.001 GiB | 2026-10-25 11:28 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/bottle_xy_visual_probe_20261025_v1 | 0.000 GiB | 2026-10-25 11:18 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/bottle_xy_visual_probe_20261025_v2 | 0.000 GiB | 2026-10-25 11:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/chunk_transition_experiments | 0.008 GiB | 2026-10-22 12:18 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/controlled_action_ablation_20261024_v1 | 0.034 GiB | 2026-10-23 14:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/cpu_smoke | 0.001 GiB | 2026-10-23 02:35 | dengjianfeng | 精确0 / 名称2 | 仅目录名/run ID 提及，具体依赖未确认 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/cpu_smoke_residual | 0.001 GiB | 2026-10-23 02:35 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/direct_chunk_v1 | 0.016 GiB | 2026-10-22 12:35 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/direct_joint_feedback_20261025_v1 | 0.004 GiB | 2026-10-25 10:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/dynamics_context_v1 | 0.006 GiB | 2026-10-22 12:16 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e21_e22_postprocess_20261102 | 0.000 GiB | 2026-11-01 14:37 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e21_sam3_20261101 | 0.010 GiB | 2026-11-01 08:13 | dengjianfeng | 精确0 / 名称2 | 仅目录名/run ID 提及，具体依赖未确认 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e21_sam3_rawfix_20261101 | 0.000 GiB | 2026-11-01 08:03 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e21_sam3_smoke_20261101 | 0.000 GiB | 2026-11-01 07:38 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e22_tianji_kinematics_20261101 | 0.000 GiB | 2026-11-01 07:36 | dengjianfeng | 精确0 / 名称3 | 仅目录名/run ID 提及，具体依赖未确认 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e24_chain_20261102 | 0.000 GiB | 2026-11-01 14:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e24_chain_20261102_bs32 | 0.000 GiB | 2026-11-02 01:25 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e24r_hand_rescore_20261102_smoke | 0.002 GiB | 2026-11-02 05:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_A1_20261103 | 18.173 GiB | 2026-11-02 19:52 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_A2_20261103 | 18.173 GiB | 2026-11-02 22:52 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_A3_20261103 | 18.173 GiB | 2026-11-02 16:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_A4_20261103_gpu2_aborted | 0.704 GiB | 2026-11-02 14:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e27a_smoke_20261103 | 0.704 GiB | 2026-11-02 14:44 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e28_full_20261103 | 0.708 GiB | 2026-11-03 03:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e28_nofk_20261103 | 0.708 GiB | 2026-11-03 03:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e28_nogate_20261103 | 0.708 GiB | 2026-11-03 03:43 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e53_m1_20261105 | 0.009 GiB | 2026-11-05 08:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e53_mount_audit_20261105 | 0.008 GiB | 2026-11-05 08:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/e69_qwen_world_20261110 | 0.079 GiB | 2026-11-10 23:15 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/embed_analysis_E_20261101_smoke | 0.039 GiB | 2026-11-01 06:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ep018_input_ablation_active_v1 | 0.002 GiB | 2026-10-22 14:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ep018_input_ablation_active_v2 | 0.002 GiB | 2026-10-22 14:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ep018_input_ablation_center_v1 | 0.002 GiB | 2026-10-22 14:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ep018_input_ablation_center_v2 | 0.002 GiB | 2026-10-22 14:23 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_E24_objaware_20261102_aborted_bs4 | 0.000 GiB | 2026-11-01 14:48 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_smoke_f0_142638 | 0.122 GiB | 2026-10-31 03:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/ffn_smoke_f1_142607 | 0.122 GiB | 2026-10-31 03:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/finger_local_attention_decoder_20261024_120e_seed20261024 | 0.003 GiB | 2026-10-24 03:25 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/finger_local_attention_decoder_20261024_120e_seed20261025 | 0.003 GiB | 2026-10-24 03:28 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/finger_local_attention_decoder_20261024_120e_seed20261026 | 0.003 GiB | 2026-10-24 03:33 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/finger_local_attention_decoder_20261024_v1 | 0.003 GiB | 2026-10-24 03:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/finger_local_shared_mixer_20261024_120e_seed20261024 | 0.001 GiB | 2026-10-24 03:58 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/finger_local_shared_mixer_20261024_120e_seed20261025 | 0.001 GiB | 2026-10-24 03:58 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/finger_local_shared_mixer_20261024_120e_seed20261026 | 0.001 GiB | 2026-10-24 03:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/flow_and_fusion_eval_v1 | 0.002 GiB | 2026-10-22 13:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/fresh_replay_ep018_v1 | 0.002 GiB | 2026-10-22 14:15 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/full84_multiscale_center_50e_sample135_v1 | 0.014 GiB | 2026-10-22 07:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/future_pos_E_smoke_20261101 | 0.000 GiB | 2026-11-01 05:51 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/gradient_path_audit_20261026 | 0.000 GiB | 2026-10-26 13:46 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/hand_synergy_5d_reconstruction_20261024_v1 | 0.001 GiB | 2026-10-23 23:17 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/hand_synergy_5d_reconstruction_20261024_v2 | 0.003 GiB | 2026-10-23 23:25 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/horizon_gate_delta10x_20261024_v1 | 0.011 GiB | 2026-10-24 07:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/horizon_gate_v2_delta10x_20261024_v1 | 0.000 GiB | 2026-10-24 07:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/horizon_gate_v3_delta10x_20261024_v1 | 0.017 GiB | 2026-10-24 07:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/horizon_gate_v4_delta10x_20261024_v1 | 0.017 GiB | 2026-10-24 07:11 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/interaction_jepa_v1_cpu_interface_20261110 | 0.000 GiB | 2026-11-10 03:52 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/interaction_jepa_v1_cpu_interface_20261110_v2 | 0.037 GiB | 2026-11-10 03:53 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/interaction_jepa_v1_gpu3_pilot16_20261110 | 0.000 GiB | 2026-11-10 03:54 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/interaction_jepa_v1_gpu3_pilot16_20261110_v2 | 0.000 GiB | 2026-11-10 03:54 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/interaction_jepa_v1_gpu3_pilot16_20261110_v3 | 0.073 GiB | 2026-11-10 03:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/latest_priority_weight_sweep_v1 | 0.000 GiB | 2026-10-22 13:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_action_flow_first20_20261027_v1 | 0.036 GiB | 2026-10-27 06:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_action_flow_first20_cont40_20261027_v1 | 0.036 GiB | 2026-10-27 06:28 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_action_mse_first20_20261027_v1 | 0.029 GiB | 2026-10-27 06:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_action_mse_first20_cont40_20261027_v1 | 0.032 GiB | 2026-10-27 06:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_action_mse_fullsplit_20261027_v1 | 0.031 GiB | 2026-10-27 07:06 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_native_abs_mse_20261027_v1 | 0.057 GiB | 2026-10-27 07:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_native_abs_rmse_20261027_v1 | 0.057 GiB | 2026-10-27 07:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_native_joint_smoke2_20261027 | 0.057 GiB | 2026-10-27 07:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_native_joint_smoke3_20261027 | 0.057 GiB | 2026-10-27 07:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_native_joint_smoke_20261027 | 0.000 GiB | 2026-10-27 07:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_native_regression_comparison_20261027 | 0.000 GiB | 2026-10-27 07:45 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_native_residual_mse_20261027_v1 | 0.057 GiB | 2026-10-27 07:29 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam24_native_residual_mse_near_20261027_v1 | 0.057 GiB | 2026-10-27 07:29 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam54_action_dynamics_20261028_1k_v1 | 0.014 GiB | 2026-10-28 09:54 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam54_action_dynamics_20261028_cross_phase_smoke_v1 | 0.014 GiB | 2026-10-28 09:58 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam54_action_dynamics_20261028_smoke_v1 | 0.014 GiB | 2026-10-28 09:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam54_action_dynamics_20261028_smoke_v2 | 0.014 GiB | 2026-10-28 09:51 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam54_action_dynamics_posfix_full_20261028_v1 | 0.014 GiB | 2026-10-28 10:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam54_action_dynamics_posfix_overfit128_20261028_v1 | 0.014 GiB | 2026-10-28 10:29 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_22d_targets_20261024_v1 | 0.032 GiB | 2026-10-24 00:02 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_22d_transfer_audit_20261024_v1 | 0.000 GiB | 2026-10-23 23:44 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_action_dynamics_detached_54d_20261027_v1 | 0.062 GiB | 2026-10-27 13:01 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_action_dynamics_full_54d_20261027_v1 | 0.062 GiB | 2026-10-27 12:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_action_dynamics_full_norm_action_54d_20261027_v1 | 0.062 GiB | 2026-10-27 13:01 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_action_dynamics_full_smoke_20261027 | 0.062 GiB | 2026-10-27 12:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_action_dynamics_policy_smoke_zero_slot_20261027 | 0.062 GiB | 2026-10-27 12:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_action_policy_only_zero_slot_54d_20261027_v1 | 0.062 GiB | 2026-10-27 13:01 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_grasp_pilot_20261028_v1 | 0.007 GiB | 2026-10-28 10:45 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_grasp_smoke_20261028_v1 | 0.001 GiB | 2026-10-28 10:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_grasp_valid_20261028_v1 | 0.007 GiB | 2026-10-28 11:07 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_matched_q_20261028_v1 | 0.007 GiB | 2026-10-28 12:31 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_matched_q_20261028_v2 | 0.015 GiB | 2026-10-28 12:37 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_multihorizon_20261028_partial_bad_offset | 0.002 GiB | 2026-10-28 11:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_multihorizon_20261028_v2 | 0.008 GiB | 2026-10-28 11:46 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_multihorizon_20261028_v3_vision_ablation | 0.020 GiB | 2026-10-28 12:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_object_pilot_20261028_v1 | 0.009 GiB | 2026-10-28 10:40 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_object_valid_20261028_v1 | 0.009 GiB | 2026-10-28 11:07 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_smoke_20261028_v1 | 0.001 GiB | 2026-10-28 10:35 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_counterfactual_smoke_20261028_v2 | 0.001 GiB | 2026-10-28 10:38 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_drawer_gr_features_20261028_v1 | 0.018 GiB | 2026-10-28 10:32 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_florence_goal_fullsplit_pilot_20261026_none | 0.037 GiB | 2026-10-26 00:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_florence_goal_fullsplit_pilot_20261026_vlm | 0.002 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_florence_goal_pilot_20261026_none | 0.036 GiB | 2026-10-25 21:49 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_florence_goal_pilot_20261026_vlm | 0.001 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_florence_goal_residual_pilot_20261026_none | 0.036 GiB | 2026-10-25 21:49 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_florence_goal_residual_pilot_20261026_vlm | 0.001 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_future_latent_parallel_54d_qvelocity_20261027_v1 | 0.058 GiB | 2026-10-27 08:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_future_latent_serial_54d_qvelocity_20261027_v1 | 0.059 GiB | 2026-10-27 08:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_future_latent_serial_detached_54d_qvelocity_20261027_v1 | 0.059 GiB | 2026-10-27 08:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_global_raw_fullflow_20261028 | 0.063 GiB | 2026-10-27 23:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_global_sam_action_fullflow_20261028 | 0.063 GiB | 2026-10-27 23:57 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_global_sam_action_fullflow_diag_20261028 | 0.063 GiB | 2026-10-28 00:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_global_sam_action_lora_matched_20261028 | 0.064 GiB | 2026-10-28 00:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_global_sam_action_lora_smoke_20261028 | 0.064 GiB | 2026-10-28 00:12 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_global_sam_action_smoke_20261028 | 0.063 GiB | 2026-10-27 23:53 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_gr_flow_20261028_v1 | 0.018 GiB | 2026-10-28 10:46 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_gr_flow_candidates_20261028_v1 | 0.002 GiB | 2026-10-28 10:49 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_gr_flow_smoke_20261028_v1 | 0.000 GiB | 2026-10-28 10:41 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_gr_flow_smoke_20261028_v2 | 0.018 GiB | 2026-10-28 10:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_gr_mse_20261028_v1 | 0.018 GiB | 2026-10-28 10:45 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_gr_mse_candidates_20261028_v1 | 0.001 GiB | 2026-10-28 10:49 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_tc_flow_20261028_v1 | 0.018 GiB | 2026-10-28 10:46 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_tc_flow_candidates_20261028_v1 | 0.002 GiB | 2026-10-28 10:49 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_flow_10k_20261029_v1 | 0.000 GiB | 2026-10-28 13:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_10k_20261029_v1 | 0.014 GiB | 2026-10-28 13:45 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_best_fullval_20261029_v1 | 0.000 GiB | 2026-10-28 13:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_best_fullval_20261029_v2 | 0.002 GiB | 2026-10-28 13:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_best_mujoco_20261029_v1 | 0.000 GiB | 2026-10-28 13:57 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_best_report_20261029_v1 | 0.000 GiB | 2026-10-28 13:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_smoke_20261029_v1 | 0.014 GiB | 2026-10-28 13:43 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_smoke_fullval_20261029_v1 | 0.002 GiB | 2026-10-28 13:49 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_smoke_mujoco_ep056_20261029_v1 | 0.000 GiB | 2026-10-28 13:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal54_terminal_mse_smoke_mujoco_ep056_20261029_v2 | 0.000 GiB | 2026-10-28 13:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_current_visual_1k_20261029_v1 | 0.014 GiB | 2026-10-28 14:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_current_visual_2k_20261029_v1 | 0.014 GiB | 2026-10-28 14:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_full_future_1k_20261029_v1 | 0.018 GiB | 2026-10-28 14:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_full_future_2k_20261029_v1 | 0.018 GiB | 2026-10-28 14:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_full_terminal_1k_20261029_v1 | 0.014 GiB | 2026-10-28 14:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_full_terminal_2k_20261029_v1 | 0.014 GiB | 2026-10-28 14:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_proprio_1k_20261029_v1 | 0.014 GiB | 2026-10-28 14:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_proprio_2k_20261029_v1 | 0.014 GiB | 2026-10-28 14:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_terminal_goal_1k_20261029_v1 | 0.014 GiB | 2026-10-28 14:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_abs_terminal_goal_2k_20261029_v1 | 0.014 GiB | 2026-10-28 14:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_rebuild_ep056_abs_full_1k_20261029_v1 | 0.014 GiB | 2026-10-28 14:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_rebuild_ep056_smoke_20261029_v1 | 0.014 GiB | 2026-10-28 14:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_residual_full_terminal_2k_20261029_v1 | 0.014 GiB | 2026-10-28 14:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_residual_full_terminal_best_eval_20261029_v1 | 0.003 GiB | 2026-10-28 14:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_residual_full_terminal_best_mujoco_20261029_v1 | 0.000 GiB | 2026-10-28 14:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_residual_full_terminal_demo_mujoco_20261029_v1 | 0.000 GiB | 2026-10-28 14:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_action_residual_proprio_2k_20261029_v1 | 0.014 GiB | 2026-10-28 14:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_reach_value_pilot_20261028_v1 | 0.000 GiB | 2026-10-28 10:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_reach_value_valid_global_20261028_v1 | 0.000 GiB | 2026-10-28 11:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_goal_reach_value_valid_spatial_20261028_v1 | 0.000 GiB | 2026-10-28 11:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_object_value_expanded_20261029_v1 | 0.000 GiB | 2026-10-28 13:29 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_object_value_test_20261028_v1 | 0.000 GiB | 2026-10-28 10:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_object_value_test_20261028_v2 | 0.000 GiB | 2026-10-28 11:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_object_value_valid_global_test_20261028_v1 | 0.000 GiB | 2026-10-28 11:09 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_object_value_valid_global_test_20261028_v2 | 0.000 GiB | 2026-10-28 11:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_object_value_valid_spatial_test_20261028_v1 | 0.000 GiB | 2026-10-28 11:09 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_object_value_valid_spatial_test_20261028_v2 | 0.000 GiB | 2026-10-28 11:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_sim_candidates_expanded_20261029_v1 | 0.001 GiB | 2026-10-28 13:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_sim_candidates_expanded_20261029_v2 | 0.015 GiB | 2026-10-28 13:25 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_sim_candidates_smoke_20261028_v1 | 0.001 GiB | 2026-10-28 10:57 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_sim_candidates_test_20261028_v1 | 0.007 GiB | 2026-10-28 10:58 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_gr_flow_sim_candidates_valid_test_20261028_v1 | 0.007 GiB | 2026-10-28 11:07 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_initial_penetration_guard_test_20261028_v1 | 0.000 GiB | 2026-10-28 11:05 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint24_abs_qfeedback_singleview_20261026_resume_v2 | 0.042 GiB | 2026-10-26 03:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint24_abs_qfeedback_singleview_20261026_v1 | 0.035 GiB | 2026-10-26 02:47 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint24_abs_singleview_20261026_v1 | 0.035 GiB | 2026-10-26 02:44 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint24_residual_qfeedback_singleview_20261026_chunkref_v1 | 0.042 GiB | 2026-10-26 03:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint24_residual_qfeedback_singleview_20261026_v1 | 0.035 GiB | 2026-10-26 03:11 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint24_residual_qfeedback_threeview_20261026_v1 | 0.042 GiB | 2026-10-26 03:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint24_residual_threeview_features_20261026_v1 | 0.054 GiB | 2026-10-26 03:29 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint_policy_24d_qonly_20261027_v1 | 0.057 GiB | 2026-10-27 07:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint_policy_24d_qvelocity_20261027_v1 | 0.058 GiB | 2026-10-27 08:06 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint_policy_54d_qonly_20261027_v1 | 0.058 GiB | 2026-10-27 08:06 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_joint_policy_54d_qvelocity_20261027_v1 | 0.061 GiB | 2026-10-27 08:27 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_learned_feedback_smoke_20261028_v3 | 0.000 GiB | 2026-10-28 11:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_object_eval_20261031 | 0.000 GiB | 2026-10-31 02:15 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_mjc_smoke_ft_20261031 | 0.119 GiB | 2026-10-31 00:41 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_multihorizon_outcome_frozen_20261028_v1 | 0.022 GiB | 2026-10-28 11:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_multihorizon_outcome_joint_20261028_v1 | 0.022 GiB | 2026-10-28 11:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_multihorizon_outcome_smoke_20261028_partial_v1 | 0.000 GiB | 2026-10-28 11:46 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_multihorizon_outcome_smoke_20261028_v2 | 0.022 GiB | 2026-10-28 11:46 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_multihorizon_outcome_smoke_final_20261028_v1 | 0.022 GiB | 2026-10-28 11:54 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_constraints_preflight_20261028 | 0.000 GiB | 2026-10-28 09:11 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_A_fair_v1_20261028 | 0.002 GiB | 2026-10-28 09:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_A_v1_20261028 | 0.003 GiB | 2026-10-28 09:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_B_raw_lewam_fair_v1_20261028 | 0.002 GiB | 2026-10-28 09:28 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_B_raw_lewam_v1_20261028 | 0.001 GiB | 2026-10-28 09:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_B_raw_lewam_v2_20261028 | 0.001 GiB | 2026-10-28 09:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_B_raw_lewam_v3_20261028 | 0.002 GiB | 2026-10-28 09:25 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_C_hybrid_lewam_v1_20261028 | 0.003 GiB | 2026-10-28 09:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_all_fair_v1_20261028 | 0.003 GiB | 2026-10-28 09:29 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_diagnose_20261028 | 0.001 GiB | 2026-10-28 09:16 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_lewam_all_v1_20261028 | 0.000 GiB | 2026-10-28 09:15 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_lewam_all_v2_20261028 | 0.005 GiB | 2026-10-28 09:18 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_hardlimit_preflight_20261028 | 0.000 GiB | 2026-10-28 09:15 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_hand_safe_lewam_all_v1_20261028 | 0.004 GiB | 2026-10-28 09:12 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v1 | 0.000 GiB | 2026-10-28 06:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v10_grasp_pose | 0.004 GiB | 2026-10-28 06:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v10_grasp_pose_preflight | 0.000 GiB | 2026-10-28 06:34 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v11_grasp_shape | 0.005 GiB | 2026-10-28 07:31 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v12_ik_dominant | 0.002 GiB | 2026-10-28 07:33 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v13_contact_latch | 0.003 GiB | 2026-10-28 07:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v13_preflight | 0.000 GiB | 2026-10-28 07:35 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v14_preflight | 0.000 GiB | 2026-10-28 07:37 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v14_staged_grasp | 0.002 GiB | 2026-10-28 07:38 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v15_fresh_q_ik | 0.004 GiB | 2026-10-28 07:45 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v16_gravity_ff | 0.006 GiB | 2026-10-28 07:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v17_waypoint_gate | 0.006 GiB | 2026-10-28 07:54 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v18_constant_target | 0.013 GiB | 2026-10-28 07:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v19_anchored_lift_test | 0.002 GiB | 2026-10-28 08:03 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v2 | 0.002 GiB | 2026-10-28 06:03 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v20_index_middle_shape | 0.001 GiB | 2026-10-28 08:05 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v21_shape_gated_lift | 0.007 GiB | 2026-10-28 08:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v22_hold45 | 0.003 GiB | 2026-10-28 08:12 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v23_transport4cm | 0.003 GiB | 2026-10-28 08:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v24_transport8cm | 0.002 GiB | 2026-10-28 08:23 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v25_transport8cm_lift3cm | 0.003 GiB | 2026-10-28 08:25 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v26_transport12cm_lift3cm | 0.002 GiB | 2026-10-28 08:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v27_transport12cm_segment4cm | 0.003 GiB | 2026-10-28 08:28 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v28_transport24cm_segment4cm | 0.004 GiB | 2026-10-28 08:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v29_transport12cm_safe | 0.003 GiB | 2026-10-28 08:38 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v3 | 0.001 GiB | 2026-10-28 06:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v30_transport24cm_safe | 0.005 GiB | 2026-10-28 08:38 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v31_transport24cm_tight08 | 0.005 GiB | 2026-10-28 08:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v32_transport24cm_tight12 | 0.005 GiB | 2026-10-28 08:44 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v33_transport12cm_tight08 | 0.003 GiB | 2026-10-28 08:45 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v4 | 0.001 GiB | 2026-10-28 06:09 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v5_lewam_only | 0.000 GiB | 2026-10-28 06:11 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v6 | 0.001 GiB | 2026-10-28 06:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v7 | 0.001 GiB | 2026-10-28 06:14 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v8_finger_metric | 0.001 GiB | 2026-10-28 06:16 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_mpc_ep056_oracle_20261028_v9_lewam_only_finger_metric | 0.001 GiB | 2026-10-28 06:16 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_outcome_pilot_20261028_v1 | 0.004 GiB | 2026-10-28 10:46 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_outcome_pilot_20261028_v2 | 0.004 GiB | 2026-10-28 10:48 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_outcome_pilot_20261028_v3 | 0.005 GiB | 2026-10-28 10:51 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_outcome_valid_global_20261028_v1 | 0.005 GiB | 2026-10-28 11:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_object_outcome_valid_spatial_20261028_v1 | 0.005 GiB | 2026-10-28 11:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_proprio_fullsplit_pilot_20261026_v1 | 0.040 GiB | 2026-10-26 01:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_proprio_qonly_fullsplit_pilot_20261026_v1 | 0.041 GiB | 2026-10-26 01:02 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_proprio_qtoken_fullsplit_pilot_20261026_v1 | 0.041 GiB | 2026-10-26 01:12 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_real_future_goal_mujoco_20261030_25frame_probe_v1 | 0.000 GiB | 2026-10-30 02:03 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_real_future_goal_mujoco_20261030_25frame_probe_v2 | 0.000 GiB | 2026-10-30 02:04 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_real_future_goal_mujoco_20261030_25frame_probe_v3 | 0.000 GiB | 2026-10-30 02:06 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_real_future_goal_mujoco_20261030_contact_probe_v1 | 0.000 GiB | 2026-10-30 02:07 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_real_future_goal_mujoco_20261030_v1 | 0.025 GiB | 2026-10-30 01:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_real_future_goal_progress_20261030_v1 | 0.049 GiB | 2026-10-30 00:46 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_rep_history_smoke_20261027_4steps | 0.000 GiB | 2026-10-27 07:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_rep_history_smoke_20261027_54d_velocity | 0.058 GiB | 2026-10-27 07:57 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_sam_mask_ablation_20261028 | 0.533 GiB | 2026-10-28 00:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_sam_mask_ablation_raw18_20261028 | 0.062 GiB | 2026-10-27 14:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_sam_mask_ablation_sam18_20261028_v3 | 0.062 GiB | 2026-10-27 18:41 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_sam_mask_ablation_smoke_20261028 | 0.062 GiB | 2026-10-27 14:03 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_tianji_wuji_22d_tc_transfer_20261024_v1 | 0.054 GiB | 2026-10-24 00:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_tianji_wuji_22d_tc_vision_only_scratch_20261024_v1 | 0.035 GiB | 2026-10-24 00:11 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_tianji_wuji_decoded_joint_supervision_20261024_pilot_delta001_incomplete | 0.017 GiB | 2026-10-24 04:09 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_tianji_wuji_decoded_joint_supervision_20261024_v1 | 0.037 GiB | 2026-10-24 04:47 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_tianji_wuji_proprio_joint_supervision_20261024_v1 | 0.311 GiB | 2026-10-24 06:38 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_upstream_R2_e2e_idm_20261031_crashed_v1 | 0.000 GiB | 2026-10-30 15:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_upstream_smoke_e2e_20261031 | 0.119 GiB | 2026-10-30 15:15 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_upstream_smoke_frozen_20261031 | 0.119 GiB | 2026-10-30 15:15 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_ablation_smoke_20261028_v3 | 0.022 GiB | 2026-10-28 12:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_ablation_v3_current | 0.022 GiB | 2026-10-28 12:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_ablation_v3_none | 0.022 GiB | 2026-10-28 12:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_ablation_v3_predicted_frozen | 0.022 GiB | 2026-10-28 12:23 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_ablation_v3_predicted_joint | 0.022 GiB | 2026-10-28 12:25 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_matched_q_smoke_20261028_v1 | 0.022 GiB | 2026-10-28 12:38 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_matched_q_v1_current | 0.022 GiB | 2026-10-28 12:40 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_matched_q_v1_none | 0.022 GiB | 2026-10-28 12:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_matched_q_v1_predicted_frozen | 0.022 GiB | 2026-10-28 12:41 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/lewam_vision_matched_q_v1_predicted_joint | 0.022 GiB | 2026-10-28 12:43 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_explicit_state_transition_20261030_smoke | 0.000 GiB | 2026-10-29 12:03 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_explicit_state_transition_20261030_smoke2 | 0.000 GiB | 2026-10-29 12:04 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_explicit_state_transition_20261030_smoke3 | 0.000 GiB | 2026-10-29 12:05 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_explicit_state_transition_20261030_smoke4 | 0.001 GiB | 2026-10-29 12:09 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_explicit_state_transition_20261030_v1 | 0.016 GiB | 2026-10-29 12:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_image_transition_full173_20261030_v1 | 1.481 GiB | 2026-10-30 04:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_image_transition_full173_20261030_v2_cache | 1.341 GiB | 2026-10-30 04:09 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_image_transition_full173_20261030_v3_rgb_cache | 1.432 GiB | 2026-10-30 04:40 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_contract_matched_ep56_f000_20261028 | 0.001 GiB | 2026-10-28 00:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_contract_matched_ep56_f200_20261028 | 0.001 GiB | 2026-10-28 00:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_ep56_f000_offset10cm | 0.001 GiB | 2026-10-28 00:34 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_ep56_f000_offset10cm_aligned_history | 0.001 GiB | 2026-10-28 00:40 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_ep56_f000_offset10cm_review_v2 | 0.001 GiB | 2026-10-28 00:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_ep56_f200_offset10cm | 0.001 GiB | 2026-10-28 00:34 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_ep56_f200_offset10cm_aligned_history | 0.001 GiB | 2026-10-28 00:40 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_ep56_f200_offset10cm_review_v2 | 0.001 GiB | 2026-10-28 00:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_v1 | 0.000 GiB | 2026-10-28 00:33 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_v2 | 0.001 GiB | 2026-10-28 00:33 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_sam_action_visual_behavior_20261028_v3 | 0.001 GiB | 2026-10-28 00:33 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_visual_transition_full173_20261030_smoke | 0.052 GiB | 2026-10-30 04:15 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_visual_transition_full173_20261030_v1 | 0.052 GiB | 2026-10-30 04:17 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_visual_transition_rgb_object_weighted_20261030_smoke | 0.058 GiB | 2026-10-30 04:41 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_visual_transition_rgb_object_weighted_20261030_v1 | 0.059 GiB | 2026-10-30 04:48 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_visual_transition_smoke_20261030 | 0.046 GiB | 2026-10-29 13:37 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/mujoco_visual_transition_train_20261030_v1 | 0.047 GiB | 2026-10-29 13:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/multiscale_future_image_action_20261030_v1 | 0.703 GiB | 2026-10-30 02:40 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/multiscale_future_image_action_train_20261030_v1 | 0.039 GiB | 2026-10-30 03:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/object_masks_qc_20261101 | 0.005 GiB | 2026-11-01 14:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/official_lewm_tianji_cube_init_100e_v1 | 0.303 GiB | 2026-10-23 01:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/policy_world_aux_from_imagewm_20261030_v1 | 0.022 GiB | 2026-10-30 03:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/q_baseline_residual_flow_20261024_v1 | 0.042 GiB | 2026-10-24 06:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/q_baseline_residual_flow_delta10x_20261024_v1 | 0.042 GiB | 2026-10-24 07:02 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen2vl_goal_cache_20261026_blacktray_v1 | 0.000 GiB | 2026-10-26 04:52 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen2vl_goal_cache_20261026_grounding_v2 | 0.000 GiB | 2026-10-26 05:27 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen2vl_goal_cache_20261026_v1 | 0.000 GiB | 2026-10-26 04:47 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen2vl_goal_grounding_pilot_20261026 | 0.004 GiB | 2026-10-26 04:32 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen2vl_lewam24_e2e_lora_multiepisode_100step_20261026_v4 | 0.001 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen2vl_lewam24_e2e_lora_pilot_20261026_v3 | 0.001 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen2vl_lewam_frozen_flow_64win_1000step_20261026_v1 | 0.002 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen2vl_lewam_frozen_head_ablation_16win_200step_20261026_v1 | 0.000 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen_dino_sam_fusion_lewam24_20261027_first20_v1 | 0.004 GiB | 2026-10-27 06:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/qwen_dino_sam_fusion_lewam24_20261027_v1 | 0.029 GiB | 2026-10-26 15:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/real_next_latent_dynamics_20261030_v1 | 0.026 GiB | 2026-10-29 21:48 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/reanchor_fusion_v1 | 0.000 GiB | 2026-10-22 13:43 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_deterministic_fullcache_ep20_v1 | 0.008 GiB | 2026-10-22 13:53 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_flow_fullcache_ep20_v1 | 0.008 GiB | 2026-10-22 13:47 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_flow_h5_v1 | 0.008 GiB | 2026-10-22 13:14 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_flow_h5_v2 | 0.010 GiB | 2026-10-22 13:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_flow_h5_v2_eval_episode_shared | 0.000 GiB | 2026-10-22 13:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_flow_h5_v2_eval_per_target | 0.000 GiB | 2026-10-22 13:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_flow_h5_v2_eval_zero | 0.000 GiB | 2026-10-22 13:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_flow_h5_v2_eval_zero_priority_sweep | 0.000 GiB | 2026-10-22 13:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/residual_head_comparison_fullcache_v1 | 0.010 GiB | 2026-10-22 13:55 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/sam_action_preprocess_contract_audit_20261028 | 0.000 GiB | 2026-10-28 00:49 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_AW_gpu_20261101 | 0.008 GiB | 2026-11-01 06:17 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_A_20261101 | 0.000 GiB | 2026-11-01 06:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_A_final_20261101 | 0.000 GiB | 2026-11-01 06:09 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_A_fp32_20261101 | 0.000 GiB | 2026-11-01 06:11 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_A_gpu_20261101 | 0.004 GiB | 2026-11-01 06:13 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_A_retry_20261101 | 0.000 GiB | 2026-11-01 06:02 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_A_w0_20261101 | 0.000 GiB | 2026-11-01 06:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_W_gpu_20261101 | 0.000 GiB | 2026-11-01 06:14 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/scorer_v3_smoke_W_retry_20261101 | 0.008 GiB | 2026-11-01 06:16 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/sim_native_joint_smoke2_20261027 | 0.000 GiB | 2026-10-27 07:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/sim_native_joint_smoke3_20261027 | 0.000 GiB | 2026-10-27 07:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/sim_native_joint_smoke_20261027 | 0.000 GiB | 2026-10-27 07:20 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/simple_gru_dynamics_full_20261028_v1 | 0.001 GiB | 2026-10-28 10:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/simple_gru_dynamics_overfit128_20261028_v1 | 0.001 GiB | 2026-10-28 10:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/simple_mlp_dynamics_full_20261028_v1 | 0.001 GiB | 2026-10-28 10:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/simple_mlp_dynamics_overfit128_20261028_v1 | 0.001 GiB | 2026-10-28 10:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/spatial_heatmap_current_scene_20261026_v1 | 0.022 GiB | 2026-10-25 13:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/spatial_heatmap_localizer_20261025_v1 | 0.217 GiB | 2026-10-25 12:14 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/spatial_localizer_multiepisode_20261025_v1 | 0.455 GiB | 2026-10-25 12:47 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/task_conditioned_image_action_future_value_pilot_20261030_v1 | 0.068 GiB | 2026-10-29 14:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/temporal_fusion_v1 | 0.000 GiB | 2026-10-22 13:11 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/time_alignment_audit_v1 | 0.000 GiB | 2026-10-22 13:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/top_active_gaze16_multiscale_50e_v1 | 0.008 GiB | 2026-10-23 02:35 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/top_center16_multiscale_50e_v1 | 0.012 GiB | 2026-10-22 06:34 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/v4_smoothing_residual_flow_v1 | 0.029 GiB | 2026-10-23 06:04 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/value_ttg_E23a_smoke2_20261101 | 0.130 GiB | 2026-11-01 07:57 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/value_ttg_E23a_smoke3_20261101 | 0.130 GiB | 2026-11-01 08:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/value_ttg_E23a_smoke_20261101 | 0.130 GiB | 2026-11-01 07:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/visual_grounding_readout_20261027_v1 | 0.029 GiB | 2026-10-26 14:51 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_cpu_half_20261029 | 0.000 GiB | 2026-10-28 15:16 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_feature_state_eval_smoke_20261030 | 0.041 GiB | 2026-10-29 05:39 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_matched_goalpool_flow_20261029_v2 | 0.045 GiB | 2026-10-29 01:51 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_matched_goalpool_mse_20261029_v2 | 0.043 GiB | 2026-10-29 05:57 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_policy_spatial_future_aux_20261029_v1 | 0.039 GiB | 2026-10-29 03:23 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_policy_spatial_future_no_action_20261030_v1 | 0.041 GiB | 2026-10-29 05:04 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_policy_spatial_future_residual_20261029_v1 | 0.041 GiB | 2026-10-29 03:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_policy_spatial_future_state_20261030_v1 | 0.000 GiB | 2026-10-29 05:12 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_policy_spatial_future_state_20261030_v4 | 0.147 GiB | 2026-10-29 13:32 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_policy_spatial_query_mse_20261029_v1 | 0.039 GiB | 2026-10-29 02:57 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_spatial_goal_motion_fk_20261029_v2 | 0.000 GiB | 2026-10-29 06:54 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_spatial_goal_motion_fk_20261029_v3 | 0.046 GiB | 2026-10-29 07:45 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_spatial_goal_motion_fk_no_action_20261029_v1 | 0.045 GiB | 2026-10-29 07:42 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_spatial_probe_capacity_matched_20261030 | 0.000 GiB | 2026-10-29 06:33 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_token_motion_audit_20261029_v1 | 0.000 GiB | 2026-10-29 04:06 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_token_motion_train_audit_20261029_v1 | 0.000 GiB | 2026-10-29 04:03 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_tokens_half_flow_stage0_20261029_v1 | 0.047 GiB | 2026-10-28 23:51 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_future_region_value_20261030_v2 | 0.000 GiB | 2026-10-29 15:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_policy_aux_smoke2_20261029 | 0.050 GiB | 2026-10-29 02:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_policy_aux_smoke3_20261029 | 0.000 GiB | 2026-10-29 02:23 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_policy_aux_smoke4_20261029 | 0.039 GiB | 2026-10-29 02:23 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_policy_aux_smoke5_20261029 | 0.039 GiB | 2026-10-29 02:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_policy_aux_smoke_20261029 | 0.000 GiB | 2026-10-29 02:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_policy_compat_smoke_20261029 | 0.039 GiB | 2026-10-29 02:25 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_policy_residual_smoke2_20261029 | 0.041 GiB | 2026-10-29 03:33 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vjepa21_world_policy_residual_smoke_20261029 | 0.041 GiB | 2026-10-29 03:32 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_action_annotation_review_20261024 | 0.001 GiB | 2026-10-23 13:23 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_action_chunks_20261024_seed3072 | 0.010 GiB | 2026-10-23 13:26 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_action_chunks_20261024_seed4096 | 0.009 GiB | 2026-10-23 13:23 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_action_chunks_20261024_seed5120 | 0.009 GiB | 2026-10-23 13:24 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_action_chunks_report_20261024 | 0.003 GiB | 2026-10-23 13:35 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_action_chunks_smoke_20261024 | 0.003 GiB | 2026-10-23 13:18 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_conditioned_lewm_full_seed3072 | 0.000 GiB | 2026-10-23 08:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_conditioned_lewm_full_seed4096 | 0.000 GiB | 2026-10-23 08:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_conditioned_lewm_full_seed5120 | 0.000 GiB | 2026-10-23 09:05 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_conditioned_lewm_full_v1 | 0.027 GiB | 2026-10-23 07:51 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_conditioned_lewm_pilot_v1 | 0.002 GiB | 2026-10-23 06:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_conditioned_lewm_pilot_v2 | 0.002 GiB | 2026-10-23 07:02 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_lewm_diagnostics_v1 | 0.000 GiB | 2026-10-23 05:17 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_lewm_infer_ep056_v1 | 0.001 GiB | 2026-10-23 03:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_comparison_v1 | 0.000 GiB | 2026-10-23 04:53 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_eval_v1 | 0.004 GiB | 2026-10-23 04:52 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_nogoal_eval_v1 | 0.004 GiB | 2026-10-23 04:53 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_nogoal_smoke_eval_v1 | 0.002 GiB | 2026-10-23 04:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_nogoal_smoke_v1 | 0.000 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_nogoal_v1 | 0.000 GiB | 2026-10-28 05:59 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_residual_smoke_eval_v1 | 0.002 GiB | 2026-10-23 03:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_residual_smoke_eval_v2 | 0.002 GiB | 2026-10-23 03:56 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_residual_smoke_eval_v3 | 0.002 GiB | 2026-10-23 04:11 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_residual_smoke_v1 | 0.000 GiB | 2026-10-23 03:50 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_residual_smoke_v2 | 0.000 GiB | 2026-10-23 03:55 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_residual_smoke_v3 | 0.000 GiB | 2026-10-24 07:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_residual_smoke_v4 | 0.000 GiB | 2026-10-24 07:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_residual_smoke_v5 | 0.000 GiB | 2026-10-24 07:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_smoke_eval_v1 | 0.002 GiB | 2026-10-23 03:48 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_smoke_v1 | 0.000 GiB | 2026-10-24 07:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_phase_policy_v1 | 0.000 GiB | 2026-10-24 07:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_compat_smoke_20261026_v1 | 0.000 GiB | 2026-10-26 00:10 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_full_dataset_annotations_20261026_v1 | 0.004 GiB | 2026-10-26 00:31 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_full_shardA_20261026_v1 | 0.001 GiB | 2026-10-26 00:30 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_full_shardB_20261026_v1 | 0.001 GiB | 2026-10-26 00:27 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_probe_20261026_v1 | 0.000 GiB | 2026-10-25 14:14 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_probe_20261026_v2 | 0.000 GiB | 2026-10-25 14:16 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_probe_20261026_v3 | 0.000 GiB | 2026-10-25 14:18 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_probe_20261026_v4 | 0.001 GiB | 2026-10-25 14:19 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_probe_20261026_v5 | 0.001 GiB | 2026-10-25 14:21 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_probe_20261026_v6 | 0.001 GiB | 2026-10-25 14:22 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_subset_annotations_20261026_v1 | 0.007 GiB | 2026-10-25 21:43 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_florence2_temporal_probe_20261026_v1 | 0.006 GiB | 2026-10-25 14:29 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_goal_frame_cache_20261026_v1 | 0.000 GiB | 2026-10-25 21:43 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_reentry_goal_frame_cache_full_20261026_v1 | 0.001 GiB | 2026-10-26 00:31 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_semantic_policy_v2 | 0.000 GiB | 2026-10-23 05:47 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_semantic_policy_v2_corrected | 0.000 GiB | 2026-10-23 05:47 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_semantic_policy_v2_eval | 0.010 GiB | 2026-10-23 05:35 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_semantic_policy_v3_eval | 0.010 GiB | 2026-10-23 05:38 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_semantic_policy_v3_tracking_dropout50 | 0.000 GiB | 2026-10-23 05:47 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_semantic_policy_v4_eval | 0.010 GiB | 2026-10-23 05:41 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_semantic_policy_v4_rollout | 0.000 GiB | 2026-10-24 07:36 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/vlm_semantic_policy_v4_smooth_eval | 0.018 GiB | 2026-10-23 06:01 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/wan_smoke_image_only_20261031 | 0.057 GiB | 2026-10-31 01:08 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/wan_smoke_latent_image_20261031 | 0.057 GiB | 2026-10-31 01:07 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/wm_viz_E_20261101 | 0.009 GiB | 2026-10-31 16:26 | dengjianfeng | 精确0 / 名称2 | 仅目录名/run ID 提及，具体依赖未确认 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/runs/wrist_inspect_20261031 | 0.000 GiB | 2026-10-31 06:06 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e98w_a_20261114_01 | 0.000 GiB | 2026-11-14 14:09 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e21_sam3_rawfix_20261101 | 0.001 GiB | 2026-11-01 08:09 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e21_sam3_test_20261101 | 0.002 GiB | 2026-11-01 08:00 | dengjianfeng | 精确0 / 名称0 | 无精确引用，但历史用途/可重建性未核实 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/lang_wam_cache_20261110 | 0.066 GiB | 2026-11-10 23:15 | dengjianfeng | 精确0 / 名称0 | 旧缓存候选；无精确路径引用，但本轮未验证生成脚本和可重建性，不满足 C 条件 |
| /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/tianji_kinematics_20261101 | 0.009 GiB | 2026-11-01 07:23 | dengjianfeng | 精确0 / 名称3 | 仅目录名/run ID 提及，具体依赖未确认 |

## 文件类型、重复与碎片

| 类别 | 文件数 | apparent GiB |
|---|---:|---:|
| checkpoint | 1385 | 220.364 |
| feature_cache | 19985 | 329.127 |
| image | 48167 | 5.472 |
| log | 2518 | 1.071 |
| other | 23495 | 53.591 |

Checkpoint 清单 1385 个；同 basename+同大小重复线索见 checkpoint_duplicate_clues.tsv；未对所有 checkpoint 全量计算 SHA，故不是重复性证明。SHA/路径/run ID 原文线索见 references_grep.txt 和 references_by_directory.txt。≥10 GiB 单文件 12 个，见 large_files.tsv。清理后最大 runs 目录为 e63_encoder_e2e_20261108（约 49.60 GiB）；e28_chain_20261103 现约 6.37 GiB。空目录扫描未完成。

项目内 symlink 清单扫描主动中止，记录首批 418 条链接；链接清单阶段没有跟随链接。文件类别扫描的文件筛选调用 isfile，可能对文件 symlink 目标额外 stat，但未读目标内容；首批链接目标在项目内，未完成清单不能排除其他外部目标。未扫描同事目录及其进程，因此他人是否软链到本项目未知。

## 活跃/风险与未完成

- 活跃，勿动：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e98_predicate_labels_train_causal_v1；未执行 du/类别扫描，大小未测。
- E111-F 按用户提供信息标注“活跃，勿动”，不操作相关进程。
- /mnt 初查 102T 总量、97T 已用、约 5.0T 可用（96%）；最近复查约 4.1T 可用（96%），共享盘占用持续变化。
- 禁止目录 /mnt/workspace/colleague-a、colleague-e、colleague-c、colleague-d、colleague-b、colleague-f 与 /mnt/workspace/dataset 未扫描；相关软链接风险未知。
- du 超时/失败：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1 [timeout_or_error]; /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/piper_events_20261106 [timeout_or_error]。

## 可释放估算与建议顺序

- B 压缩预计 0.000–0.001 GiB；C 全部删除上限 0.000 GiB；D 中缓存候选不计入。
- 若用户批准全部 B/C，粗估释放 0.000–0.001 GiB；C 的可再生性与成本未实测，因此此值是上限区间。
- 建议顺序（只提议）：先确认 run E 与外部链接风险；再确认 B 日志审计要求并选择压缩归档；逐个确认 C 的源数据/脚本/重建成本后，再由用户逐类批准；A 和活跃目录冻结。
- 模型生成标注只作为训练目标/训练池筛选依据；未用于测试真值、阈值或 checkpoint 选型。

## E28 特征缓存清理后更新（用户授权）

- 删除 runs/e28_chain_20261103 下 6 个 value/*/value/features 目录中的 72 个普通 .npy 文件；保留状态文件、全部 checkpoint、日志、报告、评估 JSON 和目录结构。
- 文件 apparent size 约 96.333 GiB；目录 du 从约 102.705 GiB 降至 6.372 GiB。删除清单：results/e112s_inventory_20261116/e28_feature_delete_manifest.tsv。

## 指定六目录清理后更新（用户授权）

- 删除 122 个 .npy/.npz 特征数组（约 146.054 GiB apparent）和 129 个旧 checkpoint（约 44.558 GiB apparent）；按父级训练子目录逐组保留最新修改的 checkpoint，共 46 组。配套 JSON/日志/脚本、报告与评估结果保留。完整删除清单：results/e112s_inventory_20261116/selected_six_cleanup_delete_manifest.tsv；计划与保留记录：selected_six_cleanup_plan.tsv。
- 清理后实际 du（GiB）：e54_ablation_20261106 11.735；embed_analysis_E_20261101 0.001；e61_endpoint_20261108 7.545；e32_jepa_distill_20261103 14.832；e33_20261104 10.868；vjepa21_cpu_half_20261029 约 0.000。251 个删除目标均已核验不存在，46 个保留 checkpoint 均存在。
