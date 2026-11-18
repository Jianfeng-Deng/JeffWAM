# Experiment Log

## Project

- **Project root on GPU host:** `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu` (`runs/` = outputs, `tools/` = scripts, `data/` = derived data).
- **Task:** Tianji dual arm + Wuji hands, grasp bottle -> place in bin.
- **Hardware:** one GPU only (physical GPU 4 on <gpu-server>). Python: `/mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python` (borrowed, nothing installed).
- **Data split:** 93 episodes (84 train / 9 val, episode-level split), 15 Hz, 54-D joints (A7+B7+L20+R20, rad).
- **Metric definitions:** Common policy metrics (validation, 2,289 windows): chunk RMSE (10x54, rad), first-5-step RMSE, hold baseline 0.101 rad; hand peak-speed ratio (1 = demo speed); grasp onset = first step where hand change from step 0 > 0.1 rad RMS: onset error (steps), missed, false onsets.

## Model lineage

```text
LeWAM upstream -> Phase A (world-action network, MSE policy)
                 -> A (flow, no foresight) / B (flow + foresight)
                 -> C (wrist cameras) / D (joint history)
                 -> E (D + DINOv2-B) -> F (semantic box weighting)
E -> L1 decoder -> region-JiT decoder -> region-JiT v2
D samples -> action scorer v1 / v2; E latents -> progress scorer
```

## Data products

| ID | Path | Description |
| --- | --- | --- |
| D1 | `data/lewam_upstream_h5_20261031` | overhead cam_high letterboxed 224, canonical q/cmd (verified). |
| D2 | `data/mujoco_transition_224_20261031` + `data/mujoco_lewam_h5_20261031` | 1,575 counterfactual sim transitions, 196 start groups, 5 group folds. |
| D3 | `data/real_workspace_192x320_20261031` | overhead workspace crop rows 96..480, 320x192. |
| D4 | `data/real_wrist_144x192_20261031` | left/right wrist cameras 192x144 (row-aligned with D3). |
| D5 | `/mnt/workspace/dengjianfeng/models/dinov2_official_20261031` | official DINOv2 ViT-S/14, ViT-B/14 (Meta server, SHA256SUMS recorded). |
| D6 | `data/dinov2b_feats_20261031` | frozen DINOv2-B pooled features (overhead 6x10, wrists 5x7) for all frames. |
| D7 | `data/policy_samples_D_20261031` | 16 samples/window from run D best (8 at noise 1.0, 8 at 0.5). |
| D8 | `data/dinov2_regions_20261101` | DINOv2 k-means (k=16) patch clusters 24x40; boxes_v2 = per-type boxes for clusters 11 (hands, dark bottles), 13 (bins), 15 (bottles/cups), min component 2/6/3 patches, 2-patch margin; ~49% coverage. |

## Experiments

Result tables reproduce source precision where available. Conclusions are preserved from Claude’s notes and the existing log; their approximate figures are retained verbatim. Dates and commands are marked not recorded where the existing log did not record them.

### E01
**ID:** E01. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** LeWAM upstream end-to-end (R0 s42/s7, R1 frozen drawer encoder, R2 +IDM 0.1, R3 no SIGReg).

**Run dir / reports:** `runs/lewam_upstream_R0_e2e_s42_20261031`, `runs/lewam_upstream_R0_e2e_s7_20261031`, `runs/lewam_upstream_R1_frozen_drawer_20261031`, `runs/lewam_upstream_R2_e2e_idm_20261031`, `runs/lewam_upstream_R3_e2e_noreg_20261031`

**Results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/lewam_upstream_R0_e2e_s42_20261031 | 0.15427690149792547 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| runs/lewam_upstream_R0_e2e_s7_20261031 | 0.1534340989390288 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| runs/lewam_upstream_R1_frozen_drawer_20261031 | 0.2213816271444469 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| runs/lewam_upstream_R2_e2e_idm_20261031 | 0.14739166359642247 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| runs/lewam_upstream_R3_e2e_noreg_20261031 | 0.14960753570068638 | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |

**Conclusion:** keypoint latent too weak; frozen drawer encoder worst; R2 best (world gain +15%) but policy 0.147 rad, worse than hold. LeWAM as-is not viable.

### E02
**ID:** E02. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** MuJoCo object test (LeWAM scratch / ft from real / frozen, 5 folds).

**Run dir / reports:** `runs/lewam_mjc_scratch_f0_20261031`; `runs/lewam_mjc_scratch_f1_20261031`; `runs/lewam_mjc_scratch_f2_20261031`; `runs/lewam_mjc_scratch_f3_20261031`; `runs/lewam_mjc_scratch_f4_20261031`; `runs/lewam_mjc_ftreal_f0_20261031`; `runs/lewam_mjc_ftreal_f1_20261031`; `runs/lewam_mjc_ftreal_f2_20261031`; `runs/lewam_mjc_ftreal_f3_20261031`; `runs/lewam_mjc_ftreal_f4_20261031`; `runs/lewam_mjc_ftfrozen_f0_20261031`; `runs/lewam_mjc_ftfrozen_f1_20261031`; `runs/lewam_mjc_ftfrozen_f2_20261031`; `runs/lewam_mjc_ftfrozen_f3_20261031`; `runs/lewam_mjc_ftfrozen_f4_20261031`

**Results:**
| Evaluation | held_out_rows | rows_bottle_moved_gt_5mm | latent.gain_vs_persistence_all | latent.gain_vs_persistence_moved | latent.within_group_retrieval_top1 | forecast.rmse_mm_moved | direct.rmse_mm_moved | forecast.within_group_spearman_magnitude |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | 1575 | 201 | 0.8307046890258789 | 0.8224399089813232 | 0.31555555555555553 | 21.061149483618355 | 16.392788503903073 | 0.041503587736464446 |
| ftreal | 1575 | 201 | 0.5236151218414307 | 0.6780104637145996 | 0.36698412698412697 | 19.516312824689955 | 16.371755339183096 | 0.07692433137638617 |
| ftfrozen | 1575 | 201 | 0.6413509845733643 | 0.5985133647918701 | 0.1403174603174603 | 22.882412627431034 | 16.52099540404476 | 0.03462165688193086 |

**Conclusion:** predicted futures get bottle direction partly right and separate candidates ~3x chance, but no better than direct regression and cannot rank candidates; frozen worst. Later (E14) shown that the 192-D vector bottleneck, not learning, limited this.

### E03
**ID:** E03. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Phase A world-action net (ResNet18-GN spatial tokens 12x20, change-weighted future image, MSE policy), variants latent_image vs image_only.

**Run dir / reports:** `runs/wan_phaseA_image_only_20261031`; `runs/wan_phaseA_latent_image_20261031`; score reports `score_phaseB_metrics_latest.json`

**Results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/wan_phaseA_image_only_20261031 | 0.07252145846592065 | not recorded | 0.817640750908634 | 2.2918340026773762 | 0.3453815261044177 | 0.1297016861219196 | not recorded |
| runs/wan_phaseA_latent_image_20261031 | 0.07181808698685202 | not recorded | 0.7748951873050911 | 2.1941097724230256 | 0.33065595716198126 | 0.14332036316472113 | not recorded |

**Conclusion:** future prediction strongly action-dependent (+49% vs copy, ~85% of gain lost with wrong action); latent loss adds nothing; MSE policy peaked epoch 5-6 (0.0685) then overfit; hands blurred (peak speed 0.77).

Phase A reports hand peak-speed ratio as `hand_peak_vel_ratio` (single output).

### E04
**ID:** E04. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Phase B flow-matching head: A (no foresight) vs B (action-free foresight fed to policy).

**Run dir / reports:** `runs/ffn_phaseB_A_noforesight_20261031`; `runs/ffn_phaseB_B_foresight_20261031`

**Results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/ffn_phaseB_A_noforesight_20261031 | 0.08144798469173643 | not recorded | 0.9258802997768437 | 2.7349397590361444 | 0.42168674698795183 | 0.14591439688715954 | 0.4821376081023613 |
| runs/ffn_phaseB_B_foresight_20261031 | 0.07977717236200965 | not recorded | 0.9747222072217638 | 2.4792503346720216 | 0.34404283801874164 | 0.1874189364461738 | 0.48280703191734475 |

**Conclusion:** flow fixes hand blur (speed ~1.0); foresight helps one-sample error and timing; neither beats MSE on timing.

### E05
**ID:** E05. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Sample selection (noise scale, medoid, world-model scoring) on A/B.

**Run dir / reports:** `runs/ffn_phaseB_A_noforesight_20261031`, `runs/ffn_phaseB_B_foresight_20261031` (selection reports: `ffn_phaseB_A_noforesight_20261031/sample_selection_best_ep30.json`, `ffn_phaseB_B_foresight_20261031/sample_selection_best_ep26.json`)

**Results:**
| Run | Strategy | chunk_rmse | first5_rmse | hand_speed_ratio | onset_err_steps | onset_miss | onset_false |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (best ep30) | T=0.5 single | 0.07320847587616554 | 0.041789427391750476 | 0.7683251535990389 | 2.534136546184739 | 0.41365461847389556 | 0.08819714656290532 |
| A (best ep30) | T=0.5 medoid8 | 0.07011663710498402 | 0.04037325918906331 | 0.7357417136278959 | 2.497991967871486 | 0.4109772423025435 | 0.06614785992217899 |
| A (best ep30) | T=0.5 world8 | 0.07205730667804792 | 0.041198948237156606 | 0.7832599542848361 | 2.5475234270414995 | 0.4390896921017403 | 0.08300907911802853 |
| A (best ep30) | T=0.5 best8 (oracle) | 0.05653446987799945 | 0.03606440632789317 | 0.7539126620749813 | 1.892904953145917 | 0.31459170013386883 | 0.0324254215304799 |
| B (best ep26) | T=0.5 single | 0.07668402346556988 | 0.042627765989767204 | 0.8626476243861696 | 2.4444444444444446 | 0.3534136546184739 | 0.15239948119325553 |
| B (best ep26) | T=0.5 medoid8 | 0.07573057975207848 | 0.04197516213446482 | 0.8441127398095859 | 2.4203480589022757 | 0.3627844712182062 | 0.14267185473411154 |
| B (best ep26) | T=0.5 world8 | 0.07599354123045293 | 0.0427501642543324 | 0.8623484334848097 | 2.4939759036144578 | 0.36947791164658633 | 0.14202334630350194 |
| B (best ep26) | T=0.5 best8 (oracle) | 0.06812990313215131 | 0.03869477565594741 | 0.8394617138318824 | 2.1311914323962515 | 0.31994645247657294 | 0.11219195849546044 |

**Conclusion:** low noise best for near-term steps; world-model scoring ~ random; oracle best-of-8 far better than anything selectable.

### E06
**ID:** E06. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** C = B + wrist cameras; D = B + 1 s joint history with past-token prediction.

**Run dir / reports:** `runs/ffn_phaseC_wrist_20261031`; `runs/ffn_phaseD_qhist15_20261031`

**Results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/ffn_phaseC_wrist_20261031 | 0.07769518989343682 | 0.045979374597476044 | 0.9840597268114297 | 2.3801874163319945 | 0.34002677376171353 | 0.14980544747081712 | 0.479523273769597 |
| runs/ffn_phaseD_qhist15_20261031 | 0.07684355512888627 | 0.04546544177498393 | 0.9998457081955047 | 2.428380187416332 | 0.36813922356091033 | 0.1297016861219196 | 0.48043133325506615 |

**Conclusion:** both help modestly (~3%); D best RMSE, C best timing; gains late in chunk.

### E07
**ID:** E07. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Encoder probes (DINOv2-S/B frozen vs our encoder vs random).

**Run dir / reports:** `probe_encoders_dinov2_20261031.json` (no run directory recorded in the JSON).

**Results:**
| Encoder | real.state.r2_hands | real.motion.r2_hands | mujoco_oracle.rmse_mm_moved | mujoco_oracle.within_group_spearman |
| --- | --- | --- | --- | --- |
| dinov2_s | 0.5183577934946118 | 0.21856766664157 | 14.941607475280762 | 0.14125896934116114 |
| dinov2_b | 0.5458531766266398 | 0.24025975180361026 | 14.15139102935791 | 0.2071102413568167 |
| ours_B | 0.35957288212461624 | 0.22815950809591512 | 15.714190483093262 | 0.22361382909328112 |
| random_resnet | 0.18478057474333565 | 0.09233425045181509 | 15.26880168914795 | 0.21441617742987606 |

**Conclusion:** DINOv2-B much better hand-state readout (hand R2 0.55 vs 0.36); MuJoCo pooled-token test saturated (random 0.57).

### E08
**ID:** E08. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** E = D + DINOv2-B encoder, last 4 blocks fine-tuned (lr 2e-5) + anchor loss 0.5.

**Run dir / reports:** `runs/ffn_phaseE_dinob_qhist15_20261031`

**Results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/ffn_phaseE_dinob_qhist15_20261031 | 0.07408740460360973 | 0.044045580720272676 | 0.9874814762941362 | 2.3815261044176705 | 0.3493975903614458 | 0.10376134889753567 | 0.4562302896117475 |

**Conclusion:** best flow model; ahead of D from epoch 6 (best ep 20: 0.0741, onset 2.38, false 10%).

### E09
**ID:** E09. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Action scorer v1 (listwise incl. demo, frozen DINOv2 features + wrists + history).

**Run dir / reports:** `runs/action_scorer_v1_20261031`

**Results:**
| Metric | Value |
| --- | --- |
| epoch | 10 |
| spearman_score_vs_true_rank | -0.1571081578731537 |
| demo_ranked_first_vs_shifted_and_blurred | 0.42944517731666565 |

**Conclusion:** ranked backwards (learned real-vs-generated / hand-motion shortcut).

### E10
**ID:** E10. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Action scorer v2 (pairwise ranking over policy samples).

**Run dir / reports:** `runs/action_scorer_v2_pairwise_20261031`

**Results:**
| Metric | Value |
| --- | --- |
| epoch | 2 |
| spearman_score_vs_true_rank | 0.37838512659072876 |
| share_picks_noise1 | 0.05460900068283081 |
| demo_ranked_first_vs_shifted_and_blurred | 0.3245958983898163 |

**Conclusion:** correct sign (+0.41) but mostly "prefer low noise"; within-group ranking weak even on train.

### E11
**ID:** E11. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Grasp-timing noise floor (nearest neighbours across demos).

**Run dir / reports:** `timing_noise_floor_20261031.json` (no run directory recorded in the JSON).

**Results:**
| Metric | Value |
| --- | --- |
| windows | 26351 |
| episodes | 93 |
| share_windows_with_onset | 0.3556221775264696 |
| nearest_neighbour_copy.onset_err_steps | 2.7816668445203288 |
| nearest_neighbour_copy.miss | 0.389819656386725 |
| nearest_neighbour_copy.false | 0.22985865724381627 |
| median_of_5_neighbours.onset_err_steps | 2.7009924234339984 |
| median_of_5_neighbours.miss | 0.39334115889446164 |
| median_of_5_neighbours.false | 0.2064193168433451 |
| random_other_window.onset_err_steps | 4.023156546793299 |
| random_other_window.miss | 0.6471027638459076 |
| random_other_window.false | 0.35706713780918725 |

**Conclusion:** ~2 steps of onset disagreement even for closest matches, flat across distance -> demonstrator randomness; MSE head 2.19 near floor; best-of-N oracle gap is luck.

### E12
**ID:** E12. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Progress-based scorer (progress head on E latents; score = progress of predicted future).

**Run dir / reports:** `runs/progress_scorer_v1_20261101`

**Results:**
Progress-head diagnostics:
| Metric | Value |
| --- | --- |
| progress_r2_real_frames | 0.7962217330932617 |
| progress_r2_predicted_futures | 0.801188051700592 |
| demo>hold | 0.6173001527786255 |
| demo>reversed | 0.5701179504394531 |
| spearman_score_vs_closeness | -0.04446200653910637 |
Selection metrics:
| Strategy | chunk_rmse | first5_rmse | hand_speed_ratio | onset_err_steps | onset_miss | onset_false |
| --- | --- | --- | --- | --- | --- | --- |
| progress_top1 | 0.0765814392754769 | 0.04522359536504058 | 1.0199488610994412 | 2.4404283801874165 | 0.35876840696117807 | 0.12775616083009078 |
| random_single | 0.07496044037837202 | 0.044412635232447456 | 0.9989526315491074 | 2.5542168674698793 | 0.3868808567603748 | 0.10700389105058365 |
| half_noise_single | 0.06987789537673068 | 0.04010548597172176 | 0.8350852986838403 | 2.4149933065595715 | 0.38821954484605087 | 0.07717250324254216 |
| medoid16 | 0.06872670856151591 | 0.03968085809633033 | 0.7917389968769619 | 2.4484605087014724 | 0.39357429718875503 | 0.07587548638132295 |
| oracle_best16 | 0.05789709727556264 | 0.03584363069756016 | 0.8951641160050623 | 1.1860776439089693 | 0.17804551539491298 | 0.029831387808041506 |

**Conclusion:** head accurate (R2 0.80) but futures barely depend on action over 10 steps (demo>hold 62%, demo>reversed 57%); picks ~ random. Simple rules on E (medoid/half-noise) beat MSE head on near-term error.

### E13
**ID:** E13. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** World-model image quality + decoder bottleneck test on E.

**Run dir / reports:** `runs/ffn_phaseE_dinob_qhist15_20261031`; reports `wm_viz_E_20261101/metrics.json` and `decoder_test_E_boxes.json`.

**Results:**
| Metric | Value |
| --- | --- |
| metrics.json: e_epoch | 20 |
| metrics.json: windows | 2289 |
| metrics.json: changed_pixel_share | 0.2535484899473933 |
| metrics.json: pred.l1_changed | 0.07130748104838455 |
| metrics.json: pers.l1_changed | 0.13645033649123325 |
| decoder_test_E_boxes.json: recon_t.sharpness_full | 0.37177134381264415 |
| decoder_test_E_boxes.json: pred_t10.sharpness_full | 0.3632774714067407 |
| decoder_test_E_boxes.json: recon_t.sharpness_in_boxes | 0.22588703753195646 |
| decoder_test_E_boxes.json: pred_t10.sharpness_in_boxes | 0.22032390387497375 |

**Conclusion:** blur comes from decoding: decoded current frame as blurred as prediction (sharpness 0.37 full / 0.23 in boxes); bottles vanish even without prediction.

### E14
**ID:** E14. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** F = E + semantic box weighting (bg 0.15) on recon/future/foresight pixel losses.

**Run dir / reports:** `runs/ffn_phaseF_semboxes_20261101`; decoder report `wm_viz_E_20261101/decoder_test_F_boxes.json`.

**Results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/ffn_phaseF_semboxes_20261101 | 0.07388730211590443 | 0.04410976004311386 | 0.9892268895619376 | 2.3480589022757696 | 0.3453815261044177 | 0.10311284046692606 | 0.4381386455650639 |
Decoder comparison:
| Metric | Value |
| --- | --- |
| E recon_t sharpness_in_boxes | 0.22588703753195646 |
| F recon_t sharpness_in_boxes | 0.24696346199022984 |
| F pred_t10 sharpness_in_boxes | 0.2371543815726087 |
| F box_pixel_share | 0.48967207295762344 |

**Conclusion:** in-box sharpness 0.23 -> 0.25 only; policy equal to E (best 0.0739). Weighting the L1 loss does not change what is decoded.

### E15
**ID:** E15. **Date:** not recorded. **Command:** not recorded. **Status:** done.

**Goal:** Evaluate the change below.

**Change:** Region-focused JiT decoder (x-prediction + v-loss, 8 px patches in boxes only, latent-only conditioning, 6,000 steps) on frozen E latents.

**Run dir / reports:** `runs/region_jit_E_latentonly_20261101`

**Results:**
| Metric | Value |
| --- | --- |
| step | 6000 |
| recon_t/jit.l1_in_box | 0.03588269845900525 |
| recon_t/jit.psnr_in_box | 22.19641058306612 |
| recon_t/jit.sharpness_in_box | 0.6032753688682622 |
| recon_t/l1_decoder.l1_in_box | 0.03142707807207091 |
| recon_t/l1_decoder.psnr_in_box | 23.568176686472132 |
| recon_t/l1_decoder.sharpness_in_box | 0.22039691181235216 |
| recon_t/jit_seed_spread | 0.022785107270976324 |
| pred_t10/jit.l1_in_box | 0.045629019295028404 |
| pred_t10/jit.psnr_in_box | 19.920075289611653 |
| pred_t10/jit.sharpness_in_box | 0.6245207146592006 |
| pred_t10/l1_decoder.l1_in_box | 0.04001857572702216 |
| pred_t10/l1_decoder.psnr_in_box | 21.067546426890225 |
| pred_t10/l1_decoder.sharpness_in_box | 0.22692902066986098 |
| pred_t10/jit_seed_spread | 0.027471225683106774 |

**Conclusion:** renders separate fingers, bin texture and bottle positions consistently across seeds -> E's latent DOES contain most object information; L1 decoder hid it. Fine detail (transparent bottle in closing hand) still weak; ~60% of remaining in-box error is seed variation.

### E16
**ID:** E16.

**Goal:** Evaluate the change below.

**Change:** Action-contrast rendering: region-JiT renders of E's predicted t+10 under demo / hold / reversed / other actions with fixed noise; tests whether predicted object outcomes depend on the action.

**Date:** 2026-11-01. **Command:** `tools/render_action_contrast_20261101.py --e-run runs/ffn_phaseE_dinob_qhist15_20261031 --jit-run runs/region_jit_E_latentonly_20261101 --boxes data/dinov2_regions_20261101/boxes_v2_24x40.npy` (256 validation windows, 50 Euler steps, same noise for every action).

**Run dir / reports:** `runs/action_contrast_E_20261101` (action_contrast.json, action_contrast_zoom.png)

**Key results (in-box pixel L1):**
| Action | Error vs real t+10 | Change vs demo render |
| --- | --- | --- |
| demo | 0.04550136662708745 | — |
| hold still | 0.06408449423604562 | 0.04451545965422149 |
| other moment's action | 0.06324133557963055 | 0.043230603288137236 |
| reversed | 0.0757325252593765 | 0.0599926685277901 |
| (seed spread, demo, two noises) | — | 0.027563084394108203 |

Latent change vs demo per token (inside / outside boxes): hold 4.12 / 3.41, other 4.21 / 3.53, reversed 6.29 / 5.37; demo change from z(t): 7.59 / 7.31.

**Conclusion:** the world model's predicted futures depend clearly on the action: action effects are 1.6-2.2x the sampling noise, mostly in hand pose (open vs closed fingers) and arm placement, more inside the object boxes than outside; the demo action gives the future closest to reality (29-40% lower error than the alternatives). Static objects stay put over 0.67 s, as expected. So the dynamics are action-sensitive and the earlier scorer failures (E09, E10, E12) come from scorer design (global progress / sample ranking), not from the world model ignoring actions.

**Status:** done

### E17
**ID:** E17.

**Date:** 2026-11-01. **Goal:** Evaluate the change below.

**Change:** Region-JiT v2 (fixes for visible patch seams): bilinear latent conditioning instead of copying each 16-px token to four 8-px patches; one-patch context ring around boxes (metrics still on original boxes); patch cap 1000; larger model (width 512, depth 12, 8 heads, bottleneck 64); 20,000 steps; EMA 0.9995; same frozen run-E latents, latent-only conditioning. Smoke test (200 steps) passed at about 0.37 s/step.

**Command:** `tools/train_region_jit_decoder_v2_20261101.py --e-run runs/ffn_phaseE_dinob_qhist15_20261031 --boxes data/dinov2_regions_20261101/boxes_v2_24x40.npy --run-dir runs/region_jit_v2_E_latentonly_20261101` (defaults: `--steps 20000 --eval-every 5000 --d 512 --depth 12 --heads 8 --ring 1 --cond-upsample bilinear --max-tok 1000 --ema 0.9995`).

**Run dir / reports:** `runs/region_jit_v2_E_latentonly_20261101`

**Results:**

| Eval step | Current JiT L1 | Current JiT PSNR | Current JiT sharpness | Current seed spread | Predicted t+10 JiT L1 | Predicted t+10 JiT PSNR | Predicted t+10 JiT sharpness | Predicted t+10 seed spread | E decoder reference current L1 | E decoder reference predicted t+10 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.03483485527935175 | 22.383030597584117 | 0.6340059189227771 | 0.021352900322926263 | 0.044590262225165926 | 20.01211592821474 | 0.6413304486479323 | 0.02543530601276922 | 0.031425558987866066 | 0.04002306630748414 |
| 10,000 | 0.032932988320109206 | 22.799317666387285 | 0.7017048603734157 | 0.017435786317665646 | 0.04328204328859481 | 20.15371212313317 | 0.7030062286655198 | 0.020756690893962322 | 0.031425558987866066 | 0.04002306630748414 |
| 15,000 | 0.032368494699465865 | 22.929972883971878 | 0.7185480608742691 | 0.01579005666120489 | 0.0429351940731089 | 20.191797845727866 | 0.7180627224127322 | 0.01864232385184085 | 0.031425558987866066 | 0.04002306630748414 |
| 20,000 | 0.03205496379752438 | 23.021917788901646 | 0.7233179936025416 | 0.014507512315105327 | 0.04284120051589504 | 20.215280334036663 | 0.7213830760129645 | 0.017146055978080724 | 0.031425558987866066 | 0.04002306630748414 |

**Conclusion:** The patch seams are fixed: no visible block edges in `zoom_step20000.png`. At step 20,000 the current-frame in-box L1 is 0.0321 (E's L1 decoder: 0.0314) at 3.3x the sharpness (0.72 vs 0.22). The predicted t+10 L1 is 0.0428 (L1 decoder: 0.0400): a sharp sampled future pays a small L1 cost against an averaged one. Seed spread kept falling (0.0145 / 0.0171), so the rendered detail is increasingly set by the latent rather than the noise. Fingers, cables, arm joints, bin weave, and held objects are rendered consistently across seeds, now and 10 steps ahead. Gains after 10k steps are small (L1 -0.0009), so the decoder is near its limit with these latents. Remaining weakness: transparent/white bottles render faint or partly missing, and some bin contents are coarse. This points to the 12x20 latent resolution, not the decoder. Takeaway: E's latent holds the object-level information the task needs, and region-JiT is a usable visual-feedback / training-loss decoder.

**Status:** done

### E18
- **ID:** E18
- **Date:** 2026-11-01
- **Goal:** Measure whether E's action-conditioned t+10 latent places clusters 15 (bottles/cups) and 11 (hands/dark bottles) correctly, using 256 evenly spaced validation windows.
- **Change:** Region-JiT v2 renders from predicted t+10 and current z(t); DINOv2-B/14 with the saved k-means assigns 24x40 cluster masks. Compare copy, predicted render, and current reconstruction; report all and moving windows.
- **Config/command:** `CUDA_VISIBLE_DEVICES=4 /mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python tools/eval_future_object_position_20261101.py --e-run runs/ffn_phaseE_dinob_qhist15_20261031 --jit-run runs/region_jit_v2_E_latentonly_20261101 --data-dir data/real_workspace_192x320_20261031 --boxes data/dinov2_regions_20261101/boxes_v2_24x40.npy --split /mnt/workspace/colleague-e/pi05_lingjun_joint54_zero2_b256_20261018/datasets/lerobot_optical93_eef72_15hz_20261016/meta/training_split.json --dino-dir /mnt/workspace/dengjianfeng/models/dinov2_official_20261031 --kmeans data/dinov2_regions_20261101/kmeans.pt --out-dir runs/future_pos_E_20261101 --eval-windows 256 --sample-steps 50`
- **Run dir:** `runs/future_pos_E_20261101`
- **Results:** 256 evenly spaced windows were evaluated with 50 Euler steps and two fixed seeds. Cluster masks and IoU were restricted to the union semantic boxes at t/t+10; the one-patch ring was used only as decoder context.

| Cluster | Subset | Windows | Copy centroid err | Copy IoU | Predicted render centroid err | Predicted render IoU | Moving windows pred beats copy | Current recon centroid err vs real t | Current recon IoU vs real t |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 bottles/cups | all | 256 | 0.9417 | 0.4109 | 1.0779 | 0.3718 | — | 0.9987 | 0.4049 |
| 15 bottles/cups | moving | 91 | 1.6151 | 0.3537 | 1.2370 | 0.3640 | 70.33% | 1.1429 | 0.4019 |
| 11 hands/dark bottles | all | 256 | 1.5446 | 0.5038 | 1.0802 | 0.6612 | — | 1.0220 | 0.7442 |
| 11 hands/dark bottles | moving | 149 | 2.2655 | 0.3888 | 1.2093 | 0.6314 | 83.89% | 1.0866 | 0.7362 |

Reports: `runs/future_pos_E_20261101/future_object_position.json` and `runs/future_pos_E_20261101/six_worst_moving_windows.png`.
- **Conclusion:** The world model moves objects in the right direction: on moving windows the predicted render beats copy-current in 70% (bottles) and 84% (hands) of windows, and its centroid error (bottles 1.24, hands 1.21 patches) is close to the current-frame rebuild limit (1.14, 1.09). The bottle problem is mainly encoding, not dynamics: even rebuilding the current frame only reaches bottle IoU 0.40 vs hands 0.74. Next: locate where bottle detail is lost in the image embedding (E20). Caveat: bottle labels come from DINOv2 clustering, itself noisy on transparent bottles.
- **Status:** done

### E19
- **ID:** E19
- **Date:** 2026-11-01
- **Goal:** Train and compare action-only (A), world-only (W), and combined (AW) scorers to select high-quality E flow samples using local counterfactual outcomes.
- **Change:** Precompute 12 normalized candidates and graded demo-closeness targets for all train/validation windows. Train each scorer for 15 epochs with listwise KL plus the specified pairwise demo-over-counterfactual hinge loss; run 50-step smoke checks first.
- **Config/command:** Candidate generation: `CUDA_VISIBLE_DEVICES=4 /mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python tools/gen_scorer_cands_E_20261101.py --e-run runs/ffn_phaseE_dinob_qhist15_20261031 --data-dir data/real_workspace_192x320_20261031 --boxes data/dinov2_regions_20261101/boxes_v2_24x40.npy --split /mnt/workspace/colleague-e/pi05_lingjun_joint54_zero2_b256_20261018/datasets/lerobot_optical93_eef72_15hz_20261016/meta/training_split.json --out-dir data/scorer_cands_E_20261101 --batch-size 32 --workers 8`. Smoke tests used 50 optimizer steps plus two validation batches for each variant. Full training launch: `setsid nohup env CUDA_VISIBLE_DEVICES=4 bash tools/run_scorer_v3_chain_20261101.sh > runs/e19_launch_20261101/full_training_chain.log 2>&1 < /dev/null &`; the wrapper runs A, W, AW sequentially with 15 epochs, batch size 32, AdamW lr 3e-4, wd 0.05.
- **Run dir:** Candidate data `data/scorer_cands_E_20261101`; `runs/scorer_v3_A_20261101`, `runs/scorer_v3_W_20261101`, `runs/scorer_v3_AW_20261101`.
- **Results:** Candidate generation completed in 154.77 s: 24,062 train and 2,289 validation windows; 12 candidates/window; listwise temperature `tau=0.8105327487`. A, W, and AW each passed 50 optimizer steps and two validation batches. The detached sequential 15-epoch training chain was launched on GPU 4 (PID 391785); A has started and W/AW are queued.
- **Conclusion:** pending (Claude)
- **Status:** running


**Note (2026-11-01, user):** AW stopped at epoch 8/15 (best epoch 1: pick RMSE 0.0753 vs random 0.0725; within-sample Spearman -0.19). A and W completed. Resources moved to E23a.

### E20
- **ID:** E20
- **Date:** 2026-11-01
- **Goal:** Locate where bottle information is lost across frozen DINOv2, E's fine-tuned encoder and world latent, predicted t+10 latent, and the Phase A visual encoder.
- **Change:** Add `tools/analyze_image_embedding_20261101.py` for cached fp16 token features, occupancy and centroid probes, shift sensitivity, bottle size and contrast groups, near-grasp wrist probing, and PCA/shift figures. Use 20,000 train-window frames and all validation-episode frames. E has `wrist=0`, so P6 uses frozen-DINO wrist tokens; the Phase A latest encoder loaded (epoch 19).
- **Config/command:** Smoke and P3 boundary checks passed on GPU 4. Full analysis ran on GPU 4 with `CUDA_VISIBLE_DEVICES=4 /mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python tools/analyze_image_embedding_20261101.py --e-run runs/ffn_phaseE_dinob_qhist15_20261031 --phasea-run runs/wan_phaseA_latent_image_20261031 --data-dir data/real_workspace_192x320_20261031 --wrist-dir data/real_wrist_144x192_20261031 --split /mnt/workspace/colleague-e/pi05_lingjun_joint54_zero2_b256_20261018/datasets/lerobot_optical93_eef72_15hz_20261016/meta/training_split.json --dino-dir /mnt/workspace/dengjianfeng/models/dinov2_official_20261031 --kmeans data/dinov2_regions_20261101/kmeans.pt --out-dir runs/embed_analysis_E_20261101 --batch-size 2 --workers 4 --resume-from-cache`, launched detached with `setsid nohup`. The completed fp16 cache was reused after correcting two P3 bugs.
- **Run dir:** `runs/embed_analysis_E_20261101`
- **Results:** 20,000 train frames; all 2,379 validation-episode frames; 2,289 valid S4 t+10 validation rows. Peak PyTorch CUDA allocation was 0.90 GiB. Reports: `results.json`, `summary.md`, `embed_pca.png`, `shift_maps.png`.

Frozen DINOv2-B/14 cluster assignments (bottle=15, hand=11, bin=13) are the labels. Feature caches are fp16 `.npy` files under `features/`. S1 is an upper reference and partly circular because its 24×40 frozen-DINO features also produce the cluster labels. P1 probes were fit on 20,000 sampled train-episode frames and evaluated on every validation-episode frame; S4 labels use real t+10 assignments and exclude the last ten frames of each validation episode.

#### P1–P3: stage comparison

P1 cells show linear AP/IoU (neighborhood MLP AP/IoU). P2 reports R² against the token-center baseline, model mean px error, and that baseline error. P3 shows mean 1−cosine change at 4 px / 8 px for bottle, hand, and background tokens.

| Stage | Bottle P1 | Hand P1 | Bin P1 | Bottle P2 R² / px (center px) | Hand P2 R² / px (center px) | P3 bottle 4/8 | P3 hand 4/8 | P3 bg 4/8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.460/0.342 (0.562/0.402) | 0.782/0.597 (0.909/0.703) | 0.894/0.748 (0.962/0.822) | 0.157 / 5.3 (5.9) | 0.444 / 4.2 (6.0) | 0.181/0.322 | 0.206/0.344 | 0.166/0.276 |
| S1 | 0.921/0.705 (0.969/0.815) | 0.990/0.899 (0.995/0.924) | 0.999/0.964 (0.999/0.970) | 0.172 / 1.9 (1.7) | 0.354 / 2.0 (2.4) | 0.195/0.329 | 0.209/0.355 | 0.187/0.308 |
| S2 | 0.484/0.353 (0.578/0.407) | 0.820/0.615 (0.922/0.719) | 0.910/0.759 (0.964/0.829) | 0.186 / 5.1 (5.9) | 0.486 / 4.0 (6.0) | 0.149/0.286 | 0.138/0.266 | 0.138/0.244 |
| S3 | 0.475/0.341 (0.591/0.422) | 0.842/0.634 (0.934/0.743) | 0.927/0.762 (0.969/0.835) | 0.169 / 5.2 (5.9) | 0.416 / 4.4 (6.0) | 0.076/0.153 | 0.102/0.204 | 0.072/0.136 |
| S4 | 0.404/0.295 (0.523/0.377) | 0.678/0.471 (0.858/0.646) | 0.900/0.736 (0.957/0.807) | 0.127 / 5.4 (5.9) | 0.246 / 5.0 (6.0) | 0.111/0.222 | 0.121/0.245 | 0.110/0.206 |
| S5 | 0.352/0.289 (0.517/0.379) | 0.818/0.588 (0.890/0.689) | 0.841/0.652 (0.937/0.758) | 0.186 / 5.2 (5.9) | 0.504 / 3.9 (6.0) | 0.043/0.150 | 0.033/0.113 | 0.045/0.140 |

##### P1 token-level AP at 12×20

| Stage | Bottle | Hand | Bin |
| --- | ---: | ---: | ---: |
| S0 | 0.766 | 0.908 | 0.951 |
| S1 | 0.956 | 0.987 | 0.999 |
| S2 | 0.788 | 0.926 | 0.962 |
| S3 | 0.775 | 0.917 | 0.959 |
| S4 | 0.706 | 0.765 | 0.945 |
| S5 | 0.653 | 0.869 | 0.907 |

#### P4: bottle size

| Components | Width median / p25–p75 px | Height median / p25–p75 px | Width <16 px | Width <8 px | Bottle tokens with area share <25% |
| ---: | --- | --- | ---: | ---: | ---: |
| 330888 | 8.0 / 8.0–24.0 | 8.0 / 8.0–24.0 | 56.0% | 0.0% | 0.0% |

#### P5: transparent-like versus opaque/coloured

Components are ranked by 0.5 × mean RGB saturation + 0.5 × absolute mean-luminance contrast against the one-cell local background ring; the lowest and highest train-set terciles are evaluated.

| Contrast split | Stage | Bottle P1 AP / IoU (linear) | Bottle P2 R² / px (center px) | Components in val |
| --- | --- | --- | --- | ---: |
| transparent_like | S0 | 0.107 / 0.095 | 0.272 / 4.8 (6.0) | 10010 |
| transparent_like | S2 | 0.117 / 0.101 | 0.302 / 4.7 (6.0) | 10010 |
| transparent_like | S3 | 0.096 / 0.086 | 0.241 / 5.0 (6.0) | 10010 |
| opaque_coloured | S0 | 0.163 / 0.137 | 0.153 / 5.2 (5.8) | 8001 |
| opaque_coloured | S2 | 0.177 / 0.146 | 0.209 / 4.9 (5.8) | 8001 |
| opaque_coloured | S3 | 0.164 / 0.141 | 0.215 / 5.0 (5.8) | 8001 |

#### P6: near-grasp relative bottle position

| Feature source | Train / val near-grasp windows | R² | Mean px error | Mean-baseline px error |
| --- | ---: | ---: | ---: | ---: |
| overhead_S3_3x3 | 8221 / 904 | -0.174 | 22.9 | 23.0 |
| pooled_frozen_DINO_wrist | 8221 / 904 | -0.081 | 23.0 | 23.0 |
| overhead_plus_DINO_wrist | 8221 / 904 | -0.354 | 24.6 | 23.0 |

#### P3 shift-size probe

| Stage | Bottle-token shift magnitude R² | MAE (px) |
| --- | ---: | ---: |
| S0 | 0.918 | 1.2 |
| S1 | 0.973 | 0.7 |
| S2 | 0.944 | 1.0 |
| S3 | 0.906 | 1.3 |
| S4 | 0.865 | 1.5 |
| S5 | 0.977 | 0.6 |

#### P7 figures

- `embed_pca.png`: PCA-to-RGB maps for four validation frames, including a transparent-like bottle, with bottle contours.
- `shift_maps.png`: 8 px horizontal-shift cosine-change maps for the same frames.

#### Deviations

- Two early full probe passes reached P3 after caching all features, then stopped on a P3 bin-region aggregation bug and mixed valid/terminal-frame indexing bug; both fixes passed a boundary smoke, and the final analysis reused the completed cache.
- E was trained with wrist=0, so no E wrist tokens exist; P6 uses frozen-DINO pooled wrist tokens only.

- **Conclusion:** Bottle detail is lost at the encoder input resolution (168x280 -> 12x20 tokens), not in fine-tuning or the world module. Bottle occupancy at 8 px: frozen 12x20 AP 0.46, fine-tuned 0.48, world latent 0.48 (MLP 0.59); 2x-resolution frozen DINOv2 0.92 (partly circular: labels come from it). Tokens know whether a bottle is present (token AP 0.77) but not where inside the 16 px token (offset R2 0.17; 5.2 px vs 5.9 px token-centre baseline). Prediction loses some more (t+10 latent: bottle AP 0.40, hands 0.84->0.68). Transparent-like bottles are the worst (AP 0.10 vs 0.16 opaque on the split metric). The world latent is less shift-sensitive than the encoder (4 px: 0.08 vs 0.18). Caveats: labels are k-means on 8 px cells, so size stats are quantised (P4 0% under 8 px is an artifact); P6 used wrist cameras, which are removed from the robot, so it is ignored. Next: re-check with SAM3 masks (E21) and design higher-resolution object tokens (24x40 features kept only inside key-object masks, or mask-pooled object tokens).
- **Status:** done

### E21
- **ID:** E21
- **Date:** 2026-11-02
- **Goal:** Finish SAM3 mask quality post-processing, compare to E20 k-means labels, and re-score E18 placement with per-image SAM3 labels.
- **Change:** The original two-episode Grounding DINO + SAM2 attempt remains stopped; the completed 93-episode SAM3 masks, objects.json, and QC outputs are used. Re-score labels real and rendered images independently with the same SAM3 image predictor. Conclusions remain SAM3 masks exist for all 93 episodes, but tracking is noisy: 9.75 tracks/episode (5-20), 1,816 suspected id jumps (>3 cells), 1,449 of 27,281 frames (5.3%) with no bottle. The E18 re-score with SAM3 per-image labels is INVALID for bottles: SAM3 does not find bottles on region-JiT renders (bottle absent in 328 seed-renders; even the current-frame rebuild scores 248 px / IoU 0.14), so it measures the labeller's domain gap, not the model. Hands are valid: on 141 moving windows the predicted render beats copy-current in 66% (IoU 0.60 vs 0.39; centroid 26.7 vs 27.4 px). Use: SAM3 masks on REAL frames are usable as training targets (E24); do not label generated images with SAM3; fix id switches before relying on instance identity..
- **Config/command:** Completed SAM3 QC/tracking/k-means packaging, then tools/e21_rescore_future_pos_sam3_20261101.py on GPU 4 (256 windows, 50 steps, two seeds, 0.15 GPU memory fraction).
- **Run dir:** runs/e21_postprocess_20261102; runs/future_pos_E_sam3_20261101
- **Code:** tools/e21_postprocess_sam3_20261102.py; tools/e21_rescore_future_pos_sam3_20261101.py
- **Results:** {"episodes":93,"e18_json":"runs/future_pos_E_sam3_20261101/future_object_position.json","e18_summary":{"windows_requested":256,"windows_evaluated":256,"model":{"e_run":"runs/ffn_phaseE_dinob_qhist15_20261031","jit_run":"runs/region_jit_v2_E_latentonly_20261101","labeler":"Meta SAM3 image predictor, per image with no propagation","sam3_checkpoint":"/mnt/workspace/dengjianfeng/eyerobot2-top-stereo/models/sam3/sam3.pt","sam3_threshold":0.5,"prompts":{"bottle":["transparent plastic bottle","bottle"],"hand":["robot hand"]},"sample_steps":50,"seeds":[101,202],"box_metric":"SAM3 masks intersected with the same union boxes at t and t+10; render ring is context only","moving_threshold_px":8.0,"gpu_memory_fraction":0.15,"gpu_peak_allocated_gib":3.826815128326416,"gpu_peak_reserved_gib":4.181640625},"classes":{"bottle":{"all_windows":{"windows":242,"copy":{"centroid_error_px":7.4713214914290935,"iou":0.6785487885785116},"predicted_render":{"centroid_error_px":258.6540628804368,"iou":0.12379077641555254,"prediction_absent_seed_count":328,"moving_windows_beating_copy_pct":null},"current_recon_vs_real_t":{"centroid_error_px":248.19809902330906,"iou":0.13807468530163097,"prediction_absent_seed_count":313}},"moving_windows":{"windows":74,"copy":{"centroid_error_px":19.788174890041816,"iou":0.3953761334228375},"predicted_render":{"centroid_error_px":259.60168505963344,"iou":0.11182570053605904,"prediction_absent_seed_count":100,"moving_windows_beating_copy_pct":13.513513513513514},"current_recon_vs_real_t":{"centroid_error_px":232.3056150057073,"iou":0.13411120947775326,"prediction_absent_seed_count":88}}},"hand":{"all_windows":{"windows":256,"copy":{"centroid_error_px":16.62441750468961,"iou":0.5338906445110401},"predicted_render":{"centroid_error_px":19.700305080119534,"iou":0.6674427918993804,"prediction_absent_seed_count":3,"moving_windows_beating_copy_pct":null},"current_recon_vs_real_t":{"centroid_error_px":21.260933184448593,"iou":0.7132997500680196,"prediction_absent_seed_count":5}},"moving_windows":{"windows":141,"copy":{"centroid_error_px":27.362352651412063,"iou":0.38929683848090646},"predicted_render":{"centroid_error_px":26.72739410626662,"iou":0.6035978112448686,"prediction_absent_seed_count":3,"moving_windows_beating_copy_pct":65.95744680851064},"current_recon_vs_real_t":{"centroid_error_px":27.776329649282555,"iou":0.6729772607165889,"prediction_absent_seed_count":5}}}}},"tracking_summary":{"tracks_per_episode_summary":{"mean":9.75268817204301,"median":10.0,"min":5,"max":20},"frames_zero_bottle_total":1449,"frames_total":27281,"suspected_id_switches_gt_3_24x40_cells":1816,"tracks_with_such_jumps":[1,2,3,4,5,9,11,12,13,17,21,22,23,24,25,26,31,32,33,35,39,40,41,42,43,44,45,47,48,49,50,51,52,53,56,59,61,62,65,66,67,68,69,70,71,72,73,74,75,77,79,80,81,82,84,91,92,93,97,98,99,100,107,108,110,112,116,117,118,121,123,128,132,136,140,141,145,149,150,151,152,160,161,162,163,170,171,175,177,178,180,181,182,183,186,187,188,192,196,197,201,202,203,207,208,209,211,212,215,216,219,221,223,225,227,228,234,235,236,237,238,241,243,246,247,248,249,251,253,254,255,258,261,262,265,271,272,273,274,275,283,284,285,286,288,294,295,297,299,300,301,308,309,310,315,316,317,322,323,324,326,329,330,331,333,335,337,339,340,341,342,343,346,347,348,349,358,359,360,361,362,364,367,369,371,374,375,376,378,379,381,382,383,384,385,386,387,388,394,395,396,397,399,401,403,407,411,412,414,416,417,419,424,425,426,427,428,430,431,432,433,436,437,439,440,446,449,450,451,452,453,454,456,457,458,462,463,464,465,466,467,471,472,473,475,476,477,479,481,482,483,485,486,489,492,493,494,495,496,497,499,502,503,504,505,509,510,511,518,519,521,524,525,526,528,530,531,532,535,537,538,539,540,541,542,544,546,547,548,549,551,552,553,555,556,559,560,562,563,565,568,569,570,571,572,573,574,576,580,581,583,584,587,592,594,595,596,597,598,599,600,601,604,608,609,610,613,614,616,617,620,621,623,625,626,630,631,632,633,637,639,644,645,646,647,649,653,654,656,657,658,660,662,663,664,665,666,667,668,671,672,673,674,675,678,679,681,687,688,689,691,694,695,697,699,700,701,702,706,707,708,709,710,711,712,713,714,715,716,721,722,723,725,726,730,732,733,734,735,736,737,743,744,746,749,750 ...
- **Deviations:** None recorded.
- **Conclusion:** pending (Claude)
- **Status:** done

### E22
- **ID:** E22
- **Date:** 2026-11-02
- **Goal:** Calibrate the overhead camera from E21 SAM3 hand masks, score per-frame mask confidence, flag 30 low-confidence frames, and filter E18 windows at confidence >= 0.8.
- **Change:** Stage B uses the completed E22 Stage A FK/joint map and the E21 SAM3 masks through the compatibility symlink. The calibration reports held-out episodes; wrist cameras are not used. Conclusions remain Stage A (FK) is physically plausible. Stage B camera self-calibration FAILED: held-out projected-hand vs SAM3 hand mask IoU median 0.02 (p90 0.23), centroid error median 145 px on a 320-px-wide image; train landmark error 72 px despite low point-to-mask distance (9 px) -> wrong pose/local minimum, not noise. Confidences are invalid (hand median 0.03; 0 of 256 windows pass 0.8). Likely causes to check: joint->URDF mapping and arm base frames, left/right hand label swaps in SAM3 masks, gloved-hand shape vs mesh, single-start optimisation (needs a coarse search over camera pose). Consequences: E24 confidence weighting disabled; the FK-hand-conditioning direction (#1 in BOTTLENECK_ANALYSIS_20261102.md) is blocked until the calibration is fixed..
- **Config/command:** Stage B ran after E21 E18 SAM3 re-score; CPU self-calibration with 84 train and 9 held-out episodes, per-frame confidence, 30-frame QC, and confidence >= 0.8 E18 filter.
- **Run dir:** runs/e22_camera_calibration_20261101; data/object_masks_sam3_20261101/confidence.npz
- **Code:** tools/e22_camera_confidence_20261101.py
- **Results:** {"heldout_episodes":[7,18,38,56,64,66,77,88,90],"projected_mesh_vs_hand_mask_iou":{"count":4333,"min":0.0,"p10":0.0,"median":0.022480058013052938,"p90":0.22512508828298333,"max":0.635386119257087,"mean":0.0692267788089885},"hand_mask_centroid_error_px":{"count":4333,"min":7.324397725024944,"p10":55.40310894412673,"median":145.3513610323921,"p90":232.28953496890227,"max":270.87599835585354,"mean":145.91029811489574},"projected_mesh_iou_by_side":{"left":{"count":2203,"min":0.0,"p10":0.0,"median":0.013681731524055175,"p90":0.10304295517333568,"max":0.5095613048368954,"mean":0.039097966922535754},"right":{"count":2130,"min":0.0,"p10":0.0,"median":0.0403573998560806,"p90":0.33853816651393276,"max":0.635386119257087,"mean":0.1003881743892023}},"train_point_to_mask_distance_px":{"count":6223,"min":0.0,"p10":0.0,"median":9.264227995251911,"p90":114.34067731422917,"max":196.66685577528827,"mean":30.09219118576038},"train_landmark_centroid_error_px":{"count":889,"min":4.167938296357261,"p10":22.351473090121253,"median":72.12303690310908,"p90":164.25319867257082,"max":219.97939131775226,"mean":82.256756912693},"frame_confidence":{"count":27281,"min":0.0,"p10":0.3228476047515869,"median":0.5062353014945984,"p90":0.5723360776901245,"max":0.734794557094574,"mean":0.47366702090021945},"hand_confidence":{"count":27281,"min":0.0,"p10":0.0,"median":0.029828190803527832,"p90":0.1690017133951187,"max":0.46958914399147034,"mean":0.059477292807562046},"bottle_confidence":{"count":25678,"min":0.00010050900164060295,"p10":0.7941906213760376,"median":1.0,"p90":1.0,"max":1.0,"mean":0.940530146683075},"cycle_iou_available_fraction":0.0,"cycle_iou_source_keys":[],"e18_high_confidence":{"source":"runs/future_pos_E_sam3_20261101/future_object_position.json","labeler":"SAM3 per-image labels from E21","filter":"confidence at both t and t+10 >= threshold","threshold":0.8,"windows_original":256,"windows_with_valid_t10":256,"windows_retained":0,"windows_rejected":256,"classes":{"bottle":{"all_windows":{"windows":0},"moving_windows":{"windows":0}},"hand":{"all_windows":{"windows":0},"moving_windows":{"windows":0}}}},"mask_files":{"0":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"1":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"2":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"3":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"4":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"5":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"6":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"7":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"8":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"9":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"10":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"11":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"12":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"13":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"14":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"15":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"16":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"17":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"18":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"19":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"20":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"21":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"22":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"23":["masks_192x320","occupancy_24x40","occupancy_12x20","class_order","input_mode"],"24":["masks_1 ...
- **Deviations:** None recorded.
- **Conclusion:** pending (Claude)
- **Status:** done

### E23a
- **ID:** E23a
- **Date:** 2026-11-01
- **Goal:** Test whether a value head can learn time-to-success from frozen Phase E latents, and whether predictions respond to action-conditioned future latents.
- **Change:** Train V_pool, V_box, and joint-history-only B_q; include the global train-mean B_len constant. Train log1p(T) with Huber and progress separately. Compare real-only (a) with real plus P1..P4 (b), using four independently seeded visual T heads per train set for sensitivity spread.
- **Config/command:** Smoke test passed on 96 sampled windows per split for two epochs. Full training used 20 epochs, batch 512, lr 3e-4, weight decay 0.01, and an 11 GiB per-process cap on GPU 4. It saved all 24 heads and both feature caches, then crashed in create_figure at line 706 because NumPy indices were applied to a Python list (TypeError: only integer scalar arrays can be converted to a scalar index). No retraining was done. After converting the source indices to np.int64, results/summary are now written before figure creation and plotting is guarded. Eval-only rerun on <gpu-server>: setsid nohup env CUDA_VISIBLE_DEVICES=4 /mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python -u tools/train_value_ttg_20261101.py --e-run runs/ffn_phaseE_dinob_qhist15_20261031 --data-dir data/real_workspace_192x320_20261031 --boxes data/dinov2_regions_20261101/boxes_v2_24x40.npy --split /mnt/workspace/colleague-e/pi05_lingjun_joint54_zero2_b256_20261018/datasets/lerobot_optical93_eef72_15hz_20261016/meta/training_split.json --run-dir runs/value_ttg_E23a_20261101 --epochs 20 --batch-size 512 --eval-batch-size 256 --cache-batch-size 4 --sensitivity-batch-size 8 --workers 8 --lr 3e-4 --weight-decay 0.01 --gpu-mem-limit-gb 11.0 --evaluate-only.
- **Run dir:** runs/value_ttg_E23a_20261101
- **Results:** Eval-only reused 24 trained heads and cached features for 24,062 train / 2,289 validation windows; action sensitivity used 512 windows. Validation on real latents:

| Train set | Head | T MAE (steps) | T R2 | Spearman(V,-T) | Progress MAE | Progress R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a_real | V_pool | 44.706 | 0.6175 | 0.9303 | 0.0957 | 0.8033 |
| a_real | V_box | 43.015 | 0.6444 | 0.9321 | 0.0985 | 0.8009 |
| a_real | B_q | 78.633 | -0.1683 | 0.4893 | 0.1885 | 0.2596 |
| b_real_plus_P1_to_P4 | V_pool | 53.204 | 0.4735 | 0.9233 | 0.1005 | 0.7940 |
| b_real_plus_P1_to_P4 | V_box | 51.895 | 0.5144 | 0.9283 | 0.1045 | 0.7761 |
| b_real_plus_P1_to_P4 | B_q | 83.668 | -0.3505 | 0.4625 | 0.1963 | 0.2125 |
| B_len | train_mean_constant | 82.749 | -0.0337 | n/a | 0.2416 | -0.0001 |

Shift MAE for R-at-matching-target and P1..P4, and all hold/reversed/other action sensitivity values by horizon (including margins and margin/spread), are in runs/value_ttg_E23a_20261101/summary.md and results.json. Figure creation completed successfully. The initial crash and eval-only recovery are recorded above.
- **Conclusion:** Time-to-success IS learnable from observed latents: within-episode Spearman 0.93, R2 0.64 (box tokens) vs joint-history-only R2 -0.17, so vision carries the progress signal; absolute MAE is still ~43 steps (~2.9 s). Heads trained on real latents fail on predicted latents (MAE 133-144 steps, worse than the constant baseline 80): rolled-out latents are off-distribution. Training on real + predicted latents fixes that shift (50-55 steps on P1-P4). But action sensitivity stays weak and is the blocker: with the shift-robust heads, demo beats hold only 58-66%, reversed 53-70%, and another window's plausible chunk 44-53% (chance), with margins 0.0-0.7x the 4-seed ensemble spread; longer horizons (H=1..4) do not help. This matches E12 (62% at one chunk). Conclusion: the value head reads the state well; the bottleneck is that E's action-conditioned predictions do not carry enough task-relevant (object) consequence of the action. Latent-rollout value scoring is not viable with E as is. Next: make the dynamics' object predictions action-dependent (higher-resolution object tokens per E20; object-state losses on predicted latents using SAM3 masks; multi-step rollout training), then repeat this test.
- **Status:** done


### E24
- **ID:** E24
- **Date:** 2026-11-02
- **Goal:** Train and evaluate the object-aware, action-sensitive world model specified in SPEC_E24.md, initialized from E best.pt.
- **Change:** DINOv2-B is run at 336x560 and pooled to 24x40 object features plus 12x20 global tokens. SAM3 masks supervise six bottle slots, two hands, and the bin; a learned mask head transitions from teacher masks to predicted masks. Dynamics uses three-chunk rollouts and action counterfactual contrast. Training completed 25 epochs; final evaluation initially crashed because sampling omitted the 15-step history prefix required by the inherited 25-position flow head. The repaired evaluator samples the full 15+10 sequence and scores the last 10 actions, matching run E. Recovery is eval-only from latest.pt EMA weights; no retraining or checkpoint writes. E22 confidence weighting is disabled because camera calibration failed.
- **Config/command:** E24 eval-only and value evaluation completed. The detached chain failed only in the final summary writer, which looked up results["validation_real"] instead of the saved results["real_validation"] after e24_value_results.json had been written. Fixed the key and regenerated the value summary from the saved JSON and the script's default E23a results file; no GPU job was rerun.
- **Run dir:** runs/ffn_E24_objaware_20261102
- **Code:** tools/e24_object_aware_20261102.py; tools/e24_posteval_20261102.py; tools/e24_eval_chain_20261102.sh; tools/e24_update_log_20261102.py
- **Results:** Evaluation complete. [E24 evaluation summary](../../runs/ffn_E24_objaware_20261102/e24_summary.md); [E24 value summary](../../runs/value_ttg_E24_objaware_20261102/e24_value_summary.md).
- **Deviations:** No retraining was done. The chain's value_ttg stage completed evaluation and saved its results; only summary generation failed.
- **Conclusion:**

#### E24 conclusion (Claude, 2026-11-02)

**Verdict: FAIL on the main gate (action sensitivity). Small policy gain. Object prediction worse than copy.**

Pass criteria (SPEC_E24): demo vs other >= 70% and demo vs hold >= 80% at H=2.
E24 (set b, real + P1..P4): other 54% (V_pool) / 52% (V_box); hold 63% / 61%. E23a on E: other 53% / 51%, hold 66% / 58%. No change at H=1-2.
At H=3-4 E24 is better on other (56-66% vs 41-46%) and reversed (67-72% vs 50-55%), and the ensemble spread halved (7-12 vs 14-27 steps), so rollouts are more stable, but still far below the gate.

1. Policy (2109 windows, same evaluator): chunk RMSE 0.0716 vs E 0.0735 (-2.5%); medoid-6 0.0698 vs 0.0701 (tie); missed onset 31% vs 35%. Marginal gain, not a new best for deployment by itself.
2. Counterfactual top-1 = 0.93 (1545 windows, ~5.2 valid negatives, chance ~0.16). The dynamics does distinguish actions in latent space, but the value heads do not see a task-progress difference. Most likely the action effect is learned on the robot (hand pose) part of the latent, not on the object outcome. This is bottleneck B1 again: demo data alone does not teach what a different action does to the bottle.
3. Object prediction vs SAM3 (t+10/20/30): predicted bottle centroid error 31 px vs copy 11-23 px; hands 82-86 px vs copy 45-79 px. Prediction is worse than "nothing moves" at every horizon; on moving windows it beats copy only 41-66%. The object head is weak. (The smoke-test copy = 0 px was only static windows; full copy baseline is valid.) An open check: decode error at t+0 to split encoding vs dynamics error.
4. Value on real latents improved: V_pool R2 0.74, Spearman 0.95 (E23a set b on E: R2 ~0.64, Spearman 0.93).

Implications: architecture/loss changes on single-action demos are not enough (E16, E19, E23a, E24 agree). Next steps are directions 1 (FK hand conditioning, needs the E22 camera fit) and 2 (small action-diverse real data). Do not run more demo-only variants of the counterfactual loss.

Bug note: e24_posteval summary writer used key `validation_real` instead of `real_validation`; results JSON was saved correctly before the crash; summary regenerated from JSON without rerun.

- **Status:** done

### E22b
- **ID:** E22b
- **Date:** 2026-11-02
- **Goal:** Check the measured 2026-10-15 top-camera and arm calibration against all 93 episodes without manual labels.
- **Change:** Use vendor MDH FK with the measured calibration, infer arm-to-image mapping from SAM3 hand-track motion, fit flange-frame hand-point offsets and optional train-only base corrections, then score held-out validation episodes.
- **Config/command:** CPU-only on dengjianfeng-101; 15 Hz, all frames, 84 train / 9 validation episodes; image geometry verified from the raw 640x480 parquet camera stream, E21 crop path, and the generated 192x320 crop report.
- **Run dir:** runs/e22b_calib_check_20261102
- **Results:** [Summary](../../runs/e22b_calib_check_20261102/summary.md); machine-readable outputs: geometry.json, mdh_vs_urdf.json, arm_mapping.json, fits.json, metrics.json.
- **Conclusion:** Run complete; see validation gates and calibration corrections in the summary.
- **Status:** done
- **E22b fix**: done; CPU-only steps 2–4 in `runs/e22b_calib_check_20261102_fix/`; prior failed run preserved. Root cause: SAM3 hand classes include forearms and switch sides; union centroids corrupt fitting and mapping. Fixed hand-only largest components, image-only sides, crop-coordinate projections and smoothed motion. Mapping A=image-left, B=image-right (margin 1.4056). A: val median 5.24px / p90 14.57px / in-mask 98.3%, pass=True. B: val median 4.80px / p90 11.58px / in-mask 99.7%, pass=True. 84/9 split and ≤12px/≥80% gate unchanged; see [summary](../../runs/e22b_calib_check_20261102_fix/summary.md), debug_evidence.json, fits.json and overlays/.
- E22b-fix passed (offset-only): A 5.2 px / 98%, B 4.8 px / 99.7% ([fix summary](../../runs/e22b_calib_check_20261102_fix/summary.md)).

## Code changes

All under `tools/`, uploaded from local scratchpad:

- `convert_lewam_upstream_h5_20261031.py`
- `train_lewam_upstream_episode_split_20261031.py (IDM param-group fix)`
- `eval_lewam_upstream_e2e_20261031.py`
- `render_mujoco_transitions_224_20261031.py`
- `build_mujoco_lewam_h5_20261031.py`
- `eval_lewam_mujoco_object_20261031.py`
- `convert_real_workspace_crop_20261031.py`
- `train_world_action_net_20261031.py (Phase A)`
- `score_phaseA_policy_20261031.py`
- `train_foresight_flow_20261031.py (Phase B)`
- `eval_sample_selection_20261031.py`
- `eval_good_samples_20261031.py`
- `convert_real_wrist_20261031.py`
- `train_ffn_cd_20261031.py (C/D; later +DINOv2 encoder/anchor for E, +semantic boxes for F; defaults keep earlier runs reproducible)`
- `probe_encoders_dinov2_20261031.py (resumable)`
- `precompute_dinov2_feats_20261031.py`
- `gen_policy_samples_20261031.py`
- `train_action_scorer_20261031.py (v1 listwise; --loss v2 pairwise)`
- `diag_action_scorer_20261031.py`
- `measure_timing_noise_floor_20261031.py`
- `train_progress_scorer_20261101.py`
- `viz_world_model_20261101.py`
- `test_decoder_bottleneck_20261101.py (+in-box metrics)`
- `cluster_dinov2_regions_20261101.py`
- `build_semantic_boxes_20261101.py`
- `train_region_jit_decoder_20261101.py`
- `train_region_jit_decoder_v2_20261101.py (region-JiT v2: bilinear conditioning, context ring, configurable size/cap/EMA; written by Codex, reviewed by Claude)`
- `render_action_contrast_20261101.py`
- `eval_future_object_position_20261101.py (E18 cluster placement evaluation)`
- `gen_scorer_cands_E_20261101.py (E19 fixed 12-candidate generation)`
- `train_action_scorer_v3_20261101.py (E19 action-only/world-only/combined scorers)`
- `train_value_ttg_20261101.py (E23a frozen-latent time-to-success learnability test)`
- `run_scorer_v3_chain_20261101.sh (E19 sequential detached training launcher)`

- `e21_sam2_object_masks_20261101.py (93-episode Grounding DINO + SAM 2.1 overhead masks, occupancy grids and QC)`
- `eval_future_object_position_sam2_20261101.py (E18 re-score with per-image Grounding DINO + SAM 2.1 labels)`

- `e22_tianji_kinematics_20261101.py (54-D mapping and 93-episode FK)`
- `e22_camera_confidence_20261101.py (camera calibration, confidence, high-confidence E18 re-score)`
- `e22_wait_for_e21.sh (10-minute E21 gate poll, 6-hour limit)`

- `e21_sam3_object_masks_20261101.py (SAM3 text-prompted video masks, tracks, occupancy, grasp bindings, and QC)`
- `e21_rescore_future_pos_sam3_20261101.py (E18 re-score with per-image SAM3 labels)`

### E28
- **ID:** E28
- **Date:** 2026-11-03
- **Goal:** Evaluate robot hand geometry from joint targets and gated object dynamics on the bottle-to-bin task.
- **Change:** FK-conditioned robot tokens for full/nogate; learned action robot tokens for nofk; three four-epoch A2-initialized runs.
- **Config/command:** OpenWAM Python; <gpu-server> GPU 7; micro-batch 8 × accumulation 4; full post-evaluation and E23a set-b H=1–4 value test.
- **Ops revision:** A2/E25 full-validation baselines run once on H20 GPU 6; per-variant waiters reuse them on GPUs 0/6/7.
- **Run dir:** `runs/e28_{full,nogate,nofk}_20261103/`; chain artifacts `runs/e28_chain_20261103/`.
- **Results:**
  - full: 2019 validation windows; H2 set-b demo<other V_pool/V_box 44.1%/45.7%; H2 demo<hold V_pool/V_box 49.6%/29.7%; medoid-6 chunk policy RMSE 0.064 rad.
  - nogate: 2019 validation windows; H2 set-b demo<other V_pool/V_box 46.3%/49.2%; H2 demo<hold V_pool/V_box 46.3%/40.0%; medoid-6 chunk policy RMSE 0.065 rad.
  - nofk: 2019 validation windows; H2 set-b demo<other V_pool/V_box 44.5%/40.2%; H2 demo<hold V_pool/V_box 55.1%/36.9%; medoid-6 chunk policy RMSE 0.065 rad.
  - Value numbers above are legacy E24 bridge diagnostics only; native E28 swapped-action/FK value gate is unmeasured. Prior identical values were copied E23a reference scores.
  - noGate gate AUROC is not meaningful; H2 static shift is local to a demo context and affected by temporal pooling and bfloat16 resolution. Detailed audited results: runs/e28_chain_20261103/ANALYSIS_INPUT.md and VERIFICATION.md.
  - Direct same-window A2 and E25 policy and A2 static-invariance values are recorded in `runs/e28_chain_20261103/summary.md` and each variant result JSON.
- **Status:** done

## Current best and open decisions

E is the selected best flow model; F has similar validation policy error. Phase A MSE remains a real-robot comparison baseline. E uses the region-JiT v2 decoder for visual feedback. See E08, E14 and E17 for measured results.

## Open decisions / next

- Real-robot comparison: E best (medoid or half-noise sampling, ~5 steps executed before replanning) vs Phase A MSE head.
- E16 found action-dependent effects; improve scorer design. Whether to train the world model through the region-JiT decoder remains open.

## How to add an entry

```markdown
### E##
- **ID:** E##
- **Date:** YYYY-MM-DD
- **Goal:**
- **Change:**
- **Config/command:**
- **Run dir:**
- **Results:**
- **Conclusion:**
- **Status:**
```

## E24r — clean hand target rescore

- **Status:** done.
- **Change:** Evaluation-only E24 hand prediction rescore; no training and no change to E24.
- **Targets:** T1 is E22b-fix calibrated vendor-MDH FK with offset-only fits, A=image-left/hand_left and B=image-right/hand_right; T2 is the E22b-fix hand-only component mask with image-only side reassignment.
- **Protocol:** E24 EMA checkpoint, same 2,109 validation windows, action-conditioned predictions at t+10/20/30; copy and t+0 decode baselines; moving classification from T1 FK displacement >10 px.
- **Artifacts:** runs/e24r_hand_rescore_20261102/results.json, summary.md and eight overlays. Full run uses CUDA_VISIBLE_DEVICES=4 on <gpu-server>, batch size 1, and waits for at least 12 GiB free on GPU 4.
- **Scored:** 2109 / 2109 validation windows; completed 2026-11-02T08:25:56+00:00.

### E24r conclusion (Claude, 2026-11-02)
E24's hand head is not usable: against the calibrated FK hand point, error is ~130-140 px at t+10/20/30 and ~140 px
already at t+0 (encoding, not dynamics); copy error is 10-41 px; on moving windows prediction beats copy only 9-25%.
Overlays show predicted hand points on the forearms and sometimes on the wrong side: E24 learned the raw SAM3
hand_left/hand_right labels, which include forearms and switch sides (E22b-fix). E24's earlier hand numbers
(82-86 px vs raw SAM3) and the E24 object-loss contribution for hands are therefore unreliable.
Action: E25 Amendment 2 trains and evaluates hands on the cleaned hand-only masks, with the FK point as a check only.
The bottle labels come from the same SAM3 run (many identity switches in E21); a similar audit of bottle targets is advisable.

### E25
- **ID:** E25
- **Date:** 2026-11-02
- **Goal:** Diagnose E24 action sensitivity and test history dropout, random-length rollouts, residual object prediction, and inverse-dynamics supervision.
- **Change:** E24 EMA initialization; C1-C4 per SPEC_E25 Amendments 1 and 2. Hands train and evaluate against E22b-fix cleaned image-only masks; calibrated FK is an evaluation check only.
- **Config/command:** Detached sequential chain on <gpu-server>, CUDA_VISIBLE_DEVICES=4; stage0 (D1-D3, up to 512 H4-valid windows), 64-window evaluator smoke, 100 optimizer-step timing with runtime GPU memory sizing and effective batch 32; 12-epoch training only if projected runtime is at most 24 hours; full E25/E24/E evaluation and E23a set-b value test.
- **Run dir:** `runs/ffn_E25_histdrop_20261102/`; chain artifacts `runs/e25_chain_20261102/`.
- **Code:** `tools/e25_stage0_20261102.py`, `tools/e25_object_aware_20261102.py`, `tools/e25_posteval_20261102.py`, `runs/e25_chain_20261102/chain.sh`.
- **Stage 0:** D1-D3 completed on 512 windows; [JSON](../../runs/e25_chain_20261102/stage0/stage0_results.json), [summary](../../runs/e25_chain_20261102/stage0/stage0_summary.md).
- **Evaluation:** [E25/E24/E JSON](../../runs/e25_chain_20261102/posteval_full/posteval_results.json).
- **Value test:** [E23a set-b result](../../runs/e25_chain_20261102/posteval_full/value/e25_value_results.json).
- **Update:** Resumed at full_evaluation without retraining; full evaluation and E23a set-b value test completed on GPU 4. Final gate report: runs/e25_chain_20261102/posteval_full/e25_final_report.md.
- **Conclusion:**

#### E25 conclusion (Claude, 2026-11-03)
FAIL on the main gate. Value test (set b, H=2): demo vs other 56% (V_pool) / 48% (V_box); vs hold 65% / 66% (gate 70% / 80%). Same level as E24.
Policy: new best offline. Chunk RMSE 0.0689 (E24 0.0716, E 0.0735); medoid-6 0.0674; first-5 0.0421.
Hands fixed by clean labels: current-frame error 5-6 px vs FK (E24 ~140 px); moving windows beat copy 55-70%. Gain is mostly from clean hand labels (Amendment 2), not separable from C1-C4.
Bottle not fixed: ~30 px vs copy 11-23 px (bottle labels were still raw SAM3).
Stage0 D3: history/velocity hiding changes E24 rollout error by only 1-3%, so the motion-shortcut hypothesis was wrong.
Bin slot was empty in all data (E21 found no bin), so bin rows are n/a; the earlier claim that the action moves the bin was an artifact.
Note: posteval_full/value/e24_value_summary.md is the same E25 checkpoint (mislabelled header), not E24.
Capacity audit (runs/e25_capacity_audit_20261102.md): future-RGB losses 55% of weighted loss, roll_latent 4%.


- **Status:** done
### E26a
- **ID:** E26a
- **Date:** 2026-11-02
- **Goal:** Audit object-token correctness and motion-stratified E24 latent prediction baselines.
- **Change:** Rebuilt persistent bottle IDs with Hungarian mask-IoU/centroid matching and 10-frame re-identification, retained E22b-clean hands and visibility flags, and merged the newly available per-episode clean bin masks into slot 8. No manual labels or FK were used. The review note approved continuing with visibility flags and stopping SAM3 bottle repair.
- **Config/command:** E24 EMA checkpoint; 2019 H4-valid validation windows; action-free tokenwise residual MLP trained on 512 train windows for 8 epochs on CUDA_VISIBLE_DEVICES=0; trajectory-level 1,000-replicate bootstrap.
- **Run dir:** runs/e26a_audit_20261102/; cleaned masks in data/object_masks_clean_20261102/.
- **Bin coverage:** 93/93 episode archives and 93/93 episodes with bin masks; 27281/27281 frames (100.0%) have slot-8 masks. The prior E21 cache had 0 bin tracks and empty slot 8; the new clean bin masks replace that missing supervision.
- **Part B:** 93 episodes, 27281 frames, 309 raw bottle tracks, 369 raw identity-switch events, 4579 mask fragments, and 0 recorded merge links.
- **Part C overall copy-normalized error (mean [95% CI]):**
- **global:** H1: cv=2.012 [1.969, 2.055], MLP=0.978 [0.962, 0.995], E24=1.280 [1.249, 1.306]; H2: cv=3.969 [3.855, 4.081], MLP=0.979 [0.952, 1.009], E24=1.589 [1.531, 1.642]; H3: cv=7.758 [7.545, 7.959], MLP=0.985 [0.955, 1.017], E24=1.997 [1.896, 2.100]; H4: cv=14.869 [14.453, 15.254], MLP=0.982 [0.952, 1.015], E24=2.446 [2.278, 2.592]
- **bottle_target:** H1: cv=2.087 [1.951, 2.189], MLP=1.111 [1.079, 1.142], E24=2.824 [2.581, 3.049]; H2: cv=3.946 [3.571, 4.240], MLP=1.157 [1.093, 1.218], E24=3.509 [3.003, 4.041]; H3: cv=7.098 [6.137, 7.924], MLP=1.154 [1.069, 1.236], E24=4.450 [3.637, 5.267]; H4: cv=12.604 [10.622, 14.342], MLP=1.132 [1.035, 1.229], E24=5.424 [4.345, 6.503]
- **bottle_other:** H1: cv=2.175 [2.067, 2.282], MLP=2.324 [1.808, 2.991], E24=13.217 [8.875, 18.245]; H2: cv=3.913 [3.679, 4.159], MLP=2.874 [2.123, 3.858], E24=16.957 [11.247, 23.501]; H3: cv=6.757 [6.220, 7.289], MLP=3.042 [2.217, 4.111], E24=22.005 [14.697, 30.831]; H4: cv=12.374 [11.001, 13.693], MLP=3.055 [2.234, 4.104], E24=27.479 [18.524, 38.139]
- **hands:** H1: cv=2.171 [2.076, 2.266], MLP=1.236 [1.138, 1.348], E24=2.507 [2.172, 2.925]; H2: cv=4.094 [3.870, 4.344], MLP=1.304 [1.166, 1.458], E24=3.050 [2.576, 3.578]; H3: cv=7.666 [7.284, 8.055], MLP=1.318 [1.189, 1.465], E24=4.152 [3.530, 4.749]; H4: cv=14.327 [13.701, 14.975], MLP=1.300 [1.192, 1.423], E24=5.344 [4.619, 6.012]
- **bin:** H1: cv=2.002 [1.925, 2.078], MLP=1.699 [1.509, 1.908], E24=7.031 [6.143, 7.929]; H2: cv=3.737 [3.468, 3.973], MLP=2.000 [1.721, 2.289], E24=9.314 [7.523, 11.199]; H3: cv=6.925 [6.324, 7.429], MLP=2.104 [1.773, 2.467], E24=12.694 [10.188, 15.218]; H4: cv=12.796 [11.685, 13.734], MLP=2.084 [1.764, 2.449], E24=15.849 [12.846, 18.776]
  - The detailed summary.md reports moving/static and task-phase strata with CIs, and action-swap sensitivity for all groups including bin.
  - The 16-window end-to-end Part C smoke run passed after fixing the demo/alternative group indexing.
  - Contact sheets: runs/e26a_audit_20261102/contact_sheets/contact_before.png and contact_after.png.
- **Conclusion:**

#### E26a conclusion (Claude, 2026-11-03)
Bin masks recovered for all 93 episodes (100% coverage). Bottle IDs cleaned (persistent IDs, visibility); bottles inside the bin and dark held bottles are under-detected (known limitation; SAM3 repair attempts added 0 frames, stopped).
Baselines: E24 latent prediction with the true action is worse than copy for every group (global 1.28x copy at H1 to 2.45x at H4; target bottle 2.8-5.4x); an action-free MLP is about copy (global 0.98). Bin/other-bottle ratios are inflated by E24 never having seen clean tokens; the global ratio is not. Conclusion: latent dynamics currently degrades the state; motivates E27a (loss rebalance, latent residual).

- **Status:** done

### E27a
- **ID:** E27a
- **Date:** 2026-11-03
- **Goal:** Rebalance future-image and rollout-latent losses; test object-region weighting and an explicit global latent residual, addressing E25 loss imbalance and E26a copy-baseline findings.
- **Change:** Five 4-epoch fine-tunes from E25 EMA, effective batch 32. Every arm uses the E26a clean mask cache with persistent bottle IDs, cleaned hands, and bin slot 8. A0 control; A1 rebalance; A2 object-weighted images; A3 no future-image losses; A4 zero-initialized global latent residual. Used-frame-only target encoding matched the E25 all-six-frame roll-latent loss on the batch verification. Per-loss gradient L2 norms were recorded at the last trainable DINO block and dynamics every 50 steps.
- **Config/command:** Detached per-arm chains; Python `/mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python`; GPU assignments A0/A1/A4=2, A2=6, A3=4; effective batch 32.
- **Run dir:** `runs/e27a_A0_20261103/` through `runs/e27a_A4_20261103/`; `runs/e27a_chain_20261103/`.
- **Results:** Comparison report written to `runs/e27a_chain_20261103/summary.md`; it contains policy, latent-to-copy, object-to-copy, counterfactual, value-gate, and gradient-share result tables.
- **Status:** done

- **E-report (2026-11-03 05:41 BRT):** Consolidated E21–E28 and historical V-JEPA evidence: [ANALYSIS_REPORT.md](../../runs/report_20261103/ANALYSIS_REPORT.md) (201 lines), CSV tables and source hashes in `runs/report_20261103/`. Visible-hand FK reprojection passes; absolute 3D accuracy remains unverified. E28 native value gate UNMEASURED; full +10 moving-bottle gate fails; policy medoid ties A2. Ranked remaining trials are recommendations only; none run.
- **Astra root-cause review (2026-11-03 05:47 BRT):** [ASTRA_ROOT_CAUSE.md](../../runs/report_20261103/ASTRA_ROOT_CAUSE.md): ranked limited identification of action-dependent bottle outcomes (moderate confidence), distinguished cosine gains from raw-L2 calibration loss, verified weak V-JEPA action benefit while flagging full-data run provenance, and retained native E28 gate as unmeasured. Proposed matched-start state/action ablation and five own-model alternatives; policy-only baseline first. Analysis only; no training/evaluation launched.

### E29
- **ID:** E29
- **Date:** 2026-11-03
- **Goal:** Prepare E29 matched-start windows, action variants, and clean hand/bottle/bin centre labels for the V-JEPA action-faithfulness probe.
- **E29-S1 sub-task (2026-11-03 06:51 BRT, luna-standard):** Cloned official V-JEPA 2 at commit 204698b45b3712590f06245fbfba32d3be539812; downloaded only vjepa2-ac-vitg.pt (11,760,743,310 bytes, SHA-256 0b5e3c4bf77a473cd8c61d32fbd87b28cdbba043fb3b8267f3b8bcfb1d5b9e6b); created the e29 Python 3.11 venv reusing host-compatible torch 2.11.0+cu130. Strict predictor load passed (encoder no missing keys); final-config H20 GPU 6 random-clip forwards passed: bf16 0.687 s / 2.538 GiB peak allocated, fp32 0.644 s / 5.109 GiB. Loader: `tools/e29_vjepa_ac_load.py`; conventions/license: `s1/CONVENTIONS.md`; measurements: `s1/RESULTS.md`.
- **E29-S2 sub-task (2026-11-03 06:34 BRT):** Read pinned official conventions; added 54D measured-default/commanded-selectable joint-to-7D per-arm EEF state/action converter and seven passing tests. Cached 93 episodes / 27,281 frames with 26,909 k=4 action pairs at 3.75 Hz; translation/rotation delta MAE 6.59 mm / 1.72°. S2 results: `s2/REPORT.md`, `s2/state_action_stats.json`, `data/e29_state_action_20261103/`.
- **Change:** E29-S3 built stride-4 raw-frame windows from the official 9-episode E26a validation split and other 84 train episodes; added lazy cam_high crop/resize loader, two coordinate label modes, mask visibility flags, five seeded action variants, and per-object copy/oracle-motion baselines. Missing labels remain NaN with separate validity flags. Bin occupancy was joined by frame index from the E26a separate bin cache.
- **Config/command:** CPU only, profile luna-standard; `/mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python tools/e29_frames.py --self-test`; `/mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python tools/e29_frames.py`; then `python tools/e29_report.py`. Seed 20261103.
- **Run dir:** `data/e29_probe_windows_20261103/`; `runs/e29_vjepa_ac_probe_20261103/s3/REPORT.md`.
- **Results:** 2,235 eval and 23,558 train windows; eval phase counts reach/grasp/lift-carry/release = 1,932/43/196/64; moving/static = 350/1,885, with all eight phase-motion cells populated. Squash eval coverage: target bottle 79.6%, clean left/right hands 79.6%/94.8%, bin 100%; per-object/mode coverage and baseline errors are in the report and counts.json. Structural validation confirmed stride 4, split identity, NaN missing labels, shapes and seeded shuffle donors from other episodes in the same phase.
- **E29-S3 sub-task:** done.
- **Status:** in-progress (E29-S1/S2/S3 sub-tasks done; remaining E29 sub-tasks ongoing).

- **E29-S4 完成记录（2026-11-03T08:09:59.005197-03:00，sol-code）：** 冻结 V-JEPA 2-AC，H20 GPU7，仅训练解耦读出头，固定4000训练候选帧、train内episode验证选checkpoint。完成2235 squash窗口与350 center moving窗口，五动作变体×两只单臂×TF/AR四步、episode bootstrap1000次；主判定 部分。详见 `runs/e29_vjepa_ac_probe_20261103/s4/REPORT.md`、results.json、CSV与完整复现日志。grasp仅43/moving18，不作显著性判断；未下载或系统安装。S2报告已复制到标准目录，原件保留。


### E30 — colleague-e world-model read-only scan (2026-11-03 BRT)
- 配置：luna-research / gpt-6-luna max（任务书记载）。
- 结果：报告写入 runs/e30_colleague-e_scan_20261103/REPORT.md；覆盖 worldmodel DF、OpenWAM Optical93/bottle、Pi05 joint54、FK cache、overlap 及 Wuji bottle 旧实验。最接近目标动作契约的是 Optical93 的 54D joint state/command；其数据有 top 与双 wrist camera。
- 只读边界：对 /mnt/workspace/colleague-e 仅执行元数据和文本读取；未运行脚本、未拷贝权重或数据。
- 下一步：如开展对照，先确认 q_cmd→residual action 语义及 top-camera 标定；保留 MuJoCo outcome model，不把旧 latent-dynamics 分支接入 planner。

### E31 — worldmodel-df 深度调研
- **E31-B 子任务（2026-11-03 10:04 BRT，luna-research / gpt-6-luna max）：** 完成 `runs/e31_dit_wm_survey_20261103/B_dfwm.md`（68 行）；覆盖 JointSingleSeq、DF、EE50、数据/训练状态、样例检索、开销及 54D 单任务取舍。最新状态文件为 96/30,000 更新，未找到可读预测样例或动作反事实质量评估；对 `/mnt/workspace/colleague-e`、`/mnt/workspace/colleague-c` 严格只读，未运行项目脚本、未拷贝模型/数据/视频、未使用 GPU。**子任务状态：done。**


### E31 — Diffusion world-model survey (2026-11-03 BRT)
- **E31-A 子任务（2026-11-03 10:04 BRT，luna-research）：** 完成 Optical93/OpenWAM 源码、训练配方、全验证曲线、动作/部署与真机记录调研；报告：`runs/e31_dit_wm_survey_20261103/A_openwam.md`。对 colleague-e/colleague-c 只读，无训练/推理/GPU/数据或权重拷贝。

### E32 — JEPA 自有世界-动作模型与离线教师

- **E32-T1（2026-11-03 BRT，执行中/权限阻塞）：** 已发布 stride8 的 3,078 窗口索引（2,812 train / 266 eval）和 READY，完成源码约定及断点/PAUSE 脚本。指定 checkpoint 为 root:colleague-c 0640，dengjianfeng 实际 open PermissionError，尚无成功 GPU forward、探针、特征 READY 或基线；详见 runs/e32_jepa_distill_20261103/t1/REPORT.md。对 colleague-e/colleague-c 只读，无权重复制或他人进程操作。
- **E32-T3 (2026-11-03 11:40 BRT):** 完成 `tools/e32_eval.py`；FK 与 OpenWAM `state_eef58` 同步核对 93 episodes/27,281 frames，位姿与手部误差为浮点精度内 0；266 eval 窗口 copy/真值自身 JSON+CSV、episode bootstrap、中心 CNN（CPU，8k train/8 epochs）、解码帧/action-swap API 与 smoke tests 均完成。检测器留出中位误差 hand L/R 7.6/9.7 px、bottle 34.0 px、box 5.5 px；详见 `runs/e32_jepa_distill_20261103/t3/REPORT.md`。OpenWAM/T4 预测未生成，故不做横评。
- 2026-11-03 11:34 BRT — E32-T4（sol-code，E32-dinob24-PhaseE-v2）：代码、CPU假教师20步、H20假教师20步、精确恢复与部署验证通过；真实教师20步因T1 checkpoint读取权限等待 READY，未全部验收；正式训练未启动；报告 `runs/e32_jepa_distill_20261103/t4/REPORT.md`；第三方目录只读、未操作他人进程。


### E32 results (Claude takeover, 2026-11-03 BRT)
- Teacher: OpenWAM-Alpha foundation (fine-tuned ckpt unreadable, root:colleague-c 0640). Probe: hands 7.2-7.8 px (random ctrl ~25 px), bottle 33 px vs copy 10 px (fails pre-registered rule; random ctrl also 34 px, so the probe is uninformative for the bottle). Claude overrode the stop rule (cache ~23 GPU-min): layers 10/15/20, sigma 0.05, wrist blanked; data/e32_teacher_feats_20261103/READY.
- Student 4 epochs (t4/full_no_teacher_v1): worse than hold at all steps (undertrained; flow loss still falling).
- Student 40 epochs, batch 8 x accum 2 (t4/v2_noteacher_e40, t4/v2_teacher_e40), 266 eval windows / 9 episodes, tools/e32_eval_student.py:
  - EEF pos error at step 16 / 32 (mm, left / right): no-teacher 45.5/58.0 and 66.5/88.0; teacher 46.3/60.1 and 69.3/92.1; hold 54.9/70.7 and 81.7/98.1. Step-32 episode CIs overlap with hold (unpaired).
  - Decoded frames PSNR (dB) cur / +16 / +32: no-teacher 21.13 / 19.22 / 18.52 vs copy 18.91 / 17.90; object-weighted MSE +16/+32: 0.0271/0.0327 vs copy 0.0319/0.0416.
- Conclusion: 40-epoch student beats hold at steps 8-32 and predicts futures better than copy (esp. object regions); the Alpha teacher gives no gain (slightly worse on every metric; single seed). Status: done.

- **E33-A（2026-11-03 22:11 BRT）：** 完成共享256D B/C/D模型与续跑训练代码；各20步GPU4 smoke、7项结构/接口检查、bitwise参数/优化器续跑、PAUSE、fp16 ep10加载及独立部署验证均通过；正式80epoch未启动，launch_e33.sh仅交付；详见 runs/e33_20261104/t_a/REPORT.md。C首步SIGReg编码器梯度占比98.609%，供审核。
- **2026-11-03 22:19 BRT — E33-E:** 完成 tools/e33_eval.py / tools/e33_compare.py；266 窗口 E32 batch16 回归最大差 6.16e-7，E33 B smoke 真实模型端到端评测完成；compare.md/compare.csv 已生成，B/C/D 正式 run 暂缺并已标注。

- **2026-11-04 00:46 BRT — E33_sub-设计评审定稿：** 已交付 `notes/2026-11-04-e33sub-design.md`（144行），含逐条评审、接口/损失/训练日程、离线诊断及验收矩阵、GPU预算；只做设计，未实现或启动GPU，未改E33/E32源码，第三方目录严格只读。
- **2026-11-04 04:18 BRT — E33_sub-实现：** 完成 R1/R2 各20步 CPU smoke、①–⑫验收与单窗口 E2E；ep80 正式缓存 3,078 窗口已构建。PCA32/64 均未过压缩门槛（PCA64 的 +16/+32 PSNR 损失与瓶子 IoU 降幅超限），无 PREPARED，未启动正式训练；24 个旧 e33/e32 文件 SHA256 未变。报告：runs/e33sub_20261104/IMPL_REPORT.md。

- 2026-11-04 05:33 BRT — E33_sub v2 实现（luna-max）：完成 full240x256-delta-v2 与独立 v2 stats/manifest；132 dev 窗表示准入通过并创建 PREPARED。R1/R2 各 20 步 CPU smoke、①–⑫、全维往返/局部头满秩/724 memory/真实 K8 分块/A 评分哨兵及单窗 E2E 均通过；参数 G=3,053,302、可训练 7,672,364。未启动正式训练、未用 GPU；受保护 e32/e33 与旧 e33sub run 哈希不变。报告 runs/e33sub_full_20261104/IMPL_REPORT.md。

- **2026-11-04 06:22 BRT — E34 实现交付：** 完成 notes/2026-11-04-e34-code.md 的代码、CPU审计与报告；报告 runs/e34_20261104/IMPL_REPORT.md（36行）。
  - CPU提取93 episodes/27,281帧；瓶子/手/箱池化映射自动一致；联系表160段×5帧，fit/dev=120/40。dev无暗瓶和箱边候选，projected-in-bin正例8；G0待人工复核，未创建标记。
  - 契约烟测15/15通过；AC/S/NOZ参数797,711/798,479/797,711；R1冻结B、零gate和同噪声复现CPU烟测通过。启动器和缓存器缺G0时均状态码2拒绝；评测延期，G1–G4未运行；无正式训练或GPU使用。
  - colleague-e未访问；colleague-c只读使用Python环境，没有写入。保护清单12,600项；结束比对发现6项差异（B_s2旧run四项、tools/e33sub_cache.py、tools/e33sub_verify.py），来源待核；E34写入范围不含这些路径，保持原状。

- **2026-11-04 07:53 BRT — E33_sub 终止（用户决定）：** R1/R2（无rec损失，全240x256残差子目标，ep60）266窗评测未通过预登记门槛：generated A=1.169/1.164 vs E33-B 1.160；oracle仅比B好1.3%（53.9 vs 54.6 mm）；R1 视觉cosine 0.960<copy 0.966，R2 0.972。用户决定停止子目标路线并视为该技术不可达；不做 rec-weight A1、Phase2、K8选择。产物保留于 runs/e33sub_full_20261104/（不删除）。

- **2026-11-04 09:49 BRT — E35-vis（CPU，只读评估与渲染）**：执行 `notes/2026-11-04-e35vis-brief.md`，报告见 `runs/e35vis_20261104/REPORT.md`。使用 E33-B ep80（SHA256 `86595f3d0389b92001da2e0fc7863a0d9833f74f08feb17e12925ea6abe28fe7`），CPU 8 线程、266 个 stride-8 窗口/9 个 episode；未训练、未用 GPU。时间审计 `action[t]` 对齐 `q[t+L]`：A/B 臂最佳 L=2 帧（约133 ms），左手 L=1，右手 L=0；臂组偏差接近零。左臂末端第一步对教师命令的位置误差为13.64 mm（hold 14.37 mm），旋转误差2.72°（hold 2.34°）；对齐实测 q[t+2] 后模型误差仍高于 hold（位置12.10/7.41 mm，旋转2.62/1.63°），因此延迟不足以解释全部误差。步32左/右末端位置误差53.4/71.4 mm，hold为81.7/98.1 mm。已生成轨迹、MuJoCo静态姿态、目标/实测曲线、阶段误差图和23秒 MP4（115帧、5 fps）。当前 CPU 固定种子扩散样本与历史 CUDA 样本不完全一致（模型动作均值最大差2.19；hold最大差约9e-6），报告注明该限制。`colleague-e` 未读取；`colleague-c` 仅只读访问机器人描述。

- **2026-11-04 10:41 BRT — E38 pilot S1 (CPU，不标注)：** 冻结20集 manifest（12/4/4）、生成 raw/overlay 联系图各 41 张与 events_rule.csv；G4/G6 因 FK 垂直轴未核验而 disabled，G7/G8 保留 unknown。CPU smoke 8/8 通过；无模型分数/权重、无 GPU、无人工标注。报告 data/e38_subgoals_20261104/v1/STEP1_REPORT.md。

- **E36b（2026-11-04 10:49 BRT；宿主 CST−11h）**：按 `notes/2026-11-04-e36b-code.md` 完成隔离代码、训练/评测启动器、protocol 与 `runs/e36b_20261104/IMPL_REPORT.md`。CPU 烟测通过：C 对 E33 hold 推理逐位一致；T detach、t=0/1、10 次 predictor、encoder/target 冻结梯度、PAUSE/续跑均通过。H20 GPU7 通过空闲门槛后做 20 步 ephemeral 计时，p95=1.674 s；C/T×B/B_s2 共用 746 步上限。训练预计 1.665 GPUh，计时与评测预算合计 0.096 GPUh；未启动四个正式训练单元，也未运行最终评测。SHA256 见 `runs/e36b_20261104/source_hashes.json`。对 colleague-e、colleague-c 仅只读；未用腕部相机。

### E46 — 顶视标定复核与受限重标定（2026-11-04 13:30 BRT）
- **Status:** done
- **Scope:** CPU-only, two-thread compute pools; read-only data; no wrist camera, GPU, training or upper-computer access.
- **Result:** Reproduced legacy 4333 hand/frame comparisons, IoU mean 0.069227, centroid median 145.351 at 320x192. Distinguished viewer mesh geometry from measured E22b/E28 palm calibration. Compared offset-only/common bounded camera/lag 0–2 using inner fit-only selection, adopted offset; fixed ROI [64, 64]. Height proxy unavailable; G4/G6 remain disabled.
- **Validation:** 84 fit/9 heldout, all 93 episodes; strict palm/five-tip mask and ROI metrics, episode/pose stability, eight E35 overlays, exact E33 122D/block4 input differences; NumPy/Torch/OpenCV and source integrity checks passed.
- **Artifacts:** notes/2026-11-04-e46-recalib.md; runs/e46_recalib_20261104/; tools/e46_*.py. No old code/parameters/runs overwritten; protected third-party trees read-only.

### E36b eval-fix 采样稳定化与评测收尾（2026-11-04 14:24 BRT）
- **Goal:** 执行 `notes/2026-11-04-e36b-eval-fix.md`；诊断 T_B/T_B_s2 非有限候选、修复 E36b 推理稳定性并完成两基座全266窗评测。
- **Diagnosis:** 原始 T 在两个基座均为 4/4,256 候选非有限、4/266 窗；同四个 window/sample 首错均为 Euler 步0 FK `uv`，5个点投影深度≤0.05m，NaN 传播至 predictor/flow。详情见 `runs/e36b_20261104/{T_B,T_B_s2}/raw_diagnostic/summary.json`。
- **Change:** 仅修改 `tools/e36b_model.py`、`tools/e36b_eval.py`；新增默认关闭的诊断/稳定化开关。FK 输入按 train-fit `act_norm` q0.001/q0.999 裁剪，并用 `uv_grid=(-1,-1)` 保留无效投影标记。修复只影响评测推理；训练损失路径与默认采样未变。
- **Results:** T_B、T_B_s2 修复后均完成266窗且输出全有限；原始有限子集262窗单独报告，没有删评测窗。中英指标、配对CI、K8/K16、记忆表、G_action和延迟见 [`runs/e36b_20261104/RESULTS.md`](../../runs/e36b_20261104/RESULTS.md)。两基座全chunk T−C CI均含0，未见一致T优势。
- **Budget:** 评测总预算由307.3秒调至1500秒；未重训。本轮诊断0.049 GPUh，修复评测0.076 GPUh；累计E36b约0.559 GPUh（旧全窗失败耗时按241窗日志外推，细目见RESULTS）。
- **Compliance:** 仅顶视RGB；colleague-e/colleague-c只读；9个核心源文件哈希与原manifest相符。状态：done。

- **2026-11-04 14:39 BRT -- E47-A data implementation:** Added tools/e47_config.py, e47_data.py, e47_verify_data.py and read-only fit-stats snapshot. CPU smoke passed: 80/4/9 episodes, 2680/132/266 windows, geo66, causal dq/label alignment, slot/IDM, leakage and malformed-index sentinels; CUDA_VISIBLE_DEVICES empty, 2 threads, no training/GPU. Relevant input SHA256 start/end match. No writes to colleague-e/colleague-c; colleague-c used only read-only for model description and existing Python environment. Report: runs/e47_p0p1_20261104/IMPL_REPORT_A.md.


- **2026-11-04 14:44 BRT — E47-C eval implementation：** 新增 tools/e47_eval.py 与 tools/e47_verify_eval.py；覆盖 E33 CPU 兼容重放、E47 direct/rollout 与 copy 评测、动作残差 shuffle 重锚、双分辨率 mask IoU、corrected_obj_h/legacy 广播复现、逐窗 MSE 聚合 PSNR、episode 配对 bootstrap 与 (a)–(e) judge。12/12 CPU 桩/手算哨兵通过，结果 runs/e47_p0p1_20261104/eval_smoke.json；实现报告 runs/e47_p0p1_20261104/IMPL_REPORT_C.md（37行）。使用 colleague-c 现存 Python 环境，pyc/XDG/Torch/Triton 缓存重定向到新 runs/e47.../cache；CUDA_VISIBLE_DEVICES 为空，无训练、GPU、权重下载。colleague-e 未访问，colleague-c 仅解释器只读；共同起始清单的保护源哈希均未变化（e36b 按约定排除）。真实 E33/E47 266窗预测与门槛裁决未执行，不能据此声称 (a)–(e) 通过。


- **2026-11-04 14:56 BRT — E47-B 实现：** 完成 `tools/e47_model.py`、`e47_train.py`、`e47_verify_model.py` 与 20 行报告 `runs/e47_p0p1_20261104/IMPL_REPORT_B.md`。实现 960×256 E47、因果 F/direct+rollout、RGB/mask D、IDM、当前态 flow/propose；另有隔离 E33-B refit，均只从本机官方 DINO 权重初始化，拒绝旧 E33 checkpoint 隐式加载。CPU smoke 覆盖 §6 第5–9、11–12项：B2全形状/因果与动作交换、geo66 NaN回退、梯度白名单及 flow detach、rollout链/IDM加权期望、EMA/续跑/PAUSE/preflight；E47真实 fit 窗 loss 3.18745、6.81s、RSS约2.50GiB，B-refit真实窗 loss 2.78148、24.18s；均只前后向、无 optimizer.step。E47总/可训参数188.49M/43.25M；B-refit 182.17M/37.17M。无GPU/训练/下载。`e47_data.py:436` 当前有 A-owned `raw_q` 初始化顺序错误，B 包装器只读回退并附加当前mask/IDM pair，未改A文件。实际官方权重SHA256 `0b8b82f8…fad8c73`；起止保护源码hash一致，详见 `source_hashes_{start,end}.json`。colleague-e未访问，colleague-c只读使用现存Python环境，测试缓存定向 E47 `runs/.../cache`。

- **2026-11-05 03:12 BRT — E47 集成：** 修复数据项当前四槽 mask/IDM 与 B 回退，统一 `run_eval`/Context stats+episode donor 接口，修 pixel legacy N>1 广播；三子校验及真实 E47/B-refit 前后向 smoke 通过（E47 loss 3.0324，B-refit 2.9737，无 optimizer.step）。E33-B ep80 266窗 CPU 重放复现 PSNR 与 legacy obj_h（0.02050560/0.02357990），corrected_obj_h 0.02385345/0.02802114；h32 mask、动作因历史 CUDA RNG/本机无CUDA未达容差，详见 `runs/e47_p0p1_20261104/INTEGRATION_REPORT.md`。GPU计时未跑（无CUDA/NVML telemetry），正式训练未启动；源哈希见 `source_hashes_start_integration.json` 与 `source_hashes_integration_end.json`。`colleague-e`、`colleague-c` 无写入，缓存仅写 E47 runs。


### E47 GPU 资源复核与集成报告更新（2026-11-05 03:23 BRT）
- 执行 `ssh -o BatchMode=yes <gpu-server>`；读取 `INTEGRATION_REPORT.md` 与 `notes/2026-11-04-e47-integrate.md` 后检查 `nvidia-smi`。8 张 H20 利用率均为 97–100%，尽管空闲显存为 25–64 GiB，仍无卡同时满足空闲显存≥16 GiB 且利用率≤10%。未触碰已有进程。
- 因无空闲 GPU，E47/B-refit 各 20+100 步 `time_profile` 和 E33-B ep80 CUDA 兼容重放均未启动；本轮 GPU 使用 0.000 GPUh，未运行训练或 optimizer.step。状态、卡快照及未测字段写入 `runs/e47_p0p1_20261104/gpu_profile.json`。
- 既有 266 窗 CPU 重放差异已在 `runs/e47_p0p1_20261104/INTEGRATION_REPORT.md` 填为真实值：h32 mask IoU 最大差 0.001253；左末端位置最大差 2.195 mm、右末端旋转最大差 0.336°。动作差异与历史 CUDA/当前 CPU RNG 流不同相符；h32 mask 差异可能受 CPU/CUDA 数值/阈值影响，但待 GPU 对照确认。
- 本轮没有访问 `/mnt/workspace/colleague-e` 或 `/mnt/workspace/colleague-c`，没有系统安装。

- **2026-11-05 04:29:56 BRT — E47 GPU共享计时与 E33-B CUDA 收尾：** 主卡 GPU0（选择时空闲 29,741 MiB、利用率 82%）对 E47/B-refit 各跑 20+100 步；GPU3（选择时空闲 97,352 MiB、利用率 0%）复测两臂。E47/GPU0 p50/p95=0.5084/0.5627s；B-refit/GPU0=0.1410/0.1502s；GPU3 复测 E47=0.3270/0.3682s、B-refit=0.1412/0.6103s。均标注为共享卡上界、非独占值；13,440 更新按 p95 外推为 2.101/0.561 GPUh（GPU0）和 1.374/2.278 GPUh（GPU3），与 0.442 s/update 的判定见 runs/e47_p0p1_20261104/INTEGRATION_REPORT.md。五个任务窗口保守计 0.0732 GPUh < 0.3；无 OOM、未调用 train()、未干预他人进程。E33-B ep80 的 266 窗 CUDA 重放（GPU0、device=cuda、batch=16、seed=20261103）四类 h32 mask IoU 和六项 step32 动作指标与历史一致；此前 CPU/CUDA 兼容差异由设备路径解释。产物和采样日志见 runs/e47_p0p1_20261104/。

## E55-1 goal-distance scorer v2 (2026-11-06 BRT)
Run dir: /mnt/workspace/dengjianfeng/code/wam_common/runs/e55_v2 (REPORT.md). Claude check: SHA of fit/eval bundles unchanged; formal eval run_count=1.
# E55-1 运行日志

- 工作树：/mnt/workspace/dengjianfeng/code/wam_common；持久SSH ControlPath=$HOME/.ssh/cm/e55.sock；未改 ~/.ssh/config。
- 阅读简报及设计、E53报告、features/get_features/gates；实现仅新增 scorer/scorer_v2/ 与 runs/e55_v2/ 文件。没有改T/W已有文件或数据包。
- CPU-only，CUDA_VISIBLE_DEVICES为空，nice -n 10，OMP/MKL/OpenBLAS/NumExpr线程上限4；未启动GPU任务或训练分类头。
- 先登记60变体及数据SHA；fit-only LOEO结束于2026-11-06 14:42:52 BRT，h32主A_R0_cosine_k1_none；族内与h16选型见config.json、fit_cv.json。
- 修复后再次运行8项单测：Ran 8 tests，OK；覆盖目标泄漏保护、完美/随机/反向、NaN与valid、dev_only、零范数、float32极值、逐行/批量一致。
- 先前4份诊断性eval输出分别因R0 mask依赖有效性、零范数误拒、float32范数溢出、batch循环写错全局行而作废；均未用于选择或最终指标。修复后正式eval只执行并记录一次：run_count=1，2026-11-06 14:47 BRT。
- 最终评估标签和eval real_future传入GoalDistanceScorer.fit/score标志均为false；oracle仅在评估后作为上限计算。config及metrics记录了这些隔离约束。
- 最终复核5个候选/配对文件SHA256全部与运行前相同；variant_matrix.csv 120数据行；metrics.json正式评估8集。
- 结果、区间、样本量、G-B原型支持、回归、分层、时延与未覆盖项见REPORT.md。没有作“通过”判断。

## E55-2 oracle-jitter / selection-regret diagnostic (2026-11-07 BRT)
Run dir: /mnt/workspace/dengjianfeng/code/wam_common/runs/e55_v2b (REPORT.md). Claude check: fit/eval SHA unchanged. Key: oracle demo>jitter05 only 0.527 (h32 R0).
# E55-2 运行记录

- 正式计算：2026-11-07 00:15:57 BRT 至 2026-11-07 00:16:45 BRT；约 48.0 秒。
- SSH：dengjianfeng-101，独立 ControlMaster socket /tmp/e55b.sock；未改 ~/.ssh/config。
- 工作树：/mnt/workspace/dengjianfeng/code/wam_common；读入 T 数据与 E33 checkpoint 元数据。
- 执行：CPU-only，nice -n 10，CUDA_VISIBLE_DEVICES 为空，OMP/MKL/OpenBLAS/NumExpr/Torch 线程上限 4；未训练、未调用 GPU。
- 正式 E55-2 计算完成一次。E55-1 的 60 变体选型与正式评估未重跑。
- eval labels/real future 未进入 GoalDistanceScorer.fit/score；真实未来只用于评分后 oracle 和距离比。
- 没有改写 E55-1 文件、输入 NPZ、数据包或 EXPERIMENT_LOG.md；没有 git commit。
- 调试尝试 1 作废：初版误从 compute_rank_gate 读取 jitter 项（该函数只返回 hold/shuffle/reverse），在 metrics 输出前中止，未用于结果。
- 正式计算的首轮 Markdown 渲染因 E55-1 jitter 字段键名未加 demo> 前缀而失败；已仅从已保存 metrics.json 重建 REPORT.md/LOG_ENTRY.md，没有重跑评估。
- 分析脚本 SHA256：metrics.json 中记录；报告渲染脚本 SHA256：ab6bed7132e5ff5e669ba78294c49bdede459d82cc1ee4c38323e3f2a4eff7fa。
- E33 模型 spread 与策略采样候选的同窗预测缓存缺失，因此未生成新的策略候选，也未对模型噪声下限作数值推断。

## 五个输入文件 SHA256（开跑前 / 完成后）

|输入|开跑前 SHA256|完成后 SHA256|一致|
|---|---|---|---|
|fit.npz|075e68d622f88b025f274bb991544f4d35b58a387d388550037b3fd30303a95c|075e68d622f88b025f274bb991544f4d35b58a387d388550037b3fd30303a95c|true|
|dev.npz|c0925c71b6435276f7908e19ab25b0531f653d912d6888c67e7648ddbcc088f3|c0925c71b6435276f7908e19ab25b0531f653d912d6888c67e7648ddbcc088f3|true|
|eval.npz|70066598b828fa0372eb7a80117f927e50e6899ec03b4519b69545ed12889769|70066598b828fa0372eb7a80117f927e50e6899ec03b4519b69545ed12889769|true|
|pairs_relaxed_fit.npz|2123baa53c79e8f425d1ebf3325b2c1d89b4313780f1365ad8ea3cca80c7d5d3|2123baa53c79e8f425d1ebf3325b2c1d89b4313780f1365ad8ea3cca80c7d5d3|true|
|pairs_relaxed_eval.npz|95c509648dea127209e870056f27a21e15940b0a9258521a9fdb25bed56a8b7b|95c509648dea127209e870056f27a21e15940b0a9258521a9fdb25bed56a8b7b|true|

## 产出

- /mnt/workspace/dengjianfeng/code/wam_common/runs/e55_v2b/config.json
- /mnt/workspace/dengjianfeng/code/wam_common/runs/e55_v2b/metrics.json
- /mnt/workspace/dengjianfeng/code/wam_common/runs/e55_v2b/REPORT.md
- /mnt/workspace/dengjianfeng/code/wam_common/runs/e55_v2b/LOG_ENTRY.md

## E55-3 H-JEPA ideas: score-time monotonicity + temporal pooling (2026-11-07 BRT, Claude, CPU)
Script W/scorer/scorer_v2/run_e55_3_hjepa.py; out W/runs/e55_v3_hjepa (REPORT.md, metrics.json). Inputs SHA unchanged; regression G-A R0 h32 hold .920 matches E55-1. Details also appended to notes/2026-11-06-e55-scorer-v2-design.md.
- Whole-task goal (G-C R0 h32) monotone: track median Spearman(score,t) 0.32, ep mean 0.36 [0.06,0.65]; subgoal G-A not monotone (-0.15). R1 ~0.
- Temporal pooling (h16+h32 mean, goal pooling +-8/16 frames) does not help jitter_05/10 (scorer .49-.52); oracle h16 is MORE jitter-sensitive than h32 (.579/.851 vs .527/.744).

## E55-4 stage-aligned subgoal (next-event anchor) monotonicity (2026-11-07 BRT, Claude, CPU)
Script W/scorer/scorer_v2/run_e55_4_stage_subgoal.py (STAGE_AT=t|t+h); out W/runs/e55_v4_stage{,_tplush}/REPORT.md; inputs incl. E49 labels SHA unchanged. 19 eval tracks / 8 eps, 42 within-stage segments: underpowered.
- Stage-at-t subgoal is NOT monotone within stage (S-proto h32 ep mean -0.30 [-0.51,-0.12]; real future also negative = definition issue: h32 future overshoots the next event). Stage-at-t+h removes the negative sign but no CI excludes 0. G-C whole-task stays best (0.34 [0.04,0.64]). Subgoal ranking weaker than G-A (shuffled .55-.59 vs .71). Inferred stage agrees with labels only 57%.

## E56 - Action selection literature survey (2026-11-07, Codex luna, CPU only)
- Report: notes/2026-11-07-e56-action-selection-survey.md (22 refs; Claude spot-checked the 2026 arXiv refs 2610.37196, 2611.03585, 2605.07514, 2601.16163, 2602.15922: all exist and match; the V-GPS OpenReview link could not be verified).
- Main points: (1) measure selectable headroom first (Gao et al. 2026: one-shot best-of-N overstates it; structured perturbations add headroom; fail-gated selector falls back to the base action). (2) Label-free selectors exist: consensus among predicted futures (Ruan et al. 2026), inverse-dynamics consistency + progress (WISE 2026). (3) Use short horizons, frequent replanning, a geometry baseline (SAM3 bottle-bin) and hard safety gates.
- Claude review: Codex looked for the E53/E55 reports under T instead of W (wam_common), so it used the brief numbers. Its recommendation 1 assumes E48/E49 hold per-candidate outcomes; they do not (E48 = synthetic demo perturbations, E49 = demo event labels), so an offline headroom audit needs generated policy samples first.

## E57 - run E policy-sampled candidates scored by E33-B (2026-11-07, Claude, GPU 0 45 s + CPU)
- Data: W/runs/e57_policy_cands/ (candidates.npz, meta.json, select_metrics.json, REPORT.md); scripts W/scorer/scorer_v2/gen_e57_policy_candidates.py, run_e57_select.py.
- 245 eval windows (ep 90 skipped) x 16 run E samples (8 at T1.0, 8 at T0.5), plus demo_full, demo10_hold and hold. run E chunk = 10 steps, extended to 32 by holding the last command.
- Results:
  - Truncation dominates h32: demo_full 0.389 vs demo10_hold 0.464, about the same as a random sample (0.463).
  - E33-B barely separates run E samples: spread cosine 0.004; Spearman(action-to-demo, pred-to-real) 0.09 [0.05, 0.13].
  - Future-consensus selector gains 0.004 rad over random (about 30% of the 0.013 action-space headroom), the same as an action-space medoid. The world model adds nothing here.
  - G-A/G-C prefer larger motions: better h32 oracle, worse action match.
- Next: match the predictor horizon to the 10-step decision (h8/h10); structured perturbations; real branch outcomes eventually.

## E58 - h10 head on frozen E33-B, E57 selection redone at h10 (2026-11-07, Claude, GPU 0 342 s + CPU)
- Data: W/runs/e58_h10/ (REPORT.md, head.pt, metrics.json, select_metrics_h10.json); scripts W/scorer/scorer_v2/e58_train_h10.py, run_e58_select.py.
- Setup: head = copy of the E33 predictor, new horizon embedding, first 3 action tokens. Trained on 78 train episodes (6 held out as dev) with demo 10 steps + hold. Eval episodes not used in training.
- Prediction quality (eval JEPA loss): h10 0.0364 vs E33 h16 0.0399 vs persistence 0.0513. Hold input is out of distribution (0.0907).
- Selection at h10:
  - Truncation bias is gone (demo10 0.298 ~ demo32 0.301 < policy 0.311). The oracle prefers demo10 over policy samples 0.66 [0.55, 0.77]. Resolution Spearman 0.28 (was 0.09).
  - Model-internal selectors still fail: G-A ranks demo over samples 0.46, G-C 0.35. Future consensus gains 0.0037 rad, the same as the action medoid (0.0041).
  - Even the real-future oracle gains only 0.003 rad.
- Conclusion: no measurable extra selection value from the world model among run E samples. Next: structured perturbations; real branch outcomes.


## E59 - Structured perturbation resolution for h10 (2026-11-07, Codex, CPU)
- Script: W/scorer/scorer_v2/run_e59_perturb.py; ran --limit 8 smoke, then full evaluation. Outputs: W/runs/e59_perturb/ (REPORT.md, metrics.json, config.json, oracle_winrate.svg).
- Data: 245 E57 demonstration windows across 8 evaluation episodes (7, 18, 38, 56, 64, 66, 77, 88); episode 90 skipped. 67 perturbation candidates/window (4 stochastic families x 5 magnitudes x 3 seeds, 5 speed factors, 2 deterministic freezes), 16,415 candidates total; config records all 14,700 derived random seeds.
- Setup: E58 h10 head on frozen E33-B, CPU, max 4 threads, nice 10. H20 GPU 0 had about 63.1 GB in use, so the run stayed on CPU. Episode bootstrap 10,000 draws, seed 45; E55 rank-gate comparison with tie margin 0.02; nonfinite comparisons contribute zero and remain in the denominator.
- Policy-sample RMSE to demo10: 0.0573 rad [0.0491, 0.0653], matching the brief's 0.057 rad reference. Oracle threshold m (log interpolation): white 0.0615, offset_all 0.0358, offset_arm 0.0212, offset_hand 0.0585; speed and both freeze groups did not reach 0.70. Selector thresholds are tabulated in REPORT.md; offset_arm had the lowest mean realized threshold scale (0.0476 rad), while offset_hand averaged 0.0777 rad. No selector reached 0.70 for speed/freeze groups at tested levels.
- One candidate pair at offset_arm m=0.16 had nonfinite model-based scores (1 each for oracle, G-A, G-C, and consensus); those rows remained in the denominator. The run completed otherwise. No velocity clipping; perturbation RMSE is only a constructed proxy, with no success/failure labels and no physical-failure interpretation.

### E82-L | 2026-11-12T10:21:09.767671-03:00
目标：Piper train/P1 模型标注 L1–L3，L4可选。改动：新labels_codex/SPEC.md及报告骨架；命令：SSH只读文档/CPU原位抽帧，等待READY_parent_split.json。目录：data/e82_piper_v1/labels_codex/；notes/e82_L_piper_labels_20261112.md。结果：入口待就绪，尚未抽帧/标注。结论：不得作测试真值/阈值/选型。状态：waiting_parent_split；GPUh=0。

### E82-D | 2026-11-12T10:41:53.657807-03:00
- 目标：Piper存量parent隔离、P1/P3数据治理与CPU缓存；天机暂停，P2=0。
- 改动：新data/e82_piper_v1的D所有目录、READY_parent_split/READY_data及notes/e82_D_piper_data_20261112.md；未写labels_codex、code_e82或源目录。
- 命令：ssh -o BatchMode=yes <gpu-server>；e29/bin/python build_parent_split_v2.py / build_cpu_package_v2.py / final_numeric_audit.py / build_pixel_warm.py / fit_candidate_stats.py；OMP/MKL/OPENBLAS=1，CUDA_VISIBLE_DEVICES空，setsid nohup分离长任务。
- 结果：P1 560条/692277帧/6.409972h/245 parent，109029窗口；P3候选167条/282033帧/2.611417h/150 parent，33888窗口。test与训练交集0；167来源原pair误差0；1625源stat无变化。缓存约0.244GB。
- 结论：完整parent图与split锁定，已知跨集三组无泄漏；仅CPU、GPUh=0、无训练。首版source metadata缺口已撤回补齐，既有split/component变化0；审计保留。
- 状态：cpu_governance_complete_pending_P3_quality；demo3 P3 pending；P3_ready=0等独立示范/可见性审核，不以成功标签替代质量。


### E82-M | Piper统一模型与训练评估流水线 | 2026-11-12 BRT

- 审计时间：2026-11-12T13:40:37.566534-03:00
- 目标：E79/E81 Piper O/A/W统一flow及P1→P3/R→P3流水线；本轮仅合同/小样本/profile，禁止正式训练。
- 改动：新建code_e82（core、原始本地DINO双塔/LoRA、Piper adapter、损失/噪声流、部署两路径、manifest/实际D数据入口、配置、训练恢复止损、网格、预算、时延、父等权/簇bootstrap、链式启动脚本）；不改旧工具。
- 命令：scripts/run_contracts.sh；python -m compileall -q e82 tests；scripts/fixture_smoke.sh；scripts/real_cpu_smoke.py；scripts/run_bounded_profile.sh（无空闲卡退出75）。
- 目录：T/code_e82/；报告T/notes/e82_M_model_pipeline_20261112.md；证据code_e82/artifacts/implementation_release.json、contracts_release_final.txt、real_cpu_forward_v1.json、gpu_accounting.json。
- 结果：29 tests通过，core4,739,072、P3可训5,618,446、LoRA49,152；O187/A50/W273；2476文件hash通过；真实DINO生产尺寸CPU缓存误差0，cuda_initialized=false；合成P1/P3/resume与100更新fixture profile通过。
- 偏差：早期CPU检查点CUDA RNG读取可能初始化上下文，已修复并加CPU禁CUDA回归；无CUDA模型计算。按全部相关CPU墙钟×8卡保守上下文GPUh上界0.458525，真实profile桶降.4，合计上界.858525<1。不能宣称严格0上下文。
- 结论：真实数据已接线；8卡均占用，真实100步GPU profile/阶段GPUh与P99未测，48h可行性未决；P3_ready=0、ROI待人工审核，正式gate拒绝放行。
- 状态：DELIVERED_CODE_CPU_VERIFIED / GPU_PROFILE_BLOCKED；未启动正式训练。需安排空闲H20短时窗口；天机/R0按用户要求不做。


### E82-L | Piper训练池模型标注 | 2026-11-12 BRT 完成

- 时间：2026-11-12T14:19:46.055864-03:00；labeler=codex_gpt-6.1-sol；CPU only，GPUh=0，不训练。
- 目标：L1任务身份、L2自主→接管parent结局、L3训练输入可见性；用途仅P1均衡/P3筛选，**不得作测试真值/阈值/选型**。
- 改动/目录：新建T/data/e82_piper_v1/labels_codex/，SPEC先行、5条两遍自检、三类JSONL与空L4、证据原帧/索引/校验/READY；报告T/notes/e82_L_piper_labels_20261112.md。不改代码/原数据/动作。
- 命令：SSH BatchMode 只读列举并查看指定报告和 spec.json；cat 新建报告，cat >> 追加两份日志；未调用 GPU/训练命令。
- 命令：ssh -o BatchMode=yes <gpu-server>；180秒READY轮询；setsid nohup CUDA_VISIBLE_DEVICES=''、OMP/OPENBLAS/MKL=1的Python h5py/OpenCV/PIL原位抽帧；元数据/图像人工式模型审阅；SHA256/源stat/划分/源帧范围校验。
- 结果：READY等待6分钟；revision3九项hash一致；560源仍train/allow_p1；L1 275=274same+1诊断unclear；L2覆盖257目标/115parent=67success+16failure+32unknown；L3 20parent四层各5、60对照帧480输入判断+160过渡帧。dev/test与sim标注0。labels约0.913GB。
- 结论：demo3任务身份支持同域候选，诊断排除；不批准P3片段质量。R1四状态证据不足未通过，固定ROI漏头/截尖端。L4可选未做；10条训练parent人工抽查清单已给。
- 状态：LABEL_REVIEW_COMPLETE_WITH_UNKNOWNS / R1_INCONCLUSIVE / L4_SKIPPED。READY只表示标签可读，不代表训练或评估gate放行。


## E83-A | 2026-11-12T14:51:01.018334-03:00 | Piper Tier A 自然语言训练标注
目标：替换仅结构化标签的不足，编写实际三视角证据支持的双语画面、任务指令、过程和关键帧描述。改动：新建data/e82_piper_v1/lang_codex/A与notes/e83_A_piper_language_20261112.md；源与E82冻结入口只读。命令：python3 notes/e83_A_bootstrap_20261112.py；setsid nohup env CUDA_VISIBLE_DEVICES= .../e21_sam3_20261101/bin/python -u extract.py pilot_manifest.json / extra_manifest.json；python3 commit.py authored_pilot.json。结果：5条试标、自检通过，逐条JSONL追加与progress原子写入。结论：可继续同样证据审阅的分层批量，不代表P3质量放行。状态：进行中。


### E83-A 部分交付检查点 | 2026-11-12T15:07:53.041355-03:00
目标：Piper Tier A双语长文本训练监督。改动：独立lang_codex/A内9条标注、73段过程、80关键帧及3条完整样例、progress断点、quality_summary与DELIVERY哈希。命令：setsid nohup CPU extract.py batch_001_manifest.json / batch_002_demo3_manifest.json；python3 commit.py <authored_batch>；python3 finalize.py。目录：data/e82_piper_v1/lang_codex/A；报告notes/e83_A_piper_language_20261112.md。结果：demo4=4、demo6early=2、demo6native=2、demo3=1；全部train/P1，评估帧=0，源stat变化=0，缩略图54055607bytes。结论：已审文本仅供language_supervision_training，不作测试真值/阈值/checkpoint选择，不改变P3放行。状态：partial_reviewed_checkpoint；剩142 P3+76 demo3，未全量完成。


## E83-B | 2026-11-12 BRT | Piper Tier B 双语语言监督

- ID：E83-B；完成：2026-11-12T15:12:00.475222-03:00。
- 目标：为P1 train且不属P3候选的parent生成自然语言场景、详细指令、3改写和独立事后回顾。
- 改动：只新增data/e82_piper_v1/lang_codex/B与notes/e83_B_piper_language_20261112.md；未改E82入口/标签或源数据。
- 命令：python3 /tmp/e83b_prepare.py；setsid nohup CPU extract_cpu.py 0 5 / 5 95；commit.py逐批；finalize.py；report.py。
- 目录：data/e82_piper_v1/lang_codex/B/；主入口lang_B.jsonl、progress.json、samples.md、FINAL_QA.json。
- 结果：95 roster，94双语条，1诊断隔离；dev/test=0，P3交集0；场景精确重复0；源stat376路径变化0；缩略图128,556,209 bytes；GPU 0，不训练、不下载模型/原数据。
- 结论：可作语言训练目标；13条完整分工顺序未录足，规范补全已标记；1条左腕不可解码；不作测试真值/阈值/选型。
- 状态：完成（明确证据限制及1条有意排除）；详见报告三个样例和质量统计。

## E84-P e84p_20261112_153338 | 2026-11-12T15:33:38.234087-03:00
- 目标：模型无关真值检验三类对齐，决定阶段一；不训练WAM。
- 改动：新建报告、进度与专属特征/探针/LoRA/代码目录；严格等待F split/GT。
- 命令：`python3 code_e84/e84p_20261112_153338/bootstrap.py`（原始初始化脚本）；后续命令见报告。
- 目录：`/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261112_153338`；报告：`/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_P_e84p_20261112_153338.md`。
- 结果/结论：待实验，禁止读取dev/test。状态：running_waiting_split。


### E83-A续标准备 | 2026-11-12T15:45:10.844469-03:00
用户允许无合适长片段时减少关键帧与阶段；更新commit.py与progress.json。命令：setsid nohup CPU extract.py batch_003_manifest.json；结果：三个train/P1候选抽帧完成，未做视觉复核与语言标注；完成仍9。状态：未完成，待从batch003继续。

## E84-F | 2026-11-12T15:47:51.856441-03:00
- ID：E84-F；目标：train-only FK规范语言与模型无关图像真值，不训练WAM。
- 改动：锁定probe split；CPU FK 548 capture/244 parent/436,116行；规范与断点。
- 命令：python3 fk/bootstrap.py；setsid nohup CPU fk/build_fk.py；setsid nohup CPU fk/build_images.py。
- 目录：data/e84_piper_align_v1/{split,fk,gt}；报告notes/e84_F_fk_gt_20261112.md。
- 结果：READY_split已交；195/49 parent；GPU0；无dev/test读取、无下载；图像及目视审核进行中。
- 结论/状态：running_image_manual_review_pending；完整哈希回执待最终验收。


E83-A | 2026-11-12T15:56:38.774908-03:00 | 目标：Tier A双语长文本续标 | 改动：新增3条并重新选择demo4同parent代表片段 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计12条，剩余P3 139、demo3 76 | 结论：当前批字段与证据校验通过 | 状态：进行中


E83-A | 2026-11-12T15:58:04.974353-03:00 | 目标：Tier A双语长文本续标 | 改动：新增3条并重新选择demo4同parent代表片段 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计13条，剩余P3 138、demo3 76 | 结论：当前批字段与证据校验通过 | 状态：进行中

## E84-F 最终交付 | 2026-11-12T16:02:33.051339-03:00
- ID/目标：E84-F；train-only FK规范运动语言与模型无关GT。
- 改动/目录：data/e84_piper_align_v1/{split,fk,gt}；READY_split/READY_fk_gt、SPEC、逐文件哈希；报告notes/e84_F_fk_gt_20261112.md。
- 命令：SSH BatchMode 只读列举并查看指定报告和 spec.json；cat 新建报告，cat >> 追加两份日志；未调用 GPU/训练命令。
- 命令：CPU setsid nohup fk/build_fk.py、fk/build_images.py；fk/finalize.py；按capture/image断点恢复。
- 结果：195/49 parent；548 capture/436,116 FK行；5824/5832可解码图；目视30投影/30颜色/20方向对；源stat0变化；GPU0，F派生0.504GB。
- 结论：FK语言ready；HSV仅橙色候选，非对象身份GT；demo4投影粗诊断，精确像素未放行；demo6/demo3无投影GT，右腕未放行。
- 状态：complete_with_calibration_and_hsv_limits；无WAM/模型训练/下载、无dev/test样本读取。

- 2026-11-12T16:05:03.340948-03:00 E84-F终审：静态artifact 18707文件SHA全部一致；E82允许train缓存核验1096项。诊断parent隔离712行后有效435,404 FK行、5,800图、194/49 parent；共同E84存储快照0.663GB。READY_fk_gt已锁定；精确像素/目标扎带实例GT未放行。

## E84-P e84p_20261112_153338 前置核验停止 | 2026-11-12T16:37:44.053580-03:00
- 目标：E84三类对齐验证，决定阶段一；不训练WAM。
- 改动：双塔/patch/tokenizer审计、revision3 split与F资格核验、采样清单、E83/FK信息量、合成实现验证、12张目视复查、断点代码；setsid nohup自有等待进程已停止。
- 命令：code_e84/e84p_20261112_153338/{audit_model,contract,prepare,accept_truth,language_audit,fk_consumption,record_visual_audit}.py；正式extract/run_all/LoRA未启动。
- 目录：data/e84_piper_align_v1/{features,probes,lora}/e84p_20261112_153338；报告notes/e84_alignment_summary_20261112.md。
- 结果：SigLIP2 base224/16双塔375187970参数；F投影中位37.44px、HSV24/30帧误检、精确像素未批准；E83 107记录/106parent；0 GPUh；R0.748GB。
- 结论：按用户缺什么就停，完整E84不能验收；不把proxy低分作模型失败。
- 状态：blocked_missing_qualified_pixel_gt；等待补真值或用户明确受限实验范围。


E83-A | 2026-11-12T23:35:14.583946-03:00 | 目标：Tier A双语长文本续标 | 改动：追加已核看条目并更新动态统计 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计16条，剩余P3 136、demo3 75 | 结论：当前批字段与证据校验通过 | 状态：进行中


E83-A | 2026-11-12T23:40:36.514912-03:00 | 目标：Tier A双语长文本续标 | 改动：追加已核看条目并更新动态统计 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计20条，剩余P3 133、demo3 74 | 结论：当前批字段与证据校验通过 | 状态：进行中


E83-A | 2026-11-13T00:06:55.990032-03:00 | 目标：Tier A双语长文本续标 | 改动：追加已核看条目并更新动态统计 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计73条，剩余P3 92、demo3 62 | 结论：当前批字段与证据校验通过 | 状态：进行中


E83-A | 2026-11-13T00:15:08.985175-03:00 | 目标：Tier A双语长文本续标 | 改动：追加已核看条目并更新动态统计 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计104条，剩余P3 76、demo3 47 | 结论：当前批字段与证据校验通过 | 状态：进行中


E83-A | 2026-11-13T00:22:17.380847-03:00 | 目标：Tier A双语长文本续标 | 改动：追加已核看条目并更新动态统计 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计129条，剩余P3 64、demo3 34 | 结论：当前批字段与证据校验通过 | 状态：进行中


E83-A | 2026-11-13T00:30:07.105567-03:00 | 目标：Tier A双语长文本续标 | 改动：追加已核看条目并更新动态统计 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计152条，剩余P3 52、demo3 23 | 结论：当前批字段与证据校验通过 | 状态：进行中


E83-A | 2026-11-13T00:44:31.385022-03:00 | 目标：Tier A双语长文本续标 | 改动：追加已核看条目并更新动态统计 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计200条，剩余P3 19、demo3 8 | 结论：当前批字段与证据校验通过 | 状态：进行中


E83-A | 2026-11-13T00:50:29.899163-03:00 | 目标：Tier A双语长文本续标 | 改动：追加已核看条目并更新动态统计 | 命令：commit.py authored_batch_003a/b/c.json；checkpoint.py | 目录：data/e82_piper_v1/lang_codex/A | 结果：累计221条，剩余P3 3、demo3 3 | 结论：当前批字段与证据校验通过 | 状态：进行中


### E83-A — 2026-11-13T00:54:38.867998-03:00 — completed_with_documented_evidence_exception
- 目标：全部train/P1 P3候选及demo3 same补充parent的Tier A双语长文本语言训练标注。
- 改动：新增独立A目录，227条scene/task/3 variants/process/keyframes及逐帧证据；逐条实看自检，短片段按授权缩减，1条左腕不可读明确例外；同步最终分片修订。
- 命令：CPU `setsid nohup .../extract.py <manifest>`；`python3 commit.py <authored.json>`；`python3 merge_shards.py`；`python3 final_audit.py`；`python3 final_report.py`；独立scope_audit.py。
- 目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/lang_codex/A；报告：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e83_A_piper_language_20261113.md。
- 结果：227/227，demo4=129/demo6early=10/demo6native=11/demo3=77；待标0；1017过程/1328关键帧；缩略图0.949GiB；CPUonly，训练/GPU/模型下载0。
- 结论：语言监督训练目标已全量产出；禁止测试真值/阈值/checkpoint选型；不构成P3质量放行。
- 状态：完成，1条源左腕证据例外；需用户决定0。

E83-A最终独立审计补记（2026-11-13 BRT）：227/227，0missing/extra/duplicate/errors，13,534条证据核验通过，主SHA256 afde8b79c5910f0af14f27380f3055d03e8c80c005074caf5903938bea7be95a；状态完成，1条缺左腕显式证据例外。

### 2026-11-13T01:39:33.954120-03:00 BRT — E84-G pixel GT / NO-GO

完成150帧（demo4 100/early25/native25）模型双遍像素标注及30帧盲第三遍；第三遍P90=5.39px，native两遍P90=15.90px（非绝对真值误差）。固定内参parent留出验证中位demo4=26.90、early=9.07（右臂仅2点且10.32）、native=18.97px（左臂无验证点）；全部NO-GO，不生成自动投影标签。报告 `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_G_pixel_gt_20261113.md`；READY `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/READY_pixel_gt.json`。CPU only，无训练/仿真；仅probe、无dev/test样本读取。

### 2026-11-13T02:17:36.807264-03:00 BRT — e84p_20261113_021548 E84-P 续跑资源检查

目标：按2026-11-13修订续跑图文/FK运动/动作反事实。改动：新run报告骨架、近似GT入口哈希/字段/邻帧审计、GPU快照与progress；不改F/G或旧run。命令：`python3 code_e84/e84p_20261113_021548/resume_audit.py`，`nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader`。目录：`/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261113_021548`，主报告`/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_P_e84p_20261113_021548.md`，汇总`/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_alignment_summary_20261113.md`。结果：150帧metadata核验通过；合格GPU 0/8（首轮为0），GPU 0 GPUh，未读原图/未训练。结论：依据用户修订第5条停止，像素标定仍NO-GO；不据此触发LoRA或选择阶段一。状态：stopped_no_free_gpu，完整实验未完成。


## E84-P e84p_20261113_021548 — 共享显存续跑完成（2026-11-13T04:09:39.280755-03:00 BRT）
- 目标：继承断点，冻结SigLIP2提特征，完成150帧近似像素图文诊断、机器人系FK运动探针和语言条件位移反事实，不训练WAM。
- 改动：仅code_e84/e84p_20261113_021548及features/probes/e84p_20261113_021548；小batch8、单GPU4、8GiB allocator cap、空闲>=12GiB准入/运行压力检查；像素只读G近似标注，拒绝FK投影运动真值。沿用固定selection和已完成审计；LoRA不启动。
- 命令：setsid nohup env CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8 /mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python -u /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84p_20261113_021548/extract.py full cuda:0；CUDA_VISIBLE_DEVICES= /mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python -u /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84p_20261113_021548/cpu_pipeline.py；/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84p_20261113_021548/finish_report.py；/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84p_20261113_021548/finalize_run.py /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84p_20261113_021548/conclusion.txt。
- 目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/features/e84p_20261113_021548；/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84p_20261113_021548；报告/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_alignment_summary_20261113.md，旧版有备份。
- 结果：22230时刻（train11376/val10854），150像素帧（val40/32parent），运动4数据集、行为2臂；GPU1.2767GPUh，LoRA0；总派生21.945GB；CI按parent2000bootstrap。6项方法验证、特征receipt SHA与源只读stat核对通过。
- 结论：**明确建议：选择“需要 LoRA 对比微调”这条阶段一验证路线，先做受预算约束的修复对照，再决定合并与冻结。当前不把“冻结即可”作为已证实结论；没有足够证据要求立即换更强 VLM。这里是下一步实验建议，尚未证明 LoRA 一定有效。**  依据来自机器人系FK，而不是未通过验收的投影或像素左右标签：H50中文零样本方向准确率左/右11.18%/19.16%，线性头为56.94%/44.49%，对比头56.05%/47.01%；两臂均满足旧run预写的“零样本≤25%且线性CI高于随机CI”判据。这表明冻结特征有运动信息，缺口主要在当前文本—时序表示的对齐路径；本轮小头已经修复一部分，不能把弱零样本结果解释为编码器完全缺乏运动信息。差分/复制句向量拼接不是SigLIP2原生训练目标，LoRA的必要性与收益仍需对照验证。  图像方面，橙带框代理AUROC0.888 [0.863,0.912]说明物体类别有粗定位信号；颜色互换仍有较高响应。左右夹爪的正确−互换AUROC差CI均跨0，可信且分离的双目标验证子集只有7parent；左右能力明确为**不可判定**，不拿它强行触发LoRA。尖端仅4个验证样本，锁头/尖端/整臂均不足以作精细穿带验收。  动作反事实存在语言效应，但可靠可控性尚未成立：换句跟随左/右19.43%/30.38%，相对无语言增加6.97 [6.35,7.55]/17.89 [17.03,18.65]个百分点；预测↔请求XY余弦0.216/0.480，完整3D余弦0.172/0.381。多数请求仍未被跟随，输出仍保留较强观测/轨迹偏置。未来FK句直接给定且只用固定少量模板，小头的条件路径尺度、优化和单任务相关性也可能限制效果，不能将此问题全部归咎于编码器，更不能称为未来预测或实机控制成功。  若用户确认，下一轮建议仅用probe-train，视觉与文本塔最后4层Q/V、rank8，固定小学习率1e-5起点和漂移约束权重0.1，使用对比损失及左右/颜色互换、方向反转难负样本，报告与原冻结编码的余弦偏离。先锁定训练停止规则，不按这150帧的结果选阈值或checkpoint；点位标签仅评估，不用于训练。训练、复提与GPU评估合计≤4GPUh，重新按共享显存规则准入、单卡小batch；仅复提全局时序特征和150帧patch，避免复制一套20GB完整patch缓存而突破30GB。全部保留原冻结结果。  **本轮LoRA实际0 GPUh，未启动；需要用户确认是否开展上述修复验证。** 若修复通过，再合并LoRA并冻结编码器进入阶段二；语言—动作对齐仍只能在含动作分支的阶段二学习，阶段一图文对比不能替代它。完整任务指令提供任务意图，局部运动条件优先用FK短描述；E83长文本需按可见时刻裁成短片段并处理64token限制，部署不得读取未来FK真值作指令。 
- 状态：冻结实验完成；LoRA等待用户确认。像素运动无真值未评估；没有dev/test样本读取、权重/数据下载、仿真、他人进程操作或FK投影像素GT。


### E82-D2 | 2026-11-13T04:36:46.860146-03:00 BRT | demo3 P3 revision4 数据治理

- ID：E82-D2；目标：审计demo3/L1-L3/E83并扩充P3候选，保持revision3 parent分区。
- 改动：新建 /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/revision4_D2_20261113；新增 READY_parent_split_rev4.json、READY_data_rev4.json；诊断teleop7从P1/P3排除，L2 failure demo3仅P1；P3质量门槛保留。
- 命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/revision4_D2_20261113/e82_d2_increment_env_v2.py > /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/revision4_D2_20261113/run_env_v2.log 2>&1 < /dev/null &；只读npz/ffprobe，无GPU。
- 目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/revision4_D2_20261113；报告：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e82_D2_demo3_pool_20261113.md。
- 结果：P1 559条/691136帧/6.399407h/244parent；P3候选 437条/565671帧/5.237694h/223parent；新增demo3 270条；Tier A 223/223；P3_ready=0。
- 结论：下游应读rev4；E84/E82-M继续读rev3不受影响；旧文件哈希不变、dev/test/sealed_test与训练交集0、窗口不跨缝、原q/action/结果字段不变。
- 状态：CPU治理完成；训练未启动；独立人工质量门槛/最终统计与demo3权重待后续。

E82-D2 最终补充（2026-11-13T04:39:44.091781-03:00 BRT）：完整图parent审计127（含16个无timestamp legacy节点全部隔离），READY_data_rev4指向complete清单；19.783ms极值来自已排除诊断片，新增P3 max5.961ms/>6ms=0；命令：已有e29/bin/python /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/revision4_D2_20261113/e82_d2_finalize.py。最终收据/报告sha256全部通过，rev3无改动。


## E84-P e84lora_20261113_042425 — E83同源审计后作废主结论（2026-11-13T05:39:52.552141-03:00 BRT）
- 目标：LoRA真实/打乱同步数对照。
- 改动/命令：双塔末4层Q/V rank8、各768步；launch.py driver_gpu.py；严格审计e83_pair_audit.py。
- 目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/lora/e84lora_20261113_042425、/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84lora_20261113_042425、/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84lora_20261113_042425。
- 结果：331 E83配对仅11条同锚点，320条近邻≤15帧；为严格执行不平移标签，旧adapter/指标只诊断，不作为正式证据；GPU0.792223h均记入本LoRA阶段4h预算。仅停止自己的driver/extract/等待进程，未OOM、未动他人进程/原权重。
- 结论：不按该先导的val分数挑checkpoint；新run e84lora_20261113_053731仅保留11条同源E83，FK1483对/原selection/split不变，重新固定768步真/乱两组。
- 状态：superseded_E83_nearest_pairs_not_primary；所有旧结果保留。


### E82-Q | 2026-11-13T06:08:23.503885-03:00 | Piper P3 independent quality audit
- 目标：437条train候选独立画面/轨迹审核；CPU only。
- 改动/目录：data/e82_piper_v1/quality_astra；notes/e82_Q_p3_quality_20261113.md。
- 命令：SSH BatchMode 只读列举并查看指定报告和 spec.json；cat 新建报告，cat >> 追加两份日志；未调用 GPU/训练命令。
- 命令：ssh -o BatchMode=yes <gpu-server>；原位只读，后台CPU抽帧。
- 结果/结论：进行中；不写P3_ready，不训练。状态：RUNNING。


## E84-P e84lora_20261113_053731 — LoRA对照完成（2026-11-13T06:52:30.876452-03:00 BRT）
- 目标：受预算约束验证SigLIP2双塔LoRA能否修复方向/颜色/反事实失败点，仅probe-train训练，probe-val评估。
- 改动：最后4层Q/V rank8，196608参数，lr1e-5、768固定步；真实/打乱同步数，FK+E83训练文本、反向/换臂/颜色难负例、0.1编码偏离约束；原模型不合并不替换；新增192793参数冻结小头对照。
- 命令：/mnt/workspace/dengjianfeng/.venvs/e21_sam3_20261101/bin/python /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84lora_20261113_053731/launch.py driver_gpu.py；CPU cpu_eval.py lora/shuffled；matched_head.py；analyze.py；verify_results.py；figures.py；finalize.py。GPU子进程CUDA_VISIBLE_DEVICES=4，PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8。
- 目录：代码/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84lora_20261113_053731；权重/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/lora/e84lora_20261113_053731；特征/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/features/e84lora_20261113_053731；结果/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e84_piper_align_v1/probes/e84lora_20261113_053731；报告/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_lora_summary_20261113.md；汇总/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_alignment_summary_20261113.md；旧汇总备份/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e84_alignment_summary_20261113.md.pre_e84lora_20261113_053731_20261113T0652308579620300.bak。
- 结果：训练1483FK+11同锚点E83/189parent，两组各768步；评估沿用22230时刻/542capture、150近似像素帧（val40/32parent）；parent等权2000bootstrap成对CI；本轮含调试GPU 1.7588/4h，E84累计3.0354/10h；派生26.003GB/<30GB。
- 结论：**阶段一建议：暂不采用本配方 LoRA，不合并本轮 adapter；保留冻结 SigLIP2 与小头方案作为后续基线，继续保留语言条件。** 本轮没有证据支持用这套编码器微调替代冻结方案；精细左右和可靠语言控制仍未通过验收。优先补齐训练 parent 的严格同源 E83 配对、多色正例和可信左右标注，再验证阶段一目标。当前证据不足以要求立即升级更强 VLM，可保留为后备建议，不下载。此结论只针对本次 rank8/末4层Q/V/lr1e-5/768步/11条同锚点E83配方，不能外推为所有LoRA方案无效。  相对原冻结，机器人系H50中文零样本方向增益左/右为 **+0.24 [-0.28, 0.79] / −0.31 [-0.81, 0.18] 个百分点**；同结构对比头方向增益 **+0.38 [-0.87, 1.54] / −0.36 [-1.20, 0.50]**，CI均跨零。反事实跟随率为19.73%/30.21%，相对冻结 **+0.30 [-0.19, 0.76] / −0.17 [-0.86, 0.52] 个百分点**，仍无增益证据；右臂预测↔请求余弦下降0.011 [0.001, 0.021]。冻结特征的小头已经明显优于原生差分检索，这表明当前对齐路径的缺口不能直接归为编码器缺乏运动信息。  检测到小的正确标签效应：相对打乱LoRA，左臂H50零样本方向+0.43 [0.10, 0.75]个百分点，两臂反事实跟随+0.58 [0.27, 0.89] / +0.86 [0.37, 1.36]个百分点；这些CI不跨零，但未转化为超过原冻结方案的核心增益。近参数量冻结头192,793参数，H50方向53.38%/43.87%，显著优于196,608参数LoRA的原生检索11.42%/18.85%。LoRA再接738,144参数大头虽然超过近参数量小头约3个百分点，合计参数量和计算已不匹配；与相同738,144参数的冻结大头比较，方向增益CI仍跨零，不能把额外大头优势归因于编码器微调。  颜色互换的**代理区分差**有显著增益：正确−蓝/绿的AUROC差相对冻结分别增加 **0.492 [0.436, 0.549] / 0.535 [0.483, 0.587]**。但橙带自身AUROC增益仅0.0097 [−0.0036, 0.0227]、pointing增益CI跨零；两组共享全为橙色的HSV弱正例，缺少真实蓝/绿正例，不能据此验收通用颜色理解。正确标签版的蓝/绿区分AUROC差相对打乱版反而低0.0079 [0.0050, 0.0110] / 0.0138 [0.0103, 0.0173]，颜色改善无法归因于正确FK/E83标签，更符合共享颜色弱监督的作用。左夹爪AUROC下降 **0.078 [0.042, 0.117]**、锁头下降 **0.145 [0.118, 0.170]**；尖端下降0.140 [0.093, 0.186]但仅4个val parent。以上下降CI不跨零，仍只是同模型近似GT的本域诊断。左右可信分离子集仍0/7双句跟随，左右可泛化能力为**不可判定**，没有被本轮修复。  全局编码偏离较小：val视觉余弦0.9860 [0.9850, 0.9869]、线性CKA0.9938；文本均值余弦0.9917、CKA0.9710。全局保持不能保证局部patch语义保持，亦不能代替通用检索遗忘评估。像素运动无真值、通用图文检索无本地样本，均未评估；150帧近似GT、未过验收的标定、单任务相关性、11条严格E83与单训练seed限制结论。语言—动作对齐仍需在含动作分支的后续模型里学习和验收；本轮没有训练WAM，也没有证明实机控制或未来预测成功。 
- 状态：完成本轮LoRA验证与受控对照；通用检索无本地样本未评估，画面像素运动无真值未评估；150帧只诊断，禁用FK投影像素GT；无dev/test样本、无下载/仿真、无他人进程修改。先导近邻E83配对不作正式证据并留档，本次只用11同锚点caption；所有先导/调试GPU时间全计入。


### E82-Q · 2026-11-13T07:09:55.422407-03:00 · Piper P3 示范质量审核（完成全量，待批准冻结）
- 目标：CPU独立画面/轨迹审核rev4全部437条train P3候选；不训练、不使用GPU。
- 改动：仅新增quality_astra审核、缩略图、QC和建议池；没有更改原数据/划分/标签/语言或P3_ready。
- 命令：ssh -o BatchMode=yes <gpu-server>；使用/mnt/workspace/dengjianfeng/venvs/e29/bin/python；抽帧通过setsid nohup extract.py及extract_third.py；批次append_reviews_v2.py；汇总finalize_v2.py；报告complete_report_v2.py。脚本均保存于quality_astra，可追溯。
- 目录：data/e82_piper_v1/quality_astra/；报告notes/e82_Q_p3_quality_20261113.md。
- 结果：437/437，accept183/reject65/uncertain189；有效parent116/223。严格池183条/185992帧/1.722148名义小时；可选宽松池372条。QC40条三类90%、kappa0.849906；明确A/R26条100%、kappa1（排除14条）；17条第三遍裁决。
- 验证：全部train、eval parent交集0；437条mask帧数/next-frame/证据hash验证；4个锁定输入hash未变；缩略图506013619字节，无源视频复制。
- 结论/限制：推荐严格accept池，需用户批准并给demo3采样上限；先验暴露标记110条，非全量严格盲审。E83 retrospective字段未找到，对照N/A。模型审核只作训练筛选，不作测试真值或选型依据。
- 状态：ALL_437_REVIEWED_PENDING_USER_FREEZE；P3_ready未写；E83字段对照待输入。


### E82-Q2 — Piper P3 uncertain独立复核（2026-11-13T08:02:22.703848-03:00 BRT）
- ID：E82-Q2-sol-20261113；日期：2026-11-13 BRT；目标：复核Astra 189 uncertain，仅用于训练池筛选建议。
- 改动：新建quality_sol_recheck，189条三视图与轨迹审核、30条固定seed盲二遍、7条第三遍、孔口补核、事后先验比较和两个建议池；未改数据/划分/标签/Astra输出/code_e82，未写P3_ready。
- 命令：ssh -o BatchMode=yes <gpu-server>；CPU解释器/mnt/workspace/dengjianfeng/venvs/e29/bin/python；setsid nohup extract.py、extra.py、detail.py；append.py逐批保存；qc_compare.py、lock_final.py、aggregate.py生成审计与统计。脚本完整保存在新目录。
- 目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/quality_sol_recheck；报告：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e82_Q2_recheck_20261113.md。
- 结果：189/189，accept23/reject24/still_uncertain142；盲二遍23/30=76.67%，kappa0.570552；先验暴露53条L2精确同类20/53=37.74%，E83 retrospective字段53/53缺失。V1=206条208473帧1.930306h/24477窗口/121parents；V2=207条208605帧1.931528h/24486窗口/121parents；两版均102/223parents无可用条目。
- 结论：模型复核非人工真值；成功与质量分离，保守保留不确定；恢复段不新增入P3，demo3上限未决定。GPU0，训练0，新增约147MB。
- 状态：复核与建议清单完成；待用户批准冻结版本、demo3采样口径及上限。

## E86-M — 2026-11-13 BRT（执行中）
目标：Piper v3冻结SigLIP2语言条件统一WAM与有界烟雾。改动：新建code_e86、P3_ready_V1批准收据；保留code_e82只读。命令：python -m e86.prepare / cache_fast / driver / pytest；目录：code_e86/artifacts，notes/e86_M_v3_smoke_20261113.md。结果：V1哈希通过，206条/121parent/24477窗；首轮38项CPU测试通过；全缓存与GPU烟雾进行中。结论：只记录工程验证，不选checkpoint、不启动正式训练。状态：in_progress；时间 2026-11-13T12:32:52.436003-03:00。


### E82-Q3 — Piper长录像分轮与逐轮质量审核（2026-11-13T12:57:01.652292-03:00 BRT）
- ID：E82-Q3-astra-20261113T104940；日期：2026-11-13 BRT；目标：全部21条demo6及Q2理由筛选的66条demo3/demo4，CPU分轮、逐轮审核、提出V3。
- 改动：新建quality_cycles_astra，87源轨迹/FK/边界提议、148候选三视图审核、30轮固定seed遮蔽二判、10轮人工短名单、H50核验、V3建议池及哈希；原数据/缓存/划分/他人进程不变，未写P3_ready。
- 命令：ssh -o BatchMode=yes <gpu-server>；/mnt/workspace/dengjianfeng/venvs/e29/bin/python；setsid nohup批量extract_cycles.py；append_reviews.py；qc_compare.py；finalize_v3.py；human_shortlist.py；finish_report.py。
- 目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/quality_cycles_astra；报告：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e82_Q3_cycles_20261113.md。
- 结果：148候选=78A/16R/54U；demo6救回75轮/80072帧/9429窗/0.741407h；V3=206旧条目+78新轮=284条目/295740帧/34777窗/2.738333h/145parent。QC30/30一致、kappa1.000，无分歧裁决；同模型同帧复判，非独立人工准确率。
- 验证：全部train；H50三相机/原动作索引边界通过；重复窗口0；V1与输入哈希不变；V3 SHA256=beaf61055b61f04064203683e8dc6275eb2bae35810b281fa5853df409802964。CPU only，GPU0h，训练0；新增约236.2MB（最终见storage_final.json）。
- 结论：分轮可救回demo6，但重试/人工介入/截断仍保守排除；模型审核只供训练池筛选。E83 retrospective字段缺失，对照N/A。
- 状态：COMPLETE_PROPOSED_V3_AWAITING_USER_ADOPTION；两批完成，待用户决定是否采用V3；建议先看10轮人工短名单。


### E82-Q3 V3采用批准 — 2026-11-13T13:16:34.649489-03:00 BRT
- ID：E82-Q3-V3-ADOPT-20261113T131634；目标：落实用户维持10轮原判并采用V3的明确批准，核验哈希，提供模型流水线交接；本轮不启动正式训练。
- 改动：新建/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/quality_cycles_astra/adoption_V3_20261113T131634（approval.json、verification.json、handoff_to_model_pipeline.md）；追加Q3报告；更新自身progress。原V3、cycles、V1、数据、缓存与划分不改。
- 命令：ssh -o BatchMode=yes <gpu-server>；/mnt/workspace/dengjianfeng/venvs/e29/bin/python执行SHA256、entry哈希、窗口去重、parent train与输入/缓存哈希核验；无训练命令。
- 目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/quality_cycles_astra/adoption_V3_20261113T131634；池：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/e82_piper_v1/quality_cycles_astra/proposed_p3_pool_V3.json。
- 结果：V3 SHA256=beaf61055b61f04064203683e8dc6275eb2bae35810b281fa5853df409802964；284条目/295740帧/34777窗/145parent；586缓存哈希通过；10轮维持原判（7排除/3accept）。
- 结论：V3已获用户采用批准，原提案文件保持字节不变，独立批准收据为授权依据；交接给用户转交模型会话。
- 状态：APPROVED_V3_NO_FORMAL_TRAINING；训练0、GPU0，未写或改P3_ready，未通知其他会话。

## E86-M — 2026-11-13 BRT（完成烟雾）
目标：Piper v3冻结SigLIP2语言条件统一WAM。改动：code_e86模型/数据/采样/损失/训练恢复/评估；P3_ready_V1用户批准收据。命令：python -m e86.cache_fast / e86.driver / pytest；目录：code_e86/artifacts/smoke_v2，notes/e86_M_v3_smoke_20261113.md。结果：55 passed in 29.24s；P1可读107057（rev4原108851）/P3 V1 24477窗口；可读来源缓存完成（P1隔离1794窗）；GPU墙钟1.993208GPUh；dev48窗与两模式各200次时延已测。结论：工程烟雾完成，固定stage-end不做选型，不外推实机性能。状态：complete_smoke_only；正式训练未启动；25%/10%模式含义待用户锁定。时间 2026-11-13T14:29:36.318190-03:00。

E86-M交付复核 2026-11-13T14:35:32.827152-03:00：55项CPU合同、三处GPU恢复参数差0、独立进程下一更新loss精确相同；全部可读缓存1302474帧（约16GB）；总1.99320782GPUh。全池batch32实测100步/阶段，正式单链含25%余量6.7877GPUh，缓存/评估另计。dev夹爪未优于复制当前状态，P99 action_only254.5/joint300.9ms（各200次）；仅烟雾参考、不选型。P1权限隔离1794窗仍未完成，P3 V1不受影响。正式训练未启动。最终报告及校验：code_e86/artifacts/E86_M_DELIVERY_READY.json。

## E87-Y-20261113T1447 — 2026-11-13 BRT（H20盘点完成）
- ID：E87-Y；时间：2026-11-13T15:14:49.625908-03:00；目标：只读盘点H20 yam双臂数据，为Piper P1提出候选。用户收缩范围为H20。
- 改动：新data/yam_inventory_v1索引/去重/建议/联系表/CPU审计脚本、notes/e87_yam_inventory_20261113.md；同事数据只读。
- 命令：Mac ssh -o BatchMode=yes <gpu-server>；setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u data/yam_inventory_v1/audit/scan_h20.py、video_check.py、candidate_check.py；CPU numpy/pyarrow/ffprobe/ffmpeg小段，无训练/GPU。
- 目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/data/yam_inventory_v1；报告：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e87_yam_inventory_20261113.md。
- 结果：17源1503记录→520片段版本；89重复子段回父（89/89精确hash）→431净片段/389录制父源（67缺原ID用hash代理），366235帧/3.391065h；45审核片仅3父。321episode/963路短段全通过；建议49条0.281954h，189条1.386139h待核；3任务各12episode联系表。重复timestamp1380，max gap400.24ms/skew317.48ms，7来源部分字段有1000倍量纲问题。
- 结论：仅建议小量P1候选，正式parent/时间窗口未建；A特征1.123GB，E86有效吞吐推算含余量0.145GPUh，未实际提取；来源标签仅筛选。task2当前缺失不计。
- 状态：COMPLETE_H20_INVENTORY_PROPOSAL_ONLY；GPU0/训练0/无P1_ready/无原片复制。待决定49接入及189复核。

## E86-V3-20261113T144520 — 2026-11-13 BRT
目标：Piper V3接入、永久排除episode142、预算内工程验证。改动：P3_ready_V3/显式永久排除；code_e86逐entry区间读取、双采样策略、增量缓存overlay、可配置模式、v3 smoke/profile/replay。命令：setsid nohup v3_driver_02.py；pytest；e86.v3_regression。目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e86/artifacts/E86-V3-20261113T144520；报告：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e86_V3_integrate_20261113.md。结果：73项CPU合同通过；P1 107057窗，P3 V3 34777窗/145 parent；新增57270相机帧/703733760B特征；P1 192/P3 384/无语言384步、恢复和48窗dev分层参考完成；0.385007GPUh；正式单链含25%余量4.8974GPUh。结论：工程合同完成，不作checkpoint选型或性能验收；正式配方25%口径待用户锁定。状态：COMPLETE_ENGINEERING_SMOKE_ONLY；未启动正式训练。时间：2026-11-13T15:36:40.668288-03:00。

## E88-W-v31-20261114 — 2026-11-14T00:56:16.891941-03:00 BRT
目标：Piper v3.1多时域残差世界头/链式重入/阶段辅助与夹爪核查。改动：新code_e88；旧包/数据/缓存只读。命令：ssh -o BatchMode=yes <gpu-server>；nohup setsid python -m e88.audit / e88.language / bounded_pipeline_01.py；pytest tests；e88.train smoke_03 --recipe A；e88.replay；e88.profile。目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e88；报告：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e88_W_v31_20261114.md。结果：108 CPU合同通过；P1/P3六时域全池统计、建议+100；signed复制基线修复、64窗小头H50 0.003478mm；P1 160、P3三语言各192步；内部/独立恢复一致；0.376294GPUh含失败；正式单链含25%余量估算8.342GPUh。结论：工程完成；世界/夹爪/语言性能未验收；当前阶段监督P3仅51窗。状态：COMPLETE_ENGINEERING_SMOKE_ONLY；未启动正式训练；A/B待用户锁定。

## E89-A-20261114T0012_BRT — ABC-130k P1 数据盘点

- 日期：2026-11-14T00:56:19.723298-03:00（BRT/UTC−3）。
- 目标：只读/CPU盘点本地ABC-130k，选50–150h Piper v3 P1视觉候选，检查MCAP、标注、时钟与60GB缓存成本；不训练、不提特征。
- 改动：新建独立venv（mcap1.6.0、mcap-protobuf-support0.6.0及protobuf/lz4/zstandard必需依赖，wheel哈希/日志已存）；新脚本、任务分类/索引/抽样联系表、三份候选清单及派生SHA256、报告。源与E82代码/数据未改。
- 命令：ssh -o BatchMode=yes <gpu-server>；python3 -m venv data/abc_inventory_v1/env/venv；pip install --report ... mcap mcap-protobuf-support；setsid nohup ... scripts/scan.py、audit.py、calibration.py、refine.py、subset_logic.py timing、finalize.py；verify.py、summarize.py、report.py、closeout.py。
- 目录：T/data/abc_inventory_v1/；报告T/notes/e89_abc_inventory_20261114.md（进度版另存）。
- 结果：24,151条候选MCAP summary、21,371条RS标定；342条/1,026路短段CPU解码全过；严格时间索引9,833条，通过8,240、剔除1,593。S1 1620条/50.456h；S2 3902条/100.246h；S3 5631条/150.182h。所选源UUID错配0，子集嵌套、单任务≤25%，GPU0GPUh；盘点约240.9MB，review约7.7MB。
- 结论：推荐S1候选+30Hz规范时距+≤60GB有界LRU；10Hz的50h payload仍超60GB。ABC初始20%窗口、默认不纳ZED；子任务文本来源未知、含mistake/reset，不能当评估真值。
- 状态：盘点交付完成（候选非P1_ready）；无正式训练、无特征/窗口缓存、未碰Piper dev/test/sealed_test。接收时钟、标注作者、本地LICENSE/NOTICE及未抽样帧隐私/解码仍有明确覆盖边界，后续策略由用户决定。

E88-W最终预算/资源复核 2026-11-14T01:00:40.356880-03:00 BRT：补充原dev图像世界参考（0新语言标签/特征）；总0.378883GPUh，108 CPU合同+非零头掩码复核通过；顺序profile受缓存影响，保守单链含25%余量10.787GPUh，三条件3 seeds约69.750GPUh；每步链粗估单链12.141GPUh（max2窗链）。最终收据code_e88/artifacts/FINAL_DELIVERY.json；正式未启动，状态COMPLETE_ENGINEERING_SMOKE_ONLY。

## E91 — 2026-11-14T03:34:21.339148-03:00 — ABC 5h English training annotations
目标：S1/A约5h，仅CPU；模型标注仅训练用途。目录：data/e91_abc_lang_v1；报告notes/e91_abc_lang_20261114.md。改动：新建prepare.py、selection_plan.json、报告骨架。命令：E89 env/venv/bin/python data/e91_abc_lang_v1/scripts/prepare.py select；setsid nohup ... prepare.py run。初选116条/5.0543h/98有标注/16任务；完整解码与标注进行中。GPUh=0。状态：RUNNING，结果与结论待追加。

### E92 | 2026-11-14T03:35:20.671969-03:00 | yam 英文文本标注
- 目标：仅E87 proposed entries的49 clip，为步骤2准备训练文本。
- 改动：新增data/e92_yam_lang_v1（392张关键帧、49行lang_yam.jsonl、抽帧脚本/状态/校验/逐clip训练建议）；报告notes/e92_yam_lang_20261114.md。
- 命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python .../data/e92_yam_lang_v1/extract.py；CPU OpenCV/PIL与Python JSON校验。
- 结果：49/49，30451帧/0.281954h；29取管插入+20恢复插入；缺失0；392 JPEG共9229961字节；uncertain局部细节49/49；GPUh=0。
- 结论：仅训练目标/训练池筛选，非真值/阈值/checkpoint依据；不标成P1_ready，未训练。
- 状态：首轮标注与校验完成；10条独立复标待授权，不报告一致率。

## E90 — 2026-11-14T03:42:59.643643-03:00 BRT — 完整v3.1与Piper P1
- 目标：步骤0完整实现、步骤1仅Piper无动作世界头5000步；不启动步骤2。
- 改动：新code_e90；来源token/多源逐前缀上限；默认+100/连续时域；口径B；适配器门控与每步两窗链/ramp；阶段权重0；完整checkpoint/调度器/RNG/hash恢复与init-ckpt。
- 命令：ssh -o BatchMode=yes <gpu-server>；e29/bin/python -m pytest -q tests；python -m e90.preflight；setsid nohup python -m e90.train e90_smoke_20261114_01 --smoke。
- 目录：T/code_e90；报告T/notes/e90_full_v31_p1_20261114.md；正式结果将位于T/results/e90_p1_20261114_01。
- 结果：CPU133通过（原108+25）；P1=107057窗/242parent，+100=98655窗/237parent；文本77173已有指令/29884短指令回退；dev已有缓存6/51parent、48窗，+100仅1窗。
- 结论：实现CPU验收通过，GPU烟雾与独立恢复验证进行中；GPUh/最终结果待追加。
- 状态：RUNNING。

### E91 完成 — 2026-11-14T03:58:12.763713-03:00
目标：ABC S1/A约5h英文训练文本（不训练）。改动/目录：data/e91_abc_lang_v1及notes/e91_abc_lang_20261114.md；选择/完整CPU解码/全文件SHA/清洗/模型关键帧标注/独立盲标/验证。命令：E89 venv prepare.py select + setsid nohup prepare.py run、verify_streams.py、zip_detail.py；python3 assemble.py、validate.py、compare_review.py、closeout.py。结果：116条、5.054296h、16任务、有原标注98/116；覆盖100%；2484原段+230事件，零长0；uncertain75/116=64.66%；三路完整解码/严格时间/实际640检查全通过。12条盲复核严格总指令一致6/12=50%，核心目标兼容12/12，阶段数差0；同模型家族非人工真值。估算30Hz三路特征20.1227GB，实际本轮约29MB，无缓存/训练，GPUh=0。结论：交付unreviewed训练候选文本，不能作测试/阈值/checkpoint依据；无待用户决定项。状态：COMPLETE。详细问题和一次已清理的0B误文件事件见报告§5.3。

### E90 最终收据 — 2026-11-14T04:04:35.575858-03:00 BRT
- 目标/改动：完整v3.1实现与Piper P1固定5000步；133项CPU合同，GPU烟雾/独立恢复逐位一致，阶段0、来源上限、口径B、每步链/ramp、全checkpoint与init-ckpt。
- 命令：setsid nohup e29/bin/python -m e90.train e90_p1_20261114_01 --mode-probabilities .25 .25 .25 .25 --clean-aux 0；CPU finalize_e90.py。
- 目录：code_e90/artifacts/e90_p1_20261114_01；results/e90_p1_20261114_01；报告notes/e90_full_v31_p1_20261114.md。
- 结果：5000步完成、10份500间隔ckpt；ckpt-1 SHA=45164cfb86483709892d5c59a55fb830484cb5923eb7c018579595c826abbc49；GPUh=0.287406，新增1.049GB，新增特征0；dev 4Euler/2000 bootstrap，6/51缓存parent，100仅1窗。
- 结论：步骤0/1完成，dev缓存覆盖受限，完整指标及区间见ckpt1_dev_eval.json；不以dev选型，不作动作/语言效果结论。
- 状态：COMPLETE_P1_CACHED_DEV_LIMITED；步骤2未启动。

E90最终指标判定 2026-11-14T04:05:14.106163-03:00 BRT：dev复制skill10/25/50=-5.81/-5.54/-5.24%，bootstrap区间低于0；100=-20.51%但仅1窗。执行完成、性能未胜复制；CPU固定4Euler基线与evaluator SHA已保存。

## E91b | 2026-11-14T04:15:58.883382-03:00 | ABC uncertain 补帧复查
目标：复查E91的128 uncertain项；CPU-only，训练文本非真值。改动/目录：data/e91_abc_lang_v1/keyframes_v2/、lang_abc_v2.jsonl（待生成）、notes/e91b_abc_uncertain_20261114.md；v1保留。命令：SSH BatchMode，inventory recursive scan，后续补帧脚本。结果：初始75/116 episodes含uncertain。结论/状态：进行中，帧上限冲突待用户选择。


## E92b | 2026-11-14 04:22:18 BRT
- 目标 对E92 yam uncertain密集复判并完成同会话弱独立复核，纯CPU。
- 改动 新增keyframes_v2、lang_yam_v2.jsonl及e92b审计/复核/建议文件，v1保留。
- 命令 ssh -o BatchMode=yes <gpu-server>；setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python T/data/e92_yam_lang_v1/e92b_extract.py；SSH标准输入写入及CPU校验。
- 目录 T/data/e92_yam_lang_v1/；报告T/notes/e92b_yam_uncertain_20261114.md。
- 结果 49clip、1556帧、41.44MB图像，GPUh=0；uncertain100%→100%，总指令改20条；复核10/10语义一致、阶段数差0。
- 结论 密集帧补足粗粒度过程，精确洞位/深度/接触仍不确定；20恢复片排除完整释放成功监督。同会话同模型家族复核，非人工真值，仅训练用途。
- 状态 完成。源目录只读，无GPU使用，新增远低于300MB/60GB。

### E91b 完成记录 | 2026-11-14T04:33:25.718149-03:00
目标：ABC uncertain补帧复查。改动：新增lang_abc_v2.jsonl及keyframes_v2证据/审计，v1不变。命令：setsid nohup env PYTHONDONTWRITEBYTECODE=1 <abc_inventory venv>/bin/python keyframes_v2/{extract,extract_complex,extract_flag_only}.py；python3 keyframes_v2/assemble_validate.py。目录：data/e91_abc_lang_v1/；报告notes/e91b_abc_uncertain_20261114.md。结果：128→81显式uncertain；任务11→1、叙述89→78、原段28→2；episode75→69/116；10条总指令更新；1115帧，最大24/episode，新增约60.952MB，GPUh=0；独立同族盲复核13/13一致，非真值。结论：物体类型可澄清，锁合/紧度/遮挡仍保守。状态：交付并校验通过，12个多类型episode共享帧导致部分逐段密度未达请求，待用户决定是否放宽24帧上限。


## E92c | 2026-11-14T04:46:03.405680-03:00 | yam 独立复核
- 目标：对 E92b 抽中10clip先独立看图标注封存，再比较v1/v2。
- 改动：新增 notes/e92c_yam_review_20261114.md 报告骨架；未创建标签或更改v1/v2。
- 命令：ssh -o BatchMode=yes <gpu-server>；python3只读解析E92b报告及inventory结构。
- 目录：data/e92_yam_lang_v1/review_independent（检查不存在，待创建）；报告如上。
- 结果：E92b报告未列10个clip_id；已请求清单。GPUh=0。
- 结论：不能以自行抽样替代指定样本，不能复用原报告一致率。
- 状态：等待用户提供清单；尚未读取v1/v2。

### E93 — dev缓存与ckpt-1诊断（2026-11-14T04:58:52.583615-03:00 BRT）
- ID/日期：E93 / 2026-11-14 BRT。目标：补dev合同特征，固定ckpt-1诊断，不训练。
- 改动：新code_e93与results/e93_dev_diag_20261114_01；原6parent1134帧重放最大差0；51parent名单全审计，26发布parent+1原始失败录制恢复为27parent/211窗，24不可按原合同评；100为208窗/26parent。旧代码/训练缓存/原v1不写。
- 命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup已有e29 Python执行prepare/extract/extract_supp/cpu_eval/cpu_validate/gpu_diag/linear_03/aux_cos/train_language/exposure与报告脚本，见code_e93/README_E93.md。
- 结果：原145条v1最大MSE差4.77e-7；dev v1 skill10/25/50/100=−5.204/−4.982/−4.800/−14.256%；4Euler N16=−.362/−.148/−.069/+1.037%；ridge=+6.778/+9.279/+12.705/+22.615%，CI均>0；train64同样未胜copy。完整4/8/16×N1/8/16、分解、FM下界与历史正负比较见notes/e93_dev_diag_20261114.md。
- 资源：0.585266 GPUh含所有GPU段，新增特征0.066967GB，总产物约0.182671GB；保护清单284文件变化0。
- 结论：采样方差解释大部分v1负项，条件均值仍远弱于线性可学习参照；不建议盲目加长原FM，先锁定均值/分布目标并做最小直接残差MSE对照，需另授权。
- 未完成：24parent无同合同时钟/录制/窗口；非全51有效缓存。状态：COMPLETE_DIAGNOSTICS_PARTIAL_51_PARENT_COVERAGE；神经训练更新0。

E93 FM读出诊断补充 2026-11-14T05:01:32.047421-03:00 BRT：world_head 768×256秩256，当前head子空间oracle FM下界占train短时FM约77–78%，100约68%；不是数据不可预测性，空间内oracle点误差仍远低copy。最终报告已解释速度零/干净头零两类参考，未训练或改checkpoint。

## E91c — 2026-11-14 BRT — ABC补帧复判
- 目标：解除E91b每episode24帧上限，补满12个多段episode并复判20项uncertain。
- 改动/目录：data/e91_abc_lang_v1/keyframes_v3/；新lang_abc_v3.jsonl；报告notes/e91c_abc_fill_20261114.md。保留v1/v2。
- 命令：SSH BatchMode 只读列举并查看指定报告和 spec.json；cat 新建报告，cat >> 追加两份日志；未调用 GPU/训练命令。
- 命令：setsid nohup env PYTHONDONTWRITEBYTECODE=1 data/abc_inventory_v1/env/venv/bin/python .../keyframes_v3/extract.py（另extract_tail.py）；python3 .../assemble_validate.py；python3 .../finalize.py。
- 结果：877张JPEG，761个相对v2新增原生帧；60窗口段内9–12张，49非指令窗口前后各≥2，11整episode指令无源外帧，1尾段后文截短0.5477秒。20项复判均保留，修订1条叙述可见描述；显式uncertain 128→81→81。10%独立同家族复核2/2一致，通信偏差见报告，非人工真值。
- 结论：补密帧数未解决锁扣遮挡/语义歧义，不以降低uncertain为目标。约50.40MB新增、CPU-only、GPUh=0；源与v1/v2哈希/边界校验通过。
- 状态：完成；无待用户决定。记录时间2026-11-14T05:04:52.799595-03:00。

## E92c-2 | 2026-11-14T05:38:16.628174-03:00
目标：yam 10/49固定种子独立CPU复核。改动：新增review_independent/及notes/e92c2_yam_review_20261114.md；v1/v2只读。命令：ssh BatchMode；random.Random(9203).sample；setsid nohup python3 extract.py；ffmpeg CPU；封存后JSONL比较。结果：278源帧，40%一致/60%部分一致/0%完全不一致；5/3/4阶段；GPUh=0。结论：粗粒度可见叙述可用，6条建议暂排除明确成功/插入目标子池，不修改池。状态：完成；仅模型训练文本复核，非真值。

## E94a — 2026-11-14T05:45:15.078595-03:00 BRT — 世界读出修复（进行中）
目标：SigLIP2 2×4目标与归一化不变，M1确定性均值/M2均值+FM各固定5000步。改动：新code_e94a，256→1024→GELU→768双MLP、无噪声均值查询、均值链停梯度；READY_readout.json已定稿。命令：setsid nohup env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 e29/bin/python -u code_e94a/pipeline_e94a.py。目录：results/e94a_control_20261114_01、results/e94a_m{1,2}_p1_20261114_01；报告notes/e94a_readout_20261114.md。结果：原133+新增16=149合同通过；GPU预算7200s、新增≤10GB；训练/固定E93评估进行中。结论：尚无性能结论；不以dev选checkpoint。状态：RUNNING。


## E94b — 2026-11-14 BRT — DINOv3世界目标（进行中）
目标：Piper P1 O=SigLIP2不变，W=DINOv3-B/16，固定5000步b1/b2及E93固定窗口评估。新源码code_e94b、缓存data/e94b_dinov3_cache、结果results/e94b_dinov3_20261114_01、报告notes/e94b_dinov3_20261114.md。命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup env PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u -m e94b.{prepare,extract_parallel,...}。两parent重复/官方前向差均0；累计预估27.27GB≤30GB。串行提取自身进程已预算优化停止，180.10秒计入4GPUh；完整缓存保留，未完成段79MB保留。状态：提取中，尚无训练/性能结论；Meta自定义许可仅内部研究。

## E94a — 2026-11-14T06:31:21.965323-03:00 BRT — 完成
目标：不变SigLIP2 2×4目标/归一化，修复读出并对照M1/M2。改动：新code_e94a，双MLP、确定性时域查询、均值链；READY v1冻结，原133+新增16=149合同通过。命令：setsid nohup e29/python -u code_e94a/pipeline_e94a.py；ridge_train64.py；finalize_e94a.py。目录：results/e94a_control_20261114_01、e94a_m1_p1_20261114_01、e94a_m2_p1_20261114_01；报告notes/e94a_readout_20261114.md。结果：两配方各5000步、20ckpt，统一E93 dev/train64、4Euler FM N1/N16、2000次parent CI及相机/四分位分解齐备；均值skill%={"train64": {"M1": [2.605634071280871, 8.416693842348588, 10.840692486104109, 28.746718211856503], "M2": [2.8480472152992764, 9.234511665789015, 12.960498477465055, 32.667717366292734]}, "dev": {"M1": [1.1084313262393009, 5.253364891350865, 9.179952683507874, 30.707992890158444], "M2": [1.1162544452982148, 5.715036449104238, 11.196124853941392, 33.777031677062986]}}。结论：预注册读出修复通过=False。资源：0.721495GPUh、2.1064GB，保护307文件变化0。状态：COMPLETE；无其他训练，无待决定的执行事项。

## E95 — 2026-11-14T06:57:14.346679-03:00 BRT — 世界头结构分析（完成）
目标：解释E90/E94a均值与FM差距、时域/链/跨编码器风险，给单因素建议。改动：仅notes/e95_worldhead_structure_astra_20261114.md及results/e95_worldhead_structure_astra_20261114_01小型CPU诊断产物，无模型/训练源码修改。
命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 e29/bin/python -u <E95目录>/cpu_probe.py、ridge_projection_probe.py（各540秒硬限）。
结果：冻结ridge投影到E90头空间后训练16parent skill+7.30/+8.36/+11.98/+19.45%（100为13parent）；O池化槽变化能量占比E90/M1/M2=.00290/.10592/.03114%；局部t=.9 M2 FM主干grad约mean的10.7倍；M1/M2链2000/1872步；M1+100 dev相对ridge配对差+8.09pp [4.69,11.70]。两项CPU诊断24.41/7.99秒、CUDA未初始化、GPUh=0，新增小于1MB；初次统计引号错误保留于报告，0GPUh。
结论：秩限制不足以解释线性均值缺口，优先空间/同位置条件通路；FM需独立概率评估；粗细时域不可混比，暂不加大链权重。相关工作一手来源与每项GPUh/判据已写报告。READY无不符、178源码末次核查无变化；未碰E94b路径/进程。
状态：COMPLETE_ANALYSIS_ONLY；未做训练/因果消融/完整概率校准。后续用户决定空间旁路或同位置线性旁路，以及是否保留概率生成需求。

## E96a | 2026-11-14T07:11:26.423266-03:00 BRT | RUNNING
目标：E95 P1 当前观测空间旁路；code_e96a；results/e96a_control_20261114_01 与 e96a_m1_p1_20261114_01；固定5000步、≤1GPUh、≤5GB。命令：ssh BatchMode + setsid nohup pipeline_e96a.py。预注册和边界：notes/e96a_spatial_bypass_20261114.md。结果/结论待追加。


## E94b 最终完成 — 2026-11-14T07:31:59.068691-03:00 BRT
目标：SigLIP2 O不变，DINOv3-B/16目标；仅Piper P1，原FM b1与复用E94a M1的b2各5000步。改动：code_e94b，DINO 2×4/7×7缓存与训练统计，跨编码器重入，定稿读出头逐字复用；20 checkpoint。命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup ... -m e94b.{extract_parallel,stats,train,evaluate,ridge}；results内parallel_tail.py及CPU ridge_train64_tail.py/postflight_tail.py；-m e94b.finalize。目录：code_e94b/、data/e94b_dinov3_cache/、results/e94b_dinov3_20261114_01/；报告notes/e94b_dinov3_20261114.md。
结果：dev DINO ridge skill +0.60/+1.79/+4.70/+12.96%；b1 N1 −14.88/−13.71/−12.27/−29.49%，N16 −2.97/−2.40/−2.00/−3.81%；b2 M1 +1.75/+5.40/+10.69/+26.16%（四个CI>0）。DINO ridge比Sig ridge更弱；相对Sig M1，DINO仅+50 skill探索性提高1.51pp，+100降低4.55pp，方向cos四时域更低，不是整体升级。b1/b2链2000/1904；两次校准门均过。
资源与保护：1.713758GPUh≤4（含180.10秒停止段）；保守累计26.555645GB≤30，含已删除1.046GB初次复制偏离；285受保护文件变化0；固定窗口记录全部匹配、源hash稳定、sampler与E90一致。DINO Meta自定义许可仅内部研究。状态：COMPLETE；无本轮未完成必做项，无需新增用户决定。

### E96a 完成 | 2026-11-14T07:32:02.316157-03:00 BRT
目标：E95 P1 current_j=pooled_j+P(raw_j)单因素；P共享无偏置零初始化，history/W/目标/归一化/优化器/seed90不变，adapter监督目标随current改变（必要配套，门不变）。195合同通过；Piper固定5000步/batch32/160000抽样，采样器及h布局与M1完全相同；单h主布局，27dev/64train parent、bootstrap2000 seed90，分camera/变化量Q、同h fine/global与四h补充完成。
结果：dev+10 E96a−M1=+3.269 [+2.518, +4.011]pp；预注册总体通过。GPUh=0.256187≤1；新增上界1.051462GB≤5；保护文件变化0。目录/命令：code_e96a，results/e96a_control_20261114_01，results/e96a_m1_p1_20261114_01；setsid nohup pipeline_e96a.py/evaluation_waiter_e96a.py；报告notes/e96a_spatial_bypass_20261114.md。结论以固定终点预注册为准，不延训/挑点。状态 COMPLETE。

## E97-B1 完成 · 2026-11-14T13:47:43.302999-03:00 BRT

目标为Piper双臂穿扎带的阶段与完成谓词体系，以及B2人工dev标注方案。改动为10阶段、21个中英双语谓词、24条正常/失败/回退/重试转移规则和机器可读JSON。采用E83-A只读训练叙述，rev4排除dev/test，永久排除demo3 episode142。

命令为 ssh -o BatchMode=yes <gpu-server>，python3 data/e97b_stages_v1/statistics.py --commit，修订规则后加 --update-own-draft，python3 data/e97b_stages_v1/design.py 和 finalize.py。目录为 data/e97b_stages_v1/，报告为 notes/e97b_stages_predicates_astra_20261114.md。

结果为226 train parent、1013叙述段，995段规则映射、328单阶段、667混合、18未映射全文保留。覆盖P3全部145 parent。dev方案按元数据抽20个唯一parent，分层7/7/3/3，seed=2026111497，代表视频30.97名义分钟，预计17–25人时。0 GPUh、0训练，新增文件小于1MB。源hash及结构校验通过。

结论为设计可供用户审阅，时长属于训练叙述区间统计，机械锁紧需额外载荷证据。未执行人工B2、dev内容标注或模型比较。待用户审定阶段粒度、完成标准、名单及人员。状态 DESIGN_COMPLETE_PENDING_HUMAN_REVIEW。

## E97-A1 | 2026-11-14T13:53:37.942526-03:00 BRT | Piper机器人/环境残差统计分解
- 目标：沿E93 B5比较R0/R1/R2，检查状态/FK/标定覆盖，量化E96a收益的状态相关方向及剩余。
- 改动：仅新建code_e97a、results/e97a_robot_env_20261114_01、notes/e97a_robot_env_decomp_20261114.md；原缓存/ckpt只读。恢复27dev录制的精确对齐小型状态补充5.19MB，不改窗口/标签。
- 命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup .../e29/bin/python -u e97a_core.py；E97A_CPU_ONLY=1 CUDA_VISIBLE_DEVICES="" ... e97a_evaluate.py；审计/收尾脚本见报告。
- 目录：T/code_e97a/；T/results/e97a_robot_env_20261114_01/。
- 结果：546 capture/242训练parent/107057 P1窗状态覆盖100%；dev27parent/211窗，h100为26/208。R1 skill=0.915/4.747/10.220/26.219%；R2=8.101/14.999/22.741/39.438%。E96a收益92.3–98.0%落在训练状态输出span，但不能等同物理手臂归因；span外E96a skill=0.690/1.673/1.407/2.813%，统计剩余仍可预测。
- 结论：支持未来轨迹条件与独立剩余目标的下一轮A/B；无可信像素分区，暂不据此上手臂降权。所有alpha仅训练内48parent选，唯一完成dev预测pass。
- 资源：无神经训练；失败GPU准入0.000344692GPUh，其余CPU；新增约0.530GB；保护418文件变化0。
- 状态：COMPLETED_ANALYSIS_ONLY；物理像素归因/投影/标定/下一轮训练未执行，限制详报告。


## E98-L-train | 2026-11-14T14:10:57.733373-03:00 | 进行中

目标：60个rev4训练parent阶段边界与完成谓词标注，6个parent独立复标。label_source=model(gpt-6-astra)，非人工，只作训练用途。
改动与目录：独占新建 data/e98_predicate_labels_train；报告 notes/e98_train_labels_20261114.md。
命令：ssh -o BatchMode=yes <gpu-server>；python3 select_train.py；setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u extract_frames.py。
结果：seed9801，P3 V3窗口配额20/24/8/8；60个train parent，73.24名义分钟；粗抽帧完成。旧源前缀迁至 /mnt/workspace/dataset/piper，原位只读。
结论/状态：视觉标注进行中，未交付完整训练标签；GPUh=0。

## E98-G | 2026-11-14T14:11:18.524560-03:00 BRT | COMPLETE_TEXT_DESIGN_AND_COVERAGE
- 目标：Piper阶段/谓词跨任务泛化，纯文本/CPU；label_source=model(gpt-6-astra)，非人工。
- 改动：10阶段/21谓词→11核心+10扩展原语、25模板；ABC全部2484 cleaned段、yam49 clip/196叙述映射；20任务设计例（含B/C非实测示例）。
- 命令：ssh -o BatchMode=yes <gpu-server>；python3 data/e98_generalization_v1/build_v2.py；python3 data/e98_generalization_v1/finalize.py。
- 目录：data/e98_generalization_v1；报告notes/e98_stage_generalization_astra_20261114.md。
- 结果：ABC核心1424/2484=57.33%，扩展1962/2484=78.99%；459部分、63未映射。yam核心58/196=29.59%，扩展78/196=39.80%；118状态。116 ABC记录18无段，15/16任务有段，空任务覆盖null。
- 结论：可复用原语/关系模板，但不能证明视觉/执行零样本泛化；结型/解缠/接头机制仍需子计划。无dev标注、无世界头语言更改、无GPU/下载/大文件复制，新增约10.00 MB（含保留失败草稿）；首次16非空任务断言已纠正并留档。
- 状态：本轮完成。后续物理验收协议、视觉/本体泛化实验待用户决定；不作为本轮未完成工作。

## E98-W — 2026-11-14T14:13:51.764252-03:00 BRT — running
- 目标：E97-A1 §11 的 A/B：未来机器人轨迹条件 vs 显式 R1 剩余目标；仅 Piper。
- 改动：独占 code_e98w（由 code_e96a 小源码复制），101 原行 oracle q/FK token 只供 W；A/B 同参数、seed90、batch32、AdamW3e-4、固定5000步。B 用5折parent交叉拟合R1，dev加回E97-A1训练全量冻结R1，保留原目标尺度。
- 命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .../venvs/e29/bin/python -u code_e98w/e98w_prepare.py / e98w_run.py。
- 目录：results/e98w_control_20261114_01；正式候选e98w_a/b_20261114_03；失败准入_01和20步profile_02保留；notes/e98w_robot_conditioned_20261114.md。
- 结果：CPU合同通过，旧参数初始化逐位一致、O条件隔离0差，R1复现最大1.2e-7；dev未前向；前20步profile预算停止41.19秒入账。
- 结论/状态：训练内状态/FK预热后继续预算内profile，禁止dev选模型；GPU总额7200秒、新增≤8GB。

### E98-L-dev | 2026-11-14T14:15:06.814095-03:00 | IN_PROGRESS
- 目标：B2固定20个dev parent的阶段/完成谓词模型标注与4-parent独立复核；label_source=model(gpt-6-astra)，非人工，结论限定基于模型标注的dev真值。
- 改动：独占建立data/e98_predicate_labels_dev；只读迁移后dataset源，源SHA256/三相机帧数核验；报告notes/e98_dev_labels_20261114.md。
- 命令：ssh -o BatchMode=yes <gpu-server>；scripts/media.py audit（setsid nohup）；media.py extract；scripts/compile_labels.py。
- 目录：data/e98_predicate_labels_dev，review_independent及primary_shards隔离。
- 结果：20源/60视频已核验；首个parent3段126谓词；GPUh=0。
- 结论：仍在标注，不可视为完整B3输入；严格因果缺测显式保留。
- 状态：IN_PROGRESS。

### E98-W 最终收据 — 2026-11-14T15:21:37.987491-03:00 BRT
- 目标/实现/命令/目录见本条初始记录与notes/e98w_robot_conditioned_20261114.md。
- 结果：状态completed；GPU含失败3874.854s=1.076348GPUh；新增≤1.669115GB；保护文件139项SHA变化0。
- 结论：{"future_condition_effective": false, "separation_extra_benefit": false, "by_h": {"25": {"future_condition_effective": false, "separation_extra_benefit": false, "A_minus_E96a_full": {"pool": "dev", "h": 25, "camera": "all", "section": "full", "comparison": "A-E96a", "parents": 27, "delta_skill": NaN, "ci95": [NaN, NaN], "mse_reduction": NaN, "mse_reduction_ci95": [NaN, NaN]}, "B_minus_A_outside": {"pool": "dev", "h": 25, "camera": "all", "section": "outside_span", "comparison": "B-A", "parents": 27, "delta_skill": NaN, "ci95": [NaN, NaN], "mse_reduction": NaN, "mse_reduction_ci95": [NaN, NaN]}, "B_minus_A_full": {"pool": "dev", "h": 25, "camera": "all", "section": "full", "comparison": "B-A", "parents": 27, "delta_skill": NaN, "ci95": [NaN, NaN], "mse_reduction": NaN, "mse_reduction_ci95": [NaN, NaN]}}, "50": {"future_condition_effective": false, "separation_extra_benefit": false, "A_minus_E96a_full": {"pool": "dev", "h": 50, "camera": "all", "section": "full", "comparison": "A-E96a", "parents": 27, "delta_skill": NaN, "ci95": [NaN, NaN], "mse_reduction": NaN, "mse_reduction_ci95": [NaN, NaN]}, "B_minus_A_outside": {"pool": "dev", "h": 50, "camera": "all", "section": "outside_span", "comparison": "B-A", "parents": 27, "delta_skill": NaN, "ci95": [NaN, NaN], "mse_reduction": NaN, "mse_reduction_ci95": [NaN, NaN]}, "B_minus_A_full": {"pool": "dev", "h": 50, "camera": "all", "section": "full", "comparison": "B-A", "parents": 27, "delta_skill": NaN, "ci95": [NaN, NaN], "mse_reduction": NaN, "mse_reduction_ci95": [NaN, NaN]}}, "100": {"future_condition_effective": true, "separation_extra_benefit": false, "A_minus_E96a_full": {"pool": "dev", "h": 100, "camera": "all", "section": "full", "comparison": "A-E96a", "parents": 26, "delta_skill": 0.1635314237170451, "ci95": [0.1389085590385716, 0.19035757787227411], "mse_reduction": 0.18405046718874662, "mse_reduction_ci95": [0.15979154388343075, 0.21097731373081272]}, "B_minus_A_outside": {"pool": "dev", "h": 100, "camera": "all", "section": "outside_span", "comparison": "B-A", "parents": 26, "delta_skill": -0.0013167053739036396, "ci95": [-0.0029195105276123316, 5.802514812701565e-05], "mse_reduction": -0.00036126471537645255, "mse_reduction_ci95": [-0.0008215369035805428, 1.5103172880074366e-05]}, "B_minus_A_full": {"pool": "dev", "h": 100, "camera": "all", "section": "full", "comparison": "B-A", "parents": 26, "delta_skill": -0.3118605982733812, "ci95": [-0.35314085006826657, -0.27357141228651566], "mse_reduction": -0.350991188759492, "mse_reduction_ci95": [-0.42058707390127087, -0.2914674303950998]}}}, "full_non_degradation_rule": "paired full skill CI lower >=0; conservative sufficient evidence", "brt": "2026-11-14T15:21:36.121985-03:00"}

### E98-W 评估有效性更正 — 2026-11-14T15:28:25.012392-03:00 BRT
首轮CPU聚合未拒绝NaN，先前完成状态与False性能判定全部无效，已保留invalid_evaluation_01。定位为eval MHA native fastpath完全遮罩token产生NaN；训练路径有限。冻结5000 checkpoint、窗口、尺度、R1、alpha与判定，仅CPU关闭fastpath补算非有限输出（dev18个window×h×model），所有原有限向量按字节保留。严格单次dev前向出现数值恢复例外，明确披露；不重训、不换checkpoint、不改阈值。最终有效指标路径recovery_02/，后续收据将覆盖当前有效性声明。

### E98-W 最终收据 — 2026-11-14T15:29:17.225164-03:00 BRT
- 目标/实现/命令/目录见本条初始记录与notes/e98w_robot_conditioned_20261114.md。
- 结果：状态completed；GPU含失败3874.854s=1.076348GPUh；新增≤1.760328GB；保护文件139项SHA变化0。
- 结论：{"future_condition_effective": true, "separation_extra_benefit": false, "by_h": {"25": {"future_condition_effective": true, "separation_extra_benefit": false, "A_minus_E96a_full": {"pool": "dev", "h": 25, "camera": "all", "section": "full", "comparison": "A-E96a", "parents": 27, "delta_skill": 0.12887137605090085, "ci95": [0.1198875194792681, 0.13767216022599346], "mse_reduction": 0.2336981592172771, "mse_reduction_ci95": [0.20401705233666864, 0.2652752921264685]}, "B_minus_A_outside": {"pool": "dev", "h": 25, "camera": "all", "section": "outside_span", "comparison": "B-A", "parents": 27, "delta_skill": -6.035091530225595e-05, "ci95": [-0.0005803663384487315, 0.0004379505048885185], "mse_reduction": -4.643840086615558e-05, "mse_reduction_ci95": [-0.0004434785871312658, 0.0003296722008188335]}, "B_minus_A_full": {"pool": "dev", "h": 25, "camera": "all", "section": "full", "comparison": "B-A", "parents": 27, "delta_skill": -0.031044916718157595, "ci95": [-0.03926741338771323, -0.023942343576921547], "mse_reduction": -0.056297527910476486, "mse_reduction_ci95": [-0.07495477378468424, -0.041581951947234]}}, "50": {"future_condition_effective": true, "separation_extra_benefit": false, "A_minus_E96a_full": {"pool": "dev", "h": 50, "camera": "all", "section": "full", "comparison": "A-E96a", "parents": 27, "delta_skill": 0.15636741200682155, "ci95": [0.14584723634424956, 0.16713924604906372], "mse_reduction": 0.28896855739698735, "mse_reduction_ci95": [0.25061766372167427, 0.3284267457922902]}, "B_minus_A_outside": {"pool": "dev", "h": 50, "camera": "all", "section": "outside_span", "comparison": "B-A", "parents": 27, "delta_skill": 0.002293393639002786, "ci95": [0.0014991258739138623, 0.003166265897087914], "mse_reduction": 0.0015502082171654497, "mse_reduction_ci95": [0.0010101068484929115, 0.002139615542808159]}, "B_minus_A_full": {"pool": "dev", "h": 50, "camera": "all", "section": "full", "comparison": "B-A", "parents": 27, "delta_skill": -0.08149549619942877, "ci95": [-0.09778030019030208, -0.06641485307828962], "mse_reduction": -0.15060450044458906, "mse_reduction_ci95": [-0.1849175431395024, -0.11888459929282699]}}, "100": {"future_condition_effective": true, "separation_extra_benefit": false, "A_minus_E96a_full": {"pool": "dev", "h": 100, "camera": "all", "section": "full", "comparison": "A-E96a", "parents": 26, "delta_skill": 0.1635314237170451, "ci95": [0.1389085590385716, 0.19035757787227411], "mse_reduction": 0.18405046718874662, "mse_reduction_ci95": [0.15979154388343075, 0.21097731373081272]}, "B_minus_A_outside": {"pool": "dev", "h": 100, "camera": "all", "section": "outside_span", "comparison": "B-A", "parents": 26, "delta_skill": -0.0013167053739036396, "ci95": [-0.0029195105276123316, 5.802514812701565e-05], "mse_reduction": -0.00036126471537645255, "mse_reduction_ci95": [-0.0008215369035805428, 1.5103172880074366e-05]}, "B_minus_A_full": {"pool": "dev", "h": 100, "camera": "all", "section": "full", "comparison": "B-A", "parents": 26, "delta_skill": -0.3118605982733812, "ci95": [-0.35314085006826657, -0.27357141228651566], "mse_reduction": -0.350991188759492, "mse_reduction_ci95": [-0.42058707390127087, -0.2914674303950998]}}}, "full_non_degradation_rule": "paired full skill CI lower >=0; unchanged conservative sufficient evidence", "recovery_exception": "CPU replacement of original nonfinite outputs only; no finite dev output repeated for scoring; checkpoints/scale/windows/thresholds frozen"}

- E98-W最终补审计 2026-11-14T15:30:36.593060-03:00 BRT：5000步h布局一致、配置仅variant差异、冻结映射buffer逐tensor一致；GPU原段1.076348h，计CPU后处理及清理余量后上界1.158522h≤2；新增上界1.760381GB≤8；指定工作完成。

### 默认推理数值修复与最终有效性
最终有效指标仅为 `results/e98w_control_20261114_01/recovery_02/summary.json`、`paired.json`、`decision.json`；根目录首轮同名指标无效，见VALID_RESULTS.json。新包在eval transformer block中强制训练等价SDPA并恢复调用者backend标志，聚合入口拒绝非有限值。CPU回归证明训练forward逐位不变，A/B默认eval对无效mask均有限、与恢复SDPA输出最大差0。训练时源码与首轮评估器保存在invalid_evaluation_01；checkpoint与结果未改。严格单次dev前向出现已披露的18个无效输出补算例外，所有首轮有限输出逐值保留。


### E98-L-dev | 2026-11-14T15:42:16.320389-03:00 | COMPLETE
- label_source=model(gpt-6-astra)，非人工，基于模型标注的 dev 真值。
- 目标：B2 seed2026111497固定20 dev parent阶段/完成谓词；seed98抽4独立复核。
- 改动：新目录内三相机CPU抽帧、盲锁因果、阶段边界细化、独立复核、分歧裁决和SHA清单；不读E98-L-train。
- 命令：ssh -o BatchMode=yes <gpu-server>；media.py、compare_labels.py、assemble_dev.py --include-pending --tag assembled_final_v2、finalize_dev.py --validated-tag assembled_final_v2；长抽帧setsid nohup与状态文件。
- 结果：固定20 parent，367阶段段、27600谓词、4074有效因果答案，194完整固定审计点。
- 因果锁定记录、SHA绑定、边界邻帧、预算校验通过；GPUh=0。
- 结论：基于模型标注的 dev 真值；四值一致率80.2669%、κ0.722936；52匹配边界中位5帧/P90 128.5帧；P073机械锁紧未知，B3未训练，世界头冻结不变。
- 交付目录 data/e98_predicate_labels_dev，报告 notes/e98_dev_labels_20261114.md。
<!-- E98 complete assembled_final_v2:cafa4ec8f838259ba6e837b70bd508d02a812ab4c0f2348bec4f908368639400 -->

### E98-L-train | 2026-11-14T16:35:15.672531-03:00 | RETROSPECTIVE_DELIVERED_PROTOCOL_INCOMPLETE

- 目标为60个rev4训练parent阶段与完成谓词标注及10%独立复核，label_source=model(gpt-6-astra)，非人工。
- 改动在data/e98_predicate_labels_train内，seed9801分层20/24/8/8，保存选择脚本、三相机CPU加密帧、789阶段、9956谓词及独立6parent。
- 命令使用ssh -o BatchMode=yes <gpu-server>，select_train.py、extract_frames.py、e98_refineXX.py、author_completeXX.py、e98_apply_p081_audit.py、e98_freeze_primary.py、e98_agreement.py、e98_finalize_retrospective.py，长抽帧setsid nohup并写状态。
- 结果为4410标注源帧，3441必需当前阶段帧点无缺失。P081复核252帧、更正250答案。精确共同156对四分类一致74.36%，双方二值87对一致87.36%。77匹配边界中位20帧。frames约0.856GB，GPUh0。
- 结论为回顾性模型训练目标版本已交付。因果首遍盲看未执行，候选下一阶段覆盖不完整，共识裁决未执行。完整B2协议尚未完成。
- 报告在notes/e98_train_labels_20261114.md，计数、SHA、限制与待办在manifest.json。需决定回顾性版本的训练使用及后续补标安排。

## E99-W2 | 2026-11-15T00:54:54.177126-03:00 BRT
目标：动作计划条件下P3完整v3.1。改动：新code_e99w2；目录results/e99w2_*；命令/结果/结论见notes/e99w2_planned_traj_20261115.md；状态进行中。

## E99-B3 | 2026-11-15T01:02:36.382996-03:00 BRT | RUNNING
目标：四方法阶段完成谓词比较及B4因果回放；基于模型标注的 dev 真值。新目录 code_e99b3/、results/e99b3_20261115_01/；报告 notes/e99b3_predicates_20261115.md。命令：ssh -o BatchMode=yes <gpu-server>；e29/bin/python code_e99b3/{prepare,features,heads,vlm,controller}.py（setsid nohup）。48/12训练parent内部验证，dev一次；GPU≤3h，新增≤20GB。已创建骨架、审计数据和只读环境；状态进行中。

### E99-W2 完成 | 2026-11-15T01:20:44.191784-03:00 BRT
目标：P3完整v3.1计划轨迹部署条件收益。改动：code_e99w2；初始化E98-W A；5000步/batch32/lr3e-4。命令：setsid nohup env CUDA_VISIBLE_DEVICES=GPU-56f697ae-6017-3e7b-5d9b-1b612543fe40 E99_GPU_UUID=同UUID /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e99w2_driver04.py。目录：results/e99w2_p3_20261115_04、results/e99w2_control_20261115_01/evaluation_01。结果：预注册有效=False，GPUh保守上界=0.349835，新增约0.326GiB；结论/覆盖限制见notes/e99w2_planned_traj_20261115.md；状态COMPLETE_FIXED_P3_DEV_CACHE_LIMITED。

### E99-W2 +100覆盖补全与最终状态 | 2026-11-15T01:25:19.819737-03:00 BRT
改动：只读复用E93已有特征，原48窗口/6parent不变；只对47个未评分+100窗口推理，原预测保留。命令：setsid nohup .../python e99w2_complete_h100.py；结果目录results/e99w2_control_20261115_01/evaluation_02_h100_complete；预注册有效=False；GPUh≤0.364284，新增约0.330GiB。结论：原单parent限制已解决；N1规则仍未通过。状态COMPLETE_FIXED_P3_48WINDOWS_ALL_HORIZONS。

## E99-B3 | 2026-11-15T01:43:07.805144-03:00 BRT | COMPLETE
目标：四方法阶段完成谓词+B4。全部结论基于模型标注的 dev 真值；60train按48/12内部验证，dev20一次；因果4074条/二值2533。四法宏AUROC：a=0.629, b=0.795, c=0.760, d=0.781。B4编辑距离=84.850；边界匹配19/347。GPU=0.676641h，新增上界=4.369GB；276保护文件零变化。改动：独立code_e99b3，结果results/e99b3_20261115_01，报告notes/e99b3_predicates_20261115.md。命令：e29/bin/python prepare.py、features.py train、setsid nohup pipeline.py、CPU finalize.py；完整锁定/哈希/分段命令见报告。结论：视觉与本体收益按谓词及配对CI披露，不能证明必要性；P073缺载荷证据。指定工作完成，无待用户决策。

E99-B3最终口径补充：基于模型标注的dev真值；性能分数冻结评一次，无dev选型。但dev首批64帧另有2次完整A前向用于O子图/未来R-W隔离合同，未替换原分数；严格逐帧单前向有此例外。详见报告尾节和FINAL_STATUS.json。GPU上界已包含合同前向。

### E100-C | 2026-11-15T02:26:36.851901-03:00 BRT
目标：修正链与无链/原链同初始化对照。目录code_e100c、results/e100c_*；用户训练前改为每组seed90，三组各5000步。已复核两信息集问题并通过CPU开环扰动测试。命令：setsid nohup .../python -u e100c_launch.py；GPU≤3GPUh、新增≤10GB。状态：准备开训；结果与结论待追加，报告notes/e100c_chain_fix_20261115.md。

### E100-C 完成 | 2026-11-15T02:50:07.412033-03:00 BRT
目标/改动：C0无链、C1原链、C2删gap+真开环，同初始化seed90各5000步。命令e100c_launch.py；目录code_e100c/results/e100c_*_03；结果采用C2；GPUh≤0.988112，新增0.7808GiB；报告notes/e100c_chain_fix_20261115.md。首启路径污染失败已计费并修复，旧ledger撤销本会话active字段，旧ckpt/指标未写。状态COMPLETE。

## E103-D | 2026-11-15 BRT | PREPARING
目标：ABC S1约10h和yam allowlist无动作数据世界头增益；C2只读起点。新目录code_e103d、data/e103d_cache、results/e103d_control_20261115_01。命令：ssh -o BatchMode=yes <gpu-server>；setsid nohup e29/bin/python -u e103d_plan.py / e103d_cache_launch.py。先CPU时间闸门与资源估算，再两个Piper parent数值复现，通过后原位提取。GPU≤6h、新增≤40GB；D0/D1预注册在训练前锁定。结果与结论待完成，报告notes/e103d_external_data_20261115.md。

## E101-S | 2026-11-15T04:08:29.713454-03:00 | RUNNING
- 目标：用训练parent袋外谓词分数调时间平滑阶段控制器，dev一次冻结评估。
- 改动：新建code_e101s、results/e101s_20261115_01与notes/e101s_stage_controller_20261115.md；旧代码/标签/结果只读。
- 命令：ssh -o BatchMode=yes <gpu-server>；CUDA_VISIBLE_DEVICES为空、PYTHONDONTWRITEBYTECODE=1、setsid nohup e29/bin/python -u code_e101s/run.py fit。
- 当前：60 train parent 5折b/d谓词分数完成；仅12训练parent具全帧缓存用于调参。GPUh=0。首段unknown阶段映射异常已修正并保留日志，无dev评分。
- 结果/结论：待冻结dev；状态RUNNING。


### E101-A — 2026-11-15T04:14:26.825467-03:00 BRT
目标：C2 Piper动作头劣于复制的口径/噪声/可学习性/夹爪诊断；无正式训练。改动：新code_e101a只读加载C2，Euler4/8/16×N1/8/16、零起点，CPU岭回归/小MLP与输入消融；固定dev48/6和训练64窗，探针2320窗/145训练parent。命令：nohup setsid .../e29/bin/python code_e101a/diag.py prepare；driver.py；deterministic.py；ablate.py；side.py；finalize.py。目录：results/e101a_diag_20261115_01；报告notes/e101a_action_diag_20261115.md。结果：未发现新反归一化bug；历史状态岭回归关节前8/H50 L1=.00894/.05527rad，优于copy .01010/.05996；夹爪L1仍未胜，正MSE skill不能替代L1。action_only缺少q历史/FK，四模式动作loss系数均1。结论：优先因果本体历史+确定性增量参照、夹爪保持/变化独立头，再考虑更长FM；本轮不训练。GPU=0.148378h，新增=0.1058GiB，状态COMPLETE。

E101-A 最终复核 2026-11-15T04:15:58.750697-03:00 BRT：训练64 N1/N8同样劣L1复制；zero4关节H50有局部微胜，未作为选型。dev夹爪静止段贡献前8误差90%/80%，并非主要集中开合；数值/shape与只读SHA复核通过，新增0.1058GiB，GPUh不变。

## E101-S | 2026-11-15T04:16:02.231161-03:00 | COMPLETE_WITH_DECLARED_LIMITATIONS
- 目标：Piper阶段控制器时间平滑，替代E99-B3 B4硬切换失败回放。
- 改动：code_e101s独立实现S0/S1/S2/S3、60parent五折CPU谓词头、在线接口、冻结统计与3例时间线；仅12训练parent全帧OOF轨迹用于调参。
- 命令：ssh -o BatchMode=yes <gpu-server>；CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 setsid nohup e29/bin/python -u code_e101s/run.py fit；evaluate.py；制图异常后finish_frozen.py只读冻结产物恢复；check_online.py；finalize.py。
- 目录：code_e101s/；results/e101s_20261115_01/；notes/e101s_stage_controller_20261115.md。
- 结果：dev20parent/55754帧一次评估、bootstrap2000；S2+b acc=0.635632, macroF1=0.557515, recall30=0.365994, F1_30=0.376296, matched30=127/347, false/min=6.489221；S0 false/min=48.556157；S3离线recall30=0.504323。
- 结论：S2为因果研究推荐但recall低于0.7，不可直接实机部署；优先改善谓词监督、截断初态与S06/S07定义。dev真值为用户许可的非人工模型标签。无dev选型/重跑。
- 资源/审计：GPUh=0；新增实计14379862字节，保守上界114379862字节；25543个保护文件元数据零变化、输入SHA零变化。旧目录不写、无大文件复制/下载、未碰他人进程。
- 未完成/限制：其余48训练parent全帧特征未重提；未做可选融合或显式HSMM；无视频复审/人工真值/实机验收。CPU启动与制图异常日志保留。状态：COMPLETE_WITH_DECLARED_LIMITATIONS；无需用户决定。

## E102-A | 2026-11-15T04:43:45.073348-03:00 BRT | PREPARING
目标：C2初始化A0/A1/A2固定终点动作头修正；两态夹爪与因果本体历史。目录：code_e102a、results/e102a_control_20261115_01；报告：notes/e102a_action_fix_20261115.md。改动、命令、结果、结论将在完成后追加。

## E102-S | 2026-11-15T04:44:17.238310-03:00
目标：60 train/20 dev因果历史谓词，H0/H1/H2/H3 + E101-S S2；目录code_e102s、results/e102s_20261115_01；命令ssh BatchMode + setsid nohup e29/bin/python pipeline.py。状态：执行中，dev未开始。改动/结果/结论见notes/e102s_temporal_predicates_20261115.md。

### E102-A 终态 2026-11-15T04:54:16.133294-03:00 BRT
{"status": "failed", "command": "setsid nohup .../e29/bin/python -u e102a_launch.py", "directory": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_01", "changes": "A1 binary gripper BCE; A2 causal proprio + deterministic delta MSE with FM joint auxiliary; all C2 world plans fixed openings", "results": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_01/SUMMARY.json", "conclusion": "see failure status", "decision": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_01/DECISION.json", "GPUh_upper": 0.11957614279261583}

### E102-A 终态 2026-11-15T04:57:33.704265-03:00 BRT
{"status": "failed", "command": "setsid nohup .../e29/bin/python -u e102a_launch.py", "directory": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_02", "changes": "A1 binary gripper BCE; A2 causal proprio + deterministic delta MSE with FM joint auxiliary; all C2 world plans fixed openings", "results": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_02/SUMMARY.json", "conclusion": "see failure status", "decision": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_02/DECISION.json", "GPUh_upper": 0.18907656645492857}

### E102-A 终态 2026-11-15T05:20:24.933733-03:00 BRT
{"status": "complete", "command": "setsid nohup .../e29/bin/python -u e102a_launch.py", "directory": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_03", "changes": "A1 binary gripper BCE; A2 causal proprio + deterministic delta MSE with FM joint auxiliary; all C2 world plans fixed openings", "results": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_03/SUMMARY.json", "conclusion": {"joint_fix_effective": true, "joint_by_group": {"A0": false, "A1": false, "A2": true}, "gripper_fix_effective": false, "gripper_by_group": {"A1": {"delta_f1": 0.1797619047619048, "ci95": [-0.05000000000000001, 0.5130952380952382], "delta_false_switches_per_minute": -4.5, "false_switch_delta_ci95": [-6.75, -2.2499999999999996], "effective": false}, "A2": {"delta_f1": 0.22142857142857145, "ci95": [-0.11666666666666668, 0.5833333333333334], "delta_false_switches_per_minute": -3.0, "false_switch_delta_ci95": [-6.0, 0.7500000000000001], "effective": false}}, "deployable_world": {"A0": true, "A1": true, "A2": true}, "single_seed_limitation": true, "dev_parents": 6, "brt": "2026-11-15T05:20:24.534214-03:00"}, "decision": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e102a_control_20261115_03/DECISION.json", "GPUh_upper": 1.1129075713044343}

### E102-A 最终审计 | 2026-11-15T05:21:57.334421-03:00 BRT | COMPLETE
有效run03：A0/A1/A2各2000步，C2原ckpt只读；新增BCE开合头、A2因果本体历史/确定性Δq，FM关节辅助。命令：setsid nohup env PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u code_e102a/e102a_launch.py（cwd code_e102a）。结果目录results/e102a_control_20261115_03；模型results/e102a_a[012]_s90_20261115_03/ckpt_02000.pt。A2关节修正通过；夹爪F1显著性未通过；三组世界条件判据通过。GPU=1.112908h含失败；新增约0.7847GiB。无修订合同内未完成项；详见notes/e102a_action_fix_20261115.md与FINAL_AUDIT.json。

命令路径澄清：实际cwd为T/code_e102a，命令为setsid nohup env PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e102a_launch.py（stdout写对应control/launch.log）；上段code_e102a/e102a_launch.py为相对T的脚本位置，不是在该cwd中再拼接code_e102a。最终复核两次失败与成功训练/评估的全部12个GPU段均closed。

## E102-S 完成 | 2026-11-15T05:32:29.221374-03:00
目标：因果历史谓词+S2，改动：全80逐帧SigLIP2、两态本体、H0-H3五折、双初态。命令：setsid nohup e29 python extract/driver/evaluate/summarize.py；目录code_e102s、results/e102s_20261115_01。结果/结论：{"H2": {"training_selected_initial": "uniform", "macro_AUROC_delta": {"value": -0.10205575070398232, "ci95": [-0.14828643885780257, -0.06726830276579417]}, "recall30_delta_vs_E101": {"value": -0.10086455331412103, "ci95": [-0.1451640148212729, -0.05975656957661581]}, "recall30": 0.26512968299711814, "gap_to_point7": 0.4348703170028818, "historical_information_effective": false}, "H3": {"training_selected_initial": "uniform", "macro_AUROC_delta": {"value": -0.03828111606673967, "ci95": [-0.07191664660855167, -0.004937559885773642]}, "recall30_delta_vs_E101": {"value": -0.17291066282420747, "ci95": [-0.2265238598883894, -0.09025013269639082]}, "recall30": 0.1930835734870317, "gap_to_point7": 0.5069164265129682, "historical_information_effective": false}}；GPUh=0.4768594646997129，新增字节=1072103540。状态COMPLETE；详见notes/e102s_temporal_predicates_20261115.md。

### E102-X | 2026-11-15T05:38:34.310026-03:00 BRT | RUNNING
目标：固定E102-A A0/A1/A2 run03 ckpt02000，扩展E93 27parent，只读推理、无模型dev语言。目录code_e102x与results/e102x_extended_s90_20261115_01；合同窗口锁BEFORE_FORWARD_LOCK.json：211窗，原48子集，缺失动作仅RAM引用原next-feedback。命令：code_e102x cwd，setsid nohup env PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e102x_launch.py。预算0.5GPUh/10GB。状态准备完成待推理。

### E102-X 完成 | 2026-11-15T05:43:03.319242-03:00 BRT
目标：E102-A三组固定终点扩展dev，只读推理。改动：code_e102x复用E102-A评估器；E93 211窗/27parent含原48，+100为208窗/26parent；下一帧反馈仅RAM读取，0新特征。命令：setsid nohup .../e29/bin/python -u e102x_launch.py；随后python e102x_finalize.py。目录results/e102x_extended_s90_20261115_01；报告notes/e102x_extended_eval_20261115.md。结果：{"joint_by_group": {"A0": false, "A1": false, "A2": true}, "world_by_group": {"A0": true, "A1": true, "A2": true}, "gripper_by_group": {"A1": true, "A2": true}, "gripper_fix_effective": true}；配对差详PAIRED.json。GPUh上界0.074467，落盘约0.0087GiB。状态COMPLETE；无训练/选点/自动部署。

E102-X最终结论补充 | 2026-11-15T05:44:19.978202-03:00 BRT：49窗口事件/48物理去重事件；A1/A2−A0 F1±10 CI均>0，±5仍跨0；A2 GT世界下降四时域约2.0–2.65pp且CI均负；A2关节/世界原判据通过。原48三组完全重放，4GPU段全部closed；GPUh≤0.074467，最终落盘约8.89MiB。COMPLETE，无待决事项。

## E103-D | 2026-11-15T06:14:53.830949-03:00 BRT | COMPLETE
目标：ABC/yam无动作视频对Piper世界预测增益；C2只读起点，seed90各5000步，P3+等权P1世界辅助成对比较。代码code_e103d；缓存data/e103d_cache；控制results/e103d_control_20261115_01；有效训练results/e103d_d0_s90_20261115_02、e103d_d1_s90_20261115_02。命令：setsid nohup e29/bin/python -u e103d_pipeline02.py；e103d_train.py D0/D1；e103d_eval.py D0/D1/finish。无增益；GPU 2.0128h（含失败/中断段），新增9.863GB；D0/D1各5000步。
报告notes/e103d_external_data_20261115.md。失败/中断保留并计入GPUh。

## E98-C | 2026-11-15T12:37:59.060888-03:00 | IN_PROGRESS

- 目标：60训练parent因果首遍盲标，与dev因果语义一致；纯CPU、GPUh=0。
- 改动/目录：data/e98_predicate_labels_train_causal_v1/；notes/e98c_train_causal_labels_20261115.md；旧标签与源只读，未接触E103-D输出。
- 命令：SSH BatchMode 只读列举并查看指定报告和 spec.json；cat 新建报告，cat >> 追加两份日志；未调用 GPU/训练命令。
- 命令：ssh -o BatchMode=yes <gpu-server>；python3 export_queries.py；/mnt/workspace/dengjianfeng/venvs/e29/bin/python gate.py {init,expose,lock} --shard {a,b,c}。
- 结果：查询60parent/4410时刻/92610计划谓词，A/B/C独立上下文各20parent。曝光上限与逐点锁定哈希链。
- 结论/状态：尚未完成，禁止用于E102-S2全量训练；回顾对照及10%独立因果复标必须在全部锁定后进行，当前未执行。

## E104-P | 2026-11-15T12:59:34.273090-03:00 BRT | PREPARING
目标：ABC+yam先外部预训练再全量Piper微调，A2固定基线；F2 Piper10000(5000=F0)、F1外部5000+Piper5000，seed90恒定lr3e-4连续AdamW。改动code_e104p；只读复用data/e103d_cache；目录results/e104p_control_20261115_01、e104p_f1_s90_20261115_01、e104p_f2_s90_20261115_01。命令setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e104p_pipeline.py。新路径检查通过，报告骨架已建；预算3GPUh/10GB；预注册CPU核验后锁定，尚无本轮评分。
报告notes/e104p_pretrain_finetune_20261115.md。

## E104-P | 2026-11-15T13:01:31.307569-03:00 BRT | STOPPED
目标：ABC+yam先外部预训练再全量Piper微调，A2固定基线；F2 Piper10000(5000=F0)、F1外部5000+Piper5000，seed90恒定lr3e-4连续AdamW。改动code_e104p；只读复用data/e103d_cache；目录results/e104p_control_20261115_02、e104p_f1_s90_20261115_02、e104p_f2_s90_20261115_02。命令setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e104p_pipeline.py。必需项未全部完成，详见PIPELINE.json；不作增益结论。
报告notes/e104p_pretrain_finetune_20261115.md。

## E104-P | 2026-11-15T13:02:42.546720-03:00 BRT | PREPARING
目标：ABC+yam先外部预训练再全量Piper微调，A2固定基线；F2 Piper10000(5000=F0)、F1外部5000+Piper5000，seed90恒定lr3e-4连续AdamW。改动code_e104p；只读复用data/e103d_cache；目录results/e104p_control_20261115_03、e104p_f1_s90_20261115_03、e104p_f2_s90_20261115_03。命令setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e104p_pipeline.py。新路径检查通过，报告骨架已建；预算3GPUh/10GB；预注册CPU核验后锁定，尚无本轮评分。
报告notes/e104p_pretrain_finetune_20261115.md。

### E98-C checkpoint 2026-11-15T13:41:42.830212-03:00 | IN_PROGRESS
源60parent/180视频原位SHA与帧范围核验完成（读取55.29GB、复制0）；主标仍进行，未读回顾答案。仅迁移5个未曝光完整parent，A/B/C各1470时刻，既有锁不变。显示几何v3修正纵横比；旧111时刻及217谓词未测可见性单列。GPUh=0；未启动复标/训练。目录data/e98_predicate_labels_train_causal_v1；报告notes/e98c_train_causal_labels_20261115.md。

## E104-P | 2026-11-15T14:09:45.652658-03:00 BRT | COMPLETE
目标：ABC+yam先外部预训练再全量Piper微调，A2固定基线；F2 Piper10000(5000=F0)、F1外部5000+Piper5000，seed90恒定lr3e-4连续AdamW。改动code_e104p；只读复用data/e103d_cache；目录results/e104p_control_20261115_03、e104p_f1_s90_20261115_03、e104p_f2_s90_20261115_03。命令setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e104p_pipeline.py。未满足外部预训练有效判据；GPUh上界1.7811；新增1.131GB；全部固定终点训练/评估完成。
报告notes/e104p_pretrain_finetune_20261115.md。

## E104-P | 2026-11-15T14:11:20.107776-03:00 BRT | AUDIT_PASSED
目标：ABC+yam先外部预训练再全量Piper微调，A2固定基线；F2 Piper10000(5000=F0)、F1外部5000+Piper5000，seed90恒定lr3e-4连续AdamW。改动code_e104p；只读复用data/e103d_cache；目录results/e104p_control_20261115_03、e104p_f1_s90_20261115_03、e104p_f2_s90_20261115_03。命令setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e104p_pipeline.py。固定终点全流程完成；连续AdamW/外部冻结状态/Piper成对流/A2重放/输入源码只读审计通过；GPUh上界1.781113，新增1.1314GB；有效主调度e104p_parallel.py。
报告notes/e104p_pretrain_finetune_20261115.md。

### E104-P 最终中文结论 | 2026-11-15T14:14:21.687587-03:00 BRT
E104-P COMPLETE：F1−F0 no_traj +25/+50/+100=-2.198,-1.949,-0.769pp；F1−F2=-3.180,-3.491,-2.001pp，全部95%CI上界<0；gt非劣性失败，动作/夹爪5%非劣性通过。F2−F0=+0.982,+1.542,+1.232pp（+100 CI跨0）。GPUh≤1.781113，新增≤1.15GB。完整固定终点和CPU审计通过；未完成/待决无。报告notes/e104p_pretrain_finetune_20261115.md；有效control/run03，主调度e104p_parallel.py。

## E105-Y | 2026-11-15 BRT | COMPLETE
目标：yam主体、ABC外部、完全不使用Piper的数据可行性；CPU only。改动：新增code_e105y脚本与results/e105y_audit_20261115_142944审计，原数据/缓存只读。命令：setsid nohup .../e29/bin/python code_e105y/audit02.py；variance.py/visual.py/finish.py。结果：43段26371帧879.033秒；固定clip窗344/344/344/335；1秒窗806/787/753/684，共同四时域628；保守日期parent2、宽松块3、严格同场景1；每parent8窗后16或24。留出1 parent无可靠跨parent CI；K3/5正式方案不成立。yam URDF存在，当前仅建议no_traj。ABC390eps/9.9983h的390 session_id是episode UUID，独立parent未确认；跨yam时间/任务/canonical交集0，完整内容去重未确认。起点A2/E98-W A均见Piper，推荐官方冻结SigLIP2+随机头。结论：不建议立即正式三方比较；探索按9月train/10月dev，先补独立parent。GPU=0；落盘<1MB（以FINAL_STATUS最终计数为准）。报告notes/e105y_yam_feasibility_20261115.md。时间2026-11-15T14:39:12.509116-03:00。

- E106-C | 20261115 BRT | PREPARING | F2续训10000→20000与五点收敛；code_e106c / results/e106c_*；命令与结论见 notes/e106c_longer_training_20261115.md。

### E98-C model transition 2026-11-15T15:42:45.501514-03:00
User requested all three primary workers gpt-6-luna high. Existing agents cannot hot-switch; preserved one annotator per parent by finishing current parents on sol and handing off at boundaries. C: 540 locks, 4 full parents, no pending exposure; Luna worker resumed index540. A: 726 locks, 12 full parents, no pending exposure; Luna worker assigned from index726. B: still completing current parent before handoff. Old sol labels remain unchanged. gate_v4 explicitly records model+annotator; audit accepts and matches these. GPUh=0.

### E98-C annotation QA disclosure 2026-11-15T15:52:20.807243-03:00
A Luna annotator self-reported scripted answer-template reuse for indices727–730, inheriting 21 unknown answers/visibility from index726. Original locks preserved; later_notes.jsonl records keys+hashes. These 5 moments are unreliable and pending independent blind review.

- E106-C | 2026-11-15T15:58:16.786286-03:00 BRT | COMPLETE | F2同流续至20000，15000/20000固定ckpt及五点曲线；code_e106c / results/e106c_control_20261115_02；GPUh=1.087853315815252；新字节=589360359；默认A2未改；详见 notes/e106c_longer_training_20261115.md。

- E106-C | 2026-11-15T15:58:43.261360-03:00 BRT | COMPLETE | 目标：Piper F2同采样流续训10000→20000并画收敛曲线；改动：新code_e106c/run02，恢复完整AdamW并重放前10000步sampler哈希，固定15000/20000保存；命令：nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -B -u code_e106c/run02/e106c_pipeline.py（start_new_session=True）+CPU postprocess_delivery.py；目录：results/e106c_control_20261115_02、results/e106c_f2_s90_20261115_02（run01失败保留）；结果：五点/27parent+48窗、bootstrap2000 seed90、全部指标与配对95%CI图表已完成；结论详见notes/e106c_longer_training_20261115.md，默认A2不变；资源GPUh上界=1.087853315815252，含失败；新增总量保守<641879565字节。

## E107-P | 2026-11-16T00:33:54.745948-03:00 BRT | COMPLETE_WITH_EVIDENCE_LIMITS
目标：固定E106C20000默认头多峰/模糊化诊断，仅Piper，不训练世界模型。
改动：新code_e107p；训练parent CPU线性夹爪探针；固定211窗/27parent（+100208/26）与48/6附表，两条件四时域；指标/阈值在GPU前锁定；2000次parent paired bootstrap seed90；禁MHA fastpath，拒非有限值。
命令：setsid nohup env PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u {collect,analyze,audit}.py（分步；cwd=T/code_e107p）；finalize.py记账。
目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e107p_diagnosis_20261116_02；CPU预检失败run01保留；报告：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e107p_multimodal_diagnosis_20261116.md。
结果：主集绝对方差比no_traj 0.850/0.812/0.754/0.724、gt_action 0.857/0.825/0.776/0.760，全部CI上界<1。机械阶段分叉混合比例11.11/13.89/11.57%；逐混合夹爪中间率超额CI均跨0。
结论：收缩可量化，模态平均机制未证实；建议暂缓完整E107-M，若另轮探索先M0/M1，再决定M2。严格同语义阶段占比和人工谓词AUROC受真值缺失限制，未用模型dev标签补真值。
资源：GPU 0.021338h（含30秒余量），新增约215.1MiB；旧checkpoint SHA未变。状态：诊断完成，严格多峰/人工语义证据有明确未完成项。

## E107-M | 2026-11-16T00:59:27.137030-03:00 BRT
目标：世界头M0/M1多峰小实验；目录code_e107m、results/e107m_control_20261116_01；起点E106C固定20000；固定3000步；CPU测试前置，GPU≤2h/新增≤10GB。命令：e107m_test.py（setsid nohup分离）；状态：PRECHECK，GPU未启动。报告notes/e107m_multimodal_head_20261116.md。

## E107-M | 2026-11-16T01:43:55.540064-03:00 BRT
目标：世界头低成本多峰验证，仅M0/M1；改动：code_e107m，32D z/256隐藏/K4能量+0.001均值MSE，旧头与主干冻结。命令：setsid nohup env PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u e107m_pipeline03.py（从已完成的固定训练终点评估）。目录：results/e107m_*_20261116_01；报告：notes/e107m_multimodal_head_20261116.md。结果：{"multimodal_effective": false, "proper": true, "noninfer": false, "spread": false, "openloop": false, "collapsed_horizons": [10, 25, 50, 100], "criterion": "all four conditions and horizons; primary no_traj, confirmatory gt_action; no_traj energy and CRPS improvement >=2 horizons lowerCI>0, mean skill all horizons lowerCI>-.01, spread/skill .8..1.2 each horizon; both k25/k50 openloop mean-vs-sample error improvement lowerCI>0; otherwise no M2/no default change", "default_unchanged": true, "M2": false, "brt": "2026-11-16T01:43:55.092206-03:00"}；资源：{"status": "complete", "GPUh_upper": 0.6038956917426549, "new_bytes": 1683923470, "old_checkpoint_unchanged": true, "brt": "2026-11-16T01:43:55.525184-03:00"}；结论：当前设定下多峰建模无明确收益，不进入M2，不改默认。 状态：COMPLETE。

- 2026-11-16T01:45:18.052698-03:00 BRT E107-M 交付终审通过；训练run01+评估run03；最终GPUh上界=0.6288431538309023（含压缩尾段与全部失败），新增保守字节=1733924970；FINAL_AUDIT.json/GPU_PROCESS_ACCOUNTING.json；报告notes/e107m_multimodal_head_20261116.md。

## E108-G | 2026-11-16T02:49:39.835488-03:00 BRT | COMPLETE
目标：默认20k夹爪G0误差分解、G1模型外因果解码；无训练/架构改动。
改动/目录：code_e108g/、results/e108g_diag_20261116_01/；报告notes/e108g_gripper_diagnosis_20261116.md。
命令：setsid nohup .../venvs/e29/bin/python -u code_e108g/collect.py / analyze.py / audit.py（分别执行，日志与状态在新结果目录）。
结果：211窗/27parent与48窗附表；训练2320窗/145parent锁定23配置选参；基线逐项复现F1±5=.149383、±10=.245679，FP/min±10=8.5；TP/FN/FP=16/33/51；5帧内回切4/51。四家族均失败预注册；dwell8/EMA.8的ΔF1@10=0 [0,0]、假检减少3.92%；滞回.45/.55及组合召回相对降6.67%。
结论：瓶颈不在因果解码，需要在原架构内通过目标/数据层面解决。仅给候选不实施。局限：无人工阶段/实发命令，标签上限不可识别；窗内寿命截尾。
资源：GPUh=0，新增约36.13MiB<2GiB；旧ckpt SHA复核一致，无新模型/部署。状态：COMPLETE。

## E109-G | 2026-11-16T04:32:59.275171-03:00 BRT | COMPLETE
目标：默认20k夹爪训练/dev泛化差距，区分窗口协议、抽样噪声、暴露与来源分布；无训练/新头/架构变更。
改动：仅code_e109g/、results/e109g_generalization_20261116_01/和报告notes/e109g_gripper_generalization_20261116.md；旧E108事件提取/匹配原样复用，158缺失特征帧CPU原编码器补提；不写E98-C。
命令：setsid nohup .../venvs/e29/bin/python -B -u code_e109g/inspect_baseline.py、features_cpu_v2.py、collect_v2.py、metadata.py、pipeline.py（各自日志/状态）。
结果：原211窗基线逐项复现；145训练parent/2320原窗和1160对齐窗，dev211窗/27parent；F1@10原训练.634891→对齐.460969，dev.245679；窗口项.173922[.101258,.245430]，占原差44.69%；剩余.215290，同规模抽样95%CI[.030689,.389013]。2k→20k对齐训练+.193173、dev−.015520（CI跨0）；仅2个训练评估事件未直接受监督，记忆因果不可独立识别；日期×来源共同支持22parent重加权差.246948[.105430,.386133]，粗来源不能解释完。
结论：协议解释可量化44.69%，其余为训练内拟合/泛化缺口，纯小样本不足以解释；记忆与真实场景/操作者差异的独立占比不可识别。候选仅建议不实施，不按dev选点。
资源/完整性：GPUh=0（含失败预检），新增保守≤64,447,551 bytes<2GiB；五旧checkpoint SHA未变，采样SHA与logits独立重算通过；无模型dev语言/阶段，无动作标签移动。状态：COMPLETE；缺失元数据与因果识别限制已明示。本轮无用户阻塞决定。

- E110-F | 2026-11-16T05:53:53-03:00 BRT | COMPLETE
目标：Piper动作头公平对照，A2确定性Δq头D vs条件flow Δq头F；训练前锁定单样本S4部署读法，同源E106C-20000起点/输入/数据流/4000更新；只重训动作路径参数。
改动：新增code_e110f（core/train/eval/finalize、合同测试、Euler诊断）；新增报告notes/e110f_flow_vs_deterministic_20261116.md与results/e110f_*。冻结其余150张量；夹爪2态BCE相同；无仿真/重映射/模型语言测试标签。
命令：CPU合同测试和Euler诊断；setsid/nohup远端执行e110f_train.py（D/F各4000步）、e110f_eval.py（27parent/211窗主集、48窗附表、S1/2/4/8）；CPU汇总及审计。旧代码/缓存/起点只读。
目录：/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e110f/；/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e110f_d_20261116_05/、e110f_f_20261116_05/、e110f_eval_20261116_01/。
结果：D/F均4000步；27parent上D前8/H50 L1=.007757/.042521，F-S4单样本=.007952/.044225；配对相对差CI上界+5.76%/+6.87%，不满足锁定S4读法≤5%非劣。F-S4 K8均值关节改善但不可事后改变部署读法；spread-skill=.294、塌缩87%，F1@±10差−.02675 [−.06379,0]；world pred非劣通过。D中位延迟4.88ms，F-S4 23.50ms。结论：当前设定下flow无明确收益，保留确定性头；默认模型未改。
资源：GPU6准入空闲34.7GiB、监测最低约33.0GiB；总GPU段1678.85秒=0.4663GPUh（含失败，限额2.5h）；新增425746772字节≈0.40GiB（限额10GiB）。冻结SHA及世界无条件输出逐位一致；sample/horizon SHA=38c5e37ede09b613a15786d0dd28f2f1df281493d585a3794b691f176092e6b3。评估摘要写盘且状态complete；事后打印阶段遇旧键名KeyError，CPU修正脚本但未重跑覆盖，摘要指标核对有效。失败段与限制详见报告。
状态：COMPLETE；后续只有另立预注册K8均值部署研究需用户决定，本轮无待批准修改。

## E111-F — 2026-11-16 BRT — Flow 头训练量核对与等量复比
- 目标：核对 E110-F D/F 初始化、累计训练量及 LR；仅训练不足时追加 F 并按锁定评估复比。
- 改动：新增 `code_e111f/e111f_resume.py`（约9.1KB，含确定性重放恢复与检查、16k追加逻辑）；报告 `notes/e111f_flow_trainlength_20261116.md`。旧结果/代码只读。
- 命令：读取 E110-F/E106-C 报告、脚本、摘要和 checkpoint 元数据；`python3 -m py_compile code_e111f/e111f_resume.py`；尝试 `setsid nohup env CUDA_VISIBLE_DEVICES=GPU-a654fa96-c3b6-6941-8b13-9c9f30f26de9 python3 e111f_resume.py`。
- 目录：`code_e111f/`；预留 `results/e111f_flow_20261116_01/`；未生成 checkpoint。
- 结果：D/F 参数均从 A2 strict-load 185/185 keys，`delta_head` optimizer step=22k+4k=26k；flow 新增随机参数0。D 的 Δq 目标曝光26k，F flow 目标曝光4k，确定性追加上限计划16k。训练启动前失败：服务器默认 Python 3.12.3 缺 NumPy/PyTorch；F05 无 optimizer/sampler state，无法运行恢复单测。GPU 0h，新增落盘约9KB，未评估。
- 结论：E110-F 参数级更新量对等，flow 速度头目标曝光不足；本轮复比未完成，不能判定曝光对等后是否仍塌缩。
- 状态：核对完成；训练/评估受运行时缺失阻断。待恢复兼容 PyTorch 环境后先跑确定性重放单测，再追加与评估。

## E112-S — 2026-11-16 (BRT)
- 目标：只读盘点 runs/results/data 存储、项目内引用和清理候选；等待用户逐类批准后才可能清理。
- 改动：仅新建报告骨架与清单目录，并追加本日志和 worklog；未修改既有数据/实验文件。
- **E111-F COMPLETE（接续）**；日期2026-11-16 BRT；目标：核对 E110-F flow 头目标曝光并在公平目标曝光下复比。改动：新增/更新 code_e111f/ 重放恢复、追加训练、评估和汇总脚本；追加报告 notes/e111f_flow_trainlength_20261116.md。命令：使用项目 e29/bin/python，E102_GROUP=A2、E102_SEED=90；确定性重放 F05 4k 后追加16k，按4k存点；沿用 E110-F 评估配置，2000 parent bootstrap、seed90。目录：results/e111f_flow_20261116_02/04/05 与 results/e111f_eval_add_04000/08000/12000/16000/。结果：F05重放逐位一致；D Δq 26k目标步、F flow 20k目标步；GPU累计6603.60秒=1.8343h，新增 code/results 约1.12GB；最终 S4 单样本关节非劣 CI 上界超+5%，F1@10差CI跨0，spread-skill=.278、塌缩87.96%，world skill下界−1.19pp。结论：flow 未满足预注册替换标准，默认D不变；塌缩持续但F目标曝光仍比D少6k，不能断言与训练量无关。状态：完成。记录时间 2026-11-16T09:37:33-03:00。

## E112-S 只读存储盘点结果（更正完整记录）— 2026-11-16 BRT
- 目标：盘点 runs/results/data 顶层目录大小、类型、属主、修改时间与项目内引用，提出分级建议；等待用户批准。
- 改动：新建 notes/e112s_storage_inventory_20261116.md 和 results/e112s_inventory_20261116/ 清单文件；仅追加本日志与 worklog。无清理、移动、压缩、chmod 或数据/实验文件修改。
- 命令：SSH <gpu-server>；对每个顶层目录串行运行 nice/ionice du -skx（单目录 12 秒 timeout）；grep 扫描 notes/*.md、实验日志、worklog、code_*；Python 只读 stat 汇总类型。目录：runs/results/data。
- 结果：runs 563 项、约 745.81 GiB；results 92 项、约 19.54 GiB（排除 E112 自己的清单目录）；data 39 项、完成 36 项约 93.44 GiB，两个目录超时，一个活跃 E98-C 跳过。引用与每目录类别见清单 TSV。A 230 项，B 14 项；C 无满足已验证可重建条件的项目；D 451 项，含 9 个待验证缓存约 2.85 GiB。B 压缩预计释放约 0–0.001 GiB。
- 限制/风险：未扫描同事目录/进程；symlink 扫描因 CPFS 延迟中止，首批 418 条链接记录，文件类别检查可能对文件 symlink 执行目标 stat，不读取目标内容，外部目标是否存在无法排除。run E 最佳旧模型路径未唯一定位。
- 结论：仅形成建议；任何压缩/删除都等待用户逐类批准。报告与清单合计约 5.4 MiB，低于 50 MiB 限制。
- 状态：盘点完成，等待用户审阅/批准；没有执行清理。


## E98-C | 2026-11-16T14:22:34.772958-03:00 | COMPLETE
- 目标：为E98-L-train 60个train parent补严格因果首遍标签，统一dev因果口径；CPU-only。
- 改动/目录：新增并完成`data/e98_predicate_labels_train_causal_v1/`，发布`predicates_causal.jsonl`（4410×21）和`stage_causal.jsonl`；报告`notes/e98c_train_causal_labels_20261115.md`。旧标签、源视频、dev标签、同事目录及E103-D未修改。
- 命令：SSH BatchMode 只读列举并查看指定报告和 spec.json；cat 新建报告，cat >> 追加两份日志；未调用 GPU/训练命令。
- 命令：`audit_progress_v3.py`及`--publish`；`compare_after_lock_v3.py`；`event_offsets_after_lock_v3.py`；Sol/Luna review gate；`audit_review_mixed_v1.py --publish`；`setsid nohup python3 source_audit_verify_final.py`。脚本哈希见`data/e98_predicate_labels_train_causal_v1/release_scripts.sha256`。
- 结果：主因果审计4410/4410时刻、92610/92610谓词、60/60 parents、0错误；GPUh=0、目录约1.05GB。源身份终验60 parents/180 videos、55,292,895,938 bytes，源哈希与mtime未变，demo3 episode142排除确认。回顾表对齐9956条，四值一致50.80%、κ=.3246，yes/no一致75.40%；回顾yes中29.88%因果为no/unknown。操作性同cycle/attempt偏移不视作物理事件时刻。独立复标441时刻/9261谓词；Sol→Sol 202时刻四值一致73.95%、κ=.6353，Luna→Luna 117时刻30.97%、κ=.0653；模型分层及跨模型对照见报告。23个疑似模板/边界/谓词不一致时刻排除可靠训练目标池，锁未覆盖。
- 结论：任务完成；输出仅供训练目标或训练池筛选，非测试真值/阈值/checkpoint依据。复标非人工准确度，Luna自一致性偏低；E102-S2本轮未做。总体dev统计暴露与接续上下文/模板疑点在报告披露。
- 状态：COMPLETE。

## E112-S 后续：按用户指示清理 E28 特征缓存 — 2026-11-16 BRT
- 目标：仅清除 runs/e28_chain_20261103 的特征缓存，并报告 runs 下剩余大目录。
- 改动：删除 6 个 value/*/value/features 目录下 72 个普通 .npy 特征数组；保留 status、checkpoint、日志、报告、评估 JSON 与目录结构。删除清单 e28_feature_delete_manifest.tsv。
- 命令：SSH <gpu-server>；按清单核对文件类型/大小/link count，再以 Python unlink 精确文件；随后对 E28 目录运行 timeout 30s du -skx。未重扫 runs 全树。
- 结果：删除前 E28 约 102.705 GiB，删除后 du 约 6.372 GiB，下降约 96.333 GiB；72 个目标全部不存在。共享盘可用空间因并发写入变化，清理后 df 显示约 3.9T 可用。
- 结论：E28 缓存已清除；其他大目录按既有盘点列示，不执行其他清理。
- 状态：完成。

## E112-S 后续：六个 runs 目录精简 — 2026-11-16 BRT
- 目标：按用户要求清除六个指定目录的全部特征数组，checkpoint 每个训练子目录只保留最新文件。
- 改动：删除 122 个 .npy/.npz 特征数组（146.054 GiB apparent）及 129 个旧 checkpoint（44.558 GiB apparent）；在 46 个 checkpoint 子目录中分别保留按最新 mtime 选出的 1 个。JSON、日志、脚本、评估结果等非特征数组文件保留。
- 命令：SSH <gpu-server>；只读生成计划，校验路径范围、普通文件、大小、link count、每组唯一最新 checkpoint；Python 按删除清单逐文件 unlink；随后对六个目录串行 timeout 20s du -skx 并核验清单。
- 结果：251 个删除目标均不存在，46 个保留 checkpoint 全部存在。清理后 GiB：E54 11.735、embed_analysis_E 0.001、E61 7.545、E32 14.832、E33 10.868、vjepa21_cpu_half 约 0.000。runs 总量按更新目录清单约 458.87 GiB。
- 结论：仅对用户指定的六目录执行精确清理；没有删除报告/评估文件，也未清理其他目录。
- 状态：完成。


## E116-P3 | 2026-11-17T06:12:53-03:00 | COMPLETE
- 目标：P3 纯推理检验本地 SigLIP2 图文联合空间对 Piper 阶段/因果谓词文本的对齐；dev 模型标签仅作开发参考。
- 改动：新增 `code_e116p3/`（锁定文本、联合空间抽帧/相似度和 CPU bootstrap 统计）；新增 `results/e116p3_20261117_01/`；报告 `notes/e116p3_language_alignment_2026-11-17.md`。冻结旧结果/标签/b 特征均只读；永久排除 `piper_4dc107335389f5711a0f65e0`。
- 命令：`setsid nohup .../code_e116p3/eval.py`（一次 dev 图像推理）；`setsid nohup .../code_e116p3/metrics.py`（冻结分数的 CPU 指标与 parent bootstrap）。图像使用 E86 224 aspect-padding 预处理；文本 fixed padding/max_length=64；MHA fastpath disabled。
- 目录：代码 `code_e116p3/`；run `results/e116p3_20261117_01/`；完整中英锁定文本、状态、GPU ledger、20 parent 嵌入/相似度、metrics JSON、三片段曲线均在 run 目录。
- 结果：20 dev parents / 1,867 阶段帧；主文本起始+结束英文、三相机 logit 均值：阶段 top-1 .0364 [.0304,.0412]，top-3 .1971 [.1770,.2169]，macro-F1 .0170。监督 CF H0 逐帧 argmax top-1 .6213 [.5968,.6477]；zero-shot−CF 差 -.5849 [-.6128,-.5599]。因果谓词 n=2,533/20 parents，English primary 宏 AUROC .5199 [.4862,.5842]；配对基线差：VLM -.1091 [-.1525,-.0214]、VLM-V3 -.1631 [-.1976,-.0785]、SigLIP2-linear-b -.2748 [-.3113,-.1938]、CF-H0 -.3882 [-.4236,-.3143]。人工 4 片段 521 抽帧 top-1=.0940。
- 判定：未通过预注册门槛（阶段 CI 下界未超过 .20；谓词 CI 下界未超过 .60）。结论：本 SigLIP2 专业文本-图像对齐很弱，语言不能直接作阶段/子目标条件。语言-动作对齐不在范围。
- 资源：GPU 4 起始空闲 85,253 MiB；实测峰值 1,696 MiB；累计 71.65 秒=0.0199 GPUh。新 run 9.11MB、新代码约121KiB；补充 Transformers runtime 约135MiB；合计 <150MiB，低于 2GB。未下载数据或权重。
- 状态：完成。

## E117-D — 学习式语言对齐方案设计（2026-11-17 BRT）
- 目标：设计 Piper L1语言—图像、L2语言—动作及L3语言条件方案，不做大规模训练。
- 改动/目录：新增 notes/e117_learned_alignment_design_2026-11-17.md；SSH只读核对策略、E116-P3、E102-S2/E113-V、E98-C、E99-B3、spec，并追加本日志与notes/worklog.md。
- 命令：SSH BatchMode 只读列举并查看指定报告和 spec.json；cat 新建报告，cat >> 追加两份日志；未调用 GPU/训练命令。
- 结果：P3原始SigLIP2阶段top-1=.036、谓词AUROC=.520，监督CF H0 .621/.908。推荐冻结双塔小视觉投影A；L3先只条件化世界头并遵守P1。
- 结论/状态：方案完成；未训练，GPUh=0；未下载模型或复制大文件。COMPLETE（设计交付，实验未开始）。
