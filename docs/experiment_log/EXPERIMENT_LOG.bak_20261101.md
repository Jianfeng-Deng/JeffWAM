# Experiment Log

## Project

- **Project root on GPU host:** `/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu` (`runs/` = outputs, `tools/` = scripts, `data/` = derived data).
- **Task:** Tianji dual arm + Wuji hands, grasp bottle -> place in bin.
- **Hardware:** one GPU only (physical GPU 4 on <gpu-server>). Python: `/mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python` (borrowed, nothing installed).
- **Data split:** 93 episodes (84 train / 9 val, episode-level split), 15 Hz, 54-D joints (A7+B7+L20+R20, rad).
- **Metric definitions:** Common policy metrics (validation, 2,289 windows): chunk RMSE (10x54, rad), first-5-step RMSE, hold baseline 0.101 rad; hand peak-speed ratio (1 = demo speed); grasp onset = first step where hand change from step 0 > 0.1 rad RMS: onset error (steps), missed, false onsets.

## Data assets

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

### E01
**Goal/what changed:** LeWAM upstream end-to-end (R0 s42/s7, R1 frozen drawer encoder, R2 +IDM 0.1, R3 no SIGReg).

**Runs:** `runs/lewam_upstream_R0_e2e_s42_20261031`, `runs/lewam_upstream_R0_e2e_s7_20261031`, `runs/lewam_upstream_R1_frozen_drawer_20261031`, `runs/lewam_upstream_R2_e2e_idm_20261031`, `runs/lewam_upstream_R3_e2e_noreg_20261031`

**Key results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/lewam_upstream_R0_e2e_s42_20261031 | 0.1543 | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) |
| runs/lewam_upstream_R0_e2e_s7_20261031 | 0.1534 | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) |
| runs/lewam_upstream_R1_frozen_drawer_20261031 | 0.2214 | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) |
| runs/lewam_upstream_R2_e2e_idm_20261031 | 0.1474 | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) |
| runs/lewam_upstream_R3_e2e_noreg_20261031 | 0.1496 | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) | n/a (see notes) |

**Conclusion:** keypoint latent too weak; frozen drawer encoder worst; R2 best (world gain +15%) but policy 0.147 rad, worse than hold. LeWAM as-is not viable.

### E02
**Goal/what changed:** MuJoCo object test (LeWAM scratch / ft from real / frozen, 5 folds).

**Runs:** `runs/lewam_mjc_scratch_f0_20261031`; `runs/lewam_mjc_scratch_f1_20261031`; `runs/lewam_mjc_scratch_f2_20261031`; `runs/lewam_mjc_scratch_f3_20261031`; `runs/lewam_mjc_scratch_f4_20261031`; `runs/lewam_mjc_ftreal_f0_20261031`; `runs/lewam_mjc_ftreal_f1_20261031`; `runs/lewam_mjc_ftreal_f2_20261031`; `runs/lewam_mjc_ftreal_f3_20261031`; `runs/lewam_mjc_ftreal_f4_20261031`; `runs/lewam_mjc_ftfrozen_f0_20261031`; `runs/lewam_mjc_ftfrozen_f1_20261031`; `runs/lewam_mjc_ftfrozen_f2_20261031`; `runs/lewam_mjc_ftfrozen_f3_20261031`; `runs/lewam_mjc_ftfrozen_f4_20261031`

**Key results:**
| Evaluation | held_out_rows | rows_bottle_moved_gt_5mm | latent.gain_vs_persistence_all | latent.gain_vs_persistence_moved | latent.within_group_retrieval_top1 | forecast.rmse_mm_moved | direct.rmse_mm_moved | forecast.within_group_spearman_magnitude |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | 1575 | 201 | 0.8307 | 0.8224 | 0.3156 | 21.06 | 16.39 | 0.04150 |
| ftreal | 1575 | 201 | 0.5236 | 0.6780 | 0.3670 | 19.52 | 16.37 | 0.07692 |
| ftfrozen | 1575 | 201 | 0.6414 | 0.5985 | 0.1403 | 22.88 | 16.52 | 0.03462 |

**Conclusion:** predicted futures get bottle direction partly right and separate candidates ~3x chance, but no better than direct regression and cannot rank candidates; frozen worst. Later (E14) shown that the 192-D vector bottleneck, not learning, limited this.

### E03
**Goal/what changed:** Phase A world-action net (ResNet18-GN spatial tokens 12x20, change-weighted future image, MSE policy), variants latent_image vs image_only.

**Runs:** `runs/wan_phaseA_image_only_20261031`; `runs/wan_phaseA_latent_image_20261031`; score reports `score_phaseB_metrics_latest.json`

**Key results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/wan_phaseA_image_only_20261031 | 0.07252 | n/a (see notes) | 0.8176 | 2.292 | 0.3454 | 0.1297 | n/a (see notes) |
| runs/wan_phaseA_latent_image_20261031 | 0.07182 | n/a (see notes) | 0.7749 | 2.194 | 0.3307 | 0.1433 | n/a (see notes) |

**Conclusion:** future prediction strongly action-dependent (+49% vs copy, ~85% of gain lost with wrong action); latent loss adds nothing; MSE policy peaked epoch 5-6 (0.0685) then overfit; hands blurred (peak speed 0.77).

Phase A reports hand peak-speed ratio as `hand_peak_vel_ratio` (single output).

### E04
**Goal/what changed:** Phase B flow-matching head: A (no foresight) vs B (action-free foresight fed to policy).

**Runs:** `runs/ffn_phaseB_A_noforesight_20261031`; `runs/ffn_phaseB_B_foresight_20261031`

**Key results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/ffn_phaseB_A_noforesight_20261031 | 0.08145 | n/a (see notes) | 0.9259 | 2.735 | 0.4217 | 0.1459 | 0.4821 |
| runs/ffn_phaseB_B_foresight_20261031 | 0.07978 | n/a (see notes) | 0.9747 | 2.479 | 0.3440 | 0.1874 | 0.4828 |

**Conclusion:** flow fixes hand blur (speed ~1.0); foresight helps one-sample error and timing; neither beats MSE on timing.

### E05
**Goal/what changed:** Sample selection (noise scale, medoid, world-model scoring) on A/B.

**Runs:** `runs/ffn_phaseB_A_noforesight_20261031`, `runs/ffn_phaseB_B_foresight_20261031` (selection reports: `ffn_phaseB_A_noforesight_20261031/sample_selection_best_ep30.json`, `ffn_phaseB_B_foresight_20261031/sample_selection_best_ep26.json`)

**Key results:**
| Run | Strategy | chunk_rmse | first5_rmse | hand_speed_ratio | onset_err_steps | onset_miss | onset_false |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (best ep30) | T=0.5 single | 0.07321 | 0.04179 | 0.7683 | 2.534 | 0.4137 | 0.08820 |
| A (best ep30) | T=0.5 medoid8 | 0.07012 | 0.04037 | 0.7357 | 2.498 | 0.4110 | 0.06615 |
| A (best ep30) | T=0.5 world8 | 0.07206 | 0.04120 | 0.7833 | 2.548 | 0.4391 | 0.08301 |
| A (best ep30) | T=0.5 best8 (oracle) | 0.05653 | 0.03606 | 0.7539 | 1.893 | 0.3146 | 0.03243 |
| B (best ep26) | T=0.5 single | 0.07668 | 0.04263 | 0.8626 | 2.444 | 0.3534 | 0.1524 |
| B (best ep26) | T=0.5 medoid8 | 0.07573 | 0.04198 | 0.8441 | 2.420 | 0.3628 | 0.1427 |
| B (best ep26) | T=0.5 world8 | 0.07599 | 0.04275 | 0.8623 | 2.494 | 0.3695 | 0.1420 |
| B (best ep26) | T=0.5 best8 (oracle) | 0.06813 | 0.03869 | 0.8395 | 2.131 | 0.3199 | 0.1122 |

**Conclusion:** low noise best for near-term steps; world-model scoring ~ random; oracle best-of-8 far better than anything selectable.

### E06
**Goal/what changed:** C = B + wrist cameras; D = B + 1 s joint history with past-token prediction.

**Runs:** `runs/ffn_phaseC_wrist_20261031`; `runs/ffn_phaseD_qhist15_20261031`

**Key results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/ffn_phaseC_wrist_20261031 | 0.07770 | 0.04598 | 0.9841 | 2.380 | 0.3400 | 0.1498 | 0.4795 |
| runs/ffn_phaseD_qhist15_20261031 | 0.07684 | 0.04547 | 0.9998 | 2.428 | 0.3681 | 0.1297 | 0.4804 |

**Conclusion:** both help modestly (~3%); D best RMSE, C best timing; gains late in chunk.

### E07
**Goal/what changed:** Encoder probes (DINOv2-S/B frozen vs our encoder vs random).

**Runs:** `probe_encoders_dinov2_20261031.json` (no run directory recorded in the JSON).

**Key results:**
| Encoder | real.state.r2_hands | real.motion.r2_hands | mujoco_oracle.rmse_mm_moved | mujoco_oracle.within_group_spearman |
| --- | --- | --- | --- | --- |
| dinov2_s | 0.5184 | 0.2186 | 14.94 | 0.1413 |
| dinov2_b | 0.5459 | 0.2403 | 14.15 | 0.2071 |
| ours_B | 0.3596 | 0.2282 | 15.71 | 0.2236 |
| random_resnet | 0.1848 | 0.09233 | 15.27 | 0.2144 |

**Conclusion:** DINOv2-B much better hand-state readout (hand R2 0.55 vs 0.36); MuJoCo pooled-token test saturated (random 0.57).

### E08
**Goal/what changed:** E = D + DINOv2-B encoder, last 4 blocks fine-tuned (lr 2e-5) + anchor loss 0.5.

**Runs:** `runs/ffn_phaseE_dinob_qhist15_20261031`

**Key results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/ffn_phaseE_dinob_qhist15_20261031 | 0.07409 | 0.04405 | 0.9875 | 2.382 | 0.3494 | 0.1038 | 0.4562 |

**Conclusion:** best flow model; ahead of D from epoch 6 (best ep 20: 0.0741, onset 2.38, false 10%).

### E09
**Goal/what changed:** Action scorer v1 (listwise incl. demo, frozen DINOv2 features + wrists + history).

**Runs:** `runs/action_scorer_v1_20261031`

**Key results:**
| Metric | Value |
| --- | --- |
| epoch | 10 |
| spearman_score_vs_true_rank | -0.1571 |
| demo_ranked_first_vs_shifted_and_blurred | 0.4294 |

**Conclusion:** ranked backwards (learned real-vs-generated / hand-motion shortcut).

### E10
**Goal/what changed:** Action scorer v2 (pairwise ranking over policy samples).

**Runs:** `runs/action_scorer_v2_pairwise_20261031`

**Key results:**
| Metric | Value |
| --- | --- |
| epoch | 2 |
| spearman_score_vs_true_rank | 0.3784 |
| share_picks_noise1 | 0.05461 |
| demo_ranked_first_vs_shifted_and_blurred | 0.3246 |

**Conclusion:** correct sign (+0.41) but mostly "prefer low noise"; within-group ranking weak even on train.

### E11
**Goal/what changed:** Grasp-timing noise floor (nearest neighbours across demos).

**Runs:** `timing_noise_floor_20261031.json` (no run directory recorded in the JSON).

**Key results:**
| Metric | Value |
| --- | --- |
| windows | 26351 |
| episodes | 93 |
| share_windows_with_onset | 0.3556 |
| nearest_neighbour_copy.onset_err_steps | 2.782 |
| nearest_neighbour_copy.miss | 0.3898 |
| nearest_neighbour_copy.false | 0.2299 |
| median_of_5_neighbours.onset_err_steps | 2.701 |
| median_of_5_neighbours.miss | 0.3933 |
| median_of_5_neighbours.false | 0.2064 |
| random_other_window.onset_err_steps | 4.023 |
| random_other_window.miss | 0.6471 |
| random_other_window.false | 0.3571 |

**Conclusion:** ~2 steps of onset disagreement even for closest matches, flat across distance -> demonstrator randomness; MSE head 2.19 near floor; best-of-N oracle gap is luck.

### E12
**Goal/what changed:** Progress-based scorer (progress head on E latents; score = progress of predicted future).

**Runs:** `runs/progress_scorer_v1_20261101`

**Key results:**
Progress-head diagnostics:
| Metric | Value |
| --- | --- |
| progress_r2_real_frames | 0.7962 |
| progress_r2_predicted_futures | 0.8012 |
| demo>hold | 0.6173 |
| demo>reversed | 0.5701 |
| spearman_score_vs_closeness | -0.04446 |
Selection metrics:
| Strategy | chunk_rmse | first5_rmse | hand_speed_ratio | onset_err_steps | onset_miss | onset_false |
| --- | --- | --- | --- | --- | --- | --- |
| progress_top1 | 0.07658 | 0.04522 | 1.020 | 2.440 | 0.3588 | 0.1278 |
| random_single | 0.07496 | 0.04441 | 0.9990 | 2.554 | 0.3869 | 0.1070 |
| half_noise_single | 0.06988 | 0.04011 | 0.8351 | 2.415 | 0.3882 | 0.07717 |
| medoid16 | 0.06873 | 0.03968 | 0.7917 | 2.448 | 0.3936 | 0.07588 |
| oracle_best16 | 0.05790 | 0.03584 | 0.8952 | 1.186 | 0.1780 | 0.02983 |

**Conclusion:** head accurate (R2 0.80) but futures barely depend on action over 10 steps (demo>hold 62%, demo>reversed 57%); picks ~ random. Simple rules on E (medoid/half-noise) beat MSE head on near-term error.

### E13
**Goal/what changed:** World-model image quality + decoder bottleneck test on E.

**Runs:** `runs/ffn_phaseE_dinob_qhist15_20261031`; reports `wm_viz_E_20261101/metrics.json` and `decoder_test_E_boxes.json`.

**Key results:**
| Metric | Value |
| --- | --- |
| metrics.json: e_epoch | 20 |
| metrics.json: windows | 2289 |
| metrics.json: changed_pixel_share | 0.2535 |
| metrics.json: pred.l1_changed | 0.07131 |
| metrics.json: pers.l1_changed | 0.1365 |
| decoder_test_E_boxes.json: recon_t.sharpness_full | 0.3718 |
| decoder_test_E_boxes.json: pred_t10.sharpness_full | 0.3633 |
| decoder_test_E_boxes.json: recon_t.sharpness_in_boxes | 0.2259 |
| decoder_test_E_boxes.json: pred_t10.sharpness_in_boxes | 0.2203 |

**Conclusion:** blur comes from decoding: decoded current frame as blurred as prediction (sharpness 0.37 full / 0.23 in boxes); bottles vanish even without prediction.

### E14
**Goal/what changed:** F = E + semantic box weighting (bg 0.15) on recon/future/foresight pixel losses.

**Runs:** `runs/ffn_phaseF_semboxes_20261101`; decoder report `wm_viz_E_20261101/decoder_test_F_boxes.json`.

**Key results:**
| Run | policy_chunk_rmse_rad | policy_first5_rmse_rad | hand_peak_vel_ratio_single | onset_abs_err_steps | onset_miss_rate | onset_false_rate | changed_region_gain_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| runs/ffn_phaseF_semboxes_20261101 | 0.07389 | 0.04411 | 0.9892 | 2.348 | 0.3454 | 0.1031 | 0.4381 |
Decoder comparison:
| Metric | Value |
| --- | --- |
| E recon_t sharpness_in_boxes | 0.2259 |
| F recon_t sharpness_in_boxes | 0.2470 |
| F pred_t10 sharpness_in_boxes | 0.2372 |
| F box_pixel_share | 0.4897 |

**Conclusion:** in-box sharpness 0.23 -> 0.25 only; policy equal to E (best 0.0739). Weighting the L1 loss does not change what is decoded.

### E15
**Goal/what changed:** Region-focused JiT decoder (x-prediction + v-loss, 8 px patches in boxes only, latent-only conditioning, 6,000 steps) on frozen E latents.

**Runs:** `runs/region_jit_E_latentonly_20261101`

**Key results:**
| Metric | Value |
| --- | --- |
| step | 6000 |
| recon_t/jit.l1_in_box | 0.03588 |
| recon_t/jit.psnr_in_box | 22.20 |
| recon_t/jit.sharpness_in_box | 0.6033 |
| recon_t/l1_decoder.l1_in_box | 0.03143 |
| recon_t/l1_decoder.psnr_in_box | 23.57 |
| recon_t/l1_decoder.sharpness_in_box | 0.2204 |
| recon_t/jit_seed_spread | 0.02279 |
| pred_t10/jit.l1_in_box | 0.04563 |
| pred_t10/jit.psnr_in_box | 19.92 |
| pred_t10/jit.sharpness_in_box | 0.6245 |
| pred_t10/l1_decoder.l1_in_box | 0.04002 |
| pred_t10/l1_decoder.psnr_in_box | 21.07 |
| pred_t10/l1_decoder.sharpness_in_box | 0.2269 |
| pred_t10/jit_seed_spread | 0.02747 |

**Conclusion:** renders separate fingers, bin texture and bottle positions consistently across seeds -> E's latent DOES contain most object information; L1 decoder hid it. Fine detail (transparent bottle in closing hand) still weak; ~60% of remaining in-box error is seed variation.

### E16
**Goal/what changed:** Action-contrast rendering: region-JiT renders of E's predicted t+10 under demo / hold / reversed / other actions with fixed noise; tests whether predicted object outcomes depend on the action.

**Date:** 2026-11-01. **Command:** `tools/render_action_contrast_20261101.py --e-run runs/ffn_phaseE_dinob_qhist15_20261031 --jit-run runs/region_jit_E_latentonly_20261101 --boxes data/dinov2_regions_20261101/boxes_v2_24x40.npy` (256 validation windows, 50 Euler steps, same noise for every action).

**Runs:** `runs/action_contrast_E_20261101` (action_contrast.json, action_contrast_zoom.png)

**Key results (in-box pixel L1):**
| Action | Error vs real t+10 | Change vs demo render |
| --- | --- | --- |
| demo | **0.0455** | — |
| hold still | 0.0641 | 0.0445 |
| other moment's action | 0.0632 | 0.0432 |
| reversed | 0.0757 | 0.0600 |
| (seed spread, demo, two noises) | — | 0.0276 |

Latent change vs demo per token (inside / outside boxes): hold 4.12 / 3.41, other 4.21 / 3.53, reversed 6.29 / 5.37; demo change from z(t): 7.59 / 7.31.

**Conclusion:** the world model's predicted futures depend clearly on the action: action effects are 1.6-2.2x the sampling noise, mostly in hand pose (open vs closed fingers) and arm placement, more inside the object boxes than outside; the demo action gives the future closest to reality (29-40% lower error than the alternatives). Static objects stay put over 0.67 s, as expected. So the dynamics are action-sensitive and the earlier scorer failures (E09, E10, E12) come from scorer design (global progress / sample ranking), not from the world model ignoring actions.

**Status:** done

### E17
**Date:** 2026-11-01. **Goal/what changed:** Region-JiT v2 (fixes for visible patch seams): bilinear latent conditioning instead of copying each 16-px token to four 8-px patches; one-patch context ring around boxes (metrics still on original boxes); patch cap 1000; larger model (width 512, depth 12, 8 heads, bottleneck 64); 20,000 steps; EMA 0.9995; same frozen run-E latents, latent-only conditioning. Smoke test (200 steps) passed at about 0.37 s/step.

**Command:** `tools/train_region_jit_decoder_v2_20261101.py --e-run runs/ffn_phaseE_dinob_qhist15_20261031 --boxes data/dinov2_regions_20261101/boxes_v2_24x40.npy --run-dir runs/region_jit_v2_E_latentonly_20261101` (defaults: `--steps 20000 --eval-every 5000 --d 512 --depth 12 --heads 8 --ring 1 --cond-upsample bilinear --max-tok 1000 --ema 0.9995`).

**Runs:** `runs/region_jit_v2_E_latentonly_20261101`

**Results:**

| Eval step | Current JiT L1 | Current JiT PSNR | Current JiT sharpness | Current seed spread | Predicted t+10 JiT L1 | Predicted t+10 JiT PSNR | Predicted t+10 JiT sharpness | Predicted t+10 seed spread | E decoder reference current L1 | E decoder reference predicted t+10 L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.034835 | 22.3830 | 0.634006 | 0.021353 | 0.044590 | 20.0121 | 0.641330 | 0.025435 | 0.031426 | 0.040023 |
| 10,000 | 0.032933 | 22.7993 | 0.701704 | 0.017436 | 0.043282 | 20.1537 | 0.703006 | 0.020757 | 0.031426 | 0.040023 |
| 15,000 | 0.032368 | 22.9300 | 0.718549 | 0.015790 | 0.042935 | 20.1918 | 0.718062 | 0.018642 | 0.031426 | 0.040023 |
| 20,000 | 0.032055 | 23.0219 | 0.723318 | 0.014508 | 0.042841 | 20.2153 | 0.721383 | 0.017146 | 0.031426 | 0.040023 |

**Conclusion:** The patch seams are fixed: no visible block edges in `zoom_step20000.png`. At step 20,000 the current-frame in-box L1 is 0.0321 (E's L1 decoder: 0.0314) at 3.3x the sharpness (0.72 vs 0.22). The predicted t+10 L1 is 0.0428 (L1 decoder: 0.0400): a sharp sampled future pays a small L1 cost against an averaged one. Seed spread kept falling (0.0145 / 0.0171), so the rendered detail is increasingly set by the latent rather than the noise. Fingers, cables, arm joints, bin weave, and held objects are rendered consistently across seeds, now and 10 steps ahead. Gains after 10k steps are small (L1 -0.0009), so the decoder is near its limit with these latents. Remaining weakness: transparent/white bottles render faint or partly missing, and some bin contents are coarse. This points to the 12x20 latent resolution, not the decoder. Takeaway: E's latent holds the object-level information the task needs, and region-JiT is a usable visual-feedback / training-loss decoder.

**Status:** done

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

## Open decisions / next

- Real-robot comparison: E best (medoid or half-noise sampling, ~5 steps executed before replanning) vs Phase A MSE head.
- If E16 shows action-dependent object outcomes: fix scorer design; if not: train the world model through the region-JiT decoder.

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
