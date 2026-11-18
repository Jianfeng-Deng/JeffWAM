# Experiment notes (author: Claude; ground truth for the log — conclusions only, numbers come from _run_summary_20261101.json)

Project root on GPU host: /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu (runs/ = outputs, tools/ = scripts, data/ = derived data)
Hardware: one GPU only (physical GPU 4 on <gpu-server>). Python: /mnt/workspace/colleague-c/OpenWAM/.venv-openwam/bin/python (borrowed, nothing installed).
Task: Tianji dual arm + Wuji hands, grasp bottle -> place in bin. 93 episodes (84 train / 9 val, episode-level split), 15 Hz, 54-D joints (A7+B7+L20+R20, rad).
Common policy metrics (validation, 2,289 windows): chunk RMSE (10x54, rad), first-5-step RMSE, hold baseline 0.101 rad;
hand peak-speed ratio (1 = demo speed); grasp onset = first step where hand change from step 0 > 0.1 rad RMS:
onset error (steps), missed, false onsets.

## Data built
- D1 data/lewam_upstream_h5_20261031: overhead cam_high letterboxed 224, canonical q/cmd (verified).
- D2 data/mujoco_transition_224_20261031 + data/mujoco_lewam_h5_20261031: 1,575 counterfactual sim transitions, 196 start groups, 5 group folds.
- D3 data/real_workspace_192x320_20261031: overhead workspace crop rows 96..480, 320x192.
- D4 data/real_wrist_144x192_20261031: left/right wrist cameras 192x144 (row-aligned with D3).
- D5 /mnt/workspace/dengjianfeng/models/dinov2_official_20261031: official DINOv2 ViT-S/14, ViT-B/14 (Meta server, SHA256SUMS recorded).
- D6 data/dinov2b_feats_20261031: frozen DINOv2-B pooled features (overhead 6x10, wrists 5x7) for all frames.
- D7 data/policy_samples_D_20261031: 16 samples/window from run D best (8 at noise 1.0, 8 at 0.5).
- D8 data/dinov2_regions_20261101: DINOv2 k-means (k=16) patch clusters 24x40; boxes_v2 = per-type boxes for clusters 11 (hands, dark bottles), 13 (bins), 15 (bottles/cups), min component 2/6/3 patches, 2-patch margin; ~49% coverage.

## Experiments (in order)
E01 LeWAM upstream end-to-end (R0 s42/s7, R1 frozen drawer encoder, R2 +IDM 0.1, R3 no SIGReg). Conclusion: keypoint latent too weak; frozen drawer encoder worst; R2 best (world gain +15%) but policy 0.147 rad, worse than hold. LeWAM as-is not viable.
E02 MuJoCo object test (LeWAM scratch / ft from real / frozen, 5 folds). Conclusion: predicted futures get bottle direction partly right and separate candidates ~3x chance, but no better than direct regression and cannot rank candidates; frozen worst. Later (E14) shown that the 192-D vector bottleneck, not learning, limited this.
E03 Phase A world-action net (ResNet18-GN spatial tokens 12x20, change-weighted future image, MSE policy), variants latent_image vs image_only. Conclusion: future prediction strongly action-dependent (+49% vs copy, ~85% of gain lost with wrong action); latent loss adds nothing; MSE policy peaked epoch 5-6 (0.0685) then overfit; hands blurred (peak speed 0.77).
E04 Phase B flow-matching head: A (no foresight) vs B (action-free foresight fed to policy). Conclusion: flow fixes hand blur (speed ~1.0); foresight helps one-sample error and timing; neither beats MSE on timing.
E05 Sample selection (noise scale, medoid, world-model scoring) on A/B. Conclusion: low noise best for near-term steps; world-model scoring ~ random; oracle best-of-8 far better than anything selectable.
E06 C = B + wrist cameras; D = B + 1 s joint history with past-token prediction. Conclusion: both help modestly (~3%); D best RMSE, C best timing; gains late in chunk.
E07 Encoder probes (DINOv2-S/B frozen vs our encoder vs random). Conclusion: DINOv2-B much better hand-state readout (hand R2 0.55 vs 0.36); MuJoCo pooled-token test saturated (random 0.57).
E08 E = D + DINOv2-B encoder, last 4 blocks fine-tuned (lr 2e-5) + anchor loss 0.5. Conclusion: best flow model; ahead of D from epoch 6 (best ep 20: 0.0741, onset 2.38, false 10%).
E09 Action scorer v1 (listwise incl. demo, frozen DINOv2 features + wrists + history). Conclusion: ranked backwards (learned real-vs-generated / hand-motion shortcut).
E10 Action scorer v2 (pairwise ranking over policy samples). Conclusion: correct sign (+0.41) but mostly "prefer low noise"; within-group ranking weak even on train.
E11 Grasp-timing noise floor (nearest neighbours across demos). Conclusion: ~2 steps of onset disagreement even for closest matches, flat across distance -> demonstrator randomness; MSE head 2.19 near floor; best-of-N oracle gap is luck.
E12 Progress-based scorer (progress head on E latents; score = progress of predicted future). Conclusion: head accurate (R2 0.80) but futures barely depend on action over 10 steps (demo>hold 62%, demo>reversed 57%); picks ~ random. Simple rules on E (medoid/half-noise) beat MSE head on near-term error.
E13 World-model image quality + decoder bottleneck test on E. Conclusion: blur comes from decoding: decoded current frame as blurred as prediction (sharpness 0.37 full / 0.23 in boxes); bottles vanish even without prediction.
E14 F = E + semantic box weighting (bg 0.15) on recon/future/foresight pixel losses. Conclusion: in-box sharpness 0.23 -> 0.25 only; policy equal to E (best 0.0739). Weighting the L1 loss does not change what is decoded.
E15 Region-focused JiT decoder (x-prediction + v-loss, 8 px patches in boxes only, latent-only conditioning, 6,000 steps) on frozen E latents. Conclusion: renders separate fingers, bin texture and bottle positions consistently across seeds -> E's latent DOES contain most object information; L1 decoder hid it. Fine detail (transparent bottle in closing hand) still weak; ~60% of remaining in-box error is seed variation.
E16 (running) Action-contrast rendering: region-JiT renders of E's predicted t+10 under demo / hold / reversed / other actions with fixed noise; tests whether predicted object outcomes depend on the action.

## Code changes (all under tools/, uploaded from local scratchpad)
convert_lewam_upstream_h5_20261031.py, train_lewam_upstream_episode_split_20261031.py (IDM param-group fix), eval_lewam_upstream_e2e_20261031.py,
render_mujoco_transitions_224_20261031.py, build_mujoco_lewam_h5_20261031.py, eval_lewam_mujoco_object_20261031.py,
convert_real_workspace_crop_20261031.py, train_world_action_net_20261031.py (Phase A), score_phaseA_policy_20261031.py,
train_foresight_flow_20261031.py (Phase B), eval_sample_selection_20261031.py, eval_good_samples_20261031.py,
convert_real_wrist_20261031.py, train_ffn_cd_20261031.py (C/D; later +DINOv2 encoder/anchor for E, +semantic boxes for F; defaults keep earlier runs reproducible),
probe_encoders_dinov2_20261031.py (resumable), precompute_dinov2_feats_20261031.py, gen_policy_samples_20261031.py,
train_action_scorer_20261031.py (v1 listwise; --loss v2 pairwise), diag_action_scorer_20261031.py, measure_timing_noise_floor_20261031.py,
train_progress_scorer_20261101.py, viz_world_model_20261101.py, test_decoder_bottleneck_20261101.py (+in-box metrics),
cluster_dinov2_regions_20261101.py, build_semantic_boxes_20261101.py, train_region_jit_decoder_20261101.py, render_action_contrast_20261101.py.

## Open decisions / next
- Real-robot comparison: E best (medoid or half-noise sampling, ~5 steps executed before replanning) vs Phase A MSE head.
- If E16 shows action-dependent object outcomes: fix scorer design; if not: train the world model through the region-JiT decoder.
