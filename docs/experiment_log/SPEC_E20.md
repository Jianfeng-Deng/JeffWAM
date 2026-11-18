# Spec E20: where is object (bottle) information lost in the image embedding? (design by Claude, 2026-11-01)

Same rules as SPEC_E18_E19.md (hosts, GPU 4 only, borrowed python, no installs, no VLM, setsid nohup, `[p]ython` patterns).
E19 may be training on GPU 4 at the same time: keep E20 memory < 12 GB, never kill E19. EXPERIMENT_LOG.md is also edited by
another Codex job: edit it only with a single atomic read-modify-write python replace right before writing, touching only the E20 entry.
Eval/probe only. No world-model training. Script: tools/analyze_image_embedding_20261101.py (+ helpers). Out: runs/embed_analysis_E_20261101.

Context: E18 showed the bottle is weakly encoded even without prediction: rebuild of the current frame hits bottle IoU 0.40 vs hands 0.74.
E's image path: 192x320 frame -> resize 168x280 -> DINOv2-B/14 (last 4 blocks fine-tuned, anchor 0.5) -> 12x20 tokens (16 px each) -> world module.

## Stages (features per token)
S0 frozen DINOv2-B at 168x280 (12x20)  — E's input setting, before fine-tuning
S1 frozen DINOv2-B at 336x560 (24x40)  — 2x resolution (upper reference; note: labels come from it, so S1 is partly circular)
S2 E fine-tuned encoder output (12x20), before the world module
S3 E world latent z_t (the tokens the decoders / dynamics use; get them via `world()` as in render_action_contrast)
S4 E predicted z^(t+10) under the demo action (labels from real t+10)
S5 Phase A ResNet18-GN encoder (12x20) if its checkpoint loads easily; otherwise skip and say so.
Wrist: frozen DINOv2-B tokens and E's wrist tokens (whatever grid E uses) for P6.

## Labels
Real frames -> frozen DINOv2-B 24x40 -> k-means (data/dinov2_regions_20261101/kmeans.pt): class masks bottle=15, hand=11, bin=13.
Bottle connected components on 24x40 (8 px cells) -> centroid (px) and width/height (px).
Frames: 20k training-episode frames (subsample) for probe fitting, all validation-episode frames (or >= 5k) for testing. Cache features as fp16 .npy.

## Probes and measurements (report bottle / hand / bin separately for every stage)
P1 occupancy at 24x40: per 12x20 token, predict the 2x2 sub-cell class occupancy (linear probe, logistic, 4 outputs per class);
   S1 predicts its own cell. Also a 3x3-neighbourhood probe (1-hidden-layer MLP on the 3x3 token patch, hidden 256) for a non-linear readout.
   Metrics on val: per-class AP, IoU at the best threshold, and token-level AP at 12x20 (any occupancy in the token).
P2 sub-token position: for tokens containing a bottle-component centroid, ridge-regress the centroid offset inside the token (dx, dy in px).
   Metrics: R2 and mean px error vs the "predict token centre" baseline. Same for hands.
P3 shift sensitivity (label-free): shift each val frame by 2, 4, 8, 16 px (horizontal and vertical, reflect pad) and re-encode (S0, S1, S2).
   Measure 1 - cosine between the original and shifted tokens (aligned by the token grid) on bottle tokens vs hand tokens vs background tokens,
   and a linear probe predicting the shift amount from the bottle tokens. Low sensitivity at 2-8 px = sub-token position is not encoded.
P4 object size: distribution of bottle width/height in px; % of bottles narrower than 16 px (one E token) and narrower than 8 px.
   Also bottle area share of each 12x20 token (how often a bottle covers < 25% of its token).
P5 transparent vs opaque: split bottle components by mean pixel saturation/brightness contrast against the local background
   (lowest-contrast tercile = "transparent-like", highest = "opaque/coloured") and repeat P1/P2 per split for S0, S2, S3.
P6 wrist: for val windows where a hand is within 2 tokens of a bottle (near grasp), ridge-probe the bottle-centroid-minus-hand-centroid
   vector (overhead labels, px) from (a) overhead S3 tokens around the hand (3x3), (b) pooled wrist tokens, (c) both. R2 and px error.
P7 figure: PCA (3 components -> RGB, fit per stage on val) for S0, S1, S2, S3 on 4 frames with bottles (incl. a transparent one),
   bottle mask contours overlaid; plus one row of P3 cosine-change maps for an 8 px shift. Save embed_pca.png, shift_maps.png.

## Report
results.json + a markdown summary table: rows = stages, columns = P1 AP/IoU (bottle, hand, bin), P2 R2/px (bottle, hand),
P3 change at 4 and 8 px (bottle, hand, bg). Separate tables for P4, P5, P6. Add a log entry E20 (template) before launch with status running;
after finishing fill Results; Conclusion = "pending (Claude)". Copy results.json, summary.md, both pngs to the local scratchpad
(prefix e20_). Final stdout: the tables and any deviation from this spec with the reason.
