# Spec E23a: can a value function learn time-to-success from E's latents? (learnability test only; design by Claude, 2026-11-01)
Rules as SPEC_E18_E19.md (GPU 4 shared with E20/E21/E22/E19-AW: stay < 12 GB, never kill them; atomic log edits; wrist cams never used).
No planning / candidate selection in this experiment.

Target: time-to-go T(t) = (last frame of episode - t) in steps (15 Hz); every demo ends in success. Train on log1p(T) (Huber), report in steps.
Also report normalized progress p = t / episode_length as a secondary target (same heads, separate output).

Inputs (E frozen: runs/ffn_phaseE_dinob_qhist15_20261031/best.pt; get z via world() as in render_action_contrast_20261101.py):
  R  real latent z(t) (12x20 tokens)
  P_k predicted latent after rolling E's dynamics on the DEMO actions for k chunks (k = 1..4, 10 steps each, chaining predicted
      latents as history), labelled with the REAL T(t + 10k). Use whatever multi-step interface wm.dyn supports; if chaining is not
      supported cleanly, document what was done.
Heads (small; same budget):
  V_pool  attention-pool over all 240 tokens -> MLP
  V_box   same but only tokens inside union boxes (boxes_v2, max-pooled to 12x20)
  B_q     baseline: 1 s joint history only (no image)    -> tells whether vision adds anything
  B_len   baseline: constant = train-mean T per relative position is NOT allowed (uses t); instead report predicting the train mean T.
Training sets: (a) real latents only; (b) real + predicted P_1..P_4 latents (the planning-relevant setting). Episode split as always.

Metrics on validation episodes (all windows):
 1. MAE (steps) and R2 of T; within-episode Spearman(V, -T); per tercile of the episode (early/mid/late).
 2. Shift test: evaluate each head on R vs P_k at the same target time; report MAE on P_k for k = 1..4 for heads trained on (a) vs (b).
 3. Action sensitivity (the key question): for 512 val windows, roll out H = 1..4 chunks with demo / hold (repeat current q) /
    reversed / other-window actions (same action family for all chunks). Report P(T_pred(demo) < T_pred(alt)) per alt and H, the mean
    margin in steps, and the margin divided by the seed/dropout spread (or by the std of T_pred over 4 dynamics noise seeds if E's dynamics is
    deterministic, use 4 heads trained with different seeds as an ensemble and report ensemble std).
 4. Figure: T_pred vs true T along 3 validation episodes (real and P_2), and demo-vs-hold curves over H.
Context: E12 progress head had R2 0.80 on real latents but almost no action sensitivity at one 10-step chunk (demo>hold 62%).
Script tools/train_value_ttg_20261101.py; run dir runs/value_ttg_E23a_20261101. Smoke test first. Log entry E23a before launch,
results after; Conclusion "pending (Claude)". Copy results json + summary.md + figure here with prefix e23a_. Final stdout: tables + deviations.
