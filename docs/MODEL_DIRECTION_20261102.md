# Is the model chasing detail? Where should it go next? (Claude, 2026-11-02)

Evidence: E25 capacity audit (`runs/e25_capacity_audit_20261102.md`), JEPA survey (`docs/lit_jepa_20261102.md`,
37 works), interaction-factorization analysis (`docs/INTERACTION_FACTORIZATION_ANALYSIS_20261102.md`), E24/E25/E26a.

## 1. Answer: yes, the training signal is dominated by RGB future prediction

E25's late-training weighted scalar losses (sum 0.981):

| Term | Weight | Weighted | Share | What it is |
|---|---:|---:|---:|---|
| future_img + fore_img | 5.0 + 5.0 | 0.536 | **55%** | RGB 192x320 at t+5, t+10 from the predicted latent (action and null-action), change-weighted (0.2 floor) |
| flow (policy) | 1.0 | 0.167 | 17% | the policy |
| object + mask | 1.0 + 1.0 | 0.140 | 14% | object masks/geometry/held, current + rollout |
| roll_latent (the JEPA term) | 0.5 | 0.038 | **4%** | cosine on all 240 pooled tokens, t+10..t+60, uniform weight |
| idm, anchor, recon, q, cf | | 0.100 | 10% | |

- The JEPA latent-prediction term is 4% of the loss; future-pixel prediction is 55%. Scalar size is not gradient share
  (not logged), but a 14x gap is large. In practice E/E24/E25 are pixel-predictive world models with a JEPA term on top.
- The pixel loss is change-weighted, so it emphasizes what moves most in pixels: the arms. The small bottle gets little.
  This fits D2 (predictions worse than copy for the bottle) and E25's results (hands now good, bottle still not).
- The decoder is small (1.35 M params, ~4.5% compute) but takes over half the training signal.
- Compute: ~70% is DINO encoding, and roughly half of that is 48 no-grad future-frame encodings per micro-batch, run for
  all 6 future chunks even though the sampled rollout length averages 3.4. Caching or encoding only the frames used is a
  cheap ~25-35% speed-up.
- Latent targets come from the online encoder (stop-grad), not an EMA target; all 240 tokens weigh the same;
  high-resolution 24x40 features are used only for current-frame masks, the dynamics runs at 12x20.

## 2. The user's decoder formulation, against this evidence

The current model already implements the strong form of "decoder and dynamics both read the latent, with mutual
constraints": it decodes the predicted future to RGB and compares with the real frame. What it lacks:
- No split between what the dynamics must predict and what is appearance only, so the dynamics is asked to predict
  texture/lighting change in moving regions.
- Region weighting by change (mostly arms), not by task objects.
- A 55% weight that makes pixels the main objective.
Proposed formulation (from the previous analysis): z = [shared | appearance | hidden], orthogonal learned subspaces;
decoder reads shared+appearance, dynamics predicts shared+hidden, appearance is carried from the current frame when
decoding the predicted future; image loss region-weighted by clean masks (hands, bottles, bin high; background low).
The survey supports a light reconstruction anchor for small data (latents can be "physically lazy"), and warns that
reconstruction-free latents can still keep distractors (CF-JEPA), so the decoder should stay, at a lower weight.

## 3. Ranked plan

1. **E27a: loss-balance ablations (cheap, decisive).** Fine-tune from E25 for 4 epochs each, same data, same eval:
   - A0 current weights (control);
   - A1 image losses 5 -> 1, roll_latent 0.5 -> 2;
   - A2 as A1, image loss weighted by clean masks (hands/bottles/bin 1.0, rest 0.05) instead of change;
   - A3 no image losses (latent + object only).
   Log per-term gradient norms at the encoder and dynamics. Judge by the action-sensitivity gate, bottle vs copy
   (moving windows), policy RMSE. Also implement target-frame caching here (speed-up for all later runs).
2. **E27b: subspace split (the user's formulation)**, built on the best of A0-A3.
3. **E26: gated robot-to-object routing**, on clean objects (E26a) and the balance found above.
4. **Small matched-start action-diverse data**, still the only test of influence/causation.
5. Later: flow policy as proposer + model as short-horizon critic; slower phase-level time scale.

Pending inputs: E25 value test (main gate) and E26a baselines with bin, both due ~14:30-15:00 BRT.
