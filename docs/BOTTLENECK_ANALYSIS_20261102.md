# Bottleneck analysis and next directions (Claude, 2026-11-02)

Evidence base: experiments E01–E23a (docs/experiment_log/EXPERIMENT_LOG.md) and the literature review
`wam_bottleneck_lit_review.md` (46 verified works, Codex terra, 2026-11-02).

## 1. Bottlenecks, ranked by how much they limit the real-robot result

| # | Bottleneck | Evidence | Why it matters |
|---|---|---|---|
| B1 | **Action effects are not identifiable from our data** (root cause) | E23a: a plausible but wrong action chunk is scored like the demo (44–53%); demo vs hold only 58–66%. E16: action effect only 1.6–2.2x sampling noise. E19: no scorer can rank E's samples. | 84 demos contain exactly one action per state, and the 1 s history + scene already predict the next 0.67 s. The dynamics can fit the data while largely ignoring the action ("action shortcut"). No loss on this data alone can teach what *another* action would have done. |
| B2 | **Small / transparent objects are lost at the input** | E20: 8-px bottle AP 0.46 at 12x20 vs 0.92 at 24x40; position inside a 16-px token barely known (5.2 vs 5.9 px baseline); transparent worst. E18: rebuild of the current bottle IoU 0.40 vs hands 0.74. | The grasp outcome depends on the bottle, the thing the latent sees worst. |
| B3 | **Prediction loses and drifts** | E20: predicted latent loses more (hand AP 0.84 -> 0.68). E23a: heads trained on real latents fail on rollouts (MAE 43 -> 133–144 steps); training on rollouts fixes the shift. | Any planning or scoring reads predicted latents. |
| B4 | **Selection target** | E09/E10/E19: demo-distance is unlearnable among good samples (E11 noise floor ~2 steps); E23a: time-to-success is learnable (Spearman 0.93) but blocked by B1. | Scoring is downstream of B1; fixing the scorer alone cannot help. |
| B5 | **No closed-loop evidence** | Every metric so far is offline; no real-robot rollout of E yet; no failure data. | Final validation is on the robot; offline proxies may not track success. |
| B6 | **Iteration speed** | One shared GPU; E24 ~20–36 h; supervision jobs died on client restarts. | Limits how many ideas we can test. |

## 2. Directions (my ranking; literature support in the review)

1. **Factorize the robot out of the world model (structural fix for B1, new).** The action's effect on the
   robot itself is known exactly: commanded joints -> forward kinematics (E22) -> projected hand pose in the image.
   Give the dynamics the *future hand geometry implied by the candidate action* (projected palm/fingertip points or
   rendered hand silhouette per step) as input, and learn only the object response (contact, grasp, carry, release).
   The model can no longer ignore the action, and hand prediction error disappears. Needs E22's camera fit.
   Backed by: IRASim / frame-aligned action conditioning, Seer/UWM (inverse dynamics), point-track WMs (ATM, Track2Act).
2. **Collect a small amount of action-diverse real data (only real fix for identifiability).** 20–40 reset-matched
   branch groups, 2–3 different actions each (~50–100 short chunks of 1–3 s): hold, slight miss, early/late close,
   different approach, recovery; plus 10–20 human corrections while running E. Balanced over stages; hold out groups.
   This also provides the first real outcomes for the value function and the first real counterfactual test.
3. **Resolution where it matters (B2).** E24's 24x40 object tokens, and next: build them from the native camera
   crop (640x384, as E21 used) instead of the upsampled 192x320 frames; evaluate on unseen transparent bottles.
4. **Rollout-consistency training (B3).** Already in E24 (3-chunk unroll); add a 1->2->4 chunk curriculum and
   re-anchor from the camera every control cycle at deployment.
5. **Counterfactual loss done carefully.** E24's InfoNCE now masks physically-equivalent negatives and static
   windows; with branch data (2) it becomes a true counterfactual loss with real alternative futures.
6. **Outcome value + uncertainty gate (B4)**, only after B1 passes the E23a test: time-to-success + stage milestones,
   trained on predicted latents, uncertainty rejection, medoid as the default pick.
7. **Real-robot baseline now (B5).** Run E (medoid of 6, replan every ~5 steps) vs the Phase A MSE head on a fixed set
   of start conditions; log success per stage. This anchors all offline numbers and produces the failure data.

Not recommended now: world-model RL fine-tuning of the policy (exploits B1 errors), unsupervised slot models
(data-hungry), more scorer variants on demo-distance targets.

## 3. Decision points for the user
- Robot time for (2) and (7): roughly one day of collection for the pilot set.
- Whether to make (1) the core of the next model (E25) once E22's camera fit is available.
