# Do we need robot / object / background disentanglement? (Claude, 2026-11-02)

Inputs: user's hypothesis; E24/E24r/E25-stage0 diagnostics; literature survey
`docs/lit_interaction_factorization_20261102.md` (36 verified works, Codex terra).

## 1. Short answer
Yes, but as **structure in the dynamics**, not as three separately learned latent spaces.
The useful factorization is about *how the action is allowed to reach each part of the scene*:

    action -> robot tokens -> (learned, gated messages) -> object tokens ;  scene tokens: mostly identity

Every token keeps its own state and is predicted as "copy + change". The action can only change robot tokens
directly; it can change an object only through a learned robot->object message, scaled by a learned gate.
No per-object rule is written: the same small network computes the gate for every robot-object pair.

## 2. Why: what the diagnostics say
- E25 stage0 D2: swapping the action moves the predicted bin by 3-10 (bin real change ~0) and the bottle by
  1.6-3.8x its real change already at 1 chunk. The current dynamics lets the action act on every token directly
  (the action is injected globally), so nothing prevents "action moves the bin".
- D2 also: even with the true action, prediction error is larger than the real change (bottle 6.8 vs 2.8): worse than
  copy. A copy+change form starts at copy by construction.
- D3: motion inputs barely matter, so the failure is not a history shortcut; it is the lack of structure.
- D1/E24r: object tokens themselves are unreliable (bottle missed 62% at t+0; hand head learned forearms).
  Any factorization only helps if the object slots are right. This is the first thing to fix.

## 3. The user's three intuitions, mapped to mechanisms

| Intuition | Keep | Change | Literature |
|---|---|---|---|
| Background and far objects: low weight / prior to stay static | Identity + residual prediction for every token; a separate action-free path for passive change | Do **not** give them low loss weight. A model can then win by predicting "nothing moves" and miss the rare grasp. Use copy-relative losses and balanced weighting of moving vs static samples. A static bin's DINO token can still change (occlusion, lighting, ViT context mixing). | Finn 2016 (CDNA copy channel), Ebert 2017, survey failure mode 4 |
| Arm-object latents carry mutual information indicating influence | Influence as a learned, state-dependent quantity per robot-object pair | Implement it as a **gate** on the action-dependent change, not as an MI-maximization loss. Unconditional MI between hand and object latents rewards shortcuts (shared background, correlated trajectories). The proper quantity is conditional MI between the action and the object's next state given the current state (CAI), and it is ~0 / not estimable when each state has one demonstrated action. | CAI (Seitzer 2021), SPARTAN 2025, FCDL 2024, NRI 2018 |
| Regions near the hand get high weight | Emphasize interaction events in training | Weight by **observed** motion/contact events (point tracks or mask motion, computed offline) rather than by hand distance. Distance is a soft input to the gate, not a rule: a held bottle moves far from where it was grasped; overlap in a top view hides depth. | Track2Act, ATM, Im2Flow2Act; survey "time and contact" |

## 4. What factorization can and cannot fix
- Can fix B (action moves everything): if the action only reaches objects through gates, an unrelated object needs
  an open gate to move, and a sparsity prior keeps most gates closed. This is a structural guarantee, not a hope.
- Can fix C (worse than copy): copy+change form plus copy-relative loss.
- Cannot fix A (identification) by itself. With one action per state the model still cannot tell whether the bottle
  moved *because of* the hand or simply *with* the hand (the demo always does both). The gate will learn
  "bottle changes when the hand is close and closing", which is the right structure and generalizes better, but
  whether the gate means causation can only be tested with matched alternative actions (the small action-diverse
  dataset). The survey reaches the same conclusion (CAI, GRADER, Poking: all rely on varied interventions).
- Robot-agnostic: robot tokens are visual (hand/arm masks), so nothing here uses FK or calibration. The joint/FK hand
  point stays an evaluation check only.

## 5. Failure modes to design against (from the survey, ranked by risk for us)
1. Bad slots: ID swaps between bottles, missing bottles, hand masks with forearms (already seen). Fix targets first.
2. Global leakage: a strong global-patch path lets the action bypass the gates; restrict action input to robot tokens
   and measure action sensitivity of non-robot tokens (Jacobian / swap test).
3. Gate collapse: all gates closed beats average loss; use weak sparsity, balanced event weighting, and reject a model
   that fixes the bin by freezing the moving bottle.
4. Contact entanglement: at grasp, hand and bottle masks overlap; keep both tokens, allow overlap, use history.
5. Inverse-action heads decoding arm appearance instead of object response (E25's C4 is at risk of this).

## 6. Proposed next steps
**E26a: target audit and motion-stratified baselines (CPU, 1-2 days).**
- Bottle tokens: D1 says 62% of active bottles are missed at t+0. Audit SAM3 bottle tracks (ID swaps, empty masks),
  build cleaned persistent bottle IDs (as done for hands).
- Bin token: D2 reports bin real change exactly 0.0000. Check whether the bin token is computed from a fixed mask or
  frame (bug) or is genuinely constant.
- Copy, constant-velocity and action-free baselines per token group, for moving vs static, approach/grasp/carry/release.

**E26: gated factored dynamics** (after E26a and E25 results).
- Tokens: robot (left/right hand + arm regions), objects (cleaned SAM3 slots), scene (global patches).
- Per token i: z_i(t+1) = z_i + d_free_i(history) + sum over robot tokens r of g_ri * m_ri(robot r, object i, action).
- Action enters only the robot tokens. One shared message MLP m and one shared gate MLP g (inputs: both tokens,
  relative image position, recent co-motion); bounded gate, weak L1 sparsity on g.
- Losses: copy-relative token loss; event-balanced weights from offline mask/track motion; keep E24's rollout loss.
- Evaluation: bin and far-object action sensitivity (should be ~0), moving-bottle beats copy, gate vs observed
  co-motion, the E23a value gate (unchanged), policy RMSE.
- Ablations: no gate (plain residual), gate without sparsity, action injected globally (current behaviour).

**Data (unchanged recommendation).** A small matched-start action-diverse set is still the only way to test whether
the learned gates mean influence. Plan it now; the factored model is the right consumer of such data.
