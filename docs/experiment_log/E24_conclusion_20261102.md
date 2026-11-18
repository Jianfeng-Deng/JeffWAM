### E24 conclusion (Claude, 2026-11-02)

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
