### E25 conclusion (Claude, 2026-11-03)
FAIL on the main gate. Value test (set b, H=2): demo vs other 56% (V_pool) / 48% (V_box); vs hold 65% / 66% (gate 70% / 80%). Same level as E24.
Policy: new best offline. Chunk RMSE 0.0689 (E24 0.0716, E 0.0735); medoid-6 0.0674; first-5 0.0421.
Hands fixed by clean labels: current-frame error 5-6 px vs FK (E24 ~140 px); moving windows beat copy 55-70%. Gain is mostly from clean hand labels (Amendment 2), not separable from C1-C4.
Bottle not fixed: ~30 px vs copy 11-23 px (bottle labels were still raw SAM3).
Stage0 D3: history/velocity hiding changes E24 rollout error by only 1-3%, so the motion-shortcut hypothesis was wrong.
Bin slot was empty in all data (E21 found no bin), so bin rows are n/a; the earlier claim that the action moves the bin was an artifact.
Note: posteval_full/value/e24_value_summary.md is the same E25 checkpoint (mislabelled header), not E24.
Capacity audit (runs/e25_capacity_audit_20261102.md): future-RGB losses 55% of weighted loss, roll_latent 4%.

### E26a conclusion (Claude, 2026-11-03)
Bin masks recovered for all 93 episodes (100% coverage). Bottle IDs cleaned (persistent IDs, visibility); bottles inside the bin and dark held bottles are under-detected (known limitation; SAM3 repair attempts added 0 frames, stopped).
Baselines: E24 latent prediction with the true action is worse than copy for every group (global 1.28x copy at H1 to 2.45x at H4; target bottle 2.8-5.4x); an action-free MLP is about copy (global 0.98). Bin/other-bottle ratios are inflated by E24 never having seen clean tokens; the global ratio is not. Conclusion: latent dynamics currently degrades the state; motivates E27a (loss rebalance, latent residual).
