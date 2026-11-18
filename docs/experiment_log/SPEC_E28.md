# SPEC E28: robot geometry from joints + gated object dynamics (Claude, 2026-11-03)

Scope: only the bottle-to-bin task on this robot (user). FK hand geometry is allowed as model input (user, 2026-11-03).

## Why
- All models fail the action gate (H=2: demo vs other ~48-61%, vs hold ~50-66%; need 70/80).
- Copy-check (runs/e27a_copycheck_20261103/report.md): global latent prediction beats copy on its cosine objective
  (A4 -25%, E25 -16%) but predictions are ~10% too large in norm and lag (+10 output closest to +5 frame).
- Decoded bottle state is worse than copy (~29 px vs 9 px); hands are good with clean labels.
- Root cause (B1): the action's effect is not identified. With FK, the robot part of the action effect becomes known
  exactly; the network only has to learn the object response, and only through the hands.

## Interface contract (all sessions must follow it)
Module `tools/e28_hand_geometry.py`, function
`hand_geometry(q54: np.ndarray|torch.Tensor (..., 54) radians) -> dict` returning, per arm a in (A=left image, B=right):
- `p_cam` (..., 2, P, 3): P hand points in the top-camera frame, metres. P = palm point (E22b-fix offset point) + flange
  origin + fingertip points if available from URDF FK (document P and point order in the module docstring).
- `uv` (..., 2, P, 2): projections in the 640x384 crop pixels (E22b intrinsics), and `uv_grid` normalized to [0,1].
- `rot6d` (..., 2, 6): flange orientation in the camera frame.
- `hand_q` (..., 2, 20): the hand joint values (as given).
Calibration: Oct 15 head_world_robot + E22b-fix fits (runs/e22b_calib_check_20261102_fix/fits.json, offset-only).
Joint order A7+B7+L20+R20; A=image-left arm, B=image-right arm (E22b-fix arm mapping). Must run on GPU in batch
(torch path) for training-time geometry of candidate/counterfactual action chunks.
Cache: `data/hand_geometry_20261103/` with per-frame geometry of measured joints for all 27,281 frames.

## Model (tools/e28_*.py), initialized from runs/e27a_A2_20261103/latest.pt where shapes match
1. Encoder, global grid, object slots (clean cache incl. bin), policy flow head: as E27a A2.
2. Robot tokens: for the current frame and each of the 10 future steps of a candidate action chunk, geometry from
   `hand_geometry(action_q)` (actions are joint targets; VERIFY absolute vs delta and convert) -> per arm a token
   (MLP over p_cam, uv, rot6d, hand_q, step embedding). Measured current q gives the current robot token.
3. Hand slots are no longer predicted from pixels in the dynamics: their future state comes from the robot tokens.
4. Gated object dynamics, per object slot i (bottles, bin) and arm a:
   dz_i = f_free(z_i, context) + sum_a g(a,i) * m(robot_a, z_i, rel_ai)
   g = sigmoid(MLP(robot_a, z_i, rel_ai)) in [0,1]; rel_ai = image-plane offset between hand points and object
   centroid/box, plus hand depth; shared MLPs for all pairs; weak L1 on g (1e-3, tune).
   The action enters ONLY through robot tokens -> gated messages (no global action injection into object or global
   tokens). Global grid: residual z_t + delta with delta driven by robot tokens + free term.
5. Norm control for global latent: loss = cosine + 0.1 * L2 after per-token LayerNorm; also log raw-L2 copy ratio.
6. Losses: policy flow (1.0); decoded object state at +10/+20/+30 as copy + residual (centroid, box, mask, held,
   in-bin) (1.0); global latent (2.0); A2 object-weighted image loss (1.0); gate L1. Drop inverse dynamics.

## Runs (4 epochs each from A2 unless noted; free GPUs, check memory, never touch others' processes)
- E28 (full).
- E28-noGate: same FK robot tokens, but the action/robot tokens are injected globally (as before) - isolates gating.
- E28-noFK: gated dynamics, robot tokens learned from the action vector without FK - isolates FK.

## Evaluation (extend tools/e27a_posteval_20261103.py)
1. Value gate (main): set-b value heads, H=1..4; native 54-D action swap, with robot geometry recomputed from the
   swapped action (hold, reversed, other window, shifted).
2. Static-object invariance under action swap: change of predicted bin / far-bottle state when only the action is
   swapped and the hand stays far (should be ~0); vs E27a A2.
3. Decoded object state vs copy (bottle, bin) at +10/+20/+30, moving vs static windows, by phase.
4. Gate diagnostics: gate value vs observed contact/co-motion events (hand-bottle mask overlap + co-motion),
   AUROC; gate on static far objects.
5. Global latent: cosine, LayerNorm-L2 and raw-L2 copy ratios.
6. Policy chunk RMSE, first-5, medoid-6 vs A2 and E25.
Pass: main gate (70/80 at H=2); bottle moving-window pred beats copy >= 60% at +10; static invariance better than A2.

## Outputs
runs/e28_{full,nogate,nofk}_20261103/, runs/e28_chain_20261103/ (status/, summary.md), log entry E28.
