# Task: Root cause of weak dynamics learning + better alternatives (profile: astra-review = gpt-6-astra, medium; user requested astra explicitly)

## Goal
Explain, with evidence from our own experiments, why the action-conditioned dynamics does not learn to respond to the action on our data (even with a properly configured frozen V-JEPA 2.1 encoder), and propose better alternatives under our constraints. Output a decision-ready report, not code.

## User's position (take as given, test it)
- The dynamics module is hard to train on the limited data.
- The previously loaded V-JEPA weights were properly configured, yet that version still performed poorly. So the encoder/weights are probably not the main cause. Check this claim against the V-JEPA logs before accepting it; say if the evidence does not support it.

## Done criteria
- [ ] Report at runs/report_20261103/ASTRA_ROOT_CAUSE.md (<=200 lines).
- [ ] Section A, evidence table: for each candidate cause, what our experiments show, strength (single seed? n?), and what would falsify it. Candidates at minimum: (1) data: 93 demos / 27k frames, one task, demonstrations with low action diversity so action is confounded with state/phase (the action is predictable from the image; counterfactual actions are out of distribution); (2) objective: latent prediction cosine vs raw L2, norm inflation, targets from the online encoder, copy baseline hard to beat at 15 fps with short horizons; (3) architecture: action enters only through global injection or via weak gates; deterministic dynamics; (4) encoder: frozen V-JEPA 12x12 tokens vs DINO finetuned, small bottle / transparent object, resolution; (5) evaluation: value gate protocol (E28 native gate unmeasured), metrics mismatch; (6) loss balance (E27a result); (7) hand geometry/FK (E28) and gating.
- [ ] Section B: your ranked root-cause verdict with confidence, and which single experiment would separate the top two causes cheaply.
- [ ] Section C: 3 to 5 better alternatives, each with: idea, why it addresses the root cause, what is the minimal experiment on our data, GPU-hours, pass criterion, risk. Consider (not limited to): training the dynamics on a state-only or low-dimensional factored representation (FK hand tokens + object state/slots) instead of dense latent prediction; direct inverse/forward consistency on low-dim states; using the policy as proposer plus a short-horizon critic; data augmentation or small action-diverse robot data (random-perturbation or policy-seeded branching) to break action-state confounding; counterfactual-contrastive or action-ranking objectives; mixed time scale / phase-level dynamics; relying on structure (FK + rigid contact priors) instead of learned physics; multi-step rollouts with consistency; model-free alternative (is a world model even necessary for the final robot success?). Be explicit where the cheapest path may be to drop the learned dynamics from the policy path.
- [ ] Section D: what NOT to try again (already refuted by our logs).
- [ ] Add one line to docs/experiment_log/EXPERIMENT_LOG.md (atomic, re-read before writing) and notes/worklog.md.

## Context
- Host files: ssh -o BatchMode=yes dengjianfeng-101; project /mnt/workspace/dengjianfeng/code/lewm_tianji_cpu. GPU host: ssh -o BatchMode=yes <gpu-server> allowed for read-only checks.
- Read: docs/experiment_handover_20261030/ (V-JEPA history, closed loop never placed the bottle, action-conditioned only ~0.04% better than action-free, ~0.28% better than persistence), docs/experiment_log/EXPERIMENT_LOG.md, docs/MODEL_DIRECTION_20261102.md, docs/INTERACTION_FACTORIZATION_ANALYSIS_20261102.md, docs/lit_jepa_20261102.md, docs/lit_dir3_pretrained_wm_20261103.md, docs/lit_dir4_latent_action_20261103.md, runs/e25_capacity_audit_20261102.md, runs/e26a_audit_20261102/summary.md, runs/e27a_chain_20261103/summary.md, runs/e27a_copycheck_20261103/report.md, runs/e28_chain_20261103/{ANALYSIS_INPUT.md,VERIFICATION.md}. Another session is writing runs/report_20261103/ANALYSIS_REPORT.md: do not edit it.
- Scope: only the current bottle-to-bin task on this Tianji dual-arm robot (not a general base model); build our own model, no other groups' encoders or weights; FK hand geometry from joints is allowed as input; never use wrist cameras. Joint order A7+B7+L20+R20 radians.

## Constraints
Analysis only: do not launch training or evaluation jobs, do not touch run outputs, no installs into system environments. Evidence not claims: cite file paths and numbers; label speculation as speculation. Times in Brasilia (BRT = host CST - 11h).

## Report back
Summary <=15 lines; report path; top-ranked root cause and the one separating experiment; the top alternative; list of numbers you could not verify.
