# E102-S causal temporal predicate scoring

New run: `results/e102s_20261115_01`; report: `notes/e102s_temporal_predicates_20261115.md`.
All source data, old code/results and labels are read-only. No world-model checkpoint is needed: the requested E99-B3 method b uses the frozen local SigLIP2 encoder directly. E100-C C2 is not used to produce these predicate scores.

Pipeline (existing e29 Python, no installs/downloads):

1. `extract.py`: train-only two-state gripper threshold, causal proprio/FK, 3-view cached-grid parity, full pooled features.
2. `driver.py`: wait for features, then CPU-only `train.py` and `tune.py`.
3. `evaluate.py`: requires both locks; one-shot dev gate, frozen scores and sequences.
4. `summarize.py`: reads only frozen predictions; parent bootstrap and protected-input audit. Can recover statistics without rerunning dev inference.

Launch long phases with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=T/code_e102s:T/code_e86 setsid nohup ...` and separate retained logs. Do not relaunch evaluate after `DEV_ONCE_STARTED.json` exists. `status.json` and phase locks record progress; all times are BRT.

H0 is exact E99-B3 b, including C=.01 and inherited binary thresholds. H1 uses five causal feature snapshots and current-minus-lag differences. H2/H3 use a 64-channel two-layer causal convolutional trunk, 21 binary predicate outputs and 21 independent unknown outputs. No stage/cycle/attempt/label is a feature. New decision thresholds are fixed .5. Missing/timestamp-gap history is zero-filled with validity; gripper inputs are binary only. Sequence histories 30/60 frames and fixed stride3 are evaluated on training OOF scores; 10 training epochs are fixed beforehand.

S2 code is copied by function from read-only E101-S. Its fixed-S01 sequence is checked for exact agreement with the prior implementation on training scores. Uniform initial prior is a separate causal variant, with the first output chosen from the frame0 posterior; it does not use a first-second lookahead. Temperature/minimum dwell and preferred initial variant are selected only on full 60-parent OOF trajectories. Stage calibration/emission training follows E101's stacked crossfit protocol; it is not fully nested cross-validation.

Dev labels are model-generated, non-human references, explicitly authorized for this experiment. No physical deployment claim is made. S06/S07 definitions are unchanged. E101 reference sequences are read-only, enabling paired parent CIs rather than subtraction from a noise-free 0.366 constant.
