# E96a / E95 P1 current spatial bypass

Only run `e96a` entrypoints in this copied source package. The copied e86/e88/e90/e94a entrypoints are read-only references and retain original destination constants; never execute their writers.

`e96a.model.WorldModel` subclasses the unchanged E94a unified readout. Its only architectural change replaces `current` with `SpatialCurrent`: the original AttentionPool parameters/modules retain their names and values, and the output is `pooled_j + bypass(raw_j)`. `bypass` is one shared bias-free Linear(768,256), initialized to zero without advancing the CPU RNG. It applies independently to each camera and each of its eight existing row-major cells. The historical pool and W heads are unchanged. No new position embedding is added.

The existing adapter loss automatically targets `current(raw).detach()`, so its target now includes the bypass. Its pooling architecture, loss, gates and calibration schedule are unchanged. Effective gated chain updates can consequently differ and must be reported.

The M1 recipe remains seed90, 5000 steps, batch32, AdamW 3e-4/1e-4, clip1, training-only original SigLIP2 residual normalization and original gated chain schedule. The new code starts from the same old parameter values and post-construction RNG. All fixed query layouts and final sampler state must match M1.

- `pipeline_e96a.py`: waits for the successful full CPU contract suite, then launches exactly one new M1 run.
- `evaluation_waiter_e96a.py`: waits for the fixed5000 completion, then invokes `eval_e96a.py`.
- `eval_e96a.py`: original E93 windows, single-h main layout; joint four-h supplemental layout; E96a/M1/M2 means and frozen ridge coefficients. Reports original normalized metrics and both fine/global raw metrics, per-camera and change-quartile decompositions, paired parent bootstrap 2000/seed90 and preregistered verdict. No fitting or checkpoint selection.
- `finalize_e96a.py`: CPU-only training/source/resource audit, small train-only spatial probe and append-only report/log receipt.

Results: `T/results/e96a_m1_p1_20261114_01` and `T/results/e96a_control_20261114_01`. Report: `T/notes/e96a_spatial_bypass_20261114.md`. No large data or checkpoint copies. Independent resource ledger caps total GPU walltime at3600s, including failures; estimated admission peak8192MiB, requires idle card with at least1.5x peak free and monitors shared free memory/actual peak. New storage <=5GB.
