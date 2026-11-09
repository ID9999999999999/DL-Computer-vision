# Nazih — verified execution audit (2026-11-08)

## Result

- PASS: all 70 old KSDD2 training images match their original SHA256 manifest.
- PASS: six archived 12-epoch baseline evaluations, 1,004 predictions per model, were rechecked.
- PASS: Poly-1 numerical gradient verification.
- BLOCKED: no original official KolektorSDD2.zip was available, and network download failed.
- BLOCKED: installed torch 2.10.0+cpu, torchvision 0.25.0+cpu, numpy 2.3.5, Pillow 12.3.0 do not match the recorded old CPU versions.
- **TRAINING COUNT: 0 of 18 new models. No fabricated results.**

## Historical 70-image baseline — reproducibility audit

The means below are derived from three previously saved seeds and are *not* outcomes of the planned training-set-size intervention. Prior external test was reused in diagnostics: do NOT describe it as a blind final test.

| Metric | CE mean | Poly-1 mean | Poly-1 − CE |
|---|---:|---:|---:|
| macro_f1 | 0.76640 | 0.75719 | -0.00921 |
| defect_recall | 0.61212 | 0.64545 | +0.03333 |
| normal_fpr | 0.05854 | 0.07420 | +0.01566 |
| auroc | 0.85675 | 0.85617 | -0.00058 |
| average_precision | 0.66238 | 0.66479 | +0.00241 |
| accuracy | 0.90538 | 0.89509 | -0.01029 |

## Next executable action (after prerequisites exist)

Provide the unmodified *original official* KSDD2 archive with SHA256
`edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`
and a CPU runtime with Python 3.12.14 / torch 2.15.0+cpu / torchvision 0.29.2+cpu / numpy 2.5.4 / Pillow 13.0.0.
Then `python scripts/runner.py --archive /path/to/KolektorSDD2.zip --stage prepare`, followed by `--stage train`, then `--stage evaluate` and `--stage report`. The existing baseline 70 is never retrained; scripts fail on mismatches.

The 140/280/420 conditions are fixed-216-update, balanced sampling and are explicitly exploratory. All comparative claims must acknowledge possible software-run determinism drift and unresolved physical-part identity independence.