# Nazih coursework — training-set size, 8 November 2026

**Author:** YASSER IDBOUZKRI, иикб-24.

## Current status: prepared and integrity-tested; no new training run has yet occurred

This package preserves the **six authentic 12-epoch evaluations** from the preceding duration experiment and **70 original KSDD2 training images**. The 70-image baseline **must not be retrained or replaced**. The new, predeclared experiment trains **18 NEW models**: 3 new sizes (140, 280, 420) × 3 seeds (20261028, 20261029, 20261030) × 2 losses (CE, Poly-1). All conditions use ResNet18 IMAGENET1K_V1, 448×448 bicubic RGB, Imagenet normalization, AdamW (LR=1e-4; weight decay=1e-4), 12 reference-equivalent rounds (not 12 full-data epochs for larger pools), batch 4, 216 optimizer steps and the same batch-label sequence per seed. Only source-image diversity changes. Raw official training/test file hashes are verified.

**Why 140/280/420?** To test whether increasing original training image diversity beyond the existing 70 improves exploratory quality. The official train split contains 246 defective canonical images, so 420 (210 per class) fits without leakage. Within each larger size we preselect new images **from the official train split only**, using seed 20261108. Selected cohorts are nested and balanced, preserving the original 70. Each epoch consumes 35 Normal + 35 Defective examples. All selected source images are visited at least once over 12 rounds. Effective passes per original image are 12 (70), 6 (140), 3 (280), or 2 (420); this is a fixed-compute learning curve, not a fixed-number-of-full-data-epochs comparison. The total number of parameter updates is held constant, avoiding confounding data amount with training duration.

**Numerical caution:** The earlier duration diagnostic could not reproduce 3-epoch checkpoint states exactly despite matching initialization/orders. Pinning software versions, seeds and order hashes here minimizes but does not prove elimination of runtime numerical differences. Since the 70-image reference is intentionally not rerun, sample-size comparisons remain exploratory rather than an exact one-factor causal experiment.

**BN control:** Reset and recalibrate running BatchNorm statistics using the same exact original 70 images, in the same order, for every new size (not all N). This avoids a second intervention (calibration population). The resulting effect is specific to this fixed compute budget and BN protocol; results should not be generalized to unconstrained training.

## Important prerequisites and blockers

- Original official `KolektorSDD2.zip` (not included; large). Official source: https://data.vicos.si/datasets/KSDD/KolektorSDD2.zip. The original SHA256 MUST equal `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`.
- CPU environment matching the archived runs: Python 3.12.14, PyTorch `2.15.0+cpu`, torchvision `0.29.2+cpu`, NumPy `2.5.4`, Pillow `13.0.0`. Script refuses to compare new runs when package versions or reference initialization/order hashes differ. CUDA is excluded because the 70-image reference was run on CPU. Installation may require an environment offering these exact versions.
- The pretrained ResNet18 ImageNet V1 checkpoint (SHA256 verified by `scripts/train.py`); torchvision downloads it once if network access is present.
- Computational budget: 18 separate 12-epoch trainings with 448×448 images on CPU can be demanding. The runner supports running one seed, size and loss at a time. Store checkpoints persistently between sessions.

## Procedure (from the study's root directory)

```bash
python scripts/preflight.py
python scripts/test_pipeline.py
python scripts/prepare.py --archive /path/to/KolektorSDD2.zip
python scripts/train.py --archive /path/to/KolektorSDD2.zip --size 140 --seed 20261028 --loss ce
```

Or use `python scripts/runner.py --archive /path/to/KolektorSDD2.zip --stage train` for resume-safe sequential training.

Repeat the *last command* for size `140`, `280`, `420`, seed `20261028`, `20261029`, `20261030`, and loss `ce`, `poly1` — no size 70. Start by checking `scripts/train.py`'s strict environment before spending time on training. Save original outputs without overwriting. **Only after all 18 checkpoint JSON + .pt files exist**, use:

```bash
python scripts/evaluate.py --archive /path/to/KolektorSDD2.zip --size 140 --seed 20261028 --loss ce
python scripts/report.py
```

Repeat evaluation for all 18 new runs, or use `python scripts/runner.py --archive /path/to/KolektorSDD2.zip --stage evaluate`. `scripts/evaluate.py` refuses to inspect official test image data before all 18 training checkpoints are present and SHA256-verified. It outputs real metrics only; `scripts/report.py` leaves pending results blank. See `RUN_Colab.ipynb` for the workflow, noting Colab's CPU and package versions must pass the strict guard.

## Verified now

- SHA256 for all 70 original training source images and zero exact-hash overlap with test manifest.
- SHA256 for all **12 archived checkpoints** (six epoch-3 and six epoch-12) matched their own original evaluation metadata, verified from `Duration_Training_Checkpoint.zip`. See `results/historical_weight_audit.json`. The 500 MB checkpoint ZIP is intentionally NOT duplicated here.
- Labels and 1,004 per-image prediction records in all six previously completed 12-epoch baseline evaluations.
- Recomputed Macro-F1 from each 2×2 confusion matrix exactly matches the stored value.
- Poly-1 gradient tests and schedules for all three new sizes × three seeds.
- No new training or test inference has been reported as finished; the official full ZIP and original CPU environment are not present in the working execution environment.

## Interpretation and research integrity

The 1,004 test images were **already inspected in earlier diagnostics**. Reusing them is permitted for an explicitly labeled exploratory learning-curve investigation, **not** an independent blind conclusion. The original 70-image trained models were also previously studied; this is an exploratory fixed-design comparison. The official KSDD2 train/test split and exact byte hashes do not prove part/specimen identity independence, which remains to be investigated. Do not select thresholds, hyperparameters, sample sizes or models based on test outcomes and then describe the same images as independent evidence.

`input/` = genuine archived data and historical results; `scripts/` = new experiment code; `results/` = verified preflight, new checkpoints and new evaluations once actually executed. The historical baseline is read-only and is never overwritten.