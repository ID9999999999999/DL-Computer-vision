# LAB 06 — Bonus Task 5C: Glare Augmentation

**Date:** 2026-11-09. **Result:** Augmentation did NOT improve glare recall versus matched unaugmented training.

## Files
- `report/bonus_augmentation.md` — experiment, method, table, five-line conclusion.
- `results/results.json` — all original/control/augmented metrics and exact augmentation configuration.
- `results/comparison_rows.csv`, `results/per_defect_delta.csv`, `results/binary_class_delta.csv` — original and per-class comparisons.
- `results/control/`, `results/glare/` — **genuinely trained** checkpoint state_dicts, prediction tables and histories.
- `scripts/task5c_glare_train.py` and `scripts/ultra_compat.py` — reproducible CPU training without the unavailable Ultralytics package.
- `scripts/verify_task5c.py` — verifies saved checkpoints against predictions and original checkpoint SHA.
- `weights/yolov8n_original_best.pt` — exact original pretrained-on-ImageNet-and-fine-tuned model, before this new experiment.
- `metadata/manifest.json` — exact original source-grouped split and image SHA-256 hashes.
- `plots/` — train-only example, validation loss and defect recall comparison.

## Reproduction
Install: `python -m pip install torch torchvision numpy pillow scikit-learn matplotlib`

Provide the **original** `demo_quality_dataset.zip` (not included: ~399 MB), then from the extracted ZIP root run:

```bash
python scripts/task5c_glare_train.py --zip PATH_TO_ORIGINAL_DATASET_ZIP --manifest metadata/manifest.json --checkpoint weights/yolov8n_original_best.pt --out reproduction
python scripts/verify_task5c.py
python scripts/verify_task5c.py --zip PATH_TO_ORIGINAL_DATASET_ZIP  # full pixel/inference verification
```

The verification script validates the INCLUDED experiment output and is not a substitute for rerunning the training. If reproducing on another machine, CPU and PyTorch versions may affect exact floating-point outputs.

## Limitations
The trial compared two **continued fine-tunings** from the original selected checkpoint, not a fresh ImageNet-start retraining. Both used the same split, seed, input size, optimizer, batch and early-stopping policy; new glare overlays were applied ONLY to training images. The original instructor's later 50/32/5/640 protocol was not used; see report. No teacher approval is claimed.
