# Verified paired CE vs Poly-1, 30 true full epochs — seed 20261028

**Date:** 11 November 2026. **Status:** Both runs completed and their results independently read from Google Drive. **Study:** separate from the archived fixed-216-optimizer-update experiment; legacy results, models, and historical baselines remain unchanged.

## Frozen protocol and verification

- Official KolektorSDD2 `train/` pool, frozen original nested 420-image subset, exactly **210 normal + 210 defective** images.
- Identical `split.json` bytes for CE and Poly-1 (independently fetched from Drive; SHA256 `1be0f86837238f81495121ae485b31d65ddf2dce512f20c540734057e516bcb0`). Validation **150 normal + 30 defective** images, from official `train/`, completely distinct by image SHA256 from both training samples and the official test hash manifest.
- Seed `20261028`; ResNet18 ImageNet1K_V1 pretrained; full fine-tuning; RGB 448 BICUBIC; AdamW lr=0.0001 wd=0.0001; batch 4; identical sample order for both losses.
- **30 actual full-dataset epochs**, each **420 images and 105 optimizer steps**, total **3,150 steps per run**.
- Device Tesla T4; PyTorch `2.14.1+cu130`, torchvision `0.29.1+cu130`, NumPy `2.4.6`, Pillow `11.3.0`.
- Best checkpoint chosen from epoch with highest **train-only-validation Macro-F1** (earliest epoch on equal scores). This is *internal validation*, not independent test.
- Both `COMPLETE.json` files present with history length=30 and `official_test_evaluated=false`. Both remote `best_model.pt` files were independently downloaded and checked against their recorded SHA256. Neither training/evaluation touched the legacy `results/` records.

## Results of paired seed (not cross-seed or cross-size averages)

| Metric | Cross-Entropy | Poly-1 |
| --- | ---: | ---: |
| Complete epochs | 30 | 30 |
| Optimizer steps | 3150 | 3150 |
| Selected best epoch | **30** | **22** |
| Best validation Macro-F1 | **0.9800** | **0.9800** |
| Best validation accuracy | **178/180 = 98.89%** | **178/180 = 98.89%** |
| Best validation defective recall | **29/30 = 96.67%** | **29/30 = 96.67%** |
| Best validation normal FPR | **1/150 = 0.67%** | **1/150 = 0.67%** |
| Confusion matrix, truth normal/defective | `[[149, 1], [1, 29]]` | `[[149, 1], [1, 29]]` |
| Epoch-30 train objective | 0.002173 CE | 0.004424 Poly-1 |
| Epoch-30 validation objective | 0.062653 CE | 0.082358 Poly-1 |
| Epoch-30 validation Macro-F1 | 0.9800 | 0.9800 |
| `best_model.pt` SHA256 | `7d3c799307ea5fc9031b1b04a0635e17c626ca89af0c7540dac52f3d424e9d50` | `adb4825f5d10ed2972b5fc49356db7e7775963f849c0da3dae2ef61b5dd71232` |

**Note:** CE and Poly-1 have different mathematical objectives, so their raw training/validation objective numbers are not a like-for-like comparison; prefer classification metrics. Identical confusion matrices do *not* prove they misclassified the exact same individual images.

## Scientific conclusion and limitations

- Under the **same full-epoch data budget, validation set, seed and model family**, both losses gave the same best **validation** Macro-F1 and the same aggregate error counts. No validated advantage for Poly-1 is established from this single seed.
- Poly-1 hit the best selected Macro-F1 at epoch 22 (also ties later), CE at 30; a best epoch earlier is not automatically proof of faster convergence or generally better performance.
- More epochs appear to have improved the original underexposed 420-image setup, but **do not compare 98% internal Validation to the older ~85% reused official Test as if they were measurements of the same data**. Different evaluation samples and budgets.
- The official 1,004-image test set has already been inspected in earlier work, so neither the old nor a future evaluation on it would constitute an independent blind test. Physical-part/group independence is not established. Balanced 420-image training subset differs from the natural class imbalance.
- To address sample-size effects and robustness, run remaining 30-full-epoch balanced pairs at **sizes 70, 140 and 280 for seed 20261028**, each against the same frozen 180-image validation split; then replicate across seeds 20261029 and 20261030 for all sizes, including 420.
- The full planned factorial design is **4 sizes × 2 losses × 3 seeds = 24 new full-epoch trainings**. **2/24 done; 22/24 remain**. The older historical 70-image models belong to the separate fixed-update protocol and do not count as 30-full-epoch runs.

## Permanent Drive evidence

- [CE epoch-30 run](https://drive.google.com/drive/folders/11FJbuU69Y8U6dhhLfNJ5kPUl-wU7GK7Z)
- [Poly-1 epoch-22 selected run](https://drive.google.com/drive/folders/1Aee26df0xD2IBgtbLigMvcDSK3zOjZDq)
- [Pilot collection](https://drive.google.com/drive/folders/11P4PVeDzSasFkAQLEIjJygp_gbYy2EWf)
