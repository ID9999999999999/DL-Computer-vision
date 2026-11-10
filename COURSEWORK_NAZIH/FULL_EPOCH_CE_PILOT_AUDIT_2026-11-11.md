# Verified 30-full-epoch CE pilot — 11 November 2026

## Provenance

- Study: **a separate, new full-epoch study**, NOT a continuation or replacement of the original 216-update study.
- Frozen training population: 420 images from KolektorSDD2 official **train/** pool, 210 normal + 210 defective.
- Internal fixed validation: 180 different official **train/** images, 150 normal + 30 defective, random seed `20261110`.
- Official test set: **not evaluated or accessed as images** in the pilot. Pre-existing test hash manifest was used solely as an exclusion guard.
- Architecture: pretrained ResNet18 ImageNet1K_V1, RGB 448x448, AdamW lr 0.0001 wd 0.0001, batch 4, Cross-Entropy loss.
- Training seed `20261028`, Colab Tesla T4, torch `2.14.1+cu130`, torchvision `0.29.1+cu130`, NumPy `2.4.6`, Pillow `11.3.0`.
- Actual training: **30 complete passes** over all 420 images (105 optimizer updates each, **3,150 total**).
- Persistent source: [Google Drive pilot results](https://drive.google.com/drive/folders/11FJbuU69Y8U6dhhLfNJ5kPUl-wU7GK7Z).
- Status confirmed by directly reading Google Drive's `COMPLETE.json`, `progress.json`, `split.json` and separately downloading and SHA256-verifying the best checkpoint on 11 November 2026.

## Audit and results

| Check | Verified value |
| --- | --- |
| Epochs in full recorded history | 30 |
| Optimizer steps | 30 × 105 = 3,150 |
| Original train rows / validation rows | 420 / 180 |
| Within each set, unique image content hashes | 420 / 180 |
| Train/validation SHA256 image overlap | **0** |
| Validation class distribution | 150 normal / 30 defective |
| Model-selection rule | Highest validation Macro-F1, first epoch on ties |
| Best selected epoch | **30** |
| Best validation Macro-F1 | **0.9800 (98%)** |
| Best validation accuracy | **178/180 = 98.89%** |
| Best epoch defective recall | **29/30 = 96.67%** |
| Best epoch normal false-positive rate | **1/150 = 0.67%** |
| Best epoch confusion matrix, truth rows normal/defective | `[[149, 1], [1, 29]]` |
| Epoch 30 train CE objective | 0.00217 |
| Epoch 30 validation CE objective | 0.06265 |
| Epoch 20 validation Macro-F1 | 0.979447 |
| Epoch 30 validation Macro-F1 | 0.980000 |
| Sum of measured per-epoch durations | ~570.54 s |
| Source official ZIP SHA256 | `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8` |
| Best checkpoint SHA256 | `7d3c799307ea5fc9031b1b04a0635e17c626ca89af0c7540dac52f3d424e9d50` |
| `split.json` SHA256 | `1be0f86837238f81495121ae485b31d65ddf2dce512f20c540734057e516bcb0` |
| Original 24 evaluation results | **Preserved, unchanged** |

**Independent SHA256 audit:** The `best_model.pt` bytes fetched from Drive produced exactly `7d3c799307ea5fc9031b1b04a0635e17c626ca89af0c7540dac52f3d424e9d50` in a separate analysis container. The `split.json` bytes similarly matched the SHA256 recorded by `COMPLETE.json`.

## Interpretation

The professor's objection to low effective epoch count has been addressed **for the first 420-image Cross-Entropy pilot** by running 30 genuine full-data epochs. Training objective fell from ~0.54480 (epoch 1) to ~0.00217 (epoch 30), while validation objective ended at ~0.06265; these trajectories justify further assessment but do not by themselves establish generalization to new industrial parts.

The validation Macro-F1 was already **0.979447 at epoch 20**, compared with **0.980000 at epoch 30**, meaning the final selected epoch offers only ~0.055 percentage points of improvement versus an earlier strong epoch. The fact that epoch 30 was selected does **not**, on its own, mean 30 epochs are insufficient or that we should extend training indefinitely. Selection based on a 180-image validation sample can be noisy.

**Strong caveat:** The validation images were withheld from official TRAIN, not a truly independent production test. Physical-part/group independence remains unproven. The natural KSDD2 official TRAIN distribution is imbalanced (2085 normal, 246 defective) while the selected 420 training images were class-balanced; the internal validation is 150/30. The previous official 1,004-image test was already inspected in the historical work. Therefore **do not report 98% as test accuracy or production performance**.

## Next scientific action

Run **Poly-1 for 420 images, seed 20261028, 30 true full epochs**, paired with the same preprocessing, train/validation rows, initialization seed and update count. Then compare both training histories, validation metrics and stability before expanding to a multi-seed, multi-size controlled full-epoch study. The evaluation/test-set policy must remain unchanged.

The new full-epoch study belongs to a different protocol from the original fixed-216-optimizer-step experiment; do not merge the two result tables without an explicit protocol distinction.
