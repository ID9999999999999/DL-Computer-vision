# Verified classification failure examples — exploratory evaluation

**Scope:** Study A, archived official TEST (1,004 images), seed `20261028`; original test set had already been inspected. These are **not** full-epoch Study B validation predictions.

All examples below are read directly from the authentic archived prediction arrays. No model retraining, rescaling, threshold adjustment or post-selection occurred.

## Error totals across paired runs

| Train images | CE FP | CE FN | Poly-1 FP | Poly-1 FN | Both wrong | CE only | Poly-1 only |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 140 | 83 | 37 | 97 | 28 | 96 | 24 | 29 |
| 280 | 31 | 31 | 21 | 33 | 46 | 16 | 8 |
| 420 | 23 | 26 | 17 | 28 | 41 | 8 | 4 |

## Selected original image-level failure examples (size=420)

| Image in official ZIP | Ground truth | CE decision | CE margin | Poly-1 decision | Poly-1 margin | Case |
|---|---|---|---:|---|---:|---|
| `test/20043.png` | Normal | Defective | +0.032 | Defective | +1.119 | Both wrong — False Positive |
| `test/20652.png` | Normal | Defective | +0.280 | Defective | +0.770 | Both wrong — False Positive |
| `test/20651.png` | Normal | Defective | +5.579 | Defective | +4.260 | Both wrong — False Positive |
| `test/20523.png` | Defective | Normal | -0.024 | Normal | -0.774 | Both wrong — False Negative |
| `test/20259.png` | Defective | Normal | -0.359 | Normal | -0.118 | Both wrong — False Negative |
| `test/20236.png` | Defective | Normal | -4.323 | Normal | -5.581 | Both wrong — False Negative |
| `test/20759.png` | Normal | Defective | +0.017 | Normal | -0.023 | CE wrong / Poly-1 correct — False Positive |
| `test/20793.png` | Normal | Defective | +0.151 | Normal | -0.426 | CE wrong / Poly-1 correct — False Positive |
| `test/20633.png` | Normal | Defective | +1.969 | Normal | -0.072 | CE wrong / Poly-1 correct — False Positive |
| `test/20719.png` | Normal | Normal | -0.018 | Defective | +0.222 | CE correct / Poly-1 wrong — False Positive |
| `test/20224.png` | Normal | Normal | -0.422 | Defective | +1.149 | CE correct / Poly-1 wrong — False Positive |
| `test/20772.png` | Defective | Defective | +1.264 | Normal | -0.042 | CE correct / Poly-1 wrong — False Negative |
| `test/20738.png` | Defective | Defective | +1.355 | Normal | -0.394 | CE correct / Poly-1 wrong — False Negative |

## Interpretation

- **False positive (FP):** a genuinely normal image classified as defective. At 420 images, CE produces 23 and Poly-1 produces 17 such errors.
- **False negative (FN):** a genuinely defective image classified as normal. At 420 images, CE produces 26 and Poly-1 produces 28 such errors. Missed defects matter in industrial inspection.
- **Trade-off:** in this one paired run Poly-1 rejects 6 fewer normal images but misses 2 more defective images. This is an observed trade-off, not evidence of a universal advantage.
- **Defect logit margin** equals defective-class logit minus normal-class logit. Positive predicts defective; negative predicts normal. Its magnitude is not a calibrated probability.
- **Distinct failures:** some images are wrong for only one loss. The models do not simply make the same errors, so aggregate Macro-F1 alone hides useful differences.

## Evidence boundaries

- The archive contains predictions, labels and exact image hashes, **not the raw test image pixels**. No visual texture, scratch type, lighting cause, or defect morphology was assessed here.
- To create a photographic gallery, obtain the official `KolektorSDD2.zip`, check its fixed SHA256 and open the individual `test/*.png` files. A separate command in `scripts/error_gallery.py` generates that gallery reproducibly without changing any model.
- These test images were evaluated previously. This is an **exploratory post-hoc error audit**, not a new independent blind-test assessment or a basis for hyperparameter tuning.
- Physical-specimen grouping and near-duplicate independence were not verified by exact image hashes alone.

Dataset: Božič et al., KolektorSDD2, ViCoS (CC BY-NC-SA 4.0).
