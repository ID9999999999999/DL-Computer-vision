# Lab 06 — Bonus Task 5C: Error-Driven Glare Augmentation

**Executed:** 2026-11-09. **Status:** genuine CPU training of both controlled arms.

## Decision and honest protocol scope
The original YOLOv8n-cls was the validation/cost-selected model. Across the four original model configurations, all observed test mistakes involved glare. Two new training arms were compared, **control** (no extra overlays) and **glare** (train-only stronger white elliptical overlays), **starting from the exact SAME selected YOLOv8n saved weights** (SHA-256 `c820e22debf8d3fe3c3ca41f7c10f55d9e7db0b2246fb90c92a8548a70a514f7`), the same seed **2473986**, same data split, batch **16**, input **224×224**, Adam LR **1e-4**, max **15 epochs**, early-stopping patience **4**. The original model was restored and its original **267/270** test count was verified.

This is a **controlled continuation from an already trained checkpoint**, not a full replay from ImageNet: package availability prevented importing Ultralytics in this offline runtime. It uses a local PyTorch checkpoint-compatible forward graph. Both arms share this limitation and use the same implementation, allowing comparison of overlay effects. Original teacher's later 50/32/5/640 requested settings were **not** satisfied by this new experiment; the actual settings are reported. Checkpoint selection used the validation cross-entropy, not Test.

## Augmentation
Only on the 1,260 training images; no augmented image was saved to or mixed into validation or held-out test. Random white ellipses alpha **150–230**, one to two overlays per selected image. Applied only to BAD training images with probability **0.75 if original defect was glare**, **0.35 for other bad defects**, and **0** for good images. Each variant is seeded deterministically by `2473986 + epoch*1,000,003 + array_index*131`. This is label-preserving for defective images; it intentionally creates stronger negative glare cases. Details: `results.json` and source script.

## Genuine measurements (held-out Test, n=270)

| Model | Test accuracy | Macro F1 | F1 good | F1 bad | Glare recall | Glare false accepts | Epoch selected | Epochs run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original archived | 0.9889 | 0.9889 | 0.9890 | 0.9888 | 0.8929 | 3 | historic | historic |
| No-overlay control | 0.9926 | 0.9926 | 0.9926 | 0.9925 | 0.9286 | 2 | 1 | 5 |
| Train-only glare | 0.9889 | 0.9889 | 0.9890 | 0.9888 | 0.9286 | 2 | 1 | 5 |

The per-defect table (`per_defect_delta.csv`) uses **each defect subtype as the positive examples versus the SAME 135 good negatives**; it is an analytical binary F1, **not a five-class confusion matrix**. `binary_class_delta.csv` contains the genuine binary good/bad per-class F1.

## Five-line conclusion
1. The test glare recall changed from **0.9286** (unaugmented continuation) to **0.9286** (glare augmentation): **no change**.
2. Overall macro F1 changed by **-0.0037** between otherwise matched new training arms.
3. Defect types with changed recall/F1 across those arms: **occlusion**; see exact values in CSV.
4. Good-class F1 changed by **-0.0036**, because one more defective occlusion image was incorrectly accepted; neither new arm falsely rejected a good image.
5. Because the test contains only **45 independent source documents**, this single seeded result does not prove a general improvement; the augmentation should be selected by validation and checked on fresh external images before deployment.

## Provenance
Original dataset ZIP, manifest and all 1,800 image SHA-256 digests validated; 210/45/45 disjoint sources. Raw original/validation/test datasets and original checkpoints were not edited. Two new tensor-only checkpoint state_dicts were saved. The teacher's written task requires a fair baseline; the report explicitly distinguishes checkpoint continuation from the original ImageNet-start training.
