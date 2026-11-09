# Task 5E — Peer Reproducibility Audit and Model Card
**9 November 2026 — original classmate `predict.py` successfully executed unchanged on our private 270-image Test.**

## Original peer README-based reproduction
- Input package: classmate-supplied `yolov8n_quality_gate.zip`, containing README, binary YOLOv8n-cls model, and inference script. The original file bytes were preserved; Windows ZIP separators were normalized to extract on Linux.
- In an Ultralytics 8.4.208 environment, we executed the **original script and README invocation**: `python predict.py --source <our private TEST> --out <private predictions.csv>`.
- Result: **exit code 0**, 270 genuine per-image predictions generated; no code patch, no retraining, fixed decision threshold 0.50.

| Original classmate script on our Dataset A Test | Result |
|---|---:|
| Images | 270 (135 good / 135 bad) |
| Accuracy | **211/270 = 78.15%** |
| Precision bad | 0.6959 |
| Recall bad | 1.0000 |
| F1 bad | 0.8207 |
| Confusion matrix, rows true [good,bad], columns predicted [good,bad] | [[76,59],[0,135]] |

- All **270 image-level class decisions** match the previous independent PyTorch compatibility evaluation (zero discrepancies). The original peer script rounds probabilities to 4 decimals; max deviation from previous unrounded adapter output was approximately **0.000051**.
- Error pattern on our independent Test: 59 clean-image false rejects, no defective-image false accepts.
- [Original script logs, raw predictions, accuracy report and SHA-256 integrity metadata — private Drive](https://drive.google.com/file/d/1BmdQlk-aY5e97KGm7phdXk8P2pN6Aw2i/view).

## Gaps and limits (do not invent evidence)
1. We reproduced the peer's **inference program on our independent Test**, **not** the peer's reported in-distribution 0.996 original-Test accuracy or their original training. Their own original labeled Test, hashes/source-group manifest, complete training code and reproducible environment lock are absent.
2. Version drift: original package does not pin Ultralytics, Python and Torch precisely. We recorded the actual working Ultralytics version.
3. Permission to publish peer weights publicly is not independently established. Therefore we saved only inference evidence, NOT the peer checkpoint or original private images.
4. Domain shift between synthetic document templates can explain lower cross-test performance, but no definitive cause is demonstrated, and the result does not disprove the peer's reported own-domain score.

## Own model card and documentation
- [Our updated model card](model_card.md): intended use, train/validation/test, 267/270 original accuracy, false accepts, latency/size, limitations, privacy/ethics and generalization concerns.
- Our own shared model's README command separately passed a clean Ubuntu GitHub Actions execution on four synthetic test inputs, documented [here](https://github.com/ID9999999999999/DL-Computer-vision/actions/runs/42516928052). That is a packaging smoke check, not a scientific accuracy experiment.
- Task 5D now independently confirmed ONNX vs PyTorch **267/270 both** on our private Test, with 0 classification disagreements.

## Completion status
- **README-driven peer inference reproducibility: completed.**
- **Full original peer training/own-test reproduction: remains unavailable without the peer's original dataset and experiment files.**
- E01 explicit third-party sharing permission and E08 instructor's formal acceptance/defense are not verified.
- [Task5E requirement audit](../docs/TASK5E_2026-11-09_AUDIT.md).

Do not claim teacher acceptance or pretend that a new adaptation/peer-side reverse test is complete.

## Reflection on our README

The documentation review found a practical barrier: the Windows launcher used the tracked `results/` directory by default, which could cause a dataset-path conflict instead of starting a fresh run. The launcher now creates a separate output directory per run. The project README also links the original dataset, lists its required folder structure, separates saved-result verification from retraining, and distinguishes the recommended local Python version from the actual Colab environment.

These changes make the reproduction path easier to follow without altering the historical checkpoints or metrics. A full independent re-training and exact result match have **not** been verified by this documentation edit.
