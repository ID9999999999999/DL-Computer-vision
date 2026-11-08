# Optional Task 5E — Peer Reproducibility Audit and Model Card
**Updated 9 November 2026. Peer-only README reproduction remains unverified. Own-share README reproduction is executed and evidenced.**

## Source artifacts and exact scope

- Original classmate-provided archive `yolov8n_quality_gate.zip` (2,716,320 bytes) was received in the conversation. It includes `README.md`, `predict.py` and a YOLOv8n-cls `.pt` checkpoint, **not the peer's labeled image dataset**.
- Its README documents good/bad image folders, 224-pixel classification, a fixed 0.5 decision threshold and an Ultralytics installation / `predict.py --source ...` inference workflow. It reports its own holdout performance and training settings (seed 24500003) but no complete independently auditable source manifest and image hashes were delivered.
- Explicit third-party redistribution permission has not been independently verified. The original classmate checkpoint and their personal data were **not** uploaded to our public GitHub repository.

## 1. What was measured with the peer's model

The classmate checkpoint **was evaluated on our Dataset A held-out 270 images using a PyTorch compatibility adapter**, not their original `predict.py` command:

| Metric | Peer checkpoint on our Test | Our original YOLOv8n on same Test |
|---|---:|---:|
| Accuracy | 211/270 = 78.15% | 267/270 = 98.89% |
| Macro-F1 | 0.7705 | 0.9889 |
| Good wrongly rejected (good → bad) | 59 | 0 |
| Bad wrongly accepted (bad → good) | 0 | 3 |

All 59 peer false alarms were on our clean INN (45) and SNILS (14) examples. Our test split contained 135 good and 135 bad images. See [cross-dataset evaluation](cross_dataset_evaluation_2026-11-09.md) and [image-level measurement evidence](https://drive.google.com/file/d/1CsRe-t3hDA8IH1ar4pQs8-zKg16JtKTF/view). This measures domain transfer onto **our** data, not a reproduction of the peer's own published result. The peer's original dataset has still not been supplied. Independently confirming classmate-training/source overlap is not possible from this archive.

## 2. Strict Task 5E peer reproduction — not yet verified

The written bonus asks to repeat **a classmate experiment using only their README**. The previous 270-image evaluation **does not** satisfy that literal criterion because we used a compatibility adapter, not the classmate's `predict.py` in an Ultralytics installation; and the peer's own original test images are absent. We cannot claim their reported original holdout performance has been reproduced. The author's explicit permission for further sharing is also not independently evidenced.

To reproduce the package's *inference workflow*, use its original archive with our private `yolo_data/test/good` and `test/bad` images and execute the README command in Ultralytics. Do not replace or invent the peer's private original holdout images. Save `stdout`, error log, CSV, software versions, model SHA and observed confusion matrix. A reproducible local wrapper is stored in [task5e_peer_readme_run.py](../scripts/task5e_peer_readme_run.py), **not yet run against the peer archive in a working executor**.

## 3. Our own README / model-sharing workflow — actually reproduced

We **did** execute the *README-style command of our own exchanged YOLOv8n package* on a fresh GitHub Actions Linux/Python 3.11 runner using current Ultralytics:

- [Successful GitHub Actions run](https://github.com/ID9999999999999/DL-Computer-vision/actions/runs/42516928052).
- [Downloadable raw smoke evidence: stdout, stderr, four-row CSV, summary JSON](https://drive.google.com/file/d/1EF69RHSmcHTv20Ym_JW7U4EbzmFF1BGS/view).
- Original checkpoint SHA-256 **c820e22debf8d3fe3c3ca41f7c10f55d9e7db0b2246fb90c92a8548a70a514f7** verified.
- Command `python predict.py --source <test> --out <predictions.csv>` returned **0**, and four synthetic input JPEGs produced exactly four image-level output rows with columns `path`, `prob_bad`, `prediction`, `true_label`.
- This is an **execution smoke test only**: these four generated images carry no meaningful generalization accuracy, and this workflow does **not** run the classmate archive.

## 4. Concrete gaps in the exchanged peer package

1. Original peer images, source manifest and file hashes are absent, so independent train/test source separation cannot be audited.
2. Exact pinned Python/Ultralytics/Torch version lock is absent, so inference compatibility is not guaranteed on a fresh machine.
3. Training code and complete logs are not self-contained in the exchanged archive; its README is aimed primarily at using the trained checkpoint, not fully reproducing the original training.
4. The README's own result cannot be compared as a same-distribution reproduction unless its originally documented held-out dataset is provided with authorization.
5. The provided model used a different source domain; accuracy on our own held-out images is a *transfer test*, not proof that its original 99%+ author figure is wrong.

## 5. Our own documentation fixes and model card

Our exchange ZIP now has a short English README, simple CLI, and model weights; its runability was verified in fresh Linux CI. The main project preserves its seed, split details, original hyperparameters and deviations, test metrics, model SHA, reproducibility script, limitations and source documents. See the [completed model card](model_card.md) for intended use, training data, accuracy, per-class results, CPU latency, size, ethics and synthetic-to-real limitations, including the new Task 5C/5D addenda. The original model's private Train/Validation/Test images remain outside GitHub.

## 6. E01–E08 requirements and boundaries

| Item | Evidence status |
|---|---|
| E01 Permission and peer README | **Partial** — original archive and README supplied, explicit redistribution permission not independently verified |
| E02 Rerun peer experiment using README alone | **Open** — no literal `predict.py` rerun of classmate checkpoint; adapter test is not sufficient |
| E03 Record concrete gaps | **Done** — seed, missing manifest, dataset, version lock and script limitations recorded |
| E04 Write audit report | **Done** — this full report, with linked real test records and CI logs |
| E05 Own model card intended use + data | **Done** |
| E06 Own metrics, latency, size and limitations | **Done** |
| E07 Ethics / privacy / domain gap | **Done** |
| E08 Submit both artifacts and explain findings | **Partial** — both reports committed to GitHub; official teacher submission/defense unverified |

**Evidence-based total: 5 complete, 2 partial, 1 open. Not a claim of teacher acceptance or completed peer-run requirement.**
