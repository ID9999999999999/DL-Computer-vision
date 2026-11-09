# Task 5E — Peer Reproducibility Audit

**Date:** 9 November 2026  
**Scope:** Reproduce one classmate's documented inference experiment using the supplied README and original files.

## 1. Materials received

The classmate supplied `yolov8n_quality_gate.zip`, containing a YOLOv8n classification checkpoint, `README.md` and `predict.py`. The files were used for a private coursework review. No permission to republish the classmate's checkpoint has been verified, so it has not been published in this repository.

The package did not provide the original training code, original held-out test images, split manifest or an exact dependency lock. These omissions limit reproduction of the **training experiment**, but not execution of the supplied **inference script**.

## 2. What was reproduced

The classmate's **unchanged `predict.py`** was executed using the supplied README instructions on the project's independent 270-image Dataset A test partition.

```bash
python predict.py --source <private_test_directory> --out <private_predictions.csv>
```

- Runtime: Ultralytics **8.4.208** (the working version was recorded; the original peer environment was not fully pinned).
- Test set: **270 images** — 135 `good`, 135 `bad`.
- Decision threshold: **0.50**.
- Script changes: **none**.
- Result: **exit code 0**, with 270 per-image predictions.

| Metric on Dataset A test set | Result |
| --- | ---: |
| Correct predictions | **211 / 270** |
| Accuracy | **78.15%** |
| Precision (`bad`) | **0.6959** |
| Recall (`bad`) | **1.0000** |
| F1 (`bad`) | **0.8207** |
| Confusion matrix (true/predicted: good, bad) | **[[76, 59], [0, 135]]** |

All 270 class predictions matched an earlier independent compatibility evaluation. The original script rounds scores to four decimals; the maximum observed score difference from the earlier unrounded output was approximately **0.000051**.

[Execution log, raw predictions and integrity records (private Drive)](https://drive.google.com/file/d/1BmdQlk-aY5e97KGm7phdXk8P2pN6Aw2i/view)

## 3. What worked, what was missing and what required a choice

| Check | Observation |
| --- | --- |
| Read and run the supplied inference instructions | **Worked.** The original script ran unchanged and produced all 270 predictions. |
| Load the supplied model | **Worked.** The checkpoint was usable with the recorded Ultralytics environment. |
| Match a separate prediction implementation | **Worked.** Zero class-label disagreements across 270 images. |
| Pin the original Python/Torch/Ultralytics environment | **Missing.** Exact original versions were not supplied as a complete lock. The working Ultralytics version is recorded above. |
| Reproduce original training | **Blocked by missing inputs.** Complete training code, the original dataset and split information were not available. |
| Reproduce the peer's reported own-test accuracy | **Not possible with the supplied files.** The original labeled test partition was unavailable. |
| Evaluate on a substitute test set | **Explicit evaluation choice.** The project's held-out Dataset A test images were used. This is a cross-dataset test, not reproduction of the peer's own-test result. |
| Publish the peer checkpoint | **Not done.** Public redistribution permission was not verified. |

**First blocking point:** the inference instructions worked. The first point preventing **full training reproducibility** was the absence of the original training procedure and corresponding dataset/split files. This is a missing-artifact limitation, **not** a claimed runtime exception from a training attempt.

## 4. Interpretation and limitations

The 59 errors are **good images predicted as bad** (false rejects). No `bad` image was predicted as `good` in this test. This result characterizes the classmate's model **on Dataset A**, not on its original test distribution. Different document generation or capture conditions could contribute to the performance difference, but no specific cause has been established.

The classmate's reported original-domain score and training procedure were **not independently reproduced**. No additional training or external dataset test is claimed.

## 5. Project documentation review

The same audit criteria were applied to the project repository:

- **Dataset access:** the main [README](../README.md) now links the original dataset and explains the required `metadata.jsonl`, `good/` and `bad/` structure.
- **Run isolation:** the Windows launcher now creates a fresh `runs/` output directory rather than writing into archived `results/`.
- **Reproduction instructions:** the README separates saved-result verification from training all four models.
- **Environment:** the recommended local Python version is distinguished from the actual recorded Colab environment.
- **Experiment history:** recorded metrics, protocols, checkpoints and historical deviations remain unchanged.

These documentation and launcher corrections have been committed. They improve usability but **do not substitute for an independently executed full retraining check**.

## 6. Deliverables and conclusion

- [Peer audit](bonus_audit_note.md) — this document.
- [Selected model card](model_card.md) — describes the selected YOLOv8n-cls classifier, measured results and limitations.

**Outcome:** the README-based **inference run** was reproducible; the peer's **original training and original-test experiment** was not reproducible from the supplied materials. The distinction is explicit so the audit can be evaluated without assuming unavailable results.
