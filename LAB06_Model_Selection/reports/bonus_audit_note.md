# Optional Task 5E — Peer exchange reproducibility audit (partial)

**9 November 2026. Permission/official acceptance is NOT independently evidenced.**

## Peer package supplied

`yolov8n_quality_gate.zip` contained: a YOLOv8n classification checkpoint, `predict.py`, and `README.md` (not the peer's raw training dataset). README names expected classes good/bad, installation command `pip install ultralytics`, invocation via `--source`, 224×224 input, 0.5 threshold and 24500003 seed; it describes a 1,260/270 synthetic train/test split and its own published test results.

## Actual external execution

Peer weights were evaluated **on our 270 held-out Test images**, without training: accuracy 0.7815 (211/270), macro F1 0.7705, 59 good→bad errors and 0 bad→good errors. Our weights evaluated on the identical split: 0.9889 (267/270), macro F1 0.9889. Raw per-image outputs and provenance are preserved in [Drive ZIP](https://drive.google.com/file/d/1CsRe-t3hDA8IH1ar4pQs8-zKg16JtKTF/view) and [evaluation report](cross_dataset_evaluation_2026-11-09.md).

## Reproducibility gaps (grounded, not conjectural results)

1. The peer package **does not include its image dataset**, image hashes or source-group manifest. It is impossible to independently verify non-overlap of our images with its training sources.
2. It supplies no frozen environment lockfile and no machine-independent latency procedure; the README specifies general installation and an author-specific laptop benchmark.
3. Its README documents one seed and hyperparameters, but does not supply a complete self-contained training reproduction pipeline and training logs in the exchanged ZIP.
4. We used a **PyTorch compatibility adapter** in our environment because Ultralytics could not be installed there. Our own model reproduced its archived confusion matrix exactly, but **this is not a literal Ultralytics rerun of the peer's README commands**. Consequently the strict 'rerun using README only' requirement is **not verified**.

## Follow-up

For a full Task 5E audit, rerun the peer `predict.py` in a working Ultralytics environment on its documented test split, with permission and class-source integrity checks. Until then this is a **partial audit**, not complete Task 5E or formal teacher acceptance.

## Our own documentation fix

Our share package now contains a short English README, a prediction script and the original checkpoint. Core repository gives source split, exact historical training deviations, manifest hash, model limitations and reproducible command examples.
