# TASK 5A — Partial Fine-Tuning of MobileNetV2

Date: 8 November 2026. Original experiment: 25 October 2026. Seed: 2473986.

## VERIFIED OUTCOME

This is a **completed measured experiment**. It loads the original frozen MobileNetV2 checkpoint with SHA256 `014226cd942aee422a04b19835eba9bf89c198939b996a88cf14b955eeed854e` and fine-tunes the final 30 named MobileNetV2 backbone layers while keeping BatchNorm frozen. The classification head also remains trainable. **No weights outside the allowed backbone region changed**, as independently checked against the original model. The test images and recorded split match the original 1,800-image source-grouped manifest (1260 train / 270 validation / 270 test), with zero source overlap and matching image hashes.

- Training: max 50 epochs; **20 actual epochs**; patience 5; best validation-loss checkpoint at **epoch 15**.
- Adam learning rate: **1e-5**. The original head LR was 1e-3 (numeric 100x reduction, despite the task sheet's "10x" wording).
- Effective batch 32: implemented as 4 gradient-accumulated microbatches of 8 due to the 4 GiB execution memory limit; this is a disclosed implementation deviation from a literal physical batch of 32.
- Runtime: Keras 3 with PyTorch CPU backend. TensorFlow was not available in this environment. Original Keras `.keras` checkpoint loaded successfully and underwent real gradient updates.
- Final test accuracy: **99.63%**; macro F1 **0.9963**, same as the archived frozen model; **1** bad image falsely accepted.
- Confusion matrix [good,bad]: `[[135, 0], [1, 134]]`.
- CPU benchmark with identical backend and 20+100 protocol: frozen 64.16 ms/image; fine-tuned 69.40 ms/image. Host timing and serialization differ from the original TensorFlow experiment.
- Saved weights: 9.80 MiB; original frozen checkpoint 11.06 MiB. Parameter count unchanged.

**Interpretation:** No improvement in final held-out accuracy or F1. The appropriate conclusion is that fine-tuning did not improve this already strong baseline on the archived synthetic test split; do not claim an improvement. Further independent real-world evaluation is necessary.

## Main files

- `train_task5a.py`: full self-contained training, validation, test-evaluation, reporting and latency protocol.
- `Task5A_FineTuning_Reproducible.ipynb`: standalone Colab notebook with the full script and Drive path setup.
- `results/mobilenetv2_finetune_top30/best.keras`: best actual fine-tuned model.
- `results/mobilenetv2_finetune_top30/results.json`: required phase-schema experiment row including metrics, errors, timing, and full history.
- `results/mobilenetv2_finetune_top30/curves.png`: train/validation loss and accuracy.
- `results/mobilenetv2_finetune_top30/confusion_matrix.png`: final test confusion matrix.
- `results/mobilenetv2_finetune_top30/errors/`: genuine misclassified test image(s), with `errors.json` index.
- `results/mobilenetv2_finetune_top30/predictions_test.json`: test probabilities.
- `reports/bonus_finetune_delta.md`: delta table and five-line conclusion.
- `train.log`: genuine training console log, from initial validation to early stopping and evaluation.

## Input sources kept outside this package

1. `demo_quality_dataset.zip` in the student's Google Drive root (399 MB).
2. `Lab06_results/manifest.json` in Google Drive (original source-grouped manifest).
3. `Lab06_results/mobilenetv2/best.keras` in Google Drive (original frozen checkpoint).

**Source integrity rule:** Do not substitute a new split or train against the archived test set. The script checks the original checkpoint SHA, all source IDs, image paths, and 1,800 hashes prior to training.

## Limitations

The original test split was previously evaluated in Task 2 and inspected during Task 4; a wholly new, independently sampled test is preferable for an unbiased future generalization claim. This completion is **on our original data**, not the classmate-exchange requirement, which remains separate. This training uses the CPU PyTorch backend because the current execution host lacks TensorFlow/GPU; it should not be misrepresented as the original TensorFlow/T4 regime.
