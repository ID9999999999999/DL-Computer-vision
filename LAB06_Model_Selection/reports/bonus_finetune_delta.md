# Task 5A — Frozen versus Top-30 Fine-Tuning Delta

**Date:** 8 November 2026. **Source split:** exact archived 70/15/15 source-grouped manifest (1260/270/270); seed **2473986**. 

## What changed

The pretrained MobileNetV2 **original frozen checkpoint** (SHA256 `014226cd942aee422a04b19835eba9bf89c198939b996a88cf14b955eeed854e`) was reloaded. We unfroze the final 30 named backbone layers, retained batch-normalization layers frozen for stable statistics, and kept the custom binary head trainable. Adam learning rate **1e-5**, max **50 epochs**, early-stopping **patience 5**, **effective batch size 32** (four microbatches of 8 because of the runtime memory cap). The teacher calls 1e-5 “10 times smaller”; relative to the original head-learning rate 1e-3, it is mathematically **100 times smaller**. Our experiment follows the explicit numeric value 1e-5.

## Frozen vs fine-tuned (same 270-image test set)

| Metric | Original frozen (archived) | Fine-tuned top-30 | Delta |
|---|---:|---:|---:|
| Accuracy | 0.9963 | 0.9963 | +0.0000 |
| Precision macro | 0.9963 | 0.9963 | +0.0000 |
| Recall macro | 0.9963 | 0.9963 | +0.0000 |
| F1 macro | 0.9963 | 0.9963 | +0.0000 |
| Incorrect test images | 1 | 1 | +0 |

**Training:** 20 epochs; 614.7 s; best epoch 15. **Train–validation accuracy gap** at last epoch: +0.0066. **Baseline CPU latency remeasured:** 64.16 ms/image vs fine-tuned 69.40 ms/image (same warm-up, 100 timed runs, Keras torch CPU, 4 threads); they are runtime- and load-dependent. **Checkpoint sizes:** old 11.06 MiB; new 9.80 MiB (serialization/optimizer state may differ, even if architecture parameter count does not).

## Five-line delta memo

1. Fine-tuning changed held-out accuracy by **+0.0000** and macro F1 by **+0.0000**, compared with the saved frozen baseline.
2. The modified classifier produced **1** test mistakes versus **1** for the baseline, so this is the measured change in false accepts/rejects.
3. The final train–validation accuracy gap is **+0.0066**. The validation-loss trajectory and gap must both be inspected before declaring overfitting.
4. End-to-end CPU latency was remeasured under the same backend: frozen **64.16 ms**, fine-tuned **69.40 ms**; checkpoint size changed from **11.06** to **9.80 MiB**, possibly due to serialization metadata.
5. **Deployment verdict:** Prefer keeping the frozen baseline; the new score does not beat its archived F1.

## Limitations

Only 300 independent source documents underlie 1,800 synthetic variants. The same held-out test set was analyzed in earlier tasks; final unbiased evaluation on fresh unseen source documents is preferable. The original baseline was trained on TensorFlow/T4; fine-tuning here used Keras with the compatible PyTorch CPU backend, with preserved source weights. The new training uses physical microbatches of 8 with gradient accumulation to match an effective 32-image update; this differs in execution memory layout from one physical 32-image batch. The original baseline test metrics are archived, while CPU latencies are rebenchmarked on the same host/backend for comparability. Training stopped according to validation loss and **test data was not used for hyperparameter selection**.
