# Bonus Task 5B — Validation threshold sweep
**Date:** 2026-11-09 | **Model:** Original YOLOv8n-cls checkpoint | **Data:** validation only, 270 saved predictions.

## Protocol
Use recorded `results/yolov8n/predictions.json` **split=val** (n=270; 135 good/135 bad), with **bad** as positive and the original saved `p_bad` probabilities. Predict bad if `p_bad >= threshold`. Evaluate 17 thresholds from 0.10 through 0.90, stepping by 0.05. No training or test-set threshold selection is performed. Select a descriptive F1 maximum on validation; ties prefer threshold closest to existing 0.50 (then smaller threshold). This is a bonus sensitivity analysis **after the original model-selection experiment**, not a retroactive claim that any new threshold was preregistered.

## Results
- Existing/default 0.50: TP=133, FP=0, TN=135, FN=2; precision=1.0000, recall=0.9852, F1=0.9925.
- F1-maximizing validation choice: **0.10**, TP=135, FP=0, TN=135, FN=0; precision=1.0000, recall=1.0000, F1=1.0000.
- Change vs default: false accepts (bad classified good) **-2**, false rejects (good classified bad) **+0**.
- Under a *hypothetical* asymmetric cost of five units per false accept (FN) and one per false reject (FP), the lowest **validation** loss among swept thresholds is **0**. This cost ratio is an illustrative assumption, **not an instructor-specified, measured, or pre-registered business cost**. Corresponding threshold(s): 0.10.

## Interpretation
A false accept (FN) allows a defective image into the next processing stage; a false reject (FP) asks the user to recapture a usable image. Therefore minimizing F1-related classification errors and minimizing a business cost are not always the same goal. The reported F1-optimal threshold is descriptive and must be validated prospectively on new external data before operational use.

- Full numeric sweep: [threshold_sweep_val.csv](../results/bonus_threshold_sweep_val.csv).
- Plot: [validation precision/recall vs threshold](../results/bonus_threshold_precision_recall.svg). This is a threshold-based precision/recall trace, not a separately measured deployment benchmark.
- **Test results in the historical report remain at threshold 0.50.** We did not use the 270 test labels to choose a new threshold.
