# Optional Task 5B — Validation threshold and error cost

**9 November 2026 · NO RETRAINING · no test-based threshold optimization**

The original YOLOv8n classifier used **P(bad) >= 0.50**. A grid of 17 thresholds was evaluated (0.10–0.90 in 0.05 increments) from the model's saved **270 validation** probabilities. The official historical model decision and test report are not changed.

## Operating policy (illustrative, not supplied by instructor)

A **false accept** (real bad document predicted good) sends an unreadable image downstream and may yield incorrect recognition. A **false reject** (real good document predicted bad) causes a user retake, wasting time but preserving correctness. To explore this asymmetry, use hypothetical validation cost `5 × FN + 1 × FP`; it is a transparent **assumption**, not a measured business cost.

| Validation threshold | TP bad | FN bad→good | FP good→bad | TN good | Precision bad | Recall bad | F1 bad | Illustrative cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 default | 133 | 2 | 0 | 135 | 1.0000 | 0.9852 | 0.9925 | 10 |
| 0.10 validation candidate | 135 | 0 | 0 | 135 | 1.0000 | 1.0000 | 1.0000 | 0 |

The grid's smallest illustrative cost was at **0.10**; it removed 2 missed bad validation images without adding false rejects **on this synthetic validation set**. This is an encouraging exploratory operating point, **not evidence of zero errors on real phone photos or a production-safe threshold**. The class-probability calibration, source diversity and genuine costs should be checked on a fresh independent dataset before changing the deployment default. Until then retain the historical 0.50 threshold as the reference.

- [Full sweep CSV](../results/bonus_threshold_yolov8n/threshold_sweep.csv)
- [Precision–recall chart](../results/bonus_threshold_yolov8n/precision_recall_curve.svg)
- [Analysis record](../results/bonus_threshold_yolov8n/results.json)
- Validated **no test examples** entered the threshold selection or cost grid.
