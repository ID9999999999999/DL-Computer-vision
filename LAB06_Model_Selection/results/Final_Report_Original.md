# Final comparison and decision memo

Generated from actual local runs; no manually invented metrics.

## Protocol

Seed 2473986. Source-grouped split {'train': 1260, 'val': 270, 'test': 270}; original Lab05 split {'train': 1230, 'validation': 228, 'test': 342}.
All variants of each source remain together. RGB, 224×224, positive class bad=1, fixed threshold 0.5. No test labels were used for checkpoint choice, thresholds, or recommendation.

## Final test comparison

| Model | Accuracy | Precision bad | Recall bad | F1 bad | CPU ms | MiB | Parameters | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mobilenetv2 | 0.9963 | 1.0000 | 0.9926 | 0.9963 | 37.69 | 11.06 | 2,422,081 | 247.9 |
| efficientnetb0 | 0.9889 | 1.0000 | 0.9778 | 0.9888 | 56.69 | 18.14 | 4,213,668 | 272.4 |
| yolov8n | 0.9889 | 1.0000 | 0.9778 | 0.9888 | 26.97 | 2.83 | 1,440,850 | 232.3 |
| yolov8s | 0.9926 | 1.0000 | 0.9852 | 0.9925 | 51.07 | 9.78 | 5,083,298 | 176.0 |

## Recommendation based on validation

Selected candidate: **yolov8n**.
The candidate passed all provisional validation and cost limits and was the fastest passing candidate.
Validation F1=0.9925; validation bad recall=0.9852.
The selected candidate also passes the same limits on final test characterization. This observation does not trigger reselection or tuning.

Alternatives: the other measured candidates remain alternatives only if validated under a new preregistered experiment or on fresh independent data. A faster model can still be unsuitable if it accepts defective documents.

## Defect analysis

Lowest measured defect recall for the selected model: glare (0.893, n=28). Ties are possible; inspect defect_analysis.csv. Glare was an expectation, not an imposed conclusion.
The failure gallery shows the most confidently wrong examples. Bad→good mistakes allow unusable documents through the first gate; good→bad mistakes cause avoidable retries.

## Training dynamics

- mobilenetv2: best validation-loss epoch 15; train/validation accuracy gap -0.006. No strong late validation-loss rise by the declared 0.03 descriptive margin. Inspect curves.png before claiming overfitting or underfitting.
- efficientnetb0: best validation-loss epoch 11; train/validation accuracy gap -0.003. Possible overfitting: validation loss finishes above its earlier minimum. Inspect curves.png before claiming overfitting or underfitting.

YOLO learning curves and epoch metrics are in each YOLO run's results.png and results.csv. Compare training and validation loss trends there; do not infer underfitting from one metric alone.

## Limits and risks

This small synthetic dataset contains only 300 independent source documents. Image counts overstate independent sample size because variants are correlated. Real camera captures and unfamiliar documents require separate validation. Results compare complete training pipelines: Keras frozen bases versus YOLO full fine-tuning, different optimizers/checkpoint criteria, and different frameworks. No causal architecture-only claim is justified.
CPU batch-one timings use the same decoded RGB images, thread count and timing scope; framework and host load still affect results. Artifact size is the actual checkpoint (including framework serialization and any optimizer state), not raw weight bytes. Training times include fit/trainer overhead and are not equivalent optimization budgets.
A 0.5 sigmoid/softmax threshold does not establish calibration. Weak performance can reflect missing quality cues in frozen ImageNet features; a fresh experiment may test low-rate fine-tuning while holding out fresh final test data.

## Completion status

Models, metrics and plots are complete only if this report generated successfully. Repository commit, student review/defense, and the in-class presentation are separate actions still required.