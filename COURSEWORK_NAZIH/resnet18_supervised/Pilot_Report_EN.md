# ResNet18 supervised engineering pilot

**Status: training completed. No independent final test; no paper addition implemented yet.**

Full ResNet18 was fine-tuned on 70 already-accessed Motor images (35 Normal, 35 Defective). Validation uses 29 already-accessed images (15 Normal, 14 Defective). Numeric suffix groups are disjoint, but physical specimen independence remains unverified. This run verifies the supervised implementation; it does not resolve the data audit blocker.

| Epoch | Training loss | Validation loss | Validation macro-F1 | Validation accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.5444 | 1.0404 | 32.56% | 48.28% |
| 2 | 0.1805 | 0.2520 | 96.54% | 96.55% |
| 3 | 0.3043 | 0.7505 | 41.55% | 55.17% |

Selected epoch: 2 (validation macro-F1; earliest tie). Selection makes this validation result optimistic as a final-performance estimate.

Selected confusion matrix (rows true Normal/Defective, columns predicted Normal/Defective): [[15, 0], [1, 13]].

Independent scikit-learn recomputation matches the in-training confusion matrix and macro-F1. Convolution and classifier weights changed, gradients remained finite, and fresh model reload reproduced logits within tolerance. See result.json for numerical evidence and software versions.

Training runtime: 93.1 seconds on two CPU threads. Full classifier parameters: 11,177,538.

Do not interpret this result as the old 83.70% frozen-feature result improving or deteriorating: labels, learning regime and split are different. The first epoch predicting every validation image as Defective is retained in the history. No extra epochs or alternative settings were tried after seeing the outcomes.

Next: establish a defensible supervised development/confirmation dataset, choose a specific paper idea and implement a matched baseline/addition comparison. Current baseline settings remain a pilot starting point, not a final optimized protocol.

Read DECISION.md for source review, limitations and reproduction commands. Model weights, configuration, split manifest, per-image predictions and code are included.
