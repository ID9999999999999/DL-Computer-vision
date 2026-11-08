# Model card — Lab06 Document Quality Gate: YOLOv8n-cls

**Version:** archived best checkpoint from 25 October 2026; card written 9 November 2026.

## Intended use

First-stage quality classification (good versus bad) for incoming Russian document photos before OCR/Telegram-bot downstream processing. Intended for **research/demo** rather than unattended production decisions. No identity recognition, document authenticity determination or detection of private data is claimed.

## Training data and method

- Dataset A: 1,800 **synthetic** document images from 300 source documents covering passport, INN and SNILS.
- Train/validation/test: 1,260/270/270, source-grouped with seed 2473986 (no source overlap recorded).
- Labels: `good` (usable capture), `bad` (blur, glare, skew, occlusion, exposure). Training classification model: ImageNet-pretrained Ultralytics YOLOv8n-cls; full fine-tuning.
- Historical settings: input **224×224**, planned 15 epochs (11 executed), batch 16, early stopping patience 4; **these differ from the later instructor brief of 640/50/32/5**.

## Validated measurements (historical Dataset A held-out test, 270 images)

| Measure | Value |
|---|---:|
| Accuracy | 0.9889 (267/270) |
| Macro F1 | 0.9889 |
| Bad precision | 1.0000 |
| Bad recall | 0.9778 |
| Confusion matrix, rows true [good,bad], columns predicted [good,bad] | [[135,0],[3,132]] |
| Mean CPU latency | 26.97 ms/image (4 threads, 20 warm-up + 100 timed, original host) |
| Original checkpoint size | 2.963 MB (2.826 MiB) |
| Parameter count | 1,440,850 |

Selection of YOLOv8n as a candidate was based on **validation gates and CPU latency**, not on the 270-image test performance.

## Transfer test (peer model, not this card's weights)

Peer's different YOLOv8n checkpoint on our test: 211/270 correct (78.15% accuracy) with 59 false alarms; see [separate external evaluation](cross_dataset_evaluation_2026-11-09.md). **Our checkpoint has NOT yet been evaluated on the peer's original held-out images.**

## Risks and limitations

Synthetic images are not a replacement for real mobile capture with diverse cameras, shadows, paper, backgrounds and demographic/document variation. The course capstone cites a 2.8–5.8 percentage-point synthetic-to-real gap, **not an experimentally measured gap for this checkpoint**. Glare was a dominant error in the archived analysis; non-standard documents and extreme perspective may generalize poorly. Input images can contain highly sensitive personal identifiers; obtain permissions, minimize retention, and do not publish real ID images or image-level names linked to real users.

## Intended deployment guardrails

Use a user-retake/verification path for low confidence and edge cases. Monitor false accepts (bad classified as good) separately from false rejects. Threshold tuning requires validation-only analysis and independent deployment-domain checks. Size/latency may change on a different machine; no TFLite/ONNX performance is claimed.

## Provenance

Actual measurement sources: `results/yolov8n/evaluation.json`, `results/yolov8n/results.json`, `results/yolov8n/training.json`, original model checkpoint in Drive. This card documents existing evidence, not new model training.
