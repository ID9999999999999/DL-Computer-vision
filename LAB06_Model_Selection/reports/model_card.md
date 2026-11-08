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

Use a user-retake/verification path for low confidence and edge cases. Monitor false accepts (bad classified as good) separately from false rejects. Threshold tuning requires validation-only analysis and independent deployment-domain checks. Size/latency may change on a different machine. A separate ONNX CPU forward-only export benchmark has been executed (9 November 2026); it has since **confirmed exact 270-image Test accuracy preservation** on the private held-out Test; it is not an end-to-end or edge-device benchmark.

## Provenance

Actual measurement sources: `results/yolov8n/evaluation.json`, `results/yolov8n/results.json`, `results/yolov8n/training.json`, original model checkpoint in Drive. This card documents existing evidence, not new model training.


## Later evidence addendum — 9 November 2026

**Task 5C:** a continued-training matched control achieved 268/270 on the same held-out Test, while targeted train-only glare augmentation achieved 267/270. Both achieved glare recall 26/28, so targeted augmentation supplied no added gain against the matched control. These checkpoints do not replace the original archived model. See [Task 5C report](bonus_augmentation.md).

**Task 5D:** a separately validated ONNX export of the original checkpoint measured 2.965 ms forward-only versus PyTorch 8.651 ms on a 4-thread GitHub Actions CPU with 20 warm-ups and 100 timed runs; ONNX size 5.501 MiB versus original PT 2.826 MiB. After the earlier synthetic smoke test, we ran **all 270 original Test images** through both engines: **267/270 correct for both (98.89%, macro-F1 0.9889), no differences in 270 class decisions, zero accuracy drop**, and maximum P(bad) difference 3.5492e-6. Same-host forward-only CPU mean: **6.104 ms PyTorch vs 2.451 ms ONNX**, 20 warm-ups and 100 timed runs. This is not a production/end-to-end or edge-device benchmark. See [Task 5D measured memo](bonus_deployment_memo.md) and [verified artifact](https://drive.google.com/file/d/1xdHClyFQhvANfbYA2fs6r365yLawbvgR/view).


**Task 5E update:** the classmate's **unchanged** `predict.py` was executed on our 270-image Test with Ultralytics: **211/270** correct, identical class predictions to the previous adapter-based comparison. This does **not** verify the peer's reported result on their own unavailable original Test images. See [Task5E final audit](bonus_audit_note.md) and [raw execution ZIP](https://drive.google.com/file/d/1BmdQlk-aY5e97KGm7phdXk8P2pN6Aw2i/view).
