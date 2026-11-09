# Lab 06 — Exact four-slide speaking script (English)
**Prepared 9 November 2026.** Designed for the submitted four-slide deck, ~5 minutes total. This is rehearsal material; not proof of delivered presentation.

## Slide 1 (60 seconds): Task and dataset
Hello. My project is document image-quality classification. The first quality-control gate decides whether a photo is usable, labeled good, or defective, labeled bad. Defects include glare, blur, skew, occlusion and exposure. The dataset contains 1,800 synthetic examples derived from 300 underlying source documents, covering INN, passports, and SNILS. The experiment uses seed 2473986 and a source-grouped split: 1,260 training, 270 validation and 270 test images. Grouping by source prevents near-identical variants from appearing in both training and test. The test set has only 45 independent sources, which limits conclusions.

## Slide 2 (90 seconds): Four measured models
I compare four pretrained classifiers on the same split: frozen MobileNetV2, frozen EfficientNetB0, YOLOv8n classification, and YOLOv8s classification. They use 224-by-224 RGB inputs in the completed experiments. Their optimization regimes differ, so this is a comparison of practical pipelines, not only architectures. Test accuracies were 99.63%, 98.89%, 98.89%, and 99.26%, respectively. CPU batch-one mean latencies were 37.69, 56.69, 26.97, and 51.07 milliseconds per image. Nano has the lowest observed latency and an approximately 2.83 MiB checkpoint. Model selection was based on a previously stated validation-and-cost rule, not on the test ranking.

## Slide 3 (75 seconds): Errors and risks
The selected YOLOv8n incorrectly accepts three defective test examples as good. All three are glare-related. On the measured glare subset, its recall is 25 out of 28, or about 89.29%. These are the three genuine YOLOv8n errors on the held-out test set. The small sample and synthetic images mean performance may fall on new cameras, lighting, or document designs. This matters because accepting defective input can create downstream document-recognition errors.

## Slide 4 (75 seconds): Decision and next steps
Under the predefined validation and CPU-cost criteria, YOLOv8n is the provisional recommendation, with 98.89% test accuracy, 0.9889 macro-F1 and 26.97 milliseconds mean CPU latency. MobileNetV2 remains a strong accuracy alternative: 99.63% accuracy but higher measured latency. The main deployment risks are false accepts caused by glare and an unknown performance gap on real mobile photos. Possible safeguards include validation-only threshold studies, improving training images and an operator review option. The actual training settings were recorded faithfully and subsequently accepted by the instructor. Thank you.
