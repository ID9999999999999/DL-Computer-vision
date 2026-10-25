# Five-minute presentation — speaking outline

1. 0:00–0:40 — Problem: reject poor document images at a fast first gate. Explain good=0, bad=1 and the cost of a missed bad image.
2. 0:40–1:20 — Dataset: 1,800 variants, 300 source groups, inn/passport/snils; show protocol.json and the leakage-safe 70/15/15 allocation. Explain why related variants cannot cross splits.
3. 1:20–2:10 — Models: MobileNetV2 and EfficientNetB0 with frozen ImageNet bases; YOLOv8 nano and small classification with full fine-tuning. Explain preprocessing and the fixed threshold.
4. 2:10–3:10 — Show comparison.csv, accuracy_vs_speed.png, and confusion_matrices.png. Explain precision, recall, F1, batch-one CPU timing and checkpoint size.
5. 3:10–4:10 — Show training curves and failure galleries. Lowest measured defect recall for yolov8n: glare (0.893); do not assume glare.
6. 4:10–5:00 — Validation-selected candidate: yolov8n. Acceptance status: passed provisional validation limits. Explain the synthetic-to-real limitation and the next independent validation study.

Before class: read Final_Report.md, inspect every plotted result and be ready to explain source grouping, model freezing, normalization, bad-class mapping, checkpoint selection and the difference between validation and test data. This is an outline; the presentation has not been delivered.
