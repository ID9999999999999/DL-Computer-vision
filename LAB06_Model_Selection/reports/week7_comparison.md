# Task 3 — Week 7 YOLOv8 classification comparison

> **Audit note:** this is a reformatting of the saved results of 25 October 2026, **not a new training run**. The experiments used max 15 epochs, batch size 16, early stopping patience 4, input 224×224 for all models (including YOLO), whereas the teacher's detailed requirements specify 50/32/5 and 640×640 for YOLO. No metrics have been altered.

## Comparison

| Model | Epochs run | Test accuracy | Macro precision | Macro recall | Macro F1 | Mean CPU ms | Checkpoint MiB | Parameters | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 15 | 99.63% | 0.9963 | 0.9963 | 0.9963 | 37.69 | 11.06 | 2,422,081 | 247.9 |
| EfficientNetB0 | 15 | 98.89% | 0.9891 | 0.9889 | 0.9889 | 56.69 | 18.14 | 4,213,668 | 272.4 |
| YOLOv8n-cls | 11 | 98.89% | 0.9891 | 0.9889 | 0.9889 | 26.97 | 2.83 | 1,440,850 | 232.3 |
| YOLOv8s-cls | 7 | 99.26% | 0.9927 | 0.9926 | 0.9926 | 51.07 | 9.78 | 5,083,298 | 176.0 |


YOLOv8n-cls reached **98.89%** test accuracy compared with MobileNetV2's **99.63%**, a decrease of **0.74 percentage points** (two additional errors on 270 test images). Its mean CPU latency was **26.97 ms/image** versus **37.69 ms** for MobileNetV2, a 10.72 ms advantage in the recorded protocol. The YOLOv8n checkpoint was **2.83 MiB** versus **11.06 MiB** for MobileNetV2; serialization conventions also differ between PyTorch and Keras. YOLOv8s-cls achieved **99.26%** test accuracy but required **51.07 ms/image**. These models differ in architecture, parameter count, head type and training regime. Because the actual study used YOLO 224×224, it does **not** measure the course-prescribed YOLO 640×640 setup.

The same source-grouped split and seed were used for all models. Every CPU timing includes 20 warm-up calls and 100 timed calls (batch 1, four threads). See [YOLOv8n JSON](../results/week7_yolov8n_cls/results.json) and [YOLOv8s JSON](../results/week7_yolov8s_cls/results.json).
