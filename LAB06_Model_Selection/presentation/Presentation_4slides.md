# Four-slide presentation — Lab 06

**Approximate speaking time: 5 minutes.** Source for the new four-slide PPTX; original eight-slide presentation is retained.

## Slide 1 — Pipeline context (60 seconds)
- First gate classifies document photos as **good** or **bad** before downstream processing.
- Defects: blur, glare, skew, occlusion, exposure.
- 1,800 synthetic images from 300 source documents; split 1,260 training / 270 validation / 270 test, grouped by document source.

## Slide 2 — Comparison and timing (90 seconds)
| Model | Epochs run | Test accuracy | Macro precision | Macro recall | Macro F1 | Mean CPU ms | Checkpoint MiB | Parameters | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 15 | 99.63% | 0.9963 | 0.9963 | 0.9963 | 37.69 | 11.06 | 2,422,081 | 247.9 |
| EfficientNetB0 | 15 | 98.89% | 0.9891 | 0.9889 | 0.9889 | 56.69 | 18.14 | 4,213,668 | 272.4 |
| YOLOv8n-cls | 11 | 98.89% | 0.9891 | 0.9889 | 0.9889 | 26.97 | 2.83 | 1,440,850 | 232.3 |
| YOLOv8s-cls | 7 | 99.26% | 0.9927 | 0.9926 | 0.9926 | 51.07 | 9.78 | 5,083,298 | 176.0 |

- The same held-out test images and seed 2473986 were used throughout.
- [Accuracy-versus-latency chart](../results/accuracy_vs_speed.png).

## Slide 3 — Failure gallery and glare risk (75 seconds)
- [True YOLOv8n error images in private Drive](https://drive.google.com/drive/folders/1080wwwEPFhfJWwaGpK4fYhDNlGAZAU4I).
- YOLOv8n false accepts: 3; glare recall 25/28 = 89.29%.
- Individual test error images: `bad_passport_glare_032_v2`, `bad_passport_glare_287_v2`, `bad_snils_glare_022_v1`.
- Training-only glare augmentation is a future mitigation, not a completed new experiment.

## Slide 4 — Decision, risks and limitations (75 seconds)
- **Choose YOLOv8n-cls** (original validation-based selection).
- Test accuracy 98.89%, macro F1 0.9889, CPU 26.97 ms/image, size 2.83 MiB.
- Alternative MobileNetV2: accuracy 99.63%, 37.69 ms/image, size 11.06 MiB.
- Biggest risks: glare false accepts and synthetic-to-real domain shift.
- **Transparent discrepancy:** course later requested max epochs 50 / batch 32 / patience 5 / YOLO 640, but saved run used 15 / 16 / 4 / YOLO 224. Do not claim otherwise.
