# Task 4 — Five-part deployment decision memo

## 1. Decision

Select **YOLOv8n-cls** as a **provisional** document-quality gate under the established validation-and-cost rule.

## 2. Evidence

On the final 270-image test split: test accuracy **98.89%**, macro F1 **0.9889**, bad-class recall **97.78%**, CPU latency **26.97 ms/image**, checkpoint size **2.83 MiB**. YOLOv8n satisfied the predeclared validation thresholds: accuracy >= 0.90, bad F1 >= 0.90, bad recall >= 0.95, CPU latency <= 50 ms and artifact <= 100 MiB. The 50 ms threshold was an *engineering choice*, not a teacher-prescribed value. It also fits beneath the course example first-gate budget of 100 ms.

## 3. Alternatives and reasons

**MobileNetV2** is more accurate (99.63%, one false accept) but slower (37.69 ms, 11.06 MiB); it is the alternative when bad-image recall is weighted more heavily. **EfficientNetB0** measured 98.89%, 56.69 ms (above provisional latency limit). **YOLOv8s-cls** measured 99.26%, 51.07 ms, slightly over the same limit. Any deployment must be validated on target hardware rather than inferred from this host's CPU measurement.

## 4. Risks and mitigation

All observed test mistakes were glare-related false accepts. The chosen model missed 3 of 28 glare cases. Synthetic augmentation and repeated document variants may overestimate field performance. The 2.8–5.8% synthetic-to-real gap cited by course materials is external context, **not measured in this project**. Candidate mitigation actions: collect fresh real-camera captures; augment glare on the training set only; evaluate threshold options using validation only; retain an operator verification loop for uncertain images. Do not tune on the current test data.

## 5. Deployment plan

Start with the existing CPU-server quality gate. An **ONNX export was subsequently completed in optional Task 5D**: the exported model preserved all 270 class predictions, and the same-host CPU forward-only benchmark improved from 6.104 ms (PyTorch) to 2.451 ms (ONNX). These timings are not full pipeline or edge-device benchmarks. See [Task 5D deployment memo](bonus_deployment_memo.md).

## Submission audit

See [full comparison](comparison.md), [four-slide speaking script](../presentation/Final_4_Slide_Script.md) and the [instructor requirement audit](../docs/INSTRUCTOR_REQUIREMENTS_AUDIT.md). Original measurements are unchanged.
