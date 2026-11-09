# Task 1 — Model selection memo

Date: 19 October 2026. Seed: 2473986. Status: written before training; no measured model results yet.

## 1. Decision context

The first pipeline gate classifies a document image as usable (good=0) or defective (bad=1). It must catch poor captures before downstream document processing, at acceptable inference cost. The supplied metadata describes 1,800 synthetic variants from 300 source documents: 600 images each for inn, passport and snils, with 900 good and 900 bad images. Bad-image defects include blur, glare, skew, occlusion and exposure. These are metadata facts, not a new verification of local image pixels.

The effective independent sample count is closer to the number of source documents than to the number of variants. Splitting individual variants independently would leak source appearance and overstate generalization. The old split contains 1230/228/342 images; the evaluation guide asks for 70/15/15. The new experiment groups sources and allocates 70/15/15 sources within each document category, producing 1260/270/270 images. The original manifest is retained. Previously test-driven decisions, if any, would require fresh final test sources; reallocation alone cannot undo prior test exposure.

## 2. Architecture comparison

| Candidate | Technical description | Why include it at the first gate? | Main risk with this dataset |
|---|---|---|---|
| MobileNetV2 | Inverted residual blocks, depthwise convolutions, linear bottlenecks [1]. ImageNet base with global average pooling, Dense(128, ReLU), Dropout(0.3), Dense(1, sigmoid). | A lightweight baseline intended for constrained inference; frozen features reduce training cost. Actual local latency and task accuracy must be measured. | Frozen semantic features may miss subtle glare or blur cues. |
| EfficientNetB0 | EfficientNet baseline within a family that balances depth, width and resolution through compound scaling [2]. ImageNet base with the same binary head. | Alternative accuracy/compute trade-off at 224×224; tests whether a different feature extractor improves quality discrimination. | More compute is not proof of better quality recall; a small synthetic set can favor shortcuts. |
| YOLOv8n-cls | Ultralytics YOLOv8 classification variant, pretrained on ImageNet, then a binary classification head [3]. This is image classification, not object detection. | A nano candidate with a practical training/inference toolchain. | Framework and full fine-tuning regime differ from the frozen Keras baselines. |
| YOLOv8s-cls | Small variant of the same classification family [3]. | Tests whether extra capacity earns sufficient validation improvement to justify cost. | Higher capacity can overfit; speed and size must be measured, not assumed. |

MobileNetV2 is the initial baseline because the gate prioritizes low inference cost and the dataset is small. This is a starting choice, not a final empirical recommendation. The final candidate is chosen by the preregistered validation rule below.

## 3. Transfer learning and training regimes

Transfer learning reuses a representation learned on a larger source task. ImageNet pretraining supplies generic visual filters; the binary head learns the local good/bad task. It does not imply that ImageNet accuracy transfers directly to document-quality accuracy.

In the frozen-base regime, only the new head is optimized. A partially unfrozen regime instead updates selected late feature blocks as well as the head. These are distinct settings: freezing the entire base is not the same as training its last blocks. The submitted Keras baseline uses a fully frozen base and explicitly calls it with training=False. This keeps batch-normalization statistics fixed [4].

In full fine-tuning, all feature stages and the head can adapt. This can learn task-specific cues but raises training cost and overfitting risk. The two YOLO variants use full fine-tuning with Adam and learning rate 1e-4. The Keras frozen heads use Adam at 1e-3. If a future Keras fine-tuning experiment is added, unfreeze the desired blocks, recompile at a lower learning rate, retain appropriate batch-normalization behavior, select using validation, and reserve fresh independent final test data if prior test analysis has influenced the change [4].

## 4. Preprocessing and fairness

All candidates receive RGB, whole-document, bilinear-resized 224×224 inputs. No new random augmentations are applied: blurring, erasing or cropping could modify the quality label or hide the defect. A custom Ultralytics classification trainer replaces its crop-based transforms with a whole-image resize; this is important because the standard toolchain uses random resized crops for training [3].

MobileNetV2 scales pixels from 0–255 to −1–1 inside the exported model [5]. Keras EfficientNetB0 contains its own input rescaling and expects 0–255 inputs, so no second division by 255 is applied [6]. YOLO uses its classifier input convention with tensors in 0–1. The binary semantic mapping is always bad=1 in evaluation, even though Ultralytics folder ordering commonly assigns bad index 0; code reads the model's class names explicitly.

All candidates use the identical source-grouped manifest and an upper limit of 15 epochs, batch size 16, seed 2473986 and early stopping patience 4. The number of optimized parameters and checkpoint-selection rules differ. The comparison evaluates practical complete pipelines; it does not isolate architecture as the sole cause of a performance difference.

## 5. Predeclared success criteria

These are proposed engineering targets because the grading guide does not supply numeric deployment limits. They must be approved or adjusted in config.json before the first run; they are not claims about teacher requirements.

| Measure | Target | Reason |
|---|---:|---|
| Accuracy | ≥ 0.90 | Overall binary performance, never reported alone |
| Precision, bad class | Report, no independent cutoff | Quantifies avoidable rejection of usable images |
| Recall, bad class | ≥ 0.95 | Prioritizes catching defective captures |
| F1, bad class | ≥ 0.90 | Balances precision and recall |
| Mean CPU latency, batch one | ≤ 50 ms/image | Provisional first-gate budget; hardware-dependent |
| Saved checkpoint | ≤ 100 MiB | Provisional artifact budget, including framework serialization |

Use threshold 0.5 throughout; no threshold search is performed. Record precision, recall, F1, macro-F1 and a confusion matrix with rows=true, columns=predicted, order good/bad. Select the fastest candidate that passes all validation and cost targets. If none pass, retain the highest validation bad-class F1 candidate as provisional and explicitly decline deployment acceptance. Do not choose a replacement based on final test performance.

Latency protocol: CPU for all candidates, four configured threads, batch one, 20 warm-up calls and 100 timed calls cycling up to 32 decoded RGB validation images held in RAM. Time resize, input conversion/normalization, forward pass and host probability output. Exclude disk reading, image decode, weight loading and warm-up. Report mean, median and p95. Device/framework differences remain a limitation. Size is the actual best checkpoint; parameters are counted from the binary model. Training time measures the training API call, including its validation and callback overhead.

## 6. Expected outputs and interpretation

The runner will save epoch curves, test predictions, per-model metrics, model-size and latency measurements, a comparison table, accuracy-versus-speed plot, side-by-side confusion matrices and failure galleries. The decision report uses actual saved outputs. Glare is a hypothesis for a difficult defect, not a result to hardcode; per-defect recall will determine the observed hardest category, and ties/small supports must be considered.

A widening train/validation gap combined with deteriorating validation loss supports an overfitting hypothesis. Poor train and validation performance can support an underfitting hypothesis, but must also prompt checks of preprocessing, labels, optimization and the frozen representation. Curves alone do not establish the cause. A perfect test score would still not prove real-world suitability because synthetic variants and source correlations limit the inference.

## References

[1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR. https://arxiv.org/abs/1801.04381

[2] Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML. https://arxiv.org/abs/1905.11946

[3] Ultralytics. *YOLOv8 model documentation* and *Image Classification*. Accessed 19 October 2026. https://docs.ultralytics.com/models/yolov8/ and https://docs.ultralytics.com/tasks/classify/

[4] Keras. *Transfer learning & fine-tuning*. Accessed 19 October 2026. https://keras.io/guides/transfer_learning/

[5] TensorFlow. *tf.keras.applications.MobileNetV2*. Accessed 19 October 2026. https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2

[6] Keras. *EfficientNet models*. Accessed 19 October 2026. https://keras.io/api/applications/efficientnet/efficientnet_models/
