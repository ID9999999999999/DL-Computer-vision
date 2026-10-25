# Defense notes — understand the submitted work

## Data and leakage

One original document can produce several good and bad images. A random image split could put one variant in training and its near-identical counterpart in test. Source-grouped splitting prevents this. The supplied metadata has 300 sources and six variants per source. The new split selects 70, 15 and 15 sources within each 100-source document class.

Training data updates weights. Validation data chooses checkpoints and supports the final recommendation. Test data characterizes the already specified experiment; it must not choose thresholds, architectures or hyperparameters. The script's fixed threshold is 0.5. Returning to training after final test inspection would need a new independent final test set.

## Metrics

Here, positive means bad. TP is a defective image rejected correctly. FN is a defective image incorrectly accepted as good. FP is a usable image rejected as bad. TN is a usable image accepted correctly.

- Precision = TP / (TP + FP): among rejected images, how many are truly defective?
- Recall = TP / (TP + FN): among defective images, how many are rejected?
- F1 = 2 × precision × recall / (precision + recall).
- Accuracy = (TP + TN) / all images.

The confusion matrix has true labels on rows and predictions on columns, ordered good then bad. Its top-left cell is TN; top-right FP; bottom-left FN; bottom-right TP. The positive-class mapping must be explicit because folder ordering can reverse numeric IDs across frameworks.

## Models

The frozen MobileNetV2 base supplies convolutional feature maps. Global average pooling averages each spatial feature map into one number. Dense(128, ReLU) learns combinations; dropout randomly suppresses some head activations during training; one sigmoid estimates p(bad). Binary cross-entropy trains that probability against 0/1 targets. EfficientNetB0 uses the same head with a different feature extractor and its own built-in input scaling.

YOLOv8-cls classifies the whole image. It does not output document bounding boxes in this project. Nano and small are different capacity variants. Both are fine-tuned from ImageNet weights. The script reads model.names to find the bad-class output column.

## Cost and interpretation

Latency includes preprocessing and forward inference for one image already decoded in memory. It excludes file reading, model loading and warm-up. All models use the same CPU target and thread setting. A GPU training time is not directly comparable to CPU training time on another machine. Actual checkpoint size includes serialization overhead and may include optimizer state; it is not a theoretical parameter-count estimate.

Fifteen epochs are an upper budget, not a guarantee of convergence. The small number of source documents makes overfitting plausible. Synthetic glare patterns may differ from camera glare. Explain observed failures using the generated gallery and per-defect counts; do not state that glare was hardest unless the measured results support it.

## What remains outside this package

The package does not claim that experiments have already run, that the targets have been achieved, that a repository commit exists, or that the five-minute presentation has been delivered. Read your real report before submitting and be able to explain each decision in your own words.
