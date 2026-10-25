# Lab 06: five-minute speaking script

The first eight slides total 300 seconds. Rehearse at your natural pace. Three appendix slides support questions.

## 1. Document quality classification (20 seconds)

This lab compares four pretrained image classifiers for a document-quality gate. The task is to accept usable images and catch defective captures before further processing. I use a grouped holdout and a validation-based selection rule, then characterize the chosen model on the test set.

## 2. Dataset and source-level split (40 seconds)

The dataset contains eighteen hundred synthetic images from three hundred source documents. Each source produces three good and three bad variants. The source document is the split unit, so variants of the same document cannot appear in training and test. The exact split is seventy, fifteen and fifteen percent, stratified by document family. The held-out test contains only forty-five independent source documents, so its two hundred and seventy images are correlated.

## 3. Training and input conventions (40 seconds)

MobileNetV2 and EfficientNetB0 keep their ImageNet feature extractors frozen and train the same binary head. The two YOLO classification variants fine-tune the full model. All candidates receive the whole document at two hundred and twenty-four pixels square. Model-specific scaling stays explicit. I avoid extra cropping or erasing because it can hide or change quality defects. These are practical pipeline comparisons, with different optimization regimes, rather than a controlled architecture-only experiment.

## 4. Final test results (40 seconds)

All four models classify every good test image correctly. Their errors are defective images accepted as good. MobileNetV2 has the highest measured test accuracy, with one error, followed by YOLO small with two. Nano and EfficientNet each miss three bad images. One image changes accuracy by about zero point three seven percentage points. These small differences on forty-five test sources do not establish a reliable general ranking. I do not select a model from this test table.

## 5. Accuracy and CPU inference cost (35 seconds)

This chart combines final test accuracy with the separately measured CPU cost. Nano takes about twenty-seven milliseconds per image, MobileNet about thirty-eight, and the other two exceed the declared fifty-millisecond mean budget. Every timing uses batch one, four configured threads, twenty warm-up calls and one hundred measured calls. The scope includes preprocessing and inference on an already decoded image. The much lower GPU timings in the training logs measure a different device and scope.

## 6. Learning curves and checkpoint choice (40 seconds)

MobileNet training and validation loss decrease overall, although validation fluctuates. Its best validation loss occurs at epoch fifteen. Nano also shows decreasing losses and reaches its best validation top-one accuracy first at epoch seven. Patience stops the run after eleven epochs. YOLO chooses by validation fitness, so a later improvement in loss does not necessarily replace its checkpoint. EfficientNet shows unstable late validation loss and uses its best epoch-eleven checkpoint. Top-five accuracy is uninformative for two classes.

## 7. Glare causes every observed test error (45 seconds)

The selected nano model misses three of the twenty-eight glare examples, so glare recall is about eighty-nine percent. It detects every measured example of the other four defects, but those subsets contain only twenty-five to twenty-nine images each. In these false accepts, glare covers part of the portrait or printed fields. One defective image receives a bad probability of only zero point zero zero three. High aggregate accuracy therefore does not guarantee a safe quality gate. This result motivates a future glare-focused experiment on fresh held-out data.

## 8. Validation-based decision and limits (40 seconds)

The predeclared rule selects the fastest candidate meeting all validation and cost gates. Nano passes, with validation bad recall of ninety-eight point five two percent and mean CPU latency of twenty-six point nine seven milliseconds. Its checkpoint is two point eight three mebibytes. MobileNet also passes and remains the measured alternative when accuracy matters more than the lowest latency. The other candidates exceed the mean CPU budget in this run. The numerical gates are provisional project targets. Before deployment, real captures and unfamiliar documents need separate validation. I keep the test set fixed and do not tune against these results.

## 9. Appendix: size and training cost (appendix)

Use this table if asked about artifact size, parameter counts or total training time. Keras files can include optimizer state while YOLO strips it. Parameters describe saved unfused models. Early stopping makes total training times unequal budgets.

## 10. Appendix: test confusion matrices (appendix)

Rows are true labels and columns are predicted labels. Good is zero and bad is one. Every off-diagonal count is a bad image classified as good, so every observed error is a false accept.

## 11. Appendix: additional learning curves (appendix)

EfficientNet reaches its lowest validation loss at epoch eleven. Its later spikes indicate instability and possible overfitting, rather than a proven cause. YOLO small reaches its best top-one level at epoch three and stops at epoch seven. Its later loss decrease does not override the validation-fitness checkpoint rule.
