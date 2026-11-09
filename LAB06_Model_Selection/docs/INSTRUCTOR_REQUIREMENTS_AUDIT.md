# Lab 06 — Instructor Requirement Cross-check

**Source:** four original instructor files in [00_INSTRUCTIONS](https://drive.google.com/drive/folders/1pRFQQHEpripLdUE4sx6Qd0sqHW7UjD7o): Task 1, Requirements for Tasks 2–4, Evaluation Guide, and Task 5A–5E.  
**Audit date:** 9 November 2026.  
**Legend:** **Met** = evidenced in current project; **Difference** = implemented differently from the literal instruction; **Pending** = requires an in-class action or missing verification; **Bonus** = optional in the written instructions.

See [current status](SUBMISSION_STATUS.md) for the only active action list. This audit preserves teacher wording and observed technical differences without creating artificial obligations.

## Task 1 — Model Selection Memo (required)

| Instructor requirement | Evidence / result | Status |
| --- | --- | --- |
| Compare ≥3 architectures; parameters, input, strengths, project risks | [Four-model comparison](Model_Selection_Memo.md) | Met |
| ≥3 credible sources from papers or official documentation | Six cited references in dated memo | Met |
| Describe first Telegram quality gate, latency, size, CPU/server and small synthetic data | Memo sections 1, 2 and 4 | Met |
| Explain frozen base, partial and full fine-tuning | Memo section 3 | Met |
| Define numeric acceptance thresholds **before** experiments | Dated 19 October memo, [config](../config.json), original runs 25 October | Met (historical record) |
| Supply Word memo in personal folder | [Task 1 Word file](https://drive.google.com/file/d/1XmGj5Tse1OaRENXPBHbm1G0jrAvIjULp/view) | Met as prepared artifact |
| Submit on time; answer three instructor questions | No verified receipt or oral record | Pending |

## Task 2 — Frozen MobileNetV2 baseline (required)

| Instructor requirement | Evidence / result | Status |
| --- | --- | --- |
| RGB, 70/15/15 split; print image/class counts | [Final report](../results/Final_Report.md), [manifest](../results/manifest.json): 1260/270/270 | Met |
| ImageNet MobileNetV2 frozen base + binary head | [Script](../lab06.py), [baseline](../reports/baseline.md) | Met |
| Preprocess with `preprocess_input(mode="tf")`, binary generator/shuffle | Equivalent MobileNetV2 scaling built into model, alternative loader rather than literal specified generator | Difference |
| Adam 0.001; callbacks, best checkpoint | [Training record](../results/mobilenetv2/training.json), original checkpoint in private Drive | Met, see setting waiver |
| Epochs 50, patience 5, batch 32 (later sheet) | Actual **15/4/16**. Instructor verbally accepted settings; no rerun required on this basis | Accepted difference |
| Test accuracy, precision, recall, F1, confusion matrix | [Result JSON](../results/mobilenetv2_frozen/results.json), [baseline report](../reports/baseline.md) | Met |
| Training curves, interpretation and warm-up latency | [Curves](../results/mobilenetv2/curves.png), report; 20 warm-up + 100 timed | Met |
| Misclassified image copies in named errors folder | [Error index](../results/mobilenetv2_frozen/errors/README.md) links one authentic private Drive JPEG | Met as private linked evidence |
| Phase-schema `results.json` | [Results](../results/mobilenetv2_frozen/results.json) derived subsequently from saved measurements | Met as reconstructed record |
| End-to-end executed notebook/script and teacher submission | [Colab run log](../results/run.log) shows actual train/evaluate/report commands; original executed notebook itself unavailable | Partial verification; hand-in pending |

## Task 3 — YOLOv8 classification (required)

| Instructor requirement | Evidence / result | Status |
| --- | --- | --- |
| Same split/class names good and bad as Task 2 | [Protocol](../results/protocol.json), source-grouped manifest | Met |
| Two variants `yolov8n-cls` and `yolov8s-cls` | [Nano JSON](../results/week7_yolov8n_cls/results.json), [Small JSON](../results/week7_yolov8s_cls/results.json) | Met |
| Keep best weights; measure training time, size, parameters | Original private Drive checkpoints; [week7 report](../reports/week7_comparison.md) | Met |
| 640px, 50 epochs, batch32, patience5 | Actual 224px/15 max/batch16/patience4; instructor verbally approved historical setup | Accepted difference |
| Test accuracy, per-class/macro precision/recall/F1, confusion | Phase-schema JSON, stored plots and [comparison](../reports/week7_comparison.md) | Met |
| Latency 20 warm-up + 100 timed | [Original evaluation records](../results/yolov8n/evaluation.json) and counterpart | Met |
| Exact command lines, seed, scripts and Week7 tag | [Colab log](../results/run.log), [launcher](../scripts/train_yolo_quality.py), [tag verification](TAGGING.md) | Met with retrospective launcher distinction |
| Compare to Task2 baseline on same test | [Week7 comparison](../reports/week7_comparison.md) | Met |
| Original in-class defense / on-time receipt | Not independently documented | Pending |

## Task 4 — Third model, comparison and presentation (required)

| Instructor requirement | Evidence / result | Status |
| --- | --- | --- |
| Third architecture with correct preprocessing | EfficientNetB0: [JSON](../results/week8_efficientnetb0/results.json), [report](../results/Final_Report.md) | Met |
| Same split, parameters and phase schema | Source-grouped 70/15/15; actual 15/4/16 accepted; retrospective phase JSON | Met / accepted difference |
| Table from recorded JSON across ≥3 models; all metrics/costs | [Comparison table](../reports/comparison.md), [source-derived CSV](../results/comparison_from_json.csv) | Met |
| Accuracy vs latency chart; side-by-side confusion matrices | [Chart](../results/accuracy_vs_speed.png), [matrices](../results/confusion_matrices.png) | Met |
| ≥5 **misclassified** test images per model | Only **1/3/3/2 true errors** exist. [Private failure-gallery indexes](../results/mobilenetv2/errors/README.md), [YOLO n](../results/yolov8n/errors/README.md), and other models | **Literal count not met; impossible without inventing errors** |
| Identify hardest defect and mitigation | [Error analysis](../reports/comparison.md): glare dominant; [decision memo](../reports/decision_memo.md) | Met |
| Five-part decision: choice, evidence, alternatives, risks, deployment | [Canonical memo](../reports/decision_memo.md) | Met |
| Exactly four slides, five-minute talk, ≥2 instructor questions | [Four-slide deck](../presentation/Lab06_Presentation_4slides.pptx), [script](../presentation/Final_4_Slide_Script.md) | Slides ready; oral delivery pending |
| Organized repository, Week8 Git tag | [Tags and provenance](TAGGING.md), code/results/reports folders | Met; Week7/8 tags label same actual original Oct 25 commit |
| Formal submission / instructor acceptance | No receipt in checked materials | Pending |

## Bonus Task 5 — not required by the main Tasks 1–4 brief

| Bonus | Exact relevant requirement | Evidence and qualification |
| --- | --- | --- |
| **5A** | Unfreeze approximately 30 layers at lr 1e-5, same split, delta, curves and train/validation gap | [Real run](../reports/bonus_finetune_delta.md): no test improvement. Batch32 via gradient accumulation, BN layers frozen, PyTorch Keras backend |
| **5B** | Validation-only threshold grid, PR curve, cost memo, chosen threshold add-on | [Validation report](../reports/bonus_threshold_memo.md); threshold 0.10 remains exploratory |
| **5C** | Train-only glare augmentation; compare matched arms and per-defect deltas | [Real two-arm continuation](../reports/bonus_augmentation.md); no extra glare benefit; not ImageNet-from-scratch retraining |
| **5D** | ONNX/TFLite export, latency/size and accuracy check, deployment memo | [ONNX report](../reports/bonus_deployment_memo.md): zero 270-image accuracy drop, 2.49× forward-only speedup |
| **5E** | Peer experiment via README, concrete audit, own model card, own README reflection | [Peer audit](../reports/bonus_audit_note.md), [model card](../reports/model_card.md); exact peer inference succeeded, **original peer training not reproduced** and explicit publication permission not verified |

## Clarifications that remove false blockers

- **Dataset B reciprocal exchange:** discussed verbally but explicitly declared **unnecessary** by the instructor in the student's 9 November report. Therefore do **not** treat it as a required missing Task 5E dataset or a Task 1–4 submission item.
- **Original hyperparameters:** instructor verbally accepted Tasks 2–4's actual settings. The historical experimental records must still show the true values.
- **Synthetic-to-real performance:** not measured. This is a limitation of the evidence, **not** a teacher-assigned mandatory new test.
- **Peer original training:** absence is disclosed in optional Task 5E. The supplied original `predict.py` was actually executed and did not fail.
- **152-item checklist:** a historical mixed-scope audit containing written requirements, bonus items, verbal Dataset B ideas, administrative steps and file-format variants. It **must not** be interpreted as a count of missing mandatory course tasks.

## Archive integrity

- Original dataset images and checkpoints remain in private Drive.
- Historical `results.json` reconstructions are labeled as derived from original logs/evaluations.
- No five-error-per-model gallery was fabricated when fewer mistakes occurred.
- The [consolidated evidence ZIP](https://drive.google.com/file/d/1ZAp0JdWW_DruPzZrG9ll96Z-Idjr1-4D/view) is an immutable **older snapshot**, not an up-to-date copy of GitHub documentation after later corrections.
