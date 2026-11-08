# LAB 06 - COMPLETE MASTER CHECKLIST
**Document Quality Classification | revised 9 November 2026**

**TOTAL:** 152 requirements / audit items - GREEN 85 | YELLOW 22 | RED 45

- 🟢 **GREEN:** completed with evidence
- 🟡 **YELLOW:** completed differently or partially; a note immediately beneath explains the deviation
- 🔴 **RED:** not completed, not confirmed, or not evidenced; unknowns are labeled explicitly

**Source distinction:** Tasks 1-4 and Task 5A-5E come from the instructor's written documents. Tasks 5A-5E are labeled *bonus* there. Dataset exchange is a **separate, reported verbal instruction**; its exact scope is not confirmed by a written brief.

**Key outcome (9 Nov):** Tasks 1–4 have real archived runs with disclosed protocol deviations; Task 5A ran on our original data (99.63% before/after). Task 1 Word memo, teacher-named YOLO training wrapper, JSON-derived comparison, model card and optional validation-only threshold audit have now been prepared. Classmate model was evaluated on our test, but **the classmate's images and their test of our model are still missing**. Official acceptance is unknown. The 8 November snapshot remains archived separately.

## Quick links
- [Master Drive folder](https://drive.google.com/drive/folders/105nt1H9wJsG-32jFrBHRqTDyB01USqta)
- [GitHub project](https://github.com/ID9999999999999/DL-Computer-vision/tree/lab06-model-selection-2473986-20261025/LAB06_Model_Selection)
- [Original instructor documents](https://drive.google.com/drive/folders/1Dz12WAZXeOE5366BCpjaO_cp-FaE-XHQ)
- [Original data and model weights](https://drive.google.com/drive/folders/1ja5uljMIcMSNLd41scx2HHV5B0SLallu)
- [Task 5A full folder](https://drive.google.com/drive/folders/1krJ2b7SXPcHBnkDQHajKBq5ZEC1fAYAP)

---

## 00 | General and project-wide requirements
*WRITTEN: Evaluation Guide + Requirements Tasks 2-4*  |  12 items: 🟢 8 / 🟡 1 / 🔴 3

- 🟢 **G01** - Use an individual GitHub repository.
- 🟢 **G02** - Use the student ID as the random seed.
  - *Evidence / note:* Seed 2473986 in the project records.
- 🟢 **G03** - Create a consistent source-grouped 70/15/15 split.
  - *Evidence / note:* 1,260 train / 270 validation / 270 test from 300 source documents.
- 🟢 **G04** - Avoid source/document leakage across train, validation and test.
  - *Evidence / note:* Original manifest checked; no overlapping source IDs.
- 🟢 **G05** - Use the test split for final evaluation, not for training or checkpoint selection.
  - *Evidence / note:* Recorded evaluation and Task 5A training use validation for selection.
- 🟢 **G06** - Report more than accuracy: macro and per-class precision, recall and F1.
- 🟢 **G07** - Generate confusion matrices and preserve measured CPU latency.
- 🟢 **G08** - Keep seed, environment, training logs and hyperparameters documented.
- 🟡 **G09** - Create the standardized phase-schema results.json for each model.
  - 🟡 **Difference:** Created later from original evaluation/training histories; not the file saved directly during the original training runs.
- 🔴 **G10** - Make the original training notebook run end-to-end without errors.
  - *Evidence / note:* Original executed Colab notebook is not in the checked GitHub package; independent full rerun not verified.
- 🔴 **G11** - Submit on time and receive instructor acknowledgment.
  - *Evidence / note:* No verified official hand-in/acceptance record.
- 🔴 **G12** - Explain and defend the pipeline, metrics and decisions orally.
  - *Evidence / note:* Instructor defense not confirmed.

## 01 | TASK 1: Model Selection Memo
*WRITTEN: Task 1 + Evaluation Guide*  |  11 items: 🟢 8 / 🟡 1 / 🔴 2

- 🟢 **T101** - Compare at least three CNN architectures with technical descriptions.
  - *Evidence / note:* Four model candidates documented.
- 🟢 **T102** - Include architecture family, native input size, approximate parameters, strengths and risks.
- 🟢 **T103** - Use at least three credible scientific or official references.
  - *Evidence / note:* Six references reported in the memo.
- 🟢 **T104** - Tie comparison to first quality-gate speed, accuracy, size, small synthetic data and CPU/server constraints.
- 🟢 **T105** - Explain transfer learning and frozen-base versus full fine-tuning.
- 🟢 **T106** - Pre-register numeric acceptance thresholds before training.
  - *Evidence / note:* Recorded in the dated selection memo.
- 🟢 **T107** - Specify a shortlist of architectures to train in later tasks.
- 🟢 **T108** - Provide a clearly structured memo and comparison table in the repository.
  - *Evidence / note:* docs/Model_Selection_Memo.md.
- 🟡 **T109** - Provide the report in the requested Word/personal-folder format.
  - 🟡 **Difference:** The verified master artifact is a GitHub Markdown memo, Word-formatted memo now at https://drive.google.com/file/d/1bbrB9GnlR-xrOtEAjCMAp_h5EUQbSOJC/view but original deadline and teacher receipt remain unverified.
- 🔴 **T110** - Submit before the class deadline.
  - *Evidence / note:* Submission timestamp/teacher receipt not verified.
- 🔴 **T111** - Answer three defense questions: pipeline latency, frozen regime and synthetic-to-real gap.
  - *Evidence / note:* Oral answers before instructor not verified.

## 02 | TASK 2: MobileNetV2 Frozen Baseline
*WRITTEN: Requirements Tasks 2-4*  |  18 items: 🟢 12 / 🟡 4 / 🔴 2

- 🟢 **T201** - Use RGB 224x224 images and a binary good/bad label mapping.
- 🟢 **T202** - Reuse the original 70/15/15 split with source grouping.
- 🟡 **T203** - Use course generator setup with preprocess_input(mode="tf"), binary class mode and specified shuffle.
  - 🟡 **Difference:** Existing trained pipeline uses normalization embedded in the Keras model and an alternative loader, rather than exactly the prescribed generator call.
- 🟢 **T204** - Load ImageNet-pretrained MobileNetV2 without the original classification top.
- 🟢 **T205** - Freeze the MobileNetV2 base and train the custom binary head.
- 🟢 **T206** - Use Adam learning rate 0.001 and binary cross-entropy.
- 🟡 **T207** - Use max 50 epochs, batch 32 and patience 5.
  - 🟡 **Difference:** Original training used max 15 epochs, batch 16 and patience 4.
- 🟢 **T208** - Use EarlyStopping and save the best checkpoint.
  - *Evidence / note:* Original best.keras is in the model folder in Drive.
- 🟢 **T209** - Save train and validation accuracy/loss curves.
- 🟢 **T210** - Evaluate only on held-out test: accuracy, macro/per-class P/R/F1.
  - *Evidence / note:* Test accuracy 99.63% (269/270).
- 🟢 **T211** - Save a test confusion matrix image.
  - *Evidence / note:* Saved confusion-matrix figures.
- 🟡 **T212** - Store individual false predictions in results/mobilenetv2_frozen/errors/.
  - 🟡 **Difference:** One true error JPEG copied to a private Drive archive and linked from GitHub error indexes; not stored as a public JPEG in that repository path.
- 🟢 **T213** - Measure CPU latency using 20 warm-up predictions and 100 timed predictions.
  - *Evidence / note:* Archived protocol recorded.
- 🟢 **T214** - Record checkpoint size and parameter count.
- 🟡 **T215** - Save results/mobilenetv2_frozen/results.json with exact phase schema.
  - 🟡 **Difference:** Compatibility JSON added retrospectively from real measurements; original run used evaluation.json/training.json.
- 🟢 **T216** - Write 5-10 lines interpreting loss curves and over/underfitting.
  - *Evidence / note:* reports/baseline.md.
- 🔴 **T217** - Include an executed end-to-end notebook/script as the formal deliverable.
  - *Evidence / note:* A reproducible workflow is present, but original executed Colab training notebook was not independently recovered/verified.
- 🔴 **T218** - Verify teacher acceptance of Task 2.
  - *Evidence / note:* No acceptance record confirmed.

## 03 | TASK 3: YOLOv8 Classification
*WRITTEN: Requirements Tasks 2-4*  |  16 items: 🟢 11 / 🟡 4 / 🔴 1

- 🟢 **T301** - Use same source-grouped dataset as Task 2; no new random re-split.
- 🟢 **T302** - Use train/validation/test with exact good and bad class folders.
  - *Evidence / note:* Original yolo_data Drive folder has train/val/test good/bad classes; naming "val" is used operationally.
- 🟢 **T303** - Train at least two variants: YOLOv8n-cls and YOLOv8s-cls.
- 🟢 **T304** - Preserve best.pt for each YOLO variant.
  - *Evidence / note:* Saved in Drive model weights folders.
- 🟡 **T305** - Train with max 50 epochs, patience 5, imgsz 640, batch 32.
  - 🟡 **Difference:** Both original YOLO runs used 15/4/224/16; 11 epochs for n and 7 for s actually executed.
- 🟢 **T306** - Use student ID as seed and record actual epochs.
  - *Evidence / note:* Seed 2473986; 11 and 7 epochs.
- 🟢 **T307** - Evaluate test top-1 accuracy correctly as fraction, not percent.
- 🟢 **T308** - Report macro and per-class P/R/F1 and confusion matrices.
- 🟢 **T309** - Measure training seconds, parameter count and best.pt size.
- 🟢 **T310** - Record 20 warm-up + 100 timed CPU predictions.
- 🟢 **T311** - Preserve Ultralytics curves, confusion plots and training artifacts.
- 🟡 **T312** - Save one valid phase-schema results.json for each model.
  - 🟡 **Difference:** Created retrospectively under results/week7_yolov8n_cls and results/week7_yolov8s_cls from archived measurements.
- 🟡 **T313** - Provide exact training commands and reproducible scripts.
  - 🟡 **Difference:** Config, args.yaml and runnable workflow are retained, but exact required scripts/train_yolo_quality.py path was not used in original run.
- 🟢 **T314** - Write a 5+ sentence comparison against own MobileNetV2 on same test split.
  - *Evidence / note:* reports/week7_comparison.md includes accuracy, latency, size and regime differences.
- 🟡 **T315** - Record a Week 7 repository commit and a Week 7 Git tag.
  - 🟡 **Difference:** Branch/commits exist, but the teacher-required Git tag has not been created/verified.
- 🔴 **T316** - Complete instructor defense / on-time Task 3 submission.
  - *Evidence / note:* Not confirmed.

## 04 | TASK 4: Third Architecture, Comparison, Decision and Presentation
*WRITTEN: Requirements Tasks 2-4*  |  21 items: 🟢 14 / 🟡 4 / 🔴 3

- 🟢 **T401** - Train third architecture (EfficientNetB0, ResNet50, MobileNetV3 or InceptionV3).
  - *Evidence / note:* EfficientNetB0 trained.
- 🟢 **T402** - Use family-correct input processing on the same source-grouped split.
  - *Evidence / note:* EfficientNetB0 was trained at 224x224 with same split.
- 🟡 **T403** - Use max 50 epochs, patience 5 and batch 32.
  - 🟡 **Difference:** Original EfficientNetB0 used max 15, patience 4 and batch 16.
- 🟢 **T404** - Save third-model checkpoint and complete test evaluation.
- 🟡 **T405** - Save a phase-schema third-model results.json.
  - 🟡 **Difference:** Generated later from archived results and stored under a compatibility path.
- 🟢 **T406** - Compare at least 3 architectures (including MobileNetV2 and YOLO).
  - *Evidence / note:* Four model configurations compared.
- 🟢 **T407** - Include accuracy, macro P/R/F1, latency, size, parameters, training time and actual epochs.
  - *Evidence / note:* New results/comparison_with_epochs.csv includes all fields.
- 🟢 **T408** - Generate final table exclusively from results.json source files.
  - 🟡 **Difference:** The corrected summary and JSON files are consistent, but the original comparison came from saved evaluation files rather than originally logged phase-schema JSON.
- 🟢 **T409** - Provide accuracy-vs-latency (or size) chart.
- 🟢 **T410** - Provide side-by-side confusion matrices.
- 🟢 **T411** - Provide a failure gallery with true and predicted labels.
  - *Evidence / note:* Real error images and indexes were retained in Drive/GitHub.
- 🟡 **T412** - Provide at least 5 distinct misclassified images per model.
  - 🟡 **Difference:** Only 1, 3, 3 and 2 actual test errors; showing 5 would fabricate evidence, so authentic smaller counts are disclosed.
- 🟢 **T413** - Identify dominant/hardest defect and give mitigation ideas.
  - *Evidence / note:* Glare identified as common false-accept cause.
- 🟢 **T414** - Provide a five-part decision memo: decision, evidence, rejected alternatives, risk/mitigation, deployment.
  - *Evidence / note:* reports/week8_decision_memo.md.
- 🟢 **T415** - Discuss CPU server now versus later TFLite/ONNX edge deployment.
  - *Evidence / note:* Documented as plan, not falsely reported as executed export.
- 🟢 **T416** - Recommend one model and justify selection using measured gates.
  - *Evidence / note:* YOLOv8n provisionally selected for measured speed.
- 🟢 **T417** - Provide presentation with exactly four slides.
  - *Evidence / note:* presentation/Lab06_Presentation_4slides.pptx.
- 🔴 **T418** - Deliver 5-minute in-class presentation and answer at least two questions.
  - *Evidence / note:* No verified instructor presentation/defense.
- 🟡 **T419** - Organize scripts/results/reports and document failed attempts.
  - 🟡 **Difference:** Compatibility folders and audit notes created; exact instructor script names and full chronological failed-attempt logging not independently verified.
- 🔴 **T420** - Create required Task 4 repository tag.
  - *Evidence / note:* No tag created.
- 🔴 **T421** - Confirm formal Task 4 submission and acceptance.
  - *Evidence / note:* No acceptance record.

## 05 | TASK 5A: Partial Fine-Tuning (Bonus in written sheet)
*WRITTEN BONUS: Task 5A; completed original-dataset experiment*  |  18 items: 🟢 14 / 🟡 3 / 🔴 1

- 🟢 **A01** - Select an already trained frozen checkpoint from Task 2 or Task 4.
  - *Evidence / note:* Original frozen MobileNetV2 best.keras was loaded and checksum verified.
- 🟢 **A02** - Use original dataset split and student seed.
  - *Evidence / note:* Verified same 1,800 files and seed 2473986; no overlap.
- 🟡 **A03** - Unfreeze only the final approximately 30 backbone layers.
  - 🟡 **Difference:** Final 30 named layers selected, but BatchNormalization layers within these 30 were intentionally left frozen.
- 🟢 **A04** - Use learning rate 1e-5.
- 🟢 **A05** - Use max epochs 50 and early stopping patience 5.
  - *Evidence / note:* 20 actual epochs; best checkpoint from epoch 15.
- 🟡 **A06** - Use batch size 32.
  - 🟡 **Difference:** Effective batch 32 via four accumulated physical microbatches of 8, due to the execution memory limit.
- 🟡 **A07** - Run using the specified Keras/TensorFlow execution environment.
  - 🟡 **Difference:** Real training executed with Keras 3 PyTorch CPU backend; original baseline used TensorFlow/T4 GPU.
- 🟢 **A08** - Run actual gradient updates and store best checkpoint.
  - *Evidence / note:* best.keras in Task5A Drive results; weights verified changed in trainable region.
- 🟢 **A09** - Save train/validation loss and accuracy curves.
- 🟢 **A10** - Measure test accuracy and macro precision/recall/F1.
  - *Evidence / note:* Test accuracy 99.63%; macro F1 0.9963.
- 🟢 **A11** - Generate test confusion matrix.
  - *Evidence / note:* [[135,0],[1,134]].
- 🟢 **A12** - Preserve genuine individual test errors / failure gallery.
  - *Evidence / note:* One real glare false accept stored, with index.
- 🟢 **A13** - Record CPU latency and model/checkpoint size.
  - *Evidence / note:* 64.16 ms frozen vs 69.40 ms fine-tuned on same new backend/host.
- 🟢 **A14** - Create results.json with regime="finetune_top30".
- 🟢 **A15** - Create frozen vs fine-tuned delta table.
  - *Evidence / note:* Accuracy and macro F1 differences both zero.
- 🟢 **A16** - Explain train-validation gap, overfitting risk and deployment recommendation in five-line memo.
  - *Evidence / note:* Recommendation: retain frozen baseline on measured evidence.
- 🟢 **A17** - Preserve reproducible script, notebook, log and full package.
  - *Evidence / note:* GitHub code and reports; Drive ZIP + fine-tuned checkpoint.
- 🔴 **A18** - Complete Task 5A on a classmate dataset.
  - *Evidence / note:* Not a requirement of the original 5A sheet; a SEPARATE reported verbal instruction remains pending.

## 06 | TASK 5B: Threshold and Cost Analysis
*WRITTEN BONUS: Task 5B; validation-only sweep and chart prepared*  |  8 items: 🟢 6 / 🟡 2 / 🔴 0

*9 Nov evidence:* [threshold sweep](../results/bonus_threshold_yolov8n/threshold_sweep.csv), [precision-recall curve](../results/bonus_threshold_yolov8n/precision_recall_curve.svg), [memo](../reports/bonus_threshold_memo.md). Validation candidate 0.10 (exploratory) versus historical default 0.50. **No test data used to choose it.** The report is staged in GitHub, not confirmed delivered to the instructor.

- 🟢 **B01** - Collect probabilities on VALIDATION only; do not tune on test.
- 🟢 **B02** - Sweep thresholds 0.10 to 0.90 in steps of 0.05.
- 🟢 **B03** - Compute bad-class precision/recall/F1 at each threshold.
- 🟢 **B04** - Plot precision-recall curve with default 0.5 and selected point.
- 🟢 **B05** - Explain false-accept versus false-reject costs in this pipeline.
- 🟢 **B06** - Compare expected validation error counts against 0.5.
- 🟡 **B07** - Save chosen_threshold into a results row and write half-page memo.
  - 🟡 **Difference:** threshold and explanation are written to an **add-on bonus result**; historical base results and deployed threshold 0.50 were not changed.
- 🟡 **B08** - Submit sweep table, curve and reports/bonus_threshold_memo.md.
  - 🟡 **Difference:** All files are prepared in GitHub, but no teacher receipt or grade is available.

## 07 | TASK 5C: Error-Driven Augmentation
*WRITTEN BONUS: Task 5C; not performed*  |  8 items: 🟢 1 / 🟡 0 / 🔴 7

- 🟢 **C01** - Identify the dominant actual failure (glare).
  - *Evidence / note:* Already identified in Task 4; this alone does not complete Task 5C.
- 🔴 **C02** - Create stronger train-only glare and/or motion-blur augmentations.
- 🔴 **C03** - Keep validation and test data unchanged.
  - *Evidence / note:* No Task 5C augmentation experiment has yet been run.
- 🔴 **C04** - Retrain best candidate with same seed and fixed protocol.
- 🔴 **C05** - Store augmentation configuration in results.json.
- 🔴 **C06** - Compare per-defect F1 before and after augmentation.
- 🔴 **C07** - Explain whether another defect class was harmed.
- 🔴 **C08** - Submit augmentation code, new results row and bonus_augmentation.md.

## 08 | TASK 5D: Deployment Export and Edge Benchmark
*WRITTEN BONUS: Task 5D; not performed*  |  7 items: 🟢 0 / 🟡 0 / 🔴 7

- 🔴 **D01** - Export selected model to TFLite or ONNX.
- 🔴 **D02** - Optionally quantize and note quantization risks.
- 🔴 **D03** - Measure model sizes before/after export.
- 🔴 **D04** - Benchmark CPU with 20 warm-up + 100 timed runs.
  - *Evidence / note:* Earlier native-model timing exists, but no exported model benchmark exists.
- 🔴 **D05** - Compare accuracy before/after export or state measured zero drop.
- 🔴 **D06** - Write justified server-versus-edge deployment memo.
  - *Evidence / note:* Task 4 gives a conceptual plan; Task 5D-specific measured memo is not completed.
- 🔴 **D07** - Deliver exported files, benchmark table, addon metrics and bonus_deployment_memo.md.

## 09 | TASK 5E: Peer Reproducibility Audit and Model Card
*WRITTEN BONUS: Task 5E; model card finished, peer audit partial*  |  8 items: 🟢 4 / 🟡 1 / 🔴 3

*9 Nov evidence:* [model card](../reports/model_card.md), [audit note](../reports/bonus_audit_note.md). The peer checkpoint was evaluated using a compatibility adapter, **not** a literal installation-and-rerun of its supplied README, so the strict E02 remains red.

- 🔴 **E01** - Get classmate permission and their README/experiment.
- 🔴 **E02** - Rerun one classmate experiment using their README only.
- 🟢 **E03** - Record concrete reproduction gaps and assumptions (seed, split, schema, versions).
- 🟡 **E04** - Write reports/bonus_audit_note.md.
- 🟢 **E05** - Write a model card for own chosen model: intended use and data.
- 🟢 **E06** - Include metrics, latency, size and generalization limitations in model card.
- 🟢 **E07** - Include ethics/privacy and synthetic-to-real gap in model card.
- 🔴 **E08** - Submit reports/model_card.md and explain audit findings.

## 10 | Dataset Exchange and New Classmate Fine-Tuning
*REPORTED VERBAL INSTRUCTION: scope not confirmed in writing*  |  14 items: 🟢 1 / 🟡 0 / 🔴 13

- 🟢 **X01** - Locate and preserve our Dataset A and its split/manifest.
  - *Evidence / note:* Root ZIP and yolo_data directories located in Drive.
- 🔴 **X02** - Confirm classmate identity and permissible private sharing.
- 🔴 **X03** - Share Dataset A copy with classmate, without unrelated results or private documents.
- 🔴 **X04** - Receive Dataset B from classmate.
- 🔴 **X05** - Validate Dataset B class definitions, counts, file integrity and licensing/privacy.
- 🔴 **X06** - Check source overlap between Dataset A and Dataset B.
- 🔴 **X07** - Create source-grouped train/validation/test for B without leakage.
- 🔴 **X08** - Measure existing model on an untouched Dataset B holdout BEFORE adaptation.
  - *Evidence / note:* Proposed scientifically controlled step; not explicitly confirmed by teacher.
- 🔴 **X09** - Fine-tune agreed model/checkpoint using Dataset B training partition.
- 🔴 **X10** - Choose checkpoint only with Dataset B validation partition.
- 🔴 **X11** - Evaluate post-adaptation on held-out B test split.
- 🔴 **X12** - Compare before/after metrics and domain-transfer limitations.
- 🔴 **X13** - Document experiment, new checkpoint and GitHub/Drive addendum.
- 🔴 **X14** - Confirm with teacher whether B-exchange replaces or supplements Task 5A.
  - *Evidence / note:* Current plan treats it as separate because written Task 5A explicitly uses our own data.

## 11 | Project Packaging, Submission and Defense
*PROJECT ADMINISTRATION; written + observed requirements*  |  11 items: 🟢 6 / 🟡 2 / 🔴 3

- 🟢 **S01** - Keep all original measurement records and checkpoints unchanged.
  - *Evidence / note:* Original runs retained; later corrections are additive.
- 🟢 **S02** - Provide corrected standardized reports and model comparisons in GitHub.
- 🟢 **S03** - Make corrected four-slide presentation available in repo and Drive.
- 🟢 **S04** - Organize a single navigable Drive master folder 00-09.
  - *Evidence / note:* LAB06_MASTER_2026-11-08.
- 🟢 **S05** - Make README and master map link GitHub, dataset and heavy checkpoints.
  - *Evidence / note:* docs/MASTER_PROJECT_MAP.md and Drive START_HERE index.
- 🟢 **S06** - Provide Task 5A final ZIP + weights in Google Drive.
- 🔴 **S07** - Create verified Week 7/Week 8 Git tags.
  - *Evidence / note:* Not created: requires identifying historical commits and tag write capability.
- 🟡 **S08** - Confirm that each deliverable matches exact teacher folder paths/names.
  - 🟡 **Difference:** Most compatibility paths and reports were added; physical weights remain in Drive by design, and some original script paths differ.
- 🟡 **S09** - Prepare a clean final hand-in bundle once teacher clarifies exchange scope.
  - 🟡 **Difference:** Curated Drive staging ZIP now exists (https://drive.google.com/file/d/1dvrnRlEyXgiUDFyVZraOaWU-iyNSlX3o/view). Peer-side test and official teacher submission remain outstanding.
  - *Evidence / note:* 08_FINAL_SUBMISSION is a staging folder, not proof of completed hand-in.
- 🔴 **S10** - Present 5-minute four-slide defense and answer 2+ instructor questions.
  - *Evidence / note:* Not verified.
- 🔴 **S11** - Record instructor acceptance and any correction requests.
  - *Evidence / note:* Not verified.

---
## Outstanding blockers / actions (not a claim of mandatory bonus work)

1. Finish reciprocal peer test: our model on classmate independent Dataset B; optionally obtain original Dataset B images and confirm whether adaptation is actually required.
2. Run the new Dataset B experiment only after verifying its labels, split, source overlap and permissions.
3. Create historical Git tags only after checking actual original commits (current connector cannot create tag refs).
4. Verify the instructor presentation, questions and official hand-in/acceptance.
5. Treat Tasks 5B-5E as optional bonuses unless the instructor explicitly assigns them.

**Historical protocol deviations not erased:** Tasks 2-4: teacher specifies 50 max epochs / patience 5 / batch 32 / YOLO 640; our original runs used 15 / 4 / 16 / 224. Task 5A separately used max 50, patience 5, effective batch 32 via 4×8, Keras-PyTorch CPU backend, and the final 30 layer region with BN frozen.


---
## Update log: 9 November 2026 (additive; no original training edits)

- Task 1: Word memo prepared, but not proven historically submitted.
- Task 3: newly authored exact-name training launcher; original YOLO checkpoints remain 224/16/15/4, and no 640-pixel training is claimed.
- Task 4: direct-from-results.json generator and four-model comparison now generated; no historical result values changed.
- Optional Task 5B: 17 validation-only threshold points, CSV and precision–recall chart. Candidate threshold 0.10 is exploratory; historical 0.5 unchanged.
- Optional Task 5E: model card finished; peer audit note prepared but full README-only rerun still unverified.
- Submission: new staging ZIP and evidence map, not official submission.
- Peer exchange: 270-image test of the classmate checkpoint on our data completed; no peer Dataset B or reciprocal result received.
- Task 5C and 5D remain optional and not performed.