# Lab 06 — Document Quality Classification

**Experiment:** 25 October 2026 · **Seed:** `2473986` · **Classes:** `good` / `bad`

This project compares four pretrained image classifiers for deciding whether a document photograph is usable or defective. It includes the training code, saved evaluation records, reports and presentation. Original images and trained checkpoints are stored separately in Google Drive.

## Results

The dataset contains **1,800 synthetic images from 300 source documents** (passport, INN and SNILS). Variants of the same source document stay in one split: **1,260 train / 270 validation / 270 test**.

| Model | Correct / 270 test images | Accuracy | CPU latency (ms/image) |
| --- | ---: | ---: | ---: |
| MobileNetV2 | 269 | 99.63% | 37.69 |
| EfficientNetB0 | 267 | 98.89% | 56.69 |
| YOLOv8n-cls | 267 | 98.89% | 26.97 |
| YOLOv8s-cls | 268 | 99.26% | 51.07 |

**MobileNetV2** has the highest observed test accuracy. **YOLOv8n-cls** is the selected model under the predefined validation-and-cost rule; its lower CPU latency and checkpoint size matter for the intended quality gate. The test set was used for final evaluation, not for selecting the winner. All observed test mistakes were defective images accepted as good.

These results describe a **source-disjoint synthetic test set**. Performance on independent real-world photos has not been measured. See the [full report](results/Final_Report.md) and the [selected model card](reports/model_card.md).

## View and verify the saved results

From the `LAB06_Model_Selection` folder:

```bash
python verify_results.py
```

This checks the recorded split, predictions, metrics, thresholds and selection decision **without retraining or downloading images**. It checks saved records, not the original image bytes or checkpoints.

Open [Review_Results.ipynb](Review_Results.ipynb) for the executed figures and analysis. [Lab06.ipynb](Lab06.ipynb) contains the training workflow but is **not** the original executed Colab session. The original run is documented through [run logs and results](results/).

## Re-run the experiment

### 1. Get the dataset

Download the [original Dataset A ZIP](https://drive.google.com/file/d/1QNHsbtpWcmeo0asG2Mrrruwy5w2lmobB/view) (Drive access may be required), then extract it. Select the directory with this structure:

```text
demo_quality_dataset/
├── metadata.jsonl
├── good/
└── bad/
```

Do not point the command to the ZIP or its parent directory. `metadata.jsonl` supplies the source IDs, image paths and class labels; the program checks the input files and source-grouped split. Images in this dataset are 640 × 640 pixels.

### 2. Prepare Python

Use **Python 3.11** for a straightforward local setup; Python 3.10 or 3.12 is also supported by the Windows launcher. Install the project dependencies in a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

**Original environment:** the saved Colab run used Python **3.13.16**, TensorFlow **2.21.0**, PyTorch **2.11.0+cu130** and Ultralytics **8.4.189**. See [environment.json](results/environment.json) and [environment.txt](results/environment.txt). The version ranges in `requirements.txt` support installation but are **not** a cross-platform exact lock; numerical results and timings can vary on a different system.

### 3. Train and evaluate

Run all four models with the same protocol, using a **new output folder**:

```bash
python lab06.py run --dataset "/path/to/demo_quality_dataset" --out "runs/reproduction_01"
```

Replace the dataset path with the extracted directory. Choose another output path for every independent experiment; **never train into the tracked `results/` folder**. The run prepares a source-grouped split, trains the four models, evaluates them and writes its own report.

**Windows shortcut:** run `RUN_WINDOWS.bat`. It opens a dataset folder picker, installs dependencies into `.venv` and stores each new run in its own date-stamped `runs/` directory. The resulting report path is printed at the end.

The historical experiment used **224 × 224 model inputs, batch size 16, maximum 15 epochs and early-stopping patience 4**; actual stopping epochs differ by model. These are the values in [config.json](config.json) and the saved measurements. They differ from a later written 640/32/50/5 prescription; the instructor has verbally accepted those differences. The recorded experiment has **not** been relabeled or retrained to match the later numbers.

## Files and reports

| File or folder | Contents |
| --- | --- |
| [lab06.py](lab06.py), [config.json](config.json) | Training, split, evaluation and fixed settings |
| [results/](results/) | Original measurements, predictions, logs and figures |
| [Final_Report.md](results/Final_Report.md) | Full methodology, results and limitations |
| [Review_Results.ipynb](Review_Results.ipynb) | Executed review of saved results; no new training |
| [presentation/](presentation/) | [Four-slide deck](presentation/Lab06_Presentation_4slides.pptx) and [speaking script](presentation/Final_4_Slide_Script.md) |
| [reports/](reports/) | Main task reports and optional experiments |
| [docs/](docs/) | Project map, audit and submission notes |

**Optional tasks:** [5A fine-tuning](reports/bonus_finetune_delta.md) · [5B validation threshold analysis](reports/bonus_threshold_memo.md) · [5C glare augmentation](reports/bonus_augmentation.md) · [5D ONNX benchmark](reports/bonus_deployment_memo.md) · [5E peer reproducibility audit](reports/bonus_audit_note.md) and [own model card](reports/model_card.md). The peer's original inference script was reproduced on our 270-image test set; their original training experiment and own test data were not available.

## Submission and scope

- [Master Drive folder](https://drive.google.com/drive/folders/105nt1H9wJsG-32jFrBHRqTDyB01USqta): original data, checkpoints and organized deliverables.
- [Consolidated review archive](https://drive.google.com/file/d/1ZAp0JdWW_DruPzZrG9ll96Z-Idjr1-4D/view): prepared for review; **not evidence of official submission or acceptance**.
- [Submission readiness](docs/FINAL_HANDIN_STATUS_2026-11-09.md) and [current checklist](docs/FULL_CHECKLIST_CURRENT_2026-11-09.md) document outstanding formal checks.

An external Dataset B exchange was discussed but is **not required by the instructor**. The original executed Colab training notebook and final instructor receipt/defense are not verified by this repository. The recorded data, checkpoints and test measurements have not been changed.
