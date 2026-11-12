# Industrial Surface Defect Classification — Project Index

**Course:** AI Implementation Technologies  
**Student:** YASSER IDBOUZKRI · Group иикб-24  
**Data:** KolektorSDD2  
**Experimental freeze:** **12 November 2026**  
**Status:** **Research and final deliverables completed for the adopted scope.** No additional training is scheduled or required by this project closeout; formal grading and submission requirements must be checked against the professor's rubric.

## 1. Canonical submission files

All final files are in [`results/`](results/). Older drafts are **not** the submission versions.

| Final artifact | Purpose |
| --- | --- |
| [`FINAL_ACADEMIC_REPORT_EN.pdf`](results/FINAL_ACADEMIC_REPORT_EN.pdf) | Printable four-page final academic report |
| [`FINAL_ACADEMIC_REPORT_EN.md`](results/FINAL_ACADEMIC_REPORT_EN.md) | Editable, versioned report text |
| [`NAZIH_FINAL_PRESENTATION.pptx`](results/NAZIH_FINAL_PRESENTATION.pptx) | Twelve-slide final presentation |
| [`FULL_EPOCH_RESULTS_SEED20261028.csv`](results/FULL_EPOCH_RESULTS_SEED20261028.csv) | Exactly eight authentic new 30-epoch results |
| [`FINAL_DELIVERABLE_SHA256.txt`](results/FINAL_DELIVERABLE_SHA256.txt) | Local SHA256 manifest for the five canonical artifacts |
| [`PROJECT_CLOSEOUT.md`](results/PROJECT_CLOSEOUT.md) | Research-scope closure, storage map, and verification limits |
| [`FULL_EPOCH_VALIDATION_F1.png`](results/FULL_EPOCH_VALIDATION_F1.png) | Follow-up internal validation chart |
| [`FIXED_BUDGET_REUSED_TEST_F1.png`](results/FIXED_BUDGET_REUSED_TEST_F1.png) | Original exploratory reused-test chart |

**Drive mirror:** [`Nazih_Coursework/03_Final_Deliverables`](https://drive.google.com/drive/folders/11_nJr8JfpFNFx5PonoLALaJg6Lns-YdE). Previous versions were moved **without deletion** to [`07_Archived_Prior_Drafts`](https://drive.google.com/drive/folders/1Bc-Jl_CTHIbxY3q2eb-_wKif9e_Git5s).

## 2. Scientific studies — do not merge their scores

**Study A — fixed 216 optimizer updates:** 24 evaluation records across four sizes (70, 140, 280, 420), two losses (Cross-Entropy, Poly-1), and three seeds. Six 70-image reference evaluations were restored unchanged; 18 new training/evaluation pairs were completed. The historical test set of 1,004 images had previously been inspected, so these are **exploratory**, not blind confirmation results. The 12 reference-equivalent training rounds correspond to 12 / 6 / 3 / 2 complete-pass equivalents for the four dataset sizes.

**Study B — 30 actual full dataset epochs:** **8/8 runs in the final adopted scope are complete**: 4 sizes × CE/Poly-1 × **seed 20261028**. Every run selects its checkpoint using the *same* 180-image holdout drawn from the official TRAIN partition, never from official TEST. At 420 images, both CE and Poly-1 achieve 98.00% **best internal validation Macro-F1**. Results across sizes have different optimization-step counts (540 / 1,050 / 2,100 / 3,150) and do not isolate image diversity from compute. The 24-run proposal (three seeds) was **not completed**: the other 16 proposed runs are future work, **not pending work for this submission**.

These two studies use different evaluation populations and training budgets. There is no legitimate direct numeric improvement claim from Study A's reused official test to Study B's training-only validation. No claim of production accuracy or specimen-independent blind testing is made.

## 3. Source, results, and audit layout

```text
COURSEWORK_NAZIH/
  README.md                         current index (this document)
  protocol.json                     frozen Study A protocol
  FULL_EPOCH_PROTOCOL.md            Study B protocol, historical design record
  scripts/                           training, evaluation, full-epoch pilot, reporting, audits
  input/                             original 70-image references and source/test manifests
  results/                           canonical final files, original evaluation JSONs and historical outputs
  data/full_epoch_pilot/            local-only/Drive-backed binary outputs (ignored by Git)
  EXTENDED_EPOCHS/                   archived alternative proposal; not part of final evidence
  02_PLAN.md, 03_ALL_CODE.ipynb      earlier M2AD development proposal, not the current KSDD2 final study
```

### Storage roles

- **GitHub:** code, protocols, lightweight metadata/results, final report, final slides, PDF and charts. **No** `.pt` weights or dataset ZIP.
- **Google Drive:** [`Nazih_Coursework`](https://drive.google.com/drive/folders/1OI3zrAtmA9ZfwH3l_DB3IiwFBK2P1_ny):
  - `Training_Size_Results/`: original 18 `.pt` weights + 18 training JSON records
  - `01_Historical_70/`: 12 genuine historical JSON files
  - `02_Evaluation_Results/`: 18 evaluation JSONs + three CSV summaries
  - `03_Final_Deliverables/`: canonical final report, PDF, PPTX, dataset tables, figures, manifest
  - `04_Integrity_And_Audits/`: local SHA256 manifest and local status
  - `05_Official_Dataset/`: original `KolektorSDD2.zip` (not redistributed in GitHub)
  - `06_Full_Epoch_Pilot/`: all eight completed full-epoch run directories, each with checkpoint and recorded history
  - `06_Extended_Epochs/`: unused empty proposal folder, excluded from completed work
  - `07_Archived_Prior_Drafts/`: previous report/presentation/figure versions, preserved
- **Windows working copy:** `C:\\Users\\YASSER\\Desktop\\nazih-project\\COURSEWORK_NAZIH`. Cloud changes do **not** automatically update local disk files unless local Git and Drive synchronization is performed.

To bring the **tracked** final artifacts to the Windows PC, from the existing repository use `git pull --ff-only origin nazih-training-set-size-20261109`. Do not reset/force-push this branch or alter `main`.

## 4. Integrity and limitations

The historical Windows audit matched 76/76 local project artifact SHA256 values and matched filenames/byte lengths for the corresponding Drive copies. This does **not** establish that every cloud object was independently SHA256-recomputed. The eight new follow-up `COMPLETE.json` histories and common validation splits were inspected from Drive; the two selected 420-image checkpoints were independently hash-verified. All scientific source-image hashes, model weights, historical raw evaluations and Git commit history must be preserved.

**Scope closed; no further model training is necessary for the agreed eight-run follow-up.** Any further seeds, independent specimen-separated blind test or deployment evaluation would constitute a new study. The formal professor submission has **not** been performed by this repository update.
