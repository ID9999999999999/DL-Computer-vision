# COURSEWORK_NAZIH — CANONICAL PROJECT INDEX

**Status as of 10 November 2026:** The user reported 18/18 new training runs and 18/18 new evaluations completed. Six earlier 70-image training/evaluation records were restored locally. The cross-condition report was generated locally from 24/24 evaluations. The GitHub branch and Drive have **not yet been fully synchronized** with those new local outputs. Do not treat the older README_EN.md status as current.

## One project; three storage roles

| Location | Authoritative role | Never do |
| --- | --- | --- |
| **GitHub**, existing repository/branch, `COURSEWORK_NAZIH/` | source code, protocol, verified manifests, historic JSON records, lightweight evaluations and summary tables, reproducible reports/figures | do not commit model `.pt` weights, full `.zip` datasets, temporary exports or secrets |
| **Google Drive**, `Nazih_Coursework/` | off-device backup of 18 model checkpoints, training metadata, official data archives (if uploaded), historical records, evaluation records and submission artifacts | never overwrite a different file without hash comparison |
| **Windows PC**, `%USERPROFILE%\Desktop\nazih-project\COURSEWORK_NAZIH\` | active working tree; produces evaluation, figures, report and slides | do not use `/content/` Colab paths or store the only copy of a checkpoint here |

The original GitHub branch is `nazih-training-set-size-20261109` in the existing repository `ID9999999999999/DL-Computer-vision`. Leave `main` unchanged unless a later deliberate merge is reviewed.

## Canonical Google Drive hierarchy

```text
Nazih_Coursework/
  Training_Size_Results/       # EXISTING; 18 .pt + 18 .json; retained in place
  01_Historical_70/            # 12 original JSON records; not yet uploaded
  02_Evaluation_Results/       # 18 eval JSON + validated summary CSV; not yet uploaded
  03_Final_Deliverables/       # report, figures, slides; not yet uploaded
  04_Integrity_And_Audits/     # generated local checksums and status; not yet uploaded
  05_Official_Dataset/        # verified original dataset ZIP if space permits; not yet uploaded
```

Folder existence alone is not proof of content synchronization.

## Canonical PC inputs / outputs

- `input/baseline_train/target_train_<seed>_<loss>.json`: 6 historical training records (from the genuine audit ZIP).
- `input/baseline_eval/eval_<seed>_<loss>_epoch12.json`: 6 historical evaluation records.
- `results/size<size>_seed<seed>_<loss>.pt` and `.json`: 18 new checkpoints and training reports; backed up already in Drive.
- `results/eval_size<size>_seed<seed>_<loss>.json`: 18 newly evaluated predictions/metrics. **Exploratory, not independent test claims**.
- `results/observed_results.csv`, `REPORT_EN.md`, `FINAL_ACADEMIC_REPORT_EN.md`, `figures/`, `NAZIH_FINAL_PRESENTATION.pptx`: derived outputs. Existence and final submission readiness must be audited locally.
- `%USERPROFILE%\Downloads\KolektorSDD2.zip`: official source archive, SHA256 `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`; never put in GitHub.
- `%USERPROFILE%\Downloads\Nazih_Execution_Audit_2026-11-08.zip`: provenance archive of historical 70-image work. Preserve without editing.

## Exact local audit and staging workflow

1. In VS Code, open `Desktop/nazih-project`; keep any local edits such as the UTF-8 fix to `scripts/report.py`.
2. Pull only this existing branch (the three organizational files created by ChatGPT are additive):
   `git pull --ff-only origin nazih-training-set-size-20261109`
3. Run `python COURSEWORK_NAZIH/scripts/organize_workspace.py`.
4. The script checks the 18 checkpoint/report SHA256 pairs, 18 matching evaluation/checkpoint IDs, six historical pairs, the official dataset SHA256 if present, and reports any missing deliverables.
5. It prepares `COURSEWORK_NAZIH/_SYNC_TO_DRIVE/` subfolders corresponding to the new Drive folders; these are *staging copies*, not automatic uploads. Upload their **contents** into the correspondingly named Drive folders.
6. After a local `git status` review, stage only verified lightweight results and local UTF-8 fix, and commit on the existing branch. Never use `git add -f` for ignored weights.
7. Validate Drive and GitHub copies by filenames, counts, SHA256 and report contents. Only then discuss moving/deleting old duplicate ZIPs.

The script **never deletes or overwrites a conflicting source**, and never retrains models.

## Verified exploratory aggregate results

| Unique training images | CE Macro-F1 | Poly-1 Macro-F1 |
| ---: | ---: | ---: |
| 70 | 76.64% | 75.72% |
| 140 | 77.86% | 79.24% |
| 280 | 84.76% | 86.10% |
| 420 | 84.80% | 86.07% |

Three paired seeds per size. The 70-image historical runs were not reproduced in the new environment. The official test images (1,004) were previously inspected; these results are **exploratory and not a blind independent confirmation**. Physical-part independence across splits remains unverified.

## Submission gate

Complete only when: all 24 evaluations checked; lightweight results committed to GitHub; 18 checkpoint pairs still backed up on Drive; evaluation/historical/deliverable/audit files independently verified in Drive; final report and PPTX visually reviewed; professor requirements independently checked. No success should be asserted merely because a script prints a path.
