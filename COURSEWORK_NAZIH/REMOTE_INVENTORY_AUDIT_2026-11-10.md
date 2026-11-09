# Cross-platform remote inventory audit — 10 November 2026

Repository: `ID9999999999999/DL-Computer-vision`
Branch: `nazih-training-set-size-20261109`
User-committed scientific outputs: `62f565df51447a3e553982bc877c6f96c86b469d` (41 files)
Local working directory last reported: `%USERPROFILE%\\Desktop\\nazih-project\\COURSEWORK_NAZIH`

## Remote inventory observed directly (Google Drive)

Source folder: https://drive.google.com/drive/folders/1OI3zrAtmA9ZfwH3l_DB3IiwFBK2P1_ny

| Folder | Inventory | Count | Matched in local SHA256 manifest by filename and byte length |
| --- | --- | ---: | ---: |
| `Training_Size_Results` | 18 PyTorch `.pt` checkpoints + 18 training `.json` | 36 | 36 |
| `02_Evaluation_Results` | 18 evaluation `.json` + three `.csv` summaries | 21 | 21 |
| `01_Historical_70/baseline_train` | historical training `.json` | 6 | 6 |
| `01_Historical_70/baseline_eval` | historical evaluation `.json` | 6 | 6 |
| `03_Final_Deliverables` | `REPORT_EN.md`, `FINAL_ACADEMIC_REPORT_EN.md`, `NAZIH_FINAL_PRESENTATION.pptx` | 3 | 3 |
| `03_Final_Deliverables/figures` | four PNG charts | 4 | 4 |
| `04_Integrity_And_Audits` | `LOCAL_INTEGRITY_MANIFEST.csv` + `LOCAL_WORKSPACE_STATUS.md` | 2 | Not included in their own source manifest |
| `05_Official_Dataset` | `KolektorSDD2.zip` | 1 | Not included in source manifest |
| **TOTAL** | **Stored files on Drive** | **79** | **76/76 eligible** |

No missing names or mismatching byte lengths were found for the 76 files described in `LOCAL_INTEGRITY_MANIFEST.csv` on Drive.

Dataset metadata: `KolektorSDD2.zip`, uploaded size **853,126,555 bytes**; the user independently verified the **local** ZIP SHA256 `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8` before upload.

## Verification limits

- The audit establishes **identity of filenames and byte lengths**, not remote SHA256 equality. Google Drive returned no usable hash field via the available metadata interface. Do not claim the Drive ZIP or 76 uploaded files have had their remote SHA256 recomputed.
- Earlier local `organize_workspace.py` verified all 18 model/training pairs, 18 evaluation reports, six historical pairs and the *local* dataset SHA256. The 18 checkpoint/report pairs were also independently SHA256-checked when backed up from Colab.
- GitHub commit `62f565df51447a3e553982bc877c6f96c86b469d` was observed in the repository with 41 files. Avoid `main` changes and avoid uploading `.pt` or ZIP files to GitHub.
- `FINAL_ACADEMIC_REPORT_EN.md` and `NAZIH_FINAL_PRESENTATION.pptx` are working drafts; existence is not proof of professor-instruction compliance or visual review.
- All 24 evaluation records are **exploratory**, because the official 1,004-image test set was previously examined. Do not present it as an untouched independent confirmation.

## Current next actions

1. Re-run the local SHA256 manifest validation and check the Git working tree branch and status after pulling this audit.
2. Optionally download a sample of the Drive-uploaded files and compare hashes locally; for absolute byte-level Drive parity, rehash **all** downloads (including the 853 MB dataset and 18 checkpoints).
3. Review the final academic report and slide deck visually against actual professor requirements before submission.
4. Keep duplicate ZIPs until all required backups and provenance records have been confirmed. Do not delete historical material as part of this audit.
