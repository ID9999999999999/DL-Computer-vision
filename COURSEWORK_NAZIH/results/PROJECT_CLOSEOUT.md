# Project closeout — 12 November 2026

**Final research scope:** Initial fixed-216-update study plus eight genuine 30-full-epoch follow-up runs (sizes 70/140/280/420, losses CE/Poly-1, seed 20261028). The remaining 16 runs of a larger proposed multi-seed experiment were **not performed** and are treated as future work, not required pending work.

## Authoritative locations

- **GitHub:** branch `nazih-training-set-size-20261109` in `ID9999999999999/DL-Computer-vision`, project subfolder `COURSEWORK_NAZIH/`. Code, original results and evaluation JSONs, protocols, integrity audits and final small deliverables are version-controlled here.
- **Google Drive:** `Nazih_Coursework/`, with `Training_Size_Results/` (18 original checkpoints, 18 training reports); `01_Historical_70/` (12 historical JSONs); `02_Evaluation_Results/` (18 evaluation JSONs + 3 CSVs); `03_Final_Deliverables/` (canonical final report, slide deck, tables and PDF); `04_Integrity_And_Audits/` (local audit metadata); `05_Official_Dataset/` (official ZIP); and `06_Full_Epoch_Pilot/` (eight actual new full-epoch runs). The separate, currently empty `06_Extended_Epochs/` is an archival alternative proposal and **not part of the final evidence**.
- **Windows working copy:** `C:\Users\YASSER\Desktop\nazih-project\COURSEWORK_NAZIH`. This is a user-maintained mirror and should be updated with `git pull --ff-only`; cloud-side operations cannot independently establish the Windows disk's final state.

## Scientific separation

- **Study A:** 24 historical/new evaluation records, 3 seeds, 216 updates per run; all mean metrics were computed on an already-inspected 1,004-image official test set. This is exploratory, not fresh blind test evidence.
- **Study B:** Eight new 30-complete-epoch runs, one seed, matched CE vs Poly-1 within size, common train-only holdout of 180 images; at 420 images both reached 98.00% best internal validation Macro-F1. This does not provide an independent production accuracy claim.
- The two studies have different evaluation populations and computational budgets and must never be pooled as equivalent results.

## Naming and retention

- `results/FINAL_ACADEMIC_REPORT_EN.md` and `results/NAZIH_FINAL_PRESENTATION.pptx` are the **canonical final** written deliverables as of 12 November 2026.
- The previous versions dated 10 November are preserved in a separate Drive archive, not silently discarded.
- No model `.pt` or full dataset `.zip` is committed to GitHub. Original evidence, SHA256 records, checkpoints, source archive, raw logs and git history are retained; do not delete them to change appearance or authorship.
- Legacy README files from 8 November are historical snapshots; `README.md` is the current project index.

## File integrity and limits

- First study: 76 source project artifacts were verified locally by SHA256 in the user's Windows runtime, and 76 names/sizes were checked against Google Drive at the time. This was **not** a full independent remote checksum of all cloud binaries.
- Follow-up: all eight complete training histories and split manifests were inspected from Drive; both 420-image selected checkpoints were independently checked. Full remote byte-level verification of all six additional checkpoint files was not performed.
- The 180-image validation partition is disjoint by source SHA256 from the 420-image nested training pool; physical-specimen independence remains uncertain.

## Submission check

The experiment and final scientific writing are completed for the agreed scope. Confirmation that the professor's complete formal instructions are satisfied is **not possible without checking the original rubric**. Do not confuse saved artifacts with submission to a university system; no external submission has been made.
