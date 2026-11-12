# Course Work — Current Status (12 November 2026)

**Student:** YASSER IDBOUZKRI · иикб-24  
**Course:** AI Implementation Technologies  
**Main index:** [README.md](README.md)  
**Submission report:** [results/FINAL_ACADEMIC_REPORT_EN.pdf](results/FINAL_ACADEMIC_REPORT_EN.pdf)  
**Presentation:** [results/NAZIH_FINAL_PRESENTATION.pptx](results/NAZIH_FINAL_PRESENTATION.pptx)

## Completed studies

**Study A — fixed compute.** Four nested training-population sizes (70/140/280/420); Cross-Entropy versus Poly-1; three seeds. The historical 70-image reference was preserved, 18 new trainings completed, and 24 evaluation records compiled. Fixed budget: 216 optimizer steps per run, or 12/6/3/2 effective full-data passes across the four sizes. The official test set was previously inspected; this is exploratory, not fresh blind confirmation.

**Study B — true full epochs.** Eight new, independent, actual **30-full-epoch** trainings were completed for one seed, 20261028, across the same four sizes and two loss functions. The same 180-image internal validation set (150 normal, 30 defective), selected only from the official training partition, was used throughout. Each size has its own total optimizer steps: 540/1050/2100/3150. The 420-image pair achieved 98.00% best internal validation Macro-F1 for both CE and Poly-1. These selected validation metrics are not test accuracy.

**Research scope frozen:** 8/8 adopted follow-up runs completed. A larger proposal involving 16 additional runs was **not executed** and is future work only. No additional training is necessary for the current report and presentation.

## Preserved records and reproducibility

Current code and fixed protocols remain in `scripts/`, `protocol.json` and `FULL_EPOCH_PROTOCOL.md`. Original training/evaluation JSONs remain in `results/` and `input/`. Eight new binary checkpoints and their full histories are backed up under Google Drive [`Nazih_Coursework/06_Full_Epoch_Pilot/`](https://drive.google.com/drive/folders/11P4PVeDzSasFkAQLEIjJygp_gbYy2EWf). The original source ZIP is maintained in `05_Official_Dataset/`; model and dataset binaries must not be committed to GitHub.

Previous versions of this README were **pre-execution plans dated 8 November 2026**. Their history is retained by Git, but their statements that training had not yet begun are no longer current. For detailed data assumptions and caveats, consult [the final report](results/FINAL_ACADEMIC_REPORT_EN.md) and [project closeout](results/PROJECT_CLOSEOUT.md).
