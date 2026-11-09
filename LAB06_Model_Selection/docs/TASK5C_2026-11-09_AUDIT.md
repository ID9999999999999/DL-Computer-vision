# Task 5C checklist addendum — 9 November 2026

This supersedes **only** section 07 (Task 5C) in the dated 8 November snapshot; the overall 152-item counts have not been recalculated here.

| ID | Status on 9 November | Evidence |
| --- | --- | --- |
| C01 | 🟢 | Dominant observed error is glare; original archived YOLOv8n missed 3 of 28 glare examples. |
| C02 | 🟢 | Source script draws white ellipse glare alpha 150–230 only on TRAIN defective images. |
| C03 | 🟢 | Original Val/Test images and all 1,800 image hashes verified; neither split receives augmentation. |
| C04 | 🟡 | Genuine new training using same seed and matched control, but **continued from trained checkpoint**, not the original ImageNet weights, CPU PyTorch-compatible implementation, and original historical 15/16/4/224 protocol instead of later 50/32/5/640 specification. |
| C05 | 🟢 | Full augmentation settings and seed persisted in `results/bonus_augmentation_results.json`. |
| C06 | 🟢 | Defect-wise and binary good/bad F1 comparison and delta CSVs; glare recall and counts. |
| C07 | 🟢 | Glare recall unchanged 26/28; occlusion recall fell from 26/26 to 25/26; no false rejects of good images. |
| C08 | 🟢 | Executable training code, two tensor checkpoints, raw predictions, full experiment package, and five-line report. |

**Task 5C on this audit: 7 green, 1 yellow, 0 red.** No instructor acceptance claimed. Do not replace original measurements with the new numbers. Full experiment: https://drive.google.com/file/d/1HyxDmT2yJ7xX_wN4R76b0F8jREVrqfjM/view
