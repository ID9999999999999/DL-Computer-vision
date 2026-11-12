# Eight verified full-epoch runs — seed 20261028 (12 Nov 2026)

## Verification
Eight of the 24 planned separate 30-true-epoch runs were independently inventoried in Google Drive: sizes 70, 140, 280, 420, each trained with CE and Poly-1 at seed 20261028. Six new run directories each contain `COMPLETE.json`, `progress.json`, `split.json`, `best_model.pt`, `resume_state.pt`. The two earlier 420-image runs were preserved intact.

All eight `COMPLETE.json` files were downloaded and parsed, and each records 30 full epochs, the correct optimizer-step count, a selected best validation epoch, a recorded 64-character checkpoint SHA256, and `official_test_evaluated=false`. The model-file hashes of the six new checkpoints have **not** been independently recomputed on the remote copies; this audit must not be interpreted as remote byte-for-byte confirmation.

The `split.json` files were independently retrieved for sizes 70/140/280/420: the same ordered 180-image validation subset was present in all four (150 normal, 30 defective), with zero hash overlap with the corresponding train images. The validation subset was frozen from the official training partition only. CE and Poly-1 use an identical split file within each size.

## Results: BEST internal validation Macro-F1 (one seed, not test accuracy)
| Training images | CE | Poly-1 | CE selected epoch | Poly-1 selected epoch | Steps in each 30-epoch run |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 70 | 0.911179340972641 | 0.9062011464304325 | 23 | 19 | 540 |
| 140 | 0.9128540305010893 | 0.934640522875817 | 12 | 30 | 1050 |
| 280 | 0.9577067669172932 | 0.9577067669172932 | 7 | 19 | 2100 |
| 420 | 0.9800000000000000 | 0.9800000000000000 | 30 | 22 | 3150 |

Selected best-epoch accuracy / defective recall / normal FPR:
- 70 CE: 0.9500 / 0.8667 / 0.0333; Poly-1: 0.9500 / 0.8000 / 0.0200.
- 140 CE: 0.9556 / 0.7667 / 0.0067; Poly-1: 0.9667 / 0.8000 / 0.0000.
- 280 CE and Poly-1: 0.9778 / 0.8667 / 0.0000.
- 420 CE and Poly-1: 0.9889 / 0.9667 / 0.0067.

## Scientific limitations
- **Only one training seed**: no mean, standard deviation or reliable statistical loss comparison can yet be reported.
- The 180-image validation has only 30 defective cases: one additional error changes defective recall by **3.33 percentage points**.
- Every size was trained for 30 COMPLETE dataset passes, so optimizer-update counts vary with training-size. The trend across sizes is NOT an isolated effect of sample size under a fixed compute budget.
- Comparing CE and Poly-1 within one size, seed and budget is paired, but comparing different sizes may confound sample diversity and optimizer steps.
- The official 1,004-image test was already examined in earlier study; do not represent these validation numbers as independent test or deployment accuracy.
- Physical-part independence not verified and the training pools were artificially balanced.
- The original 216-step study and its 24 evaluation results are preserved unchanged.

## Evidence
- [Google Drive full-epoch outputs](https://drive.google.com/drive/folders/11P4PVeDzSasFkAQLEIjJygp_gbYy2EWf)
- Drive `BATCH_SEED20261028_SIZES70_140_280.json`, six separate `COMPLETE.json` files and two original 420-image `COMPLETE.json`.
- [420-image paired audit](FULL_EPOCH_PAIRED_420_AUDIT_2026-11-11.md)

## Next
The remaining 16 full-epoch runs cover seeds 20261029 and 20261030, four sizes and two losses. Run in small, Drive-resumable GPU batches because Colab free GPU quotas interrupted earlier work. Do not conduct test-set-based model selection.
