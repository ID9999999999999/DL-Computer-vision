# Industrial Surface Defect Classification

**Student:** YASSER IDBOUZKRI · **Group:** иикб-24  
**Course:** AI Implementation Technologies · **Dataset:** KolektorSDD2

## Contents

- `REPORT.pdf` — final scientific report.
- `PRESENTATION.pptx` — final defense slides.
- `PROJECT/CODE/CODE.zip` — Python source, frozen configuration, test scripts and reproducibility instructions.
- `PROJECT/RESULTS/RESULTS.zip` — original evaluations, training histories, validation splits, reproducible audit, and traceable error analysis.
- `PROJECT/MODELS/` — 18 original and eight full-epoch checkpoints, plus the historical checkpoint archive (Google Drive submission).

## Verify results from a clean directory

Extract `CODE.zip` and `RESULTS.zip` side by side. With Python 3.11+, run:

```bash
python CODE/scripts/audit_submission.py --results RESULTS --out AUDIT
```

This **CPU-only, standard-library** check regenerates error cases, confirms the 18 new and six historical evaluation records, validates the eight 30-epoch run histories, and checks common validation splits. The exact steps to restore source images and inspect training are in `CODE/README.md`.

## Scope and limitations

The original comparison used 216 updates per run. The follow-up completed **eight real 30-full-epoch trainings**, comparing Cross-Entropy and Poly-1 at 70, 140, 280 and 420 images (seed 20261028). Both 420-image models attained **98.00% selected internal validation Macro-F1**, not independent test accuracy. The official test was previously inspected; physical-specimen independence and clean GPU retraining are not established.
