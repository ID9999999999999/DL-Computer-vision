# Industrial Surface Defect Classification

**Student:** YASSER IDBOUZKRI · **Group:** иикб-24  
**Course:** AI Implementation Technologies · **Dataset:** KolektorSDD2

## Submission

- **REPORT.pdf** — experimental methodology and findings.
- **PRESENTATION.pptx** — final defense slides.
- **PROJECT/CODE/CODE.zip** — executable source, locked settings and reference data.
- **PROJECT/RESULTS/RESULTS.zip** — original evaluation records, 30-epoch histories and verified error cases.
- **PROJECT/MODELS/** — 26 verified checkpoints and historical archive.

## Check

Extract both ZIPs into their named folders. From `PROJECT/CODE`, run:

```bash
python scripts/verify_submission.py --results-zip ../RESULTS/RESULTS.zip
```

Add `--models-dir ../MODELS` to verify all 26 checkpoint SHA256 values. This reproducibility audit was successfully run from a fresh extraction without retraining. The 30-epoch scores are **internal validation**, not independent test accuracy. The earlier official test was already inspected.

**Full technical provenance:** [research source](https://github.com/ID9999999999999/DL-Computer-vision/tree/nazih-training-set-size-20261109/COURSEWORK_NAZIH). The official dataset is distributed separately under CC BY-NC-SA 4.0.
