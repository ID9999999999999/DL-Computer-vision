# Industrial Surface Defect Classification

**Student:** YASSER IDBOUZKRI · **Group:** иикб-24  
**Course:** AI Implementation Technologies · **Dataset:** KolektorSDD2

## Submission

- **REPORT.pdf** — experimental method, results, limitations and references.
- **PRESENTATION.pptx** — 12-slide project defense.
- **PROJECT/** — source, models and original observed results:
  - **CODE/** — runnable source archive and the full-epoch training script.
  - **MODELS/** — 26 checkpoints: 18 fixed-budget models and eight 30-epoch models.
  - **RESULTS/** — original evaluations, historical records and new training histories.

## Verified scope

Fixed-budget study: 24 exploratory evaluations, 216 optimizer updates per model. Full-epoch study: eight genuine 30-epoch trainings (70–420 images; CE and Poly-1; seed 20261028). At 420 images, both losses achieved **98.00% best internal validation Macro-F1**; this is not independent test accuracy.

**Code and results:** also versioned in GitHub. **Binary checkpoints:** stored in the linked Google Drive submission because of their size. Original research evidence and dataset remain preserved in the research archive.
