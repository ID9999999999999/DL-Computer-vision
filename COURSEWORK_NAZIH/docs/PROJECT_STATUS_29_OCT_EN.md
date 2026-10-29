# Course Work Nazih — execution status (29 October 2026)

**Verified:** 6 automated offline unit checks pass; ResNet18 supervised checkpoint exists, matches archived SHA256, loads and predicts on 4 known Normal Motor images; manifest has 99 distinct SHA256 image identifiers; development split is 70 train / 29 validation, with no overlap of listed image paths or numeric groups.

**Source issue:** Only 4 of the manifest's 99 actual PNG images are stored locally. The missing 95 cannot be downloaded in this restricted environment. The public Motor.zip is approximately 15 GB; the provided download script fetches exact selected entries by HTTP byte ranges with SHA256 checks instead of downloading the whole ZIP.

**Not completed:** Poly-1 training and paired comparison. Any exact Poly-1 score reported before running the notebook would be fabricated. No valid new independent specimen-level test set exists; independence remains unverified. A clean confirmatory test requires a genuinely new specimen-disjoint split or separate external dataset.

**Deliverables:** one-slide editable PPTX and PDF, teacher-ready English plan, 3-minute speaking script, runnable Colab notebook, and original 45-MB executable comparison package.
