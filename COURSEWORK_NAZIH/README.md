# COURSE WORK NAZIH — Metal Defect Classification

**Course:** Deep Learning for Computer Vision · **Status (29 Oct 2026):** supervised ResNet18 baseline completed as an exploratory pilot; controlled Poly-1 Loss experiment prepared but **not trained**.

## Research question

Does Poly-1 Cross-Entropy (Leng et al., *PolyLoss*, ICLR 2022) improve **macro-F1** or **Defective recall** over ordinary Cross-Entropy when the same ResNet18, dataset split, ImageNet initialization, optimizer and training budget are used?

**Task:** single-image binary supervised classification: `0 Normal` / `1 Defective` using M2AD Motor images (A000/I01).

## One entry point

**Run with Google Colab:** [Open the notebook](notebooks/MOTOR_COMPARISON_COLAB.ipynb) in Colab and follow cells from top to bottom. The notebook clones this GitHub repository, installs prerequisites, verifies manifest checks, fetches the needed source images, trains both variants, and downloads a ZIP of actual results. **GPU recommended.**

To run on a computer with Python / PyTorch:

```bash
python -m pip install -r requirements.txt
python offline_preflight.py
python -m unittest test_pipeline.py -v
python download_missing.py --workers 4
python compare_poly1.py --mode both --epsilon 1.0 --device auto
```

Files required by the training code are kept together inside this folder; run commands **here**. Fetching M2AD files uses byte-range requests and can fail if the provider changes its API; it must fail visibly, not silently substitute data. Each image is checked against its pilot SHA256.

## Organization

| Path | Purpose |
|---|---|
| `notebooks/MOTOR_COMPARISON_COLAB.ipynb` | Your Colab entry point |
| `compare_poly1.py` | Matched CE versus Poly-1 training and evaluation |
| `download_missing.py` + `archive_io.py` | Recover exactly the missing M2AD images |
| `offline_preflight.py` + `test_pipeline.py` | Dataset and code integrity checks |
| `resnet18_supervised/` | Historical baseline configuration, data manifest, training script, verified pilot metrics |
| `docs/` | Professor's checkpoint plan, speaking script, limitations |
| `slides/` | 30 October single-slide PowerPoint |
| `comparison_results/` (generated) | Models, predictions and metrics from new actual runs — ignored by Git |

## Current verified evidence

| Item | Value |
|---|---|
| Historical pilot | ResNet18 trained on Normal/Defective with cross-entropy |
| Train / validation | 70 / 29 original images; physical-object independence unverified |
| Pilot validation accuracy | **96.55%** (28/29), NOT final test accuracy |
| Pilot macro-F1 | **96.54%**, NOT final test macro-F1 |
| Poly-1 results | **Not available: training not completed** |

## Do not commit dataset/weights

Raw M2AD images, downloaded datasets, trained `.pt`/`.pth` checkpoints and training outputs stay **outside GitHub** to avoid unnecessary huge files, copyright/license ambiguities, and repository clutter. The historical checkpoint remains in the previous locally delivered archive; its SHA256 is documented in `resnet18_supervised/result.json`. Every experiment is associated with hashes and a manifest.

## Deadlines

30 October: plan + two papers + one slide. 6 November: dataset + baseline. 13 November: paper idea & matched comparison. 20 November: metrics/error analysis. 27 November: final experiments & report draft. 4 December: defense (3 December internal deadline).

For transparent limitations, see [data provenance](docs/DATA_PROVENANCE_AND_LIMITATIONS.md).
