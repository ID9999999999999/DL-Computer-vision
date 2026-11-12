# Saved Model Checkpoints

- **18** `fixed216_*.pt`: 216-update study; sizes 140, 280, 420; three seeds; CE and Poly-1.
- **8** `full30_*.pt`: 30-complete-epoch study; sizes 70, 140, 280, 420; seed 20261028; CE and Poly-1.
- **Historical 70-image source archive**: `HISTORICAL_70_CHECKPOINTS.part01` through `.part08`, eight consecutive parts of the preserved `Duration_Training_Checkpoint.zip`. Their exact sizes and SHA256 values appear in `HISTORICAL_70_PARTS.txt`. Reassemble in numerical order; the reconstructed archive must have SHA256 `403cc252f7736ca35f448a2f48e85fa2c40690ea61aa23b65eb0de3d69e73733`.

The eight full-epoch checkpoint selection records and the original evaluation logs are in `../RESULTS/RESULTS.zip`. All 26 modern `.pt` model files and the eight historical archive parts were **copied**, not moved, from the preserved research data. GitHub holds the source code and evaluation metadata; large model binaries are stored here on Google Drive.
