# Data provenance and scientific limits

- This project transforms the M2AD Motor category into a **binary supervised** classifier. Class 0 = Normal; Class 1 = Defective. A single RGB image (view A000, illumination I01) produces one decision.
- `resnet18_supervised/manifest.json` contains **99 historical development images**, grouped by a numeric suffix as an unproven proxy for specimen identity. There are 70 training and 29 validation images.
- The previous anomaly detection and dual-illumination final-test images were inspected. The 99-image supervised pilot's development validation is **not a new blind test**.
- **No physical-specimen independence is established**. Splits cannot be called specimen-level until mapping and capture provenance are verified.
- Exact SHA-256 checks protect files against corruption and duplicate image bytes; they cannot alone rule out visually near-duplicate images or different captures of one specimen.
- No untouched independent confirmation set is available in this archive. Before reporting test performance, get a new truly held-out, properly grouped set with no model selection on it.
- The prior CE pilot reached validation macro-F1 0.96535 and accuracy 0.96552 (28/29). These numbers **must not** be described as final test metrics.
- The Poly-1 comparison is prepared but **not executed**. Never claim an improvement before running and inspecting both methods.
- The 44.8 MB historical `best_model.pt` is **not committed** to Git; it is retained in the previously delivered local project archive.

## Sources
- https://github.com/hustCYQ/M2AD
- https://huggingface.co/datasets/ChengYuQi99/M2AD
- https://openreview.net/forum?id=gSdSJoenupI
