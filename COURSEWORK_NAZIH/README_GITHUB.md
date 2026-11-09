# Nazih · Training-set-size study · Colab launcher

**Author:** YASSER IDBOUZKRI (иикб-24). Original 70-image baseline is **read-only**. New sizes: 140, 280, 420; three seeds and CE vs Poly-1. **Status:** source code and historical JSON evidence packaged; **0/18 new model trainings completed**.

## Use

Open `COLAB_START.ipynb` from this branch in Google Colab, then run its cells **one at a time**. It clones the GitHub branch; downloads the original full KSDD2 archive from its official publisher (or asks you to provide it if unavailable); verifies its SHA256; recovers the 70 historical training images and checks per-image SHA256; runs safety tests; refuses a mismatched historical CPU environment before any training.

### Why images and weights are not in GitHub

The original dataset has a non-commercial license and is distributed at https://www.vicos.si/resources/kolektorsdd2/. It is not rehosted in this repository. The 500 MB archived baseline checkpoints are also not uploaded; their SHA256 audit and evaluation JSON remain archived, but training new sizes must not overwrite the old results. The new experiments may need substantial CPU time. Save results to Google Drive between sessions.

### Known caveats

* The old 70 baseline was trained under Python 3.12.14, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, NumPy 2.5.4 and Pillow 13.0.0. **Default Colab T4 is not an exact match**. A mismatched setup will stop; that is deliberate.
* The official dataset may not be accessible for automatic download; the code does not fabricate a substitute.
* The previously examined 1,004-image test is **exploratory, not untouched**. Part-level independence has not been proven.
* There is no scientific basis to claim new 140/280/420 results until actual execution produces checkpoints and metrics.

See `README_EN.md` for the full protocol and `protocol.json` for locked experimental settings.