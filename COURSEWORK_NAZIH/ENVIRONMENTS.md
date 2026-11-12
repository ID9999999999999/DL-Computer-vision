# Executed software environments

The two experimental protocols used different runtime environments; they must not be conflated.

| Component | Fixed-216-update protocol (historical CPU reference) | 30-full-epoch follow-up (Google Colab GPU) |
| --- | --- | --- |
| Python | 3.12.14 | 3.13.16 |
| PyTorch | 2.15.0+cpu | 2.14.1+cu130 |
| torchvision | 0.29.2+cpu | 0.29.1+cu130 |
| NumPy | 2.5.4 | 2.4.6 |
| Pillow | 13.0.0 | 11.3.0 |
| Device | CPU | Tesla T4 |

The historical reference environment has an exact-version guard in `scripts/train.py`. The full-epoch code records the executed CUDA device and package versions in its own immutable `COMPLETE.json` records. CUDA wheel indexes and runtime compatibility depend on the host; do **not** install a CPU-only PyTorch wheel for GPU training and do not misreport an environment as identical without checking it.

`requirements.txt` describes the historical CPU-oriented dependency set; it is not a claim that both studies used one software version. The independent archive verification script uses only Python's standard library.
