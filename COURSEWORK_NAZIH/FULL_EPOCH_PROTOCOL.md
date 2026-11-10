# Full-Epoch Follow-up Study — Separate protocol v1 (10 Nov 2026)

## Why a second experiment is necessary

The original 24 evaluation results (6 archived 70-image references + 18 new runs) are preserved exactly. The original run uses **12 rounds of 70 image exposures per run**, **216 optimizer steps** in total, and deliberately fixes compute rather than counting complete passes over the dataset. For training sizes 70/140/280/420, this is **12/6/3/2 effective complete dataset passes** respectively. Increasing a reported `epochs` field or extending the old runs would be scientifically misleading. This new experiment must never overwrite old models/results or retroactively rename 12 rounds as 12 full epochs.

## Full-epoch pilot, before expanding to further comparisons

- Name: `full_epochs_v1_validation_from_train_20261110`.
- Start: **420 training images**, original balanced 210+210 pool from existing frozen `results/size_manifests.json`.
- Split: **180 distinct validation images** chosen from the unused **official TRAIN** images: 150 normal and 30 defective. There are only 36 defective training samples left after the size-420 pool; reserving 30 minimizes overlap risk and leaves six unused. Selection seed `20261110`; deterministic class-stratified selection, with fixed source image hashes. Selection does not use test images.
- No official test images may be opened during pilot validation, selection, stopping, or model comparison. The already-used 1,004 test images cannot act as a fresh confirmatory set.
- ResNet18 ImageNet1K_V1 pretrained, full fine-tuning, RGB 448 x 448 BICUBIC with ImageNet normalization.
- AdamW, lr `1e-4`, weight decay `1e-4`, batch size 4, **30 actual full-data epochs**, no augmentation.
- 420 / 4 = **105 optimization steps per epoch**, total **3,150 optimizer steps** for one 30-epoch run. Compare with 216 historical steps; these are different experiments, not directly controlled comparisons.
- Pilot run: `size=420, seed=20261028, loss=ce` only, to check feasibility, signs of training stability, validation history and whether more training is justified. Do **not** run all 18 variants until pilot review.
- Fixed epoch count 30; save each epoch's validation Macro-F1, Accuracy, defective recall and normal FPR. Select earliest epoch attaining best validation Macro-F1. No threshold tuning on validation or test.
- Seeds, training image ordering and validation selection are deterministic; CPU vs GPU/software differences mean exact bitwise equivalence to historical runs is neither expected nor claimed.
- Save split, progress, resume state, and best checkpoint ONLY under `COURSEWORK_NAZIH/data/full_epoch_pilot/n420_seed20261028_ce/` (Git-ignored); original `results/` untouched. Resume is epoch-boundary based; repeated interrupted epochs can be retrained after crash, without changing old experiments.
- If expanding to fair CE vs Poly-1 inference, repeat same full-epoch protocol and train/validation split on paired seeds. If other dataset sizes are used, their validation must remain independent of their respective training pools; a common frozen validation pool is preferable and must exclude the largest (420) pool.
- Full-data epoch comparisons need consistent sampling, class composition, validation policy and compute accounting. A class-balanced 420-pool is not representative of the naturally imbalanced KSDD2 training distribution; report both that fact and the 30/150 validation distribution.
- Group-level leakage cannot be ruled out without reliable physical-part IDs. This limitation must remain explicit.

## Invocation

Open VS Code Jupyter Python cell on the machine containing the verified KSDD2 archive and local project.

Data and integrity preflight (does not train):

```python
from pathlib import Path
import subprocess,sys
repo=Path.home()/"Desktop"/"nazih-project"
subprocess.run(["git","pull","--ff-only","origin","nazih-training-set-size-20261109"],cwd=repo,check=True)
script=repo/"COURSEWORK_NAZIH"/"scripts"/"full_epoch_pilot.py"
archive=Path.home()/"Downloads"/"KolektorSDD2.zip"
subprocess.run([sys.executable,str(script),"--archive",str(archive),"--check-only"],check=True)
```

Then for pilot training on a GPU-equipped runtime that has the files, run:

```shell
python COURSEWORK_NAZIH/scripts/full_epoch_pilot.py --archive /path/to/KolektorSDD2.zip --size 420 --seed 20261028 --loss ce --epochs 30 --device cuda
```

CPU is intentionally guarded because 30 full 448 px ResNet18 epochs are costly. With an explicit decision to use CPU, add `--allow-cpu --device cpu`. The program is synchronous; no cloud job is started by committing it.

## Gate before expanding scope

1. Verify the preflight passes; confirm validation hashes disjoint from train/test hashes.
2. Review validation Macro-F1 trajectory, possible overfitting, train objective and class-specific recall/FPR for CE pilot.
3. Independently verify exact duration and logs of 30 epochs and integrity hash of selected checkpoint.
4. Only then select follow-up paired CE vs Poly-1 full-epoch experiments, keeping old 216-step analysis intact.
5. Never report new metrics until experiments have actually executed; never present selected validation scores as independent test estimates.

## How this responds to the professor

Earlier work answered: *At a fixed optimization budget (216 updates), how do training set diversity and loss differ?*

This new pilot addresses a different question: *What happens when every training image is visited at each of 30 complete epochs, with a disjoint training-only validation set?*

Both are academically legitimate designs if labeled accurately. Neither one should overwrite the other.
