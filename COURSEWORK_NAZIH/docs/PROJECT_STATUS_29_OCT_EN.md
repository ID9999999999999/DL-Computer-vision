# Course Work Nazih — execution status (29 October 2026)

## Historic pilot (executed, local archival package)
- ResNet18 supervised binary training completed on 99 historical M2AD Motor development images: 70 train and 29 validation.
- Pilot validation macro-F1 ~96.54%; accuracy 96.55% (28/29). These are **not** independent test metrics.
- The original local archival package contained 4 image files, the 44.8 MB ResNet18 checkpoint and 6 passing unit tests, including checkpoint reload/inference sanity.
- The model's checkpoint hash and all 99 image hashes are recorded under `resnet18_supervised/`.

## GitHub source repository (this directory)
- **28 organized code, metadata, documentation, slide and notebook files committed.**
- Full M2AD source images **are not stored here**; the manifest lists 99 required file paths, and their byte integrity is verified upon retrieval.
- Large historical `best_model.pt` is **not stored here**. Consequently, a clean GitHub clone runs 5 offline unit tests with 1 checkpoint-dependent test skipped. This is deliberate, not a failed test.
- `offline_preflight.py` reports 99 images absent until downloaded. `download_missing.py` is designed to fetch them using HTTP ranges; remote service behavior has not been verified in this current environment.

## Incomplete scientific work
- No Poly-1 training or matched CE-versus-Poly-1 metrics have been executed.
- The historical development validation is not a blind test; physical-specimen independence has not been established.
- Any future confirmatory testing requires truly independent specimens / verified grouping.
- Next execution: open the GitHub-hosted `notebooks/MOTOR_COMPARISON_COLAB.ipynb` in Colab, run download and matched training, and preserve outputs as developmental results with candid limitations.

## Deliverables
- GitHub source includes the editable 30 October PPTX slide, project plan and speaking script.
- Previous conversation ZIP archives retain baseline checkpoint and supplementary historical material; they were intentionally not re-uploaded wholesale to the GitHub course repository.
