# Model checkpoints

**26 `.pt` weights are saved in the same MODELS folder in the Drive submission:**
https://drive.google.com/drive/folders/1_8iTUOrcSy1LxEauvfEjv3cdyQEW8y6b

- `fixed216_*.pt` — 18 original trained models (sizes 140/280/420, three seeds, two losses).
- `full30_*.pt` — eight 30-full-epoch best checkpoints (sizes 70/140/280/420, seed 20261028, two losses).
- The six historical 70-image fixed-budget evaluations are preserved in `RESULTS.zip`; their archived weights were not part of the 18 new original checkpoint files.

GitHub deliberately excludes large model binaries. Run histories, split manifests and original evaluations are in `PROJECT/RESULTS/RESULTS.zip`. Official dataset remains stored under `Nazih_Coursework/05_Official_Dataset/`.
