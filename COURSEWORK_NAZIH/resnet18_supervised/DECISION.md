# ResNet18 baseline decision and scope

Decision: full ImageNet-initialized ResNet18, with a two-output classifier, is the practical baseline for the next supervised experiment. The previous frozen-feature ResNet18 versus AnomalyDINO study is retained as preliminary evidence, not proof of universal architecture superiority.

The new pilot fine-tunes all ResNet18 parameters with cross entropy on Normal and Defective images. This is materially different from the previous normal-reference nearest-neighbour detector. Do not compare their scores as though only a module changed.

## Current development data

One image per directory object: Motor view 000, light 01. Existing selected files yield 50 normal and 49 defective examples; the formerly reserved bending_2_030 object is still excluded. All previous selections/inspection are disclosed. There is NO new final test partition. The 99 images are reassigned to training/validation for an explicitly exploratory engineering run. Original files/results are preserved.

Numeric suffixes are grouped conservatively before a fixed hashed split: 35 groups for training and 15 for validation. This joins potentially related normal/defective names, but numeric suffixes have NOT been established as physical specimen IDs. Thus this is a precaution, not proof of independence. The pilot validation is used for checkpoint selection and must not be presented as an independent generalization estimate.

## Source review

M2AD paper section 3.1 describes engineered defects, aligned acquisition and annotation/detectability procedures. The reviewed passage does not supply a row-level before/after specimen map. We therefore do not infer verified identity from directory naming.

- https://arxiv.org/html/2505.10996v1
- https://github.com/hustCYQ/M2AD

## Pilot protocol

3 epochs, seed 20261028, batch size 4, 448x448 full-frame bicubic RGB images, ImageNet normalization, AdamW learning rate 1e-4 and weight decay 1e-4, unweighted cross entropy, no augmentation. These are initial engineering settings, not optimized or proven best. Argmax gives Normal/Defective decisions without test-derived threshold selection. Best validation macro-F1 checkpoint, earliest on ties.

The script checks source image hashes, numeric group disjointness, both classes, finite gradients, actual convolution/head parameter updates, and checkpoint reload parity. These checks establish that the intended training pipeline runs, not robustness or scientific improvement.

## Remaining scientific work

Select and source-audit one paper idea. Compare the SAME trained ResNet18 baseline with/without that idea under a frozen development protocol and justified search budgets. Resolve data provenance and reserve truly independent confirmation before final claims. Keep all pilot failures and avoid choosing an addition solely to improve reused validation results. Do not claim the paper addition is implemented or that this pilot completes the course project.

## Reproduction

Place this folder beside the original motor_comparison, motor_classification and motor_final data folders. Run prepare.py and train.py in the recorded PyTorch environment. Source images and pretrained checkpoint are not duplicated in this package. best_model.pt is the newly trained two-class checkpoint and is included. The ImageNet initialization hash is checked by train.py.
