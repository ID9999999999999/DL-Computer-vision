# Industrial Surface Defect Classification: Cross-Entropy and Poly-1 under Two Training Budgets

**Course:** AI Implementation Technologies  
**Student:** YASSER IDBOUZKRI  
**Group:** иикб-24  
**Dataset:** KolektorSDD2  
**Final experimental freeze:** 12 November 2026  
**Study status:** Original fixed-update study completed; independent 30-full-epoch follow-up completed for one random seed. No further training is included in this final scope.

## Abstract

This study examines binary classification of industrial surface images (normal versus defective) using ImageNet-pretrained ResNet18 and two loss functions: standard Cross-Entropy (CE) and Poly-1. Two **distinct experimental protocols** are reported rather than merged. The initial fixed-compute study compares 70, 140, 280, and 420 training images at 216 optimizer updates, with three seeds and previously archived reference runs. After a concern about the limited number of complete dataset passes, an additional study trained all eight combinations of four image counts and two losses for **30 genuine full epochs** at seed 20261028. A common 180-image holdout drawn exclusively from the official training partition served for model selection in the follow-up. The new study achieved best internal validation Macro-F1 from 90.62–91.12% with 70 images up to 98.00% with 420 images. CE and Poly-1 tied at 280 and 420 images. These validation results are exploratory; neither demonstrates independent production performance.

## 1. Research question and scope

The objective is to compare CE and Poly-1 for normal/defective classification and investigate associations between training-set size and predictive metrics. The work is divided into two analyses with different compute budgets and evaluation populations. **Scores from these two analyses must not be compared as though they were computed on the same holdout.** The project is considered complete at the documented eight-run, single-seed follow-up; a larger 24-run extension is a possible future study, not completed work or an academic obligation established here.

## 2. Dataset and verification

KolektorSDD2 provides **2,331 official training images** (2,085 normal and 246 defective) and **1,004 official test images** (894 normal and 110 defective). The official source archive was checked against SHA256 `edcdb486809b24f1d17b785e30c52fafc5999554dd5fe18ddf77b61ceb6f36a8`. The selected training subsets contain 70, 140, 280, and 420 images, balanced equally between normal and defective and nested in ascending size. Exact image hashes were recorded for provenance; overlap by identical file content was excluded between selected training and the relevant evaluation subsets. **Physical-part or manufacturing-specimen independence was not established**, and equal hashes cannot detect all near-duplicates.

In the follow-up experiment, the same **180 validation images** were selected once from otherwise unused images in the official *training* partition, comprising 150 normal and 30 defective. This selection excludes the entire frozen 420-image training pool, so validation is disjoint from the 70-, 140-, 280-, and 420-image training subsets. The official test images were not used for follow-up model selection. One additional defective classification error changes defective recall by **3.33 percentage points**, making the reported recall sensitive to sample size.

## 3. Model and fixed settings

All experiments use **ResNet18**, initialized with torchvision ImageNet1K V1 weights, with a new linear two-class classification head and end-to-end fine-tuning. Input images are converted to RGB, bicubically resized to **448 × 448**, and normalized using ImageNet channel means and standard deviations. Optimizer: **AdamW**, learning rate `1e-4`, weight decay `1e-4`, batch size **4**. No data augmentation was used. Poly-1 follows the classification loss form **Poly-1 = CE + (1 - p_true)** with epsilon set to 1; `p_true` denotes the predicted probability assigned to the correct class. The follow-up holds the image order, seed, architecture and training hyperparameters constant within each CE/Poly-1 pair.

## 4. Study A — fixed optimizer-update budget

The initial study contained **24 evaluation records**, corresponding to four image-count conditions, two losses and three seeds. The 70-image source was an archived historical reference; **18 new checkpoint trainings** were produced for the other image-count conditions. Every condition used **216 optimizer updates** and 12 reference-equivalent rounds of 70 image exposures, not 12 complete passes through each larger training set. The effective complete-pass equivalents were **12, 6, 3, and 2** for sizes 70, 140, 280, and 420 respectively. BatchNorm running statistics were recalibrated under a fixed 70-image protocol. The values below are means across three seeds and relate to the **previously inspected** official 1,004-image test set.

| Images | CE Macro-F1 | Poly-1 Macro-F1 | CE AUROC | Poly-1 AUROC |
|---:|---:|---:|---:|---:|
| 70 | 76.64% | 75.72% | 85.68% | 85.62% |
| 140 | 77.86% | 79.24% | 88.25% | 88.95% |
| 280 | 84.76% | 86.10% | 89.67% | 90.80% |
| 420 | 84.80% | 86.07% | 92.27% | 93.07% |

The strongest mean Macro-F1 in the fixed-216-update study was **86.10%** for 280 images with Poly-1, while the highest mean AUROC was **93.07%** for 420 images with Poly-1. Poly-1 had higher mean Macro-F1 at 140, 280 and 420 images but not at 70 images. The historical 70-image references were not retrained in the same numerical environment as the new checkpoints. These results describe the fixed-budget exploratory setup and do not establish causality or a blind held-out generalization result.

## 5. Study B — 30 complete dataset epochs

To address the limited effective passes in Study A, a **new, separate experiment** was conducted. Each model was trained for exactly **30 complete epochs** on every image in its selected training subset. For 70, 140, 280 and 420 images, this produces **540, 1,050, 2,100 and 3,150 optimizer updates per model**. Eight real full-epoch training runs were completed: four sizes × two losses × **one seed (20261028)**. The best checkpoint of each run was chosen at the earliest epoch achieving the highest **internal validation Macro-F1** across the 30 recorded epochs. Validation was measured on the **same 180-image set in all eight conditions**.

| Images | Updates/run | CE best epoch | CE Validation Macro-F1 | Poly-1 best epoch | Poly-1 Validation Macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 70 | 540 | 23 | 91.12% | 19 | 90.62% |
| 140 | 1,050 | 12 | 91.29% | 30 | 93.46% |
| 280 | 2,100 | 7 | 95.77% | 19 | 95.77% |
| 420 | 3,150 | 30 | 98.00% | 22 | 98.00% |

At 420 images, both losses selected models with **98.00% validation Macro-F1** and **178/180 = 98.89% accuracy**. Their aggregate confusion matrices were each `[[149, 1], [1, 29]]` (ground truth rows: normal, defective). The 420-image CE model was selected from epoch 30; Poly-1 from epoch 22. The selected epoch by itself does not demonstrate general faster convergence. At 280 images the best Macro-F1 values were also equal (95.77%); Poly-1 was higher at 140 images (93.46% versus 91.29%), while CE was slightly higher at 70 images (91.12% versus 90.62%). **The follow-up does not show a consistent Poly-1 advantage.**

## 6. Interpretation

The second protocol directly resolves the concern that 420-image models had received only approximately two complete passes in the fixed-update study: both 420-image models were trained for 30 real full passes, and the other three sizes received the same nominal epoch count. The increasing internal validation Macro-F1 trend is consistent with useful additional training information, but this interpretation is limited in two major ways. First, the number of optimizer updates also rises with training-set size, so sample diversity and total update count are **confounded** in Study B. Second, each reported checkpoint was chosen using the same finite validation set; these are selected validation metrics rather than an independent final performance estimate. **The approximately 98% internal validation score must not be contrasted directly with the older 86% previously-used test score** because the protocols, data and selection rules differ.

## 7. Scientific limitations

- **Follow-up replication:** only one training seed is completed in Study B; there are no three-seed full-epoch averages or variability estimates.
- **Validation selection optimism:** the highest of 30 validation measurements was selected for each run.
- **Small defective validation sample:** 30 defective cases limit precision of recall comparisons.
- **Compute-size confounding:** 30 passes imply different total update budgets across training sizes.
- **Distribution shift:** balanced training sets do not reproduce the natural defect prevalence; the 150/30 validation mix is also a choice rather than a deployment estimate.
- **Specimen grouping:** physical-part independence and near-duplicate isolation are not fully verified.
- **Prior test exposure:** the 1,004-image official test had already been inspected in earlier experiments; Study A is exploratory and cannot be claimed as fresh blind confirmation.
- **Environment differences:** archived 70-image fixed-budget runs and subsequent conditions used historically different runtime environments.

## 8. Conclusion

A complete eight-run, 30-full-epoch follow-up was executed and saved separately from the original fixed-216-update study, without modifying historical outputs. The new data demonstrate that both losses can achieve **98.00% best internal validation Macro-F1** for the 420-image condition in this one-seed setup. No loss function consistently dominates across all four sizes. The original fixed-update study and the follow-up answer **different experimental questions**; both should be reported with their own evaluation population, compute budget and limitations. This work is final **for the agreed scope of eight full-epoch runs**, not for a hypothetical 24-run multi-seed extension. An untouched, specimen-separated evaluation set and multiple replicated seeds would be necessary for stronger generalization or statistical claims.

## References

1. He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. https://arxiv.org/abs/1512.03385
2. Leng, Z. et al. (2022). *PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions*. ICLR. https://openreview.net/forum?id=gSdSJoenupI
3. ViCoS Laboratory. *Kolektor Surface-Defect Dataset 2 (KolektorSDD2)*. https://www.vicos.si/resources/kolektorsdd2/

## Reproducibility and project locations

- Source code, protocols, archived results, audits: https://github.com/ID9999999999999/DL-Computer-vision/tree/nazih-training-set-size-20261109/COURSEWORK_NAZIH
- Original fixed-update checkpoints and records: `Nazih_Coursework/Training_Size_Results`, `01_Historical_70`, `02_Evaluation_Results` on Google Drive.
- New full-epoch checkpoints, training histories, frozen validation manifests and resume state: `Nazih_Coursework/06_Full_Epoch_Pilot` on Google Drive.
- New best-checkpoint SHA256 values are recorded in each `COMPLETE.json`. For the six newest full-epoch runs, remote content hashes have not been independently recomputed; do not claim a full remote bitwise audit.
