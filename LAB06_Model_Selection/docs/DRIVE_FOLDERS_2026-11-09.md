# LAB06 — Organized Drive folders and evidence (9 November 2026)

**Single entry point:** [START HERE](https://docs.google.com/document/d/1Pv5PZu81vR2gaNotgyRmq1Q3MaGgj3PtHheSu2wBfYU/edit) | **Pause-point handoff:** [RESUME LATER](https://docs.google.com/document/d/1QT9NeGwCxnpNiH2KiIMf6Peiipf8th8AfVnRJwM5eKQ/edit). This organization changed folders/indexes only; it did not change underlying experiments, source images, model weights or evaluation records.

| Folder | Content |
|---|---|
| [00_INSTRUCTIONS](https://drive.google.com/drive/folders/1pRFQQHEpripLdUE4sx6Qd0sqHW7UjD7o) | Original professor documents |
| [01_DATASET_A](https://drive.google.com/drive/folders/1r508sgO35TrcMeyU7DpzeHJMiAmlaAQN) | Source dataset / manifest pointers |
| [02_TASK_1_MODEL_SELECTION](https://drive.google.com/drive/folders/1geA-JYwJvq_ds0yo2vm2FkFi6GZ6aiv8) | Model-selection memo, Word hand-in preparation |
| [03_TASK_2_MOBILENETV2](https://drive.google.com/drive/folders/17GXTP8W0eou3I056C-cbhPxOT7WuvpLZ) | Frozen baseline |
| [04_TASK_3_YOLOV8](https://drive.google.com/drive/folders/16JtJ6hbtZ9XEMtn7kcV5Jm8SXkyLA0kJ) | Nano/small classifiers |
| [05_TASK_4_COMPARISON](https://drive.google.com/drive/folders/1utnuczpJDwz1kLhPHpPFGc6uPFAC8DlF) | Third architecture, comparison and presentation |
| [06_TASK_5A_FINE_TUNING](https://drive.google.com/drive/folders/1ND_PZ2r28X4DO5VPGtDuAGT6O1URXc29) | MobileNetV2 top-layer fine-tuning |
| [07_DATASET_EXCHANGE_PENDING](https://drive.google.com/drive/folders/1LqZO5O-KI2YayEHcr13VC8x9L9xlP0RE) | Peer checkpoint evaluation on our Test; Dataset B still missing |
| [08_FINAL_SUBMISSION](https://drive.google.com/drive/folders/16ioAY5oTkkspCM5iKLnHtJ3XEbKKb_ml) | **One current review ZIP**, four-slide deck and Word memo |
| [09_ARCHIVE_AND_ORIGINALS](https://drive.google.com/drive/folders/1miKvv0YAUmhssLLpBaQzmNw48MQce1HD) | Old checklists, old notes, original pointers |
| [10_BONUS_TASK_5B_THRESHOLD](https://drive.google.com/drive/folders/16oyRxD_zSO18fI8m65zXbNlbfN7bTVLh) | Validation-only threshold |
| [11_BONUS_TASK_5C_AUGMENTATION](https://drive.google.com/drive/folders/1CUj5NbtQEtWzytQ1FTkm_w4XAKMjsJNJ) | Negative glare-augmentation experiment with controls |
| [12_BONUS_TASK_5D_ONNX_DEPLOYMENT](https://drive.google.com/drive/folders/1XvbwxxuOcNsV95CHv6Tqr3yN9hYs8k04) | ONNX model, paired 270 Test evaluation, CPU timing |
| [13_BONUS_TASK_5E_PEER_AUDIT](https://drive.google.com/drive/folders/1rS2aomfbqzEb13f149Sy02hcHsTwfUA3) | Peer original predict.py execution; model card |

**Use only** [Lab06_COMPLETE_SUBMISSION_REVIEW_FINAL_2026-11-09.zip](https://drive.google.com/file/d/1ZAp0JdWW_DruPzZrG9ll96Z-Idjr1-4D/view) as the current **review** package. It is not officially submitted or teacher-approved. Older draft ZIPs are inside [90_PREVIOUS_SUBMISSION_DRAFTS](https://drive.google.com/drive/folders/1P6bBsAjuDga8jctuq7isRTn3HxF2FnnB); dated checklists are inside [90_SUPERSEDED_CHECKLISTS](https://drive.google.com/drive/folders/15RBc4K9lCZmyg9OaQCBwDdT4E9ds9sYj).

**Latest verified checklist**: [PDF](https://drive.google.com/file/d/1Gr6IGAAVCjB7o8H8_1rfQ3A4i8iewawu/view) | [Markdown](https://drive.google.com/file/d/1NhEt9AuMYn1RZPtJwPf2FDQsTDb-LfHf/view) = **100 green / 25 yellow / 27 red** of 152, including optional sections.

**Scientific boundaries:** Task 5C was matched checkpoint continuation, not full ImageNet restart. Task 5D preserves original 267/270 predictions on Test after ONNX export; forward-only CPU timings are not end-to-end deployment speeds. Task 5E literally ran the peer's inference script on *our* Test, not their missing original Test. Historical 50/32/5/640 vs 15/16/4/224 training differences remain disclosed; no formal receipt/defense or Dataset B has arrived.

**Historical tags:** [week7](https://github.com/ID9999999999999/DL-Computer-vision/tree/lab06-week7-verified), [week8](https://github.com/ID9999999999999/DL-Computer-vision/tree/lab06-week8-verified) reference the **same October 25 original experiments commit**.
