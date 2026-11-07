# efficientnetb0 — individual test errors

The model produced **3** true misclassifications on the original 270-image held-out test set. These are not synthetic additions or examples from another split.

[Open its private Google Drive error-image folder](https://drive.google.com/drive/folders/1TvpXuSSObizsLdPyaPZh-UB3QebsIsqM).

| Image | True label | Predicted label | P(bad) | Photo |
|---|---|---|---:|---|
| bad_inn_glare_272_v2.jpg | bad | good | 0.377036 | [Original JPEG copy](https://drive.google.com/file/d/1myb2jWbB1X7c8EzVU540F9pvk2up9PI1/view) |
| bad_inn_glare_282_v3.jpg | bad | good | 0.492901 | [Original JPEG copy](https://drive.google.com/file/d/1s3t4apAS4h3JEGkVoUQkpA8sDsjvE150/view) |
| bad_inn_glare_292_v1.jpg | bad | good | 0.253009 | [Original JPEG copy](https://drive.google.com/file/d/1YV8Lxj4fWow8X5dbHB4876ByGgxiClNQ/view) |

The course document requests at least five errors per model, but only 3 genuine errors exist. We have not fabricated any. Image copies are kept privately in Drive rather than published to GitHub due to the document-image content.
