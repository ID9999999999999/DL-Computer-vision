# yolov8n — individual test errors

The model produced **3** true misclassifications on the original 270-image held-out test set. These are not synthetic additions or examples from another split.

[Open its private Google Drive error-image folder](https://drive.google.com/drive/folders/1080wwwEPFhfJWwaGpK4fYhDNlGAZAU4I).

| Image | True label | Predicted label | P(bad) | Photo |
|---|---|---|---:|---|
| bad_passport_glare_032_v2.jpg | bad | good | 0.003062 | [Original JPEG copy](https://drive.google.com/file/d/1P3tWXHdlhPoV6ocxhRB3rzuuxvY4Hvwt/view) |
| bad_passport_glare_287_v2.jpg | bad | good | 0.028967 | [Original JPEG copy](https://drive.google.com/file/d/1eiC1awwGA-33MwjcTBUatOICb7DoUC2l/view) |
| bad_snils_glare_022_v1.jpg | bad | good | 0.200910 | [Original JPEG copy](https://drive.google.com/file/d/1Vp3wofplQMrfS-ErJgJa1QME0HINAUy9/view) |

The course document requests at least five errors per model, but only 3 genuine errors exist. We have not fabricated any. Image copies are kept privately in Drive rather than published to GitHub due to the document-image content.
