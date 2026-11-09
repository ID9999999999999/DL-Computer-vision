# mobilenetv2 — individual test errors

The model produced **1** true misclassifications on the original 270-image held-out test set. These are not synthetic additions or examples from another split.

[Open its private Google Drive error-image folder](https://drive.google.com/drive/folders/1IuyyEkGSoXXBZZtQSwitVkSR7tvrmqeK).

| Image | True label | Predicted label | P(bad) | Photo |
|---|---|---|---:|---|
| bad_passport_glare_082_v1.jpg | bad | good | 0.002805 | [Original JPEG copy](https://drive.google.com/file/d/1yCyq72aJweFHDTYkjB93Rxn1jOq276M9/view) |

The course document requests at least five errors per model, but only 1 genuine errors exist. We have not fabricated any. Image copies are kept privately in Drive rather than published to GitHub due to the document-image content.
