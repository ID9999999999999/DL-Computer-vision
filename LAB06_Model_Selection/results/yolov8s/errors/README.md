# yolov8s — individual test errors

The model produced **2** true misclassifications on the original 270-image held-out test set. These are not synthetic additions or examples from another split.

[Open its private Google Drive error-image folder](https://drive.google.com/drive/folders/1ROo4pmzIXx1YHYPheFKYvXc7nuXxpuyD).

| Image | True label | Predicted label | P(bad) | Photo |
|---|---|---|---:|---|
| bad_passport_glare_032_v2.jpg | bad | good | 0.438321 | [Original JPEG copy](https://drive.google.com/file/d/198jfAD_wNjToSw2KxBI3qY3nSkmi7YFX/view) |
| bad_passport_glare_287_v2.jpg | bad | good | 0.436832 | [Original JPEG copy](https://drive.google.com/file/d/1e1wFjW7HTnqZUM8csdu7R7tTPsF04aCr/view) |

The course document requests at least five errors per model, but only 2 genuine errors exist. We have not fabricated any. Image copies are kept privately in Drive rather than published to GitHub due to the document-image content.
