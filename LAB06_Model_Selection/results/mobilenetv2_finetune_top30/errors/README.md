# Task 5A — genuine false accept

The fine-tuned MobileNetV2 produced exactly **one** error on the 270-image test set: `bad_passport_glare_082_v1.jpg`, true label bad, predicted good. The original JPEG is preserved privately in [Google Drive](https://drive.google.com/drive/folders/1jmTF4e2BTRrW283JQWU2SdzB9uMXtgvd); it is not published as a GitHub image because the experiment involves document imagery. Its prediction and probability are recorded in [errors.json](../errors.json). Neither an artificial fifth error nor synthetic test predictions were added.
