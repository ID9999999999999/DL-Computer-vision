# Source scripts — compatibility directory

The original working CLI is kept unchanged at [../lab06.py](../lab06.py), with local launcher [../run_local.py](../run_local.py). The original [training notebook](../Lab06.ipynb) is an **unexecuted** workflow; original training happened separately on Colab, with measurements and logs preserved in [../results](../results).

Do not falsely label a copied or newly created wrapper as the actually executed training script. The original exact invocation is described in the project README. The detailed instructor specification of `scripts/train_yolo_quality.py` was not the structure used for the October 25 run.


## Added 9 November 2026

- `train_yolo_quality.py`: expected Task 3 script path; verify-only mode by default when specified, or explicitly `--train` for **a new run** with the instructor's 50 epochs / 32 batch / patience 5 / 640 pixels. Historical trained weights retain their original settings.
- `collect_results.py`: recomputes a four-model table solely from the archived standardized `results.json` files; does not alter original metrics.
- `tests/test_submission_tools.py`: tested on synthetic fixtures, **not** an end-to-end 640-pixel training run.
