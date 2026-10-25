# Lab 06: Model selection for document quality classification

Seed **2473986**. Experiment completed in Colab on 25 October 2026.

## Start here

1. Read [the final report](results/Final_Report.md).
2. Open [Review_Results.ipynb](Review_Results.ipynb) for executed verification and plots. It uses saved outputs and requires no GPU, images or model downloads.
3. Present [Lab06_Presentation.pptx](presentation/Lab06_Presentation.pptx). The first eight slides take about five minutes. Three appendix slides support questions. Speaker notes contain the English script.
4. Read [Model_Selection_Memo.md](docs/Model_Selection_Memo.md) for the pre-run choices and architecture references.

## Reproduce the experiment

The image dataset and model weights are external to this repository. Preserve the original `demo_quality_dataset` with `metadata.jsonl`, `good/` and `bad/`, and the four best model files in the original Drive output. The review archive does not contain them.

From this folder, install `requirements.txt` in a dedicated environment. Exact versions from the completed run are in `results/environment.json` and `results/environment.txt`. GPU packages may need platform-specific installation.

```bash
python -m pip install -r requirements.txt
python lab06.py run --dataset /path/to/demo_quality_dataset --out /path/to/NEW_lab06_results
```

Always use a **new output folder** for a new experiment. Do not point a training run at the packaged `results/` snapshot. `Lab06.ipynb` is the unexecuted training workflow. The original training happened in Colab through the same CLI and its evidence is in `results/run.log` and the model histories. The results-review notebook executes only analysis and displays the real saved plots.

## Main findings

MobileNetV2 achieves 99.63% test accuracy with one error out of 270 images. The predefined validation-and-cost rule selects YOLOv8n-cls, the fastest passing pipeline at 26.97 ms/image on CPU. All observed test mistakes involve glare. These are results on a small synthetic dataset, not proof of deployment robustness.

## Files

| Location | Purpose |
|---|---|
| `lab06.py`, `config.json` | Training, evaluation and fixed protocol |
| `Lab06.ipynb` | Unexecuted training workflow |
| `Review_Results.ipynb` | Executed analysis of completed results |
| `docs/` | Rationale, defense preparation and verification |
| `results/` | Measured metrics, raw probabilities, histories, logs and plots |
| `presentation/` | Editable slides and five-minute script |
| `tests/` | Protocol and selection tests |

Model selection uses validation metrics and cost only. Test characterization uses fixed threshold 0.5 and never triggers reselection. The original automatically generated report remains as `results/Final_Report_Original.md`. The revised report adds interpretations and links without changing measured results.
