> Historical preparation note. Training completed on 25 October 2026. See README.md and results/Final_Report.md for the current submission status.

# Verification record

Prepared on 19 October 2026. No real model has been trained in this environment.

Passed: Python source parsing and compilation of all notebook code cells; eight automated protocol/report tests. These cover exact grouped allocation, deterministic seeds, rejection of malformed source groups, the bad-positive metric convention, validation-only model selection, failed-acceptance status, path containment, pixel/hash checks, original-metadata preservation, detection of changed images, and report/chart generation on artificial temporary fixtures.

The actual supplied metadata was inspected: 1,800 records, 300 sources, seed 2473986. The prepared split has 1,260 training, 270 validation and 270 test records with zero source-group overlap.

Not executed here: package installation in the target environment, pretrained-weight downloads, TensorFlow training, Ultralytics training, inference on the user's images, or hardware latency measurements. TensorFlow, PyTorch and Ultralytics were unavailable in this workspace. The integration is prepared code and requires a real local run; errors must be corrected using its log if any occur.

No artificial predictions, plots or experiment results are included as real results. Test fixtures were created in temporary directories and removed.

Recheck protocol/report tests (after installing requirements):

```bash
python -m unittest discover -s tests -v
```
