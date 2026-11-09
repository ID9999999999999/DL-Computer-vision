# Task 5E — Requirement Check (9 November 2026)

| Instructor criterion | Status | Evidence |
| --- | --- | --- |
| Re-run one peer experiment from the README | **Partially satisfied** | The peer's original inference script ran unchanged on the 270-image Dataset A test set; the original training experiment could not be reproduced from the supplied package. |
| Concrete audit gaps | **Documented** | [Audit note](../reports/bonus_audit_note.md): absent original training code, dataset/splits and exact environment lock |
| State what worked and what had to be chosen | **Documented** | Working inference run, exit code 0, choice of substitute held-out test dataset, recorded environment |
| Explain the first obstacle | **Documented** | No inference runtime failure; missing training artifacts are the first obstacle to full original-experiment reproduction |
| Complete own model card | **Done** | [Model card](../reports/model_card.md): intended use, training data, metrics, latency, size, limitations and privacy/ethics |
| Address synthetic-to-real gap | **Done, with limitation** | Course reference separated from unmeasured model-specific gap |
| Reflect on and repair own README | **Done (documentation)** | [Project README](../README.md), Windows launcher and run-directory isolation updated |
| Submit to instructor / receive acceptance | **Not verified** | Files are in the repository; no instructor submission receipt or evaluation confirmed |
| Explicit peer permission to re-run | **Not independently verified** | Peer shared the inference package, but formal permission to reproduce and public redistribution are not documented; no peer weights republished |

**Assessment boundary:** the two deliverables are prepared, but full original-experiment reproduction, formal peer permission and instructor acceptance cannot be claimed without evidence. No new training or classmate Dataset B exchange is required for these written reports.
