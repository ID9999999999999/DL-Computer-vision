# Task 5D — requirement audit (9 November 2026)
| ID | Current status | Evidence |
|---|---|---|
| D01 | ✅ Done | Real ONNX export with SHA verification |
| D02 | ✅ Optional quantization not used; risks documented | Deployment memo explains model-size/performance and confidence risks |
| D03 | ✅ Done | PT 2,963,154 bytes; ONNX 5,767,752 bytes |
| D04 | ✅ Done | Same local CPU, 4 threads, 20 warmups + 100 timed forward passes per runtime: PT 6.104ms; ONNX 2.451ms |
| D05 | ✅ Done | Original 270 Test, SHA verified; 267/270 PT = 267/270 ONNX; zero drop and 0 label disagreements |
| D06 | ✅ Done | [Measured deployment memo](../reports/bonus_deployment_memo.md) |
| D07 | ✅ Done (artifact prepared; teacher acceptance unverified) | [ZIP with ONNX, CSV, JSON, logs and executed source code](https://drive.google.com/file/d/1zIEXnesFzi4JPrzxuNRbYaemqEUEbQ0w/view) |

**Task 5D technically completed.** The original data and checkpoint remain private/unaltered. Forward-only CPU results are not end-to-end or edge hardware benchmarks.
