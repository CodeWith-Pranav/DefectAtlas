# Verification record

Runtime: Node 22, CPU ONNX Runtime. Training: Python 3.12.14, PyTorch 2.15.0+cpu, seed17. Training and model artifacts were preserved during deployment fixes.

## Automated evidence

11 checks passed locally and in GitHub Actions (e.g. commit0528f26, run41888631094). Coverage: six Python-to-ONNX golden image scores (three tensors × two models); RGB preprocessing of gallery pixels; real PNG/ONNX inspection; non-square warning; invalid/large input; complete train/test inventory and split disjointness; frozen confusion counts; public report shape; API methods/routes/origin/body bounds; real POST inference. These correctness checks do not constitute load testing or a security audit.

## Live browser evidence

- feature-v1 on test/broken_large/000.png: score6.472, threshold4.386, ratio1.48×, flagged; server-processing observation134.9ms (not a performance benchmark).
- Uploaded healthy example using the file picker and ran ae-v1: ratio0.80×, below threshold.
- Both inspections appeared in the local queue. Human normal review and notes saved; the pending→normal history appeared. Two records and the saved review survived reload.
- Reviews and benchmark JSON exports produced completed browser downloads; downloaded bytes were not independently inspected.
- Benchmark view showed the measured AUROC0.984/0.610, per-condition counts and training curve. Model/data view showed 167/42/83 split and seed17.
- A startup navigation race was observed while registry data was loading. Fixed by disabling workspace navigation until initialization completes plus guarding view selection. A fresh live page showed disabled startup controls, then opened Review queue without an application-origin console error. Browser extension metadata errors were present and are unrelated.
- Desktop layout inspected visually. Dedicated mobile viewport, screen-reader, real production-domain images, pixel-level segmentation and concurrency/load tests were not performed.

## Deployment

Public source: https://github.com/CodeWith-Pranav/DefectAtlas . Existing Vercel project prj_Td5f8saX3fIcxX4Lm7HenxmYujjv is Git-linked. Initial config validation required include/exclude glob strings instead of arrays; corrected without model changes. Startup-fix deployment dpl_DXhZd4kF1KczCPRgRbzY3sYKxttB from commitd7ce3c5e42cc8f6396ddda8dad20d1d72ad61c45 reached READY. Public URL https://defectatlas.vercel.app/ . Future pushes run CI and Vercel deployments independently; CI is not configured as a blocking promotion gate.
