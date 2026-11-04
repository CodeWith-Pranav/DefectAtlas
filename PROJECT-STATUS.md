# DefectAtlas checkpoint

Implemented and trained: bottle-category denoising CNN autoencoder and pretrained MobileNet spatial baseline; complete 209/83 official train/test inventory, split into 167 fitting and 42 healthy validation observations. 355 dataset PNG files downloaded and hashed. Training/export completed, selecting autoencoder epoch 35 by validation MSE.

Measured test results: feature-v1 AUROC0.984, TP58/FN5/FP2/TN18; ae-v1 AUROC0.610, TP36/FN27/FP7/TN13. Thresholds derive only from healthy validation scores. Both results and the defect-heavy test prevalence are disclosed.

Validation: 11 automated tests passed, including exported Python/ONNX score comparisons, pixel preprocessing, frozen confusion counts, complete split provenance, image/request bounds and actual API inference. UI JavaScript syntax check passed. Dedicated browser QA is pending.

Resources: public GitHub CodeWith-Pranav/DefectAtlas was created once; Vercel project defectatlas, prj_Td5f8saX3fIcxX4Lm7HenxmYujjv, created once with output public and Node22. Source upload, Git connection, CI execution, deployment and browser QA remain pending. Do not recreate these resources.

Scope/limits: non-commercial MVTec AD bottle demo under CC BY-NC-SA 4.0; no calibrated probabilities, industrial safety or validated segmentation. Review records are browser-local and do not sync across devices. Preserve SPEC.md, ARCHITECTURE.md and README.md contracts when resuming.
