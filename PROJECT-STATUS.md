# DefectAtlas checkpoint — publication complete

## Completed implementation

Full MVTec AD bottle dataset acquired (355 PNG files including masks):209 healthy training,83 test. Fitting/healthy-validation split167/42, seed17. Denoising convolutional autoencoder trained35 epochs, checkpoint chosen by healthy-validation MSE. Frozen ImageNet MobileNet early-feature spatial baseline, exported ONNX models, reproducibility source, manifest, golden fixtures and licenses included.

Test results: feature-v1 AUROC0.984, TP58/FN5/FP2/TN18; ae-v1 AUROC0.610, TP36/FN27/FP7/TN13. Validation-only thresholds; no test-label model/threshold selection. Weak autoencoder result is retained. The defect-heavy test prevalence and small healthy test set are disclosed.

11 automated tests passed locally and in GitHub Actions. Live checks covered both CNNs, upload, model response overlays, local inspection storage, review edits/history, reload persistence, benchmark/model-card evidence and export actions. A detected startup navigation race was fixed and checked live. See VERIFICATION.md for details and limits.

## Resources and deployment

Public GitHub: CodeWith-Pranav/DefectAtlas. Vercel project defectatlas, prj_Td5f8saX3fIcxX4Lm7HenxmYujjv, is connected to the repository; production URL https://defectatlas.vercel.app/. Startup-fix commitd7ce3c5e42cc8f6396ddda8dad20d1d72ad61c45 deployed READY as dpl_DXhZd4kF1KczCPRgRbzY3sYKxttB. Later documentation-only commits update provenance and trigger the normal pipeline. Do not recreate resources, retrain models or rerun unchanged local checks when resuming; first inspect current remote commit and deployment.

No required implementation milestone remains. Optional future work: dedicated mobile/accessibility QA, additional independently evaluated product categories, pixel-mask localization metrics or shared authenticated review storage.

## Preserve scope

Non-commercial MVTec AD bottle research demo under CC BY-NC-SA4.0. No calibrated probabilities, industrial safety, full15-category support or validated segmentation. Reviews are browser-local and do not sync across devices or retrain the models. Read SPEC.md, ARCHITECTURE.md, README.md and BEGINNER-GUIDE.md before changes.
