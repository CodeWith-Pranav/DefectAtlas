# DefectAtlas specification

A non-commercial visual anomaly-inspection research app for the MVTec AD bottle category. Original benchmark: 209 healthy training images, 83 test images (20 healthy and 63 defects). No claim to support every product category.

## Required behavior

- Train a small denoising convolutional autoencoder from healthy images only; select checkpoint by healthy validation loss.
- Compare with frozen early ImageNet MobileNetV3 Small features and a spatial diagonal-Gaussian normal model.
- Derive independent model thresholds from healthy validation images; reserve test defect labels for final reporting.
- Upload or select an attributed example, return actual model anomaly score and 12×12 response overlay, identify processing and threshold policies.
- Store up to 30 inspections locally, record human decisions and review history, show model/reviewer disagreements, export reviews and benchmark evidence.
- Report image AUROC, AP, recall, false alarms and defect-type counts, training history, dataset/split provenance and inference latency scope.
- Public GitHub, Vercel-only hosting, reproducible training and tests. No external paid AI API.

## Limits

Scores are not calibrated defect probabilities. Heatmaps are model responses, not validated masks or precise segmentations. Small low-resolution category-specific evidence cannot establish field reliability or machine safety. Browser-local review records do not synchronize across devices or train the models. MVTec data/derived examples carry CC BY-NC-SA 4.0 attribution and terms.
