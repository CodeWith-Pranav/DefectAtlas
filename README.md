# DefectAtlas

A non-commercial visual anomaly-inspection lab with a trained denoising convolutional autoencoder, a frozen pretrained-feature baseline, response heatmaps and a browser-local human review trail. Category: **MVTec AD bottle only**.

## Run

Node 22: `npm ci --onnxruntime-node-install=skip`, `npm test`, `npm start`. Open http://localhost:3000. ONNX Runtime executes the exported CNNs on CPU; sharp handles image decoding. No paid AI API, database, secrets or Python runtime is needed to serve the app.

## What is working

- Upload PNG/JPEG/WebP or select an attributed benchmark example.
- Run either actual CNN and inspect the raw anomaly score, validation threshold, score/threshold ratio and a coarse 12×12 model-response overlay.
- View image AUROC, AP, recall, false alarms, defect-type results and the autoencoder loss history.
- Save up to 30 inspections in this browser, add a human decision and notes, preserve a review-change history and compare reviewer/model disagreements.
- Export reviews with normalized thumbnails and response maps, or export benchmark evidence as JSON.
- Review provenance, training/validation split, preprocessing, model limitations and licensing in the model card.

## Training and measured evidence

All 209 official healthy training images are split by seed 17 into **167 fitting and 42 healthy-validation images**. The autoencoder trains for 35 epochs with training-only Gaussian denoising noise (σ0.02); the lowest healthy-validation MSE selects epoch 35. Its encoder and decoder have 16,683 parameters. The fixed first four MobileNetV3 Small ImageNet feature blocks have 10,488 parameters; normal fitting images provide per-location feature means/stds. These pretrained weights were not trained by this project.

Images are RGB at 96×96. A 12×12 anomaly map is generated from pooled reconstruction error or mean squared standardized pretrained-feature deviations. The image score averages its largest eight values. Each model's threshold is fixed at the 95th percentile of healthy-validation image scores; the overlay scale is the 99th percentile of healthy-validation map values. No test defect labels select a checkpoint, model or threshold. The pretrained baseline is the predetermined UI default.

Held-out test: **83 images, 63 defective and 20 healthy**. The test is defect-heavy (75.9% prevalence), so its AP should not be interpreted as a production estimate.

| Model | AUROC | AP | Defects caught | Healthy false alarms |
| --- | --- | --- | --- | --- |
| Autoencoder ae-v1 | 0.610 | 0.870 | 36/63 | 7/20 |
| Pretrained spatial baseline feature-v1 | 0.984 | 0.995 | 58/63 | 2/20 |

The weak autoencoder result is retained. Low reconstruction loss does not imply useful anomaly separation. With only 20 healthy test images, the false-positive rate estimate is uncertain. These results validate neither field reliability nor industrial safety.

## Reproduce training

Python 3.12. Install CPU `torch==2.15.0` and `torchvision==0.29.2` from the official PyTorch CPU index (`https://download.pytorch.org/whl/cpu`), then install `requirements-train.txt` in a virtual environment. Run `python download-data.py` from the checkout. It acquires the complete bottle category under `../defectatlas-data`, checks PNG format, records SHA-256 values and compares mirror LFS hashes when present. Expected inventory: 209 train images, 83 test images and 63 masks.

Run `python train.py`. It writes the PyTorch checkpoint, two ONNX exports, model card, labeled resized gallery, explicit split paths, measured results, loss history and Python golden inference fixtures. Then run `npm test` to validate exported inference and preprocessing. Training runs offline and is not repeated during deployment. The legacy TorchScript ONNX exporter is explicitly used for stable opset-17 exports and emits a deprecation notice in current PyTorch.

Do not train on a partial download or retune on test labels. If you iterate after studying test results, reserve new independent data before making a new generalization claim.

## API and storage

- `GET /api/health`: application readiness and version.
- `GET /api/report`: frozen metrics, provenance and labeled gallery.
- `POST /api/inspect`: `{ "model":"feature-v1", "image":"data:image/png;base64,..." }`; returns score, threshold, ratio, review flag, normalized thumbnail, 12×12 normalized map and server-processing time.

Image formats are bounded PNG/JPEG/WebP, maximum 1.8 MB decoded bytes and 16 megapixels; request body limit 2.8 MB. The browser limits original uploads to 8 MB and downsizes to at most 512 pixels before transfer. Server preprocessing center-samples a 96×96 nearest-neighbor RGB grid after orientation. Non-square inputs emit a warning. Input validation/origin checking is not authentication or a comprehensive rate-limiting system.

Application code does not persist submitted images on the server. Browser-local reviews use localStorage and do not sync across devices. A full queue stops saving new inspections rather than silently discarding old work. Export backups before clearing browser data. Hosting providers may retain request metadata; use only non-confidential images.

## Scope and limits

This is a one-category low-resolution research demonstration. Phone images, other products, novel lighting/backgrounds or rotated views may be out of distribution. Scores/ratios are not calibrated defect probabilities. The response overlay is not a verified segmentation; pixel-level localization quality is not evaluated. Human review labels are not ground truth and are not used for retraining. No production traffic, safety validation, throughput or downtime-savings claim is made.

## Deployment

Vercel only. Existing project settings: no framework preset, output directory `public`, Node 22, install command `npm ci --omit=dev --onnxruntime-node-install=skip`. The Node API includes both ONNX files and model card. Non-Linux-x64 ONNX binaries are excluded to keep the function small. GitHub Actions installs the pinned CPU runtime and executes the checks; training does not run in CI.

See SPEC.md, ARCHITECTURE.md, BEGINNER-GUIDE.md and PROJECT-STATUS.md.

## Attribution

MVTec AD: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, CVPR 2019. https://www.mvtec.com/research-teaching/datasets/mvtec-ad . Dataset and adapted examples: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/ . Non-commercial research/portfolio use. Model artifacts are distributed for non-commercial educational/research use with the applicable dataset attribution. Read LICENSE-NOTICE.md and the retained torchvision license. MVTec and pretrained-model authors do not endorse this project.
