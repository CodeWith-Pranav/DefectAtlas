# DefectAtlas: explain what you built

## The problem

A factory may have many healthy product images and few labeled defects. A normality model learns what healthy images look like and flags unusual responses for a human. This app demonstrates that approach on one public bottle benchmark. It does not certify defects on arbitrary products.

## The two models

**Autoencoder:** a convolutional encoder compresses an image; a decoder reconstructs it. It trains on healthy images with a little noise added to its inputs. High reconstruction error can suggest something unusual, but it can also reflect harmless variation. This simple autoencoder is weak on the test set; keep that result and explain it.

**Pretrained baseline:** early MobileNetV3 Small convolutions already contain visual filters learned on ImageNet. We freeze those weights and fit normal mean/std statistics at each feature-map position. Unusually large feature deviations produce the anomaly response. It is a simple spatial Gaussian baseline, not PatchCore, a large foundation model or a generative AI API.

## Trace one inspection

1. The user selects an example or uploads an image. The browser reduces uploaded images before sending them.
2. The same-origin Node API validates and decodes the image, orients RGB pixels and samples 96×96 inputs.
3. ONNX Runtime runs the selected CNN. Reconstruction residuals or standardized feature distances create a 12×12 map.
4. The largest eight values are averaged into an image score. The model's frozen healthy-validation threshold determines whether to flag the image.
5. The browser draws an overlay using a fixed validation-derived scale and saves the result locally if storage is available.
6. A human can accept, correct or mark the result uncertain. The review history is editable but does not retrain the models.

A raw score of 5.0 or a ratio of 2× threshold is not a 50%, 100% or 200% failure probability. The score's units differ between models.

## Demonstration

Select a healthy benchmark example and run the pretrained baseline, then choose a broken-large example. Explain the known test labels shown in the gallery; do not present them as fresh user validation. Change overlay opacity to inspect the same input beneath the coarse response. Switch to the autoencoder and rerun. Compare the measured AUROC and false alarms in Benchmark. Open Review queue, add a human decision and notes, save it and export the review record. Then inspect Model & data for training provenance and limitations.

## Interview questions you should understand

- Why healthy-only training? Anomalies are diverse and may not have labels. Our threshold still needs held-out normal data to set an acceptable review policy.
- Why train/validation/test separation? Training fits weights/statistics; validation selects the checkpoint and threshold; test reports final evidence. Inspecting test results repeatedly changes how honestly you can claim independence in future work.
- What does transfer learning mean here? We reuse frozen ImageNet filters. The autoencoder's weights are trained here; MobileNet's original ImageNet training is not our contribution.
- Why did the autoencoder fail? Reconstruction training optimizes pixel similarity, not anomaly separation. A small low-resolution model may blur defects, reconstruct anomalies or respond strongly to harmless variation.
- Why show AUROC and AP? They assess ranking without choosing one threshold. AP depends strongly on anomaly prevalence; this test has 63 defects in 83 images. Report confusion counts at the chosen threshold too.
- Why are heatmaps not segmentations? We show a low-resolution response grid. No pixel-level mask metric or production localization assessment was run.
- Why test Python versus ONNX? Exported models and preprocessing must represent the same computation. Golden tensors verify inference parity; exact 96px gallery PNGs verify the pixel channel/scaling path.
- Where is the database? Results are stored in browser localStorage. There is no account system or cross-device sync; a real multi-reviewer deployment would require access control, durable storage and retention rules.
- What does latency mean? Our number measures server preprocessing/inference and may include cold start. It excludes the network and does not demonstrate throughput under load.

## Run and change it

Node 22 is enough to serve: install from package-lock, run tests, then `npm start`. Python is only for reproducing training. Edit `public/index.html` for interface and local review behavior, `inference.mjs` for preprocessing/inference, `api/index.mjs` for HTTP rules and `train.py` for offline model experiments. Keep version names and source artifacts consistent after retraining.

A good personal contribution is evaluating a second category independently, measuring localization against the supplied masks, or replacing spatial diagonal statistics with a rigorously evaluated memory-bank method. Use new validation/test discipline and measure improvements; do not inflate the current one-category evidence.

Be candid about assistance and explain the code you reviewed and changed. Do not invent production users, industrial deployments or commercial rights to the dataset.
