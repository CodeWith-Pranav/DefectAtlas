# DefectAtlas architecture

`download-data.py` obtains the complete bottle category, records SHA-256 values and verifies mirror LFS hashes when supplied. The raw images live outside the app checkout and are not all redistributed.

`train.py` deterministically splits official healthy training images into fitting and validation sets with seed 17, trains a small CNN autoencoder and retains its lowest healthy-validation MSE checkpoint. A frozen early MobileNet feature extractor uses ImageNet weights; healthy fitting images provide its spatial means and standard deviations. Both image thresholds use the 95th percentile of healthy-validation scores. Held-out test labels report metrics only.

Artifacts: `autoencoder.pt` (training checkpoint), `autoencoder.onnx`, `features.onnx`, `model-card.json` (statistics, provenance, metrics and labeled resized examples), `golden.json` (Python reference tensors/scores), `data-manifest.json` (raw-file hashes). Neural features and reconstruction are real model outputs.

`inference.mjs` decodes bounded PNG/JPEG/WebP inputs with sharp, orients RGB data and samples a 96×96 grid. ONNX Runtime executes on CPU. Anomaly maps are 12×12; image score averages the largest eight map values. Heat intensity uses a fixed healthy-validation reference, not per-image contrast normalization. Sessions may be reused in memory; user records are never kept on the server.

`api/index.mjs` exposes GET health/report and POST inspect. It enforces image/body/decoded-pixel limits and origin validation, but this is a public demo without user authentication. Origin validation is not a complete abuse-prevention mechanism.

`public/index.html` provides the responsive studio, benchmark and model card, image upload and localStorage review queue. Uploads are reduced to at most 512 pixels in the browser before transfer. Saved results use normalized thumbnails and preserve model version, score, response map and review trail; exports provide a portable backup. There is no cross-device sync.

Vercel serves only public assets and the Node function. Native ONNX Linux x64 binaries are retained; other platforms are excluded from the function bundle. Training remains offline and does not run during deployment.
