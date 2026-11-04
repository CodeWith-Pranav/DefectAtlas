import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import * as ort from "onnxruntime-node";
export const card = JSON.parse(
  readFileSync(new URL("./model-card.json", import.meta.url), "utf8"),
);
const sessions = new Map();
async function session(model) {
  if (!sessions.has(model)) {
    const filename = model === "ae-v1" ? "autoencoder.onnx" : "features.onnx";
    sessions.set(
      model,
      ort.InferenceSession.create(
        fileURLToPath(new URL("./" + filename, import.meta.url)),
        {
          executionProviders: ["cpu"],
          intraOpNumThreads: 1,
          interOpNumThreads: 1,
        },
      ).catch((e) => {
        sessions.delete(model);
        throw e;
      }),
    );
  }
  return sessions.get(model);
}
export function validModel(model) {
  if (!["ae-v1", "feature-v1"].includes(model))
    throw Error("Unknown model version");
  return model;
}
export async function preprocess(data) {
  if (
    typeof data !== "string" ||
    data.length > 2600000 ||
    !/^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(data)
  )
    throw Error("Use a PNG, JPEG or WebP image below 1.8 MB after preparation");
  const bytes = Buffer.from(data.slice(data.indexOf(",") + 1), "base64");
  if (bytes.length > 1800000) throw Error("Image exceeds 1.8 MB limit");
  let decoded;
  try {
    decoded = await sharp(bytes, {
      limitInputPixels: 16000000,
      animated: false,
    })
      .rotate()
      .toColourspace("srgb")
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
  } catch {
    throw Error(
      "Image could not be decoded, or exceeds the 16 megapixel limit",
    );
  }
  const { width, height, channels } = decoded.info;
  if (width < 16 || height < 16 || channels !== 3)
    throw Error("Use an RGB-compatible image at least 16 × 16 pixels");
  const x = new Float32Array(3 * 96 * 96),
    rgb = Buffer.alloc(3 * 96 * 96);
  // Exact center-sampled nearest-neighbor grid, matching Pillow NEAREST on square inputs.
  for (let y = 0; y < 96; y++)
    for (let xx = 0; xx < 96; xx++) {
      const sy = Math.min(height - 1, Math.floor(((y + 0.5) * height) / 96)),
        sx = Math.min(width - 1, Math.floor(((xx + 0.5) * width) / 96));
      for (let c = 0; c < 3; c++) {
        const v = decoded.data[(sy * width + sx) * 3 + c];
        x[c * 96 * 96 + y * 96 + xx] = v / 255;
        rgb[(y * 96 + xx) * 3 + c] = v;
      }
    }
  const thumbnail =
    "data:image/png;base64," +
    (
      await sharp(rgb, { raw: { width: 96, height: 96, channels: 3 } })
        .png()
        .toBuffer()
    ).toString("base64");
  return {
    tensor: x,
    thumbnail,
    dimensions: { width, height },
    warnings:
      Math.abs(width / height - 1) > 0.1
        ? [
            "Image stretched to square; this differs from aligned bottle benchmark images.",
          ]
        : [],
  };
}
export async function inferTensor(tensor, model) {
  validModel(model);
  if (!(tensor instanceof Float32Array) || tensor.length !== 27648)
    throw Error("Expected a 1×3×96×96 float32 tensor");
  const engine = await session(model),
    output = await engine.run({
      image: new ort.Tensor("float32", tensor, [1, 3, 96, 96]),
    });
  const m = new Float64Array(144);
  if (model === "ae-v1") {
    const rec = output.reconstruction.data;
    for (let y = 0; y < 96; y++)
      for (let x = 0; x < 96; x++) {
        let v = 0;
        for (let c = 0; c < 3; c++) {
          const i = c * 9216 + y * 96 + x;
          v += (rec[i] - tensor[i]) ** 2 / 3;
        }
        m[Math.floor(y / 8) * 12 + Math.floor(x / 8)] += v / 64;
      }
  } else {
    const f = output.features.data;
    const channels = card.featureShape[0];
    for (let i = 0; i < 144; i++)
      for (let c = 0; c < channels; c++) {
        const n = c * 144 + i;
        m[i] +=
          ((f[n] - card.featureMean[n]) / card.featureStd[n]) ** 2 / channels;
      }
  }
  const score =
    [...m]
      .sort((a, b) => b - a)
      .slice(0, 8)
      .reduce((a, b) => a + b, 0) / 8;
  const threshold = card.models[model].threshold,
    scale = card.models[model].pixelScale;
  return {
    model,
    version: card.version,
    category: "bottle",
    score,
    threshold,
    ratio: score / threshold,
    flagged: score >= threshold,
    map: Array.from(m, (v) =>
      Math.max(0, Math.min(1, v / Math.max(scale, 1e-12))),
    ),
    mapSize: 12,
    heatScale: scale,
  };
}
export async function inspect(data, model = card.modelPolicy) {
  validModel(model);
  const start = performance.now(),
    p = await preprocess(data),
    r = await inferTensor(p.tensor, model);
  return {
    ...r,
    thumbnail: p.thumbnail,
    dimensions: p.dimensions,
    warnings: p.warnings,
    inferenceMs: Math.round((performance.now() - start) * 10) / 10,
    latencyScope:
      "Server preprocessing and inference; may include model cold start; excludes network",
  };
}
export function report() {
  return {
    version: card.version,
    seed: card.seed,
    category: card.category,
    models: card.models,
    history: card.history,
    bestEpoch: card.bestEpoch,
    dataset: card.dataset,
    gallery: card.gallery,
    runtime: card.runtime,
    pretrained: card.pretrained,
    thresholdPolicy: card.thresholdPolicy,
    scorePolicy: card.scorePolicy,
    modelPolicy: card.modelPolicy,
  };
}
