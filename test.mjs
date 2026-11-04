import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import sharp from "sharp";
import {
  card,
  preprocess,
  inferTensor,
  inspect,
  report,
  validModel,
} from "./inference.mjs";
import api from "./api/index.mjs";
const golden = JSON.parse(
  readFileSync(new URL("./golden.json", import.meta.url)),
);
test("exported CNNs reproduce Python image scores", async () => {
  for (const g of golden) {
    const b = Buffer.from(g.tensor, "base64"),
      x = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
    for (const model of ["ae-v1", "feature-v1"]) {
      const r = await inferTensor(x, model);
      assert.ok(
        Math.abs(r.score - g.scores[model]) <
          Math.max(1e-5, Math.abs(g.scores[model]) * 1e-4),
        model + ": " + r.score + " vs " + g.scores[model],
      );
      assert.equal(r.map.length, 144);
      assert.ok(r.map.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
      assert.equal(r.flagged, r.score >= r.threshold);
    }
  }
});
test("96px gallery preprocessing preserves PNG pixel values", async () => {
  const g = card.gallery[0],
    p = await preprocess(g.image),
    raw = await sharp(Buffer.from(g.image.split(",")[1], "base64"))
      .raw()
      .toBuffer();
  for (let y = 0; y < 96; y++)
    for (let x = 0; x < 96; x++)
      for (let c = 0; c < 3; c++)
        assert.ok(
          Math.abs(
            p.tensor[c * 9216 + y * 96 + x] - raw[(y * 96 + x) * 3 + c] / 255,
          ) < 3e-8,
        );
});
test("PNG inspection returns computed response and normalized thumbnail", async () => {
  const r = await inspect(card.gallery[0].image, "feature-v1");
  assert.equal(r.category, "bottle");
  assert.ok(Number.isFinite(r.score) && r.score >= 0);
  assert.ok(r.thumbnail.startsWith("data:image/png;base64,"));
  assert.ok(r.inferenceMs >= 0);
});
test("non-square images warn about domain/preprocessing change", async () => {
  const b = await sharp({
    create: { width: 48, height: 96, channels: 3, background: "#777" },
  })
    .png()
    .toBuffer();
  const r = await preprocess("data:image/png;base64," + b.toString("base64"));
  assert.equal(r.tensor.length, 27648);
  assert.equal(r.warnings.length, 1);
});
test("unknown model and malformed or oversized images rejected", async () => {
  assert.throws(() => validModel("fake"));
  for (const image of [
    null,
    "https://example.com/image.png",
    "data:image/svg+xml;base64,AAAA",
    "data:image/png;base64,AAAA",
    "data:image/png;base64," + "A".repeat(2600001),
  ])
    await assert.rejects(() => preprocess(image));
});
test("complete train and test provenance; split disjointness", () => {
  const d = card.dataset;
  assert.equal(d.training + d.validation, 209);
  assert.equal(d.test, 83);
  assert.equal(d.testDefects, 63);
  assert.equal(d.testHealthy, 20);
  assert.equal(new Set([...d.trainFiles, ...d.validationFiles]).size, 209);
  assert.ok(d.trainFiles.every((p) => p.startsWith("train/good/")));
  assert.ok(d.validationFiles.every((p) => p.startsWith("train/good/")));
});
test("frozen test confusion matrices reproduced without inference", () => {
  for (const [model, m] of Object.entries(card.models)) {
    let tp = 0,
      fp = 0,
      tn = 0,
      fn = 0;
    for (const r of card.testRows) {
      const pred = r.scores[model] >= m.threshold;
      if (pred && r.label) tp++;
      else if (pred) fp++;
      else if (r.label) fn++;
      else tn++;
    }
    assert.deepEqual(
      { tp, fp, tn, fn },
      { tp: m.tp, fp: m.fp, tn: m.tn, fn: m.fn },
    );
    assert.equal(
      Object.values(m.perType).reduce((s, r) => s + r.n, 0),
      83,
    );
  }
});
test("report contains public metrics and attribution, not spatial weights", () => {
  const r = report();
  assert.equal(r.featureMean, undefined);
  assert.equal(r.seed, 17);
  assert.equal(r.gallery.length, 8);
  assert.equal(r.dataset.license, "CC BY-NC-SA 4.0");
});
test("API GET health/report and route/method validation", async () => {
  for (const p of ["health", "report"])
    assert.equal(
      (await api.fetch(new Request("https://app.test/api/" + p))).status,
      200,
    );
  assert.equal(
    (await api.fetch(new Request("https://app.test/api/missing"))).status,
    404,
  );
  assert.equal(
    (
      await api.fetch(
        new Request("https://app.test/api/inspect", { method: "DELETE" }),
      )
    ).status,
    405,
  );
});
test("API rejects invalid JSON, foreign origins and large request bodies", async () => {
  const req = (body, headers = {}) =>
    new Request("https://app.test/api/inspect", {
      method: "POST",
      body,
      headers,
    });
  assert.equal((await api.fetch(req("{"))).status, 400);
  assert.equal(
    (await api.fetch(req("{}", { origin: "https://foreign.test" }))).status,
    403,
  );
  assert.equal((await api.fetch(req(" ".repeat(2800001)))).status, 413);
});
test("API POST runs actual model inference", async () => {
  const res = await api.fetch(
    new Request("https://app.test/api/inspect", {
      method: "POST",
      headers: { origin: "https://app.test" },
      body: JSON.stringify({ image: card.gallery[0].image, model: "ae-v1" }),
    }),
  );
  assert.equal(res.status, 200);
  const r = await res.json();
  assert.equal(r.model, "ae-v1");
  assert.ok(Number.isFinite(r.ratio));
});
