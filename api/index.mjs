import { inspect, report } from "../inference.mjs";
export default {
  async fetch(req) {
    const url = new URL(req.url),
      headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
      },
      json = (d, status = 200) => Response.json(d, { status, headers });
    if (req.method === "GET") {
      if (url.pathname === "/api/health")
        return json({ status: "ok", app: "DefectAtlas", version: "1.0.0" });
      if (url.pathname === "/api/report") return json(report());
      return json({ error: "Not found" }, 404);
    }
    if (req.method !== "POST")
      return json({ error: "Method not allowed" }, 405);
    if (url.pathname !== "/api/inspect")
      return json({ error: "Not found" }, 404);
    if (req.headers.get("origin") && req.headers.get("origin") !== url.origin)
      return json({ error: "Foreign origin rejected" }, 403);
    try {
      const reader = req.body?.getReader();
      if (!reader) throw Error("JSON body required");
      let size = 0,
        body = "";
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        size += value.length;
        if (size > 2800000) {
          await reader.cancel();
          return json({ error: "Request exceeds 2.8 MB" }, 413);
        }
        body += decoder.decode(value, { stream: true });
      }
      body += decoder.decode();
      const input = JSON.parse(body);
      if (!input || typeof input !== "object" || Array.isArray(input))
        throw Error("JSON object required");
      return json(await inspect(input.image, input.model));
    } catch (e) {
      return json({ error: e.message }, 400);
    }
  },
};
