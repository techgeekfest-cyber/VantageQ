// Copy MapLibre's ESM web worker (and the shared module it imports) into public/maplibre/ so the
// static export can serve it. Next's bundler does not emit this worker by itself; without it no
// GeoJSON layer renders. Runs automatically before `npm run dev` / `npm run build`.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "maplibre-gl", "dist");
const dst = join(root, "public", "maplibre");
mkdirSync(dst, { recursive: true });
for (const f of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) copyFileSync(join(src, f), join(dst, f));
console.log("copied MapLibre worker to public/maplibre/");
