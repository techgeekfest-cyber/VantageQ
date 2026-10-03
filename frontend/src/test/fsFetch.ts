import fs from "node:fs";
import path from "node:path";

export const PUBLIC_DIR = path.resolve(__dirname, "../../public");

/** fetch() stand-in serving files from public/ (the committed demo snapshot). */
export function fsFetch(overrides: Record<string, unknown> = {}, missing: string[] = []) {
  return async (url: string) => {
    const name = url.split("/").pop() as string;
    if (missing.includes(name)) return { ok: false, status: 404, json: async () => ({}) };
    if (name in overrides) return { ok: true, status: 200, json: async () => overrides[name] };
    const file = path.join(PUBLIC_DIR, url);
    if (!fs.existsSync(file)) return { ok: false, status: 404, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(file, "utf-8")) };
  };
}

export function readDemo(name: string) {
  return JSON.parse(fs.readFileSync(path.join(PUBLIC_DIR, "demo/trishuli", name), "utf-8"));
}
