import type { NextConfig } from "next";

// Static export: the dashboard is plain HTML/JS reading the committed demo snapshot in
// public/demo/trishuli/. No server, API layer or database.
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
