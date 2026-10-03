"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import KpiCards from "./KpiCards";
import Legend from "./Legend";
import { LimitationsPanel, MethodologyPanel } from "./MethodologyPanel";
import SettlementPanel from "./SettlementPanel";
import { DEMO_BASE, kpis, loadDashboardData, MISSING_DATA_MESSAGE, settlementsSorted } from "@/lib/data";
import { defaultVisibility } from "@/lib/layers";
import type { LoadResult } from "@/lib/types";

const FloodMap = dynamic(() => import("./FloodMap"), { ssr: false });

const fmtDate = (iso: string) =>
  new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });

export default function Dashboard({ base = DEMO_BASE }: { base?: string }) {
  const [result, setResult] = useState<LoadResult | null>(null);
  const [visibility, setVisibility] = useState(defaultVisibility);

  useEffect(() => {
    loadDashboardData(base).then(setResult);
  }, [base]);

  if (!result) return <main className="vq-state">Loading Trishuli demo data…</main>;
  if (result.status !== "ok") {
    return (
      <main className="vq-state vq-state-error" role="alert">
        <h1>VantageQ</h1>
        <p>{result.status === "missing" ? MISSING_DATA_MESSAGE : `Demo data could not be read: ${result.message}`}</p>
        {result.status === "missing" ? <p className="vq-muted">Missing: {result.missing.join(", ")}</p> : null}
      </main>
    );
  }

  const data = result.data;
  const m = data.manifest;
  return (
    <div className="vq-app">
      <header className="vq-header">
        <div>
          <h1>VantageQ</h1>
          <p className="vq-tagline">Satellite-Powered Flood Intelligence for Disaster Response</p>
        </div>
        <dl className="vq-meta">
          <div><dt>AOI</dt><dd>Trishuli Valley, Nepal</dd></div>
          <div><dt>Event date</dt><dd>{fmtDate(m.event_date)}</dd></div>
          <div><dt>Post-event Sentinel-1</dt><dd>{fmtDate(m.scenes.post.datetime)}</dd></div>
          <div><dt>Model</dt><dd>Kuro Siwo-trained U-Net</dd></div>
        </dl>
      </header>
      <p className="vq-banner">
        Research prototype. Results are indicators derived from an unvalidated model prediction and pre-event
        OpenStreetMap, not confirmed damage or isolation.
      </p>
      <main className="vq-main">
        <section className="vq-map-wrap" aria-label="Map">
          <FloodMap data={data} visibility={visibility} />
          <Legend visibility={visibility} onToggle={(id) => setVisibility((v) => ({ ...v, [id]: !v[id] }))} />
        </section>
        <aside className="vq-side">
          <KpiCards items={kpis(data)} />
          <SettlementPanel settlements={settlementsSorted(data)} />
          <MethodologyPanel data={data} />
          <LimitationsPanel />
        </aside>
      </main>
      <footer className="vq-footer">
        Map data © OpenStreetMap contributors (ODbL), historical snapshot {data.infrastructure.osm_snapshot}.
        Contains modified Copernicus Sentinel data 2026. Model trained on Kuro Siwo (CC BY 4.0). Educational
        prototype for the IIT Mandi Multimodal AI Hackathon 2026.
      </footer>
    </div>
  );
}
