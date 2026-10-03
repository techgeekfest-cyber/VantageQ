import type { DashboardData } from "@/lib/types";

const day = (iso: string) => iso.slice(0, 10);

export function MethodologyPanel({ data }: { data: DashboardData }) {
  const { manifest: m, infrastructure: s } = data;
  return (
    <details className="vq-panel" open>
      <summary><h2>Methodology</h2></summary>
      <dl className="vq-dl">
        <dt>Data</dt>
        <dd>
          Sentinel-1 GRD (IW, VV+VH), track {m.scenes.post.relative_orbit} {m.scenes.post.orbit_state}: post{" "}
          {day(m.scenes.post.datetime)}, pre {day(m.scenes.pre1.datetime)} &amp; {day(m.scenes.pre2.datetime)}.
          Historical OSM snapshot {s.osm_snapshot} (pre-event). Model trained on the Kuro Siwo dataset.
        </dd>
        <dt>Model</dt>
        <dd>{m.model.name}; {m.model.input}. Frozen checkpoint, not tuned on Trishuli.</dd>
        <dt>Impact rules (prototype, not calibrated)</dt>
        <dd>
          <ul>
            <li>Roads: ≥ 10 m overlap with predicted flood</li>
            <li>Bridges: intersect predicted flood</li>
            <li>Buildings: &gt; 0.01 m² overlap with predicted flood</li>
            <li>Settlements: simplified road-graph connectivity (vehicle roads)</li>
          </ul>
        </dd>
      </dl>
    </details>
  );
}

export function LimitationsPanel() {
  return (
    <details className="vq-panel" open>
      <summary><h2>Limitations</h2></summary>
      <ul className="vq-limits">
        <li>The Trishuli flood prediction is not validated against ground truth.</li>
        <li>Terrain-related SAR false positives are possible in steep Himalayan valleys.</li>
        <li>A bridge intersecting predicted flood does not establish bridge damage.</li>
        <li>&ldquo;Potentially cut off&rdquo; is a graph indicator, not confirmed isolation.</li>
        <li>Footpaths are excluded from the connectivity graph.</li>
        <li>The road graph is limited to the analysis AOI; access outside it is not considered.</li>
        <li>All results depend on the frozen model&rsquo;s prediction (one post-event image, 2 days after the event).</li>
      </ul>
    </details>
  );
}
