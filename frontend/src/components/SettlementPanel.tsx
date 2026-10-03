import { CUT_OFF_NOTE, statusInfo } from "@/lib/format";
import type { Settlement } from "@/lib/types";

const km = (v: number) => `${v.toLocaleString("en-US", { maximumFractionDigits: 2 })} km`;

export default function SettlementPanel({ settlements }: { settlements: Settlement[] }) {
  return (
    <section className="vq-panel" aria-label="Settlements">
      <h2>Settlements ({settlements.length} mapped in OSM)</h2>
      <ul className="vq-settlements">
        {settlements.map((s) => {
          const info = statusInfo(s.status);
          return (
            <li key={s.osm_id} className={`vq-settlement-row vq-tone-${info.tone}`} data-testid={`settlement-${s.osm_id}`}>
              <div className="vq-settlement-head">
                <strong>{s.name ?? `OSM ${s.osm_id}`}</strong>
                {s.name_local ? <span className="vq-muted"> {s.name_local}</span> : null}
                <span className="vq-muted"> · {s.place}</span>
              </div>
              <div className={`vq-badge vq-badge-${info.tone}`}>{info.label}</div>
              <div className="vq-muted">
                Reachable road network: {km(s.reachable_road_km_before)} before → {km(s.reachable_road_km_after)} after
              </div>
            </li>
          );
        })}
      </ul>
      <p className="vq-note">{CUT_OFF_NOTE}</p>
    </section>
  );
}
