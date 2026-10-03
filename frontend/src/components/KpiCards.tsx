import type { Kpi } from "@/lib/data";

export default function KpiCards({ items }: { items: Kpi[] }) {
  return (
    <section className="vq-kpis" aria-label="Key figures">
      {items.map((k) => (
        <div key={k.id} className="vq-kpi" data-testid={`kpi-${k.id}`}>
          <div className="vq-kpi-value">{k.value}</div>
          <div className="vq-kpi-label">{k.label}</div>
          <div className="vq-kpi-detail">{k.detail}</div>
        </div>
      ))}
    </section>
  );
}
