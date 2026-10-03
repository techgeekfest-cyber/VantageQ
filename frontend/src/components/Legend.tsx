import { LAYERS, type LayerDef } from "@/lib/layers";

function Swatch({ layer }: { layer: LayerDef }) {
  const c = layer.color;
  switch (layer.swatch) {
    case "fill":
      return <span className="vq-swatch" style={{ background: c, opacity: 0.6, border: `1px solid ${c}` }} />;
    case "line":
      return <span className="vq-swatch vq-swatch-line" style={{ borderTopColor: c }} />;
    case "dashed":
      return <span className="vq-swatch vq-swatch-line" style={{ borderTopColor: c, borderTopStyle: "dashed" }} />;
    case "circle":
      return <span className="vq-swatch vq-swatch-circle" style={{ background: c }} />;
    case "star":
      return <span className="vq-swatch vq-swatch-star" style={{ color: c }}>●</span>;
  }
}

interface Props {
  visibility: Record<string, boolean>;
  onToggle: (id: string) => void;
}

/** Legend and layer control in one: each entry is a toggle. */
export default function Legend({ visibility, onToggle }: Props) {
  return (
    <fieldset className="vq-legend" aria-label="Map layers">
      <legend>Layers</legend>
      {LAYERS.map((l) => (
        <label key={l.id} className="vq-legend-item">
          <input type="checkbox" checked={Boolean(visibility[l.id])} onChange={() => onToggle(l.id)} />
          <Swatch layer={l} />
          <span>{l.label}</span>
        </label>
      ))}
    </fieldset>
  );
}
