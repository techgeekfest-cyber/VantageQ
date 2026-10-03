// Wording and popup content. Deliberately never says "damaged": overlap with a predicted flood
// is an indicator only.

import type { Settlement } from "./types";

export const CUT_OFF_NOTE =
  "Potentially cut off = disconnected in the simplified road graph after removing flood-affected edges. " +
  "Not confirmed real-world isolation.";

export const STATUS_INFO: Record<string, { label: string; tone: "alert" | "ok" | "neutral" }> = {
  potentially_cut_off: { label: "Potentially cut off — simplified graph indicator", tone: "alert" },
  connected: { label: "Connected in simplified road graph", tone: "ok" },
  not_on_main_network_before: { label: "Already separate from main road network (before flood)", tone: "neutral" },
  no_road_within_snap_distance: { label: "No mapped road within snap distance", tone: "neutral" },
};

export function statusInfo(status: string) {
  return STATUS_INFO[status] ?? { label: status.replaceAll("_", " "), tone: "neutral" as const };
}

export interface PopupContent {
  title: string;
  rows: [string, string][];
  note?: string;
}

type Props = Record<string, unknown>;
const str = (v: unknown) => (v === null || v === undefined || v === "" ? "—" : String(v));
const num = (v: unknown, unit: string, digits = 0) =>
  typeof v === "number" ? `${v.toLocaleString("en-US", { maximumFractionDigits: digits })} ${unit}` : "—";

export function roadPopup(p: Props): PopupContent {
  return {
    title: p.name ? String(p.name) : `OSM way ${p.osm_id}`,
    rows: [
      ["OSM way", str(p.osm_id)],
      ["Highway type", str(p.highway) + (p.ref ? ` (${p.ref})` : "")],
      ["Total length (in AOI)", num(p.length_m, "m")],
      ["Length inside predicted flood", num(p.affected_length_m, "m", 1)],
      ["Share inside predicted flood", num(p.affected_pct, "%", 1)],
    ],
    note: "Potentially affected road: ≥ 10 m overlap with the predicted flood (prototype rule).",
  };
}

export function bridgePopup(p: Props, intersects: boolean): PopupContent {
  return {
    title: p.name ? String(p.name) : `Bridge, OSM way ${p.osm_id}`,
    rows: [
      ["OSM way", str(p.osm_id)],
      ["Type", `${str(p.highway)} bridge`],
      ["Status", intersects ? "Intersects predicted flood" : "Does not intersect predicted flood"],
      ["Length inside predicted flood", num(p.affected_length_m, "m", 1)],
    ],
    note: intersects
      ? "Overlap with predicted flood only; this does not establish bridge damage. Bridges cross rivers by design."
      : undefined,
  };
}

export function buildingPopup(p: Props): PopupContent {
  return {
    title: `Building, OSM ${str(p.osm_type)} ${str(p.osm_id)}`,
    rows: [
      ["Building tag", str(p.building)],
      ["Footprint area", num(p.footprint_m2, "m²", 1)],
      ["Area inside predicted flood", num(p.flooded_m2, "m²", 1)],
    ],
    note: "Potentially affected: footprint overlaps the predicted flood (> 0.01 m²). Not a damage assessment.",
  };
}

export function settlementPopup(s: Partial<Settlement>): PopupContent {
  const info = statusInfo(String(s.status));
  return {
    title: s.name ? `${s.name}${s.name_local ? ` (${s.name_local})` : ""}` : `Settlement ${s.osm_id}`,
    rows: [
      ["Place type", str(s.place)],
      ["Status", info.label],
      ["Reachable road before → after", `${num(s.reachable_road_km_before, "km", 2)} → ${num(s.reachable_road_km_after, "km", 2)}`],
      ["Distance to nearest road node", num(s.snap_distance_m, "m")],
    ],
    note: s.status === "potentially_cut_off" ? CUT_OFF_NOTE : undefined,
  };
}

export function floodPopup(): PopupContent {
  return {
    title: "Predicted flood extent",
    rows: [["Source", "Frozen Kuro Siwo U-Net, class “flood”"], ["Post-event image", "Sentinel-1, 28 Aug 2026"]],
    note: "Model prediction; not validated against ground truth.",
  };
}

const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string);

export function popupHtml(c: PopupContent): string {
  const rows = c.rows.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("");
  return `<div class="vq-popup"><h4>${esc(c.title)}</h4><table>${rows}</table>${c.note ? `<p>${esc(c.note)}</p>` : ""}</div>`;
}
