// The only place that loads and interprets the demo snapshot. Components receive parsed data and
// derived values from here; no component re-parses JSON or recomputes numbers.

import type {
  ConnectivitySummary, DashboardData, FeatureCollection, InfrastructureSummary, LayerId, LoadResult, Manifest,
  Settlement,
} from "./types";

export const DEMO_BASE = "/demo/trishuli";
export const MISSING_DATA_MESSAGE = "Demo data not found. Run scripts/build_demo_snapshot.py.";

export const LAYER_FILES: Record<LayerId, string> = {
  flood_extent: "flood_extent.geojson",
  roads_reference: "roads_reference.geojson",
  affected_roads: "affected_roads.geojson",
  affected_road_sections: "affected_road_sections.geojson",
  bridges: "bridges.geojson",
  affected_bridges: "affected_bridges.geojson",
  affected_buildings: "affected_buildings.geojson",
  settlements: "settlements.geojson",
};
const JSON_FILES = { manifest: "manifest.json", infrastructure: "infrastructure_summary.json",
  connectivity: "connectivity_summary.json" } as const;

type Fetch = (url: string) => Promise<{ ok: boolean; status: number; json: () => Promise<unknown> }>;

function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** Throws with a readable message if a required field is missing (no silent zeroes). */
function validate(m: Manifest, inf: InfrastructureSummary, con: ConnectivitySummary): void {
  const checks: [string, unknown][] = [
    ["infrastructure.flood_area_km2_utm", inf?.flood_area_km2_utm],
    ["infrastructure.roads.segments_affected", inf?.roads?.segments_affected],
    ["infrastructure.roads.affected_length_km", inf?.roads?.affected_length_km],
    ["infrastructure.bridges.affected", inf?.bridges?.affected],
    ["infrastructure.buildings.affected", inf?.buildings?.affected],
    ["infrastructure.settlements.potentially_cut_off", inf?.settlements?.potentially_cut_off],
  ];
  const bad = checks.filter(([, v]) => !isNum(v)).map(([k]) => k);
  if (!m?.scenes?.post?.datetime) bad.push("manifest.scenes.post");
  if (!Array.isArray(con?.settlements)) bad.push("connectivity.settlements");
  if (bad.length) throw new Error(`Invalid demo data, missing fields: ${bad.join(", ")}`);
}

export async function loadDashboardData(base: string = DEMO_BASE, fetchFn: Fetch = fetch as Fetch): Promise<LoadResult> {
  const names = [...Object.values(JSON_FILES), ...Object.values(LAYER_FILES)];
  let responses;
  try {
    responses = await Promise.all(names.map((n) => fetchFn(`${base}/${n}`)));
  } catch {
    return { status: "missing", missing: names };
  }
  const missing = names.filter((_, i) => !responses[i].ok);
  if (missing.length) return { status: "missing", missing };
  try {
    const parsed = await Promise.all(responses.map((r) => r.json()));
    const byName = Object.fromEntries(names.map((n, i) => [n, parsed[i]]));
    const manifest = byName[JSON_FILES.manifest] as Manifest;
    const infrastructure = byName[JSON_FILES.infrastructure] as InfrastructureSummary;
    const connectivity = byName[JSON_FILES.connectivity] as ConnectivitySummary;
    validate(manifest, infrastructure, connectivity);
    const layers = Object.fromEntries(
      (Object.keys(LAYER_FILES) as LayerId[]).map((id) => [id, byName[LAYER_FILES[id]] as FeatureCollection]),
    ) as Record<LayerId, FeatureCollection>;
    return { status: "ok", data: { manifest, infrastructure, connectivity, layers } };
  } catch (e) {
    return { status: "error", message: e instanceof Error ? e.message : String(e) };
  }
}

export interface Kpi {
  id: string;
  label: string;
  value: string;
  detail: string;
}

const fmt = (n: number, digits = 0) =>
  n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** KPI cards, all values taken from infrastructure_summary.json. */
export function kpis(d: DashboardData): Kpi[] {
  const s = d.infrastructure;
  return [
    { id: "flood", label: "Predicted flood extent", value: `${fmt(s.flood_area_km2_utm, 3)} km²`,
      detail: `${fmt(s.flood_area_km2_utm * 100, 1)} ha · frozen model, class "flood"` },
    { id: "roads", label: "Road ways potentially affected", value: fmt(s.roads.segments_affected),
      detail: `of ${fmt(s.roads.segments_total)} mapped road ways` },
    { id: "road_length", label: "Road length inside predicted flood", value: `${fmt(s.roads.affected_length_km * 1000)} m`,
      detail: `of ${fmt(s.roads.length_km_total, 1)} km of mapped roads` },
    { id: "bridges", label: "Bridges intersecting predicted flood", value: fmt(s.bridges.affected),
      detail: `of ${fmt(s.bridges.total)} mapped bridges` },
    { id: "buildings", label: "Buildings potentially affected", value: fmt(s.buildings.affected),
      detail: `of ${fmt(s.buildings.total)} mapped buildings` },
    { id: "settlements", label: "Settlements potentially cut off", value: fmt(s.settlements.potentially_cut_off),
      detail: `of ${fmt(s.settlements.analyzed)} mapped settlements` },
  ];
}

const STATUS_ORDER = ["potentially_cut_off", "connected", "not_on_main_network_before", "no_road_within_snap_distance"];

export function settlementsSorted(d: DashboardData): Settlement[] {
  return [...d.connectivity.settlements].sort(
    (a, b) => STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status) || (a.name ?? "").localeCompare(b.name ?? ""),
  );
}

export type Bounds = [number, number, number, number]; // west, south, east, north

/** Bounding box of all coordinates in the given feature collections (null if there are none). */
export function featureBounds(collections: FeatureCollection[]): Bounds | null {
  let b: Bounds | null = null;
  const visit = (c: unknown): void => {
    if (!Array.isArray(c)) return;
    if (typeof c[0] === "number" && typeof c[1] === "number") {
      const [x, y] = c as number[];
      b = b ? [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)] : [x, y, x, y];
      return;
    }
    c.forEach(visit);
  };
  for (const fc of collections) for (const f of fc.features) visit(f.geometry.coordinates);
  return b;
}

/** Extent of the impact layers: predicted flood, flooded road sections, bridges intersecting predicted
 * flood, potentially affected buildings and potentially cut-off settlements. */
export function impactBounds(d: DashboardData): Bounds | null {
  const L = d.layers;
  const cutOff = { type: "FeatureCollection" as const,
    features: L.settlements.features.filter((f) => f.properties.status === "potentially_cut_off") };
  return featureBounds([L.flood_extent, L.affected_road_sections, L.affected_bridges, L.affected_buildings, cutOff]);
}

/** Midpoint of a (Multi)LineString's coordinates, for placing a marker on short features like bridges. */
export function lineMidpoint(geometry: { type: string; coordinates: unknown }): [number, number] | null {
  const coords = (geometry.type === "MultiLineString"
    ? (geometry.coordinates as number[][][]).flat()
    : (geometry.coordinates as number[][])) ?? [];
  if (!coords.length) return null;
  const [x, y] = coords.reduce(([sx, sy], [cx, cy]) => [sx + cx, sy + cy], [0, 0]);
  return [x / coords.length, y / coords.length];
}

/** Centroid of a polygon's outer ring (vertex mean), for marking small building footprints. */
export function polygonCenter(geometry: { type: string; coordinates: unknown }): [number, number] | null {
  const ring = geometry.type === "Polygon" ? (geometry.coordinates as number[][][])[0]
    : geometry.type === "MultiPolygon" ? (geometry.coordinates as number[][][][])[0]?.[0] : undefined;
  if (!ring?.length) return null;
  const pts = ring.slice(0, -1);
  const [x, y] = pts.reduce(([sx, sy], [cx, cy]) => [sx + cx, sy + cy], [0, 0]);
  return [x / pts.length, y / pts.length];
}

/** Point FeatureCollection placed at each feature's midpoint/centre, keeping its properties. */
export function markerPoints(fc: FeatureCollection, kind: "line" | "polygon"): FeatureCollection {
  const features = fc.features.flatMap((f) => {
    const p = kind === "line" ? lineMidpoint(f.geometry) : polygonCenter(f.geometry);
    return p ? [{ type: "Feature" as const, geometry: { type: "Point", coordinates: p }, properties: f.properties }] : [];
  });
  return { type: "FeatureCollection", features };
}
