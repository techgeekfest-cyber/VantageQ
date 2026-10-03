// Shapes of the committed demo snapshot (scripts/build_demo_snapshot.py). Only fields the
// dashboard reads are typed.

export type Geometry = { type: string; coordinates: unknown };
export type Feature<P = Record<string, unknown>> = { type: "Feature"; geometry: Geometry; properties: P };
export type FeatureCollection<P = Record<string, unknown>> = { type: "FeatureCollection"; features: Feature<P>[] };

export interface InfrastructureSummary {
  aoi_wgs84: [number, number, number, number];
  osm_snapshot: string;
  osm_source: string;
  flood_area_km2_utm: number;
  flood_min_pixels: number;
  rules: Record<string, string>;
  roads: { segments_total: number; segments_affected: number; length_km_total: number; affected_length_km: number };
  paths_not_in_graph: { segments: number; length_km: number };
  bridges: { total: number; affected: number; near_flood: number };
  buildings: { total: number; affected: number };
  settlements: { analyzed: number; potentially_cut_off: number };
}

export type SettlementStatus =
  | "potentially_cut_off"
  | "connected"
  | "not_on_main_network_before"
  | "no_road_within_snap_distance";

export interface Settlement {
  osm_id: number;
  name: string | null;
  name_local: string | null;
  place: string;
  status: SettlementStatus | string;
  snap_distance_m: number;
  reachable_road_km_before: number;
  reachable_road_km_after: number;
  access_reduced: boolean;
}

export interface ConnectivitySummary {
  definition: string;
  blocked_edge_rule: string;
  max_snap_m: number;
  network: {
    nodes: number;
    edges: number;
    blocked_edges: number;
    components_before: number;
    components_after: number;
    main_network_km_before: number;
    main_network_km_after: number;
  };
  settlements: Settlement[];
  status_counts: Record<string, number>;
}

export interface Scene {
  id: string;
  datetime: string;
  platform: string;
  orbit_state: string;
  relative_orbit: number;
}

export interface Manifest {
  title: string;
  aoi_name: string;
  aoi_wgs84: [number, number, number, number];
  event_date: string;
  scenes: { post: Scene; pre1: Scene; pre2: Scene };
  model: { name: string; input: string; checkpoint_sha256_16: string; checkpoint_epoch: number };
}

export type LayerId =
  | "flood_extent"
  | "roads_reference"
  | "affected_roads"
  | "affected_road_sections"
  | "bridges"
  | "affected_bridges"
  | "affected_buildings"
  | "settlements";

export interface DashboardData {
  manifest: Manifest;
  infrastructure: InfrastructureSummary;
  connectivity: ConnectivitySummary;
  layers: Record<LayerId, FeatureCollection>;
}

export type LoadResult =
  | { status: "ok"; data: DashboardData }
  | { status: "missing"; missing: string[] }
  | { status: "error"; message: string };
