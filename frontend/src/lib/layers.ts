// Single definition of every map layer: used by the map (rendering + popups) and the legend.

import { bridgePopup, buildingPopup, floodPopup, roadPopup, type PopupContent } from "./format";
import type { LayerId } from "./types";

export type Swatch = "fill" | "line" | "dashed" | "circle" | "star";

export interface LayerDef {
  id: string;            // toggle id
  label: string;         // legend label
  source: LayerId;       // demo snapshot layer
  color: string;
  swatch: Swatch;
  defaultVisible: boolean;
  popup?: (props: Record<string, unknown>) => PopupContent;
}

export const COLORS = {
  flood: "#2b8cbe",
  road: "#7a7a7a",
  affectedRoad: "#e34a33",
  affectedSection: "#99000d",
  bridge: "#4d4d4d",
  affectedBridge: "#7a0177",
  building: "#fd8d3c",
  cutOff: "#cb181d",
  connected: "#238b45",
  neutral: "#6a51a3",
} as const;

// Order = legend order (top to bottom) and reverse draw order on the map.
export const LAYERS: LayerDef[] = [
  { id: "cutoff", label: "Settlements (red = potentially cut off)", source: "settlements", color: COLORS.cutOff,
    swatch: "star", defaultVisible: true },
  { id: "affected_bridges", label: "Bridges intersecting predicted flood", source: "affected_bridges",
    color: COLORS.affectedBridge, swatch: "circle", defaultVisible: true, popup: (p) => bridgePopup(p, true) },
  { id: "affected_buildings", label: "Buildings potentially affected", source: "affected_buildings",
    color: COLORS.building, swatch: "circle", defaultVisible: true, popup: buildingPopup },
  { id: "affected_road_sections", label: "Road sections inside predicted flood", source: "affected_road_sections",
    color: COLORS.affectedSection, swatch: "line", defaultVisible: true, popup: roadPopup },
  { id: "affected_roads", label: "Rest of potentially affected road ways", source: "affected_roads",
    color: COLORS.affectedRoad, swatch: "dashed", defaultVisible: true, popup: roadPopup },
  { id: "flood_extent", label: "Predicted flood extent (frozen model)", source: "flood_extent", color: COLORS.flood,
    swatch: "fill", defaultVisible: true, popup: () => floodPopup() },
  { id: "bridges", label: "Other bridges (OSM)", source: "bridges", color: COLORS.bridge, swatch: "circle",
    defaultVisible: false, popup: (p) => bridgePopup(p, Boolean(p.affected)) },
  { id: "roads_reference", label: "Road network (OSM, 2026-08-25)", source: "roads_reference", color: COLORS.road,
    swatch: "line", defaultVisible: true },
];

export const defaultVisibility = (): Record<string, boolean> =>
  Object.fromEntries(LAYERS.map((l) => [l.id, l.defaultVisible]));
