import { describe, expect, it } from "vitest";
import { fsFetch, readDemo } from "@/test/fsFetch";
import { kpis, LAYER_FILES, loadDashboardData, markerPoints, settlementsSorted } from "./data";
import type { DashboardData } from "./types";

async function demo(): Promise<DashboardData> {
  const r = await loadDashboardData("/demo/trishuli", fsFetch());
  if (r.status !== "ok") throw new Error(`demo data not loaded: ${JSON.stringify(r)}`);
  return r.data;
}

describe("loadDashboardData", () => {
  it("loads the committed demo snapshot with every layer", async () => {
    const d = await demo();
    expect(Object.keys(d.layers).sort()).toEqual(Object.keys(LAYER_FILES).sort());
    expect(d.layers.flood_extent.features.length).toBeGreaterThan(0);
    expect(d.layers.affected_buildings.features).toHaveLength(d.infrastructure.buildings.affected);
    expect(d.layers.affected_roads.features).toHaveLength(d.infrastructure.roads.segments_affected);
    expect(d.layers.affected_bridges.features).toHaveLength(d.infrastructure.bridges.affected);
  });

  it("reports missing files instead of returning placeholder data", async () => {
    const r = await loadDashboardData("/demo/trishuli", fsFetch({}, ["infrastructure_summary.json"]));
    expect(r).toEqual({ status: "missing", missing: ["infrastructure_summary.json"] });
  });

  it("rejects summaries with missing numbers (no silent zeroes)", async () => {
    const bad = readDemo("infrastructure_summary.json");
    delete bad.buildings.affected;
    const r = await loadDashboardData("/demo/trishuli", fsFetch({ "infrastructure_summary.json": bad }));
    expect(r.status).toBe("error");
    if (r.status === "error") expect(r.message).toContain("infrastructure.buildings.affected");
  });

  it("treats a network failure as missing data", async () => {
    const r = await loadDashboardData("/demo/trishuli", async () => { throw new Error("offline"); });
    expect(r.status).toBe("missing");
  });
});

describe("derived values", () => {
  it("KPI values come from infrastructure_summary.json", async () => {
    const d = await demo();
    const s = readDemo("infrastructure_summary.json");
    const byId = Object.fromEntries(kpis(d).map((k) => [k.id, k]));
    expect(byId.flood.value).toBe(`${s.flood_area_km2_utm.toFixed(3)} km²`);
    expect(byId.roads.value).toBe(String(s.roads.segments_affected));
    expect(byId.road_length.value).toBe(`${Math.round(s.roads.affected_length_km * 1000)} m`);
    expect(byId.bridges.value).toBe(String(s.bridges.affected));
    expect(byId.buildings.value).toBe(String(s.buildings.affected));
    expect(byId.settlements.value).toBe(String(s.settlements.potentially_cut_off));
    expect(byId.buildings.detail).toContain(s.buildings.total.toLocaleString("en-US"));
  });

  it("orders settlements with potentially cut-off first", async () => {
    const d = await demo();
    const rows = settlementsSorted(d);
    expect(rows).toHaveLength(d.connectivity.settlements.length);
    const firstOther = rows.findIndex((r) => r.status !== "potentially_cut_off");
    expect(rows.slice(firstOther).every((r) => r.status !== "potentially_cut_off")).toBe(true);
  });

  it("places marker points at line midpoints", () => {
    const fc = markerPoints({ type: "FeatureCollection", features: [{ type: "Feature", properties: { a: 1 },
      geometry: { type: "LineString", coordinates: [[0, 0], [2, 4]] } }] }, "line");
    expect(fc.features[0].geometry.coordinates).toEqual([1, 2]);
    expect(fc.features[0].properties).toEqual({ a: 1 });
  });
});
