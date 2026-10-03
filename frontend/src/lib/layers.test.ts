import { describe, expect, it } from "vitest";
import { LAYER_FILES } from "./data";
import { defaultVisibility, LAYERS } from "./layers";

describe("map layers", () => {
  it("defines every required layer, each backed by a snapshot file", () => {
    const ids = LAYERS.map((l) => l.id);
    for (const required of ["flood_extent", "affected_road_sections", "affected_roads", "affected_bridges",
      "affected_buildings", "cutoff", "roads_reference"]) {
      expect(ids).toContain(required);
    }
    for (const l of LAYERS) expect(Object.keys(LAYER_FILES)).toContain(l.source);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("shows the impact layers by default and keeps the extra bridge layer optional", () => {
    const v = defaultVisibility();
    expect(v.flood_extent && v.affected_road_sections && v.affected_bridges && v.affected_buildings && v.cutoff).toBe(true);
    expect(v.bridges).toBe(false);
  });

  it("gives clickable impact layers a popup", () => {
    for (const id of ["flood_extent", "affected_road_sections", "affected_roads", "affected_bridges", "affected_buildings"]) {
      expect(LAYERS.find((l) => l.id === id)?.popup).toBeTypeOf("function");
    }
  });
});
