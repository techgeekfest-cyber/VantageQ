import { describe, expect, it } from "vitest";
import { readDemo } from "@/test/fsFetch";
import { kpis } from "./data";
import {
  bridgePopup, buildingPopup, CUT_OFF_NOTE, floodPopup, popupHtml, roadPopup, settlementPopup, STATUS_INFO,
} from "./format";
import { LAYERS } from "./layers";

const texts = (c: { title: string; rows: [string, string][]; note?: string }) =>
  [c.title, c.note ?? "", ...c.rows.flat()].join(" ");

describe("wording", () => {
  it("never labels anything as damaged", () => {
    const bridge = readDemo("affected_bridges.geojson").features[0].properties;
    const road = readDemo("affected_roads.geojson").features[0].properties;
    const bld = readDemo("affected_buildings.geojson").features[0].properties;
    const all = [
      texts(bridgePopup(bridge, true)), texts(bridgePopup(bridge, false)), texts(roadPopup(road)),
      texts(buildingPopup(bld)), texts(floodPopup()), ...LAYERS.map((l) => l.label),
      ...Object.values(STATUS_INFO).map((s) => s.label),
    ].join(" ").toLowerCase();
    // "damage" may only appear negated ("does not establish bridge damage", "not a damage assessment").
    expect(all).not.toMatch(/\bdamaged\b/);
    for (const m of all.matchAll(/damage/g)) {
      const before = all.slice(Math.max(0, (m.index ?? 0) - 40), m.index);
      expect(before).toMatch(/not|no /);
    }
  });

  it("bridge popup says it intersects predicted flood", () => {
    const p = readDemo("affected_bridges.geojson").features[0].properties;
    const c = bridgePopup(p, true);
    expect(c.rows).toContainEqual(["Status", "Intersects predicted flood"]);
    expect(c.title).toBe(p.name);
  });

  it("road popup shows type, total and affected length and share", () => {
    const p = readDemo("affected_roads.geojson").features[0].properties;
    const rows = Object.fromEntries(roadPopup(p).rows);
    expect(rows["Highway type"]).toContain(p.highway);
    expect(rows["Total length (in AOI)"]).toContain("m");
    expect(rows["Length inside predicted flood"]).toContain(String(Math.round(p.affected_length_m)));
    expect(rows["Share inside predicted flood"]).toContain("%");
  });

  it("settlement popup uses the graph-indicator wording for cut-off settlements only", () => {
    const sets = readDemo("connectivity_summary.json").settlements;
    for (const s of sets) {
      const c = settlementPopup(s);
      const cut = s.status === "potentially_cut_off";
      expect(c.rows.find(([k]) => k === "Status")?.[1]).toBe(STATUS_INFO[s.status].label);
      expect(c.note === CUT_OFF_NOTE).toBe(cut);
    }
    expect(STATUS_INFO.potentially_cut_off.label).toBe("Potentially cut off — simplified graph indicator");
  });

  it("escapes HTML in popups", () => {
    expect(popupHtml({ title: "<b>x</b>", rows: [["a", "<i>"]] })).not.toContain("<b>");
  });
});
