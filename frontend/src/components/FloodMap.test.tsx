import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { featureBounds, impactBounds, loadDashboardData } from "@/lib/data";
import { defaultVisibility } from "@/lib/layers";
import { fsFetch } from "@/test/fsFetch";
import FloodMap from "./FloodMap";

// Minimal MapLibre stand-in (jsdom has no WebGL): fires "load" and records camera calls.
const maps = vi.hoisted(() => [] as { fitBounds: ReturnType<typeof vi.fn>; options: { bounds: number[][] } }[]);
vi.mock("maplibre-gl", () => {
  class FakeMap {
    fitBounds = vi.fn();
    constructor(public options: { bounds: number[][] }) { maps.push(this); }
    on(event: string, a: unknown) { if (event === "load" && typeof a === "function") setTimeout(a as () => void, 0); }
    addControl() {} addSource() {} addLayer() {} setLayoutProperty() {} remove() {}
    getLayer() { return true; }
    isStyleLoaded() { return true; }
    getCanvas() { return { style: {} }; }
    queryRenderedFeatures() { return []; }
  }
  class Marker {
    el: HTMLElement;
    constructor(o: { element: HTMLElement }) { this.el = o.element; }
    setLngLat() { return this; } setPopup() { return this; } addTo() { return this; } remove() {}
    getElement() { return this.el; }
  }
  class Popup { setHTML() { return this; } setLngLat() { return this; } addTo() { return this; } }
  class Control {}
  return { Map: FakeMap, Marker, Popup, NavigationControl: Control, ScaleControl: Control, setWorkerUrl: vi.fn() };
});

describe("FloodMap", () => {
  it("'Zoom to impacts' fits the map to the extent of the loaded impact layers", async () => {
    const r = await loadDashboardData("/demo/trishuli", fsFetch());
    if (r.status !== "ok") throw new Error("demo data not loaded");
    const data = r.data;
    render(<FloodMap data={data} visibility={defaultVisibility()} />);

    const button = await screen.findByRole("button", { name: "Zoom to impacts" });
    await waitFor(() => expect(button).toBeEnabled());
    const map = maps.at(-1)!;
    const [w, s, e, n] = data.manifest.aoi_wgs84;
    expect(map.options.bounds).toEqual([[w, s], [e, n]]); // default view stays the whole AOI

    fireEvent.click(button);
    const expected = impactBounds(data)!;
    expect(map.fitBounds).toHaveBeenCalledTimes(1);
    const [[fw, fs], [fe, fn]] = map.fitBounds.mock.calls[0][0];
    expect([fw, fs, fe, fn]).toEqual(expected);

    // Derived from the layers, inside the AOI and smaller than it, and containing every flood/bridge vertex.
    expect(fw).toBeGreaterThanOrEqual(w);
    expect(fe).toBeLessThanOrEqual(e);
    expect((fe - fw) * (fn - fs)).toBeLessThan((e - w) * (n - s));
    const [bw, bs, be, bn] = featureBounds([data.layers.flood_extent, data.layers.affected_bridges])!;
    expect(bw >= fw && bs >= fs && be <= fe && bn <= fn).toBe(true);
  });
});
