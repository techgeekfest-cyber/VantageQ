import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fsFetch, readDemo } from "@/test/fsFetch";
import { MISSING_DATA_MESSAGE } from "@/lib/data";
import { LAYERS } from "@/lib/layers";
import Dashboard from "./Dashboard";

// MapLibre needs WebGL, which jsdom lacks: stub the dynamically imported map.
vi.mock("next/dynamic", () => ({
  default: () => function MapStub() {
    return <div data-testid="flood-map" />;
  },
}));

afterEach(() => vi.unstubAllGlobals());

describe("Dashboard", () => {
  it("renders KPIs, settlements, layers, methodology and limitations from the demo snapshot", async () => {
    vi.stubGlobal("fetch", fsFetch());
    render(<Dashboard />);
    const s = readDemo("infrastructure_summary.json");

    expect(await screen.findByTestId("kpi-buildings")).toHaveTextContent(String(s.buildings.affected));
    expect(screen.getByTestId("kpi-bridges")).toHaveTextContent("Bridges intersecting predicted flood");
    expect(screen.getByTestId("kpi-settlements")).toHaveTextContent(String(s.settlements.potentially_cut_off));
    expect(screen.getByTestId("flood-map")).toBeInTheDocument();

    for (const st of readDemo("connectivity_summary.json").settlements) {
      const row = screen.getByTestId(`settlement-${st.osm_id}`);
      expect(row).toHaveTextContent(st.name);
      if (st.status === "potentially_cut_off") {
        expect(within(row).getByText("Potentially cut off — simplified graph indicator")).toBeInTheDocument();
      }
    }
    expect(screen.getByText(/Not confirmed real-world isolation/)).toBeInTheDocument();

    const layers = screen.getByRole("group", { name: "Map layers" });
    expect(within(layers).getAllByRole("checkbox")).toHaveLength(LAYERS.length);
    const flood = within(layers).getByLabelText("Predicted flood extent (frozen model)");
    expect(flood).toBeChecked();
    fireEvent.click(flood);
    expect(flood).not.toBeChecked();

    expect(screen.getAllByText(/2026-08-25T00:00:00Z/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Footpaths are excluded/)).toBeInTheDocument();
    expect(screen.getByText(/does not establish bridge damage/)).toBeInTheDocument();
    expect(document.body.textContent?.toLowerCase()).not.toMatch(/\bdamaged\b/);
  });

  it("shows a clear message and no figures when demo data are missing", async () => {
    vi.stubGlobal("fetch", fsFetch({}, ["manifest.json", "infrastructure_summary.json"]));
    render(<Dashboard />);
    expect(await screen.findByRole("alert")).toHaveTextContent(MISSING_DATA_MESSAGE);
    expect(screen.queryByTestId("kpi-buildings")).not.toBeInTheDocument();
    expect(screen.queryByTestId("flood-map")).not.toBeInTheDocument();
  });
});
