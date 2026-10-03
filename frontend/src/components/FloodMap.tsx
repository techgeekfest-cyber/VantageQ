"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { GeoJSONSourceSpecification, Map as MlMap, MapMouseEvent, Marker } from "maplibre-gl";
import { impactBounds, markerPoints } from "@/lib/data";
import { popupHtml, settlementPopup, statusInfo } from "@/lib/format";
import { COLORS, LAYERS } from "@/lib/layers";
import type { DashboardData, Settlement } from "@/lib/types";

// Map sub-layers per toggle id (a toggle can drive several MapLibre layers).
const SUBLAYERS: Record<string, string[]> = {
  flood_extent: ["flood_extent-fill", "flood_extent-line"],
  roads_reference: ["roads_reference-line"],
  affected_roads: ["affected_roads-line"],
  affected_road_sections: ["affected_road_sections-line"],
  bridges: ["bridges-line", "bridges-point"],
  affected_bridges: ["affected_bridges-line", "affected_bridges-point"],
  affected_buildings: ["affected_buildings-fill", "affected_buildings-point"],
};

// Served from public/maplibre/ (copied from node_modules by scripts/copy-maplibre-worker.mjs).
const WORKER_URL = "/maplibre/maplibre-gl-worker.mjs";

const BASEMAP = {
  version: 8 as const,
  sources: {
    osm: {
      type: "raster" as const,
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "Basemap © OpenStreetMap contributors (current tiles, orientation only)",
    },
  },
  layers: [{ id: "osm", type: "raster" as const, source: "osm", paint: { "raster-saturation": -0.6, "raster-opacity": 0.85 } }],
};

interface Props {
  data: DashboardData;
  visibility: Record<string, boolean>;
}

export default function FloodMap({ data, visibility }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const markersRef = useRef<Marker[]>([]);
  const visRef = useRef(visibility);
  visRef.current = visibility;
  const [ready, setReady] = useState(false);
  const impacts = useMemo(() => impactBounds(data), [data]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const maplibregl = await import("maplibre-gl");
      if (cancelled || !container.current) return;
      maplibregl.setWorkerUrl(WORKER_URL);
      const [w, s, e, n] = data.manifest.aoi_wgs84;
      const map = new maplibregl.Map({ container: container.current, style: BASEMAP, bounds: [[w, s], [e, n]],
        fitBoundsOptions: { padding: 24 }, attributionControl: { compact: true } });
      mapRef.current = map;
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

      map.on("load", () => {
        const L = data.layers;
        const add = (id: string, d: object) => map.addSource(id, { type: "geojson", data: d as GeoJSONSourceSpecification["data"] });
        add("aoi", { type: "Feature", properties: {}, geometry: { type: "Polygon",
          coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } });
        add("roads_reference", L.roads_reference);
        add("flood_extent", L.flood_extent);
        add("affected_roads", L.affected_roads);
        add("affected_road_sections", L.affected_road_sections);
        add("bridges", L.bridges);
        add("bridges_pt", markerPoints(L.bridges, "line"));
        add("affected_bridges", L.affected_bridges);
        add("affected_bridges_pt", markerPoints(L.affected_bridges, "line"));
        add("affected_buildings", L.affected_buildings);
        add("affected_buildings_pt", markerPoints(L.affected_buildings, "polygon"));

        map.addLayer({ id: "aoi-line", type: "line", source: "aoi",
          paint: { "line-color": "#222", "line-width": 1.2, "line-dasharray": [3, 2] } });
        map.addLayer({ id: "roads_reference-line", type: "line", source: "roads_reference",
          paint: { "line-color": COLORS.road, "line-opacity": 0.8,
            "line-width": ["match", ["get", "highway"], ["primary", "trunk", "secondary"], 2.6, 1.1] } });
        map.addLayer({ id: "flood_extent-fill", type: "fill", source: "flood_extent",
          paint: { "fill-color": COLORS.flood, "fill-opacity": 0.55 } });
        map.addLayer({ id: "flood_extent-line", type: "line", source: "flood_extent",
          paint: { "line-color": COLORS.flood, "line-width": 1.2 } });
        map.addLayer({ id: "affected_roads-line", type: "line", source: "affected_roads",
          paint: { "line-color": COLORS.affectedRoad, "line-width": 3, "line-dasharray": [2, 1.5] } });
        map.addLayer({ id: "affected_road_sections-line", type: "line", source: "affected_road_sections",
          layout: { "line-cap": "butt" }, paint: { "line-color": COLORS.affectedSection, "line-width": 7 } });
        map.addLayer({ id: "affected_buildings-fill", type: "fill", source: "affected_buildings",
          paint: { "fill-color": COLORS.building, "fill-outline-color": "#7f2704" } });
        map.addLayer({ id: "affected_buildings-point", type: "circle", source: "affected_buildings_pt", maxzoom: 16.5,
          paint: { "circle-radius": 4, "circle-color": COLORS.building, "circle-stroke-color": "#4d1a00", "circle-stroke-width": 0.8 } });
        map.addLayer({ id: "bridges-line", type: "line", source: "bridges", filter: ["!=", ["get", "affected"], true],
          paint: { "line-color": COLORS.bridge, "line-width": 4 } });
        map.addLayer({ id: "bridges-point", type: "circle", source: "bridges_pt", filter: ["!=", ["get", "affected"], true],
          paint: { "circle-radius": 5, "circle-color": "#fff", "circle-stroke-color": COLORS.bridge, "circle-stroke-width": 2 } });
        map.addLayer({ id: "affected_bridges-line", type: "line", source: "affected_bridges",
          paint: { "line-color": COLORS.affectedBridge, "line-width": 6 } });
        map.addLayer({ id: "affected_bridges-point", type: "circle", source: "affected_bridges_pt",
          paint: { "circle-radius": 7, "circle-color": COLORS.affectedBridge, "circle-stroke-color": "#fff", "circle-stroke-width": 2 } });

        // Popups: the topmost clicked layer wins.
        const clickable = LAYERS.filter((l) => l.popup).flatMap((l) => (SUBLAYERS[l.id] ?? []).map((sub) => ({ sub, def: l })));
        map.on("click", (ev: MapMouseEvent) => {
          const hits = map.queryRenderedFeatures(ev.point, { layers: clickable.map((c) => c.sub) });
          if (!hits.length) return;
          const def = clickable.find((c) => c.sub === hits[0].layer.id)?.def;
          if (!def?.popup) return;
          new maplibregl.Popup({ maxWidth: "320px" }).setLngLat(ev.lngLat)
            .setHTML(popupHtml(def.popup(hits[0].properties ?? {}))).addTo(map);
        });
        for (const { sub } of clickable) {
          map.on("mouseenter", sub, () => { map.getCanvas().style.cursor = "pointer"; });
          map.on("mouseleave", sub, () => { map.getCanvas().style.cursor = ""; });
        }

        // Settlements as labelled HTML markers (no glyph server needed).
        markersRef.current = L.settlements.features.map((f) => {
          const p = f.properties as unknown as Settlement;
          const el = document.createElement("button");
          el.className = `vq-settlement vq-tone-${statusInfo(p.status).tone}`;
          el.type = "button";
          el.setAttribute("aria-label", `${p.name ?? p.osm_id}: ${statusInfo(p.status).label}`);
          el.innerHTML = `<span class="vq-dot"></span><span class="vq-name"></span>`;
          (el.querySelector(".vq-name") as HTMLElement).textContent = p.name ?? String(p.osm_id);
          const popup = new maplibregl.Popup({ offset: 14, maxWidth: "320px" }).setHTML(popupHtml(settlementPopup(p)));
          return new maplibregl.Marker({ element: el, anchor: "left", offset: [-7, 0] })
            .setLngLat(f.geometry.coordinates as [number, number]).setPopup(popup).addTo(map);
        });
        applyVisibility(map, markersRef.current, visRef.current);
        setReady(true);
      });
    })();
    return () => {
      cancelled = true;
      markersRef.current.forEach((m) => m.remove());
      mapRef.current?.remove();
      mapRef.current = null;
      setReady(false);
    };
  }, [data]);

  useEffect(() => {
    const map = mapRef.current;
    if (map?.isStyleLoaded()) applyVisibility(map, markersRef.current, visibility);
  }, [visibility]);

  const zoomToImpacts = () => {
    if (!mapRef.current || !impacts) return;
    const [w, s, e, n] = impacts;
    mapRef.current.fitBounds([[w, s], [e, n]], { padding: 60, maxZoom: 16.5, duration: 800 });
  };

  return (
    <>
      <div ref={container} className="vq-map" data-testid="flood-map" role="region" aria-label="Flood impact map" />
      <button type="button" className="vq-zoom-btn" onClick={zoomToImpacts} disabled={!ready || !impacts}>
        Zoom to impacts
      </button>
    </>
  );
}

function applyVisibility(map: MlMap, markers: Marker[], visibility: Record<string, boolean>) {
  for (const [id, subs] of Object.entries(SUBLAYERS)) {
    for (const sub of subs) {
      if (map.getLayer(sub)) map.setLayoutProperty(sub, "visibility", visibility[id] ? "visible" : "none");
    }
  }
  for (const m of markers) m.getElement().style.display = visibility.cutoff ? "" : "none";
}
