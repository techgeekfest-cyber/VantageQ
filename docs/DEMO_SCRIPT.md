# VantageQ — 3-Minute Live Demo Script

For the current dashboard (`cd frontend && npm run dev`, then <http://localhost:3000>) and the
committed situation report. All numbers below are what the dashboard shows. They come from the
committed demo snapshot (`frontend/public/demo/trishuli/infrastructure_summary.json`,
`connectivity_summary.json`).

**Before the demo**

- Start the dashboard and let the map load (the basemap needs internet; the analytical layers do not).
- Have `docs/output/VantageQ_Trishuli_Situation_Report.pdf` open in a second window.
- Leave the map on its default whole-AOI view. Close any popups.

**Wording rules for the presenter:** say *predicted flood extent*, *potentially affected*,
*bridges intersecting predicted flood*, *potentially cut off*, *simplified graph indicator*,
*unvalidated Trishuli prediction*. Never claim confirmed damage, confirmed real-world
isolation, validated Trishuli accuracy, or a production emergency-response system.

---

## 0:00–0:20 — Problem

> "After a flood, responders need three answers fast: where is the water, which roads, bridges and
> buildings may be affected, and which communities may have lost road access. That is the Track B
> problem: mapping flood impact from space. VantageQ is an end-to-end prototype that produces those
> three answers from satellite radar and open map data."

*(Point at the header: VantageQ, "Satellite-Powered Flood Intelligence for Disaster Response".)*

## 0:20–0:45 — Input and data

*(Point at the header fields: AOI, event date, post-event Sentinel-1, model.)*

> "Our case study is the Trishuli Valley in Nepal, flood event of 26 August 2026: a small area
> around Betrawati. Monsoon cloud makes optical imagery unreliable, so we use Sentinel-1 radar. One
> post-event image from 28 August and two pre-event images from 16 and 4 August, all from the same
> orbit track, are stacked into six channels. A U-Net trained on the Kuro Siwo flood dataset
> predicts no water, permanent water or flood for every 10-metre pixel."

## 0:45–1:20 — Flood intelligence

*(The predicted flood layer is visible in blue. Click **Zoom to impacts**.)*

> "This blue layer is the predicted flood extent: 0.120 square kilometres, about 0.55 percent of the
> analysed area. 'Zoom to impacts' fits the map to everything the analysis flagged. The patches sit
> around Betrawati, by the Falaakhu River bridge, and along the valley highway."

> "This is an AI prediction, not validated ground truth. On held-out Kuro Siwo test events the
> model reached flood IoU 0.31, with high precision but low recall. We have no labels for Trishuli,
> so we do not report a Trishuli accuracy."

## 1:20–1:55 — Infrastructure

*(Point at the KPI cards, then at the map: dark-red road sections, purple bridge markers, orange
building markers. Optionally click a dark-red road section to show its popup.)*

> "We overlay OpenStreetMap as it was the day before the event, a historical snapshot, so no
> post-flood edits leak in. Three road ways are potentially affected, with 189 metres of road inside
> the predicted flood, all on the main valley highway. Two bridges intersect the predicted flood:
> the Falaakhu River Bridge and a footbridge. 36 buildings are potentially affected."

> "These are overlap rules, not damage assessments. A road needs at least 10 metres inside the
> predicted flood, a prototype rule. And a bridge crosses water by design, so 'intersecting
> predicted flood' does not establish bridge damage or impassability."

## 1:55–2:30 — Connectivity

*(Click the **Betrawati** marker, then the **Bhainse** marker. Each popup shows status and reachable
road length.)*

> "We build a simple road graph from the same historical OSM and remove the road edges that the
> predicted flood blocks. Betrawati drops from 31 kilometres of reachable road network to 0.7, and
> Bhainse to 0.39, so both are flagged potentially cut off. The third settlement, Naubisephat, was
> already separate from the main network in the mapped roads, so it is not counted."

> "Potentially cut off = disconnected in the simplified road graph after removing flood-affected
> edges. This is NOT confirmed real-world isolation. Footpaths are excluded, and the graph stops at
> the analysis boundary. Bhainse sits on that boundary, so its result is the least certain."

*(Optionally scroll the sidebar to the Methodology and Limitations panels.)*

## 2:30–2:50 — Situation report

*(Switch to `docs/output/VantageQ_Trishuli_Situation_Report.pdf`.)*

> "The same structured outputs that feed the dashboard also generate this one-page situation report:
> flood overview, infrastructure, connectivity, methodology and limitations. Every number is read
> from the analysis files, nothing is typed by hand, and regenerating it gives an identical PDF."

## 2:50–3:00 — Close

> "VantageQ connects satellite flood prediction to infrastructure impact and connectivity
> intelligence, transparently and reproducibly, with every limitation stated. It is a research
> prototype, and the Trishuli prediction is still unvalidated."

---

### Fallbacks

- **No internet:** the basemap tiles do not load, but all analytical layers, popups and KPIs still
  work (they come from the committed snapshot).
- **Map slow to appear:** narrate from the KPI cards and settlement panel; they load from the same
  data.
- **Clicked the wrong feature:** close the popup with ×; "Zoom to impacts" restores the impact view.

### Numbers used (from the committed snapshot)

| Item | Value |
|---|---|
| Predicted flood extent | 0.120 km² (0.55 % of analysed AOI pixels) |
| Road ways potentially affected | 3 of 118 (189 m inside predicted flood) |
| Bridges intersecting predicted flood | 2 of 10 (Falaakhu River Bridge, Tupche Aama ko pool footbridge) |
| Buildings potentially affected | 36 of 4,877 |
| Settlements potentially cut off | 2 of 3: Betrawati (31.13 → 0.70 km), Bhainse (31.13 → 0.39 km) |
| Kuro Siwo held-out test (not Trishuli) | flood IoU 0.307, F1 0.470, precision 0.852, recall 0.324 |
| Sentinel-1 | post 2026-08-28; pre 2026-08-16, 2026-08-04; relative orbit 85 ascending |
| OSM snapshot | 2026-08-25T00:00:00Z (pre-event) |
