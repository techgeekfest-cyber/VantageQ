# VantageQ — Judge Q&A

Short answers (20–40 s spoken) using only facts recorded in this repository. **Measured** marks
numbers from our own runs; **hypothesis** marks explanations we have not verified. Sources are in
brackets.

Terminology: *predicted flood extent*, *potentially affected*, *bridges intersecting predicted
flood*, *potentially cut off* (a *simplified graph indicator*), *unvalidated Trishuli prediction*.

---

### Data and model

**1. Why Sentinel-1 rather than Sentinel-2?**
Radar sees through cloud, day and night, and calm open water is dark in its backscatter. Floods
here come with monsoon cloud. Measured: for August–September 2026, Sentinel-2 tiles over Trishuli
were mostly 50–99 % cloudy, and no post-event scene below about 38 % cloud appeared until around
20 September, three weeks after the event. Sentinel-2 stays optional and secondary.
[docs/DATA_SOURCES.md §4]

**2. Why Kuro Siwo?**
It is a listed training dataset built for this task: 43 flood events with expert labels for
no water, permanent water and flood. Each sample has one post-event and two pre-event Sentinel-1
images, which matches what we can retrieve operationally. It ships official event-level
train/val/test splits and a documented SNAP preprocessing chain we could reproduce.
[docs/DATA_SOURCES.md §1, docs/KUROSIWO_DATASET.md]

**3. Why U-Net?**
It is the official Kuro Siwo baseline (U-Net with ResNet-18 encoder via
segmentation_models_pytorch): simple, strong for pixel segmentation, and trainable on a laptop GPU.
Using the official configuration keeps our results interpretable against the dataset's own setup.
We deliberately avoided more complex architectures. [docs/KUROSIWO_PREPROCESSING.md, ml/model.py]

**4. Why six channels / three acquisition dates?**
VV and VH for the post-event image and two pre-event images: [post VV, post VH, pre1 VV, pre1 VH,
pre2 VV, pre2 VH]. That is exactly Kuro Siwo's input order. The pre-event images let the model
separate new water from water or dark surfaces that were already there. All three come from the
same orbit track, so pixels compare like for like. [ml/preprocessing.py]

**5. How is VV/VH normalised?**
As Kuro Siwo does, on linear σ⁰ (not dB): clamp to [0, 0.15], NaN → 0.15, then VV (x − 0.0953)/0.0427
and VH (x − 0.0264)/0.0215. These are fixed global statistics with no per-image normalisation. A single
shared function is used for training and for Trishuli inference.
[docs/KUROSIWO_PREPROCESSING.md, ml/preprocessing.py]

**6. Why event-level train/validation/test separation?**
Neighbouring tiles from one flood are highly correlated, so a random tile split would leak and
inflate scores. We used Kuro Siwo's official event lists, and our scripts refuse to run if an event
or tile appears in two splits. The test set was evaluated once, after the model was frozen, and
not used for any tuning. [docs/TRAINING_EXPERIMENT.md, docs/TEST_EVALUATION.md]

**7. What are the Kuro Siwo test results?**
Measured on 94 samples from 6 unseen official test events: flood IoU 0.307, F1 0.470, precision
0.852, recall 0.324. When the model predicts flood it is usually right, but it misses about two
thirds. Results vary by event (flood IoU 0.066–0.691). The largest error is flood labelled as
permanent water (39.6 % of test flood pixels, mostly one Australian event). The training set was
small: 252 samples from 14 events. [docs/TEST_EVALUATION.md]

**8. Why don't you report Trishuli accuracy?**
Because we have no Trishuli ground truth among our permitted inputs. The only published reference,
EMSR927, is forbidden as a production input and reserved for a separate evaluation-only comparison
we have not built. So the Trishuli map is an unvalidated prediction, and Kuro Siwo test numbers
cannot be transferred to Himalayan terrain. [docs/TRISHULI_INFERENCE.md]

### Trishuli pipeline

**9. How did you select the Trishuli Sentinel-1 scenes?**
We searched the Copernicus Data Space STAC catalogue and kept IW VV+VH scenes covering the area.
We grouped them by relative orbit and direction, then took the first acquisition after the event
day and the two latest before it, all on the same track. That gave relative orbit 85 ascending:
post 28 August (two days after the event), pre 16 and 4 August. All three were fetched onto one
shared EPSG:3857 grid, and identical geometry was checked.
[docs/SENTINEL1_SMOKE_TEST.md, backend/vantageq/satellite/sentinel1.py]

**10. Why historical OSM?**
The rules forbid post-event OSM edits. After a disaster, mappers often update the affected area,
which would leak outcome information into our inputs. We query OSM as it existed on 2026-08-25T00:00:00Z,
the day before the event. [docs/TRISHULI_IMPACT_ANALYSIS.md]

**11. Why Overpass attic instead of anonymous ohsome extraction?**
Measured on 2026-10-04: ohsome v1 extraction endpoints return HTTP 403, and v2 extraction requires
an API key. Overpass "attic" queries with a [date:…] parameter return OSM as it was at that instant,
without an account. Cross-check: Overpass returned 210 highway ways in the area, matching ohsome's
anonymous count of 118 roads plus 92 paths. [docs/TRISHULI_IMPACT_ANALYSIS.md §1]

### Impact and connectivity rules

**12. How is an affected road defined?**
Each OSM road way, clipped to the area, is intersected with the predicted flood polygon in metres
(UTM). It is potentially affected if at least 10 m lies inside the predicted flood. That threshold is a
prototype rule (roughly one pixel), not calibrated. Result: 3 road ways, 189 m inside predicted
flood. [docs/TRISHULI_IMPACT_ANALYSIS.md §6]

**13. How is a potentially affected building defined?**
A building is potentially affected if more than 0.01 m² of its footprint lies inside the predicted
flood. The tiny threshold only excludes edge contact and floating-point slivers. Result: 36 of 4,877
mapped buildings. This is overlap, not a structural assessment. [docs/TRISHULI_IMPACT_ANALYSIS.md §8]

**14. What does "bridge intersecting predicted flood" mean?**
An OSM bridge way with some of its length inside the predicted flood is reported among the bridges
intersecting predicted flood. That does not establish bridge damage or impassability. Bridges cross water by design, so a river predicted as flood will always flag the
bridge over it. Result: 2 of 10, the Falaakhu River Bridge on NH42 and a footbridge.
[docs/TRISHULI_IMPACT_ANALYSIS.md §7]

**15. How is a settlement classified as potentially cut off?**
We build a NetworkX graph of vehicle roads (junctions as nodes) and remove edges with at least
10 m inside the predicted flood. A settlement is potentially cut off if its nearest road node was in
the main network before and is not after: disconnected in the simplified road graph after
removing flood-affected edges. Betrawati drops from 31.13 to 0.70 km of reachable road and
Bhainse to 0.39 km. It is a graph indicator, not confirmed real-world isolation.
[docs/TRISHULI_IMPACT_ANALYSIS.md §9–10]

**16. Why are footpaths excluded?**
To keep the prototype's graph to vehicle-usable roads, which matter for access by vehicle. This is a
real limitation: in hill villages footpaths are often the actual access. 92 footpaths, about 34 km,
are excluded from the graph, so a settlement flagged potentially cut off may still be reachable on
foot. [docs/TRISHULI_IMPACT_ANALYSIS.md §4, §11]

### Limitations

**17. What are the biggest limitations?**
- The Trishuli flood prediction is unvalidated.
- The training data has no Himalayan terrain, and test recall is low (0.324).
- Steep terrain can create false positives.
- Impact rules are overlap only.
- The connectivity graph excludes footpaths and stops at the area boundary, which matters most for
  Bhainse.
- There is one post-event image, and the dashboard basemap needs internet.

[docs/TRISHULI_INFERENCE.md §7, docs/TRISHULI_IMPACT_ANALYSIS.md §11]

**18. What causes terrain-related false positives?**
Measured: about two thirds of Trishuli VV pixels exceed Kuro Siwo's 0.15 clamp because steep slopes
facing the radar are bright, and slopes facing away are dark. On validation, the model predicted
flood on dark sloped terrain in an event with no flood. At Trishuli, the "permanent water"
predictions form thin streaks on slopes that are dark on both dates. Hypothesis: radar shadow and
back-slope darkness resemble water; we have no slope/shadow mask yet to confirm or suppress it.
[docs/TRISHULI_INFERENCE.md §6, docs/TRAINING_EXPERIMENT.md §6]

**19. What happens if the Sentinel-1 acquisition is too late?**
Flood water can recede before the satellite passes, so the predicted extent can underestimate the
peak. Over Trishuli, one satellite revisits each track every 12 days, giving about three passes
per 12 days across tracks. We used the closest same-track post image, two days after the event. A
second track (descending, relative orbit 19) had its post image ten days later. If no same-track
before/after set exists, the pipeline stops rather than mixing geometries.
[docs/DATA_SOURCES.md §2–3]

**20. What would you improve with more training data?**
Train on more of Kuro Siwo and, critically, on labelled mountainous floods, since none of our
training events are Himalayan. We would also check whether flood/permanent-water confusion and
terrain false positives drop. That is a hypothesis to test, not a result. Other next steps: DEM- and
shadow-aware filtering, and independent Trishuli validation. [docs/TRISHULI_INFERENCE.md §7]

### Compliance and design

**21. How does VantageQ comply with the forbidden-data restriction?**
Production inputs are only Sentinel-1, the Copernicus DEM (used inside Sentinel Hub for
orthorectification), and historical pre-event OSM; training uses Kuro Siwo. No Copernicus EMS map
including EMSR927, no UNOSAT or other damage map, and no post-event OSM edit was used for inputs,
training, thresholds or model selection. EMSR927 is reserved for a separate evaluation-only
comparison that is not yet built. [README.md, docs/DATA_SOURCES.md]

**22. What parts are AI vs deterministic GIS?**
The AI component is the U-Net that predicts flood per pixel. Everything else is deterministic code:
- scene selection and preprocessing;
- tiled inference assembly;
- vectorising the flood map;
- road, bridge and building intersections;
- the road graph;
- the dashboard snapshot and report generation.

No language model writes any number or text in the outputs. [docs/ARCHITECTURE.md, backend/, ml/]

**23. What makes this different from just producing a flood mask?**
The mask is only the first step. VantageQ turns it into decision-oriented indicators: which road
sections, bridges and buildings overlap the predicted flood, and which settlements lose road
connectivity in a simplified graph. It delivers those in an interactive dashboard and a one-page
situation report generated from the same structured outputs, with limitations stated alongside.

**24. What would you do for a judge-selected new AOI/date?**
The building blocks are generic: catalogue search and same-track selection, shared-grid retrieval,
the frozen model, historical-OSM retrieval for any bbox and date, and the impact and graph
functions. But the current scripts are configured for Trishuli; there is no input form. We would:
1. set the new AOI, event date and track in the scripts;
2. confirm a same-track before/after set exists (CDSE credentials are needed for retrieval);
3. run inference;
4. fetch OSM dated before the event and run the impact analysis;
5. rebuild the snapshot and report.

The output would again be an unvalidated prediction. [scripts/infer_trishuli.py,
scripts/analyze_trishuli_impact.py]
