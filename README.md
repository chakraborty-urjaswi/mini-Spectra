# Project Skylark — mini-Spectra: Drone Stockpile Volumetrics

> **Tagline:** End-to-end drone stockpile auditing — automatic toe detection, 95% confidence intervals, and book-vs-physical reconciliation.

---

## What this project does

Stockpile volume measurement is Skylark Drones' headline product for mining and cement (Tata Steel, UltraTech). This project reproduces that pipeline from scratch and then attacks the four cases where drone volumetrics still breaks:

| Module | Pain point addressed |
|--------|----------------------|
| **A. Toe & base-surface estimation** | Manual toe drawing creates operator-to-operator variation (±10–15%); uneven ground makes a flat base plane wrong |
| **B. Volume uncertainty (Monte Carlo)** | Every tool reports one number; auditors need ±% confidence intervals |
| **C. Book vs. physical reconciliation** | Is a 6% gap real loss or measurement noise? Statistical test gives the answer |
| **D. Flight-design study** | Doming error without GCPs; optimal overlap/height trade-off |
| **E. Messy-scene handling** | Moving trucks on piles create false height spikes |

📊 **Phase 1 Inspection & Synthetic Validation Report:** [`reports/phase1_results.md`](reports/phase1_results.md)

---

## Project structure

```
mini-Spectra/
├── data/
│   ├── raw/
│   │   ├── odm_samples/        # Downloaded ODM benchmark datasets
│   │   └── synthetic/          # Blender-rendered drone images (generated)
│   ├── processed/
│   │   ├── dsm/                # Digital Surface Models (GeoTIFF)
│   │   ├── orthophoto/         # Orthorectified map images
│   │   └── pointcloud/         # LAZ/LAS point clouds from ODM
│   └── ground_truth/           # Known volumes for validation
├── src/
│   ├── data/
│   │   ├── download.py         # Fetch ODM datasets + metadata
│   │   ├── inspect.py          # EXIF/image QA checks
│   │   └── clean.py            # Image cleaning pipeline
│   ├── pipeline/
│   │   ├── odm_runner.py       # Docker ODM wrapper
│   │   └── dsm_loader.py       # Load + validate DSM outputs
│   ├── analysis/
│   │   ├── toe_detection.py    # Automatic stockpile boundary (Module A)
│   │   ├── base_surface.py     # Ground interpolation under pile (Module A)
│   │   ├── volume.py           # Volume calculation (Modules A+B)
│   │   ├── uncertainty.py      # Monte Carlo error propagation (Module B)
│   │   └── reconciliation.py   # Book vs physical gap test (Module C)
│   └── utils/
│       ├── geo.py              # CRS helpers, reprojection
│       └── viz.py              # Standard plotting helpers
├── notebooks/
│   ├── 01_data_access.ipynb    # THIS PHASE
│   ├── 02_baseline_pipeline.ipynb
│   ├── 03_toe_and_base.ipynb
│   └── 04_uncertainty.ipynb
├── tests/
│   └── test_volume.py
├── reports/figures/
├── requirements.txt
└── README.md
```

---

## Phase plan

| Week | Milestone |
|------|-----------|
| 1–2  | **Baseline pipeline** — ODM datasets in, DSM out, flat-plane volume |
| 3–4  | **Synthetic yard generator** — Blender piles of known volume |
| 5–7  | **Module A** — Automatic toe + base surface, operator comparison |
| 8–9  | **Module B+C** — Uncertainty bands + reconciliation test |
| 10   | **Module D** — Flight-design doming study |
| 11–12| Dashboard + write-up |

---

## Data sources

1. **ODM benchmark datasets** — `OpenDroneMap/ODMdata` (GitHub): Brighton Beach, Toledo, Aukerman, Wietrznia
2. **Community collection** — `natowi/photogrammetry_datasets` (CSV index, 350+ datasets)
3. **Synthetic** — Blender 3.x with drone-camera rig (Week 3–4)
4. **Real pile** — sub-250g drone flight at local construction site (Week 10)

---

## Key references

- DroneDeploy: accurate stockpile measurements in mining
- Propeller: defensible stockpile reconciliation
- Lowe et al. (2020): reducing dome errors through UAV flight design (ResearchGate)
- Carbonneau & Dietrich (2017): UAV-SfM for high-relief terrain (Wiley ESPL)
