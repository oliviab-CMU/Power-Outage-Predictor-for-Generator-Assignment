# Power Outage Forecasting and Backup Generator Pre-Positioning

A county-level hourly power-outage forecasting system for Michigan's 83 counties at 24-hour and 48-hour horizons, paired with an optimization step that recommends where to pre-position five mobile backup generators.

This repository contains the final submission for **95-828 Machine Learning for Problem Solving (Sections A & B)** at Carnegie Mellon University. Authors: Lisa Chung, Olivia Brown, Chloe Bonson, and Seongsu Kim.

The full project write-up — including the policy recommendation, ethical discussion, and detailed methodology — is in [`Final Report.docx`](Final%20Report.docx). The deep technical walkthrough of the modeling pipeline is in [`MODEL.md`](MODEL.md).

---

## Problem

Severe-weather-driven power outages are forecast at the **county-hour** level for **Michigan's 83 counties** under two constraints:

1. **No future weather is available.** The model only sees past outages and observed weather up to the prediction time.
2. **The target is heavily sparse.** About 70.5% of county-hour entries are zero. The 10% highest-volume hours account for 53.3% of all outages — predicting the absence of outages is easy, predicting the spikes is not.

The training period is **April 1 – June 30, 2023** (2,161 hours) with **109 weather features** per county. Two test sets are provided: 24-hour and 48-hour forecast windows, starting at 2023-06-30 01:00.

The deliverable has two parts:

| Part | Output | File |
|---|---|---|
| **I — Forecasting** | Non-negative outage predictions for every (county, hour) in the 24h and 48h test windows | `pred_24h.csv`, `pred_48h.csv` |
| **II — Generator placement** | Five FIPS codes for pre-positioning mobile backup generators (1,000-household capacity each) | `generator_counties.txt` |

---

## Solution summary

The final forecasting model is a **log-space Ridge stacking ensemble** of three complementary models:

| Model | Role | Validation 24h RMSE | Validation 48h RMSE |
|---|---|---|---|
| **LightGBM (MIMO)** | Strongest on engineered tabular features | 21.82 | 22.85 |
| **Temporal Convolutional Network (TCN)** | Captures sequential structure directly | 22.48 | 22.92 |
| **SARIMAX + weather exog** | Classical anchor, best at spikes individually | 33.50 | 35.89 |
| **Zero baseline** | Sanity check (predict 0 always) | 24.23 | 25.52 |
| **Ensemble (final)** | Log-Ridge stacking of the three above | **14.24** | **19.18** |

The ensemble is selected as the submission because it beats every individual model on both horizons and is most stable across the 24h → 48h degradation. SARIMAX looks worse than zero in average RMSE (it predicts small nonzero values during quiet hours), but is still useful in the ensemble because it captures spikes better than the others — its spike RMSE is 352.95 vs. 499.73 (TCN) and 476.13 (LightGBM).

For **spike events** (county-hours where truth > 162 outages, the 90th percentile of non-zero validation values), the ensemble's spike RMSE is 254.76 — roughly half the zero baseline's 519.29. This is the number that matters for operational decisions.

### Recommended generator counties

```
[26115, 26163, 26125, 26093, 26099]
Monroe, Wayne, Oakland, Livingston, Macomb
```

Ranking is based on a **spike-aware priority score** that combines (a) households one generator could actually serve under its 1,000-HH capacity and (b) a surge factor comparing predicted outages to the recent 20-day baseline. The traditional (capacity-only) score produces the same top-5 (rank correlation 0.98 with the spike-aware score), so the recommendation is robust to the scoring formula.

---

## Repository layout (what gets pushed to GitHub)

```
.
├── README.md                ← you are here
├── MODEL.md                 ← technical deep-dive on the modeling pipeline
├── Final Report.docx        ← full project write-up (problem, results, policy, ethics)
├── outage_forecast.ipynb    ← final pipeline that produces the submission artifacts
└── demo.ipynb               ← original course starter notebook
```

The pipeline expects `data/train.nc`, `data/test_24h_demo.nc`, and `data/test_48h_demo.nc` to be present locally — the data files are too large to commit and the demo test files contain random noise (used only for format sanity-checks). Output artifacts (`pred_24h.csv`, `pred_48h.csv`, `generator_counties.txt`) are written to a `results/` subdirectory by the notebook.

### Submission file format

Both `pred_24h.csv` and `pred_48h.csv` must satisfy these sanity checks (verified inside `outage_forecast.ipynb`):

- Columns: `timestamp, location, pred`
- 24h file: **1,992 rows** (24 hours × 83 counties)
- 48h file: **3,984 rows** (48 hours × 83 counties)
- All `pred` values are non-negative floats
- `generator_counties.txt` contains a Python list literal of 5 FIPS codes

---

## How to run

The full pipeline lives in **`outage_forecast.ipynb`** and runs end-to-end in a CUDA-capable Jupyter environment.

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate power-outage-ml
```

Key dependencies: `python=3.11`, `numpy<2`, `pytorch` (with CUDA 12.1), `lightgbm`, `xgboost`, `statsmodels`, `xarray`, `netCDF4`, `scikit-learn`, `matplotlib`.

### 2. Place the data

The notebook expects:

```
data/
  train.nc
  test_24h_demo.nc
  test_48h_demo.nc
```

### 3. Run the notebook

Open `outage_forecast.ipynb` in Jupyter and run all cells top-to-bottom. The notebook has 11 numbered sections:

| # | Section | What it does |
|---|---|---|
| 1 | Setup & dependencies | Installs packages, picks CUDA if available |
| 2 | Load data | Reads NetCDF, prints shapes & sparsity |
| 3 | EDA | Saves `eda.png`, `eda_overview.png` |
| 4 | Feature engineering | PCA, lags, rolling stats, storm flag, time encodings |
| 5 | Model A: LightGBM MIMO | Tweedie loss, per-county multi-output regressors |
| 6 | Model B: TCN | Dilated causal convolutions, Huber loss in log space |
| 7 | Model C: SARIMAX | Seasonal (2,1,2)(1,0,1)[24] + weather PC1 as exogenous |
| 8 | Validation + ensemble | Log-space Ridge stacking, RMSE + spike-RMSE |
| 9 | Retrain on full data | Refits all three models using all 2,161 hours |
| 10 | Save submissions | Writes `pred_24h.csv`, `pred_48h.csv` |
| 11 | Generator placement | Computes both scores, saves top-5 FIPS lists |

All outputs land in a `results/` subdirectory created by the notebook.

---

## What the model is good at (and not)

This matters for how the output should be used:

- ✅ **Reliable for the absence of significant outages** — the vast majority of county-hours, the model is highly accurate because the target is zero.
- ⚠️ **Provides a weak but real ranking signal** for which counties face elevated risk during an event.
- ❌ **Not a precise quantitative forecast of outage magnitude during storms.** The spike RMSE of 254.76 is roughly 18× the overall RMSE of 14.24 — a decision-maker who treats the overall number as a measure of storm-time reliability will overestimate the model's usefulness.
- ❌ **No forward-looking weather.** The model can only react to a storm after it has started affecting the grid. The biggest single improvement for any operational deployment would be to feed in NWP forecasts for the 5–7 day horizon.

The pre-positioning recommendation should be treated as a **relative prioritization signal**, not a precise estimate. We pair it with the recommendation to incorporate local infrastructure (hospitals, shelters, substation locations) before any real deployment.

---

## Ethics & responsible use

This is a resource-allocation model — its output decides which communities get backup power during emergencies. Several limitations have direct ethical consequences and should be considered before any operational use:

- **Single-season training data.** The model was trained on three months of spring–summer 2023, one storm season. Communities whose outage histories are underrepresented in that window — for whatever reason, including historical underinvestment in grid maintenance or data collection gaps — will be systematically deprioritized if the model is treated as ground truth.
- **County-level granularity hides neighborhood inequality.** A forecast assigned to Wayne County cannot distinguish between neighborhoods with robust infrastructure and low-income communities with aging grid assets and fewer recovery resources. The model has no demographic awareness, and aggregating to the county can mask where the help is most needed *within* a county.
- **The headline RMSE is misleading.** The overall RMSE of 14.24 is dominated by the 70% of county-hours where outages are zero. A decision-maker who interprets it as a measure of forecast reliability during a storm will dramatically over-trust the model precisely when reliability matters most. The spike RMSE of 254.76 is the honest number for operational decisions.
- **No causal model of vulnerability.** The ranking reflects historical outage patterns, not underlying fragility. Two counties with similar predicted outages may have very different real-world need based on population vulnerability, critical infrastructure, and recovery capacity — none of which the model sees.

**Mitigations built into the design:**

- The **surge adjustment** in the generator scoring formula explicitly prevents the ranking from being dominated by the largest counties, so smaller counties facing unusually severe events remain competitive.
- **Distributing** one generator to each of five counties (rather than concentrating) hedges against forecast error and broadens geographic coverage.
- The recommendation is framed as a **relative prioritization signal** paired with an explicit instruction to supplement with local infrastructure data (hospitals, shelters, nursing homes) before deployment.
- The final report (section 5.8) discusses these issues in more depth and recommends that any operational deployment involve domain experts, utility stakeholders, and affected communities — not treat the model as a self-sufficient authority.

See [`Final Report.docx`](Final%20Report.docx) → Section 5.8 for the full ethical discussion.

---

## Results on the demo test sets

The `test_24h_demo.nc` and `test_48h_demo.nc` files are **noise** provided by the course staff for sanity-checking the submission format only — they are not the real test set. The real test set is hidden. RMSEs on the demo files do not reflect the actual evaluation.

The RMSEs quoted in the table above are from the **validation split** (last 324 hours of the training period, 2023-06-16 13:00 → 2023-06-30 00:00), which is the cleanest estimate of generalization to the real hidden test set.

---

## References

- Final report: [`Final Report.docx`](Final%20Report.docx)
- Pipeline: [`outage_forecast.ipynb`](outage_forecast.ipynb)
- Technical walkthrough: [`MODEL.md`](MODEL.md)
- Starter notebook: [`demo.ipynb`](demo.ipynb)
- Course: 95-828 Machine Learning for Problem Solving, Carnegie Mellon University
- See `Final Report.docx` → References for the full citation list.
