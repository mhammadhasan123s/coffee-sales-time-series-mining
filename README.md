# Coffee Sales: Sequence Analysis & ARIMA Forecasting

> **Course:** STQD6414 — Data Mining
> **Assignment:** Report Assignment
> **Institution:** Universiti Kebangsaan Malaysia (UKM)
> **Team:** Liang Haizhu (P156872) · Dong Qianqi (P161798) · Zhang Ruixuan (P165398) · Mhamad Shhab Aldeen Hasan (P166175)
> **Dataset:** [Coffee Sales — Kaggle](https://www.kaggle.com/datasets/ihelon/coffee-sales/data)

---

## Research Questions

1. How do customer coffee preferences evolve over repeat purchases?
2. Which coffee categories drive long-term customer loyalty?
3. Can we forecast short-term weekly Latte sales with ARIMA?

---

## Repository Structure

```
stqd6414-coffee-sales/
├── data/
│   └── index_1.csv                  ← raw transaction data (3,636 rows)
├── R/
│   └── coffee_sales_analysis.R      ← SINGLE script: full pipeline
├── docs/
│   └── coffee_sale_report.docx      ← full written report
└── README.md
```

---

## Dataset

| Variable | Type | Description |
|---|---|---|
| `date` | character → Date | Transaction date (YYYY/M/D) |
| `datetime` | character | Full timestamp |
| `cash_type` | character → factor | card / cash |
| `card` | character | Anonymised card ID (NA for cash) |
| `money` | numeric | Transaction amount |
| `coffee_name` | character → factor | Coffee type purchased (8 categories) |

- **3,636 transactions** from **March 2024 to March 2025**
- **89 missing card values** — all from cash transactions (filled with "Cash")
- **Zero negative prices** (no outliers)
- **14 raw coffee name variants → 8 standardised** after trimws + title case

---

## Pipeline

```
Raw CSV → Cleaning → Sequence Analysis (TraMineR) → ARIMA Forecasting
```

| Section | Content | Key Output |
|---|---|---|
| 1 · Loading | Read CSV portably | Data profile |
| 2 · Cleaning | Date parsing, NA fill, outliers, name standardisation, factors | Clean dataset |
| 3 · Sequence Analysis | Filter card customers ≥2 purchases → build sequences → TraMineR plots | 3 visualisations |
| 4 · ARIMA Forecasting | ADF test → differencing → ACF/PACF → model comparison → 8-week forecast | Forecast table + plot |
| 5 · Summary | Findings, business recommendations | — |

---

## Key Findings

### Customer Behaviour (Sequence Analysis)

Customer purchase evolution follows 3 distinct stages:

| Stage | Purchases | Behaviour |
|---|---|---|
| Exploration | T1–T23 | Diverse choices, no dominant category |
| Concentration | T31–T71 | Latte and Hot Chocolate emerge as favourites |
| Loyalty | T79–T124 | Preferences solidify to 2–3 core types |

**Core loyalty categories:** Latte · Hot Chocolate · Espresso

### ARIMA Forecasting (Latte Weekly Sales)

| Item | Detail |
|---|---|
| Stationarity | Non-stationary (ADF p > 0.05) → first-order differencing (d=1) |
| Model selected | **ARIMA(2,1,1)** |
| AIC | Lowest among all candidates (~362.7) |
| Ljung-Box p | > 0.05 (residuals are white noise) |
| Forecast horizon | 8 weeks with 50% confidence interval |

**Why ARIMA(2,1,1) over ARIMA(2,1,2)?**
In ARIMA(2,1,2), the MA(2) coefficient had |coefficient| < standard error (t-stat < 2), indicating it only fits noise. Removing it gives ARIMA(2,1,1) — lower AIC, all coefficients significant, no redundancy.

---

## Fixes Applied to Original Script

| # | Issue | Fix |
|---|---|---|
| 1 | `setwd("D:/zahra/...")` hardcoded Windows path | Removed — use `setwd()` to the repo root |
| 2 | `choose.files()` interactive file picker | Replaced with `read.csv("index_1.csv", ...)` |
| 3 | **ADF comment: "p > 0.05, we reject H0"** — wrong | Fixed: p > 0.05 = **FAIL TO REJECT H0** → non-stationary |
| 4 | CI multiplier `0.69` for "50% CI" (gives ~51%) | Fixed to `0.6745` (exact 50% CI z-score) |
| 5 | No `par(mfrow=c(1,1))` reset after multi-plot blocks | Added resets throughout |
| 6 | NA state label unlabelled in legend | Added explicit "No Purchase" label |
| 7 | `install.packages()` inside script | Moved to safe `if (!requireNamespace(...))` pattern |
| 8 | Forecast table not printed — only plot shown | Added formatted `forecast_table` printout |

---

## New Skills Demonstrated in This Project

| Skill | Tool / Method |
|---|---|
| **Sequence data mining** | TraMineR: `seqdef()`, `seqdplot()`, `seqmsplot()`, `seqiplot()` |
| **Customer behaviour analysis** | Building personal purchase chains from transaction logs |
| **Time series analysis** | `ts()`, `adf.test()`, ACF/PACF interpretation |
| **ARIMA modelling** | Manual order selection + `auto.arima()` as benchmark |
| **Model comparison** | AIC criterion + Ljung-Box residual test |
| **Statistical forecasting** | `predict()` with confidence intervals |
| **Data wrangling** | `dplyr` group operations, `lubridate` date handling, `stringr` text cleaning |

---

## How to Run

### Requirements
- R ≥ 4.1
- Packages: `forecast`, `tseries`, `TraMineR`, `dplyr`, `lubridate`, `stringr`, `tidyr`, `tidyverse`
  (all auto-installed by the script)

### Steps
```r
setwd("path/to/stqd6414-coffee-sales")
source("R/coffee_sales_analysis.R")
```

---

## GitHub Repo Suggestions

**Name:** `stqd6414-coffee-sales-mining`

**Description:**
> Customer purchase sequence analysis (TraMineR) and ARIMA sales forecasting on a real coffee shop transaction dataset. Identifies loyalty evolution stages and forecasts weekly Latte demand 8 weeks ahead. STQD6414 Data Mining, UKM.
