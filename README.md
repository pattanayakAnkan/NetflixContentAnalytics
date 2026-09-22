# 🎬 Netflix Content Analytics (2008–2021)

**A comprehensive college-level Data Analytics project exploring Netflix's global content strategy through catalog composition, geographic production, genre dynamics, audience targeting, and content acquisition patterns.**

| | |
|---|---|
| **Author** | Ankan |
| **Dataset** | Netflix Movies and TV Shows — Kaggle (shivamb) |
| **Tools** | Python · Pandas · NumPy · SciPy · Plotly · Streamlit |
| **Dashboard** | `streamlit run Ankan_NetflixContentAnalytics.py` |

---

## Project Overview

Netflix grew from 2 catalog titles added in 2008 to a peak of 2,016 additions in a single year (2019), spanning content from 123 countries across 42 genre categories. This project performs a full end-to-end data analytics pipeline on Netflix's publicly available content catalog to uncover the strategic patterns behind what the platform acquires, when it acquires it, and from where.

The entire project — data cleaning, feature engineering, exploratory data analysis, statistical analysis, KPI calculation, and interactive Streamlit dashboard — is implemented in a **single Python file**: `Ankan_NetflixContentAnalytics.py`.

---

## Problem Statement

Can the composition and evolution of Netflix's own catalog reveal the content strategy decisions that drove the platform's global growth between 2008 and 2021?

Specifically:
- How has Netflix's catalog grown over time, and when did that growth peak?
- Is Netflix shifting from a Movie-first to a TV Show-first strategy?
- Which countries and genres dominate the catalog, and are there regional content patterns?
- What is Netflix's true audience maturity profile — is it primarily an adult platform?
- How quickly does Netflix acquire content relative to its original release date?

---

## Objectives

### Main Objective
Perform a comprehensive exploratory and descriptive analysis of Netflix's content catalog (2008–2021) to characterise growth trajectory, content mix evolution, geographic strategy, genre dynamics, audience targeting, and acquisition behaviour — delivered through an interactive analytical dashboard.

### Specific Objectives
1. Clean and engineer the dataset — fix corrupted rows, parse dates, split mixed-type duration column, create derived features
2. Characterise catalog composition and growth — year-by-year additions, Movies vs. TV Shows trend
3. Analyse geographic content production — 123 countries, choropleth map, country-level type splits
4. Map genre landscape and co-occurrence — 42 genres, top pairs, genre trends 2015–2021
5. Profile audience targeting — classify 14 rating codes into Adult / Teen / Family tiers
6. Analyse movie runtime patterns — descriptive statistics, genre-level comparison, outlier documentation
7. Examine TV show season longevity — season count distribution, country breakdown
8. Quantify content acquisition lag — Movies vs. TV Shows comparison, trend over years
9. Calculate 25 headline KPIs — total titles, type ratio, top country, top genre, adult %, median lag, and more
10. Build and deploy an interactive 9-tab Streamlit dashboard with Plotly visualisations

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Netflix Movies and TV Shows |
| **Source** | Kaggle — shivamb/netflix-shows |
| **URL** | https://www.kaggle.com/datasets/shivamb/netflix-shows |
| **File** | `netflix_titles.csv` |
| **File size** | 3.2 MB |
| **Rows** | 8,807 |
| **Columns** | 12 |
| **Coverage** | Content added to Netflix: 2008-01-01 to 2021-09-25 |
| **Licence** | CC0 1.0 Public Domain |

### Column Descriptions

| Column | Type | Nulls | Description |
|---|---|---|---|
| `show_id` | string | 0 | Unique identifier (e.g. `s1`) |
| `type` | categorical | 0 | `"Movie"` or `"TV Show"` |
| `title` | string | 0 | Display title |
| `director` | string | 2,634 (29.9%) | Director name(s), comma-separated |
| `cast` | string | 825 (9.4%) | Actor list, comma-separated |
| `country` | string | 831 (9.4%) | Production country/countries |
| `date_added` | string→datetime | 10 (0.1%) | Date added to Netflix catalog |
| `release_year` | integer | 0 | Original release year (1925–2021) |
| `rating` | categorical | 4 (+ 3 corrupted) | Content maturity rating |
| `duration` | string | 3 (corrupted) | `"X min"` for Movies, `"X Season(s)"` for TV Shows |
| `listed_in` | string | 0 | 1–3 comma-separated genre tags |
| `description` | string | 0 | 1–3 sentence synopsis |

### Known Data Quality Issues (all fixed in pipeline)

| Issue | Rows | Fix Applied |
|---|---|---|
| `rating` column contains duration values (CSV column shift) | 3 | Moved to `duration`; set `rating = NaN` |
| `date_added` leading/trailing whitespace | Many | `str.strip()` before parsing |
| `rating` null values | 4 (+3 above) | Filled with `"NR"` (Not Rated) |
| `director` / `cast` / `country` nulls | 2,634 / 825 / 831 | Filled with `"Unknown"` |
| `duration` mixed units (min vs. Seasons) | All | Split into `runtime_minutes` and `seasons_count` |
| `date_added` stored as human-readable string | All | Parsed with `pd.to_datetime(format="%B %d, %Y")` |

---

## Technologies Used

| Category | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | All source code |
| Data Analysis | pandas | ≥ 2.0.0 | DataFrames, cleaning, aggregation, feature engineering |
| Numerical | NumPy | ≥ 1.24.0 | Array operations, `nan`, `where`, quantile arithmetic |
| Statistics | SciPy | ≥ 1.10.0 | `pearsonr`, `describe` |
| Visualisation | Plotly | ≥ 5.15.0 | All 21 interactive charts |
| Dashboard | Streamlit | ≥ 1.28.0 | 9-tab web dashboard |
| Dataset access | Kaggle CLI | ≥ 1.5.0 | Auto-download if CSV not present |
| Standard library | os, sys, pathlib, itertools, collections, subprocess, warnings | built-in | File I/O, combinatorics, utilities |

---

## Project Structure

```
project/
│
├── Ankan_NetflixContentAnalytics.py   ← Complete Python source + Streamlit dashboard
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
├── Ankan_ProjectReport.docx            ← Formal project report
└── netflix_titles.csv                  ← Dataset (download from Kaggle or auto-downloaded)
```

> **`Ankan_NetflixContentAnalytics.py`** contains the entire project in one file, organised into 17 clearly labelled sections across 42 functions:
>
> | Section | Content |
> |---|---|
> | 1 | Imports & configuration (colour palette, constants) |
> | 2 | Dataset acquisition (`ensure_dataset()`, Kaggle CLI fallback) |
> | 3 | Data loading (`load_raw_data()` — `@st.cache_data`) |
> | 4 | Data understanding (`data_understanding_report()`) |
> | 5 | Data cleaning (`clean_data()` — 4 targeted fixes + log) |
> | 6 | Missing value handling (`handle_missing_values()`) |
> | 7 | Duplicate handling (`check_duplicates()`) |
> | 8 | Data type correction (`correct_dtypes()`) |
> | 9 | Transformation & feature engineering (`transform_and_engineer()`) |
> | 10 | Exploded helper DataFrames (`build_exploded_frames()`) |
> | 11 | KPI calculation (`calculate_kpis()` — 25 KPIs) |
> | 12 | Outlier analysis (`outlier_report()`) |
> | 13 | Statistical analysis (`statistical_summary()`) |
> | 14 | 21 visualisation functions (`fig_*`) |
> | 15 | Insight generation (`generate_insights()`) |
> | 16 | Automated tests (`run_tests()` — 15 assertions) |
> | 17 | Streamlit dashboard (`main()` — 9-tab layout) |

---

## Data Analytics Workflow

### 1. Data Acquisition
`ensure_dataset()` checks whether `netflix_titles.csv` exists in the working directory. If not, it attempts to download it via the Kaggle CLI (`kaggle datasets download -d shivamb/netflix-shows --unzip`). If the CLI is unavailable, the user is prompted to place the file manually. No credentials are stored in the source code.

### 2. Data Loading
`load_raw_data()` reads the CSV using `pd.read_csv()` with an explicit `int32` dtype hint for `release_year`. The raw DataFrame is preserved as a reference snapshot before any modification. The function is decorated with `@st.cache_data` so it runs only once per session.

### 3. Data Cleaning
`clean_data()` applies four targeted, logged fixes:
- **Fix 1** — 3 rows where a duration value (`"66 min"`, `"74 min"`, `"84 min"`) was placed in the `rating` column due to a CSV column shift. Values moved to `duration`; `rating` set to `NaN`.
- **Fix 2** — Whitespace stripped from `date_added` strings to prevent datetime parse failures.
- **Fix 3** — 7 remaining `rating` nulls (4 original + 3 from Fix 1) filled with `"NR"` (Not Rated).
- **Fix 4** — Assert that `duration` has zero nulls after fixes; log any unexpected remainder.

### 4. Data Preprocessing
`handle_missing_values()` applies a column-specific strategy:
- `director`, `cast`, `country` → filled with `"Unknown"` (preserves rows for non-talent analysis)
- `date_added` → kept as `NaT`; excluded per-analysis with `.dropna(subset=["date_added_parsed"])`

`correct_dtypes()` converts `date_added` to `datetime64`, `type` and `rating` to `category` dtype, and `release_year` to `int16`.

`transform_and_engineer()` creates 11 derived columns:

| Derived Column | Source | Purpose |
|---|---|---|
| `runtime_minutes` | `duration` (Movies) | Numeric runtime for Movie analysis |
| `seasons_count` | `duration` (TV Shows) | Numeric season count |
| `added_year` | `date_added_parsed` | Temporal trend analysis |
| `added_month` | `date_added_parsed` | Seasonality analysis |
| `added_month_name` | `date_added_parsed` | Chart labels |
| `content_lag_years` | `added_year − release_year` | Acquisition lag analysis |
| `audience_tier` | `rating` | Grouped maturity tiers |
| `primary_country` | `country` (first value) | Single-country analysis & choropleth |
| `primary_genre` | `listed_in` (first value) | Single-genre assignment |
| `decade_released` | `release_year` | Decadal grouping |
| `date_added_parsed` | `date_added` | Parsed datetime |

### 5. Exploratory Data Analysis
12 analytical questions are answered through dedicated `fig_*` functions. Each function receives the filtered DataFrame and returns a Plotly figure. Analysis dimensions include:
- **Temporal** — growth trajectory, YoY rates, monthly seasonality
- **Geographic** — 123 countries, choropleth, country-level type splits
- **Categorical** — 42 genres, genre co-occurrence matrix, rating distribution
- **Distributional** — movie runtime histogram and genre-level box plots
- **Relational** — release year vs. acquisition lag scatter

### 6. Statistical Analysis
`statistical_summary()` computes:
- Full five-number summary + mean, std, skewness, kurtosis for movie runtimes
- Lag statistics (mean, median, std) separately for Movies and TV Shows
- **Pearson r = −0.984** (release year vs. content lag) — strong negative correlation: older-released titles have longer lags
- Genre concentration **HHI = 0.0636** — near-zero, indicating a diverse genre distribution
- Count of negative lags (14 titles added before their official release year)

### 7. KPI Calculation
`calculate_kpis()` computes 25 headline KPIs dynamically from the filtered DataFrame. All values update live when sidebar filters are applied. Key KPIs include total titles, type ratio, peak year, top country, top genre, average runtime, adult content %, median lag, and YoY growth rate.

### 8. Visualization
21 Plotly figures covering:
- Dual-axis growth timeline (bar + cumulative line)
- 100% stacked type-share bar
- Monthly addition heat-map (Year × Month)
- YoY growth rate bar
- World choropleth
- Grouped country bar
- Genre frequency bar + co-occurrence heatmap
- Genre trend multi-line
- Runtime histogram + genre box plot
- Audience tier donut + by-country stacked bar
- Content lag box plots + trend line + release-year scatter
- Director and actor bar charts
- TV season distribution + country breakdown
- Seasonal monthly bar

All charts use the Plotly dark template with a Netflix-inspired colour palette (Movies = blue, TV Shows = red).

### 9. Dashboard
A 9-tab Streamlit dashboard with global sidebar filters — see [Dashboard](#dashboard) section below.

### 10. Insight Generation
`generate_insights()` produces 12 structured insight cards, each containing a **Finding**, **Evidence** (with exact numbers from `kpis` and `stat_summary`), and **Implication**. Insights are rendered as expandable cards in the dashboard's Insights tab.

---

## Dashboard

### Sections (9 Tabs)

| Tab | Content |
|---|---|
| 🏠 Overview | Project brief, analytical questions, workflow diagram, test results |
| 📊 KPI Dashboard | 10 live KPI metric cards + 4 summary charts |
| 🗃 Dataset | Raw data sample (searchable), data quality report, column profiles, outlier report |
| 🔬 EDA | Content Mix · Geography · Genres · Duration & Ratings sub-tabs |
| 📈 Trends | Growth timeline, YoY rate, monthly heat-map, genre trends |
| 🗂 Segments | By Country · By Genre & Rating · By Talent sub-tabs |
| 🔗 Relationships | Content lag analysis, correlation, genre co-occurrence, TV seasons |
| 🔎 Data Explorer | Full free-form search and filter of all 8,807 rows |
| 💡 Insights | 12 written insight cards, data quality summary, statistical table, limitations |

### KPIs (all computed dynamically)

| KPI | Full-dataset value |
|---|---|
| Total Titles | 8,807 |
| Movies | 6,131 (69.6%) |
| TV Shows | 2,676 (30.4%) |
| Peak Addition Year | 2019 (2,016 titles) |
| Total Countries | 123 |
| Total Genres | 42 |
| Top Country | United States (3,690 titles) |
| Top Genre | International Movies (2,752 appearances) |
| Avg Movie Runtime | 99.6 min |
| Adult Content | 45.5% |
| Median Content Lag | 1 year |
| Latest YoY Growth | −20.3% (2020→2021) |

### Sidebar Filters

| Filter | Column | Behaviour |
|---|---|---|
| Content Type | `type` | Multiselect: Movie / TV Show |
| Year Added | `date_added` → `added_year` | Range slider: 2008–2021 |
| Country | `primary_country` | Multiselect: top 20 producing countries |
| Genre | `primary_genre` | Multiselect: top 20 genres |
| Audience Tier | `audience_tier` | Multiselect: Adult / Teen / Family & Kids / Unrated |

All KPIs, charts, and tables respond to filter changes. A live counter — **"Showing X of 8,807 titles"** — is displayed in the sidebar at all times.

---

## Installation

### Prerequisites
- Python 3.10 or later
- `netflix_titles.csv` in the project folder (or Kaggle credentials configured for auto-download)

### Step 1 — Clone or download the project

Place all four project files in one folder:
```
Ankan_NetflixContentAnalytics.py
requirements.txt
README.md
Ankan_ProjectReport.docx
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

**Activate on Windows:**
```bash
venv\Scripts\activate
```

**Activate on macOS / Linux:**
```bash
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Place the dataset

**Option A — Manual (recommended):**
Download `netflix_titles.csv` from https://www.kaggle.com/datasets/shivamb/netflix-shows and place it in the same folder as `Ankan_NetflixContentAnalytics.py`.

**Option B — Auto-download via Kaggle API:**
Configure your Kaggle credentials (`~/.kaggle/kaggle.json` or environment variable `KAGGLE_API_TOKEN`), then the app will download the file automatically on first run.

---

## Running the Dashboard

```bash
streamlit run Ankan_NetflixContentAnalytics.py
```

The dashboard opens at **http://localhost:8501** in your default browser.

To run on a specific port:
```bash
streamlit run Ankan_NetflixContentAnalytics.py --server.port 8502
```

To run in headless mode (e.g. on a server):
```bash
streamlit run Ankan_NetflixContentAnalytics.py --server.headless true
```

---

## Key Findings

All findings below are computed directly from `netflix_titles.csv` — no values are estimated or assumed.

### 1. Catalog growth peaked in 2019 and has since declined
Netflix added a peak of **2,016 titles in 2019**. Additions fell by **20.3%** in 2021 relative to 2020 — the first sustained decline in the dataset, consistent with COVID-19 production disruptions and a strategic shift away from bulk catalog acquisition.

### 2. Netflix is Movie-dominant but TV Shows are growing faster
The catalog is **69.6% Movies** (6,131) and **30.4% TV Shows** (2,676). However, the proportional share of TV Show additions has grown consistently year-on-year since 2016, indicating a strategic shift toward serialised content.

### 3. The United States dominates, but India is a distant #2
The US produced **3,690 titles** (2,752 Movies, 938 TV Shows). India is #2 with **1,046 titles** (11.9% of the total catalog) — ahead of the UK (806), Canada (445), and France (393). India's content is almost entirely Movie-driven (962 Movies vs. only 84 TV Shows).

### 4. Japan and South Korea follow a TV-first model
Japan contributed **199 TV Shows vs. 119 Movies** and South Korea **170 TV Shows vs. 61 Movies** — the only two countries in the top 8 where TV Shows outnumber Movies. Every other top-8 country is Movie-dominant. This reflects Anime (Japan) and K-Drama (South Korea) as TV-native formats.

### 5. Netflix is primarily an adult platform by content volume
**45.5%** of the catalog is adult-rated (TV-MA, R, or NC-17). TV-MA alone accounts for **36.4%** of all titles. Teen/General content (TV-14, TV-PG, PG-13) adds another 39.9%, leaving only **13.6%** for Family & Kids content.

### 6. Movie runtimes conform tightly to theatrical norms
Movie runtimes average **99.6 minutes** (median 98 min, std 28.3 min) with mild right skew (skewness = 0.2034). Genre-level averages reveal meaningful variation: Documentaries (81.6 min) and Children & Family (79.9 min) are notably shorter, while Dramas (113.1 min) and Action & Adventure (113.5 min) run longest.

### 7. TV Shows reach Netflix far faster than Movies
The median content acquisition lag is **0 years for TV Shows** (many are Netflix originals or same-year acquisitions) vs. **2 years for Movies**. Mean lags diverge further (2.30 yr vs. 5.73 yr) due to archival film acquisitions. 67.4% of TV Shows are added within 1 year of release; only 49.6% of Movies achieve this.

### 8. Release year and lag are strongly negatively correlated
Pearson r = **−0.984** (p ≈ 0.000): older films have far longer lags, pulling the overall mean lag (4.69 yr) well above the median (1 yr). The 27 titles released before 1960 average over 60 years of lag. For contemporary content, the median is the more informative measure.

### 9. 67% of TV Shows have only one season
**67.0%** of TV Show entries list exactly 1 season (average: 1.76 seasons). The distribution is heavily right-skewed; only a handful of shows reach 10+ seasons. This reflects a mix of cancelled originals, single-season licensed content, and ongoing series not yet renewed at the time of data collection.

### 10. Stand-Up Comedy peaked in 2018 and sharply declined
Stand-Up Comedy additions peaked at **89 titles in 2018**, then fell progressively to **17 titles in 2021** — a 81% decline from peak. Documentaries followed a similar pattern, peaking in 2017 and declining thereafter, suggesting a deliberate investment cycle rather than sustained genre strategy.

### 11. Dramas + International Movies is the dominant genre pairing
The most common genre co-occurrence is **(Dramas, International Movies)** with **1,483 co-occurrences**. The genre HHI of **0.0636** confirms a diverse distribution — no single genre dominates disproportionately once the origin-tag nature of "International Movies" is understood.

### 12. Indian talent dominates individual-level statistics
**Anupam Kher** leads all actors with **43 appearances** across the catalog. 8 of the top 15 most-appearing actors are Bollywood performers. **Rajiv Chilaka** leads directors with **22 titles** (primarily Indian children's animation). This directly reflects India's position as the #2 producing country.

---

## Limitations

1. **Dataset ends September 2021** — no analysis of post-2021 Netflix strategy, content additions, or catalog changes.
2. **No viewership or engagement data** — analysis reflects what content *exists*, not what is *popular* or *watched*.
3. **`director` is missing for 29.9% of titles** — director-level analysis covers only ~70% of the catalog; TV shows are severely underrepresented.
4. **`country` reflects production origin, not regional availability** — a US-produced title may not be on Netflix in every country.
5. **Netflix's genre taxonomy is non-standard** — `"International Movies"` is an origin marker, not a content genre; `"Dramas"` appears in both Movie and TV Show genres as separate tags. Genre comparisons across the taxonomy require careful interpretation.
6. **Removed titles are not tracked** — the dataset is a snapshot of what was available at collection time, not a complete historical log of all titles ever on Netflix.
7. **Multi-country titles are counted in each country** — co-production titles inflate individual country counts; only 831 titles with null country are excluded.
8. **The mean content lag is inflated by archival content** — 27 pre-1960 titles with 60–93 year lags significantly skew the mean. The median (1 year) is more representative for contemporary acquisitions.
9. **No revenue, cost, or subscriber data** — business-impact analysis is not possible from this dataset alone.

---

## Future Scope

1. **Extend with post-2021 data** — integrate newer Kaggle snapshots or the Netflix API to track post-2021 strategy shifts (password-sharing crackdown, ad-supported tier).
2. **Cross-platform comparison** — merge with similar datasets from Disney+, Amazon Prime, and HBO Max to compare content strategies across streaming platforms.
3. **IMDb/Rotten Tomatoes rating enrichment** — join on title + release year to analyse the relationship between content quality scores and catalog acquisition decisions.
4. **NLP on descriptions** — apply topic modelling (LDA) or keyword extraction to `description` to identify thematic clusters beyond the existing genre taxonomy.
5. **Recommendation system prototype** — content-based filtering using genre, cast, and description TF-IDF vectors as a natural extension of the genre co-occurrence analysis.
6. **Time-series forecasting** — fit a simple trend model to annual addition counts to project catalog growth trajectories.
7. **Geospatial enrichment** — augment country data with GDP, internet penetration, or Netflix subscriber counts by region to test whether content acquisition correlates with market size.

---

## References

1. **Primary dataset:**
   Bansal, S. (2021). *Netflix Movies and TV Shows* [Dataset]. Kaggle.
   https://www.kaggle.com/datasets/shivamb/netflix-shows
   Licence: CC0 1.0 Universal (Public Domain Dedication)

2. **Streamlit documentation:**
   Streamlit Inc. (2024). *Streamlit Docs*. https://docs.streamlit.io

3. **Plotly documentation:**
   Plotly Technologies Inc. (2024). *Plotly Python Graphing Library*. https://plotly.com/python/

4. **Pandas documentation:**
   The pandas development team. (2024). *pandas documentation*. https://pandas.pydata.org/docs/

5. **SciPy documentation:**
   Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261–272. https://docs.scipy.org/doc/scipy/

---

*Project completed as part of a college-level Data Analytics coursework submission.*
