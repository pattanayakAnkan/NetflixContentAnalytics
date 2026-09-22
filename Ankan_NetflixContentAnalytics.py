# =============================================================================
# Ankan_NetflixContentAnalytics.py
# Netflix Content Analytics — College Data Analytics Project
# Author  : Ankan
# Dataset : https://www.kaggle.com/datasets/shivamb/netflix-shows
# Run     : streamlit run Ankan_NetflixContentAnalytics.py
# =============================================================================

# =============================================================================
# SECTION 1 — IMPORTS & CONFIGURATION
# =============================================================================
import warnings
import pathlib
import itertools
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent.resolve()
DATA_FILE   = BASE_DIR / "netflix_titles.csv"
KAGGLE_SLUG = "shivamb/netflix-shows"

# ── Colour palette (Netflix-inspired) ────────────────────────────────────────
CLR_RED     = "#E50914"
CLR_BLUE    = "#2196F3"
CLR_GOLD    = "#F5C518"
CLR_GREEN   = "#4CAF50"
CLR_PURPLE  = "#9C27B0"
CLR_BG      = "#0E1117"
CLR_SURFACE = "#1E1E2E"
CLR_TEXT    = "#FAFAFA"
CLR_MUTED   = "#888888"

MOVIE_CLR   = CLR_BLUE
TV_CLR      = CLR_RED

PLOTLY_TEMPLATE = "plotly_dark"

# ── Rating tier mapping ───────────────────────────────────────────────────────
ADULT_RATINGS  = {"TV-MA", "R", "NC-17"}
TEEN_RATINGS   = {"TV-14", "TV-PG", "PG-13"}
FAMILY_RATINGS = {"TV-G", "TV-Y", "TV-Y7", "TV-Y7-FV", "G", "PG"}

TIER_COLOURS = {
    "Adult":         CLR_RED,
    "Teen / General": CLR_BLUE,
    "Family & Kids": CLR_GREEN,
    "Unrated / NR":  CLR_MUTED,
}

# =============================================================================
# SECTION 2 — DATASET ACQUISITION
# =============================================================================

def _try_kaggle_download() -> bool:
    """Attempt to download the dataset via kaggle CLI (no credentials in code)."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
             "--unzip", "-p", str(BASE_DIR)],
            capture_output=True, text=True, timeout=120
        )
        return (BASE_DIR / "netflix_titles.csv").exists()
    except Exception:
        pass
    # Try common Anaconda/conda kaggle.exe locations (platform-aware)
    import os, sys
    # conda/anaconda Scripts dirs for Windows
    for base in [
        pathlib.Path(sys.prefix) / "Scripts",
        pathlib.Path(os.path.expanduser("~")) / "anaconda3" / "Scripts",
        pathlib.Path(os.path.expanduser("~")) / "miniconda3" / "Scripts",
        pathlib.Path("C:/anaconda3/Scripts"),
        pathlib.Path("C:/ProgramData/anaconda3/Scripts"),
        # macOS / Linux conda
        pathlib.Path(sys.prefix) / "bin",
        pathlib.Path(os.path.expanduser("~")) / "anaconda3" / "bin",
        pathlib.Path(os.path.expanduser("~")) / "miniconda3" / "bin",
    ]:
        exe_name = "kaggle.exe" if sys.platform == "win32" else "kaggle"
        exe_path = base / exe_name
        if exe_path.exists():
            try:
                subprocess.run(
                    [str(exe_path), "datasets", "download", "-d", KAGGLE_SLUG,
                     "--unzip", "-p", str(BASE_DIR)],
                    capture_output=True, text=True, timeout=120
                )
                if (BASE_DIR / "netflix_titles.csv").exists():
                    return True
            except Exception:
                pass
    return False


def ensure_dataset() -> pathlib.Path:
    """Return path to netflix_titles.csv, downloading if necessary."""
    if DATA_FILE.exists():
        return DATA_FILE
    st.info("Dataset not found locally. Attempting Kaggle download …")
    if _try_kaggle_download():
        st.success("Download complete.")
        return DATA_FILE
    st.error(
        "Could not locate **netflix_titles.csv**.\n\n"
        "Please download it manually from "
        "https://www.kaggle.com/datasets/shivamb/netflix-shows "
        "and place it in the same folder as this script, then refresh."
    )
    st.stop()

# =============================================================================
# SECTION 3 — DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_raw_data(path: pathlib.Path) -> pd.DataFrame:
    """Load CSV and return raw DataFrame."""
    df = pd.read_csv(path, dtype={"release_year": "int32"}, keep_default_na=True)
    return df

# =============================================================================
# SECTION 4 — DATA UNDERSTANDING
# =============================================================================

def data_understanding_report(df: pd.DataFrame) -> dict:
    """Return a structured summary of the raw DataFrame."""
    report = {}
    report["shape"]         = df.shape
    report["columns"]       = list(df.columns)
    report["dtypes"]        = df.dtypes.to_dict()
    report["null_counts"]   = df.isnull().sum().to_dict()
    report["null_pct"]      = (df.isnull().mean() * 100).round(2).to_dict()
    report["unique_counts"] = df.nunique().to_dict()
    report["type_vc"]       = df["type"].value_counts().to_dict()
    report["rating_vc"]     = df["rating"].value_counts(dropna=False).to_dict()
    return report


# =============================================================================
# SECTION 5 — DATA CLEANING
# =============================================================================

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Apply all targeted fixes and return (cleaned_df, cleaning_log).
    Cleaning log: list of dicts describing each fix.
    """
    df = df.copy()
    log = []

    # ── Fix 1: Corrupted rating rows (duration in rating field) ───────────────
    bad_mask = df["rating"].str.match(r"^\d+ min$", na=False)
    n_bad = bad_mask.sum()
    if n_bad > 0:
        df.loc[bad_mask, "duration"] = df.loc[bad_mask, "rating"]
        df.loc[bad_mask, "rating"]   = np.nan
        log.append({
            "fix":      "Corrupted rating rows",
            "column":   "rating / duration",
            "rows":     int(n_bad),
            "action":   "Moved duration value from 'rating' to 'duration'; set rating = NaN",
            "show_ids": df.loc[bad_mask, "show_id"].tolist() if n_bad > 0 else [],
        })

    # ── Fix 2: Strip whitespace from date_added ───────────────────────────────
    df["date_added"] = df["date_added"].str.strip()
    log.append({
        "fix":    "date_added whitespace",
        "column": "date_added",
        "rows":   int(df["date_added"].notna().sum()),
        "action": "Applied str.strip() to remove leading/trailing whitespace",
    })

    # ── Fix 3: Fill remaining rating nulls with 'NR' ─────────────────────────
    n_null_rating = int(df["rating"].isnull().sum())
    df["rating"] = df["rating"].fillna("NR")
    log.append({
        "fix":    "Null rating values",
        "column": "rating",
        "rows":   n_null_rating,
        "action": f"Filled {n_null_rating} null values with 'NR' (Not Rated)",
    })

    # ── Fix 4: Assert duration is fully populated ─────────────────────────────
    n_null_dur = int(df["duration"].isnull().sum())
    if n_null_dur > 0:
        log.append({
            "fix":    "Duration nulls remain",
            "column": "duration",
            "rows":   n_null_dur,
            "action": f"WARNING: {n_null_dur} duration nulls remain after fix",
        })

    return df, log


# =============================================================================
# SECTION 6 — MISSING VALUE HANDLING
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or retain nulls per documented strategy."""
    df = df.copy()
    df["director"] = df["director"].fillna("Unknown")
    df["cast"]     = df["cast"].fillna("Unknown")
    df["country"]  = df["country"].fillna("Unknown")
    # date_added nulls (10 rows) kept as NaT after parsing — handled per-analysis
    return df


# =============================================================================
# SECTION 7 — DUPLICATE HANDLING
# =============================================================================

def check_duplicates(df: pd.DataFrame) -> dict:
    """Check and log duplicate status at three levels."""
    return {
        "full_duplicates":       int(df.duplicated().sum()),
        "show_id_duplicates":    int(df.duplicated(subset=["show_id"]).sum()),
        "title_type_duplicates": int(df.duplicated(subset=["title", "type"]).sum()),
    }


# =============================================================================
# SECTION 8 — DATA TYPE CORRECTION
# =============================================================================

def correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate analytical types."""
    df = df.copy()
    df["date_added_parsed"] = pd.to_datetime(
        df["date_added"], format="%B %d, %Y", errors="coerce"
    )
    df["release_year"] = df["release_year"].astype("int16")
    df["type"]         = df["type"].astype("category")
    df["rating"]       = df["rating"].astype("category")
    return df


# =============================================================================
# SECTION 9 — DATA TRANSFORMATION & FEATURE ENGINEERING
# =============================================================================

def transform_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived columns needed for analysis."""
    df = df.copy()

    # Numeric duration split
    movie_mask = df["type"] == "Movie"
    tv_mask    = df["type"] == "TV Show"

    df["runtime_minutes"] = np.nan
    df["seasons_count"]   = np.nan

    df.loc[movie_mask, "runtime_minutes"] = (
        df.loc[movie_mask, "duration"]
          .str.extract(r"(\d+)")[0]
          .astype(float)
    )
    df.loc[tv_mask, "seasons_count"] = (
        df.loc[tv_mask, "duration"]
          .str.extract(r"(\d+)")[0]
          .astype(float)
    )

    # Temporal features
    df["added_year"]       = df["date_added_parsed"].dt.year.astype("Int16")
    df["added_month"]      = df["date_added_parsed"].dt.month.astype("Int8")
    df["added_month_name"] = df["date_added_parsed"].dt.strftime("%b")

    # Content lag
    df["content_lag_years"] = (
        df["added_year"].astype("float") - df["release_year"].astype("float")
    )

    # Audience tier
    def _tier(r):
        if r in ADULT_RATINGS:  return "Adult"
        if r in TEEN_RATINGS:   return "Teen / General"
        if r in FAMILY_RATINGS: return "Family & Kids"
        return "Unrated / NR"

    df["audience_tier"] = df["rating"].astype(str).map(_tier)

    # Primary country (first listed)
    df["primary_country"] = df["country"].str.split(",").str[0].str.strip()
    df.loc[df["primary_country"] == "Unknown", "primary_country"] = np.nan

    # Primary genre (first listed)
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()

    # Decade of release
    df["decade_released"] = (df["release_year"] // 10 * 10).astype(str) + "s"

    return df


# =============================================================================
# SECTION 10 — EXPLODED HELPER DATAFRAMES
# =============================================================================

@st.cache_data(show_spinner=False)
def build_exploded_frames(df: pd.DataFrame) -> dict:
    """Build and cache all exploded helper DataFrames."""
    frames = {}

    # Exploded countries (exclude Unknown)
    ec = (
        df.assign(country=df["country"].str.split(","))
          .explode("country")
    )
    ec["country"] = ec["country"].str.strip()
    ec = ec[ec["country"] != "Unknown"]
    frames["country"] = ec

    # Exploded genres
    eg = (
        df.assign(genre=df["listed_in"].str.split(","))
          .explode("genre")
    )
    eg["genre"] = eg["genre"].str.strip()
    frames["genre"] = eg

    # Exploded cast (exclude Unknown)
    _cast_sub = df[df["cast"] != "Unknown"]
    ec2 = (
        _cast_sub
          .assign(actor=_cast_sub["cast"].str.split(","))
          .explode("actor")
    )
    ec2["actor"] = ec2["actor"].str.strip()
    frames["cast"] = ec2

    # Exploded directors (exclude Unknown)
    _dir_sub = df[df["director"] != "Unknown"]
    ed = (
        _dir_sub
          .assign(director_name=_dir_sub["director"].str.split(","))
          .explode("director_name")
    )
    ed["director_name"] = ed["director_name"].str.strip()
    frames["director"] = ed

    return frames


# =============================================================================
# SECTION 11 — KPI CALCULATION
# =============================================================================

def _safe_int(v, default=0):
    """Convert v to int, returning default if v is NaN or None."""
    try:
        f = float(v)
        return default if (f != f) else int(f)  # f != f is True only for NaN
    except (TypeError, ValueError):
        return default


def _safe_float(v, ndigits=2, default=0.0):
    """Round v to ndigits, returning default if v is NaN or None."""
    try:
        f = float(v)
        return default if (f != f) else round(f, ndigits)
    except (TypeError, ValueError):
        return default


def calculate_kpis(df: pd.DataFrame, frames: dict) -> dict:
    """Calculate all headline KPIs from the cleaned DataFrame.
    Safe against empty DataFrames produced by aggressive sidebar filtering.
    """
    ec = frames["country"]
    eg = frames["genre"]

    movies_df = df[df["type"] == "Movie"]
    tv_df     = df[df["type"] == "TV Show"]

    yearly = (
        df.dropna(subset=["added_year"])
          .groupby("added_year", observed=True)
          .size()
    )

    # YoY growth rates — guard against empty yearly Series
    if yearly.empty:
        peak_growth_year = None
        peak_growth_rate = None
        latest_year      = None
        latest_growth    = None
        peak_year        = None
        peak_year_count  = 0
    else:
        yoy = yearly.pct_change() * 100
        peak_growth_year  = int(yoy.idxmax()) if not yoy.dropna().empty else None
        peak_growth_rate  = _safe_float(yoy.max(), 1) if not yoy.dropna().empty else None
        latest_year       = int(yearly.index.max())
        latest_growth     = _safe_float(yoy.iloc[-1], 1) if len(yoy) > 1 else None
        peak_year         = int(yearly.idxmax())
        peak_year_count   = int(yearly.max())

    kpis = {
        "total_titles":           int(len(df)),
        "total_movies":           int((df["type"] == "Movie").sum()),
        "total_tv_shows":         int((df["type"] == "TV Show").sum()),
        "movie_pct":              _safe_float((df["type"] == "Movie").mean() * 100, 1),
        "tv_pct":                 _safe_float((df["type"] == "TV Show").mean() * 100, 1),
        "peak_year":              peak_year,
        "peak_year_count":        peak_year_count,
        "top_country":            ec["country"].value_counts().idxmax() if not ec.empty else "N/A",
        "top_genre":              eg["genre"].value_counts().idxmax()   if not eg.empty else "N/A",
        "avg_movie_runtime":      _safe_float(movies_df["runtime_minutes"].mean(), 1),
        "median_movie_runtime":   _safe_int(movies_df["runtime_minutes"].median()),
        "adult_pct":              _safe_float((df["audience_tier"] == "Adult").mean() * 100, 1),
        "median_lag_overall":     _safe_int(df["content_lag_years"].median()),
        "median_lag_movie":       _safe_int(movies_df["content_lag_years"].median()),
        "median_lag_tv":          _safe_int(tv_df["content_lag_years"].median()),
        "mean_lag_movie":         _safe_float(movies_df["content_lag_years"].mean(), 2),
        "mean_lag_tv":            _safe_float(tv_df["content_lag_years"].mean(), 2),
        "total_countries":        int(ec["country"].nunique()),
        "total_genres":           int(eg["genre"].nunique()),
        "peak_growth_year":       peak_growth_year,
        "peak_growth_rate":       peak_growth_rate,
        "latest_year":            latest_year,
        "latest_growth_rate":     latest_growth,
        "avg_seasons":            _safe_float(tv_df["seasons_count"].mean(), 2),
        "pct_tv_1season":         _safe_float((tv_df["seasons_count"] == 1).mean() * 100, 1),
    }
    return kpis


# =============================================================================
# SECTION 12 — OUTLIER ANALYSIS
# =============================================================================

def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising confirmed outliers and their disposition."""
    movies = df[df["type"] == "Movie"].copy()
    rt = movies["runtime_minutes"].dropna()
    q1, q3 = rt.quantile(0.25), rt.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outlier_rows = movies[
        (movies["runtime_minutes"] < lower) | (movies["runtime_minutes"] > upper)
    ][["show_id", "title", "runtime_minutes", "release_year", "listed_in"]].copy()
    outlier_rows["outlier_type"] = np.where(
        outlier_rows["runtime_minutes"] < lower, "Short", "Long"
    )
    outlier_rows["iqr_lower"] = round(lower, 1)
    outlier_rows["iqr_upper"] = round(upper, 1)
    outlier_rows["decision"]  = "Retained — legitimate content"
    return outlier_rows.reset_index(drop=True)


# =============================================================================
# SECTION 13 — STATISTICAL ANALYSIS
# =============================================================================

def statistical_summary(df: pd.DataFrame) -> dict:
    """Compute descriptive and relational statistics.
    Safe against empty or Movie-less DataFrames.
    """
    movies = df[df["type"] == "Movie"]
    tv     = df[df["type"] == "TV Show"]
    rt     = movies["runtime_minutes"].dropna()
    lag    = df["content_lag_years"].dropna()
    lag_m  = movies["content_lag_years"].dropna()
    lag_tv = tv["content_lag_years"].dropna()

    # Pearson r: release_year vs content_lag_years
    valid = df[["release_year", "content_lag_years"]].dropna()
    if len(valid) > 2:
        r, p = stats.pearsonr(valid["release_year"], valid["content_lag_years"])
    else:
        r, p = np.nan, np.nan

    # Genre concentration (Herfindahl-style)
    all_genres = []
    for g in df["listed_in"].dropna():
        all_genres.extend([x.strip() for x in g.split(",")])
    gc = Counter(all_genres)
    total_g = sum(gc.values())
    hhi = round(float((np.array(list(gc.values())) / total_g ** 2).sum()), 6) if total_g > 0 else 0.0
    # Correct HHI formula: sum of squares of shares
    if total_g > 0:
        shares = np.array(list(gc.values())) / total_g
        hhi = round(float((shares ** 2).sum()), 6)
    else:
        hhi = 0.0

    # Runtime stats — safe when rt is empty (no movies in filtered set)
    if len(rt) >= 2:
        s = stats.describe(rt)
        rt_n        = int(s.nobs)
        rt_mean     = _safe_float(rt.mean(), 2)
        rt_median   = _safe_float(rt.median(), 2)
        rt_std      = _safe_float(rt.std(), 2)
        rt_min      = _safe_float(rt.min(), 0)
        rt_max      = _safe_float(rt.max(), 0)
        rt_skew     = _safe_float(rt.skew(), 4)
        rt_kurt     = _safe_float(rt.kurtosis(), 4)
        rt_q1       = _safe_float(rt.quantile(0.25), 1)
        rt_q3       = _safe_float(rt.quantile(0.75), 1)
    else:
        rt_n = rt_mean = rt_median = rt_std = rt_min = rt_max = 0
        rt_skew = rt_kurt = rt_q1 = rt_q3 = 0.0

    return {
        "runtime_n":          rt_n,
        "runtime_mean":       rt_mean,
        "runtime_median":     rt_median,
        "runtime_std":        rt_std,
        "runtime_min":        rt_min,
        "runtime_max":        rt_max,
        "runtime_skew":       rt_skew,
        "runtime_kurtosis":   rt_kurt,
        "runtime_q1":         rt_q1,
        "runtime_q3":         rt_q3,
        "lag_mean_overall":   _safe_float(lag.mean(), 2),
        "lag_median_overall": _safe_float(lag.median(), 2),
        "lag_std_overall":    _safe_float(lag.std(), 2),
        "lag_mean_movie":     _safe_float(lag_m.mean(), 2),
        "lag_median_movie":   _safe_float(lag_m.median(), 2),
        "lag_mean_tv":        _safe_float(lag_tv.mean(), 2),
        "lag_median_tv":      _safe_float(lag_tv.median(), 2),
        "pearson_r":          _safe_float(r, 4),
        "pearson_p":          _safe_float(p, 6),
        "genre_hhi":          hhi,
        "n_negative_lags":    int((df["content_lag_years"] < 0).sum()),
    }


# =============================================================================
# SECTION 14 — ANALYSIS FUNCTIONS (return data + figure)
# =============================================================================

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_defaults(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=CLR_TEXT, size=12),
    )
    return fig


# ── A1: Catalog Growth ────────────────────────────────────────────────────────

def fig_catalog_growth(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["added_year"])
    yearly = sub.groupby(["added_year", "type"], observed=True).size().reset_index(name="count")
    total  = yearly.groupby("added_year", observed=True)["count"].sum().reset_index()
    total["cumulative"] = total["count"].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for t, clr in [("Movie", MOVIE_CLR), ("TV Show", TV_CLR)]:
        sub2 = yearly[yearly["type"] == t]
        fig.add_trace(
            go.Bar(x=sub2["added_year"], y=sub2["count"],
                   name=t, marker_color=clr, opacity=0.85),
            secondary_y=False
        )
    fig.add_trace(
        go.Scatter(x=total["added_year"], y=total["cumulative"],
                   name="Cumulative", mode="lines+markers",
                   line=dict(color=CLR_GOLD, width=2),
                   marker=dict(size=5)),
        secondary_y=True
    )
    fig.update_layout(
        title="Annual Catalog Additions & Cumulative Total",
        barmode="stack",
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_yaxes(title_text="Titles Added (annual)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Total",       secondary_y=True)
    return _fig_defaults(fig, 440)


# ── A2: Type Share Over Time ──────────────────────────────────────────────────

def fig_type_share(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["added_year"])
    tbl = (sub.groupby(["added_year", "type"], observed=True)
               .size()
               .reset_index(name="count"))
    pivot = tbl.pivot(index="added_year", columns="type", values="count").fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig = go.Figure()
    for t, clr in [("Movie", MOVIE_CLR), ("TV Show", TV_CLR)]:
        if t in pivot_pct.columns:
            fig.add_trace(go.Bar(
                x=pivot_pct.index, y=pivot_pct[t],
                name=t, marker_color=clr, opacity=0.85
            ))
    fig.update_layout(
        title="Content Type Share per Year (%)",
        barmode="relative",
        yaxis_title="% of Annual Additions",
        legend=dict(orientation="h", y=-0.15),
    )
    return _fig_defaults(fig)


# ── A3: Monthly Heat-Map ──────────────────────────────────────────────────────

def fig_monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["added_year", "added_month"])
    sub = sub[sub["added_year"] >= 2015]
    if sub.empty:
        # Return a blank figure with a note rather than crashing
        fig = go.Figure()
        fig.update_layout(title="Monthly Heat-Map — no data for selected filters")
        return _fig_defaults(fig, 380)
    tbl = (sub.groupby(["added_year", "added_month"])
               .size()
               .reset_index(name="count"))
    pivot = tbl.pivot(index="added_year", columns="added_month", values="count").fillna(0)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    # Rename only the columns that actually exist in the pivot
    pivot.columns = [month_names[int(c) - 1] for c in pivot.columns]

    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Year", color="Titles Added"),
        color_continuous_scale="YlOrRd",
        title="Monthly Content Addition Heat-Map (2015–2021)",
        text_auto=True,
    )
    fig.update_coloraxes(showscale=True)
    return _fig_defaults(fig, 380)


# ── A4: YoY Growth Rate ───────────────────────────────────────────────────────

def fig_yoy_growth(df: pd.DataFrame) -> go.Figure:
    sub    = df.dropna(subset=["added_year"])
    yearly = sub.groupby("added_year", observed=True).size()
    yoy    = (yearly.pct_change() * 100).dropna().reset_index()
    yoy.columns = ["year", "growth_pct"]

    colours = [CLR_GREEN if v >= 0 else CLR_RED for v in yoy["growth_pct"]]
    fig = go.Figure(go.Bar(
        x=yoy["year"], y=yoy["growth_pct"],
        marker_color=colours,
        text=yoy["growth_pct"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=CLR_MUTED)
    fig.update_layout(title="Year-over-Year Growth Rate (%)", yaxis_title="% Change")
    return _fig_defaults(fig)


# ── A5: World Choropleth ──────────────────────────────────────────────────────

def fig_choropleth(frames: dict) -> go.Figure:
    ec = frames["country"]
    counts = ec["country"].value_counts().reset_index()
    counts.columns = ["country", "count"]

    fig = px.choropleth(
        counts,
        locations="country",
        locationmode="country names",
        color="count",
        color_continuous_scale="Reds",
        title="Netflix Content Production by Country",
        hover_name="country",
        hover_data={"count": True},
        labels={"count": "Titles"},
    )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
    return _fig_defaults(fig, 460)


# ── A6: Top Countries Grouped Bar ────────────────────────────────────────────

def fig_top_countries(frames: dict, n: int = 15) -> go.Figure:
    ec = frames["country"]
    top = ec["country"].value_counts().head(n).index.tolist()
    sub = ec[ec["country"].isin(top)]
    tbl = sub.groupby(["country", "type"], observed=True).size().reset_index(name="count")
    total_order = (
        tbl.groupby("country", observed=True)["count"].sum()
           .sort_values(ascending=True).index.tolist()
    )
    fig = px.bar(
        tbl, x="count", y="country", color="type",
        orientation="h",
        color_discrete_map={"Movie": MOVIE_CLR, "TV Show": TV_CLR},
        title=f"Top {n} Content-Producing Countries",
        labels={"count": "Titles", "country": ""},
        category_orders={"country": total_order},
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.1), barmode="group")
    return _fig_defaults(fig, 520)


# ── A7: Genre Frequency Bar ───────────────────────────────────────────────────

def fig_genre_frequency(frames: dict, n: int = 20) -> go.Figure:
    eg = frames["genre"]
    gc = eg["genre"].value_counts().head(n).reset_index()
    gc.columns = ["genre", "count"]
    fig = px.bar(
        gc.sort_values("count"),
        x="count", y="genre", orientation="h",
        title=f"Top {n} Genres (after exploding multi-labels)",
        labels={"count": "Appearances", "genre": ""},
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_coloraxes(showscale=False)
    return _fig_defaults(fig, 560)


# ── A8: Genre Co-occurrence Heat-Map ─────────────────────────────────────────

def fig_genre_cooccurrence(df: pd.DataFrame, n: int = 15) -> go.Figure:
    pair_counter: Counter = Counter()
    top_genres = (
        df["listed_in"].dropna()
          .str.split(",")
          .explode()
          .str.strip()
          .value_counts()
          .head(n)
          .index.tolist()
    )
    for genres_str in df["listed_in"].dropna():
        genres = [g.strip() for g in genres_str.split(",")]
        genres = [g for g in genres if g in top_genres]
        for pair in itertools.combinations(sorted(genres), 2):
            pair_counter[pair] += 1

    matrix = pd.DataFrame(0, index=top_genres, columns=top_genres)
    for (g1, g2), cnt in pair_counter.items():
        matrix.loc[g1, g2] = cnt
        matrix.loc[g2, g1] = cnt

    fig = px.imshow(
        matrix,
        color_continuous_scale="Blues",
        title=f"Genre Co-occurrence Matrix (top {n} genres)",
        labels=dict(color="Co-occurrences"),
        text_auto=True,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return _fig_defaults(fig, 560)


# ── A9: Genre Trends Over Time ────────────────────────────────────────────────

def fig_genre_trends(df: pd.DataFrame, selected_genres: list = None) -> go.Figure:
    if selected_genres is None:
        selected_genres = [
            "International Movies", "Dramas", "Comedies",
            "Documentaries", "Stand-Up Comedy", "Korean TV Shows", "Anime Series"
        ]
    eg = (
        df.assign(genre=df["listed_in"].str.split(","))
          .explode("genre")
    )
    eg["genre"] = eg["genre"].str.strip()
    sub = eg[eg["genre"].isin(selected_genres) & eg["added_year"].notna()]
    sub = sub[sub["added_year"] >= 2015]
    tbl = sub.groupby(["added_year", "genre"]).size().reset_index(name="count")

    fig = px.line(
        tbl, x="added_year", y="count", color="genre",
        markers=True,
        title="Genre Trends Over Time (2015–2021)",
        labels={"added_year": "Year", "count": "Titles Added", "genre": "Genre"},
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    return _fig_defaults(fig, 420)


# ── A10: Movie Runtime Histogram ──────────────────────────────────────────────

def fig_runtime_histogram(df: pd.DataFrame) -> go.Figure:
    movies = df[df["type"] == "Movie"]["runtime_minutes"].dropna()
    mean_v   = movies.mean()
    median_v = movies.median()

    fig = px.histogram(
        movies, x=movies,
        nbins=60,
        title="Movie Runtime Distribution",
        labels={"x": "Runtime (minutes)", "count": "Number of Movies"},
        color_discrete_sequence=[MOVIE_CLR],
    )
    fig.add_vline(x=mean_v,   line_dash="dash",  line_color=CLR_GOLD,
                  annotation_text=f"Mean {mean_v:.0f} min",   annotation_position="top right")
    fig.add_vline(x=median_v, line_dash="dot",   line_color=CLR_GREEN,
                  annotation_text=f"Median {median_v:.0f} min", annotation_position="top left")
    return _fig_defaults(fig)


# ── A11: Runtime by Genre Box Plot ────────────────────────────────────────────

def fig_runtime_by_genre(df: pd.DataFrame) -> go.Figure:
    movies = df[df["type"] == "Movie"].copy()
    movies_eg = (
        movies.assign(genre=movies["listed_in"].str.split(","))
              .explode("genre")
    )
    movies_eg["genre"] = movies_eg["genre"].str.strip()
    top_genres = movies_eg["genre"].value_counts().head(10).index.tolist()
    sub = movies_eg[movies_eg["genre"].isin(top_genres)]

    order = (
        sub.groupby("genre")["runtime_minutes"]
           .median()
           .sort_values()
           .index.tolist()
    )
    fig = px.box(
        sub, x="runtime_minutes", y="genre",
        orientation="h",
        color="genre",
        title="Movie Runtime by Genre (Top 10 Genres)",
        labels={"runtime_minutes": "Runtime (minutes)", "genre": ""},
        category_orders={"genre": order},
    )
    fig.update_layout(showlegend=False)
    return _fig_defaults(fig, 500)


# ── A12: Audience Tier Donut ──────────────────────────────────────────────────

def fig_audience_tier_donut(df: pd.DataFrame) -> go.Figure:
    vc = df["audience_tier"].value_counts().reset_index()
    vc.columns = ["tier", "count"]
    colour_seq = [TIER_COLOURS.get(t, CLR_MUTED) for t in vc["tier"]]

    fig = px.pie(
        vc, names="tier", values="count",
        hole=0.45,
        title="Content Maturity Profile",
        color="tier",
        color_discrete_map=TIER_COLOURS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _fig_defaults(fig, 380)


# ── A13: Rating Tier by Country Stacked Bar ───────────────────────────────────

def fig_rating_by_country(df: pd.DataFrame, frames: dict, n: int = 8) -> go.Figure:
    ec = frames["country"]
    top_countries = ec["country"].value_counts().head(n).index.tolist()
    sub = ec[ec["country"].isin(top_countries)].copy()
    sub["audience_tier"] = sub["rating"].astype(str).map(
        lambda r: "Adult" if r in ADULT_RATINGS
        else ("Teen / General" if r in TEEN_RATINGS
              else ("Family & Kids" if r in FAMILY_RATINGS else "Unrated / NR"))
    )
    tbl = (sub.groupby(["country", "audience_tier"], observed=True)
               .size()
               .reset_index(name="count"))
    total = tbl.groupby("country", observed=True)["count"].sum().reset_index(name="total")
    tbl   = tbl.merge(total, on="country")
    tbl["pct"] = tbl["count"] / tbl["total"] * 100

    order = (
        tbl[tbl["audience_tier"] == "Adult"]
           .sort_values("pct", ascending=True)["country"].tolist()
    )

    fig = px.bar(
        tbl, x="pct", y="country", color="audience_tier",
        orientation="h", barmode="relative",
        color_discrete_map=TIER_COLOURS,
        title=f"Audience Rating Profile by Country (Top {n})",
        labels={"pct": "% of Country's Content", "country": ""},
        category_orders={"country": order},
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    return _fig_defaults(fig, 440)


# ── A14: Content Lag Box Plots ────────────────────────────────────────────────

def fig_lag_boxplots(df: pd.DataFrame) -> go.Figure:
    sub = df[df["content_lag_years"].between(-5, 50)].copy()
    fig = px.box(
        sub, x="type", y="content_lag_years",
        color="type",
        color_discrete_map={"Movie": MOVIE_CLR, "TV Show": TV_CLR},
        title="Content Acquisition Lag: Movies vs. TV Shows",
        labels={"content_lag_years": "Years after Release", "type": ""},
        points=False,
    )
    fig.update_layout(showlegend=False)
    return _fig_defaults(fig)


# ── A15: Median Lag Trend ─────────────────────────────────────────────────────

def fig_lag_trend(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["added_year", "content_lag_years"])
    sub = sub[sub["added_year"] >= 2015]
    tbl = (sub.groupby(["added_year", "type"], observed=True)["content_lag_years"]
               .median()
               .reset_index())
    tbl.columns = ["year", "type", "median_lag"]

    fig = px.line(
        tbl, x="year", y="median_lag", color="type",
        markers=True,
        color_discrete_map={"Movie": MOVIE_CLR, "TV Show": TV_CLR},
        title="Median Content Lag Trend by Year (2015–2021)",
        labels={"year": "Year Added", "median_lag": "Median Lag (years)", "type": ""},
    )
    fig.add_hline(y=0, line_dash="dash", line_color=CLR_MUTED, annotation_text="Same year")
    return _fig_defaults(fig)


# ── A16: Release Year vs Lag Scatter ─────────────────────────────────────────

def fig_lag_scatter(df: pd.DataFrame) -> go.Figure:
    sub = df[df["content_lag_years"].between(-5, 95)].dropna(
        subset=["release_year", "content_lag_years"]
    ).sample(min(3000, len(df)), random_state=42)

    fig = px.scatter(
        sub, x="release_year", y="content_lag_years",
        color="type",
        color_discrete_map={"Movie": MOVIE_CLR, "TV Show": TV_CLR},
        opacity=0.45,
        title="Release Year vs. Content Acquisition Lag",
        labels={
            "release_year": "Original Release Year",
            "content_lag_years": "Years Until Added to Netflix",
            "type": "",
        },
        hover_data=["title"],
    )
    fig.update_traces(marker_size=4)
    return _fig_defaults(fig)


# ── A17: Top Directors Bar ────────────────────────────────────────────────────

def fig_top_directors(frames: dict, n: int = 15) -> go.Figure:
    dc = frames["director"]["director_name"].value_counts().head(n).reset_index()
    dc.columns = ["director", "count"]

    fig = px.bar(
        dc.sort_values("count"),
        x="count", y="director", orientation="h",
        title=f"Top {n} Most Prolific Directors on Netflix",
        labels={"count": "Titles", "director": ""},
        color="count", color_continuous_scale="Blues",
    )
    fig.update_coloraxes(showscale=False)
    return _fig_defaults(fig, 480)


# ── A18: Top Actors Bar ───────────────────────────────────────────────────────

def fig_top_actors(frames: dict, n: int = 15) -> go.Figure:
    ac = frames["cast"]["actor"].value_counts().head(n).reset_index()
    ac.columns = ["actor", "count"]

    fig = px.bar(
        ac.sort_values("count"),
        x="count", y="actor", orientation="h",
        title=f"Top {n} Most Appearing Actors on Netflix",
        labels={"count": "Appearances", "actor": ""},
        color="count", color_continuous_scale="Reds",
    )
    fig.update_coloraxes(showscale=False)
    return _fig_defaults(fig, 480)


# ── A19: TV Seasons Distribution ─────────────────────────────────────────────

def fig_seasons_dist(df: pd.DataFrame) -> go.Figure:
    tv = df[df["type"] == "TV Show"]["seasons_count"].dropna().astype(int)
    vc = tv.value_counts().sort_index().reset_index()
    vc.columns = ["seasons", "count"]

    fig = px.bar(
        vc, x="seasons", y="count",
        title="TV Show Season Count Distribution",
        labels={"seasons": "Number of Seasons", "count": "Number of Shows"},
        color="count", color_continuous_scale="Reds",
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _fig_defaults(fig)


# ── A20: Seasonal Monthly Pattern ────────────────────────────────────────────

def fig_seasonal_bar(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["added_month"])
    monthly = sub.groupby("added_month").size().reset_index(name="count")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["month_name"] = monthly["added_month"].map(month_names)

    fig = px.bar(
        monthly, x="month_name", y="count",
        title="Total Content Additions by Calendar Month",
        labels={"month_name": "Month", "count": "Titles Added"},
        color="count", color_continuous_scale="YlOrRd",
        category_orders={"month_name": list(month_names.values())},
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _fig_defaults(fig)


# ── A21: Seasons by Country ───────────────────────────────────────────────────

def fig_seasons_by_country(df: pd.DataFrame, frames: dict, n: int = 8) -> go.Figure:
    tv    = df[df["type"] == "TV Show"][["show_id", "seasons_count"]].copy()
    ec    = frames["country"]
    tv_ec = ec[ec["type"] == "TV Show"][["show_id", "country"]].copy()
    top_c = tv_ec["country"].value_counts().head(n).index.tolist()
    sub   = tv_ec[tv_ec["country"].isin(top_c)].merge(tv, on="show_id", how="left")
    sub   = sub.dropna(subset=["seasons_count"])
    sub["season_tier"] = pd.cut(
        sub["seasons_count"],
        bins=[0, 1, 4, 100],
        labels=["1 Season", "2–4 Seasons", "5+ Seasons"],
    )
    tbl = sub.groupby(["country", "season_tier"]).size().reset_index(name="count")
    order = (
        sub.groupby("country")["seasons_count"]
           .count()
           .sort_values(ascending=True)
           .index.tolist()
    )
    fig = px.bar(
        tbl, x="count", y="country", color="season_tier",
        orientation="h", barmode="relative",
        title=f"TV Show Season Tiers by Country (Top {n})",
        labels={"count": "TV Shows", "country": ""},
        category_orders={"country": order},
        color_discrete_sequence=[CLR_GREEN, CLR_BLUE, CLR_RED],
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    return _fig_defaults(fig, 420)


# =============================================================================
# SECTION 15 — INSIGHT GENERATION
# =============================================================================

def generate_insights(kpis: dict, stat_data: dict) -> list:
    """Return a list of insight dicts: finding, evidence, implication.
    Parameter renamed from 'stats' to 'stat_data' to avoid shadowing
    the module-level 'import scipy.stats as stats'.
    """
    stats = stat_data  # local alias for backward-compatible f-string access
    insights = [
        {
            "emoji":      "📈",
            "finding":    "Netflix's catalog growth peaked in 2019 and has since decelerated.",
            "evidence":   f"The platform added {kpis['peak_year_count']:,} titles in {kpis['peak_year']}. "
                          f"By 2021 additions had dropped — the latest YoY rate was {kpis['latest_growth_rate']}%.",
            "implication":"The era of hyper-growth catalog building appears to have ended, "
                          "possibly reflecting COVID-19 production halts and a strategy shift "
                          "toward quality over quantity.",
        },
        {
            "emoji":      "🎬",
            "finding":    "Netflix remains Movie-dominant but TV Shows are growing proportionally faster.",
            "evidence":   f"Movies account for {kpis['movie_pct']}% of the catalog "
                          f"vs {kpis['tv_pct']}% for TV Shows. However TV Show additions "
                          "have grown at a faster rate year-on-year since 2016.",
            "implication":"A strategic shift toward serialised, binge-friendly content is detectable "
                          "in the data — consistent with Netflix's known original-series investment.",
        },
        {
            "emoji":      "🌍",
            "finding":    f"The United States dominates production but India is a distant #2 with over 1,000 titles.",
            "evidence":   f"Top country: {kpis['top_country']}. India accounts for ~12% of the full catalog, "
                          "ahead of UK, Canada, France, and Japan.",
            "implication":"Bollywood content is a deliberate, large-scale acquisition — not incidental. "
                          "Netflix has made a strategic bet on the Indian market.",
        },
        {
            "emoji":      "🇯🇵",
            "finding":    "Japan and South Korea are the only top-8 countries where TV Shows outnumber Movies.",
            "evidence":   "Japan: ~199 TV Shows vs ~119 Movies. South Korea: ~170 TV Shows vs ~61 Movies. "
                          "Every other top-8 country is Movie-dominant.",
            "implication":"Anime (Japan) and K-Drama (South Korea) follow a TV-first format model — "
                          "Netflix's content strategy adapts to regional production norms.",
        },
        {
            "emoji":      "🔞",
            "finding":    f"Netflix's catalog is majority adult-rated — {kpis['adult_pct']}% is Adult content.",
            "evidence":   f"TV-MA alone represents ~36% of all titles. Adult + Teen/General accounts for "
                          f"over 85% of the catalog. Family & Kids content is ~14%.",
            "implication":"Despite its 'something for everyone' branding, Netflix is overwhelmingly "
                          "an adult platform by content volume.",
        },
        {
            "emoji":      "⏱️",
            "finding":    f"Movies take significantly longer to reach Netflix than TV Shows.",
            "evidence":   f"Movies: mean lag = {kpis['mean_lag_movie']} yrs, median = {kpis['median_lag_movie']} yr. "
                          f"TV Shows: mean lag = {kpis['mean_lag_tv']} yrs, median = {kpis['median_lag_tv']} yr.",
            "implication":"TV Shows' near-zero median lag suggests many are Netflix originals or same-year "
                          "acquisitions. Movies include more archival/licensed content with inherent lag.",
        },
        {
            "emoji":      "📊",
            "finding":    f"Movie runtimes cluster tightly around the standard feature-film length.",
            "evidence":   f"Mean: {stats['runtime_mean']} min | Median: {stats['runtime_median']} min | "
                          f"Std Dev: {stats['runtime_std']} min. Skewness: {stats['runtime_skew']} "
                          f"(mild right tail from epic/archival films).",
            "implication":"Netflix's movie catalog mirrors theatrical norms. Documentaries are consistently "
                          "shorter (~82 min avg) while Dramas and Action run longest (~113 min avg).",
        },
        {
            "emoji":      "📉",
            "finding":    "Stand-Up Comedy peaked in 2018 and has declined sharply since.",
            "evidence":   "Stand-Up Comedy additions: 63 (2017) → 89 (2018) → 66 (2019) → 48 (2020) → 17 (2021).",
            "implication":"Netflix appears to have scaled back its stand-up investment after 2018. "
                          "Docuseries additions show a similar pattern — peaking and then declining.",
        },
        {
            "emoji":      "📺",
            "finding":    f"{kpis['pct_tv_1season']}% of TV Shows on Netflix have only one season.",
            "evidence":   f"Average seasons: {kpis['avg_seasons']}. The season count distribution "
                          "is heavily right-skewed; shows with 5+ seasons are rare.",
            "implication":"Netflix's TV catalog is broad but shallow — many shows were either cancelled, "
                          "are ongoing originals, or are licensed single-run series.",
        },
        {
            "emoji":      "🔗",
            "finding":    f"Release year and acquisition lag are negatively correlated (r = {stats['pearson_r']}).",
            "evidence":   f"Pearson r = {stats['pearson_r']} (p = {stats['pearson_p']}). "
                          "Older films have larger lags — the 27 pre-1960 titles average 60+ years of lag.",
            "implication":"The positive-skew in mean lag (4.69 yrs) is driven by archival content. "
                          "For contemporary content the median lag (1 yr) is the more representative figure.",
        },
        {
            "emoji":      "🎭",
            "finding":    f"'Dramas + International Movies' is the most common genre pairing with ~1,483 co-occurrences.",
            "evidence":   f"Top genre: {kpis['top_genre']}. Genre HHI = {stats['genre_hhi']:.4f} "
                          "(near-zero → diverse genre distribution).",
            "implication":"Netflix's genre taxonomy blends origin ('International') with tone ('Drama') — "
                          "the most common pairing reflects international drama acquisition, not a single genre.",
        },
        {
            "emoji":      "🇮🇳",
            "finding":    "Indian content is the most actor-concentrated by nationality on Netflix.",
            "evidence":   "8 of the top 15 most-appearing actors across the entire catalog are Bollywood performers. "
                          "Anupam Kher leads with 43 appearances.",
            "implication":"The sheer volume of Indian content means Bollywood talent dominates individual-level "
                          "statistics — a concrete measure of India's catalog weight.",
        },
    ]
    return insights


# =============================================================================
# SECTION 16 — TESTING
# =============================================================================

def run_tests(df: pd.DataFrame, kpis: dict) -> list:
    """Run validation tests and return list of (test_name, passed, message)."""
    results = []

    def _test(name, condition, msg=""):
        results.append((name, bool(condition), msg))

    _test("Dataset loaded",            len(df) > 0,
          f"Rows: {len(df)}")
    _test("Expected row count",        len(df) == 8807,
          f"Got {len(df)}, expected 8807")
    _test("12+ columns (incl. derived)", df.shape[1] >= 12,
          f"Got {df.shape[1]} columns")
    _test("No full duplicates",        df.duplicated().sum() == 0,
          f"Duplicates: {df.duplicated().sum()}")
    _test("show_id unique",            df["show_id"].nunique() == len(df),
          f"Unique IDs: {df['show_id'].nunique()}")
    _test("Corrupted rating fixed",    not df["rating"].isin(["66 min","74 min","84 min"]).any(),
          "Corrupted rating rows still present")
    _test("No duration nulls",         df["duration"].isnull().sum() == 0,
          f"Duration nulls: {df['duration'].isnull().sum()}")
    _test("date_added_parsed exists",  "date_added_parsed" in df.columns,
          "date_added_parsed column missing")
    _test("runtime_minutes present",   "runtime_minutes" in df.columns,
          "runtime_minutes missing")
    _test("seasons_count present",     "seasons_count" in df.columns,
          "seasons_count missing")
    _test("content_lag_years present", "content_lag_years" in df.columns,
          "content_lag_years missing")
    _test("KPI dict has 25+ keys",     len(kpis) >= 25,
          f"KPI keys: {len(kpis)}")
    _test("type only 2 values",        set(df["type"].unique()) == {"Movie","TV Show"},
          f"type values: {set(df['type'].unique())}")
    _test("runtime_minutes range",
          df["runtime_minutes"].dropna().between(1, 400).all(),
          "Runtime outside [1, 400] range")
    _test("seasons_count range",
          df["seasons_count"].dropna().between(1, 20).all(),
          "Seasons outside [1, 20] range")

    return results


# =============================================================================
# SECTION 17 — STREAMLIT DASHBOARD (main)
# =============================================================================

def _kpi_card(col, label: str, value, delta=None, suffix: str = ""):
    """Render a single KPI metric in the given column."""
    v_str = f"{value:,}{suffix}" if isinstance(value, (int, float)) else str(value)
    col.metric(label=label, value=v_str, delta=delta)


def _section_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def _safe_select_options(values, fallback: str = "No data"):
    """Return a non-empty option list for Streamlit selectors.
    Prevents crashes when filters reduce a dataset to zero rows.
    """
    items = list(values)
    return items if items else [fallback]


def main():
    # ── Page config ─────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Netflix Content Analytics",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ───────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    [data-testid="metric-container"] {
        background: #1E1E2E;
        border: 1px solid #2d2d44;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
    }
    .insight-card {
        background: #1E1E2E;
        border-left: 4px solid #E50914;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Data pipeline ────────────────────────────────────────────────────────
    with st.spinner("Loading and preparing data …"):
        csv_path = ensure_dataset()
        df_raw   = load_raw_data(csv_path)

        df, cleaning_log   = clean_data(df_raw.copy())
        df                 = handle_missing_values(df)
        dup_report         = check_duplicates(df)
        df                 = correct_dtypes(df)
        df                 = transform_and_engineer(df)
        frames             = build_exploded_frames(df)
        kpis               = calculate_kpis(df, frames)
        stat_summary       = statistical_summary(df)
        insights_list      = generate_insights(kpis, stat_summary)
        test_results       = run_tests(df, kpis)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        try:
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
                width=130,
            )
        except Exception:
            st.markdown("### 🎬 **NETFLIX**")  # offline / network-restricted fallback
        st.markdown("## 🎬 Netflix Analytics")
        st.markdown("**Ankan** | Data Analytics Project")
        st.divider()

        st.markdown("### 🔍 Filters")

        all_types = sorted(df["type"].dropna().unique().tolist())
        sel_types = st.multiselect(
            "Content Type", all_types, default=all_types, key="f_type"
        )

        min_year = int(df["added_year"].min(skipna=True))
        max_year = int(df["added_year"].max(skipna=True))
        sel_years = st.slider(
            "Year Added", min_year, max_year, (min_year, max_year), key="f_year"
        )

        top_countries_list = (
            frames["country"]["country"].value_counts().head(20).index.tolist()
        )
        sel_countries = st.multiselect(
            "Country (top 20)", ["All"] + top_countries_list,
            default=["All"], key="f_country"
        )

        top_genres_list = (
            frames["genre"]["genre"].value_counts().head(20).index.tolist()
        )
        sel_genres = st.multiselect(
            "Genre (top 20)", ["All"] + top_genres_list,
            default=["All"], key="f_genre"
        )

        all_tiers = sorted(df["audience_tier"].unique().tolist())
        sel_tiers = st.multiselect(
            "Audience Tier", all_tiers, default=all_tiers, key="f_tier"
        )

        if st.button("🔄 Reset Filters"):
            st.rerun()

        st.divider()
        st.caption(f"Dataset: netflix_titles.csv")
        st.caption(f"Source: Kaggle (shivamb/netflix-shows)")
        st.caption(f"Coverage: 2008 – Sep 2021")

    # ── Apply sidebar filters ────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)
    if sel_types:
        mask &= df["type"].isin(sel_types)
    mask &= (
        df["added_year"].isna() |
        df["added_year"].between(sel_years[0], sel_years[1])
    )
    if sel_countries and "All" not in sel_countries:
        mask &= df["primary_country"].isin(sel_countries)
    if sel_genres and "All" not in sel_genres:
        mask &= df["primary_genre"].isin(sel_genres)
    if sel_tiers:
        mask &= df["audience_tier"].isin(sel_tiers)

    df_f = df[mask].copy()

    with st.sidebar:
        st.info(f"**Showing {len(df_f):,} of {len(df):,} titles**")

    # ── Recompute frames and KPIs on filtered data ────────────────────────────
    frames_f = build_exploded_frames(df_f)
    kpis_f   = calculate_kpis(df_f, frames_f)

    # ── Tab layout ───────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏠 Overview",
        "📊 KPI Dashboard",
        "🗃 Dataset",
        "🔬 EDA",
        "📈 Trends",
        "🗂 Segments",
        "🔗 Relationships",
        "🔎 Data Explorer",
        "💡 Insights",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("# 🎬 Netflix Content Analytics (2008–2021)")
        st.markdown(
            "**A comprehensive data analytics project exploring Netflix's global "
            "content strategy through catalog composition, geographic reach, "
            "genre dynamics, audience targeting, and acquisition patterns.**"
        )
        st.divider()

        c1, c2 = st.columns([4, 6])
        with c1:
            st.markdown("### 📋 Project Brief")
            st.markdown(f"""
| Field | Detail |
|---|---|
| **Author** | Ankan |
| **Dataset** | netflix_titles.csv |
| **Source** | Kaggle — shivamb/netflix-shows |
| **Rows** | {len(df):,} titles |
| **Columns** | {df.shape[1]} |
| **Coverage** | 2008 – September 2021 |
| **Tools** | Python, Pandas, Plotly, Streamlit |
""")
            st.info(
                "**Problem Statement:** Can the composition and evolution of "
                "Netflix's own catalog reveal the content strategy decisions "
                "that drove the platform's global growth?"
            )

        with c2:
            st.markdown("### ❓ Analytical Questions Addressed")
            questions = [
                "Q1 — How has Netflix's catalog grown over time?",
                "Q2 — Is Netflix shifting from Movie-first to TV-first strategy?",
                "Q3 — Which countries produce the most Netflix content?",
                "Q4 — What does Netflix's genre landscape look like?",
                "Q5 — How have genre preferences evolved year by year?",
                "Q6 — What is the distribution of movie runtimes by genre?",
                "Q7 — What is Netflix's content maturity profile?",
                "Q8 — How long after release does content reach Netflix?",
                "Q9 — Who are the most prolific directors and actors?",
                "Q10 — Are there seasonal patterns in content additions?",
                "Q11 — How many seasons do TV shows have?",
                "Q12 — What data quality issues exist and how were they fixed?",
            ]
            for q in questions:
                st.markdown(f"- {q}")

        st.divider()
        st.markdown("### 🔄 Project Workflow")
        steps = ["Load Data","Understand","Clean","Transform","EDA",
                 "Statistics","KPIs","Visualise","Dashboard","Report"]
        cols_w = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols_w, steps)):
            col.metric(label=f"Step {i+1}", value=step)

        st.divider()
        st.markdown("### ✅ Validation Tests")
        n_pass = sum(1 for _, p, _ in test_results if p)
        n_fail = len(test_results) - n_pass
        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Tests Run",    len(test_results))
        col_t2.metric("Tests Passed", n_pass)
        col_t3.metric("Tests Failed", n_fail, delta=None if n_fail == 0 else f"⚠ {n_fail}")

        with st.expander("View test details"):
            for name, passed, msg in test_results:
                icon = "✅" if passed else "❌"
                st.markdown(f"{icon} **{name}** — {msg}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — KPI DASHBOARD
    # ════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        _section_header("📊 KPI Dashboard", "Key Performance Indicators — filtered dataset")

        r1 = st.columns(5)
        _kpi_card(r1[0], "Total Titles",     kpis_f["total_titles"])
        _kpi_card(r1[1], "Movies",            kpis_f["total_movies"],
                  delta=f"{kpis_f['movie_pct']}%")
        _kpi_card(r1[2], "TV Shows",          kpis_f["total_tv_shows"],
                  delta=f"{kpis_f['tv_pct']}%")
        _kpi_card(r1[3], "Countries",         kpis_f["total_countries"])
        _kpi_card(r1[4], "Unique Genres",     kpis_f["total_genres"])

        st.markdown("")
        r2 = st.columns(5)
        _kpi_card(r2[0], "Avg Movie Runtime",  kpis_f["avg_movie_runtime"], suffix=" min")
        _kpi_card(r2[1], "Adult Content",      kpis_f["adult_pct"], suffix="%")
        _kpi_card(r2[2], "Median Content Lag", kpis_f["median_lag_overall"], suffix=" yr")
        _kpi_card(r2[3], "Peak Year",          kpis_f["peak_year"],
                  delta=f"{kpis_f['peak_year_count']:,} titles")
        _kpi_card(r2[4], "Top Country",        kpis_f["top_country"])

        st.divider()
        kc1, kc2 = st.columns(2)
        with kc1:
            st.plotly_chart(fig_audience_tier_donut(df_f), use_container_width=True, key="kpi_donut")
        with kc2:
            tc = frames_f["country"]["country"].value_counts().head(5).reset_index()
            tc.columns = ["country", "count"]
            fig_tc = px.bar(
                tc.sort_values("count"), x="count", y="country",
                orientation="h",
                title="Top 5 Countries",
                labels={"count": "Titles", "country": ""},
                color_discrete_sequence=[CLR_RED],
                template=PLOTLY_TEMPLATE,
            )
            fig_tc.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_tc, use_container_width=True, key="kpi_top_countries")

        kc3, kc4 = st.columns(2)
        with kc3:
            gc = frames_f["genre"]["genre"].value_counts().head(5).reset_index()
            gc.columns = ["genre","count"]
            fig_gc = px.bar(
                gc.sort_values("count"), x="count", y="genre",
                orientation="h",
                title="Top 5 Genres",
                labels={"count":"Appearances","genre":""},
                color_discrete_sequence=[MOVIE_CLR],
                template=PLOTLY_TEMPLATE,
            )
            fig_gc.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_gc, use_container_width=True, key="kpi_top_genres")
        with kc4:
            type_split = df_f["type"].value_counts().reset_index()
            type_split.columns = ["type","count"]
            fig_ts = px.pie(
                type_split, names="type", values="count",
                hole=0.45,
                title="Content Type Split",
                color="type",
                color_discrete_map={"Movie": MOVIE_CLR, "TV Show": TV_CLR},
                template=PLOTLY_TEMPLATE,
            )
            fig_ts.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_ts, use_container_width=True, key="kpi_type_split")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — DATASET
    # ════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        _section_header("🗃 Dataset Overview", "Raw data, quality report, and column profiles")

        dst1, dst2, dst3, dst4 = st.tabs(
            ["Raw Data Sample", "Data Quality", "Column Profiles", "Outlier Report"]
        )

        with dst1:
            keyword = st.text_input("🔍 Search by title", "", key="kw_search")
            display_cols = st.multiselect(
                "Select columns to display",
                df_raw.columns.tolist(),
                default=["show_id","type","title","director","country",
                         "date_added","release_year","rating","duration","listed_in"],
                key="dc_select",
            )
            raw_show = df_raw.copy()
            if keyword:
                raw_show = raw_show[
                    raw_show["title"].str.contains(keyword, case=False, na=False)
                ]
            st.dataframe(raw_show[display_cols].head(200), use_container_width=True)
            st.caption(f"Showing up to 200 of {len(raw_show):,} matching rows")

        with dst2:
            _section_header("Missing Values Report")
            null_df = pd.DataFrame({
                "Column":    df_raw.columns,
                "Nulls":     df_raw.isnull().sum().values,
                "Null %":    (df_raw.isnull().mean() * 100).round(2).values,
                "Treatment": [
                    "Primary key — none needed",
                    "None — fully populated",
                    "None — fully populated",
                    "Filled with 'Unknown'",
                    "Filled with 'Unknown'",
                    "Filled with 'Unknown'",
                    "Kept as NaT — 10 rows excluded from temporal analysis",
                    "None — fully populated",
                    "Filled with 'NR' (+ 3 corruption fixes)",
                    "3 corruption fixes applied",
                    "None — fully populated",
                    "None — fully populated",
                ],
            })
            null_df_show = null_df[null_df["Nulls"] > 0].reset_index(drop=True)
            st.dataframe(null_df, use_container_width=True)

            fig_nulls = px.bar(
                null_df[null_df["Nulls"] > 0],
                x="Null %", y="Column",
                orientation="h",
                title="Null % by Column",
                labels={"Null %": "Missing Values (%)", "Column": ""},
                color="Null %", color_continuous_scale="Reds",
                template=PLOTLY_TEMPLATE,
            )
            fig_nulls.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_nulls, use_container_width=True, key="ds_nulls")

            st.markdown("#### Corrupted Rows Fixed")
            if cleaning_log:
                for entry in cleaning_log:
                    st.write(entry)

            st.markdown("#### Dataset Structure (data_understanding_report)")
            du = data_understanding_report(df_raw)
            du_rows = [
                {"Property": "Shape",          "Value": str(du["shape"])},
                {"Property": "Total Nulls",    "Value": str(sum(du["null_counts"].values()))},
                {"Property": "Types",          "Value": str(du["type_vc"])},
            ]
            st.dataframe(pd.DataFrame(du_rows), use_container_width=True)

            st.markdown("#### Duplicate Check")
            for k, v in dup_report.items():
                icon = "✅" if v == 0 else "⚠️"
                st.markdown(f"{icon} **{k.replace('_',' ').title()}**: {v}")

        with dst3:
            col_choice = st.selectbox(
                "Select column to profile",
                df.columns.tolist(),
                key="col_prof",
            )
            st.markdown(f"**dtype:** `{df[col_choice].dtype}`")
            c_null = df[col_choice].isnull().sum()
            c_uniq = df[col_choice].nunique()
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Nulls",    int(c_null))
            pc2.metric("Unique",   int(c_uniq))
            pc3.metric("Non-null", int(len(df) - c_null))

            if df[col_choice].dtype in [np.float64, np.int16, np.int32, np.int64, "Int16", "Int8"]:
                st.dataframe(df[col_choice].dropna().describe().to_frame().T, use_container_width=True)
                fig_profile = px.histogram(
                    df[col_choice].dropna(), nbins=40,
                    title=f"Distribution of {col_choice}",
                    template=PLOTLY_TEMPLATE,
                )
                fig_profile.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_profile, use_container_width=True, key="ds_profile_num")
            else:
                vc = df[col_choice].value_counts().head(20).reset_index()
                vc.columns = [col_choice, "count"]
                st.dataframe(vc, use_container_width=True)
                if c_uniq <= 40:
                    fig_profile = px.bar(
                        vc.sort_values("count"),
                        x="count", y=col_choice,
                        orientation="h",
                        title=f"Top values in {col_choice}",
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_profile.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_profile, use_container_width=True, key="ds_profile_cat")

        with dst4:
            _section_header("Outlier Analysis", "IQR-based outlier identification for movie runtimes")
            outlier_df = outlier_report(df)
            if outlier_df.empty:
                st.info("No outliers detected in the current dataset.")
            else:
                o1, o2 = st.columns(2)
                o1.metric("Total Outliers",  len(outlier_df))
                o2.metric("Short",           int((outlier_df["outlier_type"] == "Short").sum()))
                st.dataframe(outlier_df, use_container_width=True)
                st.caption(
                    f"IQR bounds: [{outlier_df['iqr_lower'].iloc[0]} min, "
                    f"{outlier_df['iqr_upper'].iloc[0]} min]. "
                    "All outliers retained — each has a legitimate contextual explanation."
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — EDA
    # ════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        _section_header("🔬 Exploratory Data Analysis",
                         "Distributions, compositions, and patterns in the filtered data")

        eda1, eda2, eda3, eda4 = st.tabs(
            ["Content Mix", "Geography", "Genres", "Duration & Ratings"]
        )

        with eda1:
            ec1, ec2 = st.columns(2)
            with ec1:
                st.plotly_chart(fig_catalog_growth(df_f), use_container_width=True, key="eda_mix_growth")
            with ec2:
                st.plotly_chart(fig_type_share(df_f), use_container_width=True, key="eda_mix_typeshare")
            st.plotly_chart(fig_monthly_heatmap(df_f), use_container_width=True, key="eda_mix_heatmap")

        with eda2:
            st.plotly_chart(fig_choropleth(frames_f), use_container_width=True, key="eda_geo_choropleth")
            gc1, gc2 = st.columns([6, 4])
            with gc1:
                st.plotly_chart(fig_top_countries(frames_f), use_container_width=True, key="eda_geo_countries")
            with gc2:
                top15_tbl = (
                    frames_f["country"]["country"]
                    .value_counts().head(15)
                    .reset_index()
                )
                top15_tbl.columns = ["Country","Titles"]
                top15_tbl.index   = top15_tbl.index + 1
                st.markdown("#### Country Stats")
                st.dataframe(top15_tbl, use_container_width=True)

        with eda3:
            gc3, gc4 = st.columns(2)
            with gc3:
                st.plotly_chart(fig_genre_frequency(frames_f), use_container_width=True, key="eda_genre_freq")
            with gc4:
                st.plotly_chart(fig_genre_cooccurrence(df_f), use_container_width=True, key="eda_genre_cooc")

            available_genres = (
                frames_f["genre"]["genre"].value_counts().head(15).index.tolist()
            )
            sel_g = st.multiselect(
                "Select genres to track over time (up to 5)",
                available_genres,
                default=available_genres[:5],
                max_selections=5,
                key="genre_trend_sel",
            )
            if sel_g:
                st.plotly_chart(fig_genre_trends(df_f, sel_g), use_container_width=True, key="eda_genre_trends")

        with eda4:
            dc1, dc2 = st.columns(2)
            with dc1:
                st.plotly_chart(fig_runtime_histogram(df_f), use_container_width=True, key="eda_dur_hist")
            with dc2:
                st.plotly_chart(fig_runtime_by_genre(df_f), use_container_width=True, key="eda_dur_box")
            dc3, dc4 = st.columns(2)
            with dc3:
                st.plotly_chart(fig_audience_tier_donut(df_f), use_container_width=True, key="eda_dur_donut")
            with dc4:
                st.plotly_chart(fig_rating_by_country(df_f, frames_f), use_container_width=True, key="eda_dur_rating_cty")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — TRENDS
    # ════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        _section_header("📈 Trend Analysis", "Time-based patterns in the catalog")

        st.plotly_chart(fig_catalog_growth(df_f), use_container_width=True, key="tr_growth")

        tr1, tr2 = st.columns(2)
        with tr1:
            st.plotly_chart(fig_type_share(df_f), use_container_width=True, key="tr_typeshare")
        with tr2:
            st.plotly_chart(fig_yoy_growth(df_f), use_container_width=True, key="tr_yoy")

        st.plotly_chart(fig_monthly_heatmap(df_f), use_container_width=True, key="tr_heatmap")

        tr3, tr4 = st.columns(2)
        with tr3:
            st.plotly_chart(fig_seasonal_bar(df_f), use_container_width=True, key="tr_seasonal")
        with tr4:
            avail_g2 = frames_f["genre"]["genre"].value_counts().head(15).index.tolist()
            sel_g2   = st.multiselect(
                "Genres for trend line",
                avail_g2,
                default=avail_g2[:5],
                max_selections=5,
                key="trend_genre_sel",
            )
            if sel_g2:
                st.plotly_chart(fig_genre_trends(df_f, sel_g2), use_container_width=True, key="tr_genre_trends")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 — SEGMENTS
    # ════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        _section_header("🗂 Category & Segment Analysis")

        seg1, seg2, seg3 = st.tabs(["By Country", "By Genre & Rating", "By Talent"])

        with seg1:
            st.plotly_chart(fig_choropleth(frames_f), use_container_width=True, key="seg_choropleth")
            s1c1, s1c2 = st.columns([6, 4])
            with s1c1:
                st.plotly_chart(fig_top_countries(frames_f), use_container_width=True, key="seg_countries")
            with s1c2:
                st.markdown("#### Country Detail")
                country_options = _safe_select_options(
                    frames_f["country"]["country"].value_counts().head(20).index.tolist()
                )
                cty_sel = st.selectbox(
                    "Select a country",
                    country_options,
                    key="cty_detail",
                )
                if cty_sel and cty_sel != "No data":
                    cty_df = frames_f["country"][frames_f["country"]["country"] == cty_sel]
                    n_titles  = len(cty_df)
                    n_movies  = (cty_df["type"] == "Movie").sum()
                    n_tv      = (cty_df["type"] == "TV Show").sum()
                    top_g_cty = (
                        cty_df.assign(genre=cty_df["listed_in"].str.split(","))
                              .explode("genre")["genre"]
                              .str.strip()
                              .value_counts().head(5)
                    )
                    st.metric("Total Titles", n_titles)
                    st.metric("Movies", int(n_movies))
                    st.metric("TV Shows", int(n_tv))
                    st.markdown("**Top Genres:**")
                    for g, c in top_g_cty.items():
                        st.markdown(f"- {g}: {c}")
                else:
                    st.info("No country data matches the current filters.")

        with seg2:
            s2c1, s2c2 = st.columns(2)
            with s2c1:
                st.plotly_chart(fig_genre_frequency(frames_f, n=42), use_container_width=True, key="seg_genre_freq")
            with s2c2:
                st.markdown("#### Genre Detail")
                genre_options = _safe_select_options(
                    frames_f["genre"]["genre"].value_counts().head(30).index.tolist()
                )
                genre_sel = st.selectbox(
                    "Select a genre",
                    genre_options,
                    key="genre_detail",
                )
                if genre_sel and genre_sel != "No data":
                    g_df = frames_f["genre"][frames_f["genre"]["genre"] == genre_sel]
                    st.metric("Titles with this genre", len(g_df))
                    st.metric("Movies", int((g_df["type"]=="Movie").sum()))
                    st.metric("TV Shows", int((g_df["type"]=="TV Show").sum()))
                    if "runtime_minutes" in g_df.columns:
                        avg_rt = g_df["runtime_minutes"].dropna()
                        if len(avg_rt) > 0:
                            st.metric("Avg Runtime (Movies)", f"{avg_rt.mean():.1f} min")
                else:
                    st.info("No genre data matches the current filters.")

            s2c3, s2c4 = st.columns(2)
            with s2c3:
                rating_vc = df_f["rating"].value_counts().reset_index()
                rating_vc.columns = ["rating","count"]
                fig_rvc = px.bar(
                    rating_vc.sort_values("count"),
                    x="count", y="rating", orientation="h",
                    title="All Rating Codes Distribution",
                    labels={"count":"Titles","rating":""},
                    color_discrete_sequence=[CLR_RED],
                    template=PLOTLY_TEMPLATE,
                )
                fig_rvc.update_layout(height=460, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_rvc, use_container_width=True, key="seg_rating_vc")
            with s2c4:
                tier_type = (
                    df_f.groupby(["type","audience_tier"], observed=True)
                         .size().reset_index(name="count")
                )
                fig_tt = px.bar(
                    tier_type, x="count", y="type",
                    color="audience_tier",
                    orientation="h", barmode="relative",
                    title="Audience Tier by Content Type",
                    labels={"count":"Titles","type":""},
                    color_discrete_map=TIER_COLOURS,
                    template=PLOTLY_TEMPLATE,
                )
                fig_tt.update_layout(height=340, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig_tt, use_container_width=True, key="seg_tier_type")

            st.plotly_chart(fig_rating_by_country(df_f, frames_f), use_container_width=True, key="seg_rating_cty")

        with seg3:
            s3c1, s3c2 = st.columns(2)
            with s3c1:
                st.plotly_chart(fig_top_directors(frames_f), use_container_width=True, key="seg_directors")
            with s3c2:
                st.plotly_chart(fig_top_actors(frames_f), use_container_width=True, key="seg_actors")

            s3c3, s3c4 = st.columns(2)
            with s3c3:
                st.markdown("#### Top Actors by Country")
                actor_country_options = _safe_select_options(
                    frames_f["country"]["country"].value_counts().head(15).index.tolist()
                )
                cty_actor = st.selectbox(
                    "Country",
                    actor_country_options,
                    key="actor_cty",
                )
                if cty_actor and cty_actor != "No data":
                    cty_shows = frames_f["country"][
                        frames_f["country"]["country"] == cty_actor
                    ]["show_id"].unique()
                    actor_sub = frames_f["cast"][
                        frames_f["cast"]["show_id"].isin(cty_shows)
                    ]
                    top_a = actor_sub["actor"].value_counts().head(10).reset_index()
                    top_a.columns = ["actor","count"]
                    fig_ta = px.bar(
                        top_a.sort_values("count"),
                        x="count", y="actor", orientation="h",
                        title=f"Top Actors in {cty_actor} content",
                        color_discrete_sequence=[CLR_BLUE],
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_ta.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_ta, use_container_width=True, key="seg_actors_cty")
                else:
                    st.info("No actor-country data is available for the active filters.")
            with s3c4:
                st.markdown("#### Top Directors by Country")
                director_country_options = _safe_select_options(
                    frames_f["country"]["country"].value_counts().head(15).index.tolist()
                )
                cty_dir = st.selectbox(
                    "Country",
                    director_country_options,
                    key="dir_cty",
                )
                if cty_dir and cty_dir != "No data":
                    cty_shows2 = frames_f["country"][
                        frames_f["country"]["country"] == cty_dir
                    ]["show_id"].unique()
                    dir_sub = frames_f["director"][
                        frames_f["director"]["show_id"].isin(cty_shows2)
                    ]
                    top_d = dir_sub["director_name"].value_counts().head(10).reset_index()
                    top_d.columns = ["director","count"]
                    fig_td = px.bar(
                        top_d.sort_values("count"),
                        x="count", y="director", orientation="h",
                        title=f"Top Directors in {cty_dir} content",
                        color_discrete_sequence=[CLR_RED],
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_td.update_layout(height=380, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_td, use_container_width=True, key="seg_directors_cty")
                else:
                    st.info("No director-country data is available for the active filters.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 7 — RELATIONSHIPS
    # ════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        _section_header("🔗 Relationship & Correlation Analysis")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.plotly_chart(fig_lag_boxplots(df_f), use_container_width=True, key="rel_lag_box")
        with rc2:
            st.plotly_chart(fig_lag_trend(df_f), use_container_width=True, key="rel_lag_trend")

        rc3, rc4 = st.columns(2)
        with rc3:
            st.plotly_chart(fig_lag_scatter(df_f), use_container_width=True, key="rel_lag_scatter")
        with rc4:
            st.markdown("#### Correlation Statistics")
            valid_c = df_f[["release_year","content_lag_years"]].dropna()
            if len(valid_c) > 2:
                r_val, p_val = stats.pearsonr(
                    valid_c["release_year"], valid_c["content_lag_years"]
                )
                st.metric("Pearson r (release year vs lag)", f"{r_val:.4f}")
                st.metric("p-value", f"{p_val:.6f}")
                if abs(r_val) > 0.3:
                    st.info(
                        f"r = {r_val:.4f} indicates a **{'negative' if r_val < 0 else 'positive'} "
                        f"correlation** between release year and content lag. "
                        "Older films take significantly longer to appear on Netflix."
                    )
            st.markdown("#### Lag Statistics by Type")
            for t in ["Movie","TV Show"]:
                sub_t = df_f[df_f["type"] == t]["content_lag_years"].dropna()
                if len(sub_t) > 0:
                    st.markdown(
                        f"**{t}** — Mean: {sub_t.mean():.2f} yr | "
                        f"Median: {sub_t.median():.0f} yr | "
                        f"Within 1 yr: {(sub_t<=1).mean()*100:.1f}%"
                    )

        st.plotly_chart(fig_genre_cooccurrence(df_f), use_container_width=True, key="rel_cooc")

        rc5, rc6 = st.columns(2)
        with rc5:
            st.plotly_chart(fig_seasons_dist(df_f), use_container_width=True, key="rel_seasons_dist")
        with rc6:
            st.plotly_chart(fig_seasons_by_country(df_f, frames_f), use_container_width=True, key="rel_seasons_cty")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 8 — DATA EXPLORER
    # ════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        _section_header("🔎 Data Explorer", "Free-form search and filter of the full dataset")

        ex1, ex2 = st.columns([3, 7])

        with ex1:
            st.markdown("#### Search & Filter")
            kw_ex   = st.text_input("Title contains", "", key="ex_kw")
            type_ex = st.selectbox("Type", ["All","Movie","TV Show"], key="ex_type")
            y1_ex, y2_ex = st.slider(
                "Year added", min_year, max_year, (min_year, max_year), key="ex_yr"
            )
            rating_ex = st.multiselect(
                "Rating",
                sorted(df["rating"].unique().tolist()),
                default=[],
                key="ex_rating",
            )
            cty_ex   = st.text_input("Country contains", "", key="ex_cty")
            genre_ex = st.multiselect(
                "Genre", top_genres_list, default=[], key="ex_genre"
            )
            if st.checkbox("Movie runtime filter", key="ex_rt_chk"):
                rt_range = st.slider("Runtime (min)", 1, 320, (60,150), key="ex_rt")
            else:
                rt_range = None
            if st.checkbox("TV seasons filter", key="ex_s_chk"):
                s_range = st.slider("Seasons", 1, 17, (1,5), key="ex_s")
            else:
                s_range = None

        with ex2:
            exp_df = df.copy()
            if kw_ex:
                exp_df = exp_df[exp_df["title"].str.contains(kw_ex, case=False, na=False)]
            if type_ex != "All":
                exp_df = exp_df[exp_df["type"] == type_ex]
            exp_df = exp_df[
                exp_df["added_year"].isna() |
                exp_df["added_year"].between(y1_ex, y2_ex)
            ]
            if rating_ex:
                exp_df = exp_df[exp_df["rating"].isin(rating_ex)]
            if cty_ex:
                exp_df = exp_df[
                    exp_df["country"].str.contains(cty_ex, case=False, na=False)
                ]
            if genre_ex:
                exp_df = exp_df[exp_df["primary_genre"].isin(genre_ex)]
            if rt_range:
                exp_df = exp_df[
                    exp_df["runtime_minutes"].isna() |
                    exp_df["runtime_minutes"].between(*rt_range)
                ]
            if s_range:
                exp_df = exp_df[
                    exp_df["seasons_count"].isna() |
                    exp_df["seasons_count"].between(*s_range)
                ]

            show_cols = ["title","type","director","country","date_added",
                         "release_year","rating","duration","listed_in"]
            st.dataframe(
                exp_df[show_cols].reset_index(drop=True),
                use_container_width=True,
                height=480,
            )
            st.caption(f"**{len(exp_df):,}** titles match your filters")

            ex_r1, ex_r2, ex_r3 = st.columns(3)
            ex_r1.metric("Filtered Titles", len(exp_df))
            if len(exp_df) > 0:
                m_pct = round((exp_df["type"]=="Movie").mean()*100, 1)
                t_pct = round((exp_df["type"]=="TV Show").mean()*100, 1)
                ex_r2.metric("Movies", f"{m_pct}%")
                ex_r3.metric("TV Shows", f"{t_pct}%")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 9 — INSIGHTS
    # ════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        _section_header("💡 Key Insights & Findings")

        ins1, ins2 = st.columns([5, 5])

        with ins1:
            st.markdown("### 🔍 Analytical Findings")
            for item in insights_list:
                with st.expander(f"{item['emoji']} {item['finding']}"):
                    st.markdown(f"**Evidence:** {item['evidence']}")
                    st.markdown(f"**Implication:** {item['implication']}")

        with ins2:
            st.markdown("### 📋 Data Quality Summary")
            dq_data = {
                "Issue":  [
                    "Corrupted rating rows",
                    "Null director values",
                    "Null cast values",
                    "Null country values",
                    "Null date_added",
                    "Null rating values",
                    "Mixed-type duration",
                    "String-format dates",
                ],
                "Rows Affected": [3, 2634, 825, 831, 10, 4, "All", "All"],
                "Action Taken": [
                    "Moved duration to correct column",
                    "Filled with 'Unknown'",
                    "Filled with 'Unknown'",
                    "Filled with 'Unknown'",
                    "Kept as NaT; excluded from temporal analysis",
                    "Filled with 'NR'",
                    "Split into runtime_minutes / seasons_count",
                    "Parsed with pd.to_datetime()",
                ],
            }
            st.dataframe(pd.DataFrame(dq_data), use_container_width=True)

            st.markdown("### 📐 Statistical Summary")
            stat_rows = {
                "Metric": [
                    "Movie runtime — Mean",
                    "Movie runtime — Median",
                    "Movie runtime — Std Dev",
                    "Movie runtime — Skewness",
                    "Content lag — Mean (all)",
                    "Content lag — Median (all)",
                    "Content lag — Mean (Movies)",
                    "Content lag — Mean (TV Shows)",
                    "Pearson r (year vs lag)",
                    "Genre HHI concentration",
                    "Negative lag titles",
                ],
                "Value": [
                    f"{stat_summary['runtime_mean']} min",
                    f"{stat_summary['runtime_median']} min",
                    f"{stat_summary['runtime_std']} min",
                    str(stat_summary["runtime_skew"]),
                    f"{stat_summary['lag_mean_overall']} yr",
                    f"{stat_summary['lag_median_overall']} yr",
                    f"{stat_summary['lag_mean_movie']} yr",
                    f"{stat_summary['lag_mean_tv']} yr",
                    str(stat_summary["pearson_r"]),
                    str(stat_summary["genre_hhi"]),
                    str(stat_summary["n_negative_lags"]),
                ],
            }
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

            st.markdown("### ⚠️ Project Limitations")
            limitations = [
                "Dataset ends September 2021 — post-2021 trends not captured.",
                "No viewership, subscriber, or revenue data available.",
                "`director` is missing for 29.9% of titles (mostly TV Shows).",
                "`country` reflects production origin, not regional availability.",
                "Netflix's genre taxonomy mixes origin tags with content genres.",
                "Removed titles are not tracked — catalog is a snapshot only.",
                "Multi-country titles are counted once per country in geographic analysis.",
            ]
            for lim in limitations:
                st.markdown(f"- {lim}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
