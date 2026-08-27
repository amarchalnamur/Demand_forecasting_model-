
monthly.py — Pipeline mensuel de forecast de la demande

 
Ce script est destiné à être relancé CHAQUE MOIS, dès qu'un nouveau mois
d'actuals est disponible dans ml_demand_forecast_base.csv.
 
A chaque exécution, il :
 
  1. Charge ml_demand_forecast_base.csv ;
  2. Détecte automatiquement le dernier mois disponible = forecast_origin ;
  3. Regarde s'il existe des forecasts historiques (dans forecast_output)
     dont le target_month est maintenant "réalisé" (actual disponible) ;
  4. Met à jour forecast_actual_comparison (table d'erreurs au grain le plus
     fin) avec les nouvelles comparaisons forecast vs actual ;
  5. Recalcule entièrement forecast_monitoring (KPI agrégés : WAPE, Bias,
     gain vs baseline, métriques rolling, etc.) à partir de
     forecast_actual_comparison ;
  6. Reconstruit le panel mensuel complet et exactement la même famille de
     features que celle utilisée en développement ;
  7. Réentraîne 6 modèles LightGBM Residual (H1 à H6) sur tout l'historique
     disponible ;
  8. Génère les forecasts au grain site x supplier x category (niveau SITE) ;
  9. Génère en parallèle les forecasts au grain supplier x category avec le
     modèle baseline "Last Quantity" (niveau SUPPLIER_CATEGORY) ;
 10. Historise les nouveaux forecasts dans forecast_output (sans écraser les
     versions précédentes) et ajoute une ligne à forecast_run.
 
             
 
 

# IMPORTS

 
import warnings
warnings.filterwarnings("ignore")
 
from pathlib import Path
from datetime import datetime
 
import numpy as np
import pandas as pd
 
from lightgbm import LGBMRegressor
 

# CONFIGURATION

 

BASE_DIR = Path(r"C:\Users\arnau\UNAMUR\projet perso")
 
INPUT_FILE = BASE_DIR / "ml_demand_forecast_base.csv"
 

OUTPUT_DIR = BASE_DIR / "forecast_tables"
 
FORECAST_OUTPUT_FILE = OUTPUT_DIR / "forecast_output.csv"
FORECAST_MONITORING_FILE = OUTPUT_DIR / "forecast_monitoring.csv"
FORECAST_COMPARISON_FILE = OUTPUT_DIR / "forecast_actual_comparison.csv"
FORECAST_RUN_FILE = OUTPUT_DIR / "forecast_run.csv"
 
RANDOM_STATE = 42
MODEL_VERSION = "v1.0"
FEATURE_VERSION = "v1.0"
 
SERIES_COLS = ["site_id", "supplier_id", "category_id"]
FORECAST_HORIZONS = [1, 2, 3, 4, 5, 6]
 
DATE_COL = "month_date"
TARGET_COL = "quantity"
 
SITE_MODEL_NAME = "LightGBM Residual"
BASELINE_MODEL_NAME = "Last Quantity"
 
FORECAST_LEVEL_SITE = "SITE"
FORECAST_LEVEL_SUPPLIER_CATEGORY = "SUPPLIER_CATEGORY"
 
ROLLING_WINDOW_MONTHS = 3
 
VERBOSE = True
 
 
def log(*args):
    if VERBOSE:
        print(*args)
 
 

# VARIABLES CALANDAIRES

 
def easter_sunday(year: int) -> pd.Timestamp:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=int(month), day=int(day))
 
 
def belgian_public_holidays(year: int) -> pd.DatetimeIndex:
 
    easter = easter_sunday(year)
 
    holidays = [
        pd.Timestamp(year, 1, 1),                      # Nouvel An
        easter + pd.Timedelta(days=1),                 # Lundi de Pâques
        pd.Timestamp(year, 5, 1),                       # Fête du travail
        easter + pd.Timedelta(days=39),                 # Ascension
        easter + pd.Timedelta(days=50),                 # Lundi de Pentecôte
        pd.Timestamp(year, 7, 21),                      # Fête nationale
        pd.Timestamp(year, 8, 15),                      # Assomption
        pd.Timestamp(year, 11, 1),                      # Toussaint
        pd.Timestamp(year, 11, 11),                     # Armistice
        pd.Timestamp(year, 12, 25),                     # Noël
    ]
 
    return pd.DatetimeIndex(sorted(holidays))
 
 
def public_holidays_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    years = range(start.year, end.year + 1)
    all_holidays = []
    for y in years:
        all_holidays.extend(list(belgian_public_holidays(y)))
    idx = pd.DatetimeIndex(sorted(all_holidays))
    return idx[(idx >= start) & (idx <= end)]
 
 
def weekdays_between(start, end) -> int:
  
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end < start:
        return 0
    dates = pd.date_range(start=start, end=end, freq="D")
    return int((dates.weekday < 5).sum())
 
 

# Vacances scolaires belges.


 
SCHOOL_HOLIDAYS_BY_YEAR = {
    2026: {
        "FLANDERS": [
            (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-31")),  # Grandes vacances
            (pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-08")),  # Toussaint
            (pd.Timestamp("2026-12-21"), pd.Timestamp("2026-12-31")),  # Noël
        ],
        "FWB": [
            (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-23")),  # Grandes vacances
            (pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-30")),  # Toussaint
            (pd.Timestamp("2026-12-21"), pd.Timestamp("2026-12-31")),  # Noël
        ],
    },
  

}
 
 
def school_holiday_weekdays_exact(year: int, region_type: str, month_start, month_end):

    periods = SCHOOL_HOLIDAYS_BY_YEAR.get(year, {}).get(region_type)
    if periods is None:
        return None
 
    total = 0
    for start, end in periods:
        overlap_start = max(month_start, start)
        overlap_end = min(month_end, end)
        if overlap_start <= overlap_end:
            total += weekdays_between(overlap_start, overlap_end)
    return total
 
 
def build_historical_month_ratio(calendar_region: pd.DataFrame) -> pd.DataFrame:

    tmp = calendar_region.copy()
    tmp["month_number"] = tmp["month_date"].dt.month
    avg = (
        tmp.groupby(["month_number", "school_holiday_region"], as_index=False)
        .agg(
            school_holiday_ratio_selected_hist=("school_holiday_ratio_selected", "mean"),
        )
    )
    return avg
 
 
# ============================================================
# 3. STANDARDIZE DATA TYPES
# ============================================================
 
CATEGORICAL_RAW_COLS = [
    "site_id",
    "supplier_id",
    "category_id",
    "category_level_1_id",
    "category_level_2_id",
    "category_level_3_id",
    "cpi_id",
    "region",
    "school_holiday_region",
    "municipality_confidence",
]
 
# Mêmes colonnes catégorielles que celles utilisées en développement.
CATEGORICAL_FEATURES = list(CATEGORICAL_RAW_COLS)
 
NUMERIC_RAW_COLS = [
    "quantity",
    "gross_spend",
    "calendar_year",
    "month_number",
    "quarter",
    "days_in_month",
    "working_days",
    "weekdays_mon_fri",
    "public_holidays",
    "public_holidays_weekdays",
    "school_holiday_weekdays_flanders",
    "school_holiday_ratio_flanders",
    "school_holiday_weekdays_fwb",
    "school_holiday_ratio_fwb",
    "school_holiday_ratio_selected",
    "school_holiday_weekdays_selected",
    "cpi_index",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "cpi_coefficient",
]
 
METADATA_COLS = [
    "region",
    "school_holiday_region",
    "municipality_confidence",
    "category_level_1_id",
    "category_level_2_id",
    "category_level_3_id",
    "cpi_id",
]
 
CALENDAR_COLS = [
    "month_date",
    "school_holiday_region",
    "days_in_month",
    "working_days",
    "weekdays_mon_fri",
    "public_holidays",
    "public_holidays_weekdays",
    "school_holiday_weekdays_flanders",
    "school_holiday_ratio_flanders",
    "school_holiday_weekdays_fwb",
    "school_holiday_ratio_fwb",
    "school_holiday_ratio_selected",
    "school_holiday_weekdays_selected",
]
 
# Mêmes features numériques "communes" que celles utilisées en développement.
NUMERIC_COMMON = [
    # Demand
    "quantity_current",
    "quantity_lag_1", "quantity_lag_2", "quantity_lag_3", "quantity_lag_4",
    "quantity_lag_5", "quantity_lag_6", "quantity_lag_9", "quantity_lag_12",
    "quantity_mean_3", "quantity_mean_6", "quantity_mean_12",
    "quantity_std_3", "quantity_std_6", "quantity_std_12",
    # Intermittence
    "demand_positive_current",
    "positive_months_3", "positive_months_6", "positive_months_12",
    "zero_ratio_3", "zero_ratio_6", "zero_ratio_12",
    "positive_quantity_mean_6", "positive_quantity_mean_12",
    "last_positive_quantity",
    # Price
    "unit_price_current",
    "unit_price_lag_1", "unit_price_lag_2", "unit_price_lag_3",
    "unit_price_lag_6", "unit_price_lag_12",
    "unit_price_mean_3",
    "unit_price_change_1m", "unit_price_change_3m",
    # CPI
    "cpi_index", "cpi_mom_pct", "cpi_yoy_pct", "cpi_coefficient",
    "cpi_index_lag_1", "cpi_index_lag_3", "cpi_index_lag_6",
    "cpi_mom_pct_lag_1", "cpi_yoy_pct_lag_1",
    "cpi_index_change_3m", "cpi_index_change_6m",
    # Current calendar
    "current_working_days",
    "current_public_holidays_weekdays",
    "current_school_holiday_ratio_selected",
    # Trend
    "time_index",
]
 
 
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
 
    log(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    log("Date range:", df[DATE_COL].min(), "->", df[DATE_COL].max())
 
    for col in CATEGORICAL_RAW_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
 
    for col in NUMERIC_RAW_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    grain_cols = ["month_date"] + SERIES_COLS
    dup = df.duplicated(grain_cols, keep=False).sum()
    log("Duplicate rows at forecast grain:", dup)
    log("Unique series:", df[SERIES_COLS].drop_duplicates().shape[0])
 
    return df
 
 
# ============================================================
# 4. PANEL + FEATURE ENGINEERING
#    (même logique / mêmes features que le pipeline de dev)
# ============================================================
 
def build_series_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = SERIES_COLS + [c for c in METADATA_COLS if c in df.columns]
    return (
        df[cols]
        .groupby(SERIES_COLS, as_index=False, observed=True)
        .first()
    )
 
 
def build_calendar_region(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CALENDAR_COLS if c in df.columns]
    return (
        df[cols]
        .groupby(["month_date", "school_holiday_region"], as_index=False, observed=True)
        .median(numeric_only=True)
    )
 
 
def build_cpi_lookup(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["month_date", "cpi_id", "cpi_index", "cpi_mom_pct", "cpi_yoy_pct", "cpi_coefficient"]
    cols = [c for c in cols if c in df.columns]
    return (
        df[cols]
        .groupby(["month_date", "cpi_id"], as_index=False, observed=True)
        .median(numeric_only=True)
    )
 
 
def build_panel(df: pd.DataFrame, series_metadata, calendar_region, cpi_lookup):
    all_months = pd.DataFrame({
        "month_date": pd.date_range(start=df["month_date"].min(), end=df["month_date"].max(), freq="MS")
    })
 
    series_keys = df[SERIES_COLS].drop_duplicates()
    panel = series_keys.merge(all_months, how="cross")
 
    actuals = df[["month_date"] + SERIES_COLS + ["quantity", "gross_spend"]].copy()
 
    panel = panel.merge(actuals, on=["month_date"] + SERIES_COLS, how="left")
    panel = panel.merge(series_metadata, on=SERIES_COLS, how="left")
    panel = panel.merge(cpi_lookup, on=["month_date", "cpi_id"], how="left")
 
    panel = panel.sort_values(SERIES_COLS + ["month_date"]).reset_index(drop=True)
 
    # -- current calendar features --
    current_calendar = calendar_region.copy()
    rename_current = {
        c: f"current_{c}" for c in calendar_region.columns
        if c not in ["month_date", "school_holiday_region"]
    }
    current_calendar = current_calendar.rename(columns=rename_current)
    panel = panel.merge(current_calendar, on=["month_date", "school_holiday_region"], how="left")
 
    # -- demand history (lags 1-12) --
    panel["quantity_current"] = panel["quantity"]
    grouped = panel.groupby(SERIES_COLS, observed=True)
    for lag in range(1, 13):
        panel[f"quantity_lag_{lag}"] = grouped["quantity"].shift(lag)
 
    # -- rolling demand features --
    recent_3 = ["quantity_current", "quantity_lag_1", "quantity_lag_2"]
    recent_6 = ["quantity_current"] + [f"quantity_lag_{i}" for i in range(1, 6)]
    recent_12 = ["quantity_current"] + [f"quantity_lag_{i}" for i in range(1, 12)]
 
    panel["quantity_mean_3"] = panel[recent_3].mean(axis=1)
    panel["quantity_mean_6"] = panel[recent_6].mean(axis=1)
    panel["quantity_mean_12"] = panel[recent_12].mean(axis=1)
    panel["quantity_std_3"] = panel[recent_3].std(axis=1)
    panel["quantity_std_6"] = panel[recent_6].std(axis=1)
    panel["quantity_std_12"] = panel[recent_12].std(axis=1)
 
    # -- intermittence features --
    panel["demand_positive_current"] = (panel["quantity_current"] > 0).astype(float)
 
    for name, cols in [("3", recent_3), ("6", recent_6), ("12", recent_12)]:
        panel[f"positive_months_{name}"] = (panel[cols] > 0).sum(axis=1)
        panel[f"zero_ratio_{name}"] = (panel[cols] == 0).mean(axis=1)
 
    panel["positive_quantity_mean_6"] = panel[recent_6].where(panel[recent_6] > 0).mean(axis=1)
    panel["positive_quantity_mean_12"] = panel[recent_12].where(panel[recent_12] > 0).mean(axis=1)
 
    # -- last positive quantity --
    panel["_positive_quantity"] = panel["quantity"].where(panel["quantity"] > 0)
    panel["last_positive_quantity"] = (
        panel.groupby(SERIES_COLS, observed=True)["_positive_quantity"].ffill()
    )
    panel.drop(columns=["_positive_quantity"], inplace=True)
 
    # -- unit price features --
    panel["unit_price_current"] = np.where(
        panel["quantity"] > 0, panel["gross_spend"] / panel["quantity"], np.nan
    )
    grouped = panel.groupby(SERIES_COLS, observed=True)
    for lag in [1, 2, 3, 6, 12]:
        panel[f"unit_price_lag_{lag}"] = grouped["unit_price_current"].shift(lag)
 
    panel["unit_price_mean_3"] = panel[
        ["unit_price_current", "unit_price_lag_1", "unit_price_lag_2"]
    ].mean(axis=1)
 
    panel["unit_price_change_1m"] = panel["unit_price_current"] / panel["unit_price_lag_1"] - 1
    panel["unit_price_change_3m"] = panel["unit_price_current"] / panel["unit_price_lag_3"] - 1
    panel["unit_price_change_1m"] = panel["unit_price_change_1m"].clip(-0.8, 3)
    panel["unit_price_change_3m"] = panel["unit_price_change_3m"].clip(-0.8, 3)
 
    # -- CPI history features --
    grouped = panel.groupby(SERIES_COLS, observed=True)
    for col in ["cpi_index", "cpi_mom_pct", "cpi_yoy_pct", "cpi_coefficient"]:
        for lag in [1, 3, 6]:
            panel[f"{col}_lag_{lag}"] = grouped[col].shift(lag)
 
    panel["cpi_index_change_3m"] = panel["cpi_index"] / panel["cpi_index_lag_3"] - 1
    panel["cpi_index_change_6m"] = panel["cpi_index"] / panel["cpi_index_lag_6"] - 1
 
    # -- targets H1-H6 --
    grouped = panel.groupby(SERIES_COLS, observed=True)
    for h in FORECAST_HORIZONS:
        panel[f"target_h{h}"] = grouped["quantity"].shift(-h)
 
    log("Target availability:")
    for h in FORECAST_HORIZONS:
        log(f"H{h}: {panel[f'target_h{h}'].notna().sum():,}")
 
    # -- future calendar features H1-H6 --
    for h in FORECAST_HORIZONS:
        future_calendar = calendar_region.copy()
        future_calendar["month_date"] = future_calendar["month_date"] - pd.DateOffset(months=h)
 
        rename_future = {
            c: f"target_{c}_h{h}" for c in calendar_region.columns
            if c not in ["month_date", "school_holiday_region"]
        }
        future_calendar = future_calendar.rename(columns=rename_future)
 
        panel = panel.merge(future_calendar, on=["month_date", "school_holiday_region"], how="left")
 
    # -- target-month seasonality --
    for h in FORECAST_HORIZONS:
        target_date = panel["month_date"] + pd.DateOffset(months=h)
        panel[f"target_month_h{h}"] = target_date.dt.month
        panel[f"target_quarter_h{h}"] = target_date.dt.quarter
        panel[f"target_month_sin_h{h}"] = np.sin(2 * np.pi * target_date.dt.month / 12)
        panel[f"target_month_cos_h{h}"] = np.cos(2 * np.pi * target_date.dt.month / 12)
 
    # -- calendar change features --
    for h in FORECAST_HORIZONS:
        panel[f"working_days_change_h{h}"] = (
            panel[f"target_working_days_h{h}"] - panel["current_working_days"]
        )
        panel[f"school_holiday_change_h{h}"] = (
            panel[f"target_school_holiday_ratio_selected_h{h}"]
            - panel["current_school_holiday_ratio_selected"]
        )
        panel[f"public_holiday_change_h{h}"] = (
            panel[f"target_public_holidays_weekdays_h{h}"]
            - panel["current_public_holidays_weekdays"]
        )
 
    # -- time trend --
    first_date = panel["month_date"].min()
    panel["time_index"] = (
        (panel["month_date"].dt.year - first_date.year) * 12
        + (panel["month_date"].dt.month - first_date.month)
    )
 
    log("Panel shape:", panel.shape)
    return panel
 
 
def get_features(panel: pd.DataFrame, horizon: int):
    future_features = [
        f"target_days_in_month_h{horizon}",
        f"target_working_days_h{horizon}",
        f"target_weekdays_mon_fri_h{horizon}",
        f"target_public_holidays_h{horizon}",
        f"target_public_holidays_weekdays_h{horizon}",
        f"target_school_holiday_weekdays_selected_h{horizon}",
        f"target_school_holiday_ratio_selected_h{horizon}",
        f"target_school_holiday_ratio_flanders_h{horizon}",
        f"target_school_holiday_ratio_fwb_h{horizon}",
        f"working_days_change_h{horizon}",
        f"school_holiday_change_h{horizon}",
        f"public_holiday_change_h{horizon}",
        f"target_month_h{horizon}",
        f"target_quarter_h{horizon}",
        f"target_month_sin_h{horizon}",
        f"target_month_cos_h{horizon}",
    ]
 
    features = NUMERIC_COMMON + future_features + CATEGORICAL_FEATURES
    features = [c for c in dict.fromkeys(features) if c in panel.columns]
    return features
 
 
def prepare_lgb_frame(panel: pd.DataFrame) -> pd.DataFrame:
    panel_lgb = panel.copy()
    for col in CATEGORICAL_FEATURES:
        if col in panel_lgb.columns:
            panel_lgb[col] = panel_lgb[col].astype("category")
    log("panel_lgb ready:", panel_lgb.shape)
    return panel_lgb
 
 
# ============================================================
# 5. FUTURE CALENDAR FOR THE FORECAST HORIZON
#    (généralisé à n'importe quelle année, avec fallback
#    historique pour les vacances scolaires si l'année n'est
#    pas dans le registre)
# ============================================================
 
def build_future_calendar_region(forecast_origin: pd.Timestamp, horizons, historical_month_ratio: pd.DataFrame) -> pd.DataFrame:
    forecast_months = [forecast_origin + pd.DateOffset(months=h) for h in horizons]
 
    future_calendar_month = []
 
    for month in forecast_months:
        month = pd.Timestamp(month)
        month_end = month + pd.offsets.MonthEnd(0)
        all_days = pd.date_range(month, month_end, freq="D")
        weekdays = all_days.weekday < 5
        weekdays_mon_fri = int(weekdays.sum())
 
        holidays_month = public_holidays_range(month, month_end)
        public_holidays = len(holidays_month)
        public_holidays_weekdays = sum(h.weekday() < 5 for h in holidays_month)
 
        working_days = weekdays_mon_fri - public_holidays_weekdays
 
        year = month.year
 
        flanders_days = school_holiday_weekdays_exact(year, "FLANDERS", month, month_end)
        fwb_days = school_holiday_weekdays_exact(year, "FWB", month, month_end)
 
        flanders_ratio = None
        fwb_ratio = None
 
        if flanders_days is not None and weekdays_mon_fri > 0:
            flanders_ratio = flanders_days / weekdays_mon_fri
 
        if fwb_days is not None and weekdays_mon_fri > 0:
            fwb_ratio = fwb_days / weekdays_mon_fri
 
        # -- fallback: registre non renseigné pour cette année --
        if flanders_ratio is None or fwb_ratio is None:
            log(
                f"[WARNING] Vacances scolaires {year} non renseignées dans "
                f"SCHOOL_HOLIDAYS_BY_YEAR pour le mois {month.date()} -> "
                f"utilisation de la moyenne historique du même mois calendaire."
            )
            hist = historical_month_ratio[historical_month_ratio["month_number"] == month.month]
 
            if flanders_ratio is None:
                row = hist[hist["school_holiday_region"] == "FLANDERS"]
                flanders_ratio = float(row["school_holiday_ratio_selected_hist"].iloc[0]) if len(row) else 0.0
                flanders_days = flanders_ratio * weekdays_mon_fri
 
            if fwb_ratio is None:
                row = hist[hist["school_holiday_region"] == "FWB"]
                fwb_ratio = float(row["school_holiday_ratio_selected_hist"].iloc[0]) if len(row) else 0.0
                fwb_days = fwb_ratio * weekdays_mon_fri
 
        future_calendar_month.append({
            "month_date": month,
            "days_in_month": len(all_days),
            "weekdays_mon_fri": weekdays_mon_fri,
            "working_days": working_days,
            "public_holidays": public_holidays,
            "public_holidays_weekdays": public_holidays_weekdays,
            "school_holiday_weekdays_flanders": flanders_days,
            "school_holiday_ratio_flanders": flanders_ratio,
            "school_holiday_weekdays_fwb": fwb_days,
            "school_holiday_ratio_fwb": fwb_ratio,
        })
 
    future_calendar_month = pd.DataFrame(future_calendar_month)
 
    # -- explode by school_holiday_region (FLANDERS / FWB / BRUSSELS_BOTH) --
    future_calendar_region = []
 
    for _, row in future_calendar_month.iterrows():
        base = row.to_dict()
 
        flanders = base.copy()
        flanders["school_holiday_region"] = "FLANDERS"
        flanders["school_holiday_weekdays_selected"] = flanders["school_holiday_weekdays_flanders"]
        flanders["school_holiday_ratio_selected"] = flanders["school_holiday_ratio_flanders"]
        future_calendar_region.append(flanders)
 
        fwb = base.copy()
        fwb["school_holiday_region"] = "FWB"
        fwb["school_holiday_weekdays_selected"] = fwb["school_holiday_weekdays_fwb"]
        fwb["school_holiday_ratio_selected"] = fwb["school_holiday_ratio_fwb"]
        future_calendar_region.append(fwb)
 
        brussels = base.copy()
        brussels["school_holiday_region"] = "BRUSSELS_BOTH"
        brussels["school_holiday_weekdays_selected"] = (
            (brussels["school_holiday_weekdays_flanders"] + brussels["school_holiday_weekdays_fwb"]) / 2
        )
        brussels["school_holiday_ratio_selected"] = (
            (brussels["school_holiday_ratio_flanders"] + brussels["school_holiday_ratio_fwb"]) / 2
        )
        future_calendar_region.append(brussels)
 
    future_calendar_region = pd.DataFrame(future_calendar_region)
    future_calendar_region["school_holiday_region"] = future_calendar_region["school_holiday_region"].astype("string")
 
    return future_calendar_region
 
 
# ============================================================
# 6. TRAIN + FORECAST — SITE x SUPPLIER x CATEGORY (LightGBM Residual)
# ============================================================
 
def prepare_forecast_origin_frame(panel_lgb, forecast_origin, future_calendar_region, horizons):
    forecast_origin_df = panel_lgb[panel_lgb["month_date"] == forecast_origin].copy()
    forecast_origin_df = forecast_origin_df[forecast_origin_df["quantity_current"].notna()].copy()
 
    log("Series forecasted:", f"{len(forecast_origin_df):,}")
 
    for h in horizons:
        target_month = forecast_origin + pd.DateOffset(months=h)
 
        future_h = future_calendar_region[future_calendar_region["month_date"] == target_month].copy()
 
        rename_cols = {
            c: f"target_{c}_h{h}" for c in future_h.columns
            if c not in ["month_date", "school_holiday_region"]
        }
        future_h = future_h.rename(columns=rename_cols)
        future_h = future_h.drop(columns="month_date")
 
        forecast_origin_df = forecast_origin_df.merge(
            future_h, on="school_holiday_region", how="left", suffixes=("", "_future")
        )
 
        for original_col in rename_cols.values():
            future_version = original_col + "_future"
            if future_version in forecast_origin_df.columns:
                forecast_origin_df[original_col] = forecast_origin_df[future_version]
                forecast_origin_df.drop(columns=future_version, inplace=True)
 
        forecast_origin_df[f"target_month_h{h}"] = target_month.month
        forecast_origin_df[f"target_quarter_h{h}"] = target_month.quarter
        forecast_origin_df[f"target_month_sin_h{h}"] = np.sin(2 * np.pi * target_month.month / 12)
        forecast_origin_df[f"target_month_cos_h{h}"] = np.cos(2 * np.pi * target_month.month / 12)
 
        forecast_origin_df[f"working_days_change_h{h}"] = (
            forecast_origin_df[f"target_working_days_h{h}"] - forecast_origin_df["current_working_days"]
        )
        forecast_origin_df[f"school_holiday_change_h{h}"] = (
            forecast_origin_df[f"target_school_holiday_ratio_selected_h{h}"]
            - forecast_origin_df["current_school_holiday_ratio_selected"]
        )
        forecast_origin_df[f"public_holiday_change_h{h}"] = (
            forecast_origin_df[f"target_public_holidays_weekdays_h{h}"]
            - forecast_origin_df["current_public_holidays_weekdays"]
        )
 
    # -- align categorical dtypes with training data --
    for col in CATEGORICAL_FEATURES:
        if col not in forecast_origin_df.columns:
            continue
        if col in panel_lgb.columns:
            training_categories = panel_lgb[col].cat.categories
            forecast_origin_df[col] = pd.Categorical(forecast_origin_df[col], categories=training_categories)
 
    return forecast_origin_df
 
 
def train_and_forecast_site(panel_lgb, forecast_origin_df, forecast_origin, horizons):
    """
    Réentraîne un modèle LightGBM Residual par horizon (H1..H6) sur tout
    l'historique disponible, et génère le forecast site x supplier x category.
    Retourne (site_forecast_df, training_rows_by_horizon).
    """
    site_forecast_blocks = []
    training_rows_by_horizon = {}
 
    for h in horizons:
        log("\n" + "=" * 80)
        log(f"TRAINING SITE MODEL — H{h}")
        log("=" * 80)
 
        target = f"target_h{h}"
        features = get_features(panel_lgb, h)
        categorical_now = [c for c in CATEGORICAL_FEATURES if c in features]
 
        train = panel_lgb[
            panel_lgb[target].notna() & panel_lgb["quantity_current"].notna()
        ].copy()
 
        y_train = train[target] - train["quantity_current"]
 
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
 
        model.fit(train[features], y_train, categorical_feature=categorical_now)
 
        training_rows_by_horizon[h] = len(train)
        log("Training rows:", f"{len(train):,}")
 
        pred_change = model.predict(forecast_origin_df[features])
        prediction = forecast_origin_df["quantity_current"].values + pred_change
        prediction = np.maximum(prediction, 0)
 
        result = forecast_origin_df[SERIES_COLS + ["quantity_current"]].copy()
        result["forecast_origin"] = forecast_origin
        result["horizon"] = h
        result["target_month"] = forecast_origin + pd.DateOffset(months=h)
        result["model"] = SITE_MODEL_NAME
        result["forecast_quantity"] = prediction
 
        site_forecast_blocks.append(result)
 
    site_forecast = pd.concat(site_forecast_blocks, ignore_index=True)
    site_forecast = site_forecast.sort_values(
        SERIES_COLS + ["target_month"]
    ).reset_index(drop=True)
 
    log("site_forecast shape:", site_forecast.shape)
 
    return site_forecast, training_rows_by_horizon
 
 
# ============================================================
# 7. BASELINE — SUPPLIER x CATEGORY (Last Quantity)
# ============================================================
 
def build_supplier_category_forecast(df, forecast_origin, horizons):
    aggregate_origin = (
        df[df["month_date"] == forecast_origin]
        .groupby(["supplier_id", "category_id"], as_index=False, observed=True)
        .agg(quantity_current=("quantity", "sum"))
    )
 
    blocks = []
    for h in horizons:
        result = aggregate_origin.copy()
        result["forecast_origin"] = forecast_origin
        result["horizon"] = h
        result["target_month"] = forecast_origin + pd.DateOffset(months=h)
        result["model"] = BASELINE_MODEL_NAME
        result["forecast_quantity"] = result["quantity_current"]
        blocks.append(result)
 
    aggregate_forecast = pd.concat(blocks, ignore_index=True)
    aggregate_forecast = aggregate_forecast.sort_values(
        ["supplier_id", "category_id", "target_month"]
    ).reset_index(drop=True)
 
    return aggregate_forecast
 
 
# ============================================================
# 8. STANDARDIZE forecast_output SCHEMA + HISTORIZATION
# ============================================================
 
FORECAST_OUTPUT_COLUMNS = [
    "forecast_origin", "target_month", "horizon", "forecast_level",
    "site_id", "supplier_id", "category_id",
    "forecast_quantity", "model",
]
 
 
def standardize_forecast_output(site_forecast: pd.DataFrame, aggregate_forecast: pd.DataFrame) -> pd.DataFrame:
    site_out = site_forecast.copy()
    site_out["forecast_level"] = FORECAST_LEVEL_SITE
    site_out["horizon"] = "H" + site_out["horizon"].astype(str)
    site_out = site_out[[
        "forecast_origin", "target_month", "horizon", "forecast_level",
        "site_id", "supplier_id", "category_id", "forecast_quantity", "model",
    ]]
 
    agg_out = aggregate_forecast.copy()
    agg_out["forecast_level"] = FORECAST_LEVEL_SUPPLIER_CATEGORY
    agg_out["horizon"] = "H" + agg_out["horizon"].astype(str)
    agg_out["site_id"] = pd.NA
    agg_out = agg_out[[
        "forecast_origin", "target_month", "horizon", "forecast_level",
        "site_id", "supplier_id", "category_id", "forecast_quantity", "model",
    ]]
 
    out = pd.concat([site_out, agg_out], ignore_index=True)
    for c in ["forecast_origin", "target_month"]:
        out[c] = pd.to_datetime(out[c])
    return out[FORECAST_OUTPUT_COLUMNS]
 
 
def historize_forecast_output(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les nouveaux forecasts à l'historique SANS supprimer les
    versions précédentes. Un même (forecast_origin, target_month, horizon,
    forecast_level, model, site_id, supplier_id, category_id) réémis lors
    d'un rerun du même mois remplace la ligne précédente (dédoublonnage),
    mais un forecast_origin différent (ex: juin vs juillet) coexiste
    toujours — c'est le comportement voulu.
    """
    if existing is None or existing.empty:
        combined = new_rows.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
 
    key_cols = [
        "forecast_origin", "target_month", "horizon", "forecast_level",
        "model", "site_id", "supplier_id", "category_id",
    ]
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(["forecast_origin", "target_month", "forecast_level", "model"])
    return combined.reset_index(drop=True)
 
 
# ============================================================
# 9. forecast_actual_comparison
# ============================================================
 
COMPARISON_COLUMNS = [
    "forecast_origin", "target_month", "horizon", "forecast_level", "model",
    "site_id", "supplier_id", "category_id",
    "forecast_quantity", "actual_quantity",
    "error", "absolute_error", "percentage_error",
    "baseline_forecast", "baseline_error", "model_better_than_baseline",
]
 
 
def _build_key(frame: pd.DataFrame, cols) -> pd.Series:
    """
    Construit une clé texte robuste aux NaN (ex: site_id vide pour les
    lignes SUPPLIER_CATEGORY) afin de pouvoir comparer / dédupliquer des
    lignes entre deux DataFrames.
    """
    return frame[cols].fillna("<NA>").astype(str).agg("|".join, axis=1)
 
 
def _actual_lookup_site(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month_date"] + SERIES_COLS, as_index=False, observed=True)
        .agg(actual_quantity=("quantity", "sum"))
    )
 
 
def _actual_lookup_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["month_date", "supplier_id", "category_id"], as_index=False, observed=True)
        .agg(actual_quantity=("quantity", "sum"))
    )
 
 
def update_actual_comparison(df: pd.DataFrame, forecast_output_history: pd.DataFrame,
                              comparison_history: pd.DataFrame, current_origin: pd.Timestamp) -> pd.DataFrame:
    if forecast_output_history is None or forecast_output_history.empty:
        return comparison_history if comparison_history is not None else pd.DataFrame(columns=COMPARISON_COLUMNS)
 
    candidates = forecast_output_history[
        forecast_output_history["target_month"] <= current_origin
    ].copy()
 
    if candidates.empty:
        return comparison_history if comparison_history is not None else pd.DataFrame(columns=COMPARISON_COLUMNS)
 
    key_cols = ["forecast_origin", "target_month", "horizon", "forecast_level", "model",
                "site_id", "supplier_id", "category_id"]
 
    if comparison_history is not None and not comparison_history.empty:
        existing_keys = _build_key(comparison_history, key_cols)
        candidate_keys = _build_key(candidates, key_cols)
        candidates = candidates[~candidate_keys.isin(set(existing_keys))]
 
    if candidates.empty:
        return comparison_history if comparison_history is not None else pd.DataFrame(columns=COMPARISON_COLUMNS)
 
    site_lookup = _actual_lookup_site(df)
    agg_lookup = _actual_lookup_agg(df)
 
    new_rows_blocks = []
 
    # -- SITE level --
    site_candidates = candidates[candidates["forecast_level"] == FORECAST_LEVEL_SITE].copy()
    if not site_candidates.empty:
        site_candidates = site_candidates.merge(
            site_lookup.rename(columns={"month_date": "target_month"}),
            on=["target_month"] + SERIES_COLS, how="left",
        )
        baseline_lookup = site_lookup.rename(
            columns={"month_date": "forecast_origin", "actual_quantity": "baseline_forecast"}
        )
        site_candidates = site_candidates.merge(
            baseline_lookup, on=["forecast_origin"] + SERIES_COLS, how="left",
        )
        new_rows_blocks.append(site_candidates)
 
    # -- SUPPLIER_CATEGORY level --
    agg_candidates = candidates[candidates["forecast_level"] == FORECAST_LEVEL_SUPPLIER_CATEGORY].copy()
    if not agg_candidates.empty:
        agg_candidates = agg_candidates.merge(
            agg_lookup.rename(columns={"month_date": "target_month"}),
            on=["target_month", "supplier_id", "category_id"], how="left",
        )
        baseline_lookup = agg_lookup.rename(
            columns={"month_date": "forecast_origin", "actual_quantity": "baseline_forecast"}
        )
        agg_candidates = agg_candidates.merge(
            baseline_lookup, on=["forecast_origin", "supplier_id", "category_id"], how="left",
        )
        new_rows_blocks.append(agg_candidates)
 
    if not new_rows_blocks:
        return comparison_history if comparison_history is not None else pd.DataFrame(columns=COMPARISON_COLUMNS)
 
    new_rows = pd.concat(new_rows_blocks, ignore_index=True)
    new_rows = new_rows[new_rows["actual_quantity"].notna()].copy()
 
    if new_rows.empty:
        return comparison_history if comparison_history is not None else pd.DataFrame(columns=COMPARISON_COLUMNS)
 
    new_rows["error"] = new_rows["forecast_quantity"] - new_rows["actual_quantity"]
    new_rows["absolute_error"] = new_rows["error"].abs()
    new_rows["percentage_error"] = np.where(
        new_rows["actual_quantity"] != 0,
        new_rows["error"] / new_rows["actual_quantity"],
        np.nan,
    )
    new_rows["baseline_error"] = new_rows["baseline_forecast"] - new_rows["actual_quantity"]
    new_rows["model_better_than_baseline"] = (
        new_rows["absolute_error"] < new_rows["baseline_error"].abs()
    )
 
    new_rows = new_rows[COMPARISON_COLUMNS]
 
    if comparison_history is None or comparison_history.empty:
        combined = new_rows.copy()
    else:
        combined = pd.concat([comparison_history, new_rows], ignore_index=True)
 
    log(f"forecast_actual_comparison: +{len(new_rows):,} new rows "
        f"(total {len(combined):,})")
 
    return combined.reset_index(drop=True)
 
 
# ============================================================
# 10. forecast_monitoring
# ============================================================
 
MONITORING_COLUMNS = [
    "forecast_origin", "actual_month", "horizon", "forecast_level", "model",
    "n", "actual_volume", "forecast_volume", "MAE", "WAPE", "Bias",
    "baseline_WAPE", "WAPE_gain_vs_baseline", "relative_gain_pct",
    "Positive_MAE", "Positive_WAPE", "Positive_rate",
    "WAPE_rolling_3m", "Bias_rolling_3m",
]
 
 
def _safe_div(a, b):
    return np.where(b != 0, a / b, np.nan)
 
 
def build_monitoring(comparison_all: pd.DataFrame) -> pd.DataFrame:
    if comparison_all is None or comparison_all.empty:
        return pd.DataFrame(columns=MONITORING_COLUMNS)
 
    def agg_group(g):
        actual_volume = g["actual_quantity"].sum()
        forecast_volume = g["forecast_quantity"].sum()
        mae = g["absolute_error"].mean()
        wape = _safe_div(g["absolute_error"].sum(), actual_volume)
        bias = _safe_div(g["error"].sum(), actual_volume)
        baseline_wape = _safe_div(g["baseline_error"].abs().sum(), actual_volume)
        gain = float(baseline_wape) - float(wape) if pd.notna(baseline_wape) and pd.notna(wape) else np.nan
        relative_gain_pct = (gain / baseline_wape * 100) if baseline_wape not in (0, np.nan) and pd.notna(baseline_wape) and baseline_wape != 0 else np.nan
 
        pos = g[g["actual_quantity"] > 0]
        positive_mae = pos["absolute_error"].mean() if len(pos) else np.nan
        positive_wape = _safe_div(pos["absolute_error"].sum(), pos["actual_quantity"].sum()) if len(pos) else np.nan
        positive_rate = (g["actual_quantity"] > 0).mean()
 
        return pd.Series({
            "n": len(g),
            "actual_volume": actual_volume,
            "forecast_volume": forecast_volume,
            "MAE": mae,
            "WAPE": float(wape),
            "Bias": float(bias),
            "baseline_WAPE": float(baseline_wape),
            "WAPE_gain_vs_baseline": gain,
            "relative_gain_pct": relative_gain_pct,
            "Positive_MAE": positive_mae,
            "Positive_WAPE": float(positive_wape) if positive_wape is not None else np.nan,
            "Positive_rate": positive_rate,
        })
 
    grouped = (
        comparison_all
        .groupby(["forecast_origin", "target_month", "horizon", "forecast_level", "model"], as_index=False)
        .apply(agg_group)
    )
    grouped = grouped.rename(columns={"target_month": "actual_month"})
 
    # -- rolling metrics (3 mois), triées par actual_month au sein de
    #    chaque (horizon, forecast_level, model) --
    grouped = grouped.sort_values(["forecast_level", "model", "horizon", "actual_month"])
    grouped["WAPE_rolling_3m"] = (
        grouped.groupby(["forecast_level", "model", "horizon"])["WAPE"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_MONTHS, min_periods=1).mean())
    )
    grouped["Bias_rolling_3m"] = (
        grouped.groupby(["forecast_level", "model", "horizon"])["Bias"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_MONTHS, min_periods=1).mean())
    )
 
    grouped = grouped.sort_values(["forecast_origin", "horizon", "forecast_level", "model"]).reset_index(drop=True)
    return grouped[MONITORING_COLUMNS]
 
 
# ============================================================
# 11. forecast_run
# ============================================================
 
RUN_COLUMNS = [
    "run_id", "forecast_origin", "training_start", "training_end",
    "model_version", "feature_version", "n_training_rows",
    "forecast_start", "forecast_end", "generated_at", "rows",
]
 
 
def build_run_record(forecast_origin, panel, training_rows_by_horizon, new_forecast_rows: pd.DataFrame) -> pd.DataFrame:
    run_id = forecast_origin.strftime("%Y_%m_01")
    training_start = panel["month_date"].min()
    training_end = forecast_origin
 
    record = {
        "run_id": run_id,
        "forecast_origin": forecast_origin,
        "training_start": training_start,
        "training_end": training_end,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "n_training_rows": sum(training_rows_by_horizon.values()),
        "forecast_start": new_forecast_rows["target_month"].min(),
        "forecast_end": new_forecast_rows["target_month"].max(),
        "generated_at": datetime.now(),
        "rows": len(new_forecast_rows),
    }
    return pd.DataFrame([record])[RUN_COLUMNS]
 
 
def append_run(existing: pd.DataFrame, new_record: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        combined = new_record.copy()
    else:
        combined = pd.concat([existing, new_record], ignore_index=True)
    combined = combined.drop_duplicates(subset=["run_id"], keep="last")
    return combined.sort_values("forecast_origin").reset_index(drop=True)
 
 
# ============================================================
# 12. I/O HELPERS
# ============================================================
 
def read_csv_if_exists(path: Path, date_cols=None, columns=None) -> pd.DataFrame:
    """
    Charge un CSV historique s'il existe. S'il n'existe pas encore (premher
    lancement) OU s'il est vide, retourne un DataFrame vide mais avec le
    bon schéma de colonnes (`columns`), pour que le reste du pipeline (qui
    teste `.empty` et concatène) se comporte de façon homogène entre le
    premier run et les runs suivants.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
 
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()
 
    if columns is not None and df.shape[1] == 0:
        return pd.DataFrame(columns=columns)
 
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
    return df
 
 
def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("Saved:", path)
 
 
# ============================================================
# 13. MAIN PIPELINE
# ============================================================
 
def main():
    log("=" * 80)
    log("MONTHLY FORECAST PIPELINE")
    log("=" * 80)
 
    # ---- 1. Charge ml_demand_forecast_base.csv ----
    df = load_data(INPUT_FILE)
 
    # ---- 2. Détecte automatiquement le forecast_origin ----
    forecast_origin = df["month_date"].max()
    log("\nDetected forecast_origin:", forecast_origin.date())
 
    # ---- Charge l'historique existant des 4 tables ----
    forecast_output_history = read_csv_if_exists(
        FORECAST_OUTPUT_FILE, date_cols=["forecast_origin", "target_month"],
        columns=FORECAST_OUTPUT_COLUMNS,
    )
    comparison_history = read_csv_if_exists(
        FORECAST_COMPARISON_FILE, date_cols=["forecast_origin", "target_month"],
        columns=COMPARISON_COLUMNS,
    )
    run_history = read_csv_if_exists(
        FORECAST_RUN_FILE,
        date_cols=["forecast_origin", "training_start", "training_end",
                   "forecast_start", "forecast_end", "generated_at"],
        columns=RUN_COLUMNS,
    )
 
    # ---- 3 & 4. Met à jour forecast_actual_comparison ----
    log("\nUpdating forecast_actual_comparison...")
    comparison_all = update_actual_comparison(
        df, forecast_output_history, comparison_history, forecast_origin
    )
    save_csv(comparison_all, FORECAST_COMPARISON_FILE)
 
    # ---- 5. Recalcule forecast_monitoring ----
    log("\nRecomputing forecast_monitoring...")
    monitoring_all = build_monitoring(comparison_all)
    save_csv(monitoring_all, FORECAST_MONITORING_FILE)
 
    # ---- 6. Reconstruit le panel + features ----
    log("\nBuilding panel & features...")
    series_metadata = build_series_metadata(df)
    calendar_region = build_calendar_region(df)
    cpi_lookup = build_cpi_lookup(df)
    panel = build_panel(df, series_metadata, calendar_region, cpi_lookup)
    panel_lgb = prepare_lgb_frame(panel)
 
    historical_month_ratio = build_historical_month_ratio(calendar_region)
    future_calendar_region = build_future_calendar_region(
        forecast_origin, FORECAST_HORIZONS, historical_month_ratio
    )
 
    forecast_origin_df = prepare_forecast_origin_frame(
        panel_lgb, forecast_origin, future_calendar_region, FORECAST_HORIZONS
    )
 
    # ---- 7 & 8. Réentraîne 6 LightGBM Residual + forecast SITE ----
    log("\nTraining LightGBM Residual models (H1-H6) & forecasting SITE level...")
    site_forecast, training_rows_by_horizon = train_and_forecast_site(
        panel_lgb, forecast_origin_df, forecast_origin, FORECAST_HORIZONS
    )
 
    # ---- 9. Forecast SUPPLIER_CATEGORY (Last Quantity) ----
    log("\nBuilding SUPPLIER_CATEGORY baseline forecast (Last Quantity)...")
    aggregate_forecast = build_supplier_category_forecast(df, forecast_origin, FORECAST_HORIZONS)
 
    # ---- 10. Historise forecast_output + forecast_run ----
    log("\nStandardizing & historizing forecast_output...")
    new_forecast_rows = standardize_forecast_output(site_forecast, aggregate_forecast)
    forecast_output_all = historize_forecast_output(forecast_output_history, new_forecast_rows)
    save_csv(forecast_output_all, FORECAST_OUTPUT_FILE)
 
    run_record = build_run_record(forecast_origin, panel, training_rows_by_horizon, new_forecast_rows)
    run_history_all = append_run(run_history, run_record)
    save_csv(run_history_all, FORECAST_RUN_FILE)
 
    log("\n" + "=" * 80)
    log("DONE.")
    log(f"forecast_origin      : {forecast_origin.date()}")
    log(f"new forecast rows    : {len(new_forecast_rows):,}")
    log(f"forecast_output total: {len(forecast_output_all):,}")
    log(f"comparison total     : {len(comparison_all):,}")
    log(f"monitoring rows      : {len(monitoring_all):,}")
    log("=" * 80)
 
 
if __name__ == "__main__":
    main()
