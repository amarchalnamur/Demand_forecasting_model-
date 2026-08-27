
# IMPORTS & CONFIGURATION


import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

RANDOM_STATE = 42


FILEPATH = r"C:\Users\arnau\UNAMUR\projet perso\ml_demand_forecast_base.csv"


SERIES_COLS = [
    "site_id",
    "supplier_id",
    "category_id",
]



AGG_COLS = ["category_id"]

DATE_COL = "month_date"
TARGET_COL = "quantity"

FORECAST_HORIZONS = [1, 2, 3, 4, 5, 6]


N_BACKTEST_ORIGINS = 6


MIN_HISTORY_MONTHS = 24

MISSING_QUANTITY_AS_ZERO = False


OUTPUT_DIR = Path("./bottom_up_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration loaded.")



# CHARGEMENT & STANDARDISATION DES DONNEES


df = pd.read_csv(FILEPATH, low_memory=False)

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
)

df = df[df[DATE_COL].notna()].copy()

categorical_raw_cols = [
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

numeric_raw_cols = [
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

for col in categorical_raw_cols:
    if col in df.columns:
        df[col] = df[col].astype("string")

for col in numeric_raw_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("Date range:", df[DATE_COL].min(), "->", df[DATE_COL].max())




# MÉTADONNÉES CALENDRIER ET CPI


metadata_cols = [
    "region",
    "school_holiday_region",
    "municipality_confidence",
    "category_level_1_id",
    "category_level_2_id",
    "category_level_3_id",
    "cpi_id",
]

metadata_cols = [c for c in metadata_cols if c in df.columns]

series_metadata = (
    df[SERIES_COLS + metadata_cols]
    .groupby(
        SERIES_COLS,
        as_index=False,
        observed=True
    )
    .first()
)

calendar_cols = [
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
calendar_cols = [c for c in calendar_cols if c in df.columns]

calendar_region = (
    df[calendar_cols]
    .groupby(
        ["month_date", "school_holiday_region"],
        as_index=False,
        observed=True
    )
    .median(numeric_only=True)
)

cpi_cols = [
    "month_date",
    "cpi_id",
    "cpi_index",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "cpi_coefficient",
]
cpi_cols = [c for c in cpi_cols if c in df.columns]

cpi_lookup = (
    df[cpi_cols]
    .groupby(
        ["month_date", "cpi_id"],
        as_index=False,
        observed=True
    )
    .median(numeric_only=True)
)

print("Metadata rows:", len(series_metadata))
print("Calendar rows:", len(calendar_region))
print("CPI rows:", len(cpi_lookup))


# PANEL MENSUEL COMPLET GRAIN SITE


all_months = pd.DataFrame({
    DATE_COL: pd.date_range(
        start=df[DATE_COL].min(),
        end=df[DATE_COL].max(),
        freq="MS"
    )
})

series_keys = df[SERIES_COLS].drop_duplicates()

panel = series_keys.merge(all_months, how="cross")

actual_cols = [
    DATE_COL,
    *SERIES_COLS,
    TARGET_COL,
]
if "gross_spend" in df.columns:
    actual_cols.append("gross_spend")

actuals = df[actual_cols].copy()

panel = panel.merge(
    actuals,
    on=[DATE_COL] + SERIES_COLS,
    how="left"
)

panel = panel.merge(
    series_metadata,
    on=SERIES_COLS,
    how="left"
)

if "cpi_id" in panel.columns and "cpi_id" in cpi_lookup.columns:
    panel = panel.merge(
        cpi_lookup,
        on=[DATE_COL, "cpi_id"],
        how="left"
    )

panel = (
    panel
    .sort_values(SERIES_COLS + [DATE_COL])
    .reset_index(drop=True)
)


if MISSING_QUANTITY_AS_ZERO:
    panel[TARGET_COL] = panel[TARGET_COL].fillna(0.0)
    if "gross_spend" in panel.columns:
        panel["gross_spend"] = panel["gross_spend"].fillna(0.0)

print("Panel shape:", panel.shape)



# CONSTRUCTION DES FEATURES


rename_current = {
    col: f"current_{col}"
    for col in calendar_region.columns
    if col not in ["month_date", "school_holiday_region"]
}

current_calendar = calendar_region.rename(columns=rename_current)

panel = panel.merge(
    current_calendar,
    on=["month_date", "school_holiday_region"],
    how="left"
)

# QUANTITE ACTUELLE

panel["quantity_current"] = panel[TARGET_COL]

grouped = panel.groupby(SERIES_COLS, observed=True)

for lag in range(1, 13):
    panel[f"quantity_lag_{lag}"] = (
        grouped[TARGET_COL].shift(lag)
    )

recent_3 = ["quantity_current", "quantity_lag_1", "quantity_lag_2"]
recent_6 = ["quantity_current"] + [f"quantity_lag_{i}" for i in range(1, 6)]
recent_12 = ["quantity_current"] + [f"quantity_lag_{i}" for i in range(1, 12)]

for name, cols in [("3", recent_3), ("6", recent_6), ("12", recent_12)]:
    panel[f"quantity_mean_{name}"] = panel[cols].mean(axis=1)
    panel[f"quantity_std_{name}"] = panel[cols].std(axis=1)

# FEATURE D'INTERMITTENCE

panel["demand_positive_current"] = (
    panel["quantity_current"] > 0
).astype(float)

for name, cols in [("3", recent_3), ("6", recent_6), ("12", recent_12)]:
    panel[f"positive_months_{name}"] = (panel[cols] > 0).sum(axis=1)
    panel[f"zero_ratio_{name}"] = (panel[cols] == 0).mean(axis=1)

panel["positive_quantity_mean_6"] = (
    panel[recent_6]
    .where(panel[recent_6] > 0)
    .mean(axis=1)
)

panel["positive_quantity_mean_12"] = (
    panel[recent_12]
    .where(panel[recent_12] > 0)
    .mean(axis=1)
)

panel["_positive_quantity"] = panel[TARGET_COL].where(panel[TARGET_COL] > 0)

panel["last_positive_quantity"] = (
    panel
    .groupby(SERIES_COLS, observed=True)["_positive_quantity"]
    .ffill()
)

panel.drop(columns="_positive_quantity", inplace=True)



# FEATURE PRIX



if "gross_spend" in panel.columns:
    panel["unit_price_current"] = np.where(
        panel[TARGET_COL] > 0,
        panel["gross_spend"] / panel[TARGET_COL],
        np.nan
    )
else:
    panel["unit_price_current"] = np.nan

grouped = panel.groupby(SERIES_COLS, observed=True)

for lag in [1, 2, 3, 6, 12]:
    panel[f"unit_price_lag_{lag}"] = (
        grouped["unit_price_current"].shift(lag)
    )

panel["unit_price_mean_3"] = panel[
    ["unit_price_current", "unit_price_lag_1", "unit_price_lag_2"]
].mean(axis=1)

panel["unit_price_change_1m"] = (
    panel["unit_price_current"] / panel["unit_price_lag_1"] - 1
).clip(-0.8, 3)

panel["unit_price_change_3m"] = (
    panel["unit_price_current"] / panel["unit_price_lag_3"] - 1
).clip(-0.8, 3)



# FEATURE CPI HISTORIQUE


grouped = panel.groupby(SERIES_COLS, observed=True)

for col in [
    "cpi_index",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "cpi_coefficient",
]:
    if col not in panel.columns:
        panel[col] = np.nan

    for lag in [1, 3, 6]:
        panel[f"{col}_lag_{lag}"] = grouped[col].shift(lag)

panel["cpi_index_change_3m"] = (
    panel["cpi_index"] / panel["cpi_index_lag_3"] - 1
)

panel["cpi_index_change_6m"] = (
    panel["cpi_index"] / panel["cpi_index_lag_6"] - 1
)


# TARGETS HORIZONS H1-H6

grouped = panel.groupby(SERIES_COLS, observed=True)

for h in FORECAST_HORIZONS:
    panel[f"target_h{h}"] = grouped[TARGET_COL].shift(-h)
    panel[f"target_date_h{h}"] = (
        panel[DATE_COL] + pd.DateOffset(months=h)
    )


# FEATURES CALANDAIRES DU MOIS CIBLE

for h in FORECAST_HORIZONS:
    future_calendar = calendar_region.copy()


    future_calendar[DATE_COL] = (
        future_calendar[DATE_COL] - pd.DateOffset(months=h)
    )

    rename_future = {
        col: f"target_{col}_h{h}"
        for col in future_calendar.columns
        if col not in [DATE_COL, "school_holiday_region"]
    }

    future_calendar = future_calendar.rename(columns=rename_future)

    panel = panel.merge(
        future_calendar,
        on=[DATE_COL, "school_holiday_region"],
        how="left"
    )


# SAISONNALITE MOIS CIBLE


for h in FORECAST_HORIZONS:
    target_date = panel[DATE_COL] + pd.DateOffset(months=h)

    panel[f"target_month_h{h}"] = target_date.dt.month
    panel[f"target_quarter_h{h}"] = target_date.dt.quarter
    panel[f"target_month_sin_h{h}"] = np.sin(
        2 * np.pi * target_date.dt.month / 12
    )
    panel[f"target_month_cos_h{h}"] = np.cos(
        2 * np.pi * target_date.dt.month / 12
    )

    if (
        f"target_working_days_h{h}" in panel.columns
        and "current_working_days" in panel.columns
    ):
        panel[f"working_days_change_h{h}"] = (
            panel[f"target_working_days_h{h}"]
            - panel["current_working_days"]
        )

    if (
        f"target_school_holiday_ratio_selected_h{h}" in panel.columns
        and "current_school_holiday_ratio_selected" in panel.columns
    ):
        panel[f"school_holiday_change_h{h}"] = (
            panel[f"target_school_holiday_ratio_selected_h{h}"]
            - panel["current_school_holiday_ratio_selected"]
        )

    if (
        f"target_public_holidays_weekdays_h{h}" in panel.columns
        and "current_public_holidays_weekdays" in panel.columns
    ):
        panel[f"public_holiday_change_h{h}"] = (
            panel[f"target_public_holidays_weekdays_h{h}"]
            - panel["current_public_holidays_weekdays"]
        )


# INDEX TEMPS

first_date = panel[DATE_COL].min()

panel["time_index"] = (
    (panel[DATE_COL].dt.year - first_date.year) * 12
    + (panel[DATE_COL].dt.month - first_date.month)
)

print("Feature engineering complete:", panel.shape)





# REGROUPEMENT DES FEATURES


numeric_common = [
    # Demand
    "quantity_current",
    "quantity_lag_1",
    "quantity_lag_2",
    "quantity_lag_3",
    "quantity_lag_4",
    "quantity_lag_5",
    "quantity_lag_6",
    "quantity_lag_9",
    "quantity_lag_12",
    "quantity_mean_3",
    "quantity_mean_6",
    "quantity_mean_12",
    "quantity_std_3",
    "quantity_std_6",
    "quantity_std_12",


    "demand_positive_current",
    "positive_months_3",
    "positive_months_6",
    "positive_months_12",
    "zero_ratio_3",
    "zero_ratio_6",
    "zero_ratio_12",
    "positive_quantity_mean_6",
    "positive_quantity_mean_12",
    "last_positive_quantity",


    "unit_price_current",
    "unit_price_lag_1",
    "unit_price_lag_2",
    "unit_price_lag_3",
    "unit_price_lag_6",
    "unit_price_lag_12",
    "unit_price_mean_3",
    "unit_price_change_1m",
    "unit_price_change_3m",


    "cpi_index",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "cpi_coefficient",
    "cpi_index_lag_1",
    "cpi_index_lag_3",
    "cpi_index_lag_6",
    "cpi_mom_pct_lag_1",
    "cpi_yoy_pct_lag_1",
    "cpi_index_change_3m",
    "cpi_index_change_6m",


    "current_working_days",
    "current_public_holidays_weekdays",
    "current_school_holiday_ratio_selected",


    "time_index",
]


    "site_id",
    "supplier_id",
    "category_id",
    "category_level_1_id",
    "category_level_2_id",
    "category_level_3_id",
    "region",
    "school_holiday_region",
    "municipality_confidence",
    "cpi_id",
]

numeric_common = [c for c in numeric_common if c in panel.columns]
categorical_features = [c for c in categorical_features if c in panel.columns]


def get_features(horizon):
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

    features = (
        numeric_common
        + future_features
        + categorical_features
    )

    return [
        col
        for col in dict.fromkeys(features)
        if col in panel.columns
    ]


panel_lgb = panel.copy()

for col in categorical_features:
    panel_lgb[col] = panel_lgb[col].astype("category")

for h in FORECAST_HORIZONS:
    print(f"H{h}: {len(get_features(h))} features")




# ORIGINES D'ENTRAINEMENT


all_dates = pd.DatetimeIndex(
    sorted(panel_lgb[DATE_COL].dropna().unique())
)

# Pour utiliser les mêmes origines pour H1...H6,
# la dernière origine doit encore avoir H6 observable.
max_h = max(FORECAST_HORIZONS)
last_origin = (
    panel_lgb[DATE_COL].max()
    - pd.DateOffset(months=max_h)
)

first_allowed_origin = (
    panel_lgb[DATE_COL].min()
    + pd.DateOffset(months=MIN_HISTORY_MONTHS)
)

eligible_origins = all_dates[
    (all_dates >= first_allowed_origin)
    & (all_dates <= last_origin)
]

if len(eligible_origins) == 0:
    raise ValueError(
        "Pas assez d'historique pour construire les backtests "
        "avec les paramètres actuels."
    )

backtest_origins = list(
    eligible_origins[-N_BACKTEST_ORIGINS:]
)

print("Backtest origins:")
for origin in backtest_origins:
    print(" -", pd.Timestamp(origin).date())





# MÉTRIQUES ET FONCTIONS DU MODÈLE


def make_model():
    return LGBMRegressor(
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


def metric_summary(data, actual_col, pred_col):
    tmp = data[[actual_col, pred_col]].dropna().copy()

    if len(tmp) == 0:
        return pd.Series({
            "n": 0,
            "actual_sum": np.nan,
            "forecast_sum": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "wape_pct": np.nan,
            "bias_pct": np.nan,
        })

    actual = tmp[actual_col].astype(float)
    pred = tmp[pred_col].astype(float)
    error = pred - actual

    denom = np.abs(actual).sum()

    return pd.Series({
        "n": len(tmp),
        "actual_sum": actual.sum(),
        "forecast_sum": pred.sum(),
        "mae": np.abs(error).mean(),
        "rmse": np.sqrt(np.mean(error ** 2)),
        "wape_pct": (
            100 * np.abs(error).sum() / denom
            if denom > 0 else np.nan
        ),
        "bias_pct": (
            100 * error.sum() / denom
            if denom > 0 else np.nan
        ),
    })



# ENTRAINEMENT AU NIVEAU SITE


site_backtest_blocks = []

for origin in backtest_origins:

    origin = pd.Timestamp(origin)

    print("\n" + "=" * 90)
    print("BACKTEST ORIGIN:", origin.date())
    print("=" * 90)

    for h in FORECAST_HORIZONS:

        target_col = f"target_h{h}"
        target_date_col = f"target_date_h{h}"
        features = get_features(h)

        categorical_now = [
            col
            for col in categorical_features
            if col in features
        ]


        train_mask = (
            panel_lgb[target_col].notna()
            & panel_lgb["quantity_current"].notna()
            & (panel_lgb[target_date_col] <= origin)
        )

        train = panel_lgb.loc[train_mask].copy()


        test_mask = (
            (panel_lgb[DATE_COL] == origin)
            & panel_lgb["quantity_current"].notna()
            & panel_lgb[target_col].notna()
        )

        test = panel_lgb.loc[test_mask].copy()

        if len(train) == 0 or len(test) == 0:
            print(
                f"H{h}: skipped "
                f"(train={len(train):,}, test={len(test):,})"
            )
            continue

        # Même logique résiduelle que dans ton prototype
        y_train = (
            train[target_col]
            - train["quantity_current"]
        )

        model = make_model()

        model.fit(
            train[features],
            y_train,
            categorical_feature=categorical_now,
        )

        pred_change = model.predict(test[features])

        prediction = (
            test["quantity_current"].to_numpy()
            + pred_change
        )

        prediction = np.maximum(prediction, 0)

        result_cols = list(dict.fromkeys(
            SERIES_COLS
            + AGG_COLS
            + [
                "region",
                "school_holiday_region",
                "quantity_current",
            ]
        ))

        result_cols = [
            c for c in result_cols
            if c in test.columns
        ]

        result = test[result_cols].copy()

        result["forecast_origin"] = origin
        result["horizon"] = h
        result["forecast_month"] = (
            origin + pd.DateOffset(months=h)
        )

        result["actual_quantity"] = (
            test[target_col].to_numpy()
        )

        result["forecast_quantity"] = prediction

        # Baseline : last quantity / quantité du mois d'origine
        result["baseline_quantity"] = (
            test["quantity_current"].to_numpy()
        )

        site_backtest_blocks.append(result)

        print(
            f"H{h}: train={len(train):,} | "
            f"test series={len(test):,}"
        )

site_backtest = pd.concat(
    site_backtest_blocks,
    ignore_index=True
)

site_backtest = (
    site_backtest
    .sort_values(
        ["forecast_origin", "horizon"]
        + SERIES_COLS
    )
    .reset_index(drop=True)
)

print("\nSite-level backtest shape:", site_backtest.shape)
display(site_backtest.head())




# AGRÉGATION BOTTOM-UP AU NIVEAU CATÉGORIE


bottom_up_group_cols = [
    "forecast_origin",
    "horizon",
    "forecast_month",
] + AGG_COLS

category_backtest = (
    site_backtest
    .groupby(
        bottom_up_group_cols,
        as_index=False,
        observed=True
    )
    .agg(
        actual_quantity=("actual_quantity", "sum"),
        forecast_quantity=("forecast_quantity", "sum"),
        baseline_quantity=("baseline_quantity", "sum"),
        n_bottom_series=("site_id", "size"),
    )
)

category_backtest["error"] = (
    category_backtest["forecast_quantity"]
    - category_backtest["actual_quantity"]
)

category_backtest["abs_error"] = (
    category_backtest["error"].abs()
)

category_backtest["baseline_error"] = (
    category_backtest["baseline_quantity"]
    - category_backtest["actual_quantity"]
)

category_backtest["baseline_abs_error"] = (
    category_backtest["baseline_error"].abs()
)

display(category_backtest.head(20))

print(
    "Aggregated bottom-up rows:",
    f"{len(category_backtest):,}"
)




# SCORES BOTTOM-UP AU NIVEAU CATÉGORIE



bottom_up_horizon_scores = (
    category_backtest
    .groupby("horizon", observed=True)
    .apply(
        lambda g: metric_summary(
            g,
            "actual_quantity",
            "forecast_quantity"
        ),
        include_groups=False
    )
    .reset_index()
)

baseline_horizon_scores = (
    category_backtest
    .groupby("horizon", observed=True)
    .apply(
        lambda g: metric_summary(
            g,
            "actual_quantity",
            "baseline_quantity"
        ),
        include_groups=False
    )
    .reset_index()
)

bottom_up_horizon_scores = bottom_up_horizon_scores.rename(
    columns={
        c: f"bottom_up_{c}"
        for c in bottom_up_horizon_scores.columns
        if c != "horizon"
    }
)

baseline_horizon_scores = baseline_horizon_scores.rename(
    columns={
        c: f"baseline_{c}"
        for c in baseline_horizon_scores.columns
        if c != "horizon"
    }
)

comparison_by_horizon = bottom_up_horizon_scores.merge(
    baseline_horizon_scores,
    on="horizon",
    how="left"
)

comparison_by_horizon["wape_improvement_pp"] = (
    comparison_by_horizon["baseline_wape_pct"]
    - comparison_by_horizon["bottom_up_wape_pct"]
)

comparison_by_horizon["mae_improvement_pct"] = np.where(
    comparison_by_horizon["baseline_mae"] > 0,
    100 * (
        comparison_by_horizon["baseline_mae"]
        - comparison_by_horizon["bottom_up_mae"]
    )
    / comparison_by_horizon["baseline_mae"],
    np.nan
)

display(comparison_by_horizon.round(3))



# SCORES PAR CATÉGORIE


category_score_group_cols = AGG_COLS + ["horizon"]

bottom_up_category_scores = (
    category_backtest
    .groupby(
        category_score_group_cols,
        observed=True
    )
    .apply(
        lambda g: metric_summary(
            g,
            "actual_quantity",
            "forecast_quantity"
        ),
        include_groups=False
    )
    .reset_index()
)

baseline_category_scores = (
    category_backtest
    .groupby(
        category_score_group_cols,
        observed=True
    )
    .apply(
        lambda g: metric_summary(
            g,
            "actual_quantity",
            "baseline_quantity"
        ),
        include_groups=False
    )
    .reset_index()
)

bottom_up_category_scores = bottom_up_category_scores.rename(
    columns={
        c: f"bottom_up_{c}"
        for c in bottom_up_category_scores.columns
        if c not in category_score_group_cols
    }
)

baseline_category_scores = baseline_category_scores.rename(
    columns={
        c: f"baseline_{c}"
        for c in baseline_category_scores.columns
        if c not in category_score_group_cols
    }
)

comparison_by_category = bottom_up_category_scores.merge(
    baseline_category_scores,
    on=category_score_group_cols,
    how="left"
)

comparison_by_category["wape_improvement_pp"] = (
    comparison_by_category["baseline_wape_pct"]
    - comparison_by_category["bottom_up_wape_pct"]
)

comparison_by_category = comparison_by_category.sort_values(
    ["horizon", "bottom_up_wape_pct"]
)

display(comparison_by_category.head(50).round(3))





# SCORE AU GRAIN SITE


site_model_scores = (
    site_backtest
    .groupby("horizon", observed=True)
    .apply(
        lambda g: metric_summary(
            g,
            "actual_quantity",
            "forecast_quantity"
        ),
        include_groups=False
    )
    .reset_index()
)

site_baseline_scores = (
    site_backtest
    .groupby("horizon", observed=True)
    .apply(
        lambda g: metric_summary(
            g,
            "actual_quantity",
            "baseline_quantity"
        ),
        include_groups=False
    )
    .reset_index()
)

print("Performance au grain site × supplier × category")
display(site_model_scores.round(3))

print("Baseline au grain site × supplier × category")
display(site_baseline_scores.round(3))






