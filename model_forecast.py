
#  IMPORTS


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42


# CONFIG GENERALE


FILEPATH = r"C:\Users\arnau\UNAMUR\projet perso\ml_demand_forecast_base.csv"

SERIES_COLS = [
    "site_id",
    "supplier_id",
    "category_id"
]

FORECAST_HORIZONS = [1, 2, 3, 4, 5, 6]

DATE_COL = "month_date"
TARGET_COL = "quantity"


FINAL_ORIGIN = pd.Timestamp("2026-06-01")


LGB_SITE_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


RF_BOTTOMUP_PARAMS = {
    "n_estimators": 250,
    "max_depth": 14,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

FORECAST_MONTHS = [
    FINAL_ORIGIN + pd.DateOffset(months=h)
    for h in FORECAST_HORIZONS
]

print("Configuration loaded.")
print("Forecast origin:", FINAL_ORIGIN.date())
for h, month in zip(FORECAST_HORIZONS, FORECAST_MONTHS):
    print(f"H{h} -> {month.date()}")


# IMPORT DES DONNEES


df = pd.read_csv(
    FILEPATH,
    low_memory=False
)

df["month_date"] = pd.to_datetime(
    df["month_date"],
    errors="coerce"
)

print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("Date range:", df["month_date"].min(), "->", df["month_date"].max())


# STANDARDIZER LES DONNEES


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
    "municipality_confidence"
]

for col in categorical_raw_cols:
    if col in df.columns:
        df[col] = df[col].astype("string")

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
    "cpi_coefficient"
]

for col in numeric_raw_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
      

# METADONNEES DE SERIES


metadata_cols = [
    "region",
    "school_holiday_region",
    "municipality_confidence",
    "category_level_1_id",
    "category_level_2_id",
    "category_level_3_id",
    "cpi_id"
]

series_metadata = (
    df[SERIES_COLS + metadata_cols]
    .groupby(
        SERIES_COLS,
        as_index=False,
        observed=True
    )
    .first()
)

print("Metadata rows:", len(series_metadata))


# LOOKUP DONNES DIM MONTH


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
    "school_holiday_weekdays_selected"
]

calendar_region = (
    df[calendar_cols]
    .groupby(
        ["month_date", "school_holiday_region"],
        as_index=False,
        observed=True
    )
    .median(numeric_only=True)
)

print(calendar_region.head())



# LOOKUP DONNEES CPI



cpi_lookup = (
    df[
        [
            "month_date",
            "cpi_id",
            "cpi_index",
            "cpi_mom_pct",
            "cpi_yoy_pct",
            "cpi_coefficient"
        ]
    ]
    .groupby(
        ["month_date", "cpi_id"],
        as_index=False,
        observed=True
    )
    .median(numeric_only=True)
)

print(cpi_lookup.head())


# CONSTRUCTION PANEL MENSUEL


all_months = pd.DataFrame({
    "month_date": pd.date_range(
        start=df["month_date"].min(),
        end=df["month_date"].max(),
        freq="MS"
    )
})

series_keys = df[SERIES_COLS].drop_duplicates()

panel = series_keys.merge(
    all_months,
    how="cross"
)

actuals = df[
    [
        "month_date",
        "site_id",
        "supplier_id",
        "category_id",
        "quantity",
        "gross_spend"
    ]
].copy()

panel = panel.merge(
    actuals,
    on=[
        "month_date",
        "site_id",
        "supplier_id",
        "category_id"
    ],
    how="left"
)

panel = panel.merge(
    series_metadata,
    on=SERIES_COLS,
    how="left"
)

panel = panel.merge(
    cpi_lookup,
    on=["month_date", "cpi_id"],
    how="left"
)

panel = (
    panel
    .sort_values(SERIES_COLS + ["month_date"])
    .reset_index(drop=True)
)

print("Panel shape:", panel.shape)


# FEATURES CALANDAIRES


current_calendar = calendar_region.copy()

rename_current = {
    col: f"current_{col}"
    for col in calendar_region.columns
    if col not in ["month_date", "school_holiday_region"]
}

current_calendar = current_calendar.rename(
    columns=rename_current
)

panel = panel.merge(
    current_calendar,
    on=["month_date", "school_holiday_region"],
    how="left"
)


# HISTORIQUE DE DEMANDE


panel["quantity_current"] = panel["quantity"]

grouped = panel.groupby(
    SERIES_COLS,
    observed=True
)

for lag in range(1, 13):
    panel[f"quantity_lag_{lag}"] = (
        grouped["quantity"].shift(lag)
    )


# FEATURE DE DEMANDE ROULANTES


recent_3 = [
    "quantity_current",
    "quantity_lag_1",
    "quantity_lag_2"
]

recent_6 = ["quantity_current"] + [
    f"quantity_lag_{i}"
    for i in range(1, 6)
]

recent_12 = ["quantity_current"] + [
    f"quantity_lag_{i}"
    for i in range(1, 12)
]

panel["quantity_mean_3"] = panel[recent_3].mean(axis=1)
panel["quantity_mean_6"] = panel[recent_6].mean(axis=1)
panel["quantity_mean_12"] = panel[recent_12].mean(axis=1)

panel["quantity_std_3"] = panel[recent_3].std(axis=1)
panel["quantity_std_6"] = panel[recent_6].std(axis=1)
panel["quantity_std_12"] = panel[recent_12].std(axis=1)


# FEATURES D'INTERMITTENCE


panel["demand_positive_current"] = (
    panel["quantity_current"] > 0
).astype(float)

for name, cols in [
    ("3", recent_3),
    ("6", recent_6),
    ("12", recent_12)
]:
    panel[f"positive_months_{name}"] = (
        (panel[cols] > 0).sum(axis=1)
    )

    panel[f"zero_ratio_{name}"] = (
        (panel[cols] == 0).mean(axis=1)
    )

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


# DERNIERES QUANTITES POSTIIVES


panel["_positive_quantity"] = (
    panel["quantity"].where(
        panel["quantity"] > 0
    )
)

panel["last_positive_quantity"] = (
    panel
    .groupby(
        SERIES_COLS,
        observed=True
    )["_positive_quantity"]
    .ffill()
)

panel.drop(
    columns=["_positive_quantity"],
    inplace=True
)


# FEATURES DE PRIX


panel["unit_price_current"] = np.where(
    panel["quantity"] > 0,
    panel["gross_spend"] / panel["quantity"],
    np.nan
)

grouped = panel.groupby(
    SERIES_COLS,
    observed=True
)

for lag in [1, 2, 3, 6, 12]:
    panel[f"unit_price_lag_{lag}"] = (
        grouped["unit_price_current"].shift(lag)
    )

panel["unit_price_mean_3"] = (
    panel[
        [
            "unit_price_current",
            "unit_price_lag_1",
            "unit_price_lag_2"
        ]
    ]
    .mean(axis=1)
)

panel["unit_price_change_1m"] = (
    panel["unit_price_current"]
    / panel["unit_price_lag_1"]
    - 1
)

panel["unit_price_change_3m"] = (
    panel["unit_price_current"]
    / panel["unit_price_lag_3"]
    - 1
)

panel["unit_price_change_1m"] = (
    panel["unit_price_change_1m"].clip(-0.8, 3)
)

panel["unit_price_change_3m"] = (
    panel["unit_price_change_3m"].clip(-0.8, 3)
)


# FEATURES CPI


grouped = panel.groupby(
    SERIES_COLS,
    observed=True
)

for col in [
    "cpi_index",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "cpi_coefficient"
]:
    for lag in [1, 3, 6]:
        panel[f"{col}_lag_{lag}"] = (
            grouped[col].shift(lag)
        )

panel["cpi_index_change_3m"] = (
    panel["cpi_index"]
    / panel["cpi_index_lag_3"]
    - 1
)

panel["cpi_index_change_6m"] = (
    panel["cpi_index"]
    / panel["cpi_index_lag_6"]
    - 1
)


# CONSTRUCTION HORIZONS


grouped = panel.groupby(
    SERIES_COLS,
    observed=True
)

for h in FORECAST_HORIZONS:
    panel[f"target_h{h}"] = (
        grouped["quantity"].shift(-h)
    )

print("Target availability:")
for h in FORECAST_HORIZONS:
    print(
        f"H{h}: "
        f"{panel[f'target_h{h}'].notna().sum():,}"
    )




# FEATURES CALANDAIRES FUTURES HORIZONS H1-H6




for h in FORECAST_HORIZONS:

    future_calendar = calendar_region.copy()

    future_calendar["month_date"] = (
        future_calendar["month_date"]
        - pd.DateOffset(months=h)
    )

    rename_future = {
        col: f"target_{col}_h{h}"
        for col in calendar_region.columns
        if col not in [
            "month_date",
            "school_holiday_region"
        ]
    }

    future_calendar = future_calendar.rename(
        columns=rename_future
    )

    panel = panel.merge(
        future_calendar,
        on=[
            "month_date",
            "school_holiday_region"
        ],
        how="left"
    )


# SAISONNALITE MOIS CIBLE


for h in FORECAST_HORIZONS:

    target_date = (
        panel["month_date"]
        + pd.DateOffset(months=h)
    )

    panel[f"target_month_h{h}"] = target_date.dt.month
    panel[f"target_quarter_h{h}"] = target_date.dt.quarter

    panel[f"target_month_sin_h{h}"] = np.sin(
        2 * np.pi * target_date.dt.month / 12
    )

    panel[f"target_month_cos_h{h}"] = np.cos(
        2 * np.pi * target_date.dt.month / 12
    )


# FEATURES CALANDAIRES CHANGEANTES


for h in FORECAST_HORIZONS:

    panel[f"working_days_change_h{h}"] = (
        panel[f"target_working_days_h{h}"]
        - panel["current_working_days"]
    )

    panel[f"school_holiday_change_h{h}"] = (
        panel[f"target_school_holiday_ratio_selected_h{h}"]
        - panel["current_school_holiday_ratio_selected"]
    )

    panel[f"public_holiday_change_h{h}"] = (
        panel[f"target_public_holidays_weekdays_h{h}"]
        - panel["current_public_holidays_weekdays"]
    )


# INDEX TEMPS


first_date = panel["month_date"].min()

panel["time_index"] = (
    (
        panel["month_date"].dt.year
        - first_date.year
    ) * 12
    +
    (
        panel["month_date"].dt.month
        - first_date.month
    )
)


# FEATURES COMPLETES


numeric_common = [


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


    "time_index"
]

categorical_features = [
    "site_id",
    "supplier_id",
    "category_id",
    "region",
    "school_holiday_region",
    "category_level_1_id",
    "category_level_2_id",
    "category_level_3_id",
    "cpi_id",
    "municipality_confidence"
]


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
        f"target_month_cos_h{horizon}"
    ]

    features = (
        numeric_common
        + future_features
        + categorical_features
    )

    features = [
        col
        for col in dict.fromkeys(features)
        if col in panel.columns
    ]

    return features


print("Example number of features for H1:", len(get_features(1)))



# PREPARATION LIGHTGBM


panel_lgb = panel.copy()

for col in categorical_features:
    if col in panel_lgb.columns:
        panel_lgb[col] = (
            panel_lgb[col]
            .astype("category")
        )

print("panel_lgb ready:", panel_lgb.shape)


# CONGES NATIONALES BELGES — JUL-DEC 2026


BELGIUM_PUBLIC_HOLIDAYS_2026 = pd.to_datetime([
    "2026-07-21",   # FETE NATIONAL
    "2026-08-15",   # ASSEMPTION
    "2026-11-01",   # TOUSSAINT
    "2026-11-11",   # Armistice
    "2026-12-25"    # NOEL
])


def weekdays_between(start, end):
    """Number of Monday-Friday days inclusive."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    dates = pd.date_range(
        start=start,
        end=end,
        freq="D"
    )

    return (dates.weekday < 5).sum()


def weekdays_in_month(month):
    month = pd.Timestamp(month)
    start = month
    end = month + pd.offsets.MonthEnd(0)
    return weekdays_between(start, end)


# PERIODES SCOLAIRES — JUL-DEC 2026


FLANDERS_HOLIDAYS = [
    (
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-08-31")
    ),
    (
        pd.Timestamp("2026-11-02"),
        pd.Timestamp("2026-11-08")
    ),
    (
        pd.Timestamp("2026-12-21"),
        pd.Timestamp("2026-12-31")
    )
]

FWB_HOLIDAYS = [
    (
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-08-23")
    ),
    (
        pd.Timestamp("2026-10-19"),
        pd.Timestamp("2026-10-30")
    ),
    (
        pd.Timestamp("2026-12-21"),
        pd.Timestamp("2026-12-31")
    )
]


def holiday_weekdays_in_month(month, holiday_periods):

    month = pd.Timestamp(month)
    month_start = month
    month_end = month + pd.offsets.MonthEnd(0)

    total = 0

    for start, end in holiday_periods:

        overlap_start = max(
            month_start,
            start
        )

        overlap_end = min(
            month_end,
            end
        )

        if overlap_start <= overlap_end:
            total += weekdays_between(
                overlap_start,
                overlap_end
            )

    return total


# CALENDRIER FUTURE — JUL TO DEC 2026


future_calendar_month = []

for month in FORECAST_MONTHS:

    month = pd.Timestamp(month)
    month_end = month + pd.offsets.MonthEnd(0)

    all_days = pd.date_range(
        month,
        month_end,
        freq="D"
    )

    weekdays = all_days.weekday < 5

    holidays_month = (
        BELGIUM_PUBLIC_HOLIDAYS_2026[
            (
                BELGIUM_PUBLIC_HOLIDAYS_2026 >= month
            )
            &
            (
                BELGIUM_PUBLIC_HOLIDAYS_2026 <= month_end
            )
        ]
    )

    public_holidays = len(holidays_month)

    public_holidays_weekdays = sum(
        holiday.weekday() < 5
        for holiday in holidays_month
    )

    weekdays_mon_fri = weekdays.sum()

    working_days = (
        weekdays_mon_fri
        - public_holidays_weekdays
    )

    flanders_days = holiday_weekdays_in_month(
        month,
        FLANDERS_HOLIDAYS
    )

    fwb_days = holiday_weekdays_in_month(
        month,
        FWB_HOLIDAYS
    )

    flanders_ratio = (
        flanders_days / weekdays_mon_fri
        if weekdays_mon_fri > 0
        else 0
    )

    fwb_ratio = (
        fwb_days / weekdays_mon_fri
        if weekdays_mon_fri > 0
        else 0
    )

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
        "school_holiday_ratio_fwb": fwb_ratio
    })

future_calendar_month = pd.DataFrame(
    future_calendar_month
)

print(future_calendar_month.to_string(index=False))


# CALENDRIER FUTUR PAR REGION


future_calendar_region = []

for _, row in future_calendar_month.iterrows():

    base = row.to_dict()

    # FLANDRE
    flanders = base.copy()
    flanders["school_holiday_region"] = "FLANDERS"
    flanders["school_holiday_weekdays_selected"] = (
        flanders["school_holiday_weekdays_flanders"]
    )
    flanders["school_holiday_ratio_selected"] = (
        flanders["school_holiday_ratio_flanders"]
    )
    future_calendar_region.append(flanders)

    # FWB
    fwb = base.copy()
    fwb["school_holiday_region"] = "FWB"
    fwb["school_holiday_weekdays_selected"] = (
        fwb["school_holiday_weekdays_fwb"]
    )
    fwb["school_holiday_ratio_selected"] = (
        fwb["school_holiday_ratio_fwb"]
    )
    future_calendar_region.append(fwb)

    # BRUSSELS — MOYENNE FWB FLANDRE
    brussels = base.copy()
    brussels["school_holiday_region"] = "BRUSSELS_BOTH"
    brussels["school_holiday_weekdays_selected"] = (
        (
            brussels["school_holiday_weekdays_flanders"]
            + brussels["school_holiday_weekdays_fwb"]
        ) / 2
    )
    brussels["school_holiday_ratio_selected"] = (
        (
            brussels["school_holiday_ratio_flanders"]
            + brussels["school_holiday_ratio_fwb"]
        ) / 2
    )
    future_calendar_region.append(brussels)

future_calendar_region = pd.DataFrame(
    future_calendar_region
)

future_calendar_region["school_holiday_region"] = (
    future_calendar_region[
        "school_holiday_region"
    ].astype("string")
)

print(
    future_calendar_region[
        [
            "month_date",
            "school_holiday_region",
            "working_days",
            "public_holidays_weekdays",
            "school_holiday_weekdays_selected",
            "school_holiday_ratio_selected"
        ]
    ].to_string(index=False)
)


# DONNEES FORECAST ORIGINE


forecast_origin_df = (
    panel_lgb[
        panel_lgb["month_date"] == FINAL_ORIGIN
    ]
    .copy()
)

forecast_origin_df = (
    forecast_origin_df[
        forecast_origin_df[
            "quantity_current"
        ].notna()
    ]
    .copy()
)

if forecast_origin_df.empty:
    raise ValueError(
        f"No forecast-origin rows found for {FINAL_ORIGIN.date()}. "
        "Check the dataset date range and FINAL_ORIGIN."
    )

print("Series forecasted:", f"{len(forecast_origin_df):,}")



# FEATURES HORIZONS H1_H6


for h in FORECAST_HORIZONS:

    target_month = (
        FINAL_ORIGIN
        + pd.DateOffset(months=h)
    )

    future_h = (
        future_calendar_region[
            future_calendar_region[
                "month_date"
            ] == target_month
        ]
        .copy()
    )

    rename_cols = {
        col: f"target_{col}_h{h}"
        for col in future_h.columns
        if col not in [
            "month_date",
            "school_holiday_region"
        ]
    }

    future_h = future_h.rename(
        columns=rename_cols
    )

    future_h = future_h.drop(
        columns="month_date"
    )

    forecast_origin_df = (
        forecast_origin_df
        .merge(
            future_h,
            on="school_holiday_region",
            how="left",
            suffixes=("", "_future")
        )
    )


    for original_col in rename_cols.values():

        future_version = (
            original_col
            + "_future"
        )

        if future_version in forecast_origin_df.columns:

            forecast_origin_df[
                original_col
            ] = (
                forecast_origin_df[
                    future_version
                ]
            )

            forecast_origin_df.drop(
                columns=future_version,
                inplace=True
            )


    forecast_origin_df[
        f"target_month_h{h}"
    ] = target_month.month

    forecast_origin_df[
        f"target_quarter_h{h}"
    ] = target_month.quarter

    forecast_origin_df[
        f"target_month_sin_h{h}"
    ] = np.sin(
        2
        * np.pi
        * target_month.month
        / 12
    )

    forecast_origin_df[
        f"target_month_cos_h{h}"
    ] = np.cos(
        2
        * np.pi
        * target_month.month
        / 12
    )


    forecast_origin_df[
        f"working_days_change_h{h}"
    ] = (
        forecast_origin_df[
            f"target_working_days_h{h}"
        ]
        - forecast_origin_df[
            "current_working_days"
        ]
    )

    forecast_origin_df[
        f"school_holiday_change_h{h}"
    ] = (
        forecast_origin_df[
            f"target_school_holiday_ratio_selected_h{h}"
        ]
        - forecast_origin_df[
            "current_school_holiday_ratio_selected"
        ]
    )

    forecast_origin_df[
        f"public_holiday_change_h{h}"
    ] = (
        forecast_origin_df[
            f"target_public_holidays_weekdays_h{h}"
        ]
        - forecast_origin_df[
            "current_public_holidays_weekdays"
        ]
    )



# VARIABLES CATEGORIELLES LIGHTGBM


for col in categorical_features:

    if col not in forecast_origin_df.columns:
        continue

    if col in panel_lgb.columns:

        training_categories = (
            panel_lgb[col].cat.categories
        )

        forecast_origin_df[col] = pd.Categorical(
            forecast_origin_df[col],
            categories=training_categories
        )




def align_lgb_categories(X_train, X_predict, categorical_cols):

    X_train = X_train.copy()
    X_predict = X_predict.copy()

    for col in categorical_cols:

        if col not in X_train.columns or col not in X_predict.columns:
            continue

        if isinstance(X_train[col].dtype, pd.CategoricalDtype):
            categories = X_train[col].cat.categories
        else:

            train_as_string = X_train[col].astype("string")
            categories = pd.Index(
                train_as_string.dropna().unique()
            )
            X_train[col] = pd.Categorical(
                train_as_string,
                categories=categories
            )

     
        X_predict[col] = pd.Categorical(
            X_predict[col].astype("string"),
            categories=categories
        )

        X_train[col] = pd.Categorical(
            X_train[col].astype("string"),
            categories=categories
        )

    return X_train, X_predict


print("LightGBM category-alignment helper ready.")



# MODELE FINAL LIGHTGBM — LIGHTGBM RESIDUAL H1-H6


final_site_models = {}
site_forecast_blocks = []

for h in FORECAST_HORIZONS:

    print("\n" + "=" * 80)
    print(f"TRAINING FINAL SITE LIGHTGBM RESIDUAL — H{h}")
    print("=" * 80)

    target = f"target_h{h}"
    features = get_features(h)

    categorical_now = [
        col
        for col in categorical_features
        if col in features
    ]


    train = panel_lgb[
        panel_lgb[target].notna()
        & panel_lgb["quantity_current"].notna()
    ].copy()

    y_train = (
        train[target]
        - train["quantity_current"]
    )


    X_train_lgb = train[features].copy()
    X_forecast_lgb = forecast_origin_df[features].copy()

    X_train_lgb, X_forecast_lgb = align_lgb_categories(
        X_train_lgb,
        X_forecast_lgb,
        categorical_now
    )


    for col in categorical_now:
        assert isinstance(
            X_train_lgb[col].dtype,
            pd.CategoricalDtype
        ), f"{col}: train is not categorical"

        assert isinstance(
            X_forecast_lgb[col].dtype,
            pd.CategoricalDtype
        ), f"{col}: forecast is not categorical"

        assert list(
            X_train_lgb[col].cat.categories
        ) == list(
            X_forecast_lgb[col].cat.categories
        ), f"{col}: category dictionaries still differ"

    model = LGBMRegressor(
        **LGB_SITE_PARAMS
    )

    model.fit(
        X_train_lgb,
        y_train,
        categorical_feature=categorical_now
    )

    final_site_models[h] = model

    print("Training rows:", f"{len(train):,}")
    print("Categorical features:", len(categorical_now))

    pred_change = model.predict(
        X_forecast_lgb
    )

    prediction = (
        forecast_origin_df[
            "quantity_current"
        ].values
        + pred_change
    )

    prediction = np.maximum(
        prediction,
        0
    )

    result = (
        forecast_origin_df[
            SERIES_COLS
            + [
                "region",
                "school_holiday_region",
                "quantity_current"
            ]
        ]
        .copy()
    )

    result["forecast_origin"] = FINAL_ORIGIN
    result["horizon"] = h
    result["forecast_month"] = (
        FINAL_ORIGIN
        + pd.DateOffset(months=h)
    )
    result["model"] = "LightGBM Residual"
    result["forecast_quantity"] = prediction

    site_forecast_blocks.append(result)


site_forecast = pd.concat(
    site_forecast_blocks,
    ignore_index=True
)

site_forecast = (
    site_forecast
    .sort_values(
        [
            "site_id",
            "supplier_id",
            "category_id",
            "forecast_month"
        ]
    )
    .reset_index(drop=True)
)

site_forecast["forecast_grain"] = (
    "site_supplier_category"
)

print("Site forecast shape:", site_forecast.shape)
display(site_forecast.head(20))


# PREPARATION RANDOM FOREST


def make_rf_pipeline(features):

    categorical_now = [
        col
        for col in categorical_features
        if col in features
    ]

    numeric_now = [
        col
        for col in features
        if col not in categorical_now
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ])

    preprocessing = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_now
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_now
        )
    ])

    rf = RandomForestRegressor(
        **RF_BOTTOMUP_PARAMS
    )

    return Pipeline([
        (
            "preprocessing",
            preprocessing
        ),
        (
            "model",
            rf
        )
    ])



# FORECAST BOTTOM-UP RANDOM FOREST H1-H6


final_rf_models = {}
rf_site_blocks = []


panel_rf = panel.copy()
forecast_origin_rf = forecast_origin_df.copy()

for col in categorical_features:
    if col in panel_rf.columns:
        panel_rf[col] = panel_rf[col].astype("string")
    if col in forecast_origin_rf.columns:
        forecast_origin_rf[col] = (
            forecast_origin_rf[col]
            .astype("string")
        )

for h in FORECAST_HORIZONS:

    print("\n" + "=" * 80)
    print(f"TRAINING BOTTOM-UP RANDOM FOREST — H{h}")
    print("=" * 80)

    target = f"target_h{h}"
    features = get_features(h)

    train = panel_rf[
        panel_rf[target].notna()
        & panel_rf["quantity_current"].notna()
    ].copy()

    model = make_rf_pipeline(
        features
    )

    model.fit(
        train[features],
        train[target]
    )

    final_rf_models[h] = model

    print("Training rows:", f"{len(train):,}")

    prediction = model.predict(
        forecast_origin_rf[features]
    )

    prediction = np.maximum(
        prediction,
        0
    )

    result = (
        forecast_origin_rf[
            SERIES_COLS
            + ["quantity_current"]
        ]
        .copy()
    )

    result["forecast_origin"] = FINAL_ORIGIN
    result["horizon"] = h
    result["forecast_month"] = (
        FINAL_ORIGIN
        + pd.DateOffset(months=h)
    )
    result["model"] = "Random Forest (internal bottom-up)"
    result["forecast_quantity"] = prediction

    rf_site_blocks.append(result)


rf_bottom_up_site_forecast = pd.concat(
    rf_site_blocks,
    ignore_index=True
)

print(
    "Internal RF site forecast shape:",
    rf_bottom_up_site_forecast.shape
)

display(
    rf_bottom_up_site_forecast.head(20)
)


# AGGREGATION RESULTATS RANDOM FOREST — BOTTOM-UP


category_forecast = (
    rf_bottom_up_site_forecast
    .groupby(
        [
            "supplier_id",
            "category_id",
            "forecast_origin",
            "horizon",
            "forecast_month"
        ],
        as_index=False,
        observed=True
    )
    .agg(
        quantity_current=(
            "quantity_current",
            "sum"
        ),
        forecast_quantity=(
            "forecast_quantity",
            "sum"
        )
    )
)

category_forecast["model"] = (
    "Random Forest Bottom-Up"
)

category_forecast["forecast_grain"] = (
    "supplier_category"
)

category_forecast = (
    category_forecast
    .sort_values(
        [
            "supplier_id",
            "category_id",
            "forecast_month"
        ]
    )
    .reset_index(drop=True)
)

print(
    "Supplier × Category forecast shape:",
    category_forecast.shape
)

display(
    category_forecast.head(20)
)



# SAUVEGARDE FORECAST FINAL


SITE_OUTPUT = (
    r"C:\Users\arnau\UNAMUR\projet perso"
    r"\forecast_site_supplier_category_H1_H6.csv"
)

CATEGORY_OUTPUT = (
    r"C:\Users\arnau\UNAMUR\projet perso"
    r"\forecast_supplier_category_RF_bottomup_H1_H6.csv"
)

site_forecast.to_csv(
    SITE_OUTPUT,
    index=False
)

category_forecast.to_csv(
    CATEGORY_OUTPUT,
    index=False
)

print("Saved:")
print(SITE_OUTPUT)
print(CATEGORY_OUTPUT)



# ECHANTILLON FORECAST



example_key = (
    category_forecast[
        ["supplier_id", "category_id"]
    ]
    .drop_duplicates()
    .iloc[0]
)

example_supplier = (
    example_key["supplier_id"]
)

example_category = (
    example_key["category_id"]
)


example_category_forecast = (
    category_forecast[
        (
            category_forecast["supplier_id"]
            == example_supplier
        )
        &
        (
            category_forecast["category_id"]
            == example_category
        )
    ]
    [
        [
            "forecast_month",
            "horizon",
            "quantity_current",
            "forecast_quantity",
            "model"
        ]
    ]
    .copy()
)


example_site_lgb = (
    site_forecast[
        (
            site_forecast["supplier_id"]
            == example_supplier
        )
        &
        (
            site_forecast["category_id"]
            == example_category
        )
    ]
    .copy()
)

lgb_site_sum = (
    example_site_lgb
    .groupby(
        [
            "forecast_month",
            "horizon"
        ],
        as_index=False
    )["forecast_quantity"]
    .sum()
    .rename(
        columns={
            "forecast_quantity":
                "LGBM_site_sum"
        }
    )
)

illustration = (
    example_category_forecast
    .rename(
        columns={
            "forecast_quantity":
                "RF_bottom_up_category"
        }
    )
    .merge(
        lgb_site_sum,
        on=[
            "forecast_month",
            "horizon"
        ],
        how="left"
    )
)

illustration["difference_between_published_levels"] = (
    illustration["RF_bottom_up_category"]
    - illustration["LGBM_site_sum"]
)

print(
    "Illustrative Supplier × Category:",
    example_supplier,
    "×",
    example_category
)

display(illustration)

print(
    "\nFirst site-level LightGBM Residual forecasts "
    "for the same Supplier × Category:"
)

display(
    example_site_lgb[
        [
            "site_id",
            "forecast_month",
            "horizon",
            "quantity_current",
            "forecast_quantity",
            "model"
        ]
    ]
    .head(20)
)


# RESUME FORECAST GLOBAL


summary_site = (
    site_forecast
    .groupby(
        [
            "horizon",
            "forecast_month"
        ],
        as_index=False
    )
    .agg(
        number_of_site_series=(
            "forecast_quantity",
            "size"
        ),
        LGBM_site_total=(
            "forecast_quantity",
            "sum"
        )
    )
)

summary_category = (
    category_forecast
    .groupby(
        [
            "horizon",
            "forecast_month"
        ],
        as_index=False
    )
    .agg(
        number_of_supplier_categories=(
            "forecast_quantity",
            "size"
        ),
        RF_bottom_up_total=(
            "forecast_quantity",
            "sum"
        )
    )
)

forecast_summary = summary_site.merge(
    summary_category,
    on=[
        "horizon",
        "forecast_month"
    ],
    how="outer"
)

display(forecast_summary)


