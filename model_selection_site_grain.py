# ============================================================
# 1. IMPORTS & GLOBAL CONFIGURATION
# ============================================================

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# LightGBM
import lightgbm as lgb
from lightgbm import LGBMRegressor


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

RANDOM_STATE = 42

FORECAST_HORIZONS = [1, 2, 3, 4, 5, 6]

SERIES_COLS = [
    "site_id",
    "supplier_id",
    "category_id"
]

TARGET = "quantity"

DATE_COL = "month_date"


# ------------------------------------------------------------
# FAST MODE
#
# True = recommended first execution
# False = heavier final test
# ------------------------------------------------------------

FAST_MODE = True


if FAST_MODE:

    RF_N_ESTIMATORS = 80
    RF_MAX_DEPTH = 10
    RF_MIN_SAMPLES_LEAF = 10

    LGB_N_ESTIMATORS = 250
    LGB_LEARNING_RATE = 0.03

else:

    RF_N_ESTIMATORS = 250
    RF_MAX_DEPTH = 14
    RF_MIN_SAMPLES_LEAF = 8

    LGB_N_ESTIMATORS = 500
    LGB_LEARNING_RATE = 0.03


print("Configuration loaded.")
print("Forecast horizons:", FORECAST_HORIZONS)
print("Fast mode:", FAST_MODE)



# ============================================================
# 2. LOAD SOURCE DATA
# ============================================================

filepath = r"C:\Users\arnau\UNAMUR\projet perso\ml_demand_forecast_base.csv"

df = pd.read_csv(
    filepath,
    low_memory=False
)

print(
    f"Dataset loaded: "
    f"{df.shape[0]:,} rows × "
    f"{df.shape[1]:,} columns"
)

df.head()




 ============================================================
# 3. DATA TYPES
# ============================================================

df["month_date"] = pd.to_datetime(
    df["month_date"],
    errors="coerce"
)


numeric_cols = [

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


for col in numeric_cols:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


categorical_cols = [

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


for col in categorical_cols:

    if col in df.columns:

        df[col] = (
            df[col]
            .astype("string")
        )


print(
    "Date range:",
    df["month_date"].min(),
    "→",
    df["month_date"].max()
)




# ============================================================
# 4. FORECAST GRAIN CHECK
# ============================================================

grain = [
    "month_date",
    "site_id",
    "supplier_id",
    "category_id"
]


duplicate_count = (
    df.duplicated(
        grain,
        keep=False
    )
    .sum()
)


print(
    "Duplicate rows at forecasting grain:",
    f"{duplicate_count:,}"
)

print(
    "Unique site × supplier × category series:",
    f"{df[SERIES_COLS].drop_duplicates().shape[0]:,}"
)



# ============================================================
# 5. REGIONAL MONTHLY CALENDAR
# ============================================================

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
        [
            "month_date",
            "school_holiday_region"
        ],
        as_index=False,
        observed=True
    )

    .median(
        numeric_only=True
    )
)


print(calendar_region.head())



# ============================================================
# 6. SERIES METADATA
# ============================================================

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

    df[
        SERIES_COLS
        + metadata_cols
    ]

    .groupby(
        SERIES_COLS,
        as_index=False,
        observed=True
    )

    .first()
)


print(
    "Series metadata rows:",
    f"{len(series_metadata):,}"
)


# ============================================================
# 7. CPI LOOKUP
# month × CPI category
# ============================================================

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
        [
            "month_date",
            "cpi_id"
        ],
        as_index=False,
        observed=True
    )

    .median(
        numeric_only=True
    )
)


print(cpi_lookup.head())



# ============================================================
# 8. COMPLETE MONTHLY PANEL
# ============================================================

all_months = pd.DataFrame({

    "month_date":
        pd.date_range(
            start=df["month_date"].min(),
            end=df["month_date"].max(),
            freq="MS"
        )
})


series_keys = (
    df[SERIES_COLS]
    .drop_duplicates()
)


panel = series_keys.merge(
    all_months,
    how="cross"
)


# ------------------------------------------------------------
# Actual demand / spend
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Static metadata
# ------------------------------------------------------------

panel = panel.merge(

    series_metadata,

    on=SERIES_COLS,

    how="left"
)


# ------------------------------------------------------------
# CPI
# ------------------------------------------------------------

panel = panel.merge(

    cpi_lookup,

    on=[
        "month_date",
        "cpi_id"
    ],

    how="left"
)


panel = (

    panel

    .sort_values(
        SERIES_COLS
        + ["month_date"]
    )

    .reset_index(drop=True)
)


print(
    f"Panel size: "
    f"{panel.shape[0]:,} rows × "
    f"{panel.shape[1]:,} columns"
)


# ============================================================
# 9. CURRENT CALENDAR FEATURES
# ============================================================

current_calendar = (
    calendar_region
    .copy()
)


rename_current = {

    col:
        f"current_{col}"

    for col in calendar_region.columns

    if col not in [
        "month_date",
        "school_holiday_region"
    ]
}


current_calendar = current_calendar.rename(
    columns=rename_current
)


panel = panel.merge(

    current_calendar,

    on=[
        "month_date",
        "school_holiday_region"
    ],

    how="left"
)



# ============================================================
# 10. QUANTITY HISTORY
# ============================================================

panel["quantity_current"] = (
    panel["quantity"]
)


grouped = panel.groupby(
    SERIES_COLS,
    observed=True
)


for lag in range(1, 13):

    panel[
        f"quantity_lag_{lag}"
    ] = (
        grouped["quantity"]
        .shift(lag)
    )


    # ============================================================
# 11. ROLLING DEMAND FEATURES
# ============================================================

recent_3 = [
    "quantity_current",
    "quantity_lag_1",
    "quantity_lag_2"
]


recent_6 = [
    "quantity_current"
] + [
    f"quantity_lag_{i}"
    for i in range(1, 6)
]


recent_12 = [
    "quantity_current"
] + [
    f"quantity_lag_{i}"
    for i in range(1, 12)
]


panel["quantity_mean_3"] = (
    panel[recent_3]
    .mean(axis=1)
)


panel["quantity_mean_6"] = (
    panel[recent_6]
    .mean(axis=1)
)


panel["quantity_mean_12"] = (
    panel[recent_12]
    .mean(axis=1)
)


panel["quantity_std_6"] = (
    panel[recent_6]
    .std(axis=1)
)


panel["quantity_std_12"] = (
    panel[recent_12]
    .std(axis=1)
)



# ============================================================
# 12. INTERMITTENT DEMAND FEATURES
# ============================================================

panel["positive_months_3"] = (
    (panel[recent_3] > 0)
    .sum(axis=1)
)


panel["positive_months_6"] = (
    (panel[recent_6] > 0)
    .sum(axis=1)
)


panel["positive_months_12"] = (
    (panel[recent_12] > 0)
    .sum(axis=1)
)


panel["zero_ratio_3"] = (
    (panel[recent_3] == 0)
    .mean(axis=1)
)


panel["zero_ratio_6"] = (
    (panel[recent_6] == 0)
    .mean(axis=1)
)


panel["zero_ratio_12"] = (
    (panel[recent_12] == 0)
    .mean(axis=1)
)


panel["positive_quantity_mean_6"] = (

    panel[recent_6]

    .where(
        panel[recent_6] > 0
    )

    .mean(axis=1)
)


panel["positive_quantity_mean_12"] = (

    panel[recent_12]

    .where(
        panel[recent_12] > 0
    )

    .mean(axis=1)
)



# ============================================================
# 13. UNIT PRICE FEATURES
# ============================================================

panel["unit_price_current"] = np.where(

    panel["quantity"] > 0,

    panel["gross_spend"]
    / panel["quantity"],

    np.nan
)


grouped = panel.groupby(
    SERIES_COLS,
    observed=True
)


for lag in [1, 2, 3, 6, 12]:

    panel[
        f"unit_price_lag_{lag}"
    ] = (

        grouped[
            "unit_price_current"
        ]
        .shift(lag)
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


# Avoid extreme ratios
panel["unit_price_change_1m"] = (
    panel["unit_price_change_1m"]
    .clip(-0.8, 3)
)


panel["unit_price_change_3m"] = (
    panel["unit_price_change_3m"]
    .clip(-0.8, 3)
)


# ============================================================
# 14. CPI HISTORY — NO FUTURE LEAKAGE
# ============================================================

grouped = panel.groupby(
    SERIES_COLS,
    observed=True
)


cpi_cols = [

    "cpi_index",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "cpi_coefficient"
]


for col in cpi_cols:

    for lag in [1, 3, 6]:

        panel[
            f"{col}_lag_{lag}"
        ] = (

            grouped[col]
            .shift(lag)
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


# ============================================================
# 15. TIME TREND
# ============================================================

first_date = (
    panel["month_date"]
    .min()
)


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

# ============================================================
# 16. H1 → H6 TARGETS
# ============================================================

grouped = panel.groupby(
    SERIES_COLS,
    observed=True
)


for horizon in FORECAST_HORIZONS:

    panel[
        f"target_h{horizon}"
    ] = (

        grouped["quantity"]
        .shift(-horizon)
    )


print("Target counts:")

for h in FORECAST_HORIZONS:

    print(
        f"H{h}: "
        f"{panel[f'target_h{h}'].notna().sum():,}"
    )



    # ============================================================
# 17. FUTURE CALENDAR H1 → H6
# ============================================================

for horizon in FORECAST_HORIZONS:

    future_calendar = (
        calendar_region
        .copy()
    )


    future_calendar["month_date"] = (

        future_calendar["month_date"]

        - pd.DateOffset(
            months=horizon
        )
    )


    rename_future = {

        col:
            f"target_{col}_h{horizon}"

        for col in calendar_region.columns

        if col not in [
            "month_date",
            "school_holiday_region"
        ]
    }


    future_calendar = (
        future_calendar
        .rename(
            columns=rename_future
        )
    )


    panel = panel.merge(

        future_calendar,

        on=[
            "month_date",
            "school_holiday_region"
        ],

        how="left"
    )


    # ============================================================
# 18. CALENDAR CHANGE FEATURES
# ============================================================

for h in FORECAST_HORIZONS:

    panel[
        f"working_days_change_h{h}"
    ] = (

        panel[
            f"target_working_days_h{h}"
        ]

        - panel[
            "current_working_days"
        ]
    )


    panel[
        f"school_holiday_change_h{h}"
    ] = (

        panel[
            f"target_school_holiday_ratio_selected_h{h}"
        ]

        - panel[
            "current_school_holiday_ratio_selected"
        ]
    )


    panel[
        f"public_holiday_change_h{h}"
    ] = (

        panel[
            f"target_public_holidays_weekdays_h{h}"
        ]

        - panel[
            "current_public_holidays_weekdays"
        ]
    )


    # ============================================================
# 19. TARGET MONTH SEASONALITY H1 → H6
# ============================================================

for h in FORECAST_HORIZONS:

    target_date = (

        panel["month_date"]

        + pd.DateOffset(
            months=h
        )
    )


    panel[
        f"target_month_h{h}"
    ] = (
        target_date.dt.month
    )


    panel[
        f"target_quarter_h{h}"
    ] = (
        target_date.dt.quarter
    )


    panel[
        f"target_month_sin_h{h}"
    ] = np.sin(

        2
        * np.pi
        * target_date.dt.month
        / 12
    )


    panel[
        f"target_month_cos_h{h}"
    ] = np.cos(

        2
        * np.pi
        * target_date.dt.month
        / 12
    )



    # ============================================================
# 20. COMMON FEATURE SET
# ============================================================

numeric_common = [

    # Current demand
    "quantity_current",

    # Quantity lags
    "quantity_lag_1",
    "quantity_lag_2",
    "quantity_lag_3",
    "quantity_lag_4",
    "quantity_lag_5",
    "quantity_lag_6",

    "quantity_lag_9",
    "quantity_lag_12",

    # Rolling demand
    "quantity_mean_3",
    "quantity_mean_6",
    "quantity_mean_12",

    "quantity_std_6",
    "quantity_std_12",

    # Intermittence
    "positive_months_3",
    "positive_months_6",
    "positive_months_12",

    "zero_ratio_3",
    "zero_ratio_6",
    "zero_ratio_12",

    "positive_quantity_mean_6",
    "positive_quantity_mean_12",

    "last_positive_quantity",

    # Price
    "unit_price_current",
    "unit_price_lag_1",
    "unit_price_lag_2",
    "unit_price_lag_3",
    "unit_price_lag_6",
    "unit_price_lag_12",

    "unit_price_change_1m",
    "unit_price_change_3m",

    # CPI
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

    # Current calendar
    "current_working_days",
    "current_public_holidays_weekdays",
    "current_school_holiday_ratio_selected",

    # Trend
    "time_index"
]




odel_categorical_features = [

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



# ============================================================
# 21. FEATURES PER HORIZON
# ============================================================

def get_features(horizon):

    future_features = [

        # Target-month calendar
        f"target_days_in_month_h{horizon}",
        f"target_working_days_h{horizon}",
        f"target_weekdays_mon_fri_h{horizon}",

        # Public holidays
        f"target_public_holidays_h{horizon}",
        f"target_public_holidays_weekdays_h{horizon}",

        # School holidays
        f"target_school_holiday_ratio_selected_h{horizon}",
        f"target_school_holiday_weekdays_selected_h{horizon}",

        f"target_school_holiday_ratio_flanders_h{horizon}",
        f"target_school_holiday_ratio_fwb_h{horizon}",

        # Calendar changes
        f"working_days_change_h{horizon}",
        f"school_holiday_change_h{horizon}",
        f"public_holiday_change_h{horizon}",

        # Target month
        f"target_month_h{horizon}",
        f"target_quarter_h{horizon}",
        f"target_month_sin_h{horizon}",
        f"target_month_cos_h{horizon}"
    ]


    features = (

        numeric_common
        + future_features
        + model_categorical_features
    )


    # Keep only existing columns
    features = [

        col
        for col in features

        if col in panel.columns
    ]


    return list(
        dict.fromkeys(features)
    )


# ============================================================
# 22. LIGHTGBM CATEGORICAL PREPARATION
# ============================================================

panel_lgb = (
    panel.copy()
)


for col in model_categorical_features:

    if col in panel_lgb.columns:

        panel_lgb[col] = (
            panel_lgb[col]
            .astype("category")
        )


# ============================================================
# 23. BACKTEST ORIGINS
# ============================================================

BACKTEST_ORIGINS = pd.to_datetime([

    "2025-04-01",
    "2025-08-01",
    "2025-12-01"

])


max_horizon = max(
    FORECAST_HORIZONS
)


print(
    "Dataset maximum date:",
    panel["month_date"].max()
)


for origin in BACKTEST_ORIGINS:

    latest_target = (
        origin
        + pd.DateOffset(
            months=max_horizon
        )
    )

    print(
        origin.date(),
        "→ H6 target:",
        latest_target.date()
    )



# ============================================================
# 24. METRICS
# ============================================================

def calculate_metrics(
    actual,
    prediction
):

    actual = np.asarray(actual)
    prediction = np.asarray(prediction)

    valid = (
        np.isfinite(actual)
        & np.isfinite(prediction)
    )

    actual = actual[valid]
    prediction = prediction[valid]


    mae = np.mean(
        np.abs(
            actual - prediction
        )
    )


    denominator = (
        np.abs(actual).sum()
    )


    wape = (
        np.abs(
            actual - prediction
        ).sum()
        / denominator
        if denominator > 0
        else np.nan
    )


    bias = (
        (
            prediction - actual
        ).sum()
        / denominator
        if denominator > 0
        else np.nan
    )


    positive = (
        actual > 0
    )


    if positive.sum() > 0:

        positive_mae = np.mean(
            np.abs(
                actual[positive]
                - prediction[positive]
            )
        )

        positive_wape = (
            np.abs(
                actual[positive]
                - prediction[positive]
            ).sum()
            / actual[positive].sum()
        )

    else:

        positive_mae = np.nan
        positive_wape = np.nan


    return {

        "MAE":
            mae,

        "WAPE":
            wape,

        "Bias":
            bias,

        "Positive_MAE":
            positive_mae,

        "Positive_WAPE":
            positive_wape,

        "Positive_rate":
            positive.mean()
    }


# ============================================================
# 25A. LINEAR REGRESSION
# ============================================================

def fit_predict_linear(
    X_train,
    y_train,
    X_test,
    features
):

    categorical = [
        col
        for col in model_categorical_features
        if col in features
    ]


    numeric = [
        col
        for col in features
        if col not in categorical
    ]


    numeric_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
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
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=3
            )
        )
    ])


    preprocessing = ColumnTransformer([

        (
            "numeric",
            numeric_pipeline,
            numeric
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical
        )
    ])


    model = Pipeline([

        (
            "preprocessing",
            preprocessing
        ),

        (
            "model",
            LinearRegression()
        )
    ])


    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    return np.maximum(
        prediction,
        0
    )




# ============================================================
# 25B. RANDOM FOREST — OPTIMIZED
# ============================================================

def fit_predict_rf(
    X_train,
    y_train,
    X_test,
    features
):

    categorical = [
        col
        for col in model_categorical_features
        if col in features
    ]


    numeric = [
        col
        for col in features
        if col not in categorical
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
                handle_unknown=
                    "use_encoded_value",

                unknown_value=-1
            )
        )
    ])


    preprocessing = ColumnTransformer([

        (
            "numeric",
            numeric_pipeline,
            numeric
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical
        )
    ])


    rf = RandomForestRegressor(

        n_estimators=
            RF_N_ESTIMATORS,

        max_depth=
            RF_MAX_DEPTH,

        min_samples_leaf=
            RF_MIN_SAMPLES_LEAF,

        max_features=0.5,

        random_state=
            RANDOM_STATE,

        n_jobs=-1
    )


    model = Pipeline([

        (
            "preprocessing",
            preprocessing
        ),

        (
            "model",
            rf
        )
    ])


    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    return np.maximum(
        prediction,
        0
    )


# ============================================================
# 25C. LIGHTGBM DIRECT
# ============================================================

def fit_predict_lgb_direct(
    train,
    test,
    target,
    features
):

    categorical = [
        col
        for col in model_categorical_features
        if col in features
    ]


    X_train = (
        train[features]
        .copy()
    )

    y_train = (
        train[target]
        .copy()
    )

    X_test = (
        test[features]
        .copy()
    )


    model = LGBMRegressor(

        objective=
            "regression_l1",

        n_estimators=
            LGB_N_ESTIMATORS,

        learning_rate=
            LGB_LEARNING_RATE,

        num_leaves=31,

        min_child_samples=30,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=0.2,

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1
    )


    model.fit(

        X_train,
        y_train,

        categorical_feature=
            categorical
    )


    prediction = model.predict(
        X_test
    )


    return np.maximum(
        prediction,
        0
    )


# ============================================================
# 25D. LIGHTGBM RESIDUAL
# ============================================================

def fit_predict_lgb_residual(
    train,
    test,
    target,
    features
):

    categorical = [
        col
        for col in model_categorical_features
        if col in features
    ]


    X_train = (
        train[features]
        .copy()
    )


    X_test = (
        test[features]
        .copy()
    )


    # -----------------------------------------
    # Target = future - current quantity
    # -----------------------------------------

    y_train = (

        train[target]

        - train[
            "quantity_current"
        ]
    )


    model = LGBMRegressor(

        objective=
            "regression_l1",

        n_estimators=
            LGB_N_ESTIMATORS,

        learning_rate=
            LGB_LEARNING_RATE,

        num_leaves=31,

        min_child_samples=30,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=0.2,

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1
    )


    model.fit(

        X_train,
        y_train,

        categorical_feature=
            categorical
    )


    predicted_change = (
        model.predict(
            X_test
        )
    )


    prediction = (

        test[
            "quantity_current"
        ].values

        + predicted_change
    )


    return np.maximum(
        prediction,
        0
    )



    # ============================================================
# 26. MASTER BACKTEST
# H1 → H6
# ============================================================

prediction_blocks = []


for origin in BACKTEST_ORIGINS:

    print("\n" + "=" * 90)
    print(
        "FORECAST ORIGIN:",
        origin.date()
    )
    print("=" * 90)


    for horizon in FORECAST_HORIZONS:

        print(
            f"\n--- H{horizon} ---"
        )


        target = (
            f"target_h{horizon}"
        )


        features = (
            get_features(
                horizon
            )
        )


        # ====================================================
        # LEAKAGE-SAFE TRAINING CUTOFF
        # ====================================================

        train_end = (

            origin

            - pd.DateOffset(
                months=horizon
            )
        )


        # Use panel_lgb so pandas categories are
        # already prepared for LightGBM

        train = panel_lgb[

            (
                panel_lgb["month_date"]
                <= train_end
            )

            & panel_lgb[target].notna()

            & panel_lgb[
                "quantity_current"
            ].notna()

        ].copy()


        test = panel_lgb[

            (
                panel_lgb["month_date"]
                == origin
            )

            & panel_lgb[target].notna()

            & panel_lgb[
                "quantity_current"
            ].notna()

        ].copy()


        if len(test) == 0:

            print(
                "No test observations."
            )

            continue


        print(
            f"Train={len(train):,} | "
            f"Test={len(test):,}"
        )


        # ====================================================
        # BASELINE
        # ====================================================

        actual = (
            test[target]
            .values
        )


        pred_last = (
            test[
                "quantity_current"
            ]
            .values
        )


        # ====================================================
        # LINEAR REGRESSION
        # ====================================================

        pred_linear = fit_predict_linear(

            train[features],
            train[target],

            test[features],

            features
        )


        # ====================================================
        # RANDOM FOREST
        # ====================================================

        pred_rf = fit_predict_rf(

            train[features],
            train[target],

            test[features],

            features
        )


        # ====================================================
        # LIGHTGBM DIRECT
        # ====================================================

        pred_lgb_direct = (
            fit_predict_lgb_direct(

                train,
                test,

                target,
                features
            )
        )


        # ====================================================
        # LIGHTGBM RESIDUAL
        # ====================================================

        pred_lgb_residual = (
            fit_predict_lgb_residual(

                train,
                test,

                target,
                features
            )
        )


        # ====================================================
        # STORE
        # ====================================================

        result = test[
            SERIES_COLS
            + ["month_date"]
        ].copy()


        result["origin"] = (
            origin
        )


        result["horizon"] = (
            horizon
        )


        result["actual"] = (
            actual
        )


        result[
            "Last quantity"
        ] = pred_last


        result[
            "Linear Regression"
        ] = pred_linear


        result[
            "Random Forest"
        ] = pred_rf


        result[
            "LightGBM Direct"
        ] = pred_lgb_direct


        result[
            "LightGBM Residual"
        ] = pred_lgb_residual


        prediction_blocks.append(
            result
        )




# ============================================================
# 27. COMBINE ALL PREDICTIONS
# ============================================================

all_predictions = pd.concat(
    prediction_blocks,
    ignore_index=True
)


print(
    "Total prediction rows:",
    f"{len(all_predictions):,}"
)


print("\nRows by horizon:")

print(
    all_predictions[
        "horizon"
    ]
    .value_counts()
    .sort_index()
)





# ============================================================
# 28. FINAL MODEL RESULTS H1 → H6
# ============================================================

MODEL_COLUMNS = [

    "Last quantity",
    "Linear Regression",
    "Random Forest",
    "LightGBM Direct",
    "LightGBM Residual"
]


final_results = []


for horizon in FORECAST_HORIZONS:

    temp = all_predictions[

        all_predictions[
            "horizon"
        ] == horizon

    ].copy()


    for model_name in MODEL_COLUMNS:

        metrics = calculate_metrics(

            temp["actual"],

            temp[model_name]
        )


        final_results.append({

            "horizon":
                f"H{horizon}",

            "model":
                model_name,

            "n":
                len(temp),

            **metrics
        })


final_results = (

    pd.DataFrame(
        final_results
    )

    .sort_values(
        [
            "horizon",
            "WAPE"
        ]
    )

    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 130)

print(
    "FINAL MODEL COMPARISON — SITE × SUPPLIER × CATEGORY — H1 TO H6"
)

print("=" * 130)


print(
    final_results
    .to_string(
        index=False
    )
)



# ============================================================
# 29. WAPE PIVOT TABLE
# ============================================================

wape_table = (

    final_results

    .pivot(
        index="horizon",
        columns="model",
        values="WAPE"
    )

    * 100
)


print("\nWAPE (%)")

print(
    wape_table
    .round(2)
    .to_string()
)


# ============================================================
# 30. POSITIVE WAPE PIVOT
# ============================================================

positive_wape_table = (

    final_results

    .pivot(
        index="horizon",
        columns="model",
        values="Positive_WAPE"
    )

    * 100
)


print("\nPOSITIVE WAPE (%)")

print(
    positive_wape_table
    .round(2)
    .to_string()
)





