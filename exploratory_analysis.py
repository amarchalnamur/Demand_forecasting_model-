
# IMPORTS


import pandas as pd
import numpy as np

FILEPATH = r"C:\Users\arnau\UNAMUR\projet perso\ml_demand_forecast_base.csv"

SERIES_COLS = [
    "site_id",
    "supplier_id",
    "category_id"
]

GRAIN_COLS = [
    "month_date",
    "site_id",
    "supplier_id",
    "category_id"
]

df = pd.read_csv(
    FILEPATH,
    low_memory=False
)

df["month_date"] = pd.to_datetime(
    df["month_date"]
)

print("Shape:", df.shape)
print(
    "Date range:",
    df["month_date"].min(),
    "->",
    df["month_date"].max()
)



# UNICITE DU GRAIN


grain_check = (
    df
    .groupby(
        GRAIN_COLS,
        observed=True
    )
    .size()
)

print("\n" + "=" * 80)
print("GRAIN CHECK")
print("=" * 80)

print(
    "Unique combinations:",
    f"{len(grain_check):,}"
)

print(
    "Combinations with duplicates:",
    f"{(grain_check > 1).sum():,}"
)



# HISTORIQUES DES SERIES


series_history = (
    df
    .groupby(
        SERIES_COLS,
        observed=True
    )
    .agg(
        first_month=("month_date", "min"),
        last_month=("month_date", "max"),
        observed_months=("month_date", "nunique")
    )
    .reset_index()
)

series_history["expected_months"] = (
    (
        series_history["last_month"].dt.year
        - series_history["first_month"].dt.year
    ) * 12
    +
    (
        series_history["last_month"].dt.month
        - series_history["first_month"].dt.month
    )
    + 1
)

series_history["history_density"] = (
    series_history["observed_months"]
    / series_history["expected_months"]
)

print("\n" + "=" * 80)
print("TIME SERIES HISTORY")
print("=" * 80)

print(
    "Number of series:",
    f"{len(series_history):,}"
)

print("\nObserved months:")
print(
    series_history["observed_months"]
    .describe(
        percentiles=[
            .10, .25, .50, .75, .90
        ]
    )
)

print("\nHistory density:")
print(
    series_history["history_density"]
    .describe(
        percentiles=[
            .10, .25, .50, .75, .90
        ]
    )
)

print(
    "\nComplete histories:",
    f"{(series_history['history_density'] == 1).mean():.1%}"
)



# ANALYSE DE DEMANDE NULLE


zero_stats = (
    df
    .assign(
        zero_demand=
            df["quantity"] == 0
    )
    .groupby(
        SERIES_COLS,
        observed=True
    )
    .agg(
        observations=("quantity", "size"),
        zero_months=("zero_demand", "sum")
    )
    .reset_index()
)

zero_stats["zero_ratio"] = (
    zero_stats["zero_months"]
    / zero_stats["observations"]
)

print("\n" + "=" * 80)
print("ZERO DEMAND")
print("=" * 80)

print(
    zero_stats["zero_ratio"]
    .describe(
        percentiles=[
            .10, .25, .50, .75, .90, .95
        ]
    )
)

print(
    "\nOverall zero rate:",
    f"{(df['quantity'] == 0).mean():.1%}"
)

print(
    "Series with >50% zero months:",
    f"{(zero_stats['zero_ratio'] > .50).mean():.1%}"
)





# ANALYSE D'OCCURENCE DE LA DEMANDE


occurrence = df[
    SERIES_COLS
    + [
        "month_date",
        "quantity"
    ]
].copy()

occurrence["demand_positive"] = (
    occurrence["quantity"] > 0
).astype(int)

previous = occurrence[
    SERIES_COLS
    + [
        "month_date",
        "demand_positive"
    ]
].copy()

previous["month_date"] = (
    previous["month_date"]
    + pd.DateOffset(months=1)
)

previous = previous.rename(
    columns={
        "demand_positive":
            "previous_demand_positive"
    }
)

transition_df = occurrence.merge(
    previous,
    on=SERIES_COLS + ["month_date"],
    how="inner"
)

transition_matrix = pd.crosstab(
    transition_df[
        "previous_demand_positive"
    ],
    transition_df[
        "demand_positive"
    ],
    normalize="index"
)

print("\n" + "=" * 80)
print("DEMAND OCCURRENCE TRANSITION")
print("=" * 80)

print(transition_matrix)






# CORRELATION DE QUANTITE EN LAGS


base_quantity = df[
    SERIES_COLS
    + [
        "month_date",
        "quantity"
    ]
].copy()

print("\n" + "=" * 80)
print("QUANTITY CORRELATION WITH LAGS")
print("=" * 80)

for lag in [
    1, 2, 3, 6, 12
]:

    previous = (
        base_quantity.copy()
    )

    previous["month_date"] = (
        previous["month_date"]
        + pd.DateOffset(months=lag)
    )

    previous = previous.rename(
        columns={
            "quantity":
                f"quantity_lag_{lag}"
        }
    )

    aligned = base_quantity.merge(
        previous,
        on=
            SERIES_COLS
            + ["month_date"],
        how="inner"
    )

    corr = (
        aligned[
            [
                "quantity",
                f"quantity_lag_{lag}"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    print(
        f"Lag {lag:2d}: "
        f"{corr:.3f} "
        f"(n={len(aligned):,})"
    )





# ANALYSE DISTRIBUTION DES QUANTITES POSITIVES


positive_quantity = (
    df.loc[
        df["quantity"] > 0,
        "quantity"
    ]
)

print("\n" + "=" * 80)
print("POSITIVE QUANTITY DISTRIBUTION")
print("=" * 80)

print(
    positive_quantity
    .describe(
        percentiles=[
            .10,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99
        ]
    )
)




# VOLATILITE DES SERIES


series_volatility = (
    df[
        df["quantity"] > 0
    ]
    .groupby(
        SERIES_COLS,
        observed=True
    )
    .agg(
        mean_quantity=("quantity", "mean"),
        std_quantity=("quantity", "std")
    )
    .reset_index()
)

series_volatility["cv_quantity"] = (
    series_volatility["std_quantity"]
    / series_volatility["mean_quantity"]
)

print("\n" + "=" * 80)
print("QUANTITY CV")
print("=" * 80)

print(
    series_volatility[
        "cv_quantity"
    ]
    .dropna()
    .describe(
        percentiles=[
            .10,
            .25,
            .50,
            .75,
            .90
        ]
    )
)



# COMPARAISON DE GRANULARITE SITE VS CATEGORIE


# SITE × SUPPLIER × CATEGORY


site_series_cols = [
    "site_id",
    "supplier_id",
    "category_id"
]

site_stats = (
    df
    .groupby(
        site_series_cols,
        observed=True
    )
    .agg(
        observed_months=("month_date", "nunique"),
        mean_quantity=("quantity", "mean"),
        std_quantity=("quantity", "std"),
        zero_ratio=(
            "quantity",
            lambda x: (x == 0).mean()
        )
    )
    .reset_index()
)

site_stats["cv_quantity"] = (
    site_stats["std_quantity"]
    / site_stats["mean_quantity"]
)


# SUPPLIER × CATEGORY


supplier_category_df = (
    df
    .groupby(
        [
            "month_date",
            "supplier_id",
            "category_id"
        ],
        as_index=False,
        observed=True
    )
    .agg(
        quantity=("quantity", "sum")
    )
)

supplier_category_stats = (
    supplier_category_df
    .groupby(
        [
            "supplier_id",
            "category_id"
        ],
        observed=True
    )
    .agg(
        observed_months=("month_date", "nunique"),
        mean_quantity=("quantity", "mean"),
        std_quantity=("quantity", "std"),
        zero_ratio=(
            "quantity",
            lambda x: (x == 0).mean()
        )
    )
    .reset_index()
)

supplier_category_stats["cv_quantity"] = (
    supplier_category_stats["std_quantity"]
    / supplier_category_stats["mean_quantity"]
)


# TABLE DES RESULTATS


granularity_comparison = pd.DataFrame({

    "Metric": [
        "Number of series",
        "Overall zero rate",
        "Median zero ratio",
        "Series >50% zero months",
        "Median observed months",
        "Mean observed months",
        "Median quantity CV",
        "Mean quantity CV"
    ],

    "Site × Supplier × Category": [

        len(site_stats),

        (df["quantity"] == 0).mean(),

        site_stats["zero_ratio"].median(),

        (
            site_stats["zero_ratio"] > 0.50
        ).mean(),

        site_stats[
            "observed_months"
        ].median(),

        site_stats[
            "observed_months"
        ].mean(),

        site_stats[
            "cv_quantity"
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).median(),

        site_stats[
            "cv_quantity"
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).mean()
    ],

    "Supplier × Category": [

        len(supplier_category_stats),

        (
            supplier_category_df["quantity"] == 0
        ).mean(),

        supplier_category_stats[
            "zero_ratio"
        ].median(),

        (
            supplier_category_stats[
                "zero_ratio"
            ] > 0.50
        ).mean(),

        supplier_category_stats[
            "observed_months"
        ].median(),

        supplier_category_stats[
            "observed_months"
        ].mean(),

        supplier_category_stats[
            "cv_quantity"
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).median(),

        supplier_category_stats[
            "cv_quantity"
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).mean()
    ]
})


print("\n" + "=" * 110)
print(
    "GRANULARITY DIAGNOSTIC — "
    "SITE × SUPPLIER × CATEGORY "
    "VS SUPPLIER × CATEGORY"
)
print("=" * 110)

print(
    granularity_comparison
    .to_string(index=False)
)
