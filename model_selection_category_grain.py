
# CONFIGURATION SUPPLIER × CATEGORY


SC_SERIES_COLS = [
    "supplier_id",
    "category_id"
]

SC_FORECAST_HORIZONS = [
    1, 2, 3, 4, 5, 6
]

# Same origins as site-level H1-H6 experiment
SC_BACKTEST_ORIGINS = pd.to_datetime([
    "2025-04-01",
    "2025-08-01",
    "2025-12-01"
])

SC_RANDOM_STATE = 42

print("Supplier × Category experiment configured.")



# AGGREGATION DEMANDE ACTUELLE
# supplier × category × month


sc_actual = (
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
        quantity=(
            "quantity",
            "sum"
        ),

        gross_spend=(
            "gross_spend",
            "sum"
        ),

        active_sites=(
            "site_id",
            "nunique"
        )
    )
)

sc_actual = (
    sc_actual
    .sort_values(
        SC_SERIES_COLS
        + ["month_date"]
    )
    .reset_index(drop=True)
)

print(
    "Rows:",
    f"{len(sc_actual):,}"
)

print(
    "Supplier × category series:",
    f"{sc_actual[SC_SERIES_COLS].drop_duplicates().shape[0]:,}"
)

print(
    "Zero observation rate:",
    f"{(sc_actual['quantity'] == 0).mean():.2%}"
)


# PANEL MENSUEL COMPLET SUPPLIER × CATEGORY


sc_months = pd.DataFrame({
    "month_date": pd.date_range(
        start=df["month_date"].min(),
        end=df["month_date"].max(),
        freq="MS"
    )
})

sc_keys = (
    sc_actual[
        SC_SERIES_COLS
    ]
    .drop_duplicates()
)


sc_panel = (
    sc_keys
    .merge(
        sc_months,
        how="cross"
    )
    .merge(
        sc_actual,
        on=
            SC_SERIES_COLS
            + ["month_date"],
        how="left"
    )
)


sc_panel = (
    sc_panel
    .sort_values(
        SC_SERIES_COLS
        + ["month_date"]
    )
    .reset_index(drop=True)
)


print(
    "Complete SC panel:",
    sc_panel.shape
)



# METADONNEES SUPPLIER × CATEGORY


sc_metadata_cols = [
    "category_level_1_id",
    "category_level_2_id",
    "category_level_3_id",
    "cpi_id"
]


sc_metadata = (
    df[
        SC_SERIES_COLS
        + sc_metadata_cols
    ]
    .groupby(
        SC_SERIES_COLS,
        as_index=False,
        observed=True
    )
    .first()
)


sc_panel = sc_panel.merge(
    sc_metadata,
    on=SC_SERIES_COLS,
    how="left"
)



# CPI AU GRAIN SUPPLIER × CATEGORY


sc_cpi = (
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
        cpi_index=(
            "cpi_index",
            "median"
        ),

        cpi_mom_pct=(
            "cpi_mom_pct",
            "median"
        ),

        cpi_yoy_pct=(
            "cpi_yoy_pct",
            "median"
        ),

        cpi_coefficient=(
            "cpi_coefficient",
            "median"
        )
    )
)


sc_panel = sc_panel.merge(
    sc_cpi,
    on=[
        "month_date",
        "supplier_id",
        "category_id"
    ],
    how="left"
)



# COMPOSITION SITE PAR CATEGORIE


site_region = (
    df[
        [
            "month_date",
            "supplier_id",
            "category_id",
            "site_id",
            "region"
        ]
    ]
    .drop_duplicates()
)


site_region["is_flanders"] = (
    site_region["region"]
    .eq("Flanders")
    .astype(float)
)

site_region["is_wallonia"] = (
    site_region["region"]
    .eq("Wallonia")
    .astype(float)
)

site_region["is_brussels"] = (
    site_region["region"]
    .eq("Brussels-Capital")
    .astype(float)
)


regional_mix = (
    site_region
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
        share_sites_flanders=(
            "is_flanders",
            "mean"
        ),

        share_sites_wallonia=(
            "is_wallonia",
            "mean"
        ),

        share_sites_brussels=(
            "is_brussels",
            "mean"
        )
    )
)


sc_panel = sc_panel.merge(
    regional_mix,
    on=[
        "month_date",
        "supplier_id",
        "category_id"
    ],
    how="left"
)



# HISTORIQUE DE DEMANDE


sc_panel["quantity_current"] = (
    sc_panel["quantity"]
)


sc_grouped = sc_panel.groupby(
    SC_SERIES_COLS,
    observed=True
)


for lag in range(1, 13):

    sc_panel[
        f"quantity_lag_{lag}"
    ] = (
        sc_grouped["quantity"]
        .shift(lag)
    )


# FEATURES DE DEMANDE ROULANTE

sc_recent_3 = [
    "quantity_current",
    "quantity_lag_1",
    "quantity_lag_2"
]

sc_recent_6 = [
    "quantity_current"
] + [
    f"quantity_lag_{i}"
    for i in range(1, 6)
]

sc_recent_12 = [
    "quantity_current"
] + [
    f"quantity_lag_{i}"
    for i in range(1, 12)
]


sc_panel["quantity_mean_3"] = (
    sc_panel[
        sc_recent_3
    ].mean(axis=1)
)

sc_panel["quantity_mean_6"] = (
    sc_panel[
        sc_recent_6
    ].mean(axis=1)
)

sc_panel["quantity_mean_12"] = (
    sc_panel[
        sc_recent_12
    ].mean(axis=1)
)


sc_panel["quantity_std_6"] = (
    sc_panel[
        sc_recent_6
    ].std(axis=1)
)

sc_panel["quantity_std_12"] = (
    sc_panel[
        sc_recent_12
    ].std(axis=1)
)



# FEATURES D'INTERMITTENCE


for window_name, cols in [
    ("3", sc_recent_3),
    ("6", sc_recent_6),
    ("12", sc_recent_12)
]:

    sc_panel[
        f"positive_months_{window_name}"
    ] = (
        (
            sc_panel[cols] > 0
        )
        .sum(axis=1)
    )

    sc_panel[
        f"zero_ratio_{window_name}"
    ] = (
        (
            sc_panel[cols] == 0
        )
        .mean(axis=1)
    )


sc_panel[
    "positive_quantity_mean_6"
] = (
    sc_panel[
        sc_recent_6
    ]
    .where(
        sc_panel[
            sc_recent_6
        ] > 0
    )
    .mean(axis=1)
)


sc_panel[
    "positive_quantity_mean_12"
] = (
    sc_panel[
        sc_recent_12
    ]
    .where(
        sc_panel[
            sc_recent_12
        ] > 0
    )
    .mean(axis=1)
)





# DERNIERES QUANTITE POSITIVE


sc_panel["_positive_quantity"] = (
    sc_panel["quantity"]
    .where(
        sc_panel["quantity"] > 0
    )
)


sc_panel[
    "last_positive_quantity"
] = (
    sc_panel
    .groupby(
        SC_SERIES_COLS,
        observed=True
    )["_positive_quantity"]
    .ffill()
)


sc_panel.drop(
    columns="_positive_quantity",
    inplace=True
)


# FEATURES NOMBRES DE SITES ACTIF PAR CATEGORIE


sc_grouped = sc_panel.groupby(
    SC_SERIES_COLS,
    observed=True
)


sc_panel["active_sites_current"] = (
    sc_panel["active_sites"]
)


for lag in [
    1, 2, 3, 6, 12
]:

    sc_panel[
        f"active_sites_lag_{lag}"
    ] = (
        sc_grouped[
            "active_sites"
        ]
        .shift(lag)
    )


sc_panel[
    "active_sites_change_1m"
] = (
    sc_panel["active_sites_current"]
    - sc_panel["active_sites_lag_1"]
)


sc_panel[
    "active_sites_change_3m"
] = (
    sc_panel["active_sites_current"]
    - sc_panel["active_sites_lag_3"]
)



# FEATURE QUANTITE PAR SITE ACTIF


sc_panel[
    "quantity_per_site_current"
] = np.where(

    sc_panel["active_sites_current"] > 0,

    sc_panel["quantity_current"]
    / sc_panel["active_sites_current"],

    np.nan
)


sc_grouped = sc_panel.groupby(
    SC_SERIES_COLS,
    observed=True
)


for lag in [
    1, 2, 3, 6, 12
]:

    sc_panel[
        f"quantity_per_site_lag_{lag}"
    ] = (
        sc_grouped[
            "quantity_per_site_current"
        ]
        .shift(lag)
    )



# FEATURE PRIX


sc_panel[
    "unit_price_current"
] = np.where(

    sc_panel["quantity"] > 0,

    sc_panel["gross_spend"]
    / sc_panel["quantity"],

    np.nan
)


sc_grouped = sc_panel.groupby(
    SC_SERIES_COLS,
    observed=True
)


for lag in [
    1, 2, 3, 6, 12
]:

    sc_panel[
        f"unit_price_lag_{lag}"
    ] = (
        sc_grouped[
            "unit_price_current"
        ]
        .shift(lag)
    )


sc_panel[
    "unit_price_change_1m"
] = (
    sc_panel["unit_price_current"]
    / sc_panel["unit_price_lag_1"]
    - 1
)


sc_panel[
    "unit_price_change_3m"
] = (
    sc_panel["unit_price_current"]
    / sc_panel["unit_price_lag_3"]
    - 1
)


sc_panel[
    "unit_price_change_1m"
] = (
    sc_panel[
        "unit_price_change_1m"
    ].clip(-0.8, 3)
)


sc_panel[
    "unit_price_change_3m"
] = (
    sc_panel[
        "unit_price_change_3m"
    ].clip(-0.8, 3)
)



# FEATURE HISTORIQUE CPI


sc_grouped = sc_panel.groupby(
    SC_SERIES_COLS,
    observed=True
)


SC_CPI_COLS = [
    "cpi_index",
    "cpi_mom_pct",
    "cpi_yoy_pct",
    "cpi_coefficient"
]


for col in SC_CPI_COLS:

    for lag in [
        1, 3, 6
    ]:

        sc_panel[
            f"{col}_lag_{lag}"
        ] = (
            sc_grouped[col]
            .shift(lag)
        )


sc_panel[
    "cpi_index_change_3m"
] = (
    sc_panel["cpi_index"]
    / sc_panel["cpi_index_lag_3"]
    - 1
)


sc_panel[
    "cpi_index_change_6m"
] = (
    sc_panel["cpi_index"]
    / sc_panel["cpi_index_lag_6"]
    - 1
)



# HORIZONS H1-H6


sc_grouped = sc_panel.groupby(
    SC_SERIES_COLS,
    observed=True
)


for horizon in SC_FORECAST_HORIZONS:

    sc_panel[
        f"target_h{horizon}"
    ] = (
        sc_grouped["quantity"]
        .shift(-horizon)
    )


print("Target availability:")

for h in SC_FORECAST_HORIZONS:

    print(
        f"H{h}: "
        f"{sc_panel[f'target_h{h}'].notna().sum():,}"
    )


# VARIABLES CALANDAIRES FUTURES AGGREGE


site_known = panel[
    panel[
        "quantity_current"
    ].notna()
].copy()


for h in SC_FORECAST_HORIZONS:

    cols_h = [

        "month_date",
        "supplier_id",
        "category_id",

        f"target_days_in_month_h{h}",
        f"target_working_days_h{h}",
        f"target_weekdays_mon_fri_h{h}",

        f"target_public_holidays_h{h}",
        f"target_public_holidays_weekdays_h{h}",

        f"target_school_holiday_ratio_selected_h{h}",
        f"target_school_holiday_weekdays_selected_h{h}",

        f"target_school_holiday_ratio_flanders_h{h}",
        f"target_school_holiday_ratio_fwb_h{h}"
    ]


    cols_h = [
        col
        for col in cols_h
        if col in site_known.columns
    ]


    numeric_h = [
        col
        for col in cols_h
        if col not in [
            "month_date",
            "supplier_id",
            "category_id"
        ]
    ]


    calendar_agg_h = (
        site_known[
            cols_h
        ]
        .groupby(
            [
                "month_date",
                "supplier_id",
                "category_id"
            ],
            as_index=False,
            observed=True
        )[numeric_h]
        .mean()
    )


    sc_panel = sc_panel.merge(
        calendar_agg_h,
        on=[
            "month_date",
            "supplier_id",
            "category_id"
        ],
        how="left"
    )



# SAISONNALITE MOIS CIBLE


for h in SC_FORECAST_HORIZONS:

    target_date = (
        sc_panel["month_date"]
        + pd.DateOffset(
            months=h
        )
    )


    sc_panel[
        f"target_month_h{h}"
    ] = (
        target_date.dt.month
    )


    sc_panel[
        f"target_quarter_h{h}"
    ] = (
        target_date.dt.quarter
    )


    sc_panel[
        f"target_month_sin_h{h}"
    ] = np.sin(
        2 * np.pi
        * target_date.dt.month
        / 12
    )


    sc_panel[
        f"target_month_cos_h{h}"
    ] = np.cos(
        2 * np.pi
        * target_date.dt.month
        / 12
    )



# INDEX DE TEMPS


sc_first_date = (
    sc_panel["month_date"]
    .min()
)


sc_panel["time_index"] = (

    (
        sc_panel[
            "month_date"
        ].dt.year
        - sc_first_date.year
    ) * 12

    +

    (
        sc_panel[
            "month_date"
        ].dt.month
        - sc_first_date.month
    )
)



# FEATURES GENERALES


sc_numeric_common = [


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

    "quantity_std_6",
    "quantity_std_12",

    "positive_months_3",
    "positive_months_6",
    "positive_months_12",

    "zero_ratio_3",
    "zero_ratio_6",
    "zero_ratio_12",

    "positive_quantity_mean_6",
    "positive_quantity_mean_12",

    "last_positive_quantity",


    "active_sites_current",
    "active_sites_lag_1",
    "active_sites_lag_2",
    "active_sites_lag_3",
    "active_sites_lag_6",
    "active_sites_lag_12",

    "active_sites_change_1m",
    "active_sites_change_3m",


    "quantity_per_site_current",
    "quantity_per_site_lag_1",
    "quantity_per_site_lag_2",
    "quantity_per_site_lag_3",
    "quantity_per_site_lag_6",
    "quantity_per_site_lag_12",


    "share_sites_flanders",
    "share_sites_wallonia",
    "share_sites_brussels",


    "unit_price_current",
    "unit_price_lag_1",
    "unit_price_lag_2",
    "unit_price_lag_3",
    "unit_price_lag_6",
    "unit_price_lag_12",

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


    "time_index"
]



# FEATURES CATEGORIELLES


sc_categorical_features = [

    "supplier_id",
    "category_id",

    "category_level_1_id",
    "category_level_2_id",
    "category_level_3_id",

    "cpi_id"
]



# FEATURES FUTURES


def get_sc_features(horizon):

    future_features = [

        f"target_days_in_month_h{horizon}",
        f"target_working_days_h{horizon}",
        f"target_weekdays_mon_fri_h{horizon}",

        f"target_public_holidays_h{horizon}",
        f"target_public_holidays_weekdays_h{horizon}",

        f"target_school_holiday_ratio_selected_h{horizon}",
        f"target_school_holiday_weekdays_selected_h{horizon}",

        f"target_school_holiday_ratio_flanders_h{horizon}",
        f"target_school_holiday_ratio_fwb_h{horizon}",

        f"target_month_h{horizon}",
        f"target_quarter_h{horizon}",
        f"target_month_sin_h{horizon}",
        f"target_month_cos_h{horizon}"
    ]


    features = (
        sc_numeric_common
        + future_features
        + sc_categorical_features
    )


    features = [
        col
        for col in features
        if col in sc_panel.columns
    ]


    return list(
        dict.fromkeys(
            features
        )
    )

# PREPARATION LIGHTGBM


sc_panel_lgb = (
    sc_panel.copy()
)


for col in sc_categorical_features:

    if col in sc_panel_lgb.columns:

        sc_panel_lgb[col] = (
            sc_panel_lgb[col]
            .astype("category")
        )



# REGRESSION LINEAIRE


def sc_fit_predict_linear(
    X_train,
    y_train,
    X_test,
    features
):

    categorical = [
        col
        for col in sc_categorical_features
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
                handle_unknown="ignore"
            )
        )
    ])


    preprocessing = (
        ColumnTransformer([
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
    )


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


    pred = model.predict(
        X_test
    )


    return np.maximum(
        pred,
        0
    )



# RANDOM FOREST


def sc_fit_predict_rf(
    X_train,
    y_train,
    X_test,
    features
):

    categorical = [
        col
        for col in sc_categorical_features
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

        n_estimators=150,

        max_depth=10,

        min_samples_leaf=5,

        max_features=0.6,

        random_state=
            SC_RANDOM_STATE,

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


    pred = model.predict(
        X_test
    )


    return np.maximum(
        pred,
        0
    )



# LIGHTGBM DIRECT


def sc_fit_predict_lgb_direct(
    train,
    test,
    target,
    features
):

    categorical = [
        col
        for col in sc_categorical_features
        if col in features
    ]


    model = LGBMRegressor(

        objective="regression_l1",

        n_estimators=350,

        learning_rate=0.03,

        num_leaves=20,

        min_child_samples=15,

        colsample_bytree=0.8,

        reg_alpha=0.2,

        reg_lambda=0.5,

        random_state=
            SC_RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1
    )


    model.fit(

        train[features],
        train[target],

        categorical_feature=
            categorical
    )


    pred = model.predict(
        test[features]
    )


    return np.maximum(
        pred,
        0
    )



# LIGHTGBM RESIDUAL


def sc_fit_predict_lgb_residual(
    train,
    test,
    target,
    features
):

    categorical = [
        col
        for col in sc_categorical_features
        if col in features
    ]


    y_residual = (
        train[target]
        - train[
            "quantity_current"
        ]
    )


    model = LGBMRegressor(

        objective="regression_l1",

        n_estimators=350,

        learning_rate=0.03,

        num_leaves=20,

        min_child_samples=15,

        colsample_bytree=0.8,

        reg_alpha=0.2,

        reg_lambda=0.5,

        random_state=
            SC_RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1
    )


    model.fit(

        train[features],
        y_residual,

        categorical_feature=
            categorical
    )


    predicted_change = (
        model.predict(
            test[features]
        )
    )


    pred = (
        test[
            "quantity_current"
        ].values
        + predicted_change
    )


    return np.maximum(
        pred,
        0
    )




# ENTRAINELENT FINAL SUPPLIER x CATEGORY


sc_prediction_blocks = []


for origin in SC_BACKTEST_ORIGINS:

    print("\n" + "=" * 90)
    print(
        "SC FORECAST ORIGIN:",
        origin.date()
    )
    print("=" * 90)


    for horizon in SC_FORECAST_HORIZONS:

        target = (
            f"target_h{horizon}"
        )

        features = (
            get_sc_features(
                horizon
            )
        )



        train_end = (
            origin
            - pd.DateOffset(
                months=horizon
            )
        )


        train = sc_panel_lgb[

            (
                sc_panel_lgb[
                    "month_date"
                ] <= train_end
            )

            & sc_panel_lgb[
                target
            ].notna()

            & sc_panel_lgb[
                "quantity_current"
            ].notna()

        ].copy()


        test = sc_panel_lgb[

            (
                sc_panel_lgb[
                    "month_date"
                ] == origin
            )

            & sc_panel_lgb[
                target
            ].notna()

            & sc_panel_lgb[
                "quantity_current"
            ].notna()

        ].copy()


        if len(test) == 0:

            continue


        print(
            f"H{horizon} | "
            f"Train={len(train):,} | "
            f"Test={len(test):,}"
        )


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


        pred_linear = (
            sc_fit_predict_linear(

                train[features],
                train[target],

                test[features],

                features
            )
        )




        pred_rf = (
            sc_fit_predict_rf(

                train[features],
                train[target],

                test[features],

                features
            )
        )




        pred_lgb_direct = (
            sc_fit_predict_lgb_direct(

                train,
                test,

                target,
                features
            )
        )



        pred_lgb_residual = (
            sc_fit_predict_lgb_residual(

                train,
                test,

                target,
                features
            )
        )



        result = test[
            SC_SERIES_COLS
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


        sc_prediction_blocks.append(
            result
        )



# RASSEMBLER LES PREDICTIONS


sc_predictions = pd.concat(
    sc_prediction_blocks,
    ignore_index=True
)


print(
    "Total forecast observations:",
    f"{len(sc_predictions):,}"
)


print(
    sc_predictions[
        "horizon"
    ]
    .value_counts()
    .sort_index()
)



# RESULTATS TEST AGGREGE


SC_MODELS = [

    "Last quantity",
    "Linear Regression",
    "Random Forest",
    "LightGBM Direct",
    "LightGBM Residual"
]


sc_final_results = []


for horizon in SC_FORECAST_HORIZONS:

    temp = sc_predictions[
        sc_predictions[
            "horizon"
        ] == horizon
    ]


    for model_name in SC_MODELS:

        metrics = calculate_metrics(

            temp["actual"],

            temp[
                model_name
            ]
        )


        sc_final_results.append({

            "horizon":
                f"H{horizon}",

            "model":
                model_name,

            "n":
                len(temp),

            **metrics
        })


sc_final_results = (

    pd.DataFrame(
        sc_final_results
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
    "FINAL MODEL COMPARISON — SUPPLIER × CATEGORY — H1 TO H6"
)

print("=" * 130)


print(
    sc_final_results
    .to_string(
        index=False
    )
)

# TABLE WAPE GLOBAL


sc_wape_table = (

    sc_final_results

    .pivot(
        index="horizon",
        columns="model",
        values="WAPE"
    )

    * 100
)


print("\nWAPE (%)")
print("=" * 90)

print(
    sc_wape_table
    .round(2)
    .to_string()
)


