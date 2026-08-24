
-- table dim_article_clean

CREATE OR REPLACE VIEW 
"@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."dim_article_clean" 
AS 

SELECT 

    CASE 
        WHEN "Article Code" = '000000000160112756' 
         AND "Supplier Code" = '0110000010' 
        THEN '000000000160112750' 
        ELSE "Article Code" 
    END AS "Article Code", 

    "Product NL", 
    "Product FR", 
    "Supplier Code", 
    "Supplier Name", 
    "Manufacturer Code", 
    "Manufacturer", 
    "Local Product Category 1", 
    "Local Product Category 2", 
    "Local Product Category 3", 
    "Local Product Category 4", 
    "Local Product Category 5", 
    "Local Product Category 6", 
    "Order Unit", 
    "Quantity (supplier files)", 
    "Brut spend (supplier files)", 
    "3x net spend", 
    "Local Product Category" 

FROM 
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."dim_article";




-- Construction de la table dim_category_clean

CREATE OR REPLACE VIEW
"@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."dim_category_clean"
AS

SELECT DISTINCT

    TRIM(CAST(category_id AS VARCHAR)) AS category_id,

    TRIM(CAST(CoderingPHLevel1 AS VARCHAR)) AS category_level_1_id,
    TRIM(CAST(LabelPHLevel1 AS VARCHAR)) AS category_level_1_name,

    TRIM(CAST(CoderingPHLevel2 AS VARCHAR)) AS category_level_2_id,
    TRIM(CAST(LabelPHLevel2 AS VARCHAR)) AS category_level_2_name,

    TRIM(CAST(CoderingPHLevel3 AS VARCHAR)) AS category_level_3_id,
    TRIM(CAST(LabelPHLevel3 AS VARCHAR)) AS category_level_3_name,

    TRIM(CAST(LabelPHLevel4 AS VARCHAR)) AS category_name,

    TRIM(CAST(CPI_id AS VARCHAR)) AS cpi_id,
    TRIM(CAST("CPI Category" AS VARCHAR)) AS cpi_category

FROM
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."Distinct categories"

WHERE
    category_id IS NOT NULL;





-- construction de la table fact_inflation_cpi_clean




CREATE OR REPLACE VIEW
"@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."fact_inflation_cpi_clean"
AS

WITH cpi_long AS (

    SELECT
        COICOP AS cpi_id,
        "New Description" AS cpi_category,
        Coef AS coefficient,
        DATE '2021-01-01' AS month_date,
        CAST("44197" AS DOUBLE) AS cpi_index
    FROM
        "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."Fact_inflation_cpi"

    UNION ALL

    SELECT
        COICOP,
        "New Description",
        Coef,
        DATE '2021-02-01',
        CAST("44228" AS DOUBLE)
    FROM
        "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."Fact_inflation_cpi"

    /* ... même principe pour tous les mois ... */

),

cpi_with_lags AS (

    SELECT
        cpi_id,
        cpi_category,
        coefficient,
        month_date,
        cpi_index,

        LAG(cpi_index, 1) OVER (
            PARTITION BY cpi_id
            ORDER BY month_date
        ) AS cpi_index_previous_month,

        LAG(cpi_index, 12) OVER (
            PARTITION BY cpi_id
            ORDER BY month_date
        ) AS cpi_index_previous_year

    FROM cpi_long
)

SELECT

    cpi_id,
    cpi_category,
    coefficient,
    month_date,
    cpi_index,

    CASE
        WHEN cpi_index_previous_month IS NULL
          OR cpi_index_previous_month = 0
        THEN NULL
        ELSE ((cpi_index / cpi_index_previous_month) - 1) * 100
    END AS cpi_mom_pct,

    CASE
        WHEN cpi_index_previous_year IS NULL
          OR cpi_index_previous_year = 0
        THEN NULL
        ELSE ((cpi_index / cpi_index_previous_year) - 1) * 100
    END AS cpi_yoy_pct

FROM cpi_with_lags

WHERE cpi_index IS NOT NULL;







-- construction de la table Fact_purchase_category_clean





CREATE OR REPLACE VIEW
"@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."fact_purchase_category_clean"
AS

SELECT
    DATE_TRUNC('MONTH', p.period_date) AS month_date,

    p.site_code AS site_id,
    p.site_name AS site_name,

    p.supplier_code AS supplier_id,

    a."Local Product Category" AS category_id,

    /* Main volume measure */
    SUM(p.quantity) AS quantity,

    /* Main monetary measure */
    SUM(p.gross_spend) AS gross_spend,

    /* Average gross unit price */
    CASE
        WHEN SUM(p.quantity) = 0 THEN NULL
        ELSE SUM(p.gross_spend) / SUM(p.quantity)
    END AS avg_gross_unit_price,

    /* Number of distinct articles actually purchased */
    COUNT(
        DISTINCT CASE
            WHEN p.quantity <> 0 THEN p.article_code
            ELSE NULL
        END
    ) AS active_article_count,

    /* Number of source purchasing rows */
    COUNT(*) AS purchase_row_count

FROM
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."Supplier_Spend_Clean_Corrected" p

LEFT JOIN
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."dim_article_clean" a
        ON p.article_code = a."Article Code"

WHERE
    p.period_date IS NOT NULL
    AND a."Local Product Category" IS NOT NULL

GROUP BY
    DATE_TRUNC('MONTH', p.period_date),
    p.site_code,
    p.site_name,
    p.supplier_code,
    a."Local Product Category";




-- nettoyage des valeures nulles de la table fact_purchase_category_clean 




CREATE OR REPLACE VIEW
"@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."fact_purchase_category_clean"
AS

SELECT

    month_date,
    site_id,
    site_name,
    supplier_id,
    category_id,

    COALESCE(quantity, 0) AS quantity,
    COALESCE(gross_spend, 0) AS gross_spend

FROM
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."fact_purchase_monthly_category";








-- Consolidation de la table dénormalisée finale pour l'extraction vers python ml_demand_forecast_base






CREATE OR REPLACE VIEW
"@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."ml_demand_forecast_base"
AS

SELECT

    /* =========================
       KEYS
       ========================= */

    f.month_date,
    f.site_id,
    f.site_name,
    f.supplier_id,
    f.category_id,


    /* =========================
       PURCHASE / TARGET
       ========================= */

    f.quantity,
    f.gross_spend,


    /* =========================
       CATEGORY
       ========================= */

    c.category_level_1_id,
    c.category_level_1_name,
    c.category_level_2_id,
    c.category_level_2_name,
    c.category_level_3_id,
    c.category_level_3_name,
    c.category_name,

    c.cpi_id,
    c.cpi_category,


    /* =========================
       SITE
       ========================= */

    s."Postal code" AS postal_code,
    s."Municipality" AS municipality,
    s."Confidence" AS municipality_confidence,
    s."Region" AS region,
    s."School holiday region" AS school_holiday_region,


    /* =========================
       CALENDAR
       ========================= */

    m."year" AS calendar_year,
    m.month_number,
    m.quarter,
    m.days_in_month,
    m.working_days,
    m.weekdays_mon_fri,
    m.public_holidays,
    m.public_holidays_weekdays,

    m.school_holiday_weekdays_flanders,
    m.school_holiday_ratio_flanders,
    m.school_holiday_weekdays_fwb,
    m.school_holiday_ratio_fwb,


    /* =========================
       SCHOOL HOLIDAY SELECTION
       ========================= */

    CASE
        WHEN s."School holiday region" = 'FLANDERS'
        THEN m.school_holiday_ratio_flanders

        WHEN s."School holiday region" = 'FWB'
        THEN m.school_holiday_ratio_fwb

        WHEN s."School holiday region" = 'BRUSSELS_BOTH'
        THEN (
            COALESCE(m.school_holiday_ratio_flanders, 0)
            +
            COALESCE(m.school_holiday_ratio_fwb, 0)
        ) / 2.0

        ELSE NULL
    END AS school_holiday_ratio_selected,


    CASE
        WHEN s."School holiday region" = 'FLANDERS'
        THEN m.school_holiday_weekdays_flanders

        WHEN s."School holiday region" = 'FWB'
        THEN m.school_holiday_weekdays_fwb

        WHEN s."School holiday region" = 'BRUSSELS_BOTH'
        THEN (
            COALESCE(m.school_holiday_weekdays_flanders, 0)
            +
            COALESCE(m.school_holiday_weekdays_fwb, 0)
        ) / 2.0

        ELSE NULL
    END AS school_holiday_weekdays_selected,


    /* =========================
       CPI
       ========================= */

    i.cpi_index,
    i.cpi_mom_pct,
    i.cpi_yoy_pct,
    i.coefficient AS cpi_coefficient


FROM
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."fact_purchase_category_clean" f


LEFT JOIN
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."dim_category_clean" c
    ON f.category_id = c.category_id


LEFT JOIN
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."dim_site_completed" s
    ON f.site_id = s.site_code


LEFT JOIN
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."dim_month_with_school_holidays" m
    ON f.month_date = m.month_date


LEFT JOIN
    "@_Arnauld.Marchal.ext.AZ@sodexo.com"."Personal workspace"."fact_inflation_cpi_clean" i
    ON c.cpi_id = i.cpi_id
    AND f.month_date = i.month_date


/* Exclusion des observations sans site,
   car le modèle prévoit au niveau site */

WHERE
    f.site_id IS NOT NULL;
