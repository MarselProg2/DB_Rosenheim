USE [ERPDEV];
GO

CREATE OR ALTER VIEW [list_views].[G14_Test_DB_Schema]
AS
WITH Base AS (
    SELECT
        [StoreName],
        [Kenngröße],
        [Wert],
        [DBEbene],
        [Position],
        [Beschreibung],
        [Kostenart],
        [ProduktNr],
        [ProduktKategorie],
        [ProduktLinie],
        [Kampagne]
    FROM [ERPDEV].[list_views].[V_LIST_LEHPE_GESAMT_DBSCHEMA]
    WHERE LTRIM(RTRIM(CONVERT(nvarchar(50), [DBEbene]))) IN ('DB1', 'DB2', 'DB3')
),
Agg AS (
    SELECT
        [StoreName],
        [ProduktKategorie],
        [ProduktLinie],
        [Kampagne],
        SUM(CASE WHEN LOWER(LTRIM(RTRIM(CONVERT(nvarchar(200), [Kenngröße])))) IN (
                'umsatzeur', 'umsatz eur', 'umsatz', 'revenue', 'sales', 'sales eur', 'saleseur', 'totalrevenue'
            ) THEN TRY_CONVERT(decimal(18, 4), [Wert]) ELSE 0 END) AS Umsatz,
        SUM(CASE WHEN LOWER(LTRIM(RTRIM(CONVERT(nvarchar(200), [Kenngröße])))) IN (
                'transferpriceeur', 'transfer price eur'
            ) THEN TRY_CONVERT(decimal(18, 4), [Wert]) ELSE 0 END) AS Transfer,
        SUM(CASE WHEN LOWER(LTRIM(RTRIM(CONVERT(nvarchar(200), [Kenngröße])))) IN (
                'commission in eur', 'commission',
                'discountaufmaterialeur', 'discount auf material eur',
                'discountaufmaterialkategorieeur', 'discount auf material kategorie eur'
            ) THEN TRY_CONVERT(decimal(18, 4), [Wert]) ELSE 0 END) AS DB2_Cost,
        SUM(CASE WHEN LOWER(LTRIM(RTRIM(CONVERT(nvarchar(200), [Kenngröße])))) IN (
                'additional procurement costs',
                'marketing campaign',
                'monthly rent',
                'monthly salary',
                'monthly social costs'
            ) THEN TRY_CONVERT(decimal(18, 4), [Wert]) ELSE 0 END) AS DB3_Cost
    FROM Base
    GROUP BY
        [StoreName],
        [ProduktKategorie],
        [ProduktLinie],
        [Kampagne]
),
Totals AS (
    SELECT
        [StoreName],
        [ProduktKategorie],
        [ProduktLinie],
        [Kampagne],
        Umsatz,
        Transfer,
        (Umsatz + Transfer) AS E1_Total,
        (Umsatz + Transfer) - DB2_Cost AS E2_Total,
        ((Umsatz + Transfer) - DB2_Cost) - DB3_Cost AS E3_Total
    FROM Agg
)
SELECT
    [StoreName],
    [Kenngröße],
    [Wert],
    [DBEbene],
    [Position],
    [Beschreibung],
    [Kostenart],
    [ProduktNr],
    [ProduktKategorie],
    [ProduktLinie],
    [Kampagne]
FROM Base

UNION ALL

SELECT
    t.[StoreName],
    'E1 Total' AS [Kenngröße],
    CAST(t.E1_Total AS decimal(18, 2)) AS [Wert],
    'DB1' AS [DBEbene],
    NULL AS [Position],
    NULL AS [Beschreibung],
    NULL AS [Kostenart],
    NULL AS [ProduktNr],
    t.[ProduktKategorie],
    t.[ProduktLinie],
    t.[Kampagne]
FROM Totals t

UNION ALL

SELECT
    t.[StoreName],
    'E2 Total' AS [Kenngröße],
    CAST(t.E2_Total AS decimal(18, 2)) AS [Wert],
    'DB2' AS [DBEbene],
    NULL AS [Position],
    NULL AS [Beschreibung],
    NULL AS [Kostenart],
    NULL AS [ProduktNr],
    t.[ProduktKategorie],
    t.[ProduktLinie],
    t.[Kampagne]
FROM Totals t

UNION ALL

SELECT
    t.[StoreName],
    'E3 Total' AS [Kenngröße],
    CAST(t.E3_Total AS decimal(18, 2)) AS [Wert],
    'DB3' AS [DBEbene],
    NULL AS [Position],
    NULL AS [Beschreibung],
    NULL AS [Kostenart],
    NULL AS [ProduktNr],
    t.[ProduktKategorie],
    t.[ProduktLinie],
    t.[Kampagne]
FROM Totals t;
GO

-- Quick check (optional)
SELECT TOP (200) *
FROM [ERPDEV].[list_views].[G14_Test_DB_Schema]

ORDER BY ProduktLinie, ProduktKategorie, DBEbene, Kenngröße;
