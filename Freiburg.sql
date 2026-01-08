CREATE OR ALTER VIEW [list_views].[Gesamt_DB_SCHEMA_Test_Freiburg]
AS
WITH Base AS (
    SELECT DISTINCT
        [StoreName],
        [Monat],
        CASE
            WHEN LTRIM(RTRIM(CONVERT(nvarchar(50), [DBEbene]))) = 'DB1' THEN 'E1'
            WHEN LTRIM(RTRIM(CONVERT(nvarchar(50), [DBEbene]))) = 'DB2' THEN 'E2'
            WHEN LTRIM(RTRIM(CONVERT(nvarchar(50), [DBEbene]))) = 'DB3' THEN 'E3'
            ELSE LTRIM(RTRIM(CONVERT(nvarchar(50), [DBEbene])))
        END AS [Ebene],
        CAST([Position] AS nvarchar(50)) AS [EPos],
        [Kenngröße],
        [ProduktKategorie],
        [ProduktLinie],
        [Wert]
    FROM [ERPDEV].[list_views].[V_LIST_G15_GESAMT_DBSCHEMA]
    WHERE [StoreName] = 'Freiburg im Breisgau'
)
,Totals AS (
    SELECT
        B.[StoreName],
        B.[Monat],
        SUM(CASE WHEN B.[Ebene] = 'E1' THEN B.[Wert] ELSE 0 END) AS [E1_Total],
        SUM(CASE WHEN B.[Ebene] = 'E2' THEN B.[Wert] ELSE 0 END) AS [E2_Total],
        SUM(CASE WHEN B.[Ebene] = 'E3' THEN B.[Wert] ELSE 0 END) AS [E3_Total],
        SUM(B.[Wert]) AS [Gesamt_Total]
    FROM Base B
    GROUP BY B.[StoreName], B.[Monat]
)
SELECT
    B.[StoreName],
    B.[Monat],
    B.[Ebene],
    B.[EPos],
    B.[Kenngröße],
    B.[ProduktKategorie],
    B.[ProduktLinie],
    B.[Wert],
    T.[E1_Total],
    T.[E2_Total],
    T.[E3_Total],
    T.[Gesamt_Total]
FROM Base B
JOIN Totals T
    ON B.[StoreName] = T.[StoreName]
    AND B.[Monat] = T.[Monat];

-- Beispiel:
SELECT *
FROM [list_views].[Gesamt_DB_SCHEMA_Test_Freiburg]
ORDER BY Monat, Ebene, EPos, Kenngröße, ProduktKategorie, ProduktLinie;