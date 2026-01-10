USE [ERPDEV];
GO

CREATE OR ALTER VIEW [list_views].[G14_Gesamt_DB_SCHEMA]
AS
WITH Base AS (
    SELECT DISTINCT
        [StoreName],
        [Monat],
        [Ebene],
        [EPos],
        [Kenngröße],
        [ProduktKategorie],
        [ProduktLinie],
        [Wert]
    FROM [ERPDEV].[list_views].[V_LIST_LEHPE_MEASURES]
   
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
    AND B.[Monat] = T.[Monat]
;
GO

-- Beispiel:
SELECT *
FROM [list_views].[G14_Gesamt_DB_SCHEMA]
WHERE StoreName = 'Freiburg im Breisgau' AND Kenngröße='SalesPric'
ORDER BY Monat, Ebene, TRY_CONVERT(int, EPos), EPos, Kenngröße, ProduktKategorie, ProduktLinie;