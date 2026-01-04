USE [ERPDEV];
GO

CREATE OR ALTER VIEW [ins_views].[V_LIST_G14_DB_CALCULATION_PRODUCT_ROSENHEIM] AS
WITH 
-- 1. Basisdaten pro Produkt (Umsatz & Variable Kosten)
ProductData AS (
    SELECT 
        [StoreName],
        [ProduktNr],
        [ProduktKategorie],
        [ProduktLinie],
        SUM(CASE WHEN [Kenngröße] = 'UmsatzEUR' THEN [Wert] ELSE 0 END) AS [Prod_Umsatz],
        SUM(CASE WHEN [Kenngröße] IN ('TransferPriceEUR', 'Additional Procurement Costs', 'Commission', 'DiscountAufMaterialEUR', 'DiscountAufMaterialKategorieEUR') THEN [Wert] ELSE 0 END) AS [Prod_VarKosten]
    FROM [ERPDEV].[list_views].[V_LIST_LEHPE_MEASURES]
    WHERE [StoreName] = 'Rosenheim'
    GROUP BY [StoreName], [ProduktNr], [ProduktKategorie], [ProduktLinie]
),
-- 2. Fixkosten der Filiale
StoreFixCosts AS (
    SELECT 
        [StoreName],
        SUM([Wert]) AS [Total_Fixkosten]
    FROM (
        SELECT DISTINCT [StoreName], [Monat], [Kenngröße], [Wert]
        FROM [ERPDEV].[list_views].[V_LIST_LEHPE_MEASURES]
        WHERE [StoreName] = 'Rosenheim'
          AND [Kenngröße] IN ('Monthly Rent', 'Monthly Salary', 'Monthly Social Costs', 'Marketing Campaign')
    ) DistinctCosts
    GROUP BY [StoreName]
),
-- 3. Gesamtumsatz der Filiale (für die Verteilung)
StoreTotalRevenue AS (
    SELECT 
        [StoreName],
        SUM(CASE WHEN [Kenngröße] = 'UmsatzEUR' THEN [Wert] ELSE 0 END) AS [Total_Umsatz]
    FROM [ERPDEV].[list_views].[V_LIST_LEHPE_MEASURES]
    WHERE [StoreName] = 'Rosenheim'
    GROUP BY [StoreName]
)
-- 4. Finale Berechnung
SELECT 
    P.[ProduktNr],
    P.[ProduktKategorie],
    P.[ProduktLinie],
    
    -- Werte
    P.[Prod_Umsatz] AS [Umsatz],
    P.[Prod_VarKosten] AS [VariableKosten],
    
    -- DB 1
    (P.[Prod_Umsatz] + P.[Prod_VarKosten]) AS [DB1],
    
    -- Anteilige Fixkosten (Umsatzanteil * Gesamtfixkosten)
    CAST(
        CASE WHEN T.[Total_Umsatz] <> 0 
             THEN (P.[Prod_Umsatz] / T.[Total_Umsatz]) * ISNULL(F.[Total_Fixkosten], 0)
             ELSE 0 
        END 
    AS DECIMAL(18,2)) AS [Fixkosten_Anteilig],
    
    -- DB 2 (DB1 - Anteilige Fixkosten)
    ((P.[Prod_Umsatz] + P.[Prod_VarKosten]) - 
        CASE WHEN T.[Total_Umsatz] <> 0 
             THEN (P.[Prod_Umsatz] / T.[Total_Umsatz]) * ISNULL(F.[Total_Fixkosten], 0)
             ELSE 0 
        END
    ) AS [DB2],
    
    -- DB 3 (Nicht berechnet)
    NULL AS [DB3]

FROM ProductData P
CROSS JOIN StoreFixCosts F
CROSS JOIN StoreTotalRevenue T
WHERE P.[Prod_Umsatz] <> 0;
GO

-- Testabfrage
SELECT * FROM [ins_views].[V_LIST_G14_DB_CALCULATION_PRODUCT_ROSENHEIM]
ORDER BY [ProduktKategorie], [ProduktNr];