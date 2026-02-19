WITH Base AS (
    SELECT DISTINCT
        StoreName, Monat, Ebene, EPos, Kenngröße, ProduktKategorie, ProduktLinie, Wert
    FROM [ERPDEV].[list_views].[V_LIST_LEHPE_MEASURES]
)