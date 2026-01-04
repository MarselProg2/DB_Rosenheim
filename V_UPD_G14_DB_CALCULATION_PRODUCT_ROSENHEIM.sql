/* 
   V_UPD_G14_INPUT
   Ein View zum Aktualisieren von bestehenden Daten (Update).
   
   Damit ein Update funktioniert, müssen die Zeilen eindeutig identifizierbar sein.
   ACHTUNG: Ersetze 'DEINE_ECHTE_TABELLE' durch den echten Tabellennamen.
*/

CREATE OR ALTER VIEW [ins_views].[V_UPD_G14_INPUT] AS
SELECT 
    [StoreName],
    [Monat],
    [ProduktNr],
    [Kenngröße], 
    [Wert]
FROM [ERPDEV].[list_views].[V_LIST_LEHPE_MEASURES]; 
GO

/*
-- Trigger für Updates (Notwendig, falls der View nicht direkt aktualisierbar ist)
-- Dieser Trigger sorgt dafür, dass das UPDATE auf die richtige Tabelle umgeleitet wird.

CREATE TRIGGER TR_UPD_G14_INPUT
ON [ins_views].[V_UPD_G14_INPUT]
INSTEAD OF UPDATE
AS
BEGIN
    -- Wir aktualisieren die echte Tabelle basierend auf den Schlüsselfeldern
    UPDATE T
    SET T.Wert = I.Wert
    FROM [ERPDEV].[dbo].[DEINE_ECHTE_TABELLE] T
    INNER JOIN inserted I 
        ON T.StoreName = I.StoreName
        AND T.Monat = I.Monat
        AND T.ProduktNr = I.ProduktNr
        AND T.Kenngröße = I.Kenngröße;
END;
*/