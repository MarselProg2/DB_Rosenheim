/* 
   V_INS_G14_INPUT
   Ein einfacher View zum Erfassen von neuen Daten.
   
   ACHTUNG: Du musst 'DEINE_ECHTE_TABELLE' durch den Namen der Tabelle ersetzen,
   in der die Daten wirklich gespeichert sind!
*/

CREATE OR ALTER VIEW [ins_views].[V_INS_G14_INPUT] AS
SELECT 
    [StoreName],
    [Monat],
    [ProduktNr],
    [Kenngröße], -- z.B. 'UmsatzEUR' oder 'Monthly Rent'
    [Wert]
FROM [ERPDEV].[list_views].[V_LIST_LEHPE_MEASURES]; -- Hier eigentlich die Basistabelle!
GO

-- Trigger, um das Schreiben zu ermöglichen (Falls der View oben nicht direkt geht)
-- Das hier brauchst du nur, wenn der View oben nicht direkt beschreibbar ist.
/*
CREATE TRIGGER TR_INS_G14_INPUT
ON [ins_views].[V_INS_G14_INPUT]
INSTEAD OF INSERT
AS
BEGIN
    INSERT INTO [ERPDEV].[dbo].[DEINE_ECHTE_TABELLE] (StoreName, Monat, ProduktNr, Kenngröße, Wert)
    SELECT StoreName, Monat, ProduktNr, Kenngröße, Wert
    FROM inserted;
END;
*/