# 📊 Benutzerhandbuch: DB Rechnung App

---

## Was ist diese App?

Die DB Rechnung App ist ein Dashboard zur Analyse der Profitabilität unserer Stores. Sie zeigt Ihnen auf einen Blick, wie viel Umsatz ein Store macht, welche Kosten anfallen und ob am Ende ein Gewinn oder Verlust herauskommt.

---

## Anmeldung

Wenn Sie die App öffnen, sehen Sie zunächst die Login-Seite.

1. Geben Sie Ihren **Benutzernamen** ein
2. Geben Sie Ihr **Passwort** ein
3. Klicken Sie auf **"Anmelden"**

Nach erfolgreicher Anmeldung gelangen Sie automatisch zum Dashboard.

Falls Sie Ihre Zugangsdaten vergessen haben, wenden Sie sich an Ihren Vorgesetzten.

---

## Berechtigungsstufen

Je nach Ihrer Rolle im Unternehmen haben Sie unterschiedliche Zugriffsrechte:

**Stufe 1 – Basis-Zugriff**
- Sie können nur den Store Rosenheim sehen
- Sie können nur Jahresdaten anzeigen

**Stufe 2 – Erweiterter Zugriff**
- Sie können Rosenheim und Freiburg sehen
- Sie können nach Jahr und Quartal filtern

**Stufe 3 – Vollzugriff**
- Sie können alle Stores sehen
- Sie können nach Jahr, Quartal und einzelnen Monaten filtern

Ihre aktuelle Berechtigungsstufe wird Ihnen oben links in der Seitenleiste angezeigt.

---

## Die Seitenleiste (links)

In der linken Seitenleiste finden Sie alle Einstellungsmöglichkeiten:

**Benutzerinformation**
Hier sehen Sie, mit welchem Benutzer Sie angemeldet sind und welche Berechtigungen Sie haben.

**Store auswählen**
Wählen Sie den Store, dessen Daten Sie analysieren möchten. Je nach Berechtigungsstufe stehen Ihnen ein oder mehrere Stores zur Verfügung.

**Zeitraum auswählen**
- Jahr: Wählen Sie das Geschäftsjahr
- Quartal: Grenzen Sie auf ein Quartal ein (Q1, Q2, Q3 oder Q4) – nur bei Stufe 2 und 3
- Monat: Wählen Sie einen einzelnen Monat – nur bei Stufe 3

**Daten aktualisieren**
Klicken Sie auf diesen Button, um die neuesten Daten aus dem System zu laden.

**Abmelden**
Klicken Sie hier, um sich von der App abzumelden.

---

## Die Übersicht (oben)

Ganz oben auf der Hauptseite sehen Sie drei wichtige Kennzahlen:

**Gesamtumsatz**
Die Summe aller Verkaufserlöse im gewählten Zeitraum.

**E3 Total Summe**
Das finale Ergebnis nach Abzug aller Kosten. Dies ist der tatsächliche Gewinn oder Verlust.

**Status**
- ✅ Profitabel – Der Store macht Gewinn
- ❌ Nicht profitabel – Der Store macht Verlust

---

## Die Haupttabelle: DB Rechnung nach Ebenen

Die große Tabelle zeigt Ihnen alle Details der Deckungsbeitragsrechnung. Die Daten sind in drei Ebenen unterteilt:

---

**E1 – Rohertrag**

Diese Ebene zeigt den Rohertrag, also die Differenz zwischen Verkaufserlösen und Einkaufskosten.

Folgende Kenngrößen werden verwendet:

- **UmsatzEUR** – Die Verkaufserlöse in Euro
- **TransferPriceEUR** – Der Einkaufspreis der verkauften Waren (als negativer Wert gespeichert)

**Berechnung E1 Total:**
UmsatzEUR + TransferPriceEUR = E1 Total (Rohertrag)

---

**E2 – Deckungsbeitrag 2**

Von E1 werden variable Kosten wie Provisionen und Rabatte abgezogen.

Folgende Kenngrößen werden verwendet:

- **Commission in EUR** – Provisionen für Verkäufer oder Partner
- **DiscountAufMaterialEUR** – Rabatte auf Materialien
- **DiscountAufMaterialKategorieEUR** – Rabatte auf Materialkategorien

**Berechnung E2 Total:**
E1 Total − Commission in EUR − DiscountAufMaterialEUR − DiscountAufMaterialKategorieEUR = E2 Total

---

**E3 – Deckungsbeitrag 3 (Endergebnis)**

Von E2 werden die Fixkosten abgezogen. Das Ergebnis zeigt den finalen Gewinn oder Verlust.

Folgende Kenngrößen werden verwendet:

- **Monthly Rent** – Die monatliche Miete für den Store
- **Monthly Salary** – Die Gehälter der Mitarbeiter
- **Monthly Social Costs** – Sozialabgaben und Nebenkosten
- **Marketing Campaign** – Kosten für Werbung und Marketing
- **Additional Procurement Costs** – Zusätzliche Beschaffungskosten

**Berechnung E3 Total:**
E2 Total − Monthly Rent − Monthly Salary − Monthly Social Costs − Marketing Campaign − Additional Procurement Costs = E3 Total (Gewinn/Verlust)

---

**Spalten der Tabelle**

Die Tabelle ist nach Produktlinien und Produktkategorien aufgeteilt. So können Sie sehen, welche Produkte am profitabelsten sind. Ganz rechts finden Sie immer die Gesamtsumme aller Produkte.

**Total-Zeilen**

Die grau hinterlegten Zeilen (E1 Total, E2 Total, E3 Total) zeigen die Zwischensummen der jeweiligen Ebene.

---

## Legende: Was bedeuten die Begriffe?

**UmsatzEUR** – Die Verkaufserlöse in Euro

**TransferPriceEUR** – Der Einkaufspreis der verkauften Waren

**Commission in EUR** – Provisionen, die an Verkäufer oder Partner gezahlt werden

**DiscountAufMaterialEUR** – Gewährte Rabatte auf Materialien

**DiscountAufMaterialKategorieEUR** – Gewährte Rabatte auf Materialkategorien

**Monthly Rent** – Die monatliche Miete für den Store

**Monthly Salary** – Die Gehälter der Mitarbeiter

**Monthly Social Costs** – Sozialabgaben und Nebenkosten für Mitarbeiter

**Marketing Campaign** – Kosten für Werbung und Marketing

**Additional Procurement Costs** – Zusätzliche Beschaffungskosten


## Tipps für die tägliche Arbeit

- Nutzen Sie den Jahresüberblick für strategische Entscheidungen
- Nutzen Sie die Quartalsansicht für saisonale Analysen
- Nutzen Sie die Monatsansicht für detaillierte Kostenkontrollen
- Die Spalte "Gesamt" rechts gibt Ihnen immer den Überblick über alle Produkte zusammen
- Die grau hinterlegten Total-Zeilen sind die wichtigsten Kennzahlen

---

## Bei Problemen

Wenn Sie Schwierigkeiten mit der Anmeldung haben oder die App nicht richtig funktioniert, wenden Sie sich bitte an Ihren Vorgesetzten oder die IT-Abteilung.

---

*Stand: Januar 2026*
