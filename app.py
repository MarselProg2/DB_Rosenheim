import streamlit as st
import pandas as pd
import pyodbc

# -----------------------------------------------------------------------------
# 1. KONFIGURATION & VERBINDUNG
# -----------------------------------------------------------------------------
DB_CONFIG = {
    "server": "edu.hdm-server.eu",
    "database": "ERPDEV",
    "user": "w25s252",
    "password": "202860",
    "driver": "{SQL Server}"
}

@st.cache_data(ttl=600)
def load_data_from_db():
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};UID={DB_CONFIG['user']};PWD={DB_CONFIG['password']};"
    )
    conn = pyodbc.connect(conn_str)
    # View laden
    query = "SELECT * FROM [ins_views].[V_LIST_G14_DB_DASHBOARD_PIVOT]"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


@st.cache_data(ttl=600)
def load_check_costs_from_db(store_name: str = "Rosenheim"):
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};UID={DB_CONFIG['user']};PWD={DB_CONFIG['password']};"
    )
    conn = pyodbc.connect(conn_str)

    # Dein SQL aus check_costs.sql, aber parametrisiert (statt hart 'Rosenheim').
    query = """
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
    WHERE [StoreName] = ?
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
ORDER BY
    B.[Monat],
    B.[Ebene],
    B.[EPos],
    B.[Kenngröße],
    B.[ProduktKategorie],
    B.[ProduktLinie];
"""

    df = pd.read_sql(query, conn, params=[store_name])
    conn.close()
    return df

# -----------------------------------------------------------------------------
# 2. UI SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DB Rechnung G14", layout="wide")
st.title("📋 Deckungsbeitragsrechnung (Detailliert)")
st.markdown("<style>.stDataFrame { font-family: 'Arial', sans-serif; }</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOGIK
# -----------------------------------------------------------------------------
try:
    df_raw = load_data_from_db()

    # Zusätzlich: Kosten-Check Daten (dein SQL)
    df_check = load_check_costs_from_db("Rosenheim")

    if df_raw.empty:
        st.warning("Keine Daten aus der Datenbank geladen. Bitte Verbindung prüfen.")
        st.stop()

    # --- ANREICHERUNG DER DATEN FÜR FILTER ---
    df_raw['Monat_dt'] = pd.to_datetime(df_raw['Monat'])
    df_raw['Jahr'] = df_raw['Monat_dt'].dt.year
    df_raw['Quartal'] = df_raw['Monat_dt'].dt.quarter

    # -----------------------------------------------------------------------------
    # UI - SIDEBAR FILTER
    # -----------------------------------------------------------------------------
    with st.sidebar:
        st.header("🗓 Filter")
        
        # --- JAHR FILTER ---
        jahre = sorted(df_raw['Jahr'].unique(), reverse=True)
        selected_jahr = st.selectbox("Jahr:", jahre)

        # Temp-Frame für dynamische Filter-Optionen
        df_jahr_filtered = df_raw[df_raw['Jahr'] == selected_jahr]

        # --- QUARTAL FILTER ---
        quartale_options = ['Alle'] + sorted(df_jahr_filtered['Quartal'].unique().tolist())
        selected_quartal = st.selectbox("Quartal:", quartale_options)
        
        # Temp-Frame für dynamische Monats-Optionen
        df_quartal_filtered = df_jahr_filtered
        if selected_quartal != 'Alle':
            df_quartal_filtered = df_jahr_filtered[df_jahr_filtered['Quartal'] == selected_quartal]

        # --- MONAT FILTER ---
        monat_options = ['Alle'] + sorted(df_quartal_filtered['Monat'].unique().tolist())
        selected_monat = st.selectbox("Monat:", monat_options)


        if st.button("🔄 Daten aktualisieren"):
            load_data_from_db.clear()
            load_check_costs_from_db.clear()
            st.rerun()

    # -----------------------------------------------------------------------------
    # DATENFILTERUNG
    # -----------------------------------------------------------------------------
    # Start with year filter
    df_filtered = df_raw[df_raw['Jahr'] == selected_jahr].copy()
    
    # Apply quarter filter
    if selected_quartal != 'Alle':
        df_filtered = df_filtered[df_filtered['Quartal'] == selected_quartal]

    # Apply month filter
    if selected_monat != 'Alle':
        df_filtered = df_filtered[df_filtered['Monat'] == selected_monat]

    if df_filtered.empty:
        st.info("Für den ausgewählten Zeitraum gibt es keine Daten.")
        st.stop()

    # -------------------------------------------------------------------------
    # ROBUSTE SPALTENAUSWAHL (verhindert KeyError wie 'Umsatz')
    # -------------------------------------------------------------------------
    def pick_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
        cols = list(frame.columns)
        # 1) exakter Treffer
        for name in candidates:
            if name in cols:
                return name
        # 2) case-insensitive / whitespace tolerant
        normalized = {str(c).strip().lower(): c for c in cols}
        for name in candidates:
            key = str(name).strip().lower()
            if key in normalized:
                return normalized[key]
        return None

    # -------------------------------------------------------------------------
    # KOSTEN-CHECK DATEN FILTERUNG (dein SQL)
    # -------------------------------------------------------------------------
    df_check_filtered = pd.DataFrame()
    if not df_check.empty and 'Monat' in df_check.columns:
        df_check = df_check.copy()
        df_check['Monat_dt'] = pd.to_datetime(df_check['Monat'])
        df_check['Jahr'] = df_check['Monat_dt'].dt.year
        df_check['Quartal'] = df_check['Monat_dt'].dt.quarter

        df_check_filtered = df_check[df_check['Jahr'] == selected_jahr].copy()
        if selected_quartal != 'Alle':
            df_check_filtered = df_check_filtered[df_check_filtered['Quartal'] == selected_quartal]
        if selected_monat != 'Alle':
            df_check_filtered = df_check_filtered[df_check_filtered['Monat'] == selected_monat]

    # -------------------------------------------------------------------------
    # KENNZAHLEN BERECHNEN (NEU)
    # -------------------------------------------------------------------------
    umsatz_col = pick_column(df_filtered, [
        'Umsatz',
        'UmsatzEUR',
        'Revenue',
        'TotalRevenue',
    ])
    db4_col = pick_column(df_filtered, [
        'DB4',
        'Betriebsergebnis',
        'OperatingResult',
    ])

    if umsatz_col is None or db4_col is None:
        missing = []
        if umsatz_col is None:
            missing.append("Umsatz")
        if db4_col is None:
            missing.append("DB4")
        st.warning(
            "In der geladenen View fehlen erwartete Spalten: "
            + ", ".join(missing)
            + ".\n\nVerfügbare Spalten:\n- "
            + "\n- ".join(map(str, df_raw.columns))
        )

    gesamtumsatz = df_filtered[umsatz_col].sum() if umsatz_col else 0
    betriebsergebnis = df_filtered[db4_col].sum() if db4_col else 0
    profitabilitaet_status = (
        "✅ Profitabel" if db4_col and betriebsergebnis > 0 else
        ("❌ Nicht Profitabel" if db4_col else "ℹ️ Status unbekannt")
    )

    # Titel für den Berichtszeitraum
    if selected_monat != 'Alle':
        zeitraum_titel = f"{selected_monat}"
    elif selected_quartal != 'Alle':
        zeitraum_titel = f"Q{selected_quartal} {selected_jahr}"
    else:
        zeitraum_titel = f"Gesamtjahr {selected_jahr}"
    
    st.subheader(f"Übersicht für: {zeitraum_titel}")

    # Darstellung der Kennzahlen in Spalten
    col1, col2, col3 = st.columns(3)
    col1.metric("Gesamtumsatz", f"{gesamtumsatz:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    col2.metric("Betriebsergebnis (DB4)", f"{betriebsergebnis:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    col3.metric("Status", profitabilitaet_status)
    st.markdown("---") # Trennlinie

    # -------------------------------------------------------------------------
    # BEREINIGUNG & TRANSFORMATION
    # -------------------------------------------------------------------------
    
    # BEREINIGUNG
    if 'ProduktKategorie' in df_filtered.columns:
        df_filtered['ProduktKategorie'] = df_filtered['ProduktKategorie'].astype(str).str.replace('Bio: ', '').str.replace('E-Bike: ', '')

    if 'ProduktLinie' not in df_filtered.columns:
         df_filtered['ProduktLinie'] = df_filtered['ProduktKategorie'].apply(lambda x: 'E-Bike' if 'E-Bike' in str(x) else 'Bio Bike')

    # Dummy-Spalte für reine Text-Zeile hinzufügen
    df_filtered['Produktgruppennennkosten'] = 0

    # TRANSFORMATION
    row_config = {
        'Umsatz':           (1, '01. Umsatz (Verkaufspreis)'),
        'VariableKosten':   (2, '(-) Var. Kosten (Transferpr., Prov., Rabatte)'),
        'DB1':              (3, '= DB I'),
        'Marketing':        (4, '(-) Produktebene Marketing (Kampagnen)'),
        'DB2':              (5, '= DB II'),
        'Produktgruppennennkosten': (5.1, 'Produktkategorie Marketing'),
        'DB3':              (6, '= DB III'), 
        'Fixkosten_Anteil': (7, '(-) Filialkosten (Miete, Personal, Sozial)'),
        'DB4':              (8, '= DB IV (Betriebsergebnis)')
    }

    valid_vars = [col for col in row_config.keys() if col in df_filtered.columns]

    if not valid_vars:
        st.error(
            "Für die Detailanalyse fehlen alle erwarteten Kennzahlen-Spalten "
            f"({', '.join(row_config.keys())}).\n\n"
            "Verfügbare Spalten:\n- "
            + "\n- ".join(map(str, df_filtered.columns))
        )
        st.stop()

    df_melted = df_filtered.melt(
        id_vars=['ProduktLinie', 'ProduktKategorie'], 
        value_vars=valid_vars, 
        var_name='Kennzahl_Raw', 
        value_name='Wert'
    )

    df_melted['SortOrder'] = df_melted['Kennzahl_Raw'].map(lambda x: row_config[x][0])
    df_melted['ZeilenName'] = df_melted['Kennzahl_Raw'].map(lambda x: row_config[x][1])

    df_pivot = df_melted.pivot_table(
        index=['SortOrder', 'ZeilenName'], 
        columns=['ProduktLinie', 'ProduktKategorie'], 
        values='Wert', 
        aggfunc='sum'
    ).reset_index()

    df_final = df_pivot.set_index('ZeilenName')
    df_final = df_final.sort_values('SortOrder')
    df_final = df_final.drop(columns=['SortOrder'])

    df_final[('Summen', 'Gesamt')] = df_final.sum(axis=1)
    df_final = df_final.sort_index(axis=1)
    df_final.index.name = None

    # -------------------------------------------------------------------------
    # DARSTELLUNG
    # -------------------------------------------------------------------------
    
    # Formatierungsfunktionen aus dem Originalskript
    def format_german(val):
        if pd.isna(val) or val == 0: return "-"
        return "{:,.2f} €".format(val).replace(",", "X").replace(".", ",").replace("X", ".")

    def style_rows(row):
        idx = str(row.name)
        if "DB" in idx or "Ergebnis" in idx:
            return ['background-color: #f2f2f2; font-weight: bold; border-top: 1px solid #aaa; color: black;'] * len(row)
        if "Umsatz" in idx:
             return ['background-color: #e6e6e6; font-weight: bold; color: black;'] * len(row)
        return [''] * len(row)

    st.subheader(f"Detailanalyse für: {zeitraum_titel}")
    
    with st.expander("ℹ Details zu den Kostensteinen (Definition)"):
        st.markdown("""
        * Variable Kosten: Enthält TransferPriceEUR (Einkauf), Commission (Provision) und Discounts.
        * Filialkosten: Enthält Monthly Rent (Miete), Monthly Salary (Gehalt) und Social Costs.
        * Verteilschlüssel: Filialkosten werden basierend auf dem Umsatzanteil umgelegt.
        """)

    st.dataframe(
        df_final.style
        .format(format_german)
        .apply(style_rows, axis=1),
        use_container_width=True,
        height=600
    )

    # -------------------------------------------------------------------------
    # KOSTEN-CHECK DARSTELLUNG (dein SQL)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Kosten-Check (Rohdaten + Monats-Totals)")

    if df_check.empty:
        st.info("Keine Kosten-Check Daten geladen (Query check_costs.sql).")
    elif df_check_filtered.empty:
        st.info("Für den ausgewählten Zeitraum gibt es keine Kosten-Check Daten.")
    else:
        # Monats-Totals (einmal je Monat)
        totals_cols = ['Monat', 'E1_Total', 'E2_Total', 'E3_Total', 'Gesamt_Total']
        df_monthly_totals = (
            df_check_filtered[totals_cols]
            .drop_duplicates()
            .sort_values('Monat')
        )

        st.dataframe(
            df_monthly_totals,
            use_container_width=True,
            height=220
        )

        st.dataframe(
            df_check_filtered.drop(columns=['Monat_dt', 'Jahr', 'Quartal'], errors='ignore'),
            use_container_width=True,
            height=420
        )

except Exception as e:
    st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")