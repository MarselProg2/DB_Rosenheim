
import streamlit as st
import pandas as pd
import pyodbc
import json
from pathlib import Path

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
def load_final_table_from_db(store_name: str):
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};UID={DB_CONFIG['user']};PWD={DB_CONFIG['password']};"
    )
    conn = pyodbc.connect(conn_str)

    # Query aus Final.sql (Base + Totals), aber mit Parameter statt festem Store.
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


def format_eur(val: float) -> str:
    try:
        if pd.isna(val):
            return "-"
        return f"{val:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Monat_dt'] = pd.to_datetime(out['Monat'])
    out['Jahr'] = out['Monat_dt'].dt.year
    return out


def calc_gesamtumsatz(df_filtered: pd.DataFrame) -> tuple[float, list[str]]:
    """Ermittelt Gesamtumsatz aus der Final-Tabelle über Kenngröße.

    Returns:
        (umsatz_summe, gefundene_kenngroessen)
    """
    if 'Kenngröße' not in df_filtered.columns or 'Wert' not in df_filtered.columns:
        return 0.0, []

    # Typische Namen aus LEHPE-Measures / Views
    candidates = {
        'umsatzeur',
        'umsatz',
        'revenue',
        'sales',
        'saleseur',
        'totalrevenue',
    }

    k = (
        df_filtered['Kenngröße']
        .astype(str)
        .str.strip()
        .str.lower()
    )
    mask = k.isin(candidates)

    found = sorted(set(df_filtered.loc[mask, 'Kenngröße'].astype(str).unique().tolist()))
    umsatz = float(df_filtered.loc[mask, 'Wert'].sum()) if mask.any() else 0.0
    return umsatz, found


@st.cache_data(ttl=600)
def load_kenngroessen_mapping() -> pd.DataFrame:
    """Lädt die Zuordnung Ebene/EPos/Kenngröße aus Kenngrößen.json (lokal im Projekt)."""
    mapping_path = Path(__file__).with_name('Kenngrößen.json')
    with mapping_path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    df_map = pd.DataFrame(raw)
    # Normalisieren für robustes Matching
    for col in ['Ebene', 'EPos', 'Kenngröße']:
        if col not in df_map.columns:
            raise ValueError(f"Kenngrößen.json fehlt Spalte: {col}")
    df_map['Ebene'] = df_map['Ebene'].astype(str).str.strip()
    df_map['EPos'] = df_map['EPos'].astype(str).str.strip()
    df_map['Kenngröße'] = df_map['Kenngröße'].astype(str).str.strip()
    df_map['_KenngroesseNorm'] = df_map['Kenngröße'].str.lower()
    df_map['_EPosNum'] = pd.to_numeric(df_map['EPos'], errors='coerce')
    return df_map


def build_ebene_table(df_filtered: pd.DataFrame, ebene: str, df_map: pd.DataFrame) -> pd.DataFrame:
    """Erstellt die Tabelle für eine Ebene basierend auf Kenngrößen.json.

    Pro Mapping-Eintrag (Ebene/EPos/Kenngröße) wird genau eine Zeile erzeugt.
    Am Ende kommt '<Ebene> Total'.
    """
    needed = {'Ebene', 'EPos', 'Kenngröße', 'ProduktLinie', 'ProduktKategorie', 'Wert'}
    if df_filtered.empty or not needed.issubset(df_filtered.columns):
        return pd.DataFrame()

    df_e = df_filtered[df_filtered['Ebene'] == ebene].copy()
    if df_e.empty:
        return pd.DataFrame()

    df_e['Wert'] = pd.to_numeric(df_e['Wert'], errors='coerce').fillna(0)

    df_e['_KenngroesseNorm'] = df_e['Kenngröße'].astype(str).str.strip().str.lower()

    # Kenngrößen ausblenden (z.B. nicht-monetäre/technische Kennzahlen)
    excluded = {'salespriceeur', 'salesamount', 'salesprice', 'sales amount', 'sales price'}
    df_e = df_e[~df_e['_KenngroesseNorm'].isin(excluded)].copy()
    if df_e.empty:
        return pd.DataFrame()

    df_map_e = df_map[df_map['Ebene'] == str(ebene)].copy()
    if df_map_e.empty:
        return pd.DataFrame()

    # Gleiche Kenngrößen auch aus dem Mapping entfernen, damit keine leeren Zeilen entstehen
    df_map_e = df_map_e[~df_map_e['_KenngroesseNorm'].isin(excluded)].copy()
    if df_map_e.empty:
        return pd.DataFrame()

    # Sortierung: EPos numerisch, dann in JSON-Reihenfolge (df_map_e ist bereits in Datei-Reihenfolge)
    df_map_e = df_map_e.sort_values(['_EPosNum'], kind='stable')

    rows = []
    row_order: list[str] = []

    for _, m in df_map_e.iterrows():
        epos = str(m['EPos'])
        k_label = str(m['Kenngröße'])
        k_norm = str(m['_KenngroesseNorm'])

        row_label = f"{epos}. {k_label}"
        row_order.append(row_label)

        tmp = (
            df_e[df_e['_KenngroesseNorm'] == k_norm]
            .groupby(['ProduktLinie', 'ProduktKategorie'], as_index=False)['Wert']
            .sum()
        )
        tmp['RowLabel'] = row_label
        rows.append(tmp)

    # Total-Zeile für die Ebene (alle Kenngrößen dieser Ebene)
    total_label = f"{ebene} Total"
    row_order.append(total_label)
    tmp_total = (
        df_e.groupby(['ProduktLinie', 'ProduktKategorie'], as_index=False)['Wert']
        .sum()
    )
    tmp_total['RowLabel'] = total_label
    rows.append(tmp_total)

    long_df = pd.concat(rows, ignore_index=True)

    pivot = long_df.pivot_table(
        index=['RowLabel'],
        columns=['ProduktLinie', 'ProduktKategorie'],
        values='Wert',
        aggfunc='sum'
    )

    pivot[('Summen', 'Gesamt')] = pivot.sum(axis=1)
    pivot = pivot.sort_index(axis=1)

    # Reihenfolge erzwingen
    pivot = pivot.reindex(row_order).fillna(0)
    pivot.index.name = None
    return pivot


def build_all_ebenen_table(df_filtered: pd.DataFrame, ebenen: list[str]) -> pd.DataFrame:
    """Kombiniert E1..E3 zu einer einzigen Tabelle (Zeilen untereinander)."""
    parts: list[pd.DataFrame] = []
    df_map = load_kenngroessen_mapping()
    for ebene in ebenen:
        part = build_ebene_table(df_filtered, ebene, df_map)
        if not part.empty:
            # Prefix, damit die Zeilen eindeutig sind (z.B. "E1 | 1. UmsatzEUR")
            part = part.copy()
            part.index = [f"{ebene} | {idx}" for idx in part.index]
            parts.append(part)

    if not parts:
        return pd.DataFrame()

    # Spalten unionieren und vertikal stapeln
    all_cols = pd.Index([])
    for p in parts:
        all_cols = all_cols.union(p.columns)
    parts = [p.reindex(columns=all_cols) for p in parts]
    combined = pd.concat(parts, axis=0)

    # Grand Total: Summe der Ebene-Total-Zeilen (vermeidet Double-Counting der Detailzeilen)
    total_row_idxs = [idx for idx in combined.index if str(idx).endswith('Total')]
    if total_row_idxs:
        combined.loc['Grand Total'] = combined.loc[total_row_idxs].sum(axis=0)

    return combined


st.set_page_config(page_title="Final Table", layout="wide")
st.title("Final Table – Kosten & Totals")

with st.sidebar:
    st.header("Filter")
    store_name = st.text_input("StoreName", value="Rosenheim")
    if st.button("🔄 Daten aktualisieren"):
        load_final_table_from_db.clear()
        st.rerun()


try:
    df_raw = load_final_table_from_db(store_name.strip() or "Rosenheim")

    if df_raw.empty:
        st.warning("Keine Daten geladen. Bitte StoreName/Verbindung prüfen.")
        st.stop()

    required_cols = {
        'StoreName', 'Monat', 'Ebene', 'EPos', 'Kenngröße',
        'ProduktKategorie', 'ProduktLinie', 'Wert',
        'E1_Total', 'E2_Total', 'E3_Total', 'Gesamt_Total'
    }
    missing_cols = sorted(required_cols.difference(df_raw.columns))
    if missing_cols:
        st.error(
            "Erwartete Spalten fehlen: " + ", ".join(missing_cols)
            + "\n\nVerfügbare Spalten:\n- "
            + "\n- ".join(map(str, df_raw.columns))
        )
        st.stop()

    df = add_time_columns(df_raw)

    with st.sidebar:
        jahre = sorted(df['Jahr'].unique(), reverse=True)
        selected_jahr = st.selectbox("Jahr", jahre)

        df_jahr = df[df['Jahr'] == selected_jahr]
        monat_options = ['Alle'] + sorted(df_jahr['Monat'].unique().tolist())
        selected_monat = st.selectbox("Monat", monat_options)

    df_filtered = df[df['Jahr'] == selected_jahr].copy()
    if selected_monat != 'Alle':
        df_filtered = df_filtered[df_filtered['Monat'] == selected_monat]

    if df_filtered.empty:
        st.info("Für den ausgewählten Zeitraum gibt es keine Daten.")
        st.stop()

    if selected_monat != 'Alle':
        zeitraum_titel = f"{selected_monat}"
    else:
        zeitraum_titel = f"Gesamtjahr {selected_jahr}"

    st.subheader(f"Übersicht: {store_name} – {zeitraum_titel}")

    # Monats-Totals einmal je Monat ziehen (Totals sind je Zeile wiederholt)
    totals_cols = ['Monat', 'E1_Total', 'E2_Total', 'E3_Total', 'Gesamt_Total']
    df_monthly_totals = (
        df_filtered[totals_cols]
        .drop_duplicates()
        .sort_values('Monat')
    )

    # Zeitraum-Summe über Monate
    sum_total = df_monthly_totals['Gesamt_Total'].sum()

    profitabel = sum_total > 0
    status_text = "✅ Profitabel" if profitabel else "❌ Nicht profitabel"

    # Gesamtumsatz (aus Kenngröße in den Rohdaten)
    gesamtumsatz, umsatz_labels = calc_gesamtumsatz(df_filtered)
    if gesamtumsatz == 0 and not umsatz_labels:
        st.info(
            "Hinweis: Gesamtumsatz konnte nicht eindeutig gefunden werden. "
            "(Es wurde nach Kenngröße = UmsatzEUR/Umsatz/Revenue etc. gesucht.)"
        )

    c0, c1, c2 = st.columns(3)
    c0.metric("Gesamtumsatz", format_eur(gesamtumsatz))
    c1.metric("Summe Gesamt", format_eur(sum_total))
    c2.metric("Status", status_text)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # EBENEN-STRUKTUR (wie app.py-Layout, aber mit Ebenen statt DB)
    # -------------------------------------------------------------------------
    st.subheader("Detailanalyse nach Ebenen")

    def format_german(val):
        if pd.isna(val) or val == 0:
            return "-"
        return "{:,.2f} €".format(val).replace(",", "X").replace(".", ",").replace("X", ".")

    def style_total_rows(row):
        idx = str(row.name)
        if idx.endswith("Total"):
            return ['background-color: #f2f2f2; font-weight: bold; border-top: 1px solid #aaa; color: black;'] * len(row)
        return [''] * len(row)

    present_ebenen = [e for e in ['E1', 'E2', 'E3'] if e in set(df_filtered['Ebene'].dropna().astype(str))]
    if not present_ebenen:
        st.info("Keine Ebenen (E1/E2/E3) in den Daten gefunden.")
    else:
        df_all = build_all_ebenen_table(df_filtered, present_ebenen)
        if df_all.empty:
            st.info("Keine Detailanalyse-Daten für Ebenen.")
        else:
            st.dataframe(
                df_all.style
                .format(format_german)
                .apply(style_total_rows, axis=1),
                use_container_width=True,
                height=520
            )

    st.markdown("---")

    with st.expander("Legende: Kenngrößen"):
        df_map = load_kenngroessen_mapping()

        excluded = {'salespriceeur', 'salesamount', 'salesprice', 'sales amount', 'sales price'}

        # Kurze, neutrale Beschreibungen (bei Unklarheit bewusst vorsichtig formuliert)
        descriptions = {
            'SalesPriceEUR': 'Verkaufspreis in EUR (Sales Price).',
            'SalesAmount': 'Verkaufsmenge / Anzahl (Sales Amount).',
            'UmsatzEUR': 'Umsatz in EUR.',
            'TransferPriceEUR': 'Transferpreis / Einkaufspreis in EUR.',
            'Commission in EUR': 'Provision / Commission in EUR.',
            'Additional Procurement Costs': 'Zusätzliche Beschaffungskosten.',
            'Marketing Campaign': 'Marketing-Kampagne (Kosten).',
            'Monthly Rent': 'Monatsmiete.',
            'Monthly Salary': 'Monatsgehälter.',
            'Monthly Social Costs': 'Monatliche Sozialkosten.',
        }

        # Einmalig auflisten (in Datei-Reihenfolge), nur Name + Beschreibung
        seen = set()
        for _, r in df_map.iterrows():
            name = str(r['Kenngröße']).strip()
            if name.lower() in excluded:
                continue
            if name in seen:
                continue
            seen.add(name)
            desc = descriptions.get(name, 'Kenngröße aus der Datenquelle (Beschreibung nicht hinterlegt).')
            st.markdown(f"- **{name}**: {desc}")

except Exception as e:
    st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
