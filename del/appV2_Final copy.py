
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
def load_final_table_from_db(store_name: str):
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};UID={DB_CONFIG['user']};PWD={DB_CONFIG['password']};"
    )
    conn = pyodbc.connect(conn_str)

    # Daten kommen aus der SQL-View: [list_views].[G14_Gesamt_DB_SCHEMA]
    query = """
SELECT
    [StoreName],
    [Monat],
    [Ebene],
    [EPos],
    [Kenngröße],
    [ProduktKategorie],
    [ProduktLinie],
    [Wert],
    [E1_Total],
    [E2_Total],
    [E3_Total],
    [Gesamt_Total]
FROM [list_views].[G14_Gesamt_DB_SCHEMA]
WHERE [StoreName] = ?
ORDER BY
    [Monat],
    [Ebene],
    [EPos],
    [Kenngröße],
    [ProduktKategorie],
    [ProduktLinie];
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
    out['Quartal'] = 'Q' + out['Monat_dt'].dt.quarter.astype(str)
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


def _pivot_for_kenngroesse(df_filtered: pd.DataFrame, kenngroesse_norm: str) -> pd.DataFrame:
    """Erzeugt eine 1-Zeilen-Pivot-Tabelle (ProduktLinie/ProduktKategorie) für eine Kenngröße."""
    needed = {'Kenngröße', 'ProduktLinie', 'ProduktKategorie', 'Wert'}
    if df_filtered.empty or not needed.issubset(df_filtered.columns):
        return pd.DataFrame()

    tmp = df_filtered.copy()
    tmp['Wert'] = pd.to_numeric(tmp['Wert'], errors='coerce').fillna(0)
    tmp['_KenngroesseNorm'] = tmp['Kenngröße'].astype(str).str.strip().str.lower()

    tmp = tmp[tmp['_KenngroesseNorm'] == kenngroesse_norm]
    if tmp.empty:
        return pd.DataFrame()

    g = (
        tmp.groupby(['ProduktLinie', 'ProduktKategorie'], as_index=False)['Wert']
        .sum()
    )

    pivot = g.pivot_table(
        index=[],
        columns=['ProduktLinie', 'ProduktKategorie'],
        values='Wert',
        aggfunc='sum'
    )
    pivot.index = ['_row']
    pivot[('Summen', 'Gesamt')] = pivot.sum(axis=1)
    pivot = pivot.sort_index(axis=1)
    return pivot


def compute_deckungsbeitraege(df_filtered: pd.DataFrame) -> dict:
    """Berechnet DB-Logik spaltenweise (nicht zeilenweise).

    - E1 Total = UmsatzEUR + TransferPriceEUR
    - E2 Total = E1 Total - Commission in EUR
    - E3 Total = E2 Total - Summe(E3 Kosten)
    """
    umsatz = _pivot_for_kenngroesse(df_filtered, 'umsatzeur')
    transfer = _pivot_for_kenngroesse(df_filtered, 'transferpriceeur')
    commission = _pivot_for_kenngroesse(df_filtered, 'commission in eur')

    e3_cost_norms = [
        'additional procurement costs',
        'marketing campaign',
        'monthly rent',
        'monthly salary',
        'monthly social costs',
    ]
    e3_cost_parts = [p for p in (_pivot_for_kenngroesse(df_filtered, k) for k in e3_cost_norms) if not p.empty]

    def _align(p: pd.DataFrame, cols: pd.Index) -> pd.DataFrame:
        if p.empty:
            return pd.DataFrame(index=['_row'], columns=cols).fillna(0)
        return p.reindex(columns=cols).fillna(0)

    # Gemeinsame Spaltenbasis
    all_cols = pd.Index([])
    for p in [umsatz, transfer, commission] + e3_cost_parts:
        if not p.empty:
            all_cols = all_cols.union(p.columns)
    if len(all_cols) == 0:
        return {
            'e1_total': pd.DataFrame(),
            'e2_total': pd.DataFrame(),
            'e3_total': pd.DataFrame(),
            'missing': ['umsatzeur', 'transferpriceeur', 'commission in eur']
        }

    umsatz_a = _align(umsatz, all_cols)
    transfer_a = _align(transfer, all_cols)
    commission_a = _align(commission, all_cols)

    e3_cost_a = pd.DataFrame(index=['_row'], columns=all_cols).fillna(0)
    for p in e3_cost_parts:
        e3_cost_a = e3_cost_a.add(_align(p, all_cols), fill_value=0)

    # TransferPriceEUR ist in vielen Datenquellen bereits als negativer Wert hinterlegt.
    # Daher hier bewusst PLUS, um kein "Minus minus" zu erzeugen.
    e1_total = umsatz_a.add(transfer_a, fill_value=0)
    e2_total = e1_total.sub(commission_a, fill_value=0)
    e3_total = e2_total.sub(e3_cost_a, fill_value=0)

    # Summen-Spalte sicherstellen (falls in all_cols nicht enthalten)
    if ('Summen', 'Gesamt') not in all_cols:
        for df_ in (e1_total, e2_total, e3_total):
            df_[('Summen', 'Gesamt')] = df_.sum(axis=1)

    missing = []
    if umsatz.empty:
        missing.append('UmsatzEUR')
    if transfer.empty:
        missing.append('TransferPriceEUR')
    if commission.empty:
        missing.append('Commission in EUR')

    return {
        'e1_total': e1_total,
        'e2_total': e2_total,
        'e3_total': e3_total,
        'missing': missing,
    }


def build_ebene_table(df_filtered: pd.DataFrame, ebene: str) -> pd.DataFrame:
    """Erstellt die Tabelle für eine Ebene basierend auf den geladenen Daten.

    Reihenfolge: nach EPos (numerisch), dann Kenngröße. Am Ende kommt '<Ebene> Total'.
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

    rows = []
    row_order: list[str] = []

    df_rows = (
        df_e[['EPos', 'Kenngröße', '_KenngroesseNorm']]
        .drop_duplicates()
        .copy()
    )
    df_rows['_EPosNum'] = pd.to_numeric(df_rows['EPos'], errors='coerce')
    df_rows = df_rows.sort_values(['_EPosNum', 'Kenngröße'], kind='stable')

    for _, m in df_rows.iterrows():
        k_label = str(m['Kenngröße'])
        k_norm = str(m['_KenngroesseNorm'])

        # Keine EPos-Nummern im Frontend anzeigen
        row_label = f"{k_label}"
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
    for ebene in ebenen:
        part = build_ebene_table(df_filtered, ebene)
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

    # DB-Logik: Total-Zeilen spaltenweise berechnen
    db = compute_deckungsbeitraege(df_filtered)
    if not db['e1_total'].empty:
        e1_idx = 'E1 | E1 Total'
        if e1_idx in combined.index:
            combined.loc[e1_idx] = db['e1_total'].reindex(columns=combined.columns).iloc[0].fillna(0).values

    if not db['e2_total'].empty:
        e2_idx = 'E2 | E2 Total'
        if e2_idx in combined.index:
            combined.loc[e2_idx] = db['e2_total'].reindex(columns=combined.columns).iloc[0].fillna(0).values

    if not db['e3_total'].empty:
        e3_idx = 'E3 | E3 Total'
        if e3_idx in combined.index:
            combined.loc[e3_idx] = db['e3_total'].reindex(columns=combined.columns).iloc[0].fillna(0).values

    return combined


st.set_page_config(page_title=" 11DB Rosenheim", layout="wide")
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
        present_quarters = sorted(df_jahr['Monat_dt'].dt.quarter.dropna().unique().tolist())
        quartal_options = ['Alle'] + [f"Q{q}" for q in present_quarters]
        selected_quartal = st.selectbox("Quartal", quartal_options)

        df_scope = df_jahr
        if selected_quartal != 'Alle':
            df_scope = df_scope[df_scope['Quartal'] == selected_quartal]

        month_map = (
            df_scope[['Monat', 'Monat_dt']]
            .drop_duplicates()
            .sort_values('Monat_dt', kind='stable')
        )
        monat_options = ['Alle'] + month_map['Monat'].astype(str).tolist()
        selected_monat = st.selectbox("Monat", monat_options)

    df_filtered = df[df['Jahr'] == selected_jahr].copy()
    if selected_quartal != 'Alle':
        df_filtered = df_filtered[df_filtered['Quartal'] == selected_quartal]
    if selected_monat != 'Alle':
        df_filtered = df_filtered[df_filtered['Monat'] == selected_monat]

    if df_filtered.empty:
        st.info("Für den ausgewählten Zeitraum gibt es keine Daten.")
        st.stop()

    if selected_monat != 'Alle':
        zeitraum_titel = f"{selected_monat}"
    elif selected_quartal != 'Alle':
        zeitraum_titel = f"{selected_quartal} {selected_jahr}"
    else:
        zeitraum_titel = f"Gesamtjahr {selected_jahr}"

    st.subheader(f"Übersicht: {store_name} – {zeitraum_titel}")

    # Deckungsbeiträge (spaltenweise, über alle Produkte/Linien/Kategorien)
    db = compute_deckungsbeitraege(df_filtered)
    sum_total = 0.0
    if not db['e3_total'].empty and ('Summen', 'Gesamt') in db['e3_total'].columns:
        sum_total = float(db['e3_total'].iloc[0][('Summen', 'Gesamt')])

    if db.get('missing'):
        st.info(
            "Hinweis: Für die Deckungsbeitrags-Rechnung fehlen Kenngrößen: "
            + ", ".join(db['missing'])
        )

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
    st.subheader("DB Rechnung nach Ebenen")

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

        # Einmalig auflisten (Reihenfolge wie in den Daten), nur Name + Beschreibung
        k_series = df_filtered['Kenngröße'].dropna().astype(str).str.strip()
        seen = set()
        for name in k_series.tolist():
            if name.lower() in excluded:
                continue
            if name in seen:
                continue
            seen.add(name)
            desc = descriptions.get(name, 'Kenngröße aus der Datenquelle (Beschreibung nicht hinterlegt).')
            st.markdown(f"- **{name}**: {desc}")

except Exception as e:
    st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
