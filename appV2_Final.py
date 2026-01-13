
import streamlit as st
import pandas as pd
import pymssql
import re
import os
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. KONFIGURATION & VERBINDUNG
# -----------------------------------------------------------------------------

load_dotenv()

DB_CONFIG = {
    "server": os.getenv("DB_SERVER"),
    "database": os.getenv("DB_DATABASE"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# -----------------------------------------------------------------------------
# 2. BENUTZER & BERECHTIGUNGEN
# -----------------------------------------------------------------------------
# Berechtigungen pro Stufe (SECURITYLEVEL aus DB)
# Level 1: Nur Rosenheim, nur Jahresansicht
# Level 2: Rosenheim + Freiburg, Quartalsansicht
# Level 3: Alle Stores, alle Filter (Monat, Quartal, Jahr)
PERMISSIONS = {
    1: {
        "stores": ["Rosenheim"],
        "can_filter_quartal": False,
        "can_filter_monat": False,
        "description": "Zugriff auf Rosenheim (nur Jahresansicht)",
    },
    2: {
        "stores": ["Rosenheim", "Freiburg im Breisgau"],
        "can_filter_quartal": True,
        "can_filter_monat": False,
        "description": "Zugriff auf alle Stores (Jahr + Quartal)",
    },
    3: {
        "stores": ["Rosenheim", "Freiburg im Breisgau"],
        "can_filter_quartal": True,
        "can_filter_monat": True,
        "description": "Voller Zugriff (alle Filter)",
    },
}


def check_login(username: str, password: str) -> bool:
    """Prüft Login-Daten gegen die Datenbank-Tabelle LOV_USER_LOGINS."""
    try:
        conn = pymssql.connect(
            server=DB_CONFIG['server'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
        )
        cursor = conn.cursor(as_dict=True)
        
        # Benutzer in der Datenbank suchen
        query = """
            SELECT [USERNAME], [USERPASS], [SECURITYLEVEL]
            FROM [dbo].[LOV_USER_LOGINS]
            WHERE [USERNAME] = %s AND [USERPASS] = %s
        """
        cursor.execute(query, (username, password))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # SECURITYLEVEL auf 1-3 begrenzen (falls andere Werte in DB)
            level = int(row['SECURITYLEVEL']) if row['SECURITYLEVEL'] else 1
            level = max(1, min(3, level))  # Zwischen 1 und 3 halten
            
            st.session_state["logged_in"] = True
            st.session_state["username"] = row['USERNAME']
            st.session_state["user_level"] = level
            st.session_state["user_name"] = f"Fachkraft Stufe {level}"
            return True
        return False
    except Exception as e:
        st.error(f"Datenbankfehler beim Login: {e}")
        return False


def logout():
    """Logout und Session zurücksetzen."""
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["user_level"] = None
    st.session_state["user_name"] = None


def get_user_permissions() -> dict:
    """Gibt die Berechtigungen des aktuellen Users zurück."""
    level = st.session_state.get("user_level", 1)
    return PERMISSIONS.get(level, PERMISSIONS[1])


def show_login_page():
    """Zeigt die Login-Seite an."""
    st.set_page_config(page_title="Login – DB Rechnung", layout="centered")
    st.title("🔐 Login")
    st.markdown("Bitte melden Sie sich an, um fortzufahren.")
    
    with st.form("login_form"):
        username = st.text_input("Benutzername")
        password = st.text_input("Passwort", type="password")
        submit = st.form_submit_button("Anmelden")
        
        if submit:
            if check_login(username, password):
                st.success(f"Willkommen, {st.session_state['user_name']}!")
                st.rerun()
            else:
                st.error("Ungültiger Benutzername oder Passwort.")
    
    st.markdown("---")
    st.caption("Login mit Benutzerdaten aus der Datenbank")


# -----------------------------------------------------------------------------
# 3. LOGIN-PRÜFUNG
# -----------------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    show_login_page()
    st.stop()

# Ab hier: User ist eingeloggt
permissions = get_user_permissions()


def load_store_names() -> list[str]:
    """Gibt die erlaubten Stores basierend auf Berechtigungsstufe zurück."""
    return permissions["stores"]

@st.cache_data(ttl=600)
def load_final_table_from_db(store_name: str):
    conn = pymssql.connect(
        server=DB_CONFIG['server'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
    )

    store = (store_name or "").strip()

    # Alle Stores (inkl. Freiburg im Breisgau, Rosenheim, etc.) aus derselben G14-View laden
    g14_view = '[list_views].[G14_Gesamt_DB_SCHEMA]'

    query = f"""
SELECT *
FROM {g14_view}
WHERE [StoreName] = %s;
"""
    df = pd.read_sql(query, conn, params=[store_name])

    conn.close()

    # Manche Views liefern (DBEbene, Position) statt (Ebene, EPos).
    # Damit die "DB Rechnung nach Ebenen" immer funktioniert, mappen wir robust – unabhängig vom Store.
    if not df.empty:
        if 'Ebene' not in df.columns and 'DBEbene' in df.columns:
            ebene_map = {
                'DB1': 'E1',
                'DB2': 'E2',
                'DB3': 'E3',
            }
            df['Ebene'] = (
                df['DBEbene']
                .astype(str)
                .str.strip()
                .map(ebene_map)
                .fillna(df['DBEbene'].astype(str).str.strip())
            )

        if 'EPos' not in df.columns and 'Position' in df.columns:
            df['EPos'] = df['Position']

    if not df.empty:
        sort_cols = [
            c for c in ['Monat', 'Ebene', 'EPos', 'Kenngröße', 'ProduktKategorie', 'ProduktLinie']
            if c in df.columns
        ]
        if sort_cols:
            df = df.sort_values(sort_cols, kind='stable')
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
        'umsatz eur',
        'umsatz',
        'revenue',
        'sales',
        'sales eur',
        'saleseur',
        'totalrevenue',
    }

    k = df_filtered['Kenngröße'].apply(normalize_kenngroesse)
    mask = k.isin(candidates)

    found = sorted(set(df_filtered.loc[mask, 'Kenngröße'].astype(str).unique().tolist()))
    umsatz = float(df_filtered.loc[mask, 'Wert'].sum()) if mask.any() else 0.0
    return umsatz, found


def normalize_kenngroesse(value) -> str:
    """Normalisiert Kenngröße-Namen für robustes Matching.

    Ziel: leichte Schreibvarianten (Whitespace, Punkte, € vs EUR, etc.) vereinheitlichen.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    s = str(value)
    s = s.replace('\u00a0', ' ')
    s = s.replace('€', ' eur ')
    s = s.strip().lower()
    # Trennzeichen vereinheitlichen
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"[._\-/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def drop_allgemein_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    """Gibt die Pivot-Tabelle unverändert zurück.

    Hinweis: Früher wurden 'Allgemein'-Spalten entfernt. Jetzt werden alle Spalten
    (inkl. Allgemein) beibehalten, um keine Daten zu manipulieren.
    """
    return pivot


def _pivot_for_kenngroesse(df_filtered: pd.DataFrame, kenngroesse_norm: str | list[str] | set[str] | tuple[str, ...]) -> pd.DataFrame:
    """Erzeugt eine 1-Zeilen-Pivot-Tabelle (ProduktLinie/ProduktKategorie) für eine Kenngröße.

    `kenngroesse_norm` kann ein String oder eine Liste von Kandidaten sein.
    """
    needed = {'Kenngröße', 'ProduktLinie', 'ProduktKategorie', 'Wert'}
    if df_filtered.empty or not needed.issubset(df_filtered.columns):
        return pd.DataFrame()

    if isinstance(kenngroesse_norm, (list, set, tuple)):
        targets = {normalize_kenngroesse(x) for x in kenngroesse_norm}
    else:
        targets = {normalize_kenngroesse(kenngroesse_norm)}

    tmp = df_filtered.copy()
    # Wichtig: In manchen Datenquellen (z.B. Freiburg) sind Linie/Kategorie NULL.
    # Ohne Fallback würden diese Zeilen in groupby/pivot "verschwinden".
    tmp['ProduktLinie'] = tmp['ProduktLinie'].fillna('Allgemein')
    tmp['ProduktKategorie'] = tmp['ProduktKategorie'].fillna('Allgemein')
    tmp['Wert'] = pd.to_numeric(tmp['Wert'], errors='coerce').fillna(0)
    tmp['_KenngroesseNorm'] = tmp['Kenngröße'].apply(normalize_kenngroesse)

    tmp = tmp[tmp['_KenngroesseNorm'].isin(targets)]
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
    pivot = drop_allgemein_columns(pivot)
    pivot = pivot.sort_index(axis=1)
    return pivot


# Erwartete Kenngrößen pro Ebene (für Validierung)
EXPECTED_KENNGROESSEN = {
    'E1': ['UmsatzEUR', 'TransferPriceEUR'],
    'E2': ['Commission in EUR', 'DiscountAufMaterialEUR', 'DiscountAufMaterialKategorieEUR'],
    'E3': ['Additional Procurement Costs', 'Marketing Campaign', 'Monthly Rent', 'Monthly Salary', 'Monthly Social Costs'],
}


def get_missing_kenngroessen(df_filtered: pd.DataFrame) -> dict:
    """Prüft welche erwarteten Kenngrößen in den Daten fehlen."""
    if 'Kenngröße' not in df_filtered.columns:
        return {ebene: kenn_list for ebene, kenn_list in EXPECTED_KENNGROESSEN.items()}
    
    # Vorhandene Kenngrößen normalisieren
    vorhandene = set(df_filtered['Kenngröße'].dropna().astype(str).str.strip().str.lower())
    
    missing_per_ebene = {}
    for ebene, expected in EXPECTED_KENNGROESSEN.items():
        missing = []
        for k in expected:
            if k.lower() not in vorhandene:
                missing.append(k)
        if missing:
            missing_per_ebene[ebene] = missing
    
    return missing_per_ebene


def compute_deckungsbeitraege(df_filtered: pd.DataFrame) -> dict:
    """Berechnet DB-Logik spaltenweise (nicht zeilenweise).

    - E1 Total = UmsatzEUR + TransferPriceEUR
    - E2 Total = E1 Total - (Commission + Discounts)
    - E3 Total = E2 Total - Summe(E3 Kosten)
    
    Hinweis: Wenn E2-Kenngrößen (Commission, Discounts) fehlen, wird E2 = E1 gesetzt.
    """
    umsatz = _pivot_for_kenngroesse(df_filtered, ['UmsatzEUR', 'Umsatz EUR', 'umsatzeur'])
    transfer = _pivot_for_kenngroesse(df_filtered, ['TransferPriceEUR', 'Transfer Price EUR', 'transferpriceeur'])
    
    # E2-Kenngrößen: Commission und Discounts
    commission = _pivot_for_kenngroesse(df_filtered, ['Commission in EUR', 'Commission', 'commission in eur', 'commission'])
    discount_material = _pivot_for_kenngroesse(df_filtered, ['DiscountAufMaterialEUR', 'discountaufmaterialeur'])
    discount_kategorie = _pivot_for_kenngroesse(df_filtered, ['DiscountAufMaterialKategorieEUR', 'discountaufmaterialkategorieeur'])

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
    for p in [umsatz, transfer, commission, discount_material, discount_kategorie] + e3_cost_parts:
        if not p.empty:
            all_cols = all_cols.union(p.columns)
    if len(all_cols) == 0:
        return {
            'e1_total': pd.DataFrame(),
            'e2_total': pd.DataFrame(),
            'e3_total': pd.DataFrame(),
            'missing_per_ebene': get_missing_kenngroessen(df_filtered),
        }

    umsatz_a = _align(umsatz, all_cols)
    transfer_a = _align(transfer, all_cols)
    commission_a = _align(commission, all_cols)
    discount_material_a = _align(discount_material, all_cols)
    discount_kategorie_a = _align(discount_kategorie, all_cols)

    e3_cost_a = pd.DataFrame(index=['_row'], columns=all_cols).fillna(0)
    for p in e3_cost_parts:
        e3_cost_a = e3_cost_a.add(_align(p, all_cols), fill_value=0)

    # TransferPriceEUR ist in vielen Datenquellen bereits als negativer Wert hinterlegt.
    # Daher hier bewusst PLUS, um kein "Minus minus" zu erzeugen.
    e1_total = umsatz_a.add(transfer_a, fill_value=0)
    
    # E2: E1 minus alle E2-Abzüge (Commission + Discounts)
    e2_abzuege = commission_a.add(discount_material_a, fill_value=0).add(discount_kategorie_a, fill_value=0)
    e2_total = e1_total.sub(e2_abzuege, fill_value=0)
    
    e3_total = e2_total.sub(e3_cost_a, fill_value=0)

    # Summen-Spalte sicherstellen (falls in all_cols nicht enthalten)
    if ('Summen', 'Gesamt') not in all_cols:
        for df_ in (e1_total, e2_total, e3_total):
            df_[('Summen', 'Gesamt')] = df_.sum(axis=1)

    return {
        'e1_total': e1_total,
        'e2_total': e2_total,
        'e3_total': e3_total,
        'missing_per_ebene': get_missing_kenngroessen(df_filtered),
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

    # Fallback für NULL Linie/Kategorie, damit die Werte in der Pivot landen.
    df_e['ProduktLinie'] = df_e['ProduktLinie'].fillna('Allgemein')
    df_e['ProduktKategorie'] = df_e['ProduktKategorie'].fillna('Allgemein')

    df_e['Wert'] = pd.to_numeric(df_e['Wert'], errors='coerce').fillna(0)

    df_e['_KenngroesseNorm'] = df_e['Kenngröße'].astype(str).str.strip().str.lower()

    # SalesPrice und SalesAmount ausblenden (nicht-monetäre/technische Kennzahlen)
    exclude_patterns = {'salesprice', 'salespriceeur', 'salesamount'}
    df_e = df_e[~df_e['_KenngroesseNorm'].isin(exclude_patterns)]
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
    pivot = drop_allgemein_columns(pivot)
    pivot = pivot.sort_index(axis=1)

    # Reihenfolge erzwingen
    pivot = pivot.reindex(row_order).fillna(0)
    pivot.index.name = None
    return pivot


def build_all_ebenen_table(df_filtered: pd.DataFrame, ebenen: list[str]) -> pd.DataFrame:
    """Kombiniert E1..E3 zu einer einzigen Tabelle (Zeilen untereinander).
    
    E2 wird auch angezeigt, wenn keine E2-Daten in der Quelle existieren,
    aber berechnet werden kann (E2 Total = E1 Total - E2-Abzüge).
    """
    parts: list[pd.DataFrame] = []
    
    # DB-Logik vorab berechnen
    db = compute_deckungsbeitraege(df_filtered)
    
    for ebene in ebenen:
        part = build_ebene_table(df_filtered, ebene)
        if not part.empty:
            # Prefix, damit die Zeilen eindeutig sind (z.B. "E1 | 1. UmsatzEUR")
            part = part.copy()
            part.index = [f"{ebene} | {idx}" for idx in part.index]
            parts.append(part)
    
    # E2 hinzufügen, falls nicht in den Daten vorhanden aber berechenbar
    if 'E2' not in ebenen and not db['e2_total'].empty:
        # E2 Total-Zeile erstellen
        e2_total_row = db['e2_total'].copy()
        e2_total_row.index = ['E2 | E2 Total']
        parts.append(e2_total_row)

    if not parts:
        return pd.DataFrame()

    # Spalten unionieren und vertikal stapeln
    all_cols = pd.Index([])
    for p in parts:
        all_cols = all_cols.union(p.columns)
    parts = [p.reindex(columns=all_cols) for p in parts]
    combined = pd.concat(parts, axis=0)
    combined = combined.fillna(0)

    # Total-Zeilen mit berechneten Werten überschreiben
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

    combined = combined.fillna(0)
    
    # Sortierung: E1, E2, E3 in richtiger Reihenfolge
    # Jede Ebene: Kenngrößen zuerst, dann Total am Ende der Ebene
    def sort_key(idx):
        # Ebene extrahieren
        if idx.startswith('E1'):
            ebene = 1
        elif idx.startswith('E2'):
            ebene = 2
        elif idx.startswith('E3'):
            ebene = 3
        else:
            ebene = 9
        
        # Total-Zeilen ans Ende der jeweiligen Ebene
        if 'E1 Total' in idx:
            return (1, 99, idx)
        elif 'E2 Total' in idx:
            return (2, 99, idx)
        elif 'E3 Total' in idx:
            return (3, 99, idx)
        else:
            # Normale Zeilen nach Position sortieren
            if 'UmsatzEUR' in idx:
                return (ebene, 1, idx)
            elif 'TransferPriceEUR' in idx:
                return (ebene, 2, idx)
            elif 'Commission' in idx:
                return (ebene, 1, idx)
            elif 'DiscountAufMaterial' in idx and 'Kategorie' not in idx:
                return (ebene, 2, idx)
            elif 'DiscountAufMaterialKategorie' in idx:
                return (ebene, 3, idx)
            elif 'Additional Procurement' in idx:
                return (ebene, 1, idx)
            elif 'Marketing Campaign' in idx:
                return (ebene, 2, idx)
            elif 'Monthly Rent' in idx:
                return (ebene, 3, idx)
            elif 'Monthly Salary' in idx:
                return (ebene, 4, idx)
            elif 'Monthly Social' in idx:
                return (ebene, 5, idx)
            else:
                return (ebene, 50, idx)
    
    combined = combined.loc[sorted(combined.index, key=sort_key)]

    return combined


st.set_page_config(page_title="DB Rechnung", layout="wide")
st.title("Final Table – Kosten & Totals")

with st.sidebar:
    # User-Info und Logout
    st.markdown(f"**Angemeldet als:** {st.session_state['user_name']}")
    st.caption(f"Berechtigung: {permissions['description']}")
    if st.button("🚪 Abmelden"):
        logout()
        st.rerun()
    
    st.markdown("---")
    st.header("Filter")
    
    store_options = load_store_names()
    default_index = 0
    if 'Rosenheim' in store_options:
        default_index = store_options.index('Rosenheim')
    store_name = st.selectbox("StoreName", store_options, index=default_index)
    
    if st.button("🔄 Daten aktualisieren"):
        load_final_table_from_db.clear()
        st.rerun()


try:
    df_raw = load_final_table_from_db(store_name.strip() or "Rosenheim")

    if df_raw.empty:
        st.warning("Keine Daten geladen. Bitte StoreName/Verbindung prüfen.")
        st.stop()

    # Minimal benötigte Spalten für Filter + Kennzahlen.
    # Ebenen-Auswertung ist optional und wird nur gezeigt, wenn Ebene/EPos vorhanden sind.
    required_cols = {
        'StoreName', 'Monat', 'Kenngröße',
        'ProduktKategorie', 'ProduktLinie', 'Wert'
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
        
        # Quartal-Filter nur für Level 2+
        selected_quartal = 'Alle'
        if permissions["can_filter_quartal"]:
            present_quarters = sorted(df_jahr['Monat_dt'].dt.quarter.dropna().unique().tolist())
            quartal_options = ['Alle'] + [f"Q{q}" for q in present_quarters]
            selected_quartal = st.selectbox("Quartal", quartal_options)
        else:
            st.caption("🔒 Quartalsfilter nicht verfügbar (Berechtigung)")

        df_scope = df_jahr
        if selected_quartal != 'Alle':
            df_scope = df_scope[df_scope['Quartal'] == selected_quartal]

        # Monat-Filter nur für Level 3
        selected_monat = 'Alle'
        if permissions["can_filter_monat"]:
            month_map = (
                df_scope[['Monat', 'Monat_dt']]
                .drop_duplicates()
                .sort_values('Monat_dt', kind='stable')
            )
            monat_options = ['Alle'] + month_map['Monat'].astype(str).tolist()
            selected_monat = st.selectbox("Monat", monat_options)
        else:
            st.caption("🔒 Monatsfilter nicht verfügbar (Berechtigung)")

    df_filtered = df[df['Jahr'] == selected_jahr].copy()
    if selected_quartal != 'Alle':
        df_filtered = df_filtered[df_filtered['Quartal'] == selected_quartal]
    if selected_monat != 'Alle':
        # Robuster Vergleich: beide Seiten als String
        df_filtered = df_filtered[df_filtered['Monat'].astype(str) == str(selected_monat)]

    # Debug: Falls leer, zeige verfügbare Werte
    if df_filtered.empty:
        with st.expander("🔍 Debug: Verfügbare Daten", expanded=True):
            st.write(f"Ausgewähltes Jahr: {selected_jahr}")
            st.write(f"Ausgewähltes Quartal: {selected_quartal}")
            st.write(f"Ausgewählter Monat: {selected_monat} (Typ: {type(selected_monat).__name__})")
            st.write(f"Verfügbare Jahre: {sorted(df['Jahr'].unique().tolist())}")
            st.write(f"Verfügbare Monate im Jahr {selected_jahr}: {df[df['Jahr'] == selected_jahr]['Monat'].unique().tolist()}")
            st.write(f"Monat-Spalte Typ: {df['Monat'].dtype}")
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

    # Hinweis über fehlende Kenngrößen pro Ebene
    missing_per_ebene = db.get('missing_per_ebene', {})
    if missing_per_ebene:
        missing_text = "**Hinweis: Folgende Kenngrößen fehlen in den Daten:**\n"
        for ebene, missing_list in missing_per_ebene.items():
            missing_text += f"- **{ebene}**: {', '.join(missing_list)}\n"
        st.warning(missing_text)

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
    c1.metric("E3 Total Summe", format_eur(sum_total))
    c2.metric("Status", status_text)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # EBENEN-STRUKTUR (wie app.py-Layout, aber mit Ebenen statt DB)
    # -------------------------------------------------------------------------
    st.subheader("DB Rechnung nach Ebenen")

    def format_german(val):
        if pd.isna(val):
            return "-"
        # 0 soll sichtbar sein (sonst wirkt die Tabelle "leer")
        return "{:,.2f} €".format(float(val)).replace(",", "X").replace(".", ",").replace("X", ".")

    def style_total_rows(row):
        idx = str(row.name)
        if idx.endswith("Total"):
            return ['background-color: #f2f2f2; font-weight: bold; border-top: 1px solid #aaa; color: black;'] * len(row)
        return [''] * len(row)

    if 'Ebene' not in df_filtered.columns or 'EPos' not in df_filtered.columns:
        st.info("Hinweis: Ebenen-Auswertung (E1/E2/E3) ist für diese Datenquelle nicht verfügbar (Spalten Ebene/EPos fehlen).")
    else:
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
        # Kurze, neutrale Beschreibungen (bei Unklarheit bewusst vorsichtig formuliert)
        descriptions = {
            'UmsatzEUR': 'Umsatz in EUR.',
            'TransferPriceEUR': 'Transferpreis / Einkaufspreis in EUR.',
            'Commission in EUR': 'Provision / Commission in EUR.',
            'DiscountAufMaterialEUR': 'Rabatt auf Material in EUR.',
            'DiscountAufMaterialKategorieEUR': 'Rabatt auf Materialkategorie in EUR.',
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
            if name in seen:
                continue
            seen.add(name)
            desc = descriptions.get(name, 'Kenngröße aus der Datenquelle (Beschreibung nicht hinterlegt).')
            st.markdown(f"- **{name}**: {desc}")

except Exception as e:
    st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
