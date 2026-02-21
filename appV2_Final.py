
import streamlit as st
import pandas as pd
import pymssql
import re

# -----------------------------------------------------------------------------
# 1. KONFIGURATION & VERBINDUNG
# -----------------------------------------------------------------------------
DB_CONFIG = {
    "server": "edu.hdm-server.eu",
    "database": "ERPDEV",
    "user": "ERP_REMOTE_USER",
    "password": "Password123",
}


def load_store_names() -> list[str]:
    """Lädt verfügbare Stores aus der DB (Fallback: Rosenheim/Freiburg)."""
    fallback = ['Rosenheim', 'Freiburg im Breisgau']
    try:
        conn = pymssql.connect(
            server=DB_CONFIG['server'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
        )
        q = """
SELECT DISTINCT [StoreName]
    FROM [list_views].[G14_Gesamt_DB_SCHEMA_FINAL]
WHERE [StoreName] IS NOT NULL
ORDER BY [StoreName];
"""
        rows = pd.read_sql(q, conn)
        conn.close()
        if 'StoreName' not in rows.columns:
            return fallback
        stores = [str(x).strip() for x in rows['StoreName'].dropna().tolist() if str(x).strip()]
        return stores or fallback
    except Exception:
        return fallback


def authenticate_user(username: str, password: str) -> tuple[bool, int | None]:
    """Prüft Login gegen ERPDEV.dbo.LOV_USER_LOGINS und liefert SECURITYLEVEL zurück."""
    u = str(username or '').strip().lower()
    p = str(password or '').strip()
    if not u or not p:
        return False, None

    try:
        conn = pymssql.connect(
            server=DB_CONFIG['server'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
        )
        cursor = conn.cursor(as_dict=True)
        cursor.execute(
            """
SELECT TOP (1) [SECURITYLEVEL]
FROM [dbo].[LOV_USER_LOGINS]
WHERE LOWER(LTRIM(RTRIM([USERNAME]))) = %s
  AND LTRIM(RTRIM([USERPASS])) = %s;
""",
            (u, p),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False, None
        try:
            return True, int(row.get('SECURITYLEVEL'))
        except Exception:
            return True, None
    except Exception:
        return False, None

    return False, None


def get_permission_level_number() -> int | None:
    """Liest SECURITYLEVEL aus Session/Query-Parametern."""
    for key in ('security_level', 'SECURITYLEVEL', 'permission_level', 'berechtigung', 'role', 'user_role'):
        value = st.session_state.get(key)
        if value not in (None, ''):
            try:
                return int(value)
            except Exception:
                pass

    try:
        params = st.query_params
        for key in ('security_level', 'SECURITYLEVEL', 'permission_level', 'berechtigung', 'role', 'user_role'):
            if key in params:
                value = params.get(key)
                if isinstance(value, list):
                    value = value[0] if value else None
                if value not in (None, ''):
                    try:
                        return int(value)
                    except Exception:
                        pass
    except Exception:
        pass

    return None


def get_permission_level() -> str:
    level = get_permission_level_number()
    if level in (1, 2, 3):
        return f'Fachkraft {level}'
    return 'Unbekannt'


def get_permission_rights(permission_level: str) -> list[str]:
    """Liefert die sichtbaren Rechte je Fachkraft-Level."""
    level = None
    m = re.search(r"(\d+)", str(permission_level))
    if m:
        level = int(m.group(1))

    rights_by_level = {
        1: [
            'Nur Store Rosenheim sichtbar.',
            'Nur Jahresdaten (kein Quartal/Monat).',
            'Lesender Zugriff auf Dashboard.',
        ],
        2: [
            'Stores Rosenheim und Freiburg sichtbar.',
            'Filter: Jahr und Quartal.',
            'Monatsfilter nicht verfügbar.',
        ],
        3: [
            'Alle verfügbaren Stores sichtbar.',
            'Filter: Jahr, Quartal und Monat.',
            'Voller Analysezugriff in der App.',
        ],
    }

    return rights_by_level.get(level, ['Keine Rechtezuordnung gefunden.'])


def get_allowed_stores(permission_level_number: int | None) -> list[str]:
    """Store-Sichtbarkeit gemäß Handbuch."""
    all_stores = load_store_names()
    if permission_level_number == 1:
        return [s for s in all_stores if s == 'Rosenheim']
    if permission_level_number == 2:
        allowed = {'Rosenheim', 'Freiburg im Breisgau'}
        return [s for s in all_stores if s in allowed]
    if permission_level_number == 3:
        return all_stores
    return ['Rosenheim']


def get_permission_summary(permission_level_number: int | None) -> str:
    """Kurztext für Berechtigungsanzeige in der Sidebar."""
    if permission_level_number == 1:
        return 'Basis-Zugriff (nur Jahr)'
    if permission_level_number == 2:
        return 'Erweiterter Zugriff (Jahr + Quartal)'
    if permission_level_number == 3:
        return 'Voller Zugriff (alle Filter)'
    return 'Unbekannt'

@st.cache_data(ttl=600)
def load_final_table_from_db(store_name: str):
    conn = pymssql.connect(
        server=DB_CONFIG['server'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
    )

    store = (store_name or "").strip()

    # Alle Stores (inkl. Freiburg im Breisgau, Rosenheim, etc.) aus der G14-Final-View laden
    g14_view = '[list_views].[G14_Gesamt_DB_SCHEMA_FINAL]'

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


def compute_deckungsbeitraege(df_filtered: pd.DataFrame) -> dict:
    """Berechnet DB-Logik spaltenweise (nicht zeilenweise).

    - E1 Total = UmsatzEUR + TransferPriceEUR
    - E2 Total = E1 Total - DiscountAufMaterialEUR - DiscountAufMaterialKategorieEUR
    - E3 Total = E2 Total - Summe(E3 Kosten) - Commission in EUR
    """
    umsatz = _pivot_for_kenngroesse(df_filtered, ['UmsatzEUR', 'Umsatz EUR', 'umsatzeur'])
    transfer = _pivot_for_kenngroesse(df_filtered, ['TransferPriceEUR', 'Transfer Price EUR', 'transferpriceeur'])
    commission = _pivot_for_kenngroesse(df_filtered, ['Commission in EUR', 'Commission', 'commission in eur', 'commission'])
    discount_material = _pivot_for_kenngroesse(
        df_filtered,
        [
            'DiscountAufMaterialEUR',
            'Discount Auf Material EUR',
            'discountaufmaterialeur',
        ]
    )
    discount_material_kategorie = _pivot_for_kenngroesse(
        df_filtered,
        [
            'DiscountAufMaterialKategorieEUR',
            'Discount Auf Material Kategorie EUR',
            'discountaufmaterialkategorieeur',
        ]
    )

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
    for p in [umsatz, transfer, commission, discount_material, discount_material_kategorie] + e3_cost_parts:
        if not p.empty:
            all_cols = all_cols.union(p.columns)
    if len(all_cols) == 0:
        return {
            'e1_total': pd.DataFrame(),
            'e2_total': pd.DataFrame(),
            'e3_total': pd.DataFrame(),
            'missing': ['umsatzeur', 'transferpriceeur', 'commission in eur', 'discountaufmaterialkategorieeur']
        }

    umsatz_a = _align(umsatz, all_cols)
    transfer_a = _align(transfer, all_cols)
    commission_a = _align(commission, all_cols)
    discount_material_a = _align(discount_material, all_cols)
    discount_material_kategorie_a = _align(discount_material_kategorie, all_cols)

    e3_cost_a = pd.DataFrame(index=['_row'], columns=all_cols).fillna(0)
    for p in e3_cost_parts:
        e3_cost_a = e3_cost_a.add(_align(p, all_cols), fill_value=0)

    # TransferPriceEUR ist in vielen Datenquellen bereits als negativer Wert hinterlegt.
    # Daher hier bewusst PLUS, um kein "Minus minus" zu erzeugen.
    e1_total = umsatz_a.add(transfer_a, fill_value=0)
    # Discounts sollen E2 immer reduzieren – unabhängig davon,
    # ob die Datenquelle Discounts als negativ oder positiv liefert.
    e2_total = (
        e1_total
        .sub(discount_material_a.abs(), fill_value=0)
        .sub(discount_material_kategorie_a.abs(), fill_value=0)
    )
    # E3-Kosten/Provision sollen E3 immer reduzieren – unabhängig davon,
    # ob die Datenquelle Kosten als negativ oder positiv liefert.
    e3_total = e2_total.sub(e3_cost_a.abs(), fill_value=0).sub(commission_a.abs(), fill_value=0)

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
    if discount_material_kategorie.empty:
        missing.append('DiscountAufMaterialKategorieEUR')

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

    # Fallback für NULL Linie/Kategorie, damit die Werte in der Pivot landen.
    df_e['ProduktLinie'] = df_e['ProduktLinie'].fillna('Allgemein')
    df_e['ProduktKategorie'] = df_e['ProduktKategorie'].fillna('Allgemein')

    df_e['Wert'] = pd.to_numeric(df_e['Wert'], errors='coerce').fillna(0)

    df_e['_KenngroesseNorm'] = df_e['Kenngröße'].apply(normalize_kenngroesse)

    # Technische Kennzahlen in der Ebenen-Tabelle ausblenden.
    hidden_metrics = {
        'salespriceeur',
        'sales price eur',
        'salesamount',
        'sales amount',
    }
    df_e = df_e[~df_e['_KenngroesseNorm'].isin(hidden_metrics)]
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
    combined = combined.fillna(0)

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

    combined = combined.fillna(0)

    return combined


st.set_page_config(page_title=" APPV211DB Rosenheim", layout="wide")
st.title("Final Table – Kosten & Totals")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state.get('logged_in'):
    st.subheader('Anmeldung')
    with st.form('login_form', clear_on_submit=False):
        login_user = st.text_input('Benutzername')
        login_pass = st.text_input('Passwort', type='password')
        submitted = st.form_submit_button('Anmelden')

    if submitted:
        ok, sec_level = authenticate_user(login_user, login_pass)
        if ok:
            st.session_state['logged_in'] = True
            st.session_state['username'] = str(login_user).strip()
            if sec_level is not None:
                st.session_state['security_level'] = int(sec_level)
            st.success('Anmeldung erfolgreich.')
            st.rerun()
        else:
            st.error('Login fehlgeschlagen. Bitte Benutzername/Passwort prüfen.')
    st.stop()

with st.sidebar:
    permission_level = get_permission_level()
    permission_level_number = get_permission_level_number()
    current_username = str(st.session_state.get('username', '')).strip() or 'Unbekannt'
    role_label = f"Fachkraft Stufe {permission_level_number}" if permission_level_number in (1, 2, 3) else permission_level
    st.markdown(
        f"**Angemeldet als:** {role_label}"
        + (f"  \nBenutzer: {current_username}" if current_username != 'Unbekannt' else "")
    )
    st.markdown(f"**Berechtigung:** {get_permission_summary(permission_level_number)}")
    if st.button("Abmelden"):
        for k in ('logged_in', 'username', 'security_level'):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()
    st.header("Filter")
    store_options = get_allowed_stores(permission_level_number)
    default_index = 0
    if 'Rosenheim' in store_options:
        default_index = store_options.index('Rosenheim')
    store_name = st.selectbox("StoreName", store_options, index=default_index)
    if st.button("🔄 Daten aktualisieren"):
        load_final_table_from_db.clear()
        load_store_names.clear()
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

    selected_monat_dt = None
    with st.sidebar:
        jahre = sorted(df['Jahr'].unique(), reverse=True)
        selected_jahr = st.selectbox("Jahr", jahre)

        df_jahr = df[df['Jahr'] == selected_jahr]
        selected_quartal = 'Alle'
        if permission_level_number in (2, 3):
            present_quarters = sorted(df_jahr['Monat_dt'].dt.quarter.dropna().unique().tolist())
            quartal_options = ['Alle'] + [f"Q{q}" for q in present_quarters]
            selected_quartal = st.selectbox("Quartal", quartal_options)

        df_scope = df_jahr
        if selected_quartal != 'Alle':
            df_scope = df_scope[df_scope['Quartal'] == selected_quartal]

        selected_monat = 'Alle'
        if permission_level_number == 3:
            month_map = (
                df_scope[['Monat', 'Monat_dt']]
                .drop_duplicates()
                .sort_values('Monat_dt', kind='stable')
            )
            month_map = month_map.copy()
            month_map['MonatLabel'] = month_map['Monat'].astype(str)
            month_lookup = dict(zip(month_map['MonatLabel'], month_map['Monat_dt']))
            monat_options = ['Alle'] + month_map['MonatLabel'].tolist()
            selected_monat = st.selectbox("Monat", monat_options)
            if selected_monat != 'Alle':
                selected_monat_dt = month_lookup.get(selected_monat)

    df_filtered = df[df['Jahr'] == selected_jahr].copy()
    if selected_quartal != 'Alle':
        df_filtered = df_filtered[df_filtered['Quartal'] == selected_quartal]
    if selected_monat != 'Alle':
        if selected_monat_dt is not None:
            df_filtered = df_filtered[df_filtered['Monat_dt'] == selected_monat_dt]
        else:
            df_filtered = df_filtered[df_filtered['Monat'].astype(str) == selected_monat]

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

    def apply_display_signs(df_table: pd.DataFrame) -> pd.DataFrame:
        """Steuert nur die Darstellung der Vorzeichen in der Ebenen-Tabelle.

        - Kostenstellen immer mit Minus
        - E1/E2/E3 (inkl. Total) mit rechnerischem Vorzeichen anzeigen
        """
        if df_table.empty:
            return df_table

        minus_rows = {
            'discountaufmaterialeur',
            'discountaufmaterialkategorieeur',
            'additional procurement costs',
            'commission',
            'commission in eur',
            'marketing campaign',
            'monthly rent',
            'monthly salary',
            'monthly social costs',
        }
        out = df_table.copy()
        for idx in out.index:
            label = str(idx)
            if '|' in label:
                label = label.split('|', 1)[1].strip()
            label_norm = normalize_kenngroesse(label)

            row_vals = pd.to_numeric(out.loc[idx], errors='coerce').fillna(0)
            if label_norm in minus_rows:
                out.loc[idx] = -row_vals.abs().values

        return out

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
                df_all_display = apply_display_signs(df_all)
                st.dataframe(
                    df_all_display.style
                    .format(format_german)
                    .apply(style_total_rows, axis=1),
                    use_container_width=True,
                    height=520
                )

    st.markdown("---")

    with st.expander("Legende: Kenngrößen"):
        # Kurze, neutrale Beschreibungen (bei Unklarheit bewusst vorsichtig formuliert)
        descriptions = {
            'SalesPriceEUR': 'Verkaufspreis in EUR (Sales Price).',
            'SalesAmount': 'Verkaufsmenge / Anzahl (Sales Amount).',
            'UmsatzEUR': 'Umsatz in EUR.',
            'TransferPriceEUR': 'Transferpreis / Einkaufspreis in EUR.',
            'DiscountAufMaterialKategorieEUR': 'Rabatt auf Materialkategorie in EUR.',
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
            if name in seen:
                continue
            seen.add(name)
            desc = descriptions.get(name, 'Kenngröße aus der Datenquelle (Beschreibung nicht hinterlegt).')
            st.markdown(f"- **{name}**: {desc}")

except Exception as e:
    st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
