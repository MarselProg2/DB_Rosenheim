import streamlit as st
import pandas as pd
import pyodbc
import re


# -----------------------------------------------------------------------------
# 1. KONFIGURATION & VERBINDUNG
# -----------------------------------------------------------------------------
DB_CONFIG = {
	"server": "edu.hdm-server.eu",
	"database": "ERPDEV",
	"user": "w25s252",
	"password": "202860",
	"driver": "{SQL Server}",
}


def _connect() -> pyodbc.Connection:
	conn_str = (
		f"DRIVER={DB_CONFIG['driver']};SERVER={DB_CONFIG['server']};"
		f"DATABASE={DB_CONFIG['database']};UID={DB_CONFIG['user']};PWD={DB_CONFIG['password']};"
	)
	return pyodbc.connect(conn_str)


G15_VIEW = "[ERPDEV].[list_views].[V_LIST_G15_GESAMT_DBSCHEMA]"


@st.cache_data(ttl=600)
def load_store_names() -> list[str]:
	conn = _connect()
	try:
		df = pd.read_sql(
			f"SELECT DISTINCT [StoreName] FROM {G15_VIEW} ORDER BY [StoreName];",
			conn,
		)
		names = df["StoreName"].dropna().astype(str).tolist() if "StoreName" in df.columns else []
		return names
	finally:
		conn.close()


@st.cache_data(ttl=600)
def load_g15_data(store_names: tuple[str, ...]) -> pd.DataFrame:
	"""Lädt Rohdaten aus der G15-View für die gewählten Stores."""
	conn = _connect()
	try:
		if not store_names:
			return pd.DataFrame()

		placeholders = ",".join(["?"] * len(store_names))
		query = f"""
SELECT *
FROM {G15_VIEW}
WHERE [StoreName] IN ({placeholders});
"""
		return pd.read_sql(query, conn, params=list(store_names))
	finally:
		conn.close()


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	out["Monat_dt"] = pd.to_datetime(out["Monat"], errors="coerce")
	out["Jahr"] = out["Monat_dt"].dt.year
	out["Quartal"] = "Q" + out["Monat_dt"].dt.quarter.astype("Int64").astype(str)
	return out


def format_eur(val: float) -> str:
	try:
		if pd.isna(val):
			return "-"
		return f"{float(val):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
	except Exception:
		return "-"


def normalize_kenngroesse(value) -> str:
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return ""
	s = str(value)
	s = s.replace("\u00a0", " ")
	s = s.replace("€", " eur ")
	s = s.strip().lower()
	s = re.sub(r"[\t\r\n]+", " ", s)
	s = re.sub(r"[._\-/]+", " ", s)
	s = re.sub(r"\s+", " ", s).strip()
	return s


def calc_gesamtumsatz(df_filtered: pd.DataFrame) -> tuple[float, list[str]]:
	if "Kenngröße" not in df_filtered.columns or "Wert" not in df_filtered.columns:
		return 0.0, []

	candidates = {
		"umsatzeur",
		"umsatz eur",
		"umsatz",
		"revenue",
		"sales",
		"sales eur",
		"saleseur",
		"totalrevenue",
	}

	k = df_filtered["Kenngröße"].apply(normalize_kenngroesse)
	mask = k.isin(candidates)
	found = sorted(set(df_filtered.loc[mask, "Kenngröße"].astype(str).unique().tolist()))
	umsatz = float(df_filtered.loc[mask, "Wert"].sum()) if mask.any() else 0.0
	return umsatz, found


def drop_allgemein_columns(pivot: pd.DataFrame) -> pd.DataFrame:
	if pivot.empty:
		return pivot
	cols = pivot.columns
	if isinstance(cols, pd.MultiIndex):
		drop_cols = []
		for c in cols:
			if any(str(level) == "Allgemein" for level in c):
				drop_cols.append(c)
		if drop_cols:
			pivot = pivot.drop(columns=drop_cols)
	return pivot


def ensure_ebene_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Mappt DBEbene/Position auf Ebene/EPos, damit die Ebenen-Logik wie in V2 funktioniert."""
	out = df.copy()

	if "Ebene" not in out.columns and "DBEbene" in out.columns:
		ebene_map = {"DB1": "E1", "DB2": "E2", "DB3": "E3"}
		out["Ebene"] = (
			out["DBEbene"].astype(str).str.strip().map(ebene_map).fillna(out["DBEbene"].astype(str).str.strip())
		)
	if "EPos" not in out.columns and "Position" in out.columns:
		out["EPos"] = out["Position"]
	return out


def _pivot_for_kenngroesse(df_filtered: pd.DataFrame, kenngroesse_norm: str | list[str] | set[str] | tuple[str, ...]) -> pd.DataFrame:
	needed = {"Kenngröße", "ProduktLinie", "ProduktKategorie", "Wert"}
	if df_filtered.empty or not needed.issubset(df_filtered.columns):
		return pd.DataFrame()

	targets = (
		{normalize_kenngroesse(x) for x in kenngroesse_norm}
		if isinstance(kenngroesse_norm, (list, set, tuple))
		else {normalize_kenngroesse(kenngroesse_norm)}
	)

	tmp = df_filtered.copy()
	tmp["ProduktLinie"] = tmp["ProduktLinie"].fillna("Allgemein")
	tmp["ProduktKategorie"] = tmp["ProduktKategorie"].fillna("Allgemein")
	tmp["Wert"] = pd.to_numeric(tmp["Wert"], errors="coerce").fillna(0)
	tmp["_KenngroesseNorm"] = tmp["Kenngröße"].apply(normalize_kenngroesse)
	tmp = tmp[tmp["_KenngroesseNorm"].isin(targets)]
	if tmp.empty:
		return pd.DataFrame()

	g = tmp.groupby(["ProduktLinie", "ProduktKategorie"], as_index=False)["Wert"].sum()
	pivot = g.pivot_table(index=[], columns=["ProduktLinie", "ProduktKategorie"], values="Wert", aggfunc="sum")
	pivot.index = ["_row"]
	pivot[("Summen", "Gesamt")] = pivot.sum(axis=1)
	pivot = drop_allgemein_columns(pivot)
	pivot = pivot.sort_index(axis=1)
	return pivot


def compute_deckungsbeitraege(df_filtered: pd.DataFrame) -> dict:
	umsatz = _pivot_for_kenngroesse(df_filtered, ["UmsatzEUR", "Umsatz EUR", "umsatzeur"])
	transfer = _pivot_for_kenngroesse(df_filtered, ["TransferPriceEUR", "Transfer Price EUR", "transferpriceeur"])
	commission = _pivot_for_kenngroesse(df_filtered, ["Commission in EUR", "Commission", "commission in eur", "commission"])

	e3_cost_norms = [
		"additional procurement costs",
		"marketing campaign",
		"monthly rent",
		"monthly salary",
		"monthly social costs",
	]
	e3_cost_parts = [p for p in (_pivot_for_kenngroesse(df_filtered, k) for k in e3_cost_norms) if not p.empty]

	def _align(p: pd.DataFrame, cols: pd.Index) -> pd.DataFrame:
		if p.empty:
			return pd.DataFrame(index=["_row"], columns=cols).fillna(0)
		return p.reindex(columns=cols).fillna(0)

	all_cols = pd.Index([])
	for p in [umsatz, transfer, commission] + e3_cost_parts:
		if not p.empty:
			all_cols = all_cols.union(p.columns)
	if len(all_cols) == 0:
		return {"e1_total": pd.DataFrame(), "e2_total": pd.DataFrame(), "e3_total": pd.DataFrame(), "missing": ["UmsatzEUR", "TransferPriceEUR", "Commission"]}

	umsatz_a = _align(umsatz, all_cols)
	transfer_a = _align(transfer, all_cols)
	commission_a = _align(commission, all_cols)

	e3_cost_a = pd.DataFrame(index=["_row"], columns=all_cols).fillna(0)
	for p in e3_cost_parts:
		e3_cost_a = e3_cost_a.add(_align(p, all_cols), fill_value=0)

	e1_total = umsatz_a.add(transfer_a, fill_value=0)
	e2_total = e1_total.sub(commission_a, fill_value=0)
	e3_total = e2_total.sub(e3_cost_a, fill_value=0)

	if ("Summen", "Gesamt") not in all_cols:
		for df_ in (e1_total, e2_total, e3_total):
			df_[("Summen", "Gesamt")] = df_.sum(axis=1)

	missing = []
	if umsatz.empty:
		missing.append("UmsatzEUR")
	if transfer.empty:
		missing.append("TransferPriceEUR")
	if commission.empty:
		missing.append("Commission/Commission in EUR")

	return {"e1_total": e1_total, "e2_total": e2_total, "e3_total": e3_total, "missing": missing}


def _norm_dbe(value) -> str:
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return ""
	return str(value).strip().upper()


def build_db_matrix(df_filtered: pd.DataFrame) -> pd.DataFrame:
	"""DB-Rechnung als Matrix (Pivot) über den aktuell gefilterten Zeitraum.

	- Spalten: ProduktLinie × ProduktKategorie
	- Zeilen: Umsatz, DB1_Kosten, DB2_Kosten, DB3_Kosten, DB1, DB2, DB3
	- Fix-/Allgemein-Kosten (ohne Produktbezug) werden proportional zum Umsatz verteilt.
	"""
	needed = {"Kenngröße", "Wert", "DBEbene", "ProduktLinie", "ProduktKategorie"}
	if df_filtered.empty or not needed.issubset(df_filtered.columns):
		return pd.DataFrame()

	df = df_filtered.copy()
	df["Wert"] = pd.to_numeric(df["Wert"], errors="coerce").fillna(0)
	df["_DBEbeneNorm"] = df["DBEbene"].apply(_norm_dbe)

	# Produkt-Spalten definieren (nur echte Produkt-Zuordnungen, keine NULLs)
	prod_mask = df["ProduktLinie"].notna() & df["ProduktKategorie"].notna()
	df_prod = df[prod_mask].copy()
	if df_prod.empty:
		return pd.DataFrame()

	# Spalten (ProduktLinie, ProduktKategorie)
	col_pairs = (
		df_prod[["ProduktLinie", "ProduktKategorie"]]
		.drop_duplicates()
		.astype(str)
		.sort_values(["ProduktLinie", "ProduktKategorie"], kind="stable")
	)
	cols_mi = pd.MultiIndex.from_frame(col_pairs)

	# Umsatz pro Produkt (Basis für Verteilung)
	umsatz_candidates = {"umsatzeur", "umsatz eur", "umsatz", "revenue", "sales", "sales eur", "saleseur", "totalrevenue"}
	df_prod["_KenngroesseNorm"] = df_prod["Kenngröße"].apply(normalize_kenngroesse)
	umsatz_prod = (
		df_prod[df_prod["_KenngroesseNorm"].isin(umsatz_candidates)]
		.groupby(["ProduktLinie", "ProduktKategorie"])["Wert"]
		.sum()
		.reindex(cols_mi, fill_value=0)
	)
	total_umsatz = float(umsatz_prod.sum())
	if total_umsatz != 0:
		shares = umsatz_prod / total_umsatz
	else:
		# Ohne Umsatz kann man Fixkosten nicht sinnvoll verteilen.
		shares = umsatz_prod * 0

	# Kostenstellen-Spalte: bevorzugt Position, sonst Kostenstelle, sonst Kenngröße
	cost_col = "Kenngröße"
	for candidate in ("Position", "Kostenstelle"):
		if candidate in df.columns:
			cost_col = candidate
			break

	def _sum_level(level: str, exclude_umsatz: bool) -> pd.Series:
		lvl_mask = df["_DBEbeneNorm"] == level
		df_lvl = df[lvl_mask].copy()
		if df_lvl.empty:
			return pd.Series(0, index=cols_mi)

		if exclude_umsatz:
			df_lvl["_KenngroesseNorm"] = df_lvl["Kenngröße"].apply(normalize_kenngroesse)
			df_lvl = df_lvl[~df_lvl["_KenngroesseNorm"].isin(umsatz_candidates)]
			if df_lvl.empty:
				return pd.Series(0, index=cols_mi)

		prod_mask_lvl = df_lvl["ProduktLinie"].notna() & df_lvl["ProduktKategorie"].notna()
		prod_part = (
			df_lvl[prod_mask_lvl]
			.groupby(["ProduktLinie", "ProduktKategorie"])["Wert"]
			.sum()
			.reindex(cols_mi, fill_value=0)
		)

		# Allgemein-Pool: alles ohne Produktzuordnung (NULL in Linie oder Kategorie)
		general_total = float(df_lvl[~prod_mask_lvl]["Wert"].sum())
		allocated = shares * general_total
		return prod_part.add(allocated, fill_value=0)

	def _sum_level_by_costcenter(level: str) -> pd.DataFrame:
		lvl_mask = df["_DBEbeneNorm"] == level
		df_lvl = df[lvl_mask].copy()
		if df_lvl.empty:
			return pd.DataFrame(index=pd.Index([], dtype=str), columns=cols_mi)

		# Umsatz-Zeilen aus Kostenstellen-Breakdown entfernen
		df_lvl["_KenngroesseNorm"] = df_lvl["Kenngröße"].apply(normalize_kenngroesse)
		df_lvl = df_lvl[~df_lvl["_KenngroesseNorm"].isin(umsatz_candidates)]
		if df_lvl.empty:
			return pd.DataFrame(index=pd.Index([], dtype=str), columns=cols_mi)

		if cost_col != "Kenngröße":
			df_lvl["_CostCenter"] = df_lvl[cost_col].where(df_lvl[cost_col].notna(), df_lvl["Kenngröße"])
		else:
			df_lvl["_CostCenter"] = df_lvl["Kenngröße"]
		df_lvl["_CostCenter"] = df_lvl["_CostCenter"].astype(str)

		prod_mask_lvl = df_lvl["ProduktLinie"].notna() & df_lvl["ProduktKategorie"].notna()
		prod_pivot = pd.DataFrame(index=pd.Index([], dtype=str), columns=cols_mi)
		if prod_mask_lvl.any():
			prod_pivot = df_lvl[prod_mask_lvl].pivot_table(
				index="_CostCenter",
				columns=["ProduktLinie", "ProduktKategorie"],
				values="Wert",
				aggfunc="sum",
				fill_value=0,
			)
			prod_pivot = prod_pivot.reindex(columns=cols_mi, fill_value=0)

		general_totals = df_lvl[~prod_mask_lvl].groupby("_CostCenter")["Wert"].sum()
		if general_totals.empty:
			allocated = pd.DataFrame(index=pd.Index([], dtype=str), columns=cols_mi)
		else:
			allocated = pd.DataFrame({cc: (shares * float(val)) for cc, val in general_totals.items()}).T
			allocated = allocated.reindex(columns=cols_mi, fill_value=0)

		out = prod_pivot.add(allocated, fill_value=0)
		out = out.fillna(0)
		out.index = out.index.astype(str)
		return out

	# Kosten pro Ebene (ohne Umsatz) + Kostenstellen-Details
	db1_costcenters = _sum_level_by_costcenter("DB1")
	db2_costcenters = _sum_level_by_costcenter("DB2")
	db3_costcenters = _sum_level_by_costcenter("DB3")

	db1_kosten = db1_costcenters.sum(axis=0) if not db1_costcenters.empty else pd.Series(0, index=cols_mi)
	db2_kosten = db2_costcenters.sum(axis=0) if not db2_costcenters.empty else pd.Series(0, index=cols_mi)
	db3_kosten = db3_costcenters.sum(axis=0) if not db3_costcenters.empty else pd.Series(0, index=cols_mi)

	# Kumulativ
	db1 = umsatz_prod.add(db1_kosten, fill_value=0)
	db2 = db1.add(db2_kosten, fill_value=0)
	db3 = db2.add(db3_kosten, fill_value=0)

	rows_list: list[tuple[str, pd.Series]] = []
	rows_list.append(("Umsatz", umsatz_prod.reindex(cols_mi, fill_value=0)))

	if not db1_costcenters.empty:
		for cc in sorted(db1_costcenters.index.tolist(), key=lambda x: str(x)):
			rows_list.append((f"DB1 | {cc}", db1_costcenters.loc[cc].reindex(cols_mi, fill_value=0)))
	rows_list.append(("DB1 Kosten", db1_kosten.reindex(cols_mi, fill_value=0)))
	rows_list.append(("DB1", db1.reindex(cols_mi, fill_value=0)))

	if not db2_costcenters.empty:
		for cc in sorted(db2_costcenters.index.tolist(), key=lambda x: str(x)):
			rows_list.append((f"DB2 | {cc}", db2_costcenters.loc[cc].reindex(cols_mi, fill_value=0)))
	rows_list.append(("DB2 Kosten", db2_kosten.reindex(cols_mi, fill_value=0)))
	rows_list.append(("DB2", db2.reindex(cols_mi, fill_value=0)))

	if not db3_costcenters.empty:
		for cc in sorted(db3_costcenters.index.tolist(), key=lambda x: str(x)):
			rows_list.append((f"DB3 | {cc}", db3_costcenters.loc[cc].reindex(cols_mi, fill_value=0)))
	rows_list.append(("DB3 Kosten", db3_kosten.reindex(cols_mi, fill_value=0)))
	rows_list.append(("DB3", db3.reindex(cols_mi, fill_value=0)))

	result = pd.DataFrame([s for _, s in rows_list], index=[name for name, _ in rows_list])
	result[("Summen", "Gesamt")] = result.sum(axis=1)
	result = result.sort_index(axis=1)
	return result


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="App V3 – G15 FR/R", layout="wide")
st.title("Final Table – Kosten & Totals (G15)")

with st.sidebar:
	st.header("Filter")
	if st.button("🔄 Daten aktualisieren"):
		load_store_names.clear()
		load_g15_data.clear()
		st.rerun()

store_options = load_store_names()

default_stores = []
for candidate in ["Rosenheim", "Freiburg im Breisgau"]:
	if candidate in store_options:
		default_stores.append(candidate)
if not default_stores and store_options:
	default_stores = store_options[:1]

with st.sidebar:
	selected_stores = st.multiselect("StoreName", store_options, default=default_stores)

df_raw = load_g15_data(tuple(selected_stores))
if df_raw.empty:
	st.warning("Keine Daten geladen. Bitte StoreName/Verbindung prüfen.")
	st.stop()

required_cols = {"StoreName", "Monat", "Kenngröße", "Wert"}
missing_cols = sorted(required_cols.difference(df_raw.columns))
if missing_cols:
	st.error(
		"Erwartete Spalten fehlen: "
		+ ", ".join(missing_cols)
		+ "\n\nVerfügbare Spalten:\n- "
		+ "\n- ".join(map(str, df_raw.columns))
	)
	st.stop()

df = add_time_columns(df_raw)
df = ensure_ebene_columns(df)

with st.sidebar:
	jahre = sorted([int(x) for x in df["Jahr"].dropna().unique().tolist()], reverse=True)
	if not jahre:
		st.warning("Keine gültigen Monate/Jahre gefunden.")
		st.stop()
	selected_jahr = st.selectbox("Jahr", jahre)

df_jahr = df[df["Jahr"] == selected_jahr]
with st.sidebar:
	present_quarters = sorted(df_jahr["Monat_dt"].dt.quarter.dropna().unique().tolist())
	quartal_options = ["Alle"] + [f"Q{q}" for q in present_quarters]
	selected_quartal = st.selectbox("Quartal", quartal_options)

df_scope = df_jahr
if selected_quartal != "Alle":
	df_scope = df_scope[df_scope["Quartal"] == selected_quartal]

with st.sidebar:
	month_map = df_scope[["Monat", "Monat_dt"]].drop_duplicates().sort_values("Monat_dt", kind="stable")
	monat_options = ["Alle"] + month_map["Monat"].astype(str).tolist()
	selected_monat = st.selectbox("Monat", monat_options)

df_filtered = df_jahr.copy()
if selected_quartal != "Alle":
	df_filtered = df_filtered[df_filtered["Quartal"] == selected_quartal]
if selected_monat != "Alle":
	df_filtered = df_filtered[df_filtered["Monat"].astype(str) == selected_monat]

if df_filtered.empty:
	st.info("Für den ausgewählten Zeitraum gibt es keine Daten.")
	st.stop()

if selected_monat != "Alle":
	zeitraum_titel = f"{selected_monat}"
elif selected_quartal != "Alle":
	zeitraum_titel = f"{selected_quartal} {selected_jahr}"
else:
	zeitraum_titel = f"Gesamtjahr {selected_jahr}"

store_title = ", ".join(selected_stores) if selected_stores else "(keine Auswahl)"
st.subheader(f"Übersicht: {store_title} – {zeitraum_titel}")

db = compute_deckungsbeitraege(df_filtered)
sum_total = 0.0
if not db["e3_total"].empty and ("Summen", "Gesamt") in db["e3_total"].columns:
	sum_total = float(db["e3_total"].iloc[0][("Summen", "Gesamt")])

if db.get("missing"):
	st.info("Hinweis: Für die Deckungsbeitrags-Rechnung fehlen Kenngrößen: " + ", ".join(db["missing"]))

profitabel = sum_total > 0
status_text = "✅ Profitabel" if profitabel else "❌ Nicht profitabel"

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

st.subheader("DB Rechnung (Matrix)")

def format_german(val):
	if pd.isna(val):
		return "-"
	return "{:,.2f} €".format(float(val)).replace(",", "X").replace(".", ",").replace("X", ".")


db_matrix = build_db_matrix(df_filtered)
if db_matrix.empty:
	st.info("DB-Matrix konnte nicht berechnet werden (fehlende Produkt-/DBEbene-Daten).")
else:
	# Profitabilität aus DB3 (Summen/Gesamt)
	if ("Summen", "Gesamt") in db_matrix.columns and "DB3" in db_matrix.index:
		sum_total = float(db_matrix.loc["DB3", ("Summen", "Gesamt")])
		profitabel = sum_total > 0
		status_text = "✅ Profitabel" if profitabel else "❌ Nicht profitabel"
		# Metrics oben aktualisieren (damit Status/Total zur Matrix passt)
		c0, c1, c2 = st.columns(3)
		c0.metric("Gesamtumsatz", format_eur(gesamtumsatz))
		c1.metric("DB3 (Summe Gesamt)", format_eur(sum_total))
		c2.metric("Status", status_text)

	st.dataframe(
		db_matrix.style.format(format_german),
		use_container_width=True,
		height=360,
	)

st.subheader("Daten (SQL-Ansicht)")

df_sql = df_filtered.copy()
sort_cols = [
	c
	for c in ["Monat_dt", "DBEbene", "Position", "Ebene", "EPos", "Kenngröße", "ProduktKategorie", "ProduktLinie"]
	if c in df_sql.columns
]
if sort_cols:
	df_sql = df_sql.sort_values(sort_cols, kind="stable")

# SQL-like Spaltenreihenfolge
preferred_order = [
	"StoreName",
	"Kenngröße",
	"Wert",
	"DBEbene",
	"Position",
	"Ebene",
	"EPos",
	"Beschreibung",
	"Kostenart",
	"ProduktNr",
	"ProduktKategorie",
	"ProduktLinie",
	"Kampagne",
	"Monat",
]

cols = [c for c in preferred_order if c in df_sql.columns] + [c for c in df_sql.columns if c not in preferred_order and c not in {"Monat_dt", "Jahr", "Quartal"}]
df_sql = df_sql[cols]

st.dataframe(df_sql, use_container_width=True, height=520)

