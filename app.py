# Ruta: app.py
# Archivo: app.py

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Underwriting", layout="wide")

import os
import requests
from dateutil.relativedelta import relativedelta

from underwriting.infrastructure.config import load_settings
from underwriting.infrastructure.syntage_client import SyntageClient
from underwriting.application.sat_service import SatService
from underwriting.application.cfdi_service import CfdiService
from underwriting.ui.sat_views import render_tax_status_cards
from underwriting.ui.cfdi_views import render_prodserv_dual_cards
from types import SimpleNamespace
from underwriting.application.buro_service import obtener_buro_moffin_por_rfc



# Soporte para Streamlit Cloud Secrets
try:
    if st.secrets:
        for k, v in st.secrets.items():
            os.environ[str(k)] = str(v)
except Exception:
    pass



st.markdown(
    """
<style>
/* ==========================
   TOP INPUT BAR - PRO LOOK
   ========================== */

/* Barra fija arriba */
div[data-testid="stExpander"]{
  position: fixed !important;
  top: 4.0rem !important;                 /* un poco más abajo */
  left: 0.5rem !important;
  right: 0.5rem !important;
  z-index: 999999 !important;

  background: #ffffff !important;         /* sólido */
  border-radius: 12px !important;
  border: 1px solid rgba(20, 20, 20, 0.08) !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
  overflow: hidden !important;
}

/* Contenedor interno */
div[data-testid="stExpander"] > details{
  border: none !important;
  border-radius: 12px !important;
  background: #ffffff !important;
}

/* Header minimal */
div[data-testid="stExpander"] > details > summary{
  padding: 0.35rem 0.75rem !important;
  font-size: 0.90rem !important;
  font-weight: 600 !important;
  line-height: 1.0 !important;
  color: rgba(20,20,20,0.75) !important;
}

/* Cuerpo sin scroll, ultra compacto */
div[data-testid="stExpander"] > details > div{
  max-height: none !important;
  overflow: visible !important;
  padding: 0.35rem 0.75rem 0.55rem 0.75rem !important;
}

/* Compacta el espacio vertical entre widgets */
div[data-testid="stExpander"] .stMarkdown,
div[data-testid="stExpander"] .stTextInput,
div[data-testid="stExpander"] .stDateInput,
div[data-testid="stExpander"] .stRadio,
div[data-testid="stExpander"] .stButton{
  margin-bottom: 0.10rem !important;
}

/* Labels más pequeños y pro */
div[data-testid="stExpander"] label{
  font-size: 0.78rem !important;
  color: rgba(20,20,20,0.70) !important;
  font-weight: 600 !important;
}

/* Inputs (RFC / carpeta / date) estilo “enterprise” */
div[data-testid="stExpander"] input,
div[data-testid="stExpander"] textarea{
  border-radius: 10px !important;
  border: 1px solid rgba(20,20,20,0.10) !important;
  background: #fbfbfc !important;
}

/* Date input container */
div[data-testid="stExpander"] div[data-baseweb="input"]{
  border-radius: 10px !important;
  background: #fbfbfc !important;
}

/* Botones generales (Calcular/Cancelar) */
div[data-testid="stExpander"] button{
  border-radius: 10px !important;
  min-height: 34px !important;
  font-size: 0.80rem !important;
  font-weight: 700 !important;
  padding: 0.25rem 0.70rem !important;
  white-space: nowrap !important;         /* evita “Rech azar” */
}

/* Botón primario más “premium” */
div[data-testid="stExpander"] button[kind="primary"]{
  border-radius: 10px !important;
}

/* Chips de rango (1M,3M,...): super compactos y elegantes */
div[data-testid="stExpander"] button[kind="secondary"]{
  padding: 0.10rem 0.35rem !important;
  min-height: 26px !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  border-radius: 999px !important;        /* pill */
  border: 1px solid rgba(20,20,20,0.12) !important;
  background: #ffffff !important;
}

/* Hover suave para chips */
div[data-testid="stExpander"] button[kind="secondary"]:hover{
  background: rgba(20,20,20,0.04) !important;
}

/* Reduce el padding interno de columnas para que todo se vea alineado */
div[data-testid="stExpander"] [data-testid="column"]{
  padding-top: 0.0rem !important;
  padding-bottom: 0.0rem !important;
}

/* Divider más discreto */
div[data-testid="stExpander"] hr{
  margin: 0.45rem 0 !important;
  border-color: rgba(20,20,20,0.08) !important;
}

/* Evita que la barra tape el contenido */
section.main > div.block-container{
  padding-top: 165px !important;
}
</style>
""",
    unsafe_allow_html=True,
)
# =============================================================================

@st.cache_resource
def get_service() -> SatService:
    settings = load_settings()
    client = SyntageClient(settings=settings)
    return SatService(client=client)


@st.cache_resource
def get_cfdi_service() -> CfdiService:
    settings = load_settings()
    client = SyntageClient(settings=settings)
    return CfdiService(client=client)


@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_tax_status(rfc: str):
    service = get_service()
    return service.get_tax_status(rfc)


@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_cfdi(rfc: str, source: str, date_from: date | None, date_to: date | None, local_dir: str):
    service = get_cfdi_service()
    if source == "local":
        return service.fetch_local_xml(local_dir)
    if source == "syntage":
        return service.fetch_syntage_xml(rfc=rfc, date_from=date_from, date_to=date_to)
    raise ValueError(f"Fuente CFDI inválida: {source!r}")


@st.cache_data(show_spinner=False)
def load_ps_catalog() -> pd.DataFrame:
    p = ROOT / "src" / "underwriting" / "assets" / "catalogo_productos_servicios_SAT.csv"
    if not p.exists():
        return pd.DataFrame(columns=["clave_prodserv", "producto"])

    df = pd.read_csv(p, dtype=str)
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    if "clave_prodserv" not in df.columns:
        df = df.rename(columns={df.columns[0]: "clave_prodserv"})
    if "producto" not in df.columns:
        for c in df.columns:
            if c in {"descripcion", "description", "desc"}:
                df = df.rename(columns={c: "producto"})
                break
    if "producto" not in df.columns and len(df.columns) > 1:
        df = df.rename(columns={df.columns[1]: "producto"})

    return df[["clave_prodserv", "producto"]].copy()

# =============================================================================
# Syntage: Concentración (Top clientes / proveedores) - con caché
# =============================================================================
SYNTAGE_BASE_URL = "https://api.syntage.com"


@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_syntage_concentration(rfc: str, kind: str, from_dt: str | None = None, to_dt: str | None = None):
    """
    kind:
      - "customer"  -> /insights/{rfc}/customer-concentration
      - "supplier"  -> /insights/{rfc}/supplier-concentration
    from_dt / to_dt: ISO UTC string, e.g. "2025-01-01T00:00:00Z"
    """
    api_key = os.getenv("SYNTAGE_API_KEY", "")
    if not api_key:
        return None

    r = (rfc or "").strip().upper()
    if not r:
        return None

    if kind == "customer":
        path = "customer-concentration"
    elif kind == "supplier":
        path = "supplier-concentration"
    else:
        return None

    url = f"{SYNTAGE_BASE_URL}/insights/{r}/{path}"

    params = {}
    if from_dt:
        params["options[from]"] = from_dt
    if to_dt:
        params["options[to]"] = to_dt

    try:
        resp = requests.get(
            url,
            headers={"X-API-Key": api_key},
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        obj = resp.json()
        data = obj.get("data")
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _conc_to_df(conc_list) -> pd.DataFrame:
    """
    conc_list: lista de dicts con keys típicas: rfc, name, total, share, transactions
    Devuelve DF con columnas display (share/total formateadas) + columnas numéricas auxiliares:
      - _share_num (0..100)
      - _total_num  (monto)
    """
    if not conc_list:
        return pd.DataFrame(columns=["name", "rfc", "share", "total", "transactions", "_share_num", "_total_num"])

    df = pd.DataFrame(conc_list)

    # Normaliza columnas esperadas
    for c in ["rfc", "name", "total", "share", "transactions"]:
        if c not in df.columns:
            df[c] = None

    # Coerciones suaves (numéricas en columnas auxiliares)
    df["name"] = df["name"].astype(str).fillna("")
    df["rfc"] = df["rfc"].astype(str).fillna("")
    df["_total_num"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0)
    df["_share_num"] = pd.to_numeric(df["share"], errors="coerce").fillna(0.0)
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype(int)

    # Orden por share desc
    df = df.sort_values("_share_num", ascending=False).head(10).reset_index(drop=True)

    # Si share viene 0..1 lo convertimos a %
    if df["_share_num"].max() <= 1.0:
        df["_share_num"] = df["_share_num"] * 100.0

    # Formatos para mostrar
    df["share"] = df["_share_num"].map(lambda x: f"{x:,.2f}%")
    df["total"] = df["_total_num"].map(lambda x: f"${x:,.2f}")

    # Orden final (incluye auxiliares al final para gráficos)
    return df[["name", "rfc", "share", "total", "transactions", "_share_num", "_total_num"]].copy()


def _top10_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """DF para mostrar en tabla Top10: quita 'transactions' y columnas auxiliares."""
    if df is None or df.empty:
        return df
    drop_cols = [c for c in ["transactions", "_share_num", "_total_num"] if c in df.columns]
    return df.drop(columns=drop_cols) if drop_cols else df

def _with_color_legend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega una columna visual de color (●) para mapear filas ↔ dona.
    No modifica datos originales.
    """
    if df is None or df.empty:
        return df

    # paleta (debe coincidir con la usada en la dona)
    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]

    d = df.copy().reset_index(drop=True)
    colors = [palette[i % len(palette)] for i in range(len(d))]

    # ● más grande + centrado
    d.insert(
        0,
        " ",
        [f"<span style='color:{c}; font-size:26px; line-height:1'>●</span>" for c in colors],
    )
    return d

def _apply_any_column_filters(df: pd.DataFrame, *, key_prefix: str) -> pd.DataFrame:
    """
    Filtro genérico por cualquier columna:
    - numéricas: rango min/max
    - fechas/datetime: rango
    - texto/categóricas: contiene + selección de valores (si son pocos)
    """
    if df is None or df.empty:
        return df

    d = df.copy()

    with st.container(border=True):
        st.markdown("#### Filtros")

        cols = list(d.columns)
        sel_cols = st.multiselect(
            "Columnas a filtrar",
            options=cols,
            default=[],
            key=f"{key_prefix}__cols",
        )

        for c in sel_cols:
            s = d[c]

            # intenta detectar datetime
            is_dt = False
            s_dt = None
            try:
                s_dt = pd.to_datetime(s, errors="coerce")
                is_dt = s_dt.notna().any() and (s_dt.notna().mean() > 0.5)
            except Exception:
                is_dt = False

            if is_dt:
                vmin = s_dt.min()
                vmax = s_dt.max()
                if pd.isna(vmin) or pd.isna(vmax):
                    st.info(f"'{c}': no hay fechas válidas para filtrar.")
                    continue

                c1, c2 = st.columns(2)
                with c1:
                    d_from = st.date_input(
                        f"{c} · Desde",
                        value=vmin.date(),
                        key=f"{key_prefix}__{c}__dt_from",
                    )
                with c2:
                    d_to = st.date_input(
                        f"{c} · Hasta",
                        value=vmax.date(),
                        key=f"{key_prefix}__{c}__dt_to",
                    )

                mask = s_dt.dt.date.between(d_from, d_to)
                d = d.loc[mask].copy()
                continue

            # numérico
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.notna().any() and (s_num.notna().mean() > 0.7):
                mn = float(s_num.min())
                mx = float(s_num.max())
                if mn == mx:
                    st.caption(f"{c}: valor único {mn}")
                    continue

                r = st.slider(
                    f"{c} · Rango",
                    min_value=mn,
                    max_value=mx,
                    value=(mn, mx),
                    key=f"{key_prefix}__{c}__num_rng",
                )
                d = d.loc[s_num.between(r[0], r[1])].copy()
                continue

            # texto/categórico
            s_txt = s.astype(str).fillna("")
            c1, c2 = st.columns([1.3, 1.7])

            with c1:
                q = st.text_input(
                    f"{c} · Contiene",
                    value="",
                    key=f"{key_prefix}__{c}__txt",
                    placeholder="Ej. 'SERVICIO', 'MX', '01010101'...",
                )

            with c2:
                uniques = sorted([u for u in s_txt.unique().tolist() if u not in {"None", "nan"}])
                use_select = len(uniques) <= 200
                selected = st.multiselect(
                    f"{c} · Valores",
                    options=uniques if use_select else [],
                    default=[],
                    key=f"{key_prefix}__{c}__vals",
                    disabled=not use_select,
                    help=None if use_select else "Demasiados valores únicos para listar.",
                )

            if q:
                d = d.loc[s_txt.str.contains(q, case=False, na=False)].copy()
                s_txt = d[c].astype(str).fillna("")

            if selected:
                d = d.loc[s_txt.isin(selected)].copy()

    return d


def _render_donut(df: pd.DataFrame, *, title: str, value_col: str = "_total_num", label_col: str = "name") -> None:
    """Gráfica de anillo (donut) usando Altair (sin matplotlib), con colores alineados a la tabla."""
    if df is None or df.empty:
        st.info("Sin datos para el periodo.")
        return
    if value_col not in df.columns or label_col not in df.columns:
        st.info("Sin columnas suficientes para graficar distribución.")
        return

    d = df[[label_col, value_col]].copy().reset_index(drop=True)
    d[label_col] = d[label_col].astype(str).fillna("")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0.0)

    if float(d[value_col].sum()) <= 0:
        st.info("Sin montos positivos para graficar distribución.")
        return

    # paleta EXACTA (debe coincidir con _with_color_legend)
    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]

    # dominio en el MISMO orden que la tabla (orden de filas)
    domain = d[label_col].tolist()
    rng = palette[: len(domain)]

    import altair as alt

    chart = (
        alt.Chart(d)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta(field=value_col, type="quantitative"),
            color=alt.Color(
                field=label_col,
                type="nominal",
                legend=None,
                sort=domain,  # fuerza orden
                scale=alt.Scale(domain=domain, range=rng),  # fuerza color por categoría
            ),
            tooltip=[
                alt.Tooltip(label_col, type="nominal"),
                alt.Tooltip(value_col, type="quantitative", format=",.2f"),
            ],
        )
        .properties(title=title, height=260)
    )

    st.altair_chart(chart, width='stretch')



# =============================================================================
# KPI helpers (idénticos a Shiny)
# =============================================================================
def _or0(x) -> float:
    try:
        if x is None:
            return 0.0
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _num_s(s) -> pd.Series:
    if isinstance(s, pd.Series):
        x = s.astype(str)
    else:
        x = pd.Series(s).astype(str)
    x = x.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
    return pd.to_numeric(x, errors="coerce").fillna(0.0)


def _ensure_header_cols(headers: pd.DataFrame | None) -> pd.DataFrame:
    if headers is None or not isinstance(headers, pd.DataFrame) or headers.empty:
        return pd.DataFrame(columns=["total", "emisor_rfc", "receptor_rfc", "tipo", "uuid"])
    H = headers.copy()
    H.columns = [str(c).strip().lower() for c in H.columns]

    # Normaliza nombres básicos si vienen con variantes
    ren = {}
    if "total" not in H.columns:
        for c in H.columns:
            if c in {"importe", "monto", "total_cfdi", "totalfactura"}:
                ren[c] = "total"
                break
    if "emisor_rfc" not in H.columns:
        for c in H.columns:
            if c in {"rfc_emisor", "emisorrfc", "emisor"}:
                ren[c] = "emisor_rfc"
                break
    if "receptor_rfc" not in H.columns:
        for c in H.columns:
            if c in {"rfc_receptor", "receptorrfc", "receptor"}:
                ren[c] = "receptor_rfc"
                break
    if "tipo" not in H.columns:
        for c in H.columns:
            if c in {"tipocfdi", "tipo_cfdi"}:
                ren[c] = "tipo"
                break
    if "uuid" not in H.columns:
        for c in H.columns:
            if c in {"id", "folio_fiscal", "foliofiscal"}:
                ren[c] = "uuid"
                break
    if ren:
        H = H.rename(columns=ren)

    # Asegura columnas
    for c in ["total", "emisor_rfc", "receptor_rfc", "tipo", "uuid"]:
        if c not in H.columns:
            H[c] = None

    H["emisor_rfc"] = H["emisor_rfc"].astype(str).str.strip().str.upper()
    H["receptor_rfc"] = H["receptor_rfc"].astype(str).str.strip().str.upper()
    H["tipo"] = H["tipo"].astype(str).str.strip().str.upper()
    H["uuid"] = H["uuid"].astype(str).str.strip()
    H["total"] = _num_s(H["total"])
    return H


def kpi_ingresos(headers: pd.DataFrame | None, rfc: str) -> float:
    H = _ensure_header_cols(headers)
    r = (rfc or "").strip().upper()
    base_pos = _or0(H.loc[(H["emisor_rfc"] == r) & (H["tipo"] == "I"), "total"].sum())
    base_neg = _or0(H.loc[(H["emisor_rfc"] == r) & (H["tipo"] == "E"), "total"].sum())
    # [CHANGE] Ingresos NO restan nómina (tipo N). Solo I emitidos - E emitidos.
    return base_pos - base_neg


def kpi_egresos(headers: pd.DataFrame | None, rfc: str, headers_emitidos: pd.DataFrame | None = None) -> float:
    H = _ensure_header_cols(headers)
    r = (rfc or "").strip().upper()
    base_pos = _or0(H.loc[(H["receptor_rfc"] == r) & (H["tipo"] == "I"), "total"].sum())
    base_neg = _or0(H.loc[(H["receptor_rfc"] == r) & (H["tipo"] == "E"), "total"].sum())
    # [KEEP] Egresos incluyen nómina: sumar CFDI tipo N emitidos por el RFC
    H_emit = _ensure_header_cols(headers_emitidos) if headers_emitidos is not None else H
    nomina_emit = _or0(H_emit.loc[(H_emit["emisor_rfc"] == r) & (H_emit["tipo"] == "N"), "total"].sum())
    return base_pos - base_neg + nomina_emit


def kpi_nomina(headers: pd.DataFrame | None, conceptos: pd.DataFrame | None, rfc: str) -> float:
    H = _ensure_header_cols(headers)
    r = (rfc or "").strip().upper()
    return _or0(H.loc[(H["emisor_rfc"] == r) & (H["tipo"] == "N"), "total"].sum())


def kpi_interes(headers: pd.DataFrame | None, conceptos: pd.DataFrame | None, rfc: str) -> float:
    H = _ensure_header_cols(headers)
    C = conceptos if isinstance(conceptos, pd.DataFrame) else pd.DataFrame()
    if H.empty or C.empty:
        return 0.0

    C2 = C.copy()
    C2.columns = [str(c).strip().lower() for c in C2.columns]

    # Normaliza campos mínimos (descripcion)
    desc_col = None
    for c in C2.columns:
        if c in {"descripcion", "description", "concepto", "conceptodescripcion"}:
            desc_col = c
            break
    if desc_col is None:
        desc = pd.Series([""] * len(C2))
    else:
        desc = C2[desc_col].astype(str).fillna("")

    # patrón "interes" tolerante (quita acentos)
    trans = str.maketrans({"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"})
    desc_n = desc.str.lower().str.translate(trans)
    rows_int = desc_n.str.contains("interes", regex=False)

    if not rows_int.any():
        return 0.0

    if "importe" not in C2.columns:
        C2["importe"] = 0
    C2["importe"] = _num_s(C2["importe"])

    if "uuid" not in C2.columns:
        # si no existe uuid, no hay forma de cruzar
        return 0.0

    by_uuid = (
        C2.loc[rows_int, ["uuid", "importe"]]
        .groupby("uuid", as_index=False)["importe"]
        .sum()
        .rename(columns={"importe": "monto_interes"})
    )
    if by_uuid.empty:
        return 0.0

    r = (rfc or "").strip().upper()
    H_rec = H.loc[H["receptor_rfc"] == r, ["uuid"]].copy()
    out = H_rec.merge(by_uuid, on="uuid", how="left")
    return _or0(out["monto_interes"].sum())

def _set_range_cb(months: int | None = None, years: int | None = None) -> None:
    today = date.today()
    if months is not None:
        st.session_state["cfdi_date_from"] = today - relativedelta(months=months)
        st.session_state["cfdi_date_to"] = today
    elif years is not None:
        st.session_state["cfdi_date_from"] = today - relativedelta(years=years)
        st.session_state["cfdi_date_to"] = today


def _reset_range_cb() -> None:
    st.session_state["cfdi_date_from"] = date.today() - timedelta(days=90)
    st.session_state["cfdi_date_to"] = date.today()

# =============================================================================
# Ventas / Gastos / Utilidad Fiscal (basado en KPIs anteriores)
# =============================================================================
def _pick_date_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    cols = [str(c).strip().lower() for c in df.columns]
    candidates = [
        "fecha",
        "fecha_emision",
        "fechaemision",
        "fecha_timbrado",
        "fechatimbrado",
        "fecha_cfdi",
        "fechacfdi",
        "fch_emision",
        "fch_timbrado",
        "date",
        "issued_at",
        "created_at",
        "timestamp",
    ]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    for c in df.columns:
        name = str(c).strip().lower()
        if "fecha" in name or "date" in name or "timbr" in name or "emisi" in name:
            return c
    return None


def _with_dt(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["dt"])
    dcol = _pick_date_col(df)
    out = df.copy()
    if dcol is None:
        out["dt"] = pd.NaT
        return out
    out["dt"] = pd.to_datetime(out[dcol], errors="coerce")
    return out

def build_clientes_proveedores_tables(
    *,
    rfc: str,
    ing_headers: pd.DataFrame | None,
    egr_headers: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tablas desde CFDI (headers):
      - Clientes: emitidas (ing_headers) agrupado por receptor_rfc
      - Proveedores: recibidas (egr_headers) agrupado por emisor_rfc

    Requiere que headers ya traiga:
      emisor_rfc, receptor_rfc, fecha, uuid, subtotal, total
    y para nombre:
      emisor_nombre, receptor_nombre   (los agregamos en el parser)
    """

    r = (rfc or "").strip().upper()

    def _first_non_empty_name(s: pd.Series) -> str:
        try:
            for v in s.astype(str).fillna("").tolist():
                v2 = v.strip()
                if v2 and v2.lower() not in {"none", "nan"}:
                    return v2
        except Exception:
            pass
        return ""

    def _prep(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(
                columns=[
                    "uuid", "fecha", "subtotal", "total", "emisor_rfc", "receptor_rfc",
                    "emisor_nombre", "receptor_nombre",
                ]
            )

        d = df.copy()

        # Asegura columnas (sin romper si faltan)
        for c in [
            "uuid", "fecha", "subtotal", "total", "emisor_rfc", "receptor_rfc",
            "emisor_nombre", "receptor_nombre",
        ]:
            if c not in d.columns:
                d[c] = None

        d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
        d["subtotal"] = pd.to_numeric(d["subtotal"], errors="coerce").fillna(0.0)
        d["total"] = pd.to_numeric(d["total"], errors="coerce").fillna(0.0)

        d["emisor_rfc"] = d["emisor_rfc"].astype(str).str.strip().str.upper()
        d["receptor_rfc"] = d["receptor_rfc"].astype(str).str.strip().str.upper()
        d["uuid"] = d["uuid"].astype(str).str.strip()

        d["emisor_nombre"] = d["emisor_nombre"].astype(str).fillna("").str.strip()
        d["receptor_nombre"] = d["receptor_nombre"].astype(str).fillna("").str.strip()

        return d

    ing = _prep(ing_headers)
    egr = _prep(egr_headers)

    # ── CLIENTES (emitidas): receptor_rfc ─────────────────────────────────────
    ing2 = ing.loc[ing["emisor_rfc"] == r].copy() if not ing.empty else ing

    clientes = (
        ing2.groupby("receptor_rfc", as_index=False)
        .agg(
            Cliente=("receptor_nombre", _first_non_empty_name),
            **{
                "# CFDI": ("uuid", "nunique"),
                "Subtotal": ("subtotal", "sum"),
                "Total": ("total", "sum"),
                "Primera fecha": ("fecha", "min"),
                "Última fecha": ("fecha", "max"),
            }
        )
        .rename(columns={"receptor_rfc": "RFC"})
    )

    # ── PROVEEDORES (recibidas): emisor_rfc ───────────────────────────────────
    egr2 = egr.loc[egr["receptor_rfc"] == r].copy() if not egr.empty else egr

    proveedores = (
        egr2.groupby("emisor_rfc", as_index=False)
        .agg(
            Proveedor=("emisor_nombre", _first_non_empty_name),
            **{
                "# CFDI": ("uuid", "nunique"),
                "Subtotal": ("subtotal", "sum"),
                "Total": ("total", "sum"),
                "Primera fecha": ("fecha", "min"),
                "Última fecha": ("fecha", "max"),
            }
        )
        .rename(columns={"emisor_rfc": "RFC"})
    )

    # Formato final
    for df in (clientes, proveedores):
        if df is not None and not df.empty:
            df["Primera fecha"] = pd.to_datetime(df["Primera fecha"], errors="coerce").dt.date
            df["Última fecha"] = pd.to_datetime(df["Última fecha"], errors="coerce").dt.date
            df["Subtotal"] = df["Subtotal"].round(2)
            df["Total"] = df["Total"].round(2)
            df.sort_values(["Total"], ascending=False, inplace=True, ignore_index=True)

    return clientes, proveedores


def _slice_headers_by_date(headers: pd.DataFrame | None, d_from: date, d_to: date) -> pd.DataFrame:
    if headers is None or not isinstance(headers, pd.DataFrame) or headers.empty:
        return pd.DataFrame()

    h = headers.copy()
    # tu header trae 'fecha'
    h["_dt"] = pd.to_datetime(h["fecha"], errors="coerce")
    start = pd.Timestamp(d_from)
    end = pd.Timestamp(d_to) + pd.Timedelta(days=1)  # inclusivo
    out = h[(h["_dt"] >= start) & (h["_dt"] < end)].drop(columns=["_dt"])
    return out.reset_index(drop=True)


def _slice_conceptos_by_uuid(conceptos: pd.DataFrame | None, uuids: pd.Series) -> pd.DataFrame:
    if conceptos is None or not isinstance(conceptos, pd.DataFrame) or conceptos.empty:
        return pd.DataFrame()
    c = conceptos.copy()
    return c[c["uuid"].isin(set(uuids.astype(str)))].reset_index(drop=True)


def _slice_cfdi_data(cfdi_base: dict, d_from: date, d_to: date) -> dict:
    """Recorta cfdi_data base al rango solicitado (sin re-descargar)."""
    ing0 = cfdi_base.get("ingresos")
    egr0 = cfdi_base.get("egresos")

    ing_h = _slice_headers_by_date(getattr(ing0, "headers", None), d_from, d_to) if ing0 is not None else pd.DataFrame()
    egr_h = _slice_headers_by_date(getattr(egr0, "headers", None), d_from, d_to) if egr0 is not None else pd.DataFrame()

    ing_c = _slice_conceptos_by_uuid(getattr(ing0, "conceptos", None), ing_h["uuid"]) if ing0 is not None else pd.DataFrame()
    egr_c = _slice_conceptos_by_uuid(getattr(egr0, "conceptos", None), egr_h["uuid"]) if egr0 is not None else pd.DataFrame()

    out = {}
    out["meta"] = cfdi_base.get("meta") or {}

    out["ingresos"] = SimpleNamespace(headers=ing_h, conceptos=ing_c)
    out["egresos"] = SimpleNamespace(headers=egr_h, conceptos=egr_c)

    out["clientes_df"] = cfdi_base.get("clientes_df", pd.DataFrame())
    out["proveedores_df"] = cfdi_base.get("proveedores_df", pd.DataFrame())

    out["emit_invoices_df"] = cfdi_base.get("emit_invoices_df", pd.DataFrame())
    out["rec_invoices_df"] = cfdi_base.get("rec_invoices_df", pd.DataFrame())


    return out


def _params_key_for_base(rfc: str, cfdi_source: str, local_dir: str) -> tuple:
    return ((rfc or "").strip().upper(), cfdi_source, str(local_dir))


def _ensure_cfdi_from_base_or_fetch(*, rfc: str, cfdi_source: str, date_from: date, date_to: date, local_dir: str) -> dict:
    """
    Base inteligente:
    - Si el rango pedido está dentro de la base -> recorta (sin descargar).
    - Si el rango pedido se sale -> descarga la UNIÓN (min(from), max(to)) y guarda como nueva base.
      (La base nunca se hace más chica.)
    """
    base_key = _params_key_for_base(rfc, cfdi_source, local_dir)

    base = st.session_state.get("cfdi_data_base")
    base_meta = st.session_state.get("cfdi_data_base_meta")  # (key, from, to)

    # Caso: ya hay base para el mismo RFC/source/local_dir
    if base is not None and base_meta is not None:
        saved_key, saved_from, saved_to = base_meta
        if saved_key == base_key:
            # Si el rango está contenido, solo recorta
            if saved_from <= date_from and date_to <= saved_to:
                st.session_state["_cfdi_last_action"] = "slice"
                return _slice_cfdi_data(base, date_from, date_to)

            # Si NO está contenido, expandimos: descargamos la UNIÓN
            new_from = min(saved_from, date_from)
            new_to = max(saved_to, date_to)

            cfdi_new_base = fetch_cfdi(
                rfc=rfc,
                source=cfdi_source,
                date_from=new_from,
                date_to=new_to,
                local_dir=local_dir,
            )
            st.session_state["cfdi_data_base"] = cfdi_new_base
            st.session_state["cfdi_data_base_meta"] = (base_key, new_from, new_to)
            st.session_state["_cfdi_last_action"] = "fetch_union"

            return _slice_cfdi_data(cfdi_new_base, date_from, date_to)

    # Caso: no hay base (o cambió RFC/source/local_dir) -> descarga exacto y guarda base
    cfdi_new_base = fetch_cfdi(
        rfc=rfc,
        source=cfdi_source,
        date_from=date_from,
        date_to=date_to,
        local_dir=local_dir,
    )
    st.session_state["cfdi_data_base"] = cfdi_new_base
    st.session_state["cfdi_data_base_meta"] = (base_key, date_from, date_to)
    st.session_state["_cfdi_last_action"] = "fetch_exact"

    return _slice_cfdi_data(cfdi_new_base, date_from, date_to)




def _period_sum_ingresos(headers_emit: pd.DataFrame | None, rfc: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    H0 = _with_dt(headers_emit)
    H = _ensure_header_cols(H0)
    H["dt"] = H0.get("dt", pd.NaT)
    r = (rfc or "").strip().upper()
    m = (H["dt"] >= start) & (H["dt"] < end)
    base_pos = _or0(H.loc[m & (H["emisor_rfc"] == r) & (H["tipo"] == "I"), "total"].sum())
    base_neg = _or0(H.loc[m & (H["emisor_rfc"] == r) & (H["tipo"] == "E"), "total"].sum())
    return base_pos - base_neg


def _period_sum_egresos(headers_rec: pd.DataFrame | None, headers_emit: pd.DataFrame | None, rfc: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    Hr0 = _with_dt(headers_rec)
    Hr = _ensure_header_cols(Hr0)
    Hr["dt"] = Hr0.get("dt", pd.NaT)

    He0 = _with_dt(headers_emit)
    He = _ensure_header_cols(He0)
    He["dt"] = He0.get("dt", pd.NaT)

    r = (rfc or "").strip().upper()

    mrec = (Hr["dt"] >= start) & (Hr["dt"] < end)
    memi = (He["dt"] >= start) & (He["dt"] < end)

    base_pos = _or0(Hr.loc[mrec & (Hr["receptor_rfc"] == r) & (Hr["tipo"] == "I"), "total"].sum())
    base_neg = _or0(Hr.loc[mrec & (Hr["receptor_rfc"] == r) & (Hr["tipo"] == "E"), "total"].sum())
    nomina_emit = _or0(He.loc[memi & (He["emisor_rfc"] == r) & (He["tipo"] == "N"), "total"].sum())
    return base_pos - base_neg + nomina_emit


def _money(x: float) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def _set_range(months: int | None = None, years: int | None = None) -> None:
    today = date.today()
    if months is not None:
        st.session_state["cfdi_date_from_value"] = today - relativedelta(months=months)
        st.session_state["cfdi_date_to_value"] = today
    elif years is not None:
        st.session_state["cfdi_date_from_value"] = today - relativedelta(years=years)
        st.session_state["cfdi_date_to_value"] = today


def render_filterable_grid(df: pd.DataFrame, *, key: str) -> None:
    """Tabla con filtro por columna (AG Grid)."""
    if df is None or df.empty:
        st.info("Sin datos para el periodo.")
        return

    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode  # type: ignore

    gb = GridOptionsBuilder.from_dataframe(df)

    # ✅ filtros por columna + “floating filter” (la barrita de filtro bajo el header)
    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        floatingFilter=True,
    )

    # UI pro: paginación y fit
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=25)
    gb.configure_grid_options(domLayout="normal")

    grid_options = gb.build()

    AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=True,
        height=420,
        key=key,
        allow_unsafe_jscode=False,
        theme="streamlit",
    )

print("BOOT 4: about to render title")
# =============================================================================
st.title("Underwriting")

# =============================================================================
# TOPBAR
# =============================================================================
st.markdown('<div id="topbar">', unsafe_allow_html=True)
# Inicializa fechas SOLO si no existen (evita warning de Streamlit)
if "cfdi_date_from" not in st.session_state:
    st.session_state["cfdi_date_from"] = date.today() - timedelta(days=90)
if "cfdi_date_to" not in st.session_state:
    st.session_state["cfdi_date_to"] = date.today()

with st.expander("Inputs", expanded=True):
    # ---- defaults de fechas (se usan en la misma línea) ----
    default_from = st.session_state.get("cfdi_date_from") or (date.today() - timedelta(days=90))
    default_to = st.session_state.get("cfdi_date_to") or date.today()

    # =============================================================================
    # FILA ÚNICA: RFC + Calcular/Cancelar + Desde/Hasta + botones de rango
    # =============================================================================
    row = st.columns(
        [6.0, 2.0, 2.0, 2.4, 2.4] + [0.55] * 8 + [0.65]
    )  # RFC, Calcular, Cancelar, Desde, Hasta, 1M..5A, ⟳

    with row[0]:
        rfc = st.text_input("RFC", placeholder="PEIC211118IS0", key="inp_rfc").strip().upper()
    rfc_valid = 12 <= len(rfc) <= 13

    with row[1]:
        run = st.button("Calcular", type="primary", use_container_width=True
, disabled=not rfc_valid, key="btn_run")
    with row[2]:
        clear = st.button("Cancelar", use_container_width=True
, key="btn_clear")

    # ✅ IMPORTANTE: date_input toma valor de los mismos keys que modifican los botones
    with row[3]:
        st.date_input("Desde", key="cfdi_date_from")
    with row[4]:
        st.date_input("Hasta", key="cfdi_date_to")


    # ✅ Botones de rango usando callbacks (actualizan cfdi_date_from/cfdi_date_to)
    row[5].button("1M", use_container_width=True
, key="rng_1m", on_click=_set_range_cb, kwargs={"months": 1})
    row[6].button("3M", use_container_width=True
, key="rng_3m", on_click=_set_range_cb, kwargs={"months": 3})
    row[7].button("6M", use_container_width=True
, key="rng_6m", on_click=_set_range_cb, kwargs={"months": 6})
    row[8].button("1A", use_container_width=True
, key="rng_1y", on_click=_set_range_cb, kwargs={"years": 1})
    row[9].button("2A", use_container_width=True
, key="rng_2y", on_click=_set_range_cb, kwargs={"years": 2})
    row[10].button("3A", use_container_width=True
, key="rng_3y", on_click=_set_range_cb, kwargs={"years": 3})
    row[11].button("4A", use_container_width=True
, key="rng_4y", on_click=_set_range_cb, kwargs={"years": 4})
    row[12].button("5A", use_container_width=True
, key="rng_5y", on_click=_set_range_cb, kwargs={"years": 5})
    row[13].button("⟳", use_container_width=True
, key="rng_reset", on_click=_reset_range_cb)

    if clear:
        st.session_state.pop("tax_status", None)
        st.session_state.pop("last_rfc", None)
        for k in ["cfdi_data", "cfdi_date_from", "cfdi_date_to"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()

    # =============================================================================
    # FILA 2: Fuente + carpeta local (se queda igual)
    # =============================================================================
    c4, c5 = st.columns([2.2, 7.8])
    with c4:
        fuente = st.radio(
            "CFDI · Fuente",
            options=["Syntage (XML)", "Local"],
            horizontal=True,
            key="cfdi_source_radio",
        )
    cfdi_source = "syntage" if fuente == "Syntage (XML)" else "local"
    with c5:
        st.text_input(
            "CFDI · Carpeta local (XMLs)",
            value=st.session_state.get("cfdi_local_dir", str(ROOT / "data" / "cfdi_xml")),
            disabled=(cfdi_source != "local"),
            key="cfdi_local_dir",
        )


st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# AUTO-REFRESH: si cambian fechas/fuente y ya hay RFC cargado, refresca CFDI sin "Calcular"
# =============================================================================
_last_rfc = st.session_state.get("last_rfc")
_last_params = st.session_state.get("_last_cfdi_params")

_current_params = (
    st.session_state.get("cfdi_source_radio"),
    st.session_state.get("cfdi_local_dir"),
    st.session_state.get("cfdi_date_from"),
    st.session_state.get("cfdi_date_to"),
)

# Solo refresca si ya existe un RFC "cargado" previamente y cambió algo del filtro
if _last_rfc and _current_params != _last_params:
    try:
        local_dir = st.session_state.get("cfdi_local_dir", str(ROOT / "data" / "cfdi_xml"))
        st.session_state["cfdi_data"] = _ensure_cfdi_from_base_or_fetch(
            rfc=_last_rfc,
            cfdi_source=("syntage" if st.session_state.get("cfdi_source_radio") == "Syntage (XML)" else "local"),
            date_from=st.session_state.get("cfdi_date_from"),
            date_to=st.session_state.get("cfdi_date_to"),
            local_dir=local_dir,
        )
        st.session_state["_last_cfdi_params"] = _current_params
    except Exception as e:
        st.error(f"Error actualizando CFDIs con el nuevo filtro: {e}")


# =============================================================================
# Tabs
# =============================================================================
tabs = st.tabs(
    [
        "Análisis SAT",
        "Facturas",
        "Buró de Crédito",
        "Modelo",
        "Salud Financiera",
        "Documentos",
        "Cuentas Bancarias",
    ]
)

if run:
    if not rfc_valid:
        st.warning("RFC inválido. Debe tener 12 o 13 caracteres.")
    else:
        with st.spinner("Consultando SAT y CFDI..."):
            try:
                st.session_state["tax_status"] = fetch_tax_status(rfc)
                st.session_state["last_rfc"] = rfc
            except Exception as e:
                st.error(f"Error consultando SAT en Syntage: {e}")

            try:
                local_dir = st.session_state.get("cfdi_local_dir", str(ROOT / "data" / "cfdi_xml"))

                # ✅ Cambio: usa base-cache + recorte (en vez de re-descargar todo siempre)
                st.session_state["cfdi_data"] = _ensure_cfdi_from_base_or_fetch(
                    rfc=rfc,
                    cfdi_source=cfdi_source,
                    date_from=st.session_state.get("cfdi_date_from"),
                    date_to=st.session_state.get("cfdi_date_to"),
                    local_dir=local_dir,
                )

                st.session_state["_last_cfdi_params"] = (
                    st.session_state.get("cfdi_source_radio"),
                    st.session_state.get("cfdi_local_dir"),
                    st.session_state.get("cfdi_date_from"),
                    st.session_state.get("cfdi_date_to"),
                )

            except Exception as e:
                st.error(f"Error consultando CFDIs: {e}")


# =============================================================================
# TAB 0: SAT
# =============================================================================
with tabs[0]:
    st.caption("SAT →  Tax Status → Actividades económicas y regímenes")

    tax_status = st.session_state.get("tax_status")
    if tax_status:
        render_tax_status_cards(tax_status)

        cfdi_data = st.session_state.get("cfdi_data") or {}
        meta = cfdi_data.get("meta") or {}
        if meta:
            st.caption(
                f"CFDI meta: emit listed={meta.get('emit_listed')} downloaded={meta.get('emit_downloaded')} "
                f"failed={meta.get('emit_failed')} | rec listed={meta.get('rec_listed')} "
                f"downloaded={meta.get('rec_downloaded')} failed={meta.get('rec_failed')} "
                f"(workers={meta.get('max_workers')})"
            )

        if not cfdi_data:
            st.info("Para ver Prod/Serv, selecciona fuente/rango arriba y presiona Calcular.")
        else:
            catalogo = load_ps_catalog()
            service = get_cfdi_service()

            ing = cfdi_data.get("ingresos")
            egr = cfdi_data.get("egresos")

            resumen_ing = (
                service.prodserv_summary_shiny(
                    rfc=rfc,
                    headers=ing.headers,
                    conceptos=ing.conceptos,
                    catalogo=catalogo,
                    tipo="I",
                    rol="emisor",
                    top_n=25,
                )
                if ing is not None and ing.conceptos is not None and ing.headers is not None
                else pd.DataFrame()
            )

            resumen_egr = (
                service.prodserv_summary_shiny(
                    rfc=rfc,
                    headers=egr.headers,
                    conceptos=egr.conceptos,
                    catalogo=catalogo,
                    tipo="I",
                    rol="receptor",
                    top_n=25,
                )
                if egr is not None and egr.conceptos is not None and egr.headers is not None
                else pd.DataFrame()
            )

            render_prodserv_dual_cards(
                title_left="Prod/Serv (resumen)_ingresos",
                df_left=resumen_ing,
                title_right="Prod/Serv (resumen)_egresos",
                df_right=resumen_egr,
            )

            # =============================================================================
            # KPIs debajo de las cards (idénticos a Shiny)
            # =============================================================================
            h_ing = ing.headers if ing is not None else None
            c_ing = ing.conceptos if ing is not None else None
            h_egr = egr.headers if egr is not None else None
            c_egr = egr.conceptos if egr is not None else None

            k_ing = kpi_ingresos(h_ing, rfc)
            k_nom = kpi_nomina(h_ing, c_ing, rfc)
            k_egr = kpi_egresos(h_egr, rfc, headers_emitidos=h_ing)
            k_int = kpi_interes(h_egr, c_egr, rfc)

            kpi_cols = st.columns(2)
            with kpi_cols[0]:
                st.metric("KPI Ingresos (I - E emitidos)", f"${k_ing:,.2f}")
                st.metric("KPI Nómina (N emitidos)", f"${k_nom:,.2f}")
            with kpi_cols[1]:
                st.metric("KPI Egresos (I - E recibidos + Nómina emitida)", f"${k_egr:,.2f}")
                st.metric("KPI Interés (conceptos recibidos)", f"${k_int:,.2f}")

            # =============================================================================
            # Ventas / Gastos / Utilidad Fiscal (tabla + gráfico)
            # =============================================================================
            today = pd.Timestamp(st.session_state.get("cfdi_date_to") or date.today())
            y2 = today.year
            y1 = y2 - 1
            y0 = y2 - 2

            def _year_window(y: int) -> tuple[pd.Timestamp, pd.Timestamp]:
                return (pd.Timestamp(year=y, month=1, day=1), pd.Timestamp(year=y + 1, month=1, day=1))

            # últimos 12 meses (por mes calendario, 12 puntos)
            start_12m = pd.Timestamp(year=today.year, month=today.month, day=1) - relativedelta(months=11)
            end_next_month = (pd.Timestamp(year=today.year, month=today.month, day=1) + relativedelta(months=1))

            ventas_y0 = _period_sum_ingresos(h_ing, rfc, *_year_window(y0))
            ventas_y1 = _period_sum_ingresos(h_ing, rfc, *_year_window(y1))
            ventas_y2 = _period_sum_ingresos(h_ing, rfc, *_year_window(y2))
            ventas_12m = _period_sum_ingresos(h_ing, rfc, start_12m, end_next_month)

            gastos_y0 = _period_sum_egresos(h_egr, h_ing, rfc, *_year_window(y0))
            gastos_y1 = _period_sum_egresos(h_egr, h_ing, rfc, *_year_window(y1))
            gastos_y2 = _period_sum_egresos(h_egr, h_ing, rfc, *_year_window(y2))
            gastos_12m = _period_sum_egresos(h_egr, h_ing, rfc, start_12m, end_next_month)

            util_y0 = ventas_y0 - gastos_y0
            util_y1 = ventas_y1 - gastos_y1
            util_y2 = ventas_y2 - gastos_y2
            util_12m = ventas_12m - gastos_12m

            with st.container(border=True):
                st.markdown("### 💲 Ventas y Utilidad Fiscal")

                tbl = pd.DataFrame(
                    {
                        str(y0): [_money(ventas_y0), _money(gastos_y0), _money(util_y0)],
                        str(y1): [_money(ventas_y1), _money(gastos_y1), _money(util_y1)],
                        str(y2): [_money(ventas_y2), _money(gastos_y2), _money(util_y2)],
                        "Últimos 12 Meses": [_money(ventas_12m), _money(gastos_12m), _money(util_12m)],
                    },
                    index=["Ventas Anuales", "Gastos Anuales", "Utilidad Fiscal"],
                )
                st.dataframe(tbl, use_container_width=True
)

            # gráfico mensual de utilidad fiscal (últimos 12 meses) con barras ventas vs gastos
            months = pd.date_range(start=start_12m, end=end_next_month, freq="MS")[:12]
            rows = []
            for m in months:
                m2 = m + relativedelta(months=1)
                v = _period_sum_ingresos(h_ing, rfc, m, m2)
                g = _period_sum_egresos(h_egr, h_ing, rfc, m, m2)
                rows.append({"Mes": m.strftime("%Y-%m"), "Ventas": float(v), "Gastos": float(g), "Utilidad Fiscal": float(v - g)})

            monthly = pd.DataFrame(rows)

            with st.container(border=True):
                st.markdown("### 📈 Utilidad Fiscal últimos 12 meses")

                import altair as alt

                d_long = monthly.melt(id_vars=["Mes"], value_vars=["Ventas", "Gastos"], var_name="Tipo", value_name="Monto")

                chart = (
                    alt.Chart(d_long)
                    .mark_bar()
                    .encode(
                        x=alt.X("Mes:N", title="Mes"),
                        xOffset=alt.XOffset("Tipo:N"),
                        y=alt.Y("Monto:Q", title="Monto"),
                        color=alt.Color("Tipo:N"),
                        tooltip=["Mes:N", "Tipo:N", alt.Tooltip("Monto:Q", format=",.2f")],
                    )
                    .properties(height=320)
                )

                st.altair_chart(chart, width='stretch')

                        
            # =============================================================================
            # Top 10 Clientes / Proveedores (Syntage)
            # =============================================================================
            # Respeta filtro Desde/Hasta del usuario (si existe)
            d_from = st.session_state.get("cfdi_date_from_value") or st.session_state.get("cfdi_date_from")
            d_to = st.session_state.get("cfdi_date_to_value") or st.session_state.get("cfdi_date_to")

            # fallback por si algo viene vacío
            if d_from is None:
                d_from = date.today() - timedelta(days=365)
            if d_to is None:
                d_to = date.today()

            # ISO UTC (incluye día completo)
            from_dt = pd.Timestamp(d_from).strftime("%Y-%m-%dT00:00:00Z")
            to_dt = pd.Timestamp(d_to).strftime("%Y-%m-%dT23:59:59Z")

            conc_customers = fetch_syntage_concentration(rfc, kind="customer", from_dt=from_dt, to_dt=to_dt)
            conc_suppliers = fetch_syntage_concentration(rfc, kind="supplier", from_dt=from_dt, to_dt=to_dt)


            df_customers = _conc_to_df(conc_customers)
            df_suppliers = _conc_to_df(conc_suppliers)

            cols_conc = st.columns(2)
            with cols_conc[0]:
                with st.container(border=True):
                    st.markdown("### 🧑‍💼 Top 10 Clientes")
                    if df_customers.empty:
                        st.info("Sin datos de concentración de clientes para el periodo.")
                    else:
                        _render_donut(df_customers, title="Distribución")
                        df_show = _with_color_legend(_top10_display_df(df_customers))
                        st.markdown(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)


            with cols_conc[1]:
                with st.container(border=True):
                    st.markdown("### 🏭 Top 10 Proveedores")
                    if df_suppliers.empty:
                        st.info("Sin datos de concentración de proveedores para el periodo.")
                    else:
                        _render_donut(df_suppliers, title="Distribución")
                        df_show = _with_color_legend(_top10_display_df(df_suppliers))
                        st.markdown(df_show.to_html(escape=False, index=False), unsafe_allow_html=True)
            # =============================================================================
            # Prod/Serv (resumen)_egresos (MISMA card que en Facturas, con filtros)
            # =============================================================================
            with st.container(border=True):
                st.markdown("Productos y servicios comprados")

                if resumen_egr is None or resumen_egr.empty:
                    st.info("Sin datos para el periodo.")
                else:
                    render_filterable_grid(resumen_egr, key="sat_prodserv_egresos_grid")
            # =============================================================================
            # Headers (MISMA tabla que Facturas -> Headers) con el MISMO formato (AG Grid)
            # =============================================================================
            with st.container(border=True):
                st.markdown("Facturas (headers)")

                headers_all = []

                if ing is not None and ing.headers is not None and not ing.headers.empty:
                    h = ing.headers.copy()
                    h["lado"] = "emitidas"
                    headers_all.append(h)

                if egr is not None and egr.headers is not None and not egr.headers.empty:
                    h = egr.headers.copy()
                    h["lado"] = "recibidas"
                    headers_all.append(h)

                headers_df = pd.concat(headers_all, ignore_index=True) if headers_all else pd.DataFrame()

                if headers_df.empty:
                    st.info("Sin headers para el periodo.")
                else:
                    render_filterable_grid(headers_df, key="sat_headers_grid")
            
            # =============================================================================
            # Clientes / Proveedores (desde CFDI XML) - con nombres
            # =============================================================================
            clientes_df = cfdi_data.get("clientes_df", pd.DataFrame())
            proveedores_df = cfdi_data.get("proveedores_df", pd.DataFrame())

            # (opcional) si estás en modo local y no existen, caes al builder XML
            if (clientes_df is None or clientes_df.empty) and cfdi_source == "local":
                clientes_df, proveedores_df = build_clientes_proveedores_tables(
                    rfc=rfc,
                    ing_headers=ing.headers if ing is not None else None,
                    egr_headers=egr.headers if egr is not None else None,
                )

            with st.container(border=True):
                st.markdown("Clientes (desde CFDI XML)")
                if clientes_df is None or clientes_df.empty:
                    st.info("Sin clientes para el periodo.")
                else:
                    render_filterable_grid(clientes_df, key="sat_clientes_xml_grid")

            with st.container(border=True):
                st.markdown("Proveedores (desde CFDI XML)")
                if proveedores_df is None or proveedores_df.empty:
                    st.info("Sin proveedores para el periodo.")
                else:
                    render_filterable_grid(proveedores_df, key="sat_proveedores_xml_grid")


    else:
        st.info("Ingresa un RFC arriba, selecciona fuente/rango CFDI y presiona Calcular.")

# =============================================================================
# TAB 1: FACTURAS
# =============================================================================
with tabs[1]:
    st.caption("CFDI (XML) → Headers / Conceptos / Prod-Serv (resumen)")
    
    cfdi_data = st.session_state.get("cfdi_data") or {}
    meta = cfdi_data.get("meta") or {}
    if meta:
        st.info(
            f"Meta descarga: emit listed={meta.get('emit_listed')} downloaded={meta.get('emit_downloaded')} failed={meta.get('emit_failed')} "
            f"(headers={meta.get('emit_headers')}) | rec listed={meta.get('rec_listed')} downloaded={meta.get('rec_downloaded')} failed={meta.get('rec_failed')} "
            f"(headers={meta.get('rec_headers')}) | workers={meta.get('max_workers')} "
        )


    if not cfdi_data:
        st.info("Selecciona RFC/fuente/rango arriba y presiona Calcular.")
    else:
        ing = cfdi_data.get("ingresos")
        egr = cfdi_data.get("egresos")

        headers_all = []
        conceptos_all = []

        if ing is not None and ing.headers is not None and not ing.headers.empty:
            h = ing.headers.copy()
            h["lado"] = "emitidas"
            headers_all.append(h)

        if egr is not None and egr.headers is not None and not egr.headers.empty:
            h = egr.headers.copy()
            h["lado"] = "recibidas"
            headers_all.append(h)

        if ing is not None and ing.conceptos is not None and not ing.conceptos.empty:
            c = ing.conceptos.copy()
            c["lado"] = "emitidas"
            conceptos_all.append(c)

        if egr is not None and egr.conceptos is not None and not egr.conceptos.empty:
            c = egr.conceptos.copy()
            c["lado"] = "recibidas"
            conceptos_all.append(c)

        headers_df = pd.concat(headers_all, ignore_index=True) if headers_all else pd.DataFrame()
        conceptos_df = pd.concat(conceptos_all, ignore_index=True) if conceptos_all else pd.DataFrame()

        subtabs = st.tabs(
            [
                "Headers",
                "Conceptos",
                "Prod/Serv (resumen)_ingresos",
                "Prod/Serv (resumen)_egresos",
            ]
        )

        with subtabs[0]:
            if headers_df.empty:
                st.info("Sin headers para el periodo.")
            else:
                st.dataframe(headers_df, use_container_width=True
, hide_index=True)

        with subtabs[1]:
            if conceptos_df.empty:
                st.info("Sin conceptos para el periodo.")
            else:
                st.dataframe(conceptos_df, use_container_width=True
, hide_index=True)

        catalogo = load_ps_catalog()
        service = get_cfdi_service()

        with subtabs[2]:
            resumen_ing = (
                service.prodserv_summary_shiny(
                    rfc=rfc,
                    headers=ing.headers,
                    conceptos=ing.conceptos,
                    catalogo=catalogo,
                    tipo="I",
                    rol="emisor",
                    top_n=25,
                )
                if ing is not None and ing.headers is not None and ing.conceptos is not None
                else pd.DataFrame()
            )
            if resumen_ing.empty:
                st.info("Sin datos para el periodo.")
            else:
                st.dataframe(resumen_ing, use_container_width=True
, hide_index=True)

        with subtabs[3]:
            resumen_egr = (
                service.prodserv_summary_shiny(
                    rfc=rfc,
                    headers=egr.headers,
                    conceptos=egr.conceptos,
                    catalogo=catalogo,
                    tipo="I",
                    rol="receptor",
                    top_n=25,
                )
                if egr is not None and egr.headers is not None and egr.conceptos is not None
                else pd.DataFrame()
            )
            if resumen_egr.empty:
                st.info("Sin datos para el periodo.")
            else:
                st.dataframe(resumen_egr, use_container_width=True
, hide_index=True)


with tabs[2]:
    st.caption("Buró de Crédito → Moffin (PF / PM)")


    if run:
        if not rfc_valid:
            st.warning("RFC inválido. Debe tener 12 (PM) o 13 (PF) caracteres.")
        else:
            with st.spinner("Consultando Buró de Crédito (Moffin)..."):
                try:
                    df_buro = obtener_buro_moffin_por_rfc(rfc)
                    
                    if df_buro.empty:
                        st.info("No se encontraron registros de buró para este RFC.")
                    else:
                        # 1. Extraemos los valores
                        fecha = df_buro["Fecha Consulta"].iloc[0] if "Fecha Consulta" in df_buro.columns else "N/A"
                        monto = df_buro["MontoTotalPagar"].iloc[0] if "MontoTotalPagar" in df_buro.columns else "0.00"


                        kpi_cols = st.columns(2)
                        with kpi_cols[0]:
                            st.metric("Fecha de consulta del buró", fecha)

                        with kpi_cols[1]:
                            st.metric("Monto total a pagar", monto)                                # -----------------------------

                        # -----------------------------
                        # 2) Separar créditos abiertos vs cerrados
                        # -----------------------------

                        # Columnas visibles (sin Fecha Consulta)
                        cols_visibles = [c for c in df_buro.columns if c not in ["Fecha Consulta", "MontoTotalPagar"]]

                        df_tabla = df_buro[cols_visibles].copy()

                        # Crear columna numérica temporal para filtrar
                        if "Monto a pagar" in df_tabla.columns:
                            df_tabla["_monto_num"] = (
                                df_tabla["Monto a pagar"]
                                .astype(str)
                                .str.replace("$", "", regex=False)
                                .str.replace(",", "", regex=False)
                                .astype(float)
                            )
                        else:
                            df_tabla["_monto_num"] = 0.0

                        # Créditos abiertos (> 0)
                        df_abiertos = df_tabla[df_tabla["_monto_num"] > 0].drop(columns="_monto_num")

                        # Créditos cerrados (= 0)
                        df_cerrados = df_tabla[df_tabla["_monto_num"] == 0].drop(columns="_monto_num")

                        # -----------------------------
                        # 3) Render UI
                        # -----------------------------

                        # Créditos abiertos
                        st.markdown("#### ✅ Créditos abiertos")
                        st.caption("Cuentas con saldo o monto pendiente de pago")

                        if df_abiertos.empty:
                            st.info("No se encontraron créditos abiertos.")
                        else:
                            st.dataframe(
                                df_abiertos,
                                use_container_width=True,
                                hide_index=True,
                            )

                        st.markdown("---")

                        # Créditos cerrados
                        st.markdown("#### ⛔ Créditos cerrados")
                        st.caption("Cuentas sin saldo pendiente")

                        if df_cerrados.empty:
                            st.info("No se encontraron créditos cerrados.")
                        else:
                            st.dataframe(
                                df_cerrados,
                                use_container_width=True,
                                hide_index=True,
                            )

                except Exception as e:
                    st.error(f"Error consultando Buró de Crédito: {e}")
    else:
        st.info("Ingresa un RFC y presiona Aceptar para consultar el buró.")

