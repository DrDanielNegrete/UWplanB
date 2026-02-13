# Ruta: src/underwriting/ui/cfdi_views.py
# Archivo: cfdi_views.py

from __future__ import annotations

import pandas as pd
import streamlit as st


def _make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out.reset_index(drop=True)

    # columnas a str + únicas
    out.columns = [str(c) for c in out.columns]
    if len(set(out.columns)) != len(out.columns):
        out.columns = _make_unique(list(out.columns))

    # celdas no serializables -> str
    for c in out.columns:
        s = out[c]
        if s.map(lambda x: isinstance(x, (list, dict, tuple, set))).any():
            out[c] = s.map(lambda x: str(x) if isinstance(x, (list, dict, tuple, set)) else x)

    return out


def _render_table_card(title: str, df: pd.DataFrame, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"### 🧾 {title}")
        if df is None or df.empty:
            st.info("Sin datos para el periodo.")
            return

        safe = _safe_df(df)
        st.dataframe(safe, use_container_width=True, hide_index=True, key=key)


def render_prodserv_dual_cards(
    *,
    title_left: str,
    df_left: pd.DataFrame,
    title_right: str,
    df_right: pd.DataFrame,
) -> None:
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        _render_table_card(title_left, df_left, key="df_prodserv_resumen_ingresos_card")
    with col2:
        _render_table_card(title_right, df_right, key="df_prodserv_resumen_egresos_card")
