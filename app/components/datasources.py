"""データ取得元の入力欄と、取得結果のプレビュー。

`Connector.options` の定義からウィジェットを機械的に描くので、
コネクタを足してもこのファイルを触る必要はない。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mllab.data.connectors import Connector, Option
from mllab.viz import theme


def connector_controls(connector: Connector, key: str, container=None) -> dict[str, Any]:
    """コネクタの入力欄を描き、取得条件の辞書を返す。"""
    ui = container if container is not None else st
    values: dict[str, Any] = {}
    for option in connector.options:
        values[option.key] = _one_control(ui, option, f"{key}_{connector.key}_{option.key}")
    return values


def _one_control(ui, option: Option, widget_key: str) -> Any:
    label_of = lambda v: option.labels.get(v, str(v))  # noqa: E731
    help_text = option.help or None

    if option.kind == "text":
        return ui.text_input(
            option.label, value=option.default or "", key=widget_key, help=help_text
        )

    if option.kind == "int":
        return int(
            ui.number_input(
                option.label,
                min_value=int(option.min if option.min is not None else 0),
                max_value=int(option.max if option.max is not None else 1_000_000),
                value=int(option.default),
                step=100,
                key=widget_key,
                help=help_text,
            )
        )

    if option.kind == "date":
        picked = ui.date_input(
            option.label,
            value=pd.to_datetime(option.default).date() if option.default else None,
            min_value=date(1940, 1, 1),
            max_value=date.today(),
            key=widget_key,
            help=help_text,
            format="YYYY-MM-DD",
        )
        return picked.isoformat() if isinstance(picked, date) else None

    if option.kind == "select":
        options = list(option.options)
        return ui.selectbox(
            option.label,
            options,
            index=options.index(option.default) if option.default in options else 0,
            format_func=label_of,
            key=widget_key,
            help=help_text,
        )

    if option.kind == "multiselect":
        return tuple(
            ui.multiselect(
                option.label,
                list(option.options),
                default=list(option.default or ()),
                format_func=label_of,
                key=widget_key,
                help=help_text,
            )
        )

    raise ValueError(f"未知の入力欄の種別: {option.kind}")


def preview(frame: pd.DataFrame, rows: int = 12) -> None:
    """先頭数行と、列ごとの型・欠測をまとめて見せる。"""
    st.dataframe(frame.head(rows), width="stretch", height=min(420, 44 + 35 * rows))

    profile = pd.DataFrame(
        {
            "列": [str(c) for c in frame.columns],
            "型": [str(t) for t in frame.dtypes],
            "欠測": [int(frame[c].isna().sum()) for c in frame.columns],
            "欠測率": [f"{frame[c].isna().mean():.1%}" for c in frame.columns],
            "ユニーク数": [int(frame[c].nunique(dropna=True)) for c in frame.columns],
        }
    )
    with st.expander(f"列の内訳（{len(frame.columns)} 列 × {len(frame):,} 行）", expanded=False):
        st.dataframe(profile, width="stretch", hide_index=True)


def quick_chart(frame: pd.DataFrame, key: str) -> None:
    """列を選ぶだけの簡易グラフ。取り込んだデータの当たりを付けるためのもの。"""
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        st.caption("数値列がないため、グラフは描けません。")
        return

    time_like = [
        c
        for c in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[c])
    ]
    x_options = time_like + [c for c in frame.columns if c not in time_like]

    col_x, col_y, col_kind = st.columns([1, 1.4, 1])
    with col_x:
        x = st.selectbox("横軸", x_options, key=f"{key}_x")
    with col_y:
        ys = st.multiselect(
            "縦軸（数値列・複数可）", numeric,
            default=numeric[: min(2, len(numeric))], key=f"{key}_y",
        )
    with col_kind:
        default_kind = "折れ線" if time_like else "散布"
        kinds = ["折れ線", "散布", "ヒストグラム"]
        kind = st.selectbox(
            "グラフ", kinds, index=kinds.index(default_kind), key=f"{key}_kind"
        )

    if not ys:
        st.caption("縦軸に列を 1 つ以上選んでください。")
        return

    # 点が多すぎると描画が重いので間引く（傾向を見るのが目的なので十分）
    limit = 5000
    plotted = frame
    if len(frame) > limit and kind != "ヒストグラム":
        plotted = frame.iloc[:: max(1, len(frame) // limit)]

    fig = go.Figure()
    for i, y in enumerate(ys):
        color = theme.class_color(i)
        if kind == "折れ線":
            fig.add_trace(
                go.Scatter(x=plotted[x], y=plotted[y], mode="lines", name=str(y),
                           line=dict(color=color, width=1.8))
            )
        elif kind == "散布":
            fig.add_trace(
                go.Scatter(x=plotted[x], y=plotted[y], mode="markers", name=str(y),
                           marker=dict(color=color, size=6, opacity=0.7,
                                       line=dict(color=theme.BG, width=0.6)))
            )
        else:
            fig.add_trace(
                go.Histogram(x=frame[y], name=str(y), nbinsx=40,
                             marker=dict(color=color, line=dict(width=0)), opacity=0.75)
            )

    fig.update_layout(
        height=380,
        barmode="overlay",
        xaxis=dict(title=None if kind == "ヒストグラム" else str(x)),
        yaxis=dict(title="件数" if kind == "ヒストグラム" else None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch", key=f"{key}_chart")

    if len(plotted) < len(frame):
        st.caption(f"描画は {len(plotted):,} 点に間引いています（全 {len(frame):,} 行）。")
