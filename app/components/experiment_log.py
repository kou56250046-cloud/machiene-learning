"""各ラボから実験を記録するための共通 UI。

ラボ側は「いまの設定」と「いまの結果」を辞書で渡すだけでよい。
どのラボからでも同じ形で記録されるので、後から横断して比較できる。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.components.layout import panel
from mllab import experiments as EX
from mllab.viz import theme


def record_panel(
    lab: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    key: str,
    default_experiment: str = "",
) -> None:
    """「この結果を記録する」欄を描く。

    Args:
        lab: どのラボからの記録か。
        params: 何を設定したか。
        metrics: どうだったか。
        key: ウィジェットの key の接頭辞。
        default_experiment: 実験名の初期値。
    """
    panel("この結果を記録する", "設定と結果を残しておくと、後から比べられます")

    usable = {k: v for k, v in metrics.items() if v is not None and np.isfinite(v)}
    if not usable:
        st.caption("記録できる数値の結果がありません。")
        return

    existing = EX.experiments()
    col_name, col_note, col_button = st.columns([1.2, 1.6, 0.8])

    with col_name:
        name = st.text_input(
            "実験名", value=default_experiment or lab, key=f"{key}_expname",
            help="同じ名前で記録すると、1 つの実験としてまとまります。",
        )
    with col_note:
        note = st.text_input(
            "メモ（任意）", value="", key=f"{key}_expnote",
            placeholder="例: 欠測を中央値で埋めた版",
        )
    with col_button:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        clicked = st.button("記録する", type="primary", key=f"{key}_explog", width="stretch")

    with st.expander("記録される内容", expanded=False):
        left, right = st.columns(2)
        left.markdown("**設定**")
        left.json(params or {})
        right.markdown("**結果**")
        right.json({k: round(float(v), 6) for k, v in usable.items()})

    if clicked:
        try:
            run_id = EX.log(name, lab, params, usable, note)
        except EX.ExperimentError as exc:
            st.error(str(exc))
        else:
            st.success(
                f"「{name}」に記録しました（{run_id}）。"
                "**実験ログ**のページで、これまでの試行と比べられます。"
            )

    if existing:
        st.caption("記録済みの実験: " + "　".join(f"`{e}`" for e in existing))


# ======================================================================
# 実験ログのページで使う図
# ======================================================================


def history_figure(frame, metric: str, height: int = 340) -> go.Figure:
    """試行を記録順に並べ、指標がどう動いたかを見る。"""
    import pandas as pd

    ordered = frame.sort_values("記録日時").reset_index(drop=True)
    values = pd.to_numeric(ordered[metric], errors="coerce")
    label = str(metric).removeprefix("結果:")

    best_so_far = values.cummax()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(values) + 1), y=values, mode="lines+markers",
            name=label, line=dict(color=theme.CYAN, width=2),
            marker=dict(size=9),
            customdata=ordered["メモ"].fillna("").to_numpy(),
            hovertemplate="%{x} 回目<br>" + label + " %{y:.4f}<br>%{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(values) + 1), y=best_so_far, mode="lines",
            name="そこまでの最良",
            line=dict(color=theme.LIME, width=2, dash="dot"),
            hovertemplate="%{x} 回目までの最良 %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        xaxis=dict(title="何回目の試行か", dtick=1),
        yaxis=dict(title=label),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def setting_effect_figure(frame, setting: str, metric: str, height: int = 340) -> go.Figure:
    """設定の値ごとに、スコアの分布を箱ひげで比べる。"""
    import pandas as pd

    values = pd.to_numeric(frame[metric], errors="coerce")
    label = str(metric).removeprefix("結果:")
    fig = go.Figure()
    for i, value in enumerate(sorted(frame[setting].dropna().unique().tolist())):
        mask = frame[setting] == value
        scores = values[mask].dropna()
        if scores.empty:
            continue
        color = theme.class_color(i)
        fig.add_trace(
            go.Box(
                y=scores, name=str(value), boxpoints="all", jitter=0.5, pointpos=0,
                marker=dict(color=color, size=8, line=dict(color=theme.BG, width=1)),
                line=dict(color=color), fillcolor=theme.rgba(color, 0.15),
                hovertemplate="%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height, showlegend=False,
        xaxis=dict(title=str(setting).removeprefix("設定:")),
        yaxis=dict(title=label),
    )
    return fig


def importance_figure(effects, height: int | None = None) -> go.Figure:
    """どの設定を変えるとスコアが大きく動いたか。"""
    shown = effects.head(12).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=shown["スコアの幅"], y=shown["設定"], orientation="h",
            marker=dict(color=shown["スコアの幅"], colorscale=theme.SEQUENTIAL,
                        showscale=False, line=dict(width=0)),
            customdata=shown["最良の値"],
            hovertemplate="%{y}<br>幅 %{x:.4f}<br>最良の値: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height or max(240, 30 * len(shown) + 90),
        showlegend=False,
        xaxis=dict(title="スコアの幅（値を変えたときの動き）"),
        yaxis=dict(title=None),
        margin=dict(l=140, r=24, t=16, b=44),
    )
    return fig
