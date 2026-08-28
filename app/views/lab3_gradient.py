"""ラボ 3 — 勾配降下法。

損失曲面の上を降りていく軌跡を見て、学習率と最適化手法の役割を掴む。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.components.cards import Kpi, kpi_row, score_color
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.viz import theme
from mllab.viz.surface import (
    OPTIMIZER_NOTES,
    OPTIMIZERS,
    SURFACES,
    descend,
    loss_curve_figure,
    surface_figure,
)

KEY = "lab3"

accent = page_header(
    number=3,
    title="勾配降下法ラボ",
    lede=(
        "モデルが「学習する」とは、パラメータを少しずつ動かして損失を下げること。"
        "その動き方が 勾配降下法 です。地形・学習率・手法を変えて、"
        "うまく谷底に着くとき／振動するとき／発散するときの違いを見てください。"
        "縦横の軸はモデルのパラメータ、色の濃さが損失の大きさです。"
    ),
)

# ---- サイドバー -------------------------------------------------------
sidebar_section("地形")
surface_key = st.sidebar.selectbox(
    "損失曲面", list(SURFACES), format_func=lambda k: SURFACES[k].label, key=f"{KEY}_surf"
)
surface = SURFACES[surface_key]
st.sidebar.caption(surface.note)

sidebar_section("最適化")
optimizer = st.sidebar.selectbox(
    "手法", list(OPTIMIZERS), format_func=lambda k: OPTIMIZERS[k], key=f"{KEY}_opt"
)
st.sidebar.caption(OPTIMIZER_NOTES[optimizer])

lr_exp = st.sidebar.slider(
    "学習率 (learning rate)",
    -4.0, 0.7, float(np.log10(surface.default_lr)), 0.05,
    key=f"{KEY}_lr_{surface_key}", format="10^%.2f",
    help="1 ステップで進む歩幅。大きすぎると飛び越えて発散します。",
)
lr = float(10.0**lr_exp)
st.sidebar.caption(f"　→ 学習率 = **{lr:.4g}**")

steps = st.sidebar.slider(
    "ステップ数", 1, 300, 60, 1, key=f"{KEY}_steps",
    help="このスライダを 1 から増やしていくと、軌跡が伸びていきます。",
)

sidebar_section("出発点")
use_custom_start = st.sidebar.toggle("出発点を自分で決める", value=False, key=f"{KEY}_custom")
if use_custom_start:
    sx = st.sidebar.slider(
        "出発点 w1", float(surface.x_range[0]), float(surface.x_range[1]),
        float(surface.start[0]), 0.05, key=f"{KEY}_sx_{surface_key}",
    )
    sy = st.sidebar.slider(
        "出発点 w2", float(surface.y_range[0]), float(surface.y_range[1]),
        float(surface.start[1]), 0.05, key=f"{KEY}_sy_{surface_key}",
    )
    start = (float(sx), float(sy))
else:
    start = surface.start

if optimizer in ("momentum", "adam"):
    momentum = st.sidebar.slider(
        "慣性 (momentum)", 0.0, 0.99, 0.90, 0.01, key=f"{KEY}_mom",
        help="前回の移動をどれだけ引き継ぐか。大きいほど転がり続けます。",
    )
else:
    momentum = 0.9

sidebar_section("比較")
compare_all = st.sidebar.toggle(
    "4 手法を同じ条件で並べる", value=False, key=f"{KEY}_cmp",
    help="同じ学習率・同じ出発点で SGD / Momentum / RMSProp / Adam を比較します。",
)


# ---- 計算 -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run(surface_key, optimizer, lr, steps, start, momentum):
    return descend(SURFACES[surface_key], optimizer, lr, steps, start, momentum)


path, losses, diverged = run(surface_key, optimizer, lr, steps, start, momentum)

final_loss = float(losses[-1]) if np.isfinite(losses[-1]) else float("nan")
dist = float(np.hypot(path[-1, 0] - surface.minimum[0], path[-1, 1] - surface.minimum[1]))
improvement = float(losses[0] - final_loss) if np.isfinite(final_loss) else float("nan")

kpi_row(
    [
        Kpi("最終の損失", "発散" if diverged else f"{final_loss:.4g}",
            sub="小さいほど良い",
            color=theme.BAD if diverged else score_color(
                abs(final_loss - surface.value(*surface.minimum)), good=0.01, bad=1.0)),
        Kpi("最小値までの距離", "—" if diverged else f"{dist:.3f}",
            sub="0 に近いほど正解に到達",
            color=theme.BAD if diverged else score_color(dist, good=0.1, bad=1.0)),
        Kpi("進んだステップ数", f"{len(path) - 1}", unit=f"／{steps}",
            sub="発散すると途中で打ち切ります"),
        Kpi("損失の減り幅", "—" if diverged else f"{improvement:.4g}",
            sub="出発点からどれだけ下がったか"),
        Kpi("学習率", f"{lr:.4g}", sub="1 ステップの歩幅", color=theme.PURPLE),
    ],
    accent,
)

if diverged:
    note(
        f"学習率 {lr:.4g} は この地形には大きすぎます — "
        f"{len(path) - 1} ステップで損失が発散しました。学習率を下げてください。",
        tone="bad",
    )
elif dist < 0.1:
    note("最小値にほぼ到達しました", tone="good")
elif len(path) - 1 == steps and dist > 1.0:
    note("まだ最小値に届いていません — ステップ数か学習率を増やしてみてください", tone="warn")

# ---- メイン図 ---------------------------------------------------------
if not compare_all:
    col_map, col_loss = st.columns([1.5, 1])

    with col_map:
        panel(
            f"{surface.label} を {OPTIMIZERS[optimizer]} で降りる",
            "緑が出発点、ピンクの × が到達点、白い星が真の最小値",
        )
        st.plotly_chart(
            surface_figure(surface, path, height=520), width="stretch"
        )

    with col_loss:
        panel("損失の推移", "ステップごとに損失がどう下がったか")
        st.plotly_chart(
            loss_curve_figure(losses, log_y=not diverged, height=245), width="stretch"
        )

        panel("ステップの歩幅", "実際に 1 歩で動いた距離")
        step_sizes = np.linalg.norm(np.diff(path, axis=0), axis=1)
        fig_s = go.Figure(
            go.Bar(
                x=np.arange(1, len(step_sizes) + 1), y=step_sizes,
                marker=dict(color=theme.PURPLE, line=dict(width=0)),
                hovertemplate="step %{x}<br>移動距離 %{y:.4g}<extra></extra>",
            )
        )
        fig_s.update_layout(
            height=200, showlegend=False,
            xaxis=dict(title="ステップ"), yaxis=dict(title="移動距離"),
            margin=dict(l=48, r=24, t=10, b=40),
        )
        st.plotly_chart(fig_s, width="stretch")
else:
    panel(
        "4 手法を同じ条件で比較",
        f"地形={surface.label} / 学習率={lr:.4g} / {steps} ステップ / 同じ出発点",
    )

    cols = st.columns(2)
    summary = []
    for i, opt in enumerate(OPTIMIZERS):
        p, l, d = run(surface_key, opt, lr, steps, start, momentum)
        summary.append((opt, l, d, p))
        fig = surface_figure(surface, p, height=330)
        fig.update_layout(showlegend=False, margin=dict(l=28, r=10, t=10, b=28))
        fig.update_xaxes(title=None, showticklabels=False)
        fig.update_yaxes(title=None, showticklabels=False)
        with cols[i % 2]:
            state = "発散" if d else f"損失 {l[-1]:.4g}"
            st.markdown(f"**{OPTIMIZERS[opt]}** — {state}")
            st.plotly_chart(fig, width="stretch", key=f"{KEY}_cmp_{opt}")

    panel("損失の推移を重ねる", "同じ条件でどれが速く下がるか")
    fig_all = go.Figure()
    for (opt, l, d, _), color in zip(summary, theme.ACCENTS):
        fig_all.add_trace(
            go.Scatter(
                x=np.arange(len(l)), y=np.where(np.isfinite(l), l, np.nan),
                mode="lines", name=OPTIMIZERS[opt] + ("（発散）" if d else ""),
                line=dict(color=color, width=2.5),
            )
        )
    positive = np.concatenate([l[np.isfinite(l) & (l > 0)] for _, l, _, _ in summary])
    fig_all.update_layout(
        height=340,
        xaxis=dict(title="ステップ"),
        yaxis=dict(title="損失", type="log" if len(positive) > 1 else "linear"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_all, width="stretch")

explain("gradient_descent")
