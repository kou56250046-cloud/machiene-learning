"""ダッシュボードのカード部品。

Streamlit の st.metric ではなく自前の HTML カードを使う理由は、
左端のアクセントバーと hover のグローで「どのラボにいるか」を色で示すため。
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

from mllab.viz import theme


@dataclass(frozen=True)
class Kpi:
    """KPI カード 1 枚。"""

    label: str
    value: str
    unit: str = ""
    sub: str = ""
    #: None ならページのアクセント色を使う
    color: str | None = None


def kpi_row(items: list[Kpi], accent: str = theme.CYAN) -> None:
    """KPI カードを横並びで描く。列数は幅に応じて自動で折り返す。"""
    cells = []
    for k in items:
        color = k.color or accent
        unit = f'<span class="unit">{html.escape(k.unit)}</span>' if k.unit else ""
        sub = f'<div class="mllab-kpi-sub">{html.escape(k.sub)}</div>' if k.sub else ""
        cells.append(
            f'<div class="mllab-kpi" style="--accent:{color}">'
            f'<div class="mllab-kpi-label">{html.escape(k.label)}</div>'
            f'<div class="mllab-kpi-value">{html.escape(k.value)}{unit}</div>'
            f"{sub}</div>"
        )
    st.html(f'<div class="mllab-kpis">{"".join(cells)}</div>')


def score_color(value: float, good: float, bad: float) -> str:
    """値の良し悪しを色に変換する。good に近ければ黄緑、bad に近ければピンク。

    good > bad（大きいほど良い）でも good < bad（小さいほど良い）でも動く。
    """
    if good >= bad:
        if value >= good:
            return theme.GOOD
        if value <= bad:
            return theme.BAD
    else:
        if value <= good:
            return theme.GOOD
        if value >= bad:
            return theme.BAD
    return theme.WARN


@dataclass(frozen=True)
class LabCard:
    """ホームに並べるラボ紹介カード。"""

    number: int
    title: str
    body: str


def lab_grid(cards: list[LabCard]) -> None:
    """ホームのラボ一覧グリッド。"""
    cells = []
    for c in cards:
        color = theme.LAB_COLORS.get(c.number, theme.CYAN)
        cells.append(
            f'<div class="mllab-card" style="--accent:{color}">'
            f'<div class="mllab-card-num">LAB {c.number:02d}</div>'
            f'<div class="mllab-card-title">{html.escape(c.title)}</div>'
            f'<div class="mllab-card-body">{c.body}</div>'
            f"</div>"
        )
    st.html(f'<div class="mllab-grid">{"".join(cells)}</div>')
