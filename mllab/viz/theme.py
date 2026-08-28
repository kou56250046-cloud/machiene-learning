"""配色トークンと Plotly ダークテンプレート。

アプリ内の色はすべてこのモジュール経由で参照する。
各ページや CSS に HEX を直接書かないこと。
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- ベース（黒基調） -------------------------------------------------
BG = "#0B0B10"          # ページ背景
SURFACE = "#15151F"     # カード背景
SURFACE_ALT = "#101018"  # サイドバー背景
BORDER = "#26263A"      # 罫線・グリッド
TEXT = "#E8E8F0"        # 本文
TEXT_MUTED = "#8A8AA3"  # 補助テキスト

# --- アクセント 5 色 ---------------------------------------------------
CYAN = "#4DD8FF"    # 水色
PINK = "#FF4D9D"    # ピンク
LIME = "#A8F04B"    # 黄緑
ORANGE = "#FF9A3C"  # オレンジ
PURPLE = "#A96BFF"  # 紫

#: 系列色の並び順。全ラボでこの順序を共有する。
ACCENTS: list[str] = [CYAN, PINK, LIME, ORANGE, PURPLE]

#: アクセント 5 色の淡色版。6 クラス以上を描き分けるときに続けて使う。
ACCENTS_LIGHT: list[str] = ["#A9E9FF", "#FFA3CB", "#D3F79E", "#FFC894", "#D0B0FF"]

#: クラス番号 → 色。10 クラスまでは色が重複しない。
CATEGORY: list[str] = ACCENTS + ACCENTS_LIGHT

# --- 状態色 -----------------------------------------------------------
GOOD = LIME      # 良好
WARN = ORANGE    # 注意
BAD = PINK       # 問題あり

#: 連続量（損失曲面・密度など）
SEQUENTIAL = [
    [0.00, "#1A1030"],
    [0.20, PURPLE],
    [0.45, PINK],
    [0.70, ORANGE],
    [1.00, LIME],
]

#: 2 クラスの決定境界（水色 ↔ ピンク、中央は背景寄り）
DIVERGING = [
    [0.00, CYAN],
    [0.35, "#1E4A5E"],
    [0.50, "#242433"],
    [0.65, "#5E1E42"],
    [1.00, PINK],
]

TEMPLATE = "mllab_dark"

#: ラボ番号 → テーマカラー（ページヘッダのアクセント）
LAB_COLORS: dict[int, str] = {
    0: CYAN,
    1: CYAN,
    2: ORANGE,
    3: PURPLE,
    4: LIME,
    5: PINK,
    6: ORANGE,
    7: PURPLE,
}


def class_color(index: int) -> str:
    """クラス番号に対応する色を返す（10 色で巡回）。

    先頭 5 色は `ACCENTS` と同じなので、2〜5 クラスのラボでは
    従来どおり彩度の高い 5 色だけが使われる。
    """
    return CATEGORY[int(index) % len(CATEGORY)]


def rgba(hex_color: str, alpha: float) -> str:
    """`#RRGGBB` を `rgba(r,g,b,a)` に変換する。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _build_template() -> go.layout.Template:
    axis = dict(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        linecolor=BORDER,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title=dict(font=dict(color=TEXT_MUTED, size=12)),
    )
    return go.layout.Template(
        layout=dict(
            colorway=ACCENTS,
            colorscale=dict(sequential=SEQUENTIAL, diverging=DIVERGING),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, family="Inter, 'Segoe UI', sans-serif", size=13),
            title=dict(font=dict(color=TEXT, size=16)),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor=BORDER,
                borderwidth=1,
                font=dict(color=TEXT_MUTED, size=11),
            ),
            hoverlabel=dict(
                bgcolor=SURFACE,
                bordercolor=BORDER,
                font=dict(color=TEXT, size=12),
            ),
            margin=dict(l=48, r=24, t=48, b=44),
        )
    )


def apply() -> None:
    """Plotly のデフォルトテンプレートを ML Lab のダークテーマにする。"""
    pio.templates[TEMPLATE] = _build_template()
    pio.templates.default = TEMPLATE
