"""ページ共通の下地。

各ラボの先頭で `page_header(...)` を 1 回呼ぶ。CSS 注入と Plotly テーマ適用も
そこでまとめて行うので、ページ側で個別に設定する必要はない。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from mllab.viz import theme

_CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"


@st.cache_data(show_spinner=False)
def _load_css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")


def bootstrap(accent: str = theme.CYAN) -> None:
    """CSS の注入と Plotly テーマの適用。ページごとのアクセント色も渡す。"""
    theme.apply()
    st.html(f"<style>{_load_css()}\n:root {{ --accent: {accent}; }}</style>")


def page_header(
    number: int,
    title: str,
    lede: str,
    eyebrow: str = "",
) -> str:
    """ラボのヘッダを描き、そのラボのアクセント色を返す。

    Args:
        number: ラボ番号（0 はホーム）。`theme.LAB_COLORS` の添字になる。
        title: ラボ名。
        lede: 何が体感できるかの短い説明。
        eyebrow: 見出しの上に出す小さな英字ラベル。

    Returns:
        このページのアクセント色（HEX）。KPI カードや図の色に使う。
    """
    accent = theme.LAB_COLORS.get(number, theme.CYAN)
    bootstrap(accent)

    label = eyebrow or (f"LAB {number:02d}" if number else "ML LAB")
    st.html(
        f"""
        <div class="mllab-header">
          <div class="mllab-eyebrow">{label}</div>
          <h1 class="mllab-title">{title}</h1>
          <p class="mllab-lede">{lede}</p>
        </div>
        """
    )
    return accent


def sidebar_section(label: str) -> None:
    """サイドバーのセクション見出し（アクセントドット付き）。"""
    st.sidebar.html(f'<div class="mllab-side-head">{label}</div>')


def panel(title: str, note: str = "") -> None:
    """本文中のセクション見出し。"""
    note_html = f'<span class="mllab-panel-note">{note}</span>' if note else ""
    st.html(
        f"""
        <div class="mllab-panel-head">
          <span class="mllab-panel-title">{title}</span>{note_html}
        </div>
        """
    )


def note(text: str, tone: str = "warn") -> None:
    """1 行の注意書きバッジ。tone は good / warn / bad。"""
    cls = {"good": "good", "warn": "", "bad": "bad"}.get(tone, "")
    st.html(f'<div class="mllab-note {cls}">{text}</div>')
