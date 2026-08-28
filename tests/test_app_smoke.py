"""各ラボページが例外なく描画できることを確かめるスモークテスト。

Streamlit の AppTest でページを実際に実行する。
中身の正しさは test_mllab.py が見ているので、ここでは
「開いて、ウィジェットを動かしても落ちない」ことだけを保証する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIEWS = ROOT / "app" / "views"

PAGES = [
    "home.py",
    "lab1_boundary.py",
    "lab2_overfitting.py",
    "lab3_gradient.py",
    "lab4_clustering.py",
    "lab5_dimreduction.py",
    "lab6_metrics.py",
    "lab7_ensemble.py",
]

# t-SNE や全モデルのスイープがあるので、ページによっては数十秒かかる
TIMEOUT = 180


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(VIEWS / page), default_timeout=TIMEOUT)
    at.run()
    return at


def _assert_clean(at: AppTest, page: str) -> None:
    assert not at.exception, (
        f"{page} で例外: " + "; ".join(str(e.message) for e in at.exception)
    )
    assert not at.error, f"{page} でエラー表示: " + "; ".join(str(e.value) for e in at.error)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page: str) -> None:
    _assert_clean(_run(page), page)


@pytest.mark.parametrize("page", PAGES[1:])
def test_page_survives_widget_changes(page: str) -> None:
    """スライダ・セレクトボックス・トグルを動かしても落ちないこと。"""
    at = _run(page)
    _assert_clean(at, page)

    # 各種ウィジェットを既定値から 1 つずらして再実行する
    for slider in list(at.sidebar.slider)[:4]:
        try:
            slider.set_value(slider.max if slider.value != slider.max else slider.min)
        except Exception:  # noqa: BLE001 - 型が合わないウィジェットは飛ばす
            continue
    # options は format_func 適用後のラベルなので、値ではなく添字で選び直す
    for select in list(at.sidebar.selectbox)[:2]:
        if len(select.options) > 1:
            select.select_index((select.index + 1) % len(select.options))
    for radio in list(at.sidebar.radio)[:2]:
        # format_func 付きの radio は options がラベルなので設定できない。その場合は飛ばす
        if len(radio.options) > 1:
            try:
                radio.set_value(radio.options[(radio.index + 1) % len(radio.options)])
            except Exception:  # noqa: BLE001
                continue
    for toggle in at.sidebar.toggle:
        toggle.set_value(not toggle.value)

    at.run()
    _assert_clean(at, f"{page}（ウィジェット操作後）")


def _html_of(at: AppTest) -> str:
    """ページが st.html で出した HTML をすべて連結して返す。"""
    return "".join(el.proto.body for el in at.get("html"))


def test_every_lab_shows_an_explanation() -> None:
    """各ラボが content/*.md の解説を表示していること。"""
    for page in PAGES[1:]:
        html = _html_of(_run(page))
        assert "mllab-explain" in html, f"{page} に解説ブロックがない"
        assert "解説ファイルが見つかりません" not in html, (
            f"{page} が参照する content/*.md が存在しない"
        )


def test_every_lab_shows_kpi_cards() -> None:
    """各ラボの上部に KPI カードが出ていること（ホームを除く）。"""
    for page in PAGES[1:]:
        html = _html_of(_run(page))
        assert "mllab-kpi" in html, f"{page} に KPI カードがない"
