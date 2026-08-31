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
    "catalog.py",
    "lab8_tabular.py",
    "lab9_timeseries.py",
    "lab10_text.py",
    "lab11_imaging.py",
    "lab12_inventory.py",
    "experiments.py",
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


# ---- カタログはデータの有無で表示が変わるので両方見る ------------------

def _seed_store(tmp_path) -> None:
    """テスト用の保存先に、気象と株価に見立てたデータを 1 件ずつ置く。"""
    import numpy as np
    import pandas as pd

    from mllab.data import store

    store.RAW_DIR = tmp_path / "raw"
    store.PROCESSED_DIR = tmp_path / "processed"
    store.ensure_dirs()

    days = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    store.save(
        "weather_tokyo",
        pd.DataFrame(
            {
                "日付": days,
                "都市": "東京",
                "最高気温": rng.normal(20, 8, 120).round(1),
                "平均気温": rng.normal(15, 8, 120).round(1),
                "降水量": rng.gamma(1, 3, 120).round(1),
            }
        ),
        label="気象データ", source="テスト", domain="timeseries",
        params={"city": "tokyo"},
    )
    store.save(
        "market_n225",
        pd.DataFrame(
            {
                "日付": days,
                "銘柄": "日経平均株価",
                "終値": rng.normal(38000, 500, 120).round(1),
                "前日比(%)": rng.normal(0, 1, 120).round(3),
            }
        ),
        label="株価・指数", source="テスト", domain="timeseries",
        params={"symbol": "^N225"},
    )


def test_catalog_renders_with_saved_data(tmp_path, monkeypatch) -> None:
    """保存済みデータがあるときの一覧・プレビュー・SQL タブが落ちないこと。"""
    from mllab.data import store

    monkeypatch.setattr(store, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path / "processed")
    _seed_store(tmp_path)

    at = _run("catalog.py")
    _assert_clean(at, "catalog.py（データあり）")

    html = _html_of(at)
    assert "mllab-kpi" in html
    # 保存済みテーブル名が SQL タブの案内に出ている
    body = " ".join(str(m.value) for m in at.get("caption"))
    assert "weather_tokyo" in body and "market_n225" in body


def test_catalog_sql_examples_match_saved_tables(tmp_path, monkeypatch) -> None:
    """例文が、実在するテーブル名だけを参照していること。"""
    from app.components import sql_examples
    from mllab.data import store

    monkeypatch.setattr(store, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path / "processed")
    _seed_store(tmp_path)

    stored = store.list_datasets()
    examples = sql_examples.build(stored)
    assert examples, "例文が 1 つも作られていない"

    names = {d.name for d in stored}
    for title, sql in examples.items():
        # 例文はすべて実行できなければならない（テーブル名・列名の綴り間違い検出）
        try:
            store.query(sql)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"例文『{title}』が実行できない: {exc}") from exc
        assert any(n in sql for n in names), f"例文『{title}』が実在テーブルを参照していない"


# ---- ウィジェットの key 衝突を静的に検出する ---------------------------

def _literal_keys(path: Path) -> list[str]:
    """ページ内の `key=...` を、f 文字列を展開しつつ集める。

    Streamlit は同じ key が 2 つあると実行時に落ちる。実際にその分岐を
    通らないと気づけないので、ソースを読んで先に見つける。
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    # KEY = "lab8" のようなモジュール定数を拾って f 文字列の展開に使う
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    constants[target.id] = node.value.value

    def render(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for piece in node.values:
                if isinstance(piece, ast.Constant):
                    parts.append(str(piece.value))
                elif isinstance(piece, ast.FormattedValue):
                    inner = piece.value
                    if isinstance(inner, ast.Name) and inner.id in constants:
                        parts.append(constants[inner.id])
                    else:
                        # 実行時にしか決まらない部分。衝突しない印を置く
                        return None
            return "".join(parts)
        return None

    keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "key":
                rendered = render(keyword.value)
                if rendered is not None:
                    keys.append(rendered)
    return keys


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p)
def test_no_duplicate_widget_keys(page: str) -> None:
    keys = _literal_keys(VIEWS / page)
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    assert not duplicates, f"{page} に重複した key があります: {duplicates}"
