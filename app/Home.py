"""ML Lab のエントリポイント。

    uv run streamlit run app/Home.py

ページの登録だけを行い、中身は app/views/ 以下に置く。
"""

from __future__ import annotations

import sys
from pathlib import Path

# app/views/*.py からも `app.components` / `mllab` を import できるようにする
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="ML Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    st.Page("views/home.py", title="ホーム", icon="🏠", url_path="home", default=True),
    st.Page("views/lab1_boundary.py", title="1 決定境界", icon="🎯", url_path="boundary"),
    st.Page("views/lab2_overfitting.py", title="2 過学習と正則化", icon="📈", url_path="overfitting"),
    st.Page("views/lab3_gradient.py", title="3 勾配降下法", icon="⛰️", url_path="gradient"),
    st.Page("views/lab4_clustering.py", title="4 クラスタリング", icon="🧩", url_path="clustering"),
    st.Page("views/lab5_dimreduction.py", title="5 次元削減", icon="🗺️", url_path="dimreduction"),
    st.Page("views/lab6_metrics.py", title="6 評価指標", icon="⚖️", url_path="metrics"),
    st.Page("views/lab7_ensemble.py", title="7 アンサンブル", icon="🌲", url_path="ensemble"),
    st.Page("views/catalog.py", title="データカタログ", icon="🗄️", url_path="catalog"),
    st.Page("views/lab8_tabular.py", title="8 テーブルデータ", icon="📋", url_path="tabular"),
    st.Page("views/lab9_timeseries.py", title="9 時系列", icon="📉", url_path="timeseries"),
    st.Page("views/lab10_text.py", title="10 テキスト", icon="📰", url_path="text"),
    st.Page("views/lab11_imaging.py", title="11 画像・信号", icon="🖼️", url_path="imaging"),
    st.Page("views/experiments.py", title="実験ログ", icon="🧾", url_path="experiments"),
]

st.navigation(
    {
        "ML LAB": PAGES[:1],
        "基礎ラボ": PAGES[1:8],
        "データ基盤": PAGES[8:9],
        "応用ラボ": PAGES[9:13],
        "記録": PAGES[13:],
    }
).run()
