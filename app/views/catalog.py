"""データカタログ — 取得・蓄積・SQL 検索。

外部から取ってきたデータを Parquet に落とし、DuckDB で SQL を打てるようにする。
以降のドメイン別ラボは、すべてここに溜めたデータを入力にする。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import sql_examples
from app.components.cards import Kpi, kpi_row
from app.components.datasources import connector_controls, preview, quick_chart
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.data import store
from mllab.data.connectors import CONNECTORS, DOMAIN_LABELS, estat
from mllab.viz import theme

KEY = "catalog"

accent = page_header(
    number=10,
    title="データカタログ",
    lede=(
        "外部の公開データを取り込んで Parquet に溜め、SQL で調べられるようにする場所です。"
        "サーバーは一切立てず、すべてローカルのファイルで完結します。"
        "ここに溜めたデータが、これから作るドメイン別ラボの入力になります。"
    ),
    eyebrow="DATA PLATFORM",
)

datasets = store.list_datasets()

kpi_row(
    [
        Kpi("保存済みデータセット", f"{len(datasets)}", sub="data/raw/*.parquet"),
        Kpi("合計行数", f"{sum(d.rows for d in datasets):,}", sub="全データセットの合計"),
        Kpi("合計サイズ", f"{sum(d.size_mb for d in datasets):.1f}", unit="MB",
            sub="Parquet は圧縮が効くので軽量です"),
        Kpi("最終取得", datasets[0].fetched_at_local if datasets else "—",
            sub=datasets[0].label if datasets else "まだ何も取得していません",
            color=theme.PURPLE),
    ],
    accent,
)

if not datasets:
    note(
        "まだデータがありません。まず「データを取得する」タブで "
        "公開データセット（通信不要）を 1 つ入れてみてください。",
        tone="warn",
    )

sidebar_section("保存場所")
st.sidebar.caption(
    "`data/raw/` に Parquet と `.meta.json` を並べて保存します。"
    "取得条件を必ず併置するので、あとから「これは何のデータか」が分かります。"
)
st.sidebar.caption(
    "取得したデータは git 管理外です（`.gitignore` 済み）。"
    "取り直せるものはリポジトリに入れない方針です。"
)

tab_fetch, tab_stored, tab_sql = st.tabs(
    ["データを取得する", "保存済みデータ", "SQL で調べる"]
)

# ======================================================================
# タブ 1: 取得
# ======================================================================
with tab_fetch:
    panel("取得元を選ぶ", "e-Stat 以外はすべて APIキー不要です")

    connector_key = st.selectbox(
        "取得元",
        list(CONNECTORS),
        format_func=lambda k: (
            f"{CONNECTORS[k].label}"
            f"（{DOMAIN_LABELS.get(CONNECTORS[k].domain, CONNECTORS[k].domain)}）"
            + ("　※要APIキー" if CONNECTORS[k].requires_key else "")
        ),
        key=f"{KEY}_source",
    )
    connector = CONNECTORS[connector_key]

    st.markdown(connector.description)
    st.caption(f"出典: {connector.source}　—　{connector.terms}")

    # e-Stat だけは鍵の状態と探し方を先に出す
    if connector.requires_key and not estat.has_key():
        st.warning("APIキーが未設定です", icon="🔑")
        st.code(connector.requires_key, language="text")
    elif connector_key == "estat":
        note("APIキーを読み込めました", tone="good")

    params = connector_controls(connector, KEY)

    if connector_key == "estat" and estat.has_key():
        with st.expander("統計表 ID をキーワードで探す", expanded=False):
            col_kw, col_btn = st.columns([3, 1])
            keyword = col_kw.text_input(
                "キーワード", value="人口推計", key=f"{KEY}_estat_kw",
                label_visibility="collapsed",
                placeholder="例: 家計調査 / 人口推計 / 消費者物価",
            )
            if col_btn.button("検索", width="stretch", key=f"{KEY}_estat_search"):
                with st.spinner("e-Stat を検索中…"):
                    st.session_state[f"{KEY}_estat_hits"] = estat.search(keyword)
            hits = st.session_state.get(f"{KEY}_estat_hits")
            if hits is not None:
                if hits.ok:
                    st.dataframe(hits.frame, width="stretch", hide_index=True, height=320)
                    st.caption("使いたい行の「統計表ID」を上の入力欄に貼り付けてください。")
                else:
                    st.error(hits.error)

    dataset_name = connector.name_for(params)
    st.caption(f"保存名（SQL のテーブル名）: `{dataset_name}`")

    if dataset_name in {d.name for d in datasets} and connector_key != "news":
        note(f"`{dataset_name}` は既にあります。取得すると上書きされます。", tone="warn")

    if st.button("取得する", type="primary", key=f"{KEY}_fetch"):
        with st.spinner(f"{connector.label} を取得中…"):
            result = connector.fetch(**params)
        st.session_state[f"{KEY}_result"] = (connector_key, dataset_name, result)

    stashed = st.session_state.get(f"{KEY}_result")
    if stashed:
        used_key, used_name, result = stashed
        if used_key != connector_key:
            # 取得元を切り替えたら前の結果は関係ないので出さない
            st.session_state.pop(f"{KEY}_result", None)
        elif not result.ok:
            st.error(result.error)
        else:
            frame = result.frame
            note(f"{len(frame):,} 行 × {len(frame.columns)} 列を取得しました", tone="good")

            for warning in result.params.get("partial_failures", []):
                st.warning(warning, icon="⚠️")

            panel("取得結果のプレビュー", "保存する前に中身を確かめてください")
            preview(frame)

            col_save, col_note = st.columns([1, 3])
            if col_save.button("この内容で保存", type="primary", key=f"{KEY}_save"):
                try:
                    saved = store.save(
                        used_name,
                        frame,
                        label=connector.label,
                        source=connector.source,
                        domain=connector.domain,
                        description=connector.description,
                        params=result.params,
                    )
                except store.StoreError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop(f"{KEY}_result", None)
                    st.success(
                        f"`{saved.name}` として保存しました"
                        f"（{saved.rows:,} 行 / {saved.size_mb:.2f} MB）。"
                    )
                    st.rerun()
            col_note.caption(
                f"保存先: `data/raw/{used_name}.parquet`　"
                f"— 取得条件は `{used_name}.meta.json` に一緒に残します。"
            )

# ======================================================================
# タブ 2: 保存済み
# ======================================================================
with tab_stored:
    if not datasets:
        st.info("まだ保存済みのデータがありません。左のタブから取得してください。")
    else:
        panel("保存済みの一覧", "新しく取得したものが上に来ます")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "テーブル名": d.name,
                        "種類": DOMAIN_LABELS.get(d.domain, d.domain),
                        "取得元": d.label,
                        "行数": d.rows,
                        "列数": len(d.columns),
                        "サイズ(MB)": round(d.size_mb, 2),
                        "取得日時": d.fetched_at_local,
                    }
                    for d in datasets
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        panel("中身を見る", "1 つ選んでプレビューとグラフを確認します")

        chosen = st.selectbox(
            "データセット",
            [d.name for d in datasets],
            format_func=lambda n: f"{n} — {next(d.label for d in datasets if d.name == n)}",
            key=f"{KEY}_pick",
        )
        meta = next(d for d in datasets if d.name == chosen)
        frame = store.load(chosen)

        kpi_row(
            [
                Kpi("行数", f"{meta.rows:,}", sub=meta.label),
                Kpi("列数", f"{len(meta.columns)}",
                    sub="／".join(meta.columns[:3]) + ("…" if len(meta.columns) > 3 else "")),
                Kpi("サイズ", f"{meta.size_mb:.2f}", unit="MB", sub="Parquet 圧縮後"),
                Kpi("取得日時", meta.fetched_at_local, sub=meta.source, color=theme.PURPLE),
            ],
            accent,
        )

        with st.expander("取得条件（何をどう取ったか）", expanded=False):
            st.json(meta.params or {})

        preview(frame)

        panel("かんたんグラフ", "列を選ぶだけ。データの当たりを付けるためのものです")
        quick_chart(frame, key=f"{KEY}_{chosen}")

        panel("削除", "外部から取り直せるデータなので、消しても失われません")
        col_del, col_warn = st.columns([1, 3])
        confirm = col_warn.checkbox(
            f"`{chosen}` を削除することを確認", key=f"{KEY}_confirm_{chosen}"
        )
        if col_del.button("削除する", disabled=not confirm, key=f"{KEY}_delete"):
            store.delete(chosen)
            st.success(f"`{chosen}` を削除しました。")
            st.rerun()

# ======================================================================
# タブ 3: SQL
# ======================================================================
with tab_sql:
    if not datasets:
        st.info("SQL を打つには、まずデータを 1 つ以上取得してください。")
    else:
        panel(
            "SQL で調べる",
            "DuckDB が Parquet を直接読みます。DB へ取り込む手順はありません",
        )

        st.caption(
            "使えるテーブル: "
            + "　".join(f"`{d.name}`（{d.rows:,} 行）" for d in datasets)
        )

        examples = sql_examples.build(datasets)
        picked = st.selectbox(
            "例文から始める",
            list(examples),
            key=f"{KEY}_example",
            help="選ぶと下の入力欄に入ります。書き換えて実行してください。",
        )

        sql = st.text_area(
            "SQL",
            value=examples[picked],
            height=170,
            key=f"{KEY}_sql_{picked}",
            label_visibility="collapsed",
        )

        col_run, col_hint = st.columns([1, 4])
        run = col_run.button("実行", type="primary", key=f"{KEY}_run")
        col_hint.caption(
            "読み取り専用です。`SELECT` / `WITH` で始まる 1 文だけ実行できます"
            "（元データを壊さないため）。列名が日本語のときは `\"平均気温\"` のように"
            "二重引用符で囲みます。"
        )

        if run:
            try:
                result_frame = store.query(sql)
            except store.StoreError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 - SQL の文法エラーなど
                st.error(f"SQL の実行に失敗しました: {exc}")
            else:
                note(f"{len(result_frame):,} 行が返りました", tone="good")
                st.dataframe(result_frame, width="stretch", height=380)
                if not result_frame.empty:
                    panel("結果をグラフにする", "")
                    quick_chart(result_frame, key=f"{KEY}_sqlchart")

explain("data_catalog")
