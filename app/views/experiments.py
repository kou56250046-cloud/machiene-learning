"""実験ログ — 「何を試して、どうだったか」を残す場所。

各ラボの「この結果を記録する」から溜まった試行を、横断して比べる。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import experiment_log as EL
from app.components.cards import Kpi, kpi_row
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab import experiments as EX
from mllab.viz import theme

KEY = "explog"

accent = page_header(
    number=12,
    title="実験ログ",
    lede=(
        "機械学習は「試して、比べて、また試す」の繰り返しです。"
        "記録していないと、半年後どころか翌週には「あのとき一番良かった設定」が分からなくなります。"
        "各ラボの下部にある「この結果を記録する」から溜めた試行を、ここで横断して比べられます。"
    ),
    eyebrow="EXPERIMENT TRACKING",
)

all_runs = EX.load_runs()
info = EX.storage_info()

sidebar_section("保存場所")
st.sidebar.caption(
    "`data/experiments/runs.parquet` に 1 ファイルで溜めています。"
    "サーバーは立てません。"
)
st.sidebar.caption(
    f"サイズ: {info['size_kb']:.1f} KB　—　"
    "取得データと同じく git 管理外です。"
)

if all_runs.empty:
    kpi_row(
        [
            Kpi("記録された試行", "0", sub="まだ何も記録されていません"),
            Kpi("実験の数", "0", sub="実験名でまとまります"),
        ],
        accent,
    )
    note(
        "まだ記録がありません。**テーブルデータ・時系列・テキスト・画像**の各ラボの下部に "
        "「この結果を記録する」欄があります。設定を変えて何度か記録すると、ここで比較できます。",
        tone="warn",
    )
    explain("experiments")
    st.stop()

experiment_names = EX.experiments()

kpi_row(
    [
        Kpi("記録された試行", f"{len(all_runs):,}", sub="全実験の合計"),
        Kpi("実験の数", f"{len(experiment_names)}", sub="実験名でまとまります"),
        Kpi("使ったラボ", f"{all_runs['ラボ'].nunique()}",
            sub="／".join(sorted(all_runs["ラボ"].dropna().unique())[:3]),
            color=theme.PURPLE),
        Kpi("記録の大きさ", f"{info['size_kb']:.1f}", unit="KB",
            sub="Parquet 1 ファイル", color=theme.ORANGE),
    ],
    accent,
)

sidebar_section("表示")
experiment = st.sidebar.selectbox(
    "実験", experiment_names, key=f"{KEY}_experiment",
    help="同じ実験名で記録した試行が 1 つにまとまります。",
)

runs = EX.runs_of(experiment)
metrics = EX.metric_columns(runs)
settings = EX.param_columns(runs)

if not metrics:
    st.error(f"「{experiment}」には数値の結果が記録されていません。")
    st.stop()

metric = st.sidebar.selectbox(
    "比べる指標", metrics,
    format_func=lambda m: str(m).removeprefix("結果:"), key=f"{KEY}_metric",
)
higher_is_better = st.sidebar.toggle(
    "大きいほど良い指標", value=True, key=f"{KEY}_higher",
    help="RMSE や誤差のように「小さいほど良い」指標のときはオフにしてください。",
)

tab_compare, tab_settings, tab_table = st.tabs(
    ["試行を比べる", "どの設定が効いたか", "記録の中身"]
)

# ======================================================================
# タブ 1: 試行を比べる
# ======================================================================
with tab_compare:
    best = EX.best_run(runs, metric, higher_is_better)
    label = str(metric).removeprefix("結果:")

    if best is not None:
        values = pd.to_numeric(runs[metric], errors="coerce").dropna()
        kpi_row(
            [
                Kpi(f"最良の {label}", f"{best[metric]:.4f}",
                    sub=best["メモ"] or f"{best['ラボ']} / {best['記録日時'][:16]}",
                    color=theme.GOOD),
                Kpi("試行数", f"{len(runs)}", sub=f"実験「{experiment}」"),
                Kpi("最悪との差",
                    f"{abs(values.max() - values.min()):.4f}",
                    sub="設定を変えて動いた幅"),
                Kpi("ばらつき", f"{values.std():.4f}" if len(values) > 1 else "—",
                    sub="標準偏差", color=theme.PURPLE),
            ],
            accent,
        )

        with st.expander("最良だったときの設定", expanded=True):
            best_settings = {
                str(c).removeprefix("設定:"): best[c]
                for c in settings
                if pd.notna(best.get(c))
            }
            if best_settings:
                st.json(best_settings)
            else:
                st.caption("設定が記録されていません。")

    panel("試行の推移", "記録した順に並べています")
    st.plotly_chart(
        EL.history_figure(runs, metric), width="stretch", key=f"{KEY}_history"
    )
    st.caption(
        "点線が「そこまでの最良」です。**平らになったら、いまの探索範囲では"
        "頭打ち**ということ。別のつまみを触るか、探索範囲を広げる合図です。"
    )

    panel("指標のまとめ", "この実験で記録した全ての指標")
    st.dataframe(EX.summary(runs).round(4), width="stretch", hide_index=True)

# ======================================================================
# タブ 2: どの設定が効いたか
# ======================================================================
with tab_settings:
    effects = EX.which_settings_matter(runs, metric)

    if effects.empty:
        st.info(
            "比較できる設定がありません。"
            "**同じ実験名のまま、設定を変えて何度か記録**してください。"
            "2 通り以上試した設定だけが比較対象になります。"
        )
    else:
        panel(
            "設定ごとのスコアの動き",
            "値を変えたときにスコアがどれだけ動いたかで並べています",
        )
        st.plotly_chart(
            EL.importance_figure(effects), width="stretch", key=f"{KEY}_effects"
        )
        st.dataframe(effects.round(4), width="stretch", hide_index=True)
        st.caption(
            "上にある設定ほど、結果への影響が大きかったものです。"
            "**下のほうにある設定は、いくら回しても結果が変わりません。**"
            "限られた時間をどこに使うかの判断材料になります。"
        )

        panel("1 つの設定を詳しく見る", "値ごとのスコアの分布")
        chosen = st.selectbox(
            "設定", [c for c in settings if runs[c].nunique(dropna=True) > 1],
            format_func=lambda c: str(c).removeprefix("設定:"), key=f"{KEY}_setting",
        )
        st.plotly_chart(
            EL.setting_effect_figure(runs, chosen, metric),
            width="stretch", key=f"{KEY}_effect",
        )
        st.caption(
            "箱の高さは、その設定を固定しても残るばらつきです。"
            "**箱同士が重なっているなら、その設定の違いには意味がない**かもしれません。"
            "他の条件が揃っていない状態で比べていないか、確かめてください。"
        )

# ======================================================================
# タブ 3: 記録の中身
# ======================================================================
with tab_table:
    panel("記録された試行", f"実験「{experiment}」の全 {len(runs)} 件")

    display_columns = ["記録日時", "ラボ", "メモ"] + metrics + settings
    display_columns = [c for c in display_columns if c in runs.columns]
    st.dataframe(
        runs[display_columns], width="stretch", hide_index=True, height=360
    )

    col_download, col_delete = st.columns([1, 1])
    with col_download:
        st.download_button(
            "この実験を CSV で保存",
            data=EX.export_csv(runs),
            file_name=f"{experiment}_runs.csv",
            mime="text/csv",
            key=f"{KEY}_download",
            width="stretch",
        )
        st.caption("他のツールへ持ち出す場合に使ってください。")

    with col_delete:
        confirm = st.checkbox(
            f"実験「{experiment}」の記録を全て削除することを確認", key=f"{KEY}_confirm"
        )
        if st.button(
            "この実験を削除", disabled=not confirm, key=f"{KEY}_delete", width="stretch"
        ):
            removed = EX.delete_experiment(experiment)
            st.success(f"{removed} 件を削除しました。")
            st.rerun()

    panel("実行環境", "同じコードでも環境が変われば結果は変わりえます")
    st.json(EX.environment())
    st.caption(
        f"記録の実体: `{info['path']}`　—　"
        "Parquet 1 ファイルなので、消したいときはこのファイルを消すだけです。"
    )

explain("experiments")
