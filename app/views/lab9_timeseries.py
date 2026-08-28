"""ラボ 9 — 時系列。

カタログに溜めた気象・株価を使い、分解 → 周期の発見 → 予測 を回す。
テーブルデータと違い「行の順序に意味がある」ことが全ての制約の source になる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import timeseries_plots as P
from app.components.cards import Kpi, kpi_row, score_color
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.data import store
from mllab.models import timeseries as TS
from mllab.viz import theme

KEY = "lab9"

accent = page_header(
    number=9,
    title="時系列ラボ",
    lede=(
        "行の順序に意味があるデータです。ランダムに分けてはいけないし、"
        "未来の情報を特徴量に混ぜてもいけない。"
        "気象と株価を切り替えてみてください。同じ「時系列」でも、"
        "予測できるものとできないものがあることが一目で分かります。"
    ),
)

# ======================================================================
# データの選択
# ======================================================================

sidebar_section("データ")

candidates = [
    dataset
    for dataset in store.list_datasets()
    if TS.find_time_columns(store.load(dataset.name))
]

if not candidates:
    st.warning(
        "日時の列を持つデータがカタログにありません。"
        "**データカタログ**で「気象データ」または「株価・指数」を取得してください。",
        icon="📭",
    )
    explain("timeseries")
    st.stop()

dataset_name = st.sidebar.selectbox(
    "データセット",
    [d.name for d in candidates],
    format_func=lambda n: f"{n} — {next(d.label for d in candidates if d.name == n)}",
    key=f"{KEY}_dataset",
)
frame = store.load(dataset_name)

time_columns = TS.find_time_columns(frame)
time_column = (
    time_columns[0]
    if len(time_columns) == 1
    else st.sidebar.selectbox("日時の列", time_columns, key=f"{KEY}_time")
)

value_columns = TS.find_value_columns(frame, time_column)
if not value_columns:
    st.error(f"`{dataset_name}` に数値の観測値がありません。")
    st.stop()

value_column = st.sidebar.selectbox(
    "観測値の列", value_columns, key=f"{KEY}_value_{dataset_name}"
)

frequency = st.sidebar.selectbox(
    "粒度", list(TS.FREQUENCIES),
    format_func=lambda k: TS.FREQUENCIES[k][1],
    key=f"{KEY}_freq",
    help="日次のままだと 1 年周期を見るのに 365 点必要です。月次に落とすと軽くなります。",
)
aggregation = st.sidebar.selectbox(
    "まとめ方", list(TS.AGGREGATIONS),
    format_func=lambda k: TS.AGGREGATIONS[k],
    index=list(TS.AGGREGATIONS).index("mean"),
    key=f"{KEY}_agg",
    help="株価の終値のように「その期間の最後の値」が欲しいときは『最後の値』を選びます。",
)

series_config = TS.SeriesConfig(time_column, value_column, frequency, aggregation)


@st.cache_data(show_spinner=False)
def load_series(_name: str, _config, frame):
    return TS.build_series(frame, _config), TS.gaps_in_series(frame, _config)


series, gaps = load_series(dataset_name, series_config, frame)

if len(series) < 24:
    st.error(f"点数が {len(series)} 点しかありません。粒度を細かくするか、期間の長いデータを選んでください。")
    st.stop()

# ======================================================================
# 分解と予測の設定
# ======================================================================

sidebar_section("分解")
default_period = TS.suggest_period(frequency)
max_period = max(2, len(series) // 2)
period = st.sidebar.number_input(
    "季節の周期（何点で 1 周するか）",
    min_value=2, max_value=max_period,
    value=min(default_period, max_period), step=1,
    key=f"{KEY}_period_{frequency}",
    help="日次なら 365（1 年）、週次なら 52、月次なら 12 が目安です。",
)
anomaly_threshold = st.sidebar.slider(
    "異常とみなす残差の大きさ", 2.0, 6.0, 3.0, 0.5, key=f"{KEY}_thresh",
    help="いつもの散らばりの何倍離れたら異常とみなすか。下げるほど多く拾います。",
)

sidebar_section("予測")
horizon = st.sidebar.select_slider(
    "何点先を予測するか", options=list(TS.HORIZON_SWEEP), value=7, key=f"{KEY}_horizon",
    help="このラボで最も効く設定です。1 点先と 90 点先では、勝つ手法が入れ替わります。",
)
n_lags = st.sidebar.slider("使うラグの数", 1, 30, 14, 1, key=f"{KEY}_lags",
                           help="何点前までを特徴量にするか。")
use_calendar = st.sidebar.toggle(
    "カレンダー特徴を使う", value=True, key=f"{KEY}_cal",
    help="月・曜日・年内位置。季節性のあるデータでは効きます。",
)
n_splits = st.sidebar.slider("時系列交差検証の分割数", 2, 8, 5, 1, key=f"{KEY}_splits")

feature_config = TS.FeatureConfig(
    n_lags=int(n_lags), windows=(7, 28), calendar=use_calendar, horizon=int(horizon)
)

# ======================================================================
# 計算
# ======================================================================


@st.cache_data(show_spinner="分解中…")
def run_decompose(_name, _config, period, series):
    return TS.decompose(series, int(period))


@st.cache_data(show_spinner="自己相関を計算中…")
def run_autocorrelation(_name, _config, n_lags, series):
    return TS.autocorrelation(series, n_lags)


@st.cache_data(show_spinner="予測を検証中…")
def run_backtests(_name, _config, _features, period, n_splits, series):
    features, target = TS.make_features(series, _features)
    results = [
        TS.backtest(key, series, features, target, n_splits,
                    _features.horizon, int(period), 0)
        for key in TS.FORECASTERS
    ]
    return features, target, results


decomposition = None
decompose_error = ""
try:
    decomposition = run_decompose(dataset_name, series_config, period, series)
except ValueError as exc:
    decompose_error = str(exc)

features, target, results = run_backtests(
    dataset_name, series_config, feature_config, period, int(n_splits), series
)
best = max(results, key=lambda r: r.mean("R2") if np.isfinite(r.mean("R2")) else -np.inf)
naive = next(r for r in results if r.key == "naive")
learned = [r for r in results if not TS.FORECASTERS[r.key].is_baseline]
best_learned = max(
    learned, key=lambda r: r.mean("R2") if np.isfinite(r.mean("R2")) else -np.inf
)

# ---- KPI --------------------------------------------------------------
kpis = [
    Kpi("点数", f"{len(series):,}",
        sub=f"{series.index.min():%Y-%m-%d} 〜 {series.index.max():%Y-%m-%d}"),
]
if decomposition is not None:
    kpis += [
        Kpi("季節性の強さ", f"{decomposition.seasonal_strength:.3f}",
            sub="1 に近いほど周期で説明できる",
            color=score_color(decomposition.seasonal_strength, good=0.6, bad=0.2)),
        Kpi("トレンドの強さ", f"{decomposition.trend_strength:.3f}",
            sub="1 に近いほど長期の向きが支配的",
            color=theme.PURPLE),
    ]
kpis += [
    Kpi("最良の手法", f"{best.mean('R2'):+.3f}", sub=f"{best.label}（R²）",
        color=score_color(best.mean("R2"), good=0.7, bad=0.2)),
    Kpi("学習 − ナイーブ", f"{best_learned.mean('R2') - naive.mean('R2'):+.3f}",
        sub="学習して得た分（R² の差）",
        color=score_color(best_learned.mean("R2") - naive.mean("R2"), good=0.02, bad=0.0)),
]
kpi_row(kpis, accent)

if gaps:
    note(f"元データに無かった {gaps:,} 点を直前の値で補完しています（休場日・欠測）", tone="warn")

if TS.FORECASTERS[best.key].is_baseline:
    note(
        f"最も成績が良いのは学習しない「{best.label}」です — "
        f"{horizon} 点先の予測では、機械学習を持ち込む価値がこのデータにはありません。"
        "予測期間を伸ばすと逆転するか確かめてみてください。",
        tone="warn",
    )
elif best.mean("R2") < 0.1:
    note(
        "どの手法も R² がほぼ 0 です — この系列は過去から予測できません。"
        "株価の変化率のように、本質的にランダムに近いデータではこうなります。",
        tone="bad",
    )

tab_series, tab_decompose, tab_forecast = st.tabs(
    ["系列を見る", "分解と周期", "予測する"]
)

# ======================================================================
# タブ 1: 系列を見る
# ======================================================================
with tab_series:
    panel(f"{value_column} の推移", f"{dataset_name} / {TS.FREQUENCIES[frequency][1]}")

    windows = st.multiselect(
        "重ねる移動平均の窓",
        [7, 28, 90, 365],
        default=[7, 365] if frequency == "D" else [3, 12],
        key=f"{KEY}_windows",
        help="短い窓はノイズを追い、長い窓は長期の向きを見せます。",
    )
    st.plotly_chart(
        P.series_figure(series, value_column, tuple(windows)),
        width="stretch", key=f"{KEY}_series",
    )
    st.caption("下部のスライダで期間を絞れます。移動平均の窓を変えると、見える構造が変わります。")

    col_stats, col_diff = st.columns(2)
    with col_stats:
        panel("基本統計", "")
        st.dataframe(
            series.describe().to_frame(value_column).round(3),
            width="stretch",
        )
    with col_diff:
        panel("差分を取ると", "トレンドを取り除いた「変化量」の系列")
        differenced = series.diff().dropna()
        st.plotly_chart(
            P.series_figure(differenced, f"{value_column} の変化量", height=300),
            width="stretch", key=f"{KEY}_diff",
        )
        st.caption(
            "元の系列に強いトレンドがあると、多くの手法がうまく働きません。"
            "差分を取ると平均が一定に近づき（定常化）、扱いやすくなります。"
        )

# ======================================================================
# タブ 2: 分解と周期
# ======================================================================
with tab_decompose:
    if decomposition is None:
        st.error(decompose_error)
    else:
        panel(
            "STL 分解",
            f"周期 {period} 点。4 つを足し合わせると元の系列に戻ります",
        )

        anomaly_points = TS.anomalies(decomposition, float(anomaly_threshold))
        st.plotly_chart(
            P.decomposition_figure(decomposition, anomaly_points),
            width="stretch", key=f"{KEY}_decomp",
        )

        kpi_row(
            [
                Kpi("季節の振れ幅", f"{decomposition.seasonal_amplitude:.3g}",
                    sub=f"{value_column} の単位"),
                Kpi("季節性の強さ", f"{decomposition.seasonal_strength:.3f}",
                    sub="0.6 以上なら周期がはっきりしている",
                    color=score_color(decomposition.seasonal_strength, good=0.6, bad=0.2)),
                Kpi("トレンドの強さ", f"{decomposition.trend_strength:.3f}",
                    sub="0.6 以上なら長期の向きが支配的",
                    color=score_color(decomposition.trend_strength, good=0.6, bad=0.2)),
                Kpi("異常点", f"{len(anomaly_points)}",
                    sub=f"残差が {anomaly_threshold:.1f} 倍以上外れた点",
                    color=theme.PINK),
            ],
            accent,
        )

        if decomposition.seasonal_strength < 0.15:
            note(
                "季節性がほとんどありません — この系列は周期では説明できません。"
                "株価のようにトレンドとノイズだけで動くデータではこうなります。",
                tone="warn",
            )

        if len(anomaly_points):
            with st.expander(f"異常とみなした {len(anomaly_points)} 点", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "日時": anomaly_points.index,
                            "実測値": series.reindex(anomaly_points.index).to_numpy(),
                            "残差": anomaly_points.to_numpy().round(3),
                        }
                    ).sort_values("残差", key=lambda s: s.abs(), ascending=False),
                    width="stretch", hide_index=True, height=280,
                )
            st.caption(
                "トレンドと季節を取り除いたうえでの外れ値です。"
                "「7 月なのに寒い」のような、季節を考慮した異常を拾えます。"
            )

        panel("自己相関", "過去の自分とどれだけ似ているか")

        acf_lags = st.slider(
            "見るラグの範囲", 10, min(500, len(series) // 2 - 1),
            min(int(period) + 30, len(series) // 2 - 1), 10,
            key=f"{KEY}_acflags",
        )
        autocorr = run_autocorrelation(dataset_name, series_config, int(acf_lags), series)
        st.plotly_chart(
            P.autocorrelation_figure(autocorr), width="stretch", key=f"{KEY}_acf"
        )

        significant = autocorr.significant_lags(8)
        if significant:
            st.caption(
                "グレーの帯は「相関が無いとしたら値が収まるはずの範囲」です。"
                f"帯を超えたラグ（先頭から {'、'.join(map(str, significant))} …）は、"
                "偶然では説明できない相関があります。"
            )
        else:
            st.caption(
                "どのラグも帯の中に収まっています。過去の値との相関が検出できない "
                "＝ ホワイトノイズに近い系列です。"
            )

# ======================================================================
# タブ 3: 予測する
# ======================================================================
with tab_forecast:
    panel(
        "予測期間を変えると勝つ手法が入れ替わる",
        "点線が学習しないベースライン、実線が学習するモデル",
    )

    @st.cache_data(show_spinner="予測期間を振って検証中…")
    def run_sweep(_name, _config, _features, period, n_splits, series):
        return TS.horizon_sweep(
            series, _features, TS.HORIZON_SWEEP,
            n_splits=n_splits, period=int(period), seed=0,
        )

    sweep = run_sweep(
        dataset_name, series_config, feature_config, period, int(n_splits), series
    )
    if sweep.empty:
        st.warning("系列が短く、予測期間を振れませんでした。")
    else:
        st.plotly_chart(P.horizon_figure(sweep), width="stretch", key=f"{KEY}_sweep")

        pivot = sweep.pivot(index="予測期間", columns="手法", values="R2").round(3)
        st.dataframe(pivot, width="stretch")

        short = sweep[sweep["予測期間"] == sweep["予測期間"].min()]
        long = sweep[sweep["予測期間"] == sweep["予測期間"].max()]
        if not short.empty and not long.empty:
            short_best = short.loc[short["R2"].idxmax()]
            long_best = long.loc[long["R2"].idxmax()]
            headline = (
                f"**{int(short_best['予測期間'])} 点先**では「{short_best['手法']}」"
                f"（R²={short_best['R2']:.3f}）、"
                f"**{int(long_best['予測期間'])} 点先**では「{long_best['手法']}」"
                f"（R²={long_best['R2']:.3f}）が最良です。"
            )
            # データによって結論が変わるので、実際の数字を見て言い分けること。
            # 「長期は学習が効く」と決め打ちすると、株価では嘘になる。
            if long_best["R2"] < 0:
                tail = (
                    "ただし長期側は最良の手法でも R² が負 — "
                    "この系列でそこまで先を予測するのは、平均を返すより悪い結果になります。"
                )
            elif short_best["手法"] == long_best["手法"]:
                tail = "どの期間でも同じ手法が最良でした。"
            else:
                tail = "短期は直近の値が強く、長期は季節性や学習したパターンが効きます。"
            st.caption(headline + tail)

    panel(
        f"{horizon} 点先を予測する",
        f"{n_splits} 分割の時系列交差検証。常に過去で学習し、未来で検証します",
    )

    st.plotly_chart(
        P.split_figure(TS.split_ranges(features, int(n_splits))),
        width="stretch", key=f"{KEY}_splits_chart",
    )
    st.caption(
        "水色が訓練、ピンクが検証です。訓練期間が必ず検証期間より前にあることを確かめてください。"
        "普通の交差検証だとこれが混ざり、未来のデータで過去を当てることになります（リーク）。"
    )

    col_box, col_table = st.columns([1.3, 1])
    with col_box:
        st.markdown("**分割ごとの R²** — ばらつきが手法間の差より大きければ、その差は運かもしれません")
        st.plotly_chart(
            P.scores_figure(results, "R2"), width="stretch", key=f"{KEY}_scores"
        )
    with col_table:
        st.markdown("**成績のまとめ**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "手法": r.label,
                        "R²": round(r.mean("R2"), 3),
                        "±": round(r.std("R2"), 3),
                        "RMSE": round(r.mean("RMSE"), 4),
                        "MAE": round(r.mean("MAE"), 4),
                    }
                    for r in results
                ]
            ).sort_values("R²", ascending=False),
            width="stretch", hide_index=True,
        )

    panel("最後の分割での予測", "実測とどれだけ重なっているか")
    shown_key = st.selectbox(
        "手法", [r.key for r in results],
        format_func=lambda k: TS.FORECASTERS[k].label,
        index=[r.key for r in results].index(best.key),
        key=f"{KEY}_shown",
    )
    shown = next(r for r in results if r.key == shown_key)
    st.caption(TS.FORECASTERS[shown_key].summary)
    st.plotly_chart(
        P.forecast_figure(shown, value_column), width="stretch", key=f"{KEY}_forecast"
    )

    if not TS.FORECASTERS[shown_key].is_baseline:
        panel("どの特徴量が効いたか", "ラグとカレンダー特徴の重要度")

        @st.cache_data(show_spinner="重要度を計算中…")
        def fit_importance(_name, _config, _features, key, X, y):
            model = TS.FORECASTERS[key].build(0)
            model.fit(X, y)
            values = getattr(model, "feature_importances_", None)
            if values is None:
                values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
            return list(X.columns), np.asarray(values, dtype=float)

        names, importances = fit_importance(
            dataset_name, series_config, feature_config, shown_key, features, target
        )
        st.plotly_chart(
            P.importance_figure(names, importances),
            width="stretch", key=f"{KEY}_imp",
        )
        st.caption(
            "「1点前」が突出していれば、モデルは実質ナイーブ予測をしています。"
            "カレンダー特徴が上位に来るなら、季節性を学習できている証拠です。"
        )

explain("timeseries")
