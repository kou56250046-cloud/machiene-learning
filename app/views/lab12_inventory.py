"""ラボ 12 — 在庫・発注シミュレーション。

予測して終わりではなく、**予測 → 発注 → 損得**まで通す最初のラボ。
ここまでのラボが精度で話を終えていたのに対し、評価は円で行う。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import sim_panels as P
from app.components import experiment_log as EL
from app.components.cards import Kpi, kpi_row, score_color
from app.components.controls import spec_controls
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.sim import core
from mllab.sim import inventory as INV
from mllab.sim.policies import z_for_service_level
from mllab.viz import theme

KEY = "lab12"
TRUTH = "実績カバー期間需要"

accent = page_header(
    number=12,
    title="在庫・発注シミュレーション",
    lede=(
        "需要を当てること自体は目的ではありません。目的は「何個仕入れるか」を決めることで、"
        "成績は精度ではなく円で出ます。1 個余った損と 1 個足りなかった損が同額でないとき、"
        "<b>RMSE を最小にする予測は利益を最大にしません。</b> それをこの画面で確かめます。"
    ),
    eyebrow="LAB 12 / SIMULATION",
)


# ======================================================================
# サイドバー
# ======================================================================

sidebar_section("世界")
world_values = spec_controls(INV.WORLD_PARAMS, KEY)
days = st.sidebar.select_slider(
    "シミュレーション期間", options=[180, 365, 730], value=365, key=f"{KEY}_days",
    format_func=lambda d: f"{d} 日",
)
seed = st.sidebar.number_input(
    "乱数シード", 0, 9999, 0, 1, key=f"{KEY}_seed",
    help="同じ値なら毎回まったく同じ需要が生成されます。",
)

sidebar_section("お金")
money_values = spec_controls(INV.MONEY_PARAMS, KEY)

sidebar_section("運用")
operation_values = spec_controls(INV.OPERATION_PARAMS, KEY)

config = INV.InventoryConfig(
    days=int(days), seed=int(seed), **world_values, **money_values, **operation_values
)

if config.margin <= 0:
    st.error(
        f"売価 {config.price:,.0f} 円が仕入原価 {config.cost:,.0f} 円を下回っています。"
        "売るほど損をする設定なので、売価を上げるか原価を下げてください。"
    )
    st.stop()

sidebar_section("予測")
forecast_key = st.sidebar.selectbox(
    "予測手法", list(INV.FORECASTS), format_func=lambda k: INV.FORECASTS[k].label,
    key=f"{KEY}_forecast",
)
st.sidebar.caption(INV.FORECASTS[forecast_key].summary)
refit_every = st.sidebar.select_slider(
    "モデルを学習し直す間隔", options=[15, 30, 60, 120], value=30, key=f"{KEY}_refit",
    format_func=lambda d: f"{d} 日ごと",
    help="毎日学習し直す現場はありません。間隔が空くほど、モデルは古い世界のまま判断します。",
)

sidebar_section("方策")
policy_key = st.sidebar.selectbox(
    "詳しく見る方策", list(INV.POLICIES), format_func=lambda k: INV.POLICIES[k].label,
    index=list(INV.POLICIES).index("quantile"), key=f"{KEY}_policy",
)
st.sidebar.caption(INV.POLICIES[policy_key].summary)
service_level = st.sidebar.slider(
    "サービスレベル（安全在庫用）", 0.50, 0.99, 0.95, 0.01, key=f"{KEY}_service",
    help="欠品しない確率として人が決める目標値。コスト構造から決まる臨界比とは別物です。",
)
fixed_quantity = st.sidebar.slider(
    "固定量方策の目標在庫", 10, 800, int(config.base_demand * config.cover_days), 10,
    key=f"{KEY}_fixed",
)

spec = INV.ForecastConfig(
    method=forecast_key, refit_every=int(refit_every), seed=int(seed)
)


# ======================================================================
# 実行（キャッシュ）
# ======================================================================


@st.cache_data(show_spinner="シミュレーション中…", max_entries=8)
def run_everything(config, spec, service_level, fixed_quantity):
    return INV.compare_policies(config, spec, service_level, fixed_quantity)


@st.cache_data(show_spinner="予測手法 × 方策を総当たり中…", max_entries=4)
def run_grid(config, spec, service_level, fixed_quantity):
    return INV.accuracy_vs_profit(config, spec, service_level, fixed_quantity)


@st.cache_data(show_spinner=False)
def build_world_frame(config):
    return INV.generate(config)


summary, ledgers, signals = run_everything(
    config, spec, float(service_level), float(fixed_quantity)
)
scores = INV.accuracy(signals)

selected_label = INV.POLICIES[policy_key].label
reference_label = INV.POLICIES[INV.REFERENCE_POLICY].label
profit = summary.set_index("方策")["利益"].astype(float)

oracle_profit = float(profit[INV.ORACLE_LABEL])
actual_profit = float(profit[selected_label])
best_profit = float(profit[reference_label])
oracle_share = actual_profit / oracle_profit if oracle_profit else float("nan")

selected_row = summary.set_index("方策").loc[selected_label]


# ======================================================================
# KPI
# ======================================================================

kpi_row(
    [
        Kpi("累積利益", f"{actual_profit:,.0f}", unit=" 円", sub=f"{selected_label}／{config.days} 日"),
        Kpi(
            "オラクル比", f"{oracle_share:.1%}",
            sub=f"上限 {oracle_profit:,.0f} 円",
            color=score_color(oracle_share, good=0.95, bad=0.75),
        ),
        Kpi(
            "欠品率", f"{selected_row['欠品率']:.1%}",
            sub=f"廃棄率 {selected_row['廃棄率']:.1%}",
            color=theme.PINK,
        ),
        Kpi(
            "臨界比 CR", f"{config.critical_ratio:.3f}",
            sub=f"Cu {config.underage:,.0f} 円 / Co {config.overage:,.0f} 円",
            color=theme.LIME,
        ),
        Kpi(
            "予測の RMSE", f"{scores['RMSE']:.1f}",
            sub=f"MAE {scores['MAE']:.1f}／偏り {scores['偏り']:+.1f}",
            color=theme.PURPLE,
        ),
    ],
    accent,
)

gap = best_profit - actual_profit
if policy_key != INV.REFERENCE_POLICY and gap > 0:
    note(
        f"{reference_label}方策なら、同じ予測のまま利益が {gap:,.0f} 円"
        f"（{gap / abs(actual_profit):.1%}）増えます — 予測ではなく決め方の差です。",
        tone="warn",
    )

tabs = st.tabs(["世界を見る", "予測する", "決める", "損得を見る", "精度と利益の関係"])


# ======================================================================
# タブ 1: 世界を見る
# ======================================================================
with tabs[0]:
    frame = build_world_frame(config)
    sim_frame = frame.tail(config.days)

    panel("需要はこうして作られている", "合成データなので、正解の仕組みを隠さずに見せます")
    st.plotly_chart(
        P.demand_figure(sim_frame), width="stretch", key=f"{KEY}_demand"
    )
    st.caption(
        "需要の平均は「基準 × 曜日 × 気温 × 販促」で作り、そこからポアソン／負の二項で"
        "実際の数を引いています。**予測モデルが拾えるのはこの構造の部分だけ**で、"
        "最後の乱数は誰にも当てられません。オラクルとの差の大半はここから来ます。"
    )

    left, right = st.columns([1, 1])
    with left:
        panel("曜日ごとの需要", "予測が拾うべき構造")
        st.plotly_chart(
            P.weekday_profile_figure(sim_frame), width="stretch", key=f"{KEY}_weekday"
        )
    with right:
        panel("需要の要約", f"{config.days} 日ぶん")
        described = sim_frame["需要"].describe()
        st.dataframe(
            pd.DataFrame(
                {
                    "項目": ["日数", "平均", "標準偏差", "最小", "中央値", "最大"],
                    "値": [
                        f"{described['count']:,.0f} 日",
                        f"{described['mean']:.1f} 個",
                        f"{described['std']:.1f} 個",
                        f"{described['min']:.0f} 個",
                        f"{described['50%']:.0f} 個",
                        f"{described['max']:.0f} 個",
                    ],
                }
            ),
            width="stretch", hide_index=True,
        )
        st.caption(
            f"平均 {described['mean']:.0f} 個に対して標準偏差が {described['std']:.0f} 個。"
            "**この散らばりの中でどこを狙うかが方策の仕事**で、"
            "散らばり自体を消すことは予測にはできません。"
        )

# ======================================================================
# タブ 2: 予測する
# ======================================================================
with tabs[1]:
    panel(
        f"{INV.FORECASTS[forecast_key].label} の予測",
        f"カバー期間 {config.cover_days} 日ぶんの需要を当てる",
    )
    st.plotly_chart(
        P.forecast_figure(signals, TRUTH), width="stretch", key=f"{KEY}_forecast_fig"
    )
    st.caption(
        f"緑の点線が**臨界比 {config.critical_ratio:.2f} の分位点**です。"
        + (
            "水色の平均線より上を走っています。欠品のほうが痛いので、多めに構えるのが正解だからです。"
            if config.critical_ratio > 0.5
            else "水色の平均線より下を走っています。廃棄のほうが痛いので、絞るのが正解だからです。"
        )
        + " この 2 本の線の差が、そのまま方策の利益の差になります。"
    )

    left, right = st.columns([1.2, 1])
    with left:
        panel("予測誤差の分布", "0 を中心に対称なら、平均としては良い予測")
        st.plotly_chart(
            P.error_histogram(signals, TRUTH), width="stretch", key=f"{KEY}_error"
        )
    with right:
        panel("精度", "ここまでがラボ 9 で見ていた世界")
        st.dataframe(
            pd.DataFrame(
                {
                    "指標": list(scores),
                    "値": [f"{v:.2f}" for v in scores.values()],
                }
            ),
            width="stretch", hide_index=True,
        )
        st.caption(
            "**この表だけを見て「良い予測ができた」と判断してはいけません。**"
            "同じ RMSE でも、次のタブの決め方次第で利益は何割も変わります。"
            "5 番目のタブで、その散らばりを実際に見られます。"
        )

    panel("手法ごとの性格", "分位点を直接学習できるかどうかが効いてきます")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "手法": item.label,
                    "分位点": "直接学習できる" if item.native_quantile else "正規分布を仮定して代用",
                    "説明": item.summary,
                }
                for item in INV.FORECASTS.values()
            ]
        ),
        width="stretch", hide_index=True,
    )

# ======================================================================
# タブ 3: 決める
# ======================================================================
with tabs[2]:
    panel("方策 — 予測を発注量に変える規則", "同じ予測でも、規則が違えば結果は変わります")

    implied = z_for_service_level(config.critical_ratio)
    chosen_z = z_for_service_level(service_level)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "方策": item.label,
                    "目標在庫の決め方": {
                        "fixed": f"{fixed_quantity:,.0f} 個で固定",
                        "mean": "予測の平均",
                        "safety": f"予測の平均 + {chosen_z:.2f}σ",
                        "quantile": f"予測の {config.critical_ratio:.0%} 分位点",
                    }[item.key],
                    "説明": item.summary,
                }
                for item in INV.POLICIES.values()
            ]
        ),
        width="stretch", hide_index=True,
    )

    if abs(chosen_z - implied) > 0.3:
        note(
            f"サービスレベル {service_level:.0%}（z={chosen_z:.2f}）は、"
            f"いまのコスト構造が示す臨界比 {config.critical_ratio:.0%}（z={implied:.2f}）と"
            f"噛み合っていません。安全在庫は必ず"
            f"{'過剰' if chosen_z > implied else '過少'}になります。",
            tone="bad" if abs(chosen_z - implied) > 0.8 else "warn",
        )

    panel(f"{selected_label} の動き", "直近 60 日")
    st.plotly_chart(
        P.decision_figure(ledgers[selected_label]),
        width="stretch", key=f"{KEY}_decision",
    )
    st.caption(
        "灰色の棒が需要、緑の線が発注量です。ピンク（欠品）とオレンジ（廃棄）の"
        "**どちらが多く立っているかで、その方策の性格が分かります。**"
    )

    panel("この 1 日は、何をどう決めたのか", "日付を動かすと内訳が変わります")
    ledger_frame = ledgers[selected_label].frame()
    index = st.slider(
        "何日目を見るか", 1, len(ledger_frame), min(30, len(ledger_frame)), 1,
        key=f"{KEY}_day",
    )
    row = ledger_frame.iloc[index - 1]
    signal = signals.loc[row["日付"]]

    detail = pd.DataFrame(
        [
            ("① 予測（平均）", f"{signal['平均']:.1f} 個"),
            ("① 予測（分位点）", f"{signal['分位点']:.1f} 個"),
            ("② 朝の在庫ポジション", f"{row['期首在庫']:.0f} 個"),
            ("③ 発注量", f"{row['発注']:.0f} 個"),
            ("④ 実際の需要", f"{row['需要']:.0f} 個"),
            ("⑤ 売れた", f"{row['販売']:.0f} 個"),
            ("⑤ 欠品", f"{row['欠品']:.0f} 個"),
            ("⑤ 廃棄", f"{row['廃棄']:.0f} 個"),
            ("⑥ その日の利益", f"{row['利益']:,.0f} 円"),
        ],
        columns=["段階", "値"],
    )
    left, right = st.columns([1, 1.3])
    with left:
        st.dataframe(detail, width="stretch", hide_index=True)
    with right:
        st.markdown(
            f"**{pd.Timestamp(row['日付']):%Y-%m-%d}（{'月火水木金土日'[pd.Timestamp(row['日付']).dayofweek]}）**"
        )
        error = row["需要"] - signal["平均"]
        st.markdown(
            f"予測は {signal['平均']:.0f} 個、実際は {row['需要']:.0f} 個で "
            f"**{abs(error):.0f} 個の{'不足' if error > 0 else '過大'}**でした。\n\n"
            + (
                f"欠品 {row['欠品']:.0f} 個 × {config.underage:,.0f} 円ぶんの損が出ています。"
                if row["欠品"] > 0
                else f"廃棄 {row['廃棄']:.0f} 個 × 原価 {config.cost:,.0f} 円ぶんを捨てています。"
                if row["廃棄"] > 0
                else "欠品も廃棄も出ていません。この日は読み切れています。"
            )
        )
        st.caption(
            "**発注は④を見る前に決めています。** 見てから決められるなら予測は要りません。"
        )

# ======================================================================
# タブ 4: 損得を見る
# ======================================================================
with tabs[3]:
    panel("方策を横並びで比べる", "同じ世界・同じ予測。違うのは決め方だけ")
    st.plotly_chart(
        P.policy_comparison_figure(summary, INV.ORACLE_LABEL),
        width="stretch", key=f"{KEY}_policies",
    )

    shown = summary.copy()
    shown["オラクル比"] = shown["利益"] / oracle_profit
    st.dataframe(
        shown.assign(
            利益=lambda d: d["利益"].map("{:,.0f}".format),
            売上=lambda d: d["売上"].map("{:,.0f}".format),
            欠品率=lambda d: d["欠品率"].map("{:.1%}".format),
            廃棄率=lambda d: d["廃棄率"].map("{:.1%}".format),
            平均在庫=lambda d: d["平均在庫"].map("{:.1f}".format),
            オラクル比=lambda d: d["オラクル比"].map("{:.1%}".format),
        ).drop(columns=["欠品日数"]),
        width="stretch", hide_index=True,
    )

    winner = summary[summary["方策"] != INV.ORACLE_LABEL].sort_values(
        "利益", ascending=False
    ).iloc[0]
    mean_profit = float(profit[INV.POLICIES["mean"].label])
    quantile_profit = float(profit[INV.POLICIES["quantile"].label])
    edge = quantile_profit - mean_profit
    if edge > 0:
        note(
            f"分位点方策が平均予測方策を {edge:,.0f} 円上回りました"
            f"（{edge / abs(mean_profit):.1%}）。予測は同じものを使っています。",
            tone="good",
        )

    panel("累積利益", "傾きの差が、毎日の取りこぼし")
    st.plotly_chart(
        P.cumulative_profit_figure(ledgers, INV.ORACLE_LABEL),
        width="stretch", key=f"{KEY}_cumulative",
    )

    left, right = st.columns(2)
    with left:
        panel(f"{selected_label} の損益内訳", "売上から何が引かれて利益になるか")
        st.plotly_chart(
            P.breakdown_waterfall(ledgers[selected_label]),
            width="stretch", key=f"{KEY}_waterfall",
        )
    with right:
        panel("オラクルとの差はどこから来たか", "次に手を入れる場所が決まります")
        parts = core.decompose(actual_profit, best_profit, oracle_profit)
        st.plotly_chart(
            P.loss_decomposition_figure(parts),
            width="stretch", key=f"{KEY}_decomp",
        )

    forecast_loss = parts["予測誤差による損失"]
    policy_loss = parts["方策の非最適性による損失"]
    if policy_loss > forecast_loss:
        st.caption(
            f"**方策のせいの損（{policy_loss:,.0f} 円）が、予測のせいの損（{forecast_loss:,.0f} 円）"
            "より大きい状態です。** いまはモデルを磨くより、発注ルールを直すほうが効きます。"
        )
    else:
        st.caption(
            f"**予測のせいの損（{forecast_loss:,.0f} 円）のほうが大きい状態です。**"
            f"方策を最良（{reference_label}）にしてもここまでしか届かないので、"
            "伸ばすなら特徴量やモデルに手を入れることになります。"
            "ただしオラクルとの差には、誰にも当てられない乱数ぶんが含まれている点に注意してください。"
        )

# ======================================================================
# タブ 5: 精度と利益の関係
# ======================================================================
with tabs[4]:
    panel("予測手法 × 方策の総当たり", "このラボの結論が出る図です")
    st.markdown(
        "予測手法を変えると横に、方策を変えると縦に動きます。"
        "**点が 1 本の右下がりの線に乗らないこと**が要点です。"
    )

    if st.button("総当たりを実行する", type="primary", key=f"{KEY}_grid_run"):
        st.session_state[f"{KEY}_grid_done"] = True

    if not st.session_state.get(f"{KEY}_grid_done"):
        st.info(
            f"{len(INV.FORECASTS) + 1} 種類の予測 × {len(INV.POLICIES)} 種類の方策を回します。"
            "10〜30 秒かかるので、ボタンを押したときだけ実行します。"
        )
    else:
        grid = run_grid(config, spec, float(service_level), float(fixed_quantity))
        st.plotly_chart(
            P.accuracy_profit_scatter(grid), width="stretch", key=f"{KEY}_scatter"
        )

        # 完全予見は手に入らないので、幅の比較からは外す
        spread = P.spread_summary(grid, exclude_method=INV.ORACLE_LABEL)
        left, right = st.columns(2)
        left.metric(
            "方策を変えて動く幅", f"{spread['方策を変えて動く幅']:,.0f} 円",
            help="予測はそのままで、決め方だけを変えたときに動く利益の幅です。",
        )
        right.metric(
            "予測を変えて動く幅", f"{spread['予測を変えて動く幅']:,.0f} 円",
            help="方策はそのままで、実在する予測手法を取り替えたときに動く幅です。完全予見は含めていません。",
        )

        if spread["方策を変えて動く幅"] > spread["予測を変えて動く幅"]:
            note(
                "<b>同じ予測のまま決め方を変えるほうが、予測を良くするより効く</b>状態です。"
                "精度の改善だけを追いかけていると、この差を取り逃します。",
                tone="good",
            )

        st.dataframe(
            grid.assign(
                RMSE=lambda d: d["RMSE"].map("{:.1f}".format),
                利益=lambda d: d["利益"].map("{:,.0f}".format),
                欠品率=lambda d: d["欠品率"].map("{:.1%}".format),
                廃棄率=lambda d: d["廃棄率"].map("{:.1%}".format),
            )[["予測手法", "方策", "RMSE", "利益", "欠品率", "廃棄率"]],
            width="stretch", hide_index=True,
        )
        st.caption(
            "一番上の行（完全予見）は RMSE が 0 なので、どの方策でも同じ利益になります。"
            "**予測が完璧なら方策の違いは消える** — 逆に言えば、"
            "方策が要るのは予測が外れるからです。"
        )

# ======================================================================
# 実験の記録
# ======================================================================

EL.record_panel(
    lab="在庫・発注",
    params={
        "予測手法": INV.FORECASTS[forecast_key].label,
        "方策": selected_label,
        "臨界比": round(config.critical_ratio, 3),
        "売価": config.price,
        "仕入原価": config.cost,
        "欠品ペナルティ": config.stockout_penalty,
        "保管費": config.holding,
        "リードタイム": config.lead_time,
        "賞味期限": config.shelf_life,
        "サービスレベル": service_level,
        "需要のばらつき": config.dispersion,
        "期間": config.days,
        "シード": config.seed,
    },
    metrics={
        "利益": actual_profit,
        "オラクル比": oracle_share,
        "欠品率": float(selected_row["欠品率"]),
        "廃棄率": float(selected_row["廃棄率"]),
        "RMSE": scores["RMSE"],
    },
    key=KEY,
    default_experiment="在庫/方策比較",
)

explain(
    "inventory",
    values={
        "p": config.price,
        "c": config.cost,
        "b": config.stockout_penalty,
        "h": config.holding,
        "Cu": config.underage,
        "Co": config.overage,
        "CR": round(config.critical_ratio, 3),
        "L": config.lead_time,
        "cover": config.cover_days,
    },
)
