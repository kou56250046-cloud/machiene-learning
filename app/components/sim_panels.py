"""業務シミュレーションの図。

シナリオが増えても使い回せるものだけをここに置く。
在庫に固有の図（需要と発注の重ね描きなど）もここにあるが、
「時間を進めて損益を積む」形はどのシナリオも同じなので、
累積損益・方策比較・損失の分解・精度と利益の散布図は共通で使える。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mllab.sim.core import Ledger
from mllab.viz import theme


# ======================================================================
# 世界を見る
# ======================================================================


def demand_figure(frame: pd.DataFrame, height: int = 420) -> go.Figure:
    """需要の推移と、その裏で効いている外生変数。

    上段が需要（販促日に印）、下段が気温。合成データなので
    「需要がなぜその形になっているか」を隠さずに見せる。
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("日々の需要", "気温（需要を動かしている外生変数）"),
    )

    fig.add_trace(
        go.Scatter(
            x=frame.index, y=frame["需要"], mode="lines", name="需要",
            line=dict(color=theme.rgba(theme.CYAN, 0.8), width=1.2),
            hovertemplate="%{x|%Y-%m-%d}<br>需要 %{y:.0f} 個<extra></extra>",
        ),
        row=1, col=1,
    )

    promo = frame[frame["販促"]]
    if len(promo):
        fig.add_trace(
            go.Scatter(
                x=promo.index, y=promo["需要"], mode="markers", name="販促日",
                marker=dict(color=theme.PINK, size=7, line=dict(color=theme.BG, width=1)),
                hovertemplate="%{x|%Y-%m-%d}<br>販促日 %{y:.0f} 個<extra></extra>",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=frame.index, y=frame["気温"], mode="lines", name="気温",
            line=dict(color=theme.ORANGE, width=1.2), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} ℃<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
        margin=dict(l=48, r=24, t=56, b=40),
    )
    fig.update_yaxes(title="個/日", row=1, col=1)
    fig.update_yaxes(title="℃", row=2, col=1)
    return fig


def weekday_profile_figure(frame: pd.DataFrame, height: int = 300) -> go.Figure:
    """曜日ごとの需要の分布。予測が拾うべき構造がここに出る。"""
    names = ["月", "火", "水", "木", "金", "土", "日"]
    fig = go.Figure()
    for i, name in enumerate(names):
        values = frame.loc[frame.index.dayofweek == i, "需要"]
        if values.empty:
            continue
        color = theme.class_color(i)
        fig.add_trace(
            go.Box(
                y=values, name=name, boxpoints=False,
                marker=dict(color=color), line=dict(color=color),
                fillcolor=theme.rgba(color, 0.15),
                hovertemplate="%{y:.0f} 個<extra></extra>",
            )
        )
    fig.update_layout(
        height=height, showlegend=False,
        xaxis=dict(title="曜日"), yaxis=dict(title="需要（個/日）"),
    )
    return fig


# ======================================================================
# 予測を見る
# ======================================================================


def forecast_figure(
    signals: pd.DataFrame, truth_column: str, days: int = 90, height: int = 380
) -> go.Figure:
    """予測と実績の重ね描き。平均と分位点がどれだけ離れているかを見る。

    **分位点の線が平均より上にあるか下にあるかが、そのまま方策の差になる。**
    """
    shown = signals.tail(days)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=shown.index, y=shown[truth_column], mode="lines", name="実績",
            line=dict(color=theme.rgba(theme.TEXT_MUTED, 0.9), width=1.4),
            hovertemplate="%{x|%Y-%m-%d}<br>実績 %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=shown.index, y=shown["平均"], mode="lines", name="予測（平均）",
            line=dict(color=theme.CYAN, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>平均 %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=shown.index, y=shown["分位点"], mode="lines", name="予測（分位点）",
            line=dict(color=theme.LIME, width=2, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>分位点 %{y:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        yaxis=dict(title="カバー期間の需要（個）"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def error_histogram(
    signals: pd.DataFrame, truth_column: str, height: int = 300
) -> go.Figure:
    """予測誤差の分布。0 を中心に左右対称なら、平均としては良い予測。

    ただし**左右対称であることと、儲かることは別**である。
    損が非対称なら、わざと偏らせたほうが利益は大きくなる。
    """
    error = (signals["平均"] - signals[truth_column]).dropna()
    fig = go.Figure(
        go.Histogram(
            x=error, nbinsx=40,
            marker=dict(color=theme.rgba(theme.CYAN, 0.55),
                        line=dict(color=theme.CYAN, width=1)),
            hovertemplate="誤差 %{x:.0f}<br>%{y} 日<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=theme.TEXT_MUTED, width=1, dash="dash"))
    fig.update_layout(
        height=height, showlegend=False,
        xaxis=dict(title="予測 − 実績（個）　右が過大予測"),
        yaxis=dict(title="日数"),
    )
    return fig


# ======================================================================
# 決める
# ======================================================================


def decision_figure(ledger: Ledger, days: int = 60, height: int = 400) -> go.Figure:
    """需要・発注・在庫の重ね描き。方策の性格がそのまま形に出る。"""
    frame = ledger.frame().tail(days)
    x = frame["日付"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x, y=frame["需要"], name="需要",
            marker=dict(color=theme.rgba(theme.TEXT_MUTED, 0.35), line=dict(width=0)),
            hovertemplate="%{x|%Y-%m-%d}<br>需要 %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=frame["発注"], mode="lines+markers", name="発注量",
            line=dict(color=theme.LIME, width=2), marker=dict(size=5),
            hovertemplate="%{x|%Y-%m-%d}<br>発注 %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=frame["欠品"], mode="lines", name="欠品",
            line=dict(color=theme.PINK, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>欠品 %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=frame["廃棄"], mode="lines", name="廃棄",
            line=dict(color=theme.ORANGE, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>廃棄 %{y:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height, barmode="overlay",
        yaxis=dict(title="個"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ======================================================================
# 損得を見る
# ======================================================================


def cumulative_profit_figure(
    ledgers: dict[str, Ledger], oracle_label: str = "", height: int = 420
) -> go.Figure:
    """方策ごとの累積利益。**傾きの差がそのまま毎日の取りこぼし。**"""
    fig = go.Figure()
    for i, (label, ledger) in enumerate(ledgers.items()):
        series = ledger.cumulative()
        if series.empty:
            continue
        is_oracle = label == oracle_label
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.to_numpy(), mode="lines", name=label,
                line=dict(
                    color=theme.TEXT_MUTED if is_oracle else theme.class_color(i),
                    width=2.6 if is_oracle else 2,
                    dash="dot" if is_oracle else "solid",
                ),
                hovertemplate="%{x|%Y-%m-%d}<br>" + label + " %{y:,.0f} 円<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        yaxis=dict(title="累積利益（円）"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def policy_comparison_figure(
    summary: pd.DataFrame, oracle_label: str, height: int | None = None
) -> go.Figure:
    """方策別の総利益。オラクルを破線で引き、そこまでの距離を見る。"""
    ordered = summary.sort_values("利益")
    oracle = float(summary.loc[summary["方策"] == oracle_label, "利益"].iloc[0])
    colors = [
        theme.TEXT_MUTED if name == oracle_label else theme.class_color(i)
        for i, name in enumerate(ordered["方策"])
    ]
    fig = go.Figure(
        go.Bar(
            x=ordered["利益"], y=ordered["方策"], orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:,.0f}" for v in ordered["利益"]],
            textposition="auto",
            customdata=np.stack(
                [ordered["欠品率"], ordered["廃棄率"], ordered["利益"] / oracle], axis=-1
            ),
            hovertemplate=(
                "%{y}<br>利益 %{x:,.0f} 円"
                "<br>欠品率 %{customdata[0]:.1%}　廃棄率 %{customdata[1]:.1%}"
                "<br>オラクル比 %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=oracle, line=dict(color=theme.TEXT_MUTED, width=1.5, dash="dash"),
        annotation_text="オラクル（上限）", annotation_position="top",
    )
    fig.update_layout(
        height=height or max(260, 46 * len(ordered) + 90),
        showlegend=False,
        xaxis=dict(title="期間を通した利益（円）"),
        yaxis=dict(title=None),
        margin=dict(l=150, r=24, t=36, b=44),
    )
    return fig


def breakdown_waterfall(ledger: Ledger, height: int = 380) -> go.Figure:
    """損益の内訳。売上から何がどれだけ引かれて利益に落ちるか。"""
    items = ledger.by_item()
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(items) + ["total"],
            x=list(items.index) + ["利益"],
            y=list(items.to_numpy()) + [0],
            connector=dict(line=dict(color=theme.BORDER)),
            increasing=dict(marker=dict(color=theme.LIME)),
            decreasing=dict(marker=dict(color=theme.PINK)),
            totals=dict(marker=dict(color=theme.CYAN)),
            hovertemplate="%{x}<br>%{y:,.0f} 円<extra></extra>",
        )
    )
    fig.update_layout(
        height=height, showlegend=False,
        yaxis=dict(title="円"), xaxis=dict(title=None),
    )
    return fig


def loss_decomposition_figure(parts: dict[str, float], height: int = 380) -> go.Figure:
    """オラクルからの距離を、予測のせいと方策のせいに分ける。

    **どちらが大きいかで、次に手を入れる場所が決まる。**
    予測誤差ぶんが大きければモデルを、方策ぶんが大きければ発注ルールを直す。
    """
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["オラクル利益", "予測誤差による損失", "方策の非最適性による損失", "実現利益"],
            y=[
                parts["オラクル利益"],
                -parts["予測誤差による損失"],
                -parts["方策の非最適性による損失"],
                0,
            ],
            connector=dict(line=dict(color=theme.BORDER)),
            increasing=dict(marker=dict(color=theme.LIME)),
            decreasing=dict(marker=dict(color=theme.PINK)),
            totals=dict(marker=dict(color=theme.CYAN)),
            hovertemplate="%{x}<br>%{y:,.0f} 円<extra></extra>",
        )
    )
    fig.update_layout(
        height=height, showlegend=False,
        yaxis=dict(title="円"), xaxis=dict(title=None),
    )
    return fig


# ======================================================================
# 精度と利益の関係
# ======================================================================


def accuracy_profit_scatter(grid: pd.DataFrame, height: int = 460) -> go.Figure:
    """横軸に予測精度、縦軸に利益。**このラボの結論が出る図。**

    右へ行くほど予測は下手。点が 1 本の線に乗らないことが要点で、
    - **同じ RMSE でも、方策次第で利益は大きく変わる**（縦の散らばり）
    - **RMSE を下げても、利益が同じだけ伸びるとは限らない**（横の傾き）
    の 2 つが同時に見える。
    """
    fig = go.Figure()
    for i, (policy, part) in enumerate(grid.groupby("方策", sort=False)):
        ordered = part.sort_values("RMSE")
        fig.add_trace(
            go.Scatter(
                x=ordered["RMSE"], y=ordered["利益"], mode="lines+markers",
                name=str(policy),
                line=dict(color=theme.class_color(i), width=1.6, dash="dot"),
                marker=dict(size=12, color=theme.class_color(i),
                            line=dict(color=theme.BG, width=1.5)),
                customdata=ordered["予測手法"],
                hovertemplate=(
                    "%{customdata}　×　" + str(policy)
                    + "<br>RMSE %{x:.1f}<br>利益 %{y:,.0f} 円<extra></extra>"
                ),
            )
        )

    # 手法名は点に添えると重なって読めなくなる。同じ手法は同じ RMSE に縦に並ぶので、
    # 列の見出しとして上端に 1 回だけ置く。RMSE が近い手法どうしは
    # 見出しも重なるので、高さを互い違いにずらす
    methods = grid.drop_duplicates("予測手法").sort_values("RMSE")
    for order, (_, row) in enumerate(methods.iterrows()):
        fig.add_vline(
            x=float(row["RMSE"]),
            line=dict(color=theme.rgba(theme.TEXT_MUTED, 0.25), width=1, dash="dot"),
        )
        fig.add_annotation(
            x=float(row["RMSE"]), y=1.0, yref="paper", yanchor="bottom",
            yshift=2 + 15 * (order % 2),
            text=str(row["予測手法"]), showarrow=False,
            font=dict(size=10, color=theme.TEXT_MUTED),
        )

    fig.update_layout(
        height=height,
        xaxis=dict(title="予測誤差 RMSE（右へ行くほど下手）"),
        yaxis=dict(title="期間を通した利益（円）"),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0),
        margin=dict(l=64, r=24, t=96, b=48),
    )
    return fig


def spread_summary(grid: pd.DataFrame, exclude_method: str = "") -> dict[str, float]:
    """散布図から読み取れることを、数字にして添える。

    「方策を変えて得られる幅」と「予測を良くして得られる幅」を並べると、
    どちらに投資すべきかが一目で分かる。

    Args:
        exclude_method: 比較から外す予測手法。**完全予見は必ず外す。**
            手に入らない予測を「良くした場合」に数えると、
            予測側の幅が不当に大きく見える。
    """
    usable = grid[grid["予測手法"] != exclude_method] if exclude_method else grid
    if usable.empty:
        return {}

    # 同じ予測のまま方策だけ変えたときの幅（最も大きい手法で測る）
    by_method = usable.groupby("予測手法")["利益"]
    policy_spread = float((by_method.max() - by_method.min()).max())

    # 同じ方策のまま予測手法だけ変えたときの幅
    by_policy = usable.groupby("方策")["利益"]
    forecast_spread = float((by_policy.max() - by_policy.min()).max())

    return {
        "方策を変えて動く幅": policy_spread,
        "予測を変えて動く幅": forecast_spread,
    }
