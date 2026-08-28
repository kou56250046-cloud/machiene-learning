"""時系列ラボの図。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mllab.models import timeseries as TS
from mllab.viz import theme


def series_figure(
    series: pd.Series,
    label: str,
    moving_windows: tuple[int, ...] = (),
    height: int = 380,
) -> go.Figure:
    """系列そのものと、移動平均の重ね描き。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index, y=series.to_numpy(), mode="lines", name=label,
            line=dict(color=theme.rgba(theme.CYAN, 0.75), width=1.2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4g}<extra></extra>",
        )
    )
    for i, window in enumerate(moving_windows):
        if window < 2 or window >= len(series):
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.rolling(window).mean().to_numpy(),
                mode="lines", name=f"移動平均 {window}",
                line=dict(color=theme.class_color(i + 3), width=2.4),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4g}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        xaxis=dict(title=None, rangeslider=dict(visible=True, thickness=0.06)),
        yaxis=dict(title=label),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def decomposition_figure(
    decomposition: TS.Decomposition,
    anomaly_points: pd.Series | None = None,
    height: int = 640,
) -> go.Figure:
    """元系列 / トレンド / 季節 / 残差 の 4 段。

    上から順に「元のかたち」「長期の向き」「毎年の繰り返し」「説明できない分」。
    足し合わせると元系列に戻る。
    """
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        subplot_titles=("元の系列", "トレンド（長期の向き）", "季節（周期の繰り返し）", "残差（説明できない分）"),
    )
    panels = [
        (decomposition.observed, theme.CYAN),
        (decomposition.trend, theme.ORANGE),
        (decomposition.seasonal, theme.LIME),
        (decomposition.residual, theme.PURPLE),
    ]
    for row, (component, color) in enumerate(panels, start=1):
        fig.add_trace(
            go.Scatter(
                x=component.index, y=component.to_numpy(), mode="lines",
                line=dict(color=color, width=1.3), showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4g}<extra></extra>",
            ),
            row=row, col=1,
        )

    if anomaly_points is not None and len(anomaly_points):
        fig.add_trace(
            go.Scatter(
                x=anomaly_points.index, y=anomaly_points.to_numpy(), mode="markers",
                name="異常", showlegend=False,
                marker=dict(color=theme.PINK, size=7, symbol="x",
                            line=dict(color=theme.BG, width=0.5)),
                hovertemplate="%{x|%Y-%m-%d}<br>残差 %{y:.4g}<extra></extra>",
            ),
            row=4, col=1,
        )

    fig.update_layout(height=height, margin=dict(l=56, r=24, t=36, b=40))
    fig.update_annotations(font=dict(size=12, color=theme.TEXT_MUTED))
    fig.update_xaxes(showgrid=False)
    return fig


def _correlogram(
    axis: str,
    lags: np.ndarray,
    values: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    color: str,
) -> list[go.Scatter | go.Bar]:
    """棒＋信頼区間の帯。帯の外に出た棒は「偶然とは言えない」。"""
    return [
        go.Scatter(
            x=np.concatenate([lags, lags[::-1]]),
            y=np.concatenate([high, low[::-1]]),
            fill="toself", fillcolor=theme.rgba(theme.TEXT_MUTED, 0.16),
            line=dict(width=0), hoverinfo="skip", showlegend=False, name="95%区間",
        ),
        go.Bar(
            x=lags, y=values,
            marker=dict(color=color, line=dict(width=0)),
            showlegend=False,
            hovertemplate=f"{axis} ラグ %{{x}}<br>相関 %{{y:.3f}}<extra></extra>",
        ),
    ]


def autocorrelation_figure(ac: TS.Autocorrelation, height: int = 420) -> go.Figure:
    """ACF と PACF を上下に並べる。"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11,
        subplot_titles=(
            "自己相関 ACF — k 点前の自分との相関",
            "偏自己相関 PACF — 途中のラグの影響を除いた分",
        ),
    )
    for trace in _correlogram("ACF", ac.lags, ac.acf_values, ac.acf_low, ac.acf_high, theme.CYAN):
        fig.add_trace(trace, row=1, col=1)
    for trace in _correlogram("PACF", ac.lags, ac.pacf_values, ac.pacf_low, ac.pacf_high, theme.PINK):
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(height=height, margin=dict(l=56, r=24, t=44, b=44), bargap=0.1)
    fig.update_annotations(font=dict(size=12, color=theme.TEXT_MUTED))
    fig.update_xaxes(title_text="ラグ（何点前か）", row=2, col=1)
    fig.update_yaxes(range=[-1.05, 1.05])
    return fig


def horizon_figure(sweep: pd.DataFrame, metric: str = "R2", height: int = 400) -> go.Figure:
    """予測期間ごとの成績。手法の優劣が入れ替わる様子を見る図。"""
    fig = go.Figure()
    for i, (label, group) in enumerate(sweep.groupby("手法", sort=False)):
        is_baseline = TS.FORECASTERS[group["key"].iloc[0]].is_baseline
        fig.add_trace(
            go.Scatter(
                x=group["予測期間"], y=group[metric], mode="lines+markers", name=label,
                line=dict(
                    color=theme.class_color(i), width=2.5,
                    dash="dot" if is_baseline else "solid",
                ),
                marker=dict(size=8),
                hovertemplate=f"{label}<br>%{{x}} 点先<br>{metric}=%{{y:.3f}}<extra></extra>",
            )
        )
    yaxis: dict = dict(title=metric)
    if metric == "R2":
        fig.add_hline(
            y=0, line=dict(color=theme.TEXT_MUTED, width=1.5, dash="dash"),
            annotation_text="R²=0（平均を返すのと同じ）",
            annotation_font=dict(color=theme.TEXT_MUTED, size=10),
        )
        # R² が大きく負になる手法があると、肝心の 0〜1 が潰れて読めなくなる。
        # −1 を下回った時点で「平均より明確に悪い」と分かれば十分なので切り詰める。
        lowest = float(sweep[metric].min())
        yaxis["range"] = [max(-1.15, lowest - 0.1), 1.05]
        if lowest < -1.15:
            fig.add_annotation(
                xref="paper", yref="paper", x=0, y=0, showarrow=False,
                text=f"※ R² が −1 を下回る手法は下端で切っています（最小 {lowest:.2f}）",
                font=dict(color=theme.TEXT_MUTED, size=10), yshift=-38,
            )
    fig.update_layout(
        height=height,
        xaxis=dict(title="何点先を予測するか", type="log",
                   tickvals=sorted(sweep["予測期間"].unique())),
        yaxis=yaxis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def forecast_figure(result: TS.BacktestResult, label: str, height: int = 400) -> go.Figure:
    """最後の分割での実測と予測の重ね描き。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.timestamps, y=result.actual, mode="lines", name="実測",
            line=dict(color=theme.rgba(theme.TEXT, 0.8), width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>実測 %{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.timestamps, y=result.predicted, mode="lines", name="予測",
            line=dict(color=theme.PINK, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>予測 %{y:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        xaxis=dict(title=None),
        yaxis=dict(title=label),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def split_figure(ranges: list[dict], height: int = 300) -> go.Figure:
    """TimeSeriesSplit がどの期間を使うかを帯で示す。

    常に「過去で学習し、未来で検証」していることが一目で分かる。
    """
    fig = go.Figure()
    for row in ranges:
        name = f"分割 {row['分割']}"
        fig.add_trace(
            go.Scatter(
                x=[row["訓練開始"], row["訓練終了"]], y=[name, name],
                mode="lines", showlegend=row["分割"] == 1, name="訓練",
                line=dict(color=theme.CYAN, width=14),
                hovertemplate="訓練 %{x|%Y-%m-%d}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row["検証開始"], row["検証終了"]], y=[name, name],
                mode="lines", showlegend=row["分割"] == 1, name="検証",
                line=dict(color=theme.PINK, width=14),
                hovertemplate="検証 %{x|%Y-%m-%d}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title=None, autorange="reversed", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=80, r=24, t=16, b=40),
    )
    return fig


def scores_figure(results: list[TS.BacktestResult], metric: str, height: int = 340) -> go.Figure:
    """手法ごとの分割スコアを箱ひげで比べる。"""
    fig = go.Figure()
    for i, result in enumerate(results):
        values = [v for v in result.scores.get(metric, []) if np.isfinite(v)]
        if not values:
            continue
        color = theme.class_color(i)
        fig.add_trace(
            go.Box(
                y=values, name=result.label, boxpoints="all", jitter=0.5, pointpos=0,
                marker=dict(color=color, size=7, line=dict(color=theme.BG, width=1)),
                line=dict(color=color), fillcolor=theme.rgba(color, 0.15),
                hovertemplate="%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height, showlegend=False,
        yaxis=dict(title=f"分割ごとの {metric}"),
        xaxis=dict(title=None),
    )
    return fig


def importance_figure(
    names: list[str], values: np.ndarray, top: int = 15, height: int | None = None
) -> go.Figure:
    """予測モデルが、どのラグ・どの特徴量を使ったか。"""
    order = np.argsort(values)[::-1][:top][::-1]
    fig = go.Figure(
        go.Bar(
            x=values[order], y=[names[i] for i in order], orientation="h",
            marker=dict(color=values[order], colorscale=theme.SEQUENTIAL,
                        showscale=False, line=dict(width=0)),
            hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height or max(260, 26 * len(order) + 90),
        showlegend=False,
        xaxis=dict(title="重要度"), yaxis=dict(title=None),
        margin=dict(l=150, r=24, t=16, b=44),
    )
    return fig
