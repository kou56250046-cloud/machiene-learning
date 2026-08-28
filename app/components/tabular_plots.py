"""テーブルデータラボの図。

Plotly でダークテーマに合わせて自前で描く（SHAP 付属の matplotlib 描画は
配色が合わず、明るい背景を前提にしているため使わない）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

from mllab.models import tabular as T
from mllab.viz import theme


def cv_scores_figure(cv_by_model: dict[str, np.ndarray], metric: str) -> go.Figure:
    """モデルごとの交差検証スコアを箱ひげで並べる。

    平均だけでなくばらつきを見せるのが目的。分割の運で 0.05 動くようなら、
    モデル間の 0.02 の差には意味がない。
    """
    fig = go.Figure()
    for i, (label, scores) in enumerate(cv_by_model.items()):
        color = theme.class_color(i)
        fig.add_trace(
            go.Box(
                y=scores,
                name=label,
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                marker=dict(color=color, size=7, line=dict(color=theme.BG, width=1)),
                line=dict(color=color),
                fillcolor=theme.rgba(color, 0.15),
                hovertemplate="%{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=340,
        showlegend=False,
        yaxis=dict(title=f"交差検証スコア（{metric}）"),
        xaxis=dict(title=None),
    )
    return fig


def confusion_figure(y_true, y_pred, labels: list[str]) -> go.Figure:
    """混同行列。行方向で正規化せず、実数と割合を併記する。"""
    matrix = confusion_matrix(y_true, y_pred)
    row_totals = matrix.sum(axis=1, keepdims=True)
    ratio = np.divide(matrix, np.where(row_totals == 0, 1, row_totals))
    text = np.array(
        [
            [f"<b>{matrix[i, j]}</b><br>{ratio[i, j]:.0%}" for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])
        ]
    )
    fig = go.Figure(
        go.Heatmap(
            z=ratio,
            text=text,
            texttemplate="%{text}",
            x=[f"予測: {c}" for c in labels],
            y=[f"実際: {c}" for c in labels],
            colorscale=theme.SEQUENTIAL,
            zmin=0,
            zmax=1,
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
            textfont=dict(size=13, color=theme.TEXT),
        )
    )
    fig.update_layout(
        height=90 + 62 * len(labels),
        xaxis=dict(side="top", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=110, r=20, t=50, b=20),
    )
    return fig


def roc_figure(y_true, y_proba, class_labels: list[str]) -> go.Figure | None:
    """ROC 曲線。多クラスは 1 対他で 1 本ずつ描く。"""
    classes = np.unique(y_true)
    if y_proba is None or len(classes) < 2:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="でたらめ",
            line=dict(color=theme.TEXT_MUTED, width=1.5, dash="dash"),
        )
    )
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=classes[1])
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{class_labels[1]}",
                line=dict(color=theme.CYAN, width=3),
                fill="tozeroy", fillcolor=theme.rgba(theme.CYAN, 0.12),
            )
        )
    else:
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve((np.asarray(y_true) == cls).astype(int), y_proba[:, i])
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{class_labels[i]} 対 その他",
                    line=dict(color=theme.class_color(i), width=2.5),
                )
            )
    fig.update_layout(
        height=380,
        xaxis=dict(title="偽陽性率", range=[-0.02, 1.02]),
        yaxis=dict(title="真陽性率", range=[-0.02, 1.02], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def prediction_scatter(y_true, y_pred) -> go.Figure:
    """回帰の実測 vs 予測。対角線に乗るほど良い。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 + 1e-9

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
            name="完全に当たった場合",
            line=dict(color=theme.TEXT_MUTED, width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_true, y=y_pred, mode="markers", name="予測",
            marker=dict(color=theme.CYAN, size=7, opacity=0.7,
                        line=dict(color=theme.BG, width=0.6)),
            hovertemplate="実測 %{x:.4g}<br>予測 %{y:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        xaxis=dict(title="実測値", range=[lo - pad, hi + pad]),
        yaxis=dict(title="予測値", range=[lo - pad, hi + pad],
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def residual_figure(y_true, y_pred) -> go.Figure:
    """残差プロット。0 の周りに均等に散らばっていれば健全。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=theme.TEXT_MUTED, width=1.5, dash="dash"))
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=residual, mode="markers", name="残差",
            marker=dict(
                color=residual, colorscale=theme.DIVERGING,
                cmid=0, size=7, opacity=0.75,
                line=dict(color=theme.BG, width=0.6),
                colorbar=dict(thickness=8, len=0.6, outlinewidth=0,
                              tickfont=dict(size=10, color=theme.TEXT_MUTED)),
            ),
            hovertemplate="予測 %{x:.4g}<br>残差 %{y:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=330, showlegend=False,
        xaxis=dict(title="予測値"), yaxis=dict(title="残差（実測 − 予測）"),
    )
    return fig


def importance_figure(
    frame: pd.DataFrame, value_column: str, error_column: str | None = None,
    top: int = 15, color: str | None = None,
) -> go.Figure:
    """特徴量重要度の横棒グラフ。上位のみ表示する。"""
    shown = frame.head(top).iloc[::-1]
    error = (
        dict(type="data", array=shown[error_column], color=theme.TEXT_MUTED)
        if error_column and error_column in shown
        else None
    )
    fig = go.Figure(
        go.Bar(
            x=shown[value_column],
            y=shown["特徴量"],
            orientation="h",
            error_x=error,
            marker=dict(
                color=shown[value_column] if color is None else color,
                colorscale=None if color else theme.SEQUENTIAL,
                showscale=False,
                line=dict(width=0),
            ),
            hovertemplate="%{y}<br>" + value_column + " = %{x:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(260, 26 * len(shown) + 90),
        showlegend=False,
        xaxis=dict(title=value_column),
        yaxis=dict(title=None),
        margin=dict(l=170, r=24, t=16, b=44),
    )
    return fig


def shap_beeswarm(explanation: T.Explanation, top: int = 12) -> go.Figure:
    """SHAP のビースウォーム。

    横軸が「予測をどちらへどれだけ動かしたか」、点の色が「その特徴量の値の大小」。
    右上がりに色が並べば「値が大きいほど予測を押し上げる」と読める。
    """
    order = np.argsort(np.abs(explanation.values).mean(axis=0))[::-1][:top]
    fig = go.Figure()

    rng = np.random.default_rng(0)
    for row, index in enumerate(reversed(order)):
        values = explanation.values[:, index]
        feature = explanation.features[:, index]
        # 値の大小を 0〜1 に正規化して色に使う（単位が違う特徴量を並べるため）
        finite = feature[np.isfinite(feature)]
        spread = float(np.ptp(finite)) if len(finite) else 0.0
        if spread > 0:
            normalized = (feature - finite.min()) / spread
        else:
            normalized = np.full_like(feature, 0.5)

        fig.add_trace(
            go.Scatter(
                x=values,
                y=row + (rng.random(len(values)) - 0.5) * 0.55,
                mode="markers",
                name=explanation.names[index],
                showlegend=False,
                marker=dict(
                    color=normalized,
                    colorscale=theme.DIVERGING,
                    cmin=0, cmax=1, size=5, opacity=0.7,
                    line=dict(width=0),
                    colorbar=dict(
                        title=dict(text="特徴量<br>の値", font=dict(size=10)),
                        tickvals=[0, 1], ticktext=["小", "大"],
                        thickness=8, len=0.5, outlinewidth=0,
                        tickfont=dict(size=10, color=theme.TEXT_MUTED),
                    )
                    if row == 0
                    else None,
                    showscale=row == 0,
                ),
                customdata=feature,
                hovertemplate=(
                    f"{explanation.names[index]}<br>"
                    "値 %{customdata:.4g}<br>SHAP %{x:.4g}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0, line=dict(color=theme.TEXT_MUTED, width=1.5, dash="dash"))
    fig.update_layout(
        height=max(300, 30 * len(order) + 100),
        xaxis=dict(title="SHAP 値（予測を押し上げる → / ← 押し下げる）"),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(order))),
            ticktext=[explanation.names[i] for i in reversed(order)],
            title=None,
        ),
        margin=dict(l=170, r=24, t=16, b=48),
    )
    return fig


def shap_waterfall(explanation: T.Explanation, row: int, top: int = 10) -> go.Figure:
    """1 件の予測について、各特徴量の寄与を積み上げる。

    「なぜこの行はこう予測されたのか」を 1 枚で示す図。
    """
    values = explanation.values[row]
    order = np.argsort(np.abs(values))[::-1][:top]
    others = float(values.sum() - values[order].sum())

    labels = [
        f"{explanation.names[i]} = {explanation.features[row, i]:.4g}" for i in order
    ]
    contributions = [float(values[i]) for i in order]
    if abs(others) > 1e-12:
        labels.append(f"その他 {len(values) - len(order)} 特徴量")
        contributions.append(others)

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=labels[::-1],
            x=contributions[::-1],
            connector=dict(line=dict(color=theme.BORDER, width=1)),
            increasing=dict(marker=dict(color=theme.PINK)),
            decreasing=dict(marker=dict(color=theme.CYAN)),
            hovertemplate="%{y}<br>寄与 %{x:+.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(300, 30 * len(labels) + 110),
        showlegend=False,
        xaxis=dict(title=f"基準値 {explanation.base_value:.4g} からの寄与"),
        yaxis=dict(title=None),
        margin=dict(l=250, r=24, t=16, b=48),
    )
    return fig


def missing_figure(frame: pd.DataFrame, top: int = 20) -> go.Figure | None:
    """列ごとの欠測率。欠測が無ければ None。"""
    rates = frame.isna().mean().sort_values(ascending=False)
    rates = rates[rates > 0].head(top)
    if rates.empty:
        return None

    fig = go.Figure(
        go.Bar(
            x=rates.to_numpy() * 100,
            y=[str(i) for i in rates.index],
            orientation="h",
            marker=dict(color=theme.ORANGE, line=dict(width=0)),
            hovertemplate="%{y}<br>欠測率 %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(220, 26 * len(rates) + 90),
        showlegend=False,
        xaxis=dict(title="欠測率 (%)"),
        yaxis=dict(title=None, autorange="reversed"),
        margin=dict(l=170, r=24, t=16, b=44),
    )
    return fig


def target_figure(y: pd.Series, task: str) -> go.Figure:
    """目的変数の分布。分類はクラス件数、回帰はヒストグラム。"""
    if task == T.CLASSIFICATION:
        counts = y.value_counts().sort_index()
        colors = [theme.class_color(i) for i in range(len(counts))]
        fig = go.Figure(
            go.Bar(
                x=[str(i) for i in counts.index],
                y=counts.to_numpy(),
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="%{x}<br>%{y} 件<extra></extra>",
            )
        )
        fig.update_layout(
            height=280, showlegend=False,
            xaxis=dict(title="クラス", type="category"), yaxis=dict(title="件数"),
        )
        return fig

    fig = go.Figure(
        go.Histogram(
            x=y.to_numpy(), nbinsx=40,
            marker=dict(color=theme.CYAN, line=dict(width=0)),
            hovertemplate="%{x}<br>%{y} 件<extra></extra>",
        )
    )
    fig.update_layout(
        height=280, showlegend=False,
        xaxis=dict(title="目的変数の値"), yaxis=dict(title="件数"),
    )
    return fig
