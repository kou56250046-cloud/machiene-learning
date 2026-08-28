"""決定境界の計算と描画。

decision_figure() が全ラボ共通のエントリポイント。
メッシュ計算は重いので、呼び出し側（Streamlit）でキャッシュすること。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.base import BaseEstimator

from mllab.viz import theme


def make_mesh(
    X: np.ndarray, resolution: int = 160, margin: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """データを囲む格子を作る。

    Returns:
        (xx, yy, grid)。xx / yy は (resolution, resolution)、grid は (resolution**2, 2)。
    """
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    pad_x = (x_max - x_min) * margin + 1e-6
    pad_y = (y_max - y_min) * margin + 1e-6
    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, resolution),
        np.linspace(y_min - pad_y, y_max + pad_y, resolution),
    )
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def decision_scores(model: BaseEstimator, grid: np.ndarray) -> tuple[np.ndarray, str]:
    """格子上のスコアを返す。

    2 クラスなら「クラス 1 らしさ」を 0〜1 で、多クラスなら予測クラス番号を返す。

    Returns:
        (scores, mode)。mode は proba / decision / label のいずれか。
    """
    classes = getattr(model, "classes_", None)
    n_classes = len(classes) if classes is not None else 2

    if n_classes == 2:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(grid)[:, 1], "proba"
        if hasattr(model, "decision_function"):
            d = np.asarray(model.decision_function(grid)).ravel()
            # tanh で潰すと、境界（0）付近のグラデーションが見やすくなる
            scale = np.percentile(np.abs(d), 95) + 1e-9
            return 0.5 * (np.tanh(d / scale) + 1.0), "decision"
    return model.predict(grid).astype(float), "label"


def _discrete_colorscale(n_classes: int) -> list[list]:
    """クラスごとにベタ塗りの帯を作るカラースケール。"""
    scale: list[list] = []
    for c in range(n_classes):
        color = theme.class_color(c)
        scale.append([c / n_classes, color])
        scale.append([(c + 1) / n_classes, color])
    return scale


def _scatter(
    X: np.ndarray,
    y: np.ndarray,
    name_prefix: str,
    symbol: str,
    size: int,
    opacity: float,
    line_color: str,
) -> list[go.Scatter]:
    traces = []
    for cls in np.unique(y):
        mask = y == cls
        traces.append(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=f"{name_prefix} クラス {int(cls)}",
                marker=dict(
                    color=theme.class_color(int(cls)),
                    size=size,
                    symbol=symbol,
                    opacity=opacity,
                    line=dict(color=line_color, width=1.2),
                ),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    return traces


def decision_figure(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    resolution: int = 160,
    title: str = "",
    show_contour: bool = True,
    height: int = 520,
) -> go.Figure:
    """決定境界のヒートマップとデータ点を重ねた図を作る。"""
    all_X = X_train if X_test is None else np.vstack([X_train, X_test])
    xx, yy, grid = make_mesh(all_X, resolution)
    scores, mode = decision_scores(model, grid)
    Z = scores.reshape(xx.shape)

    fig = go.Figure()

    if mode == "label":
        n_cls = int(Z.max()) + 1
        fig.add_trace(
            go.Heatmap(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                colorscale=_discrete_colorscale(n_cls),
                zmin=0,
                zmax=n_cls - 1,
                showscale=False,
                showlegend=False,
                opacity=0.30,
                hoverinfo="skip",
            )
        )
    else:
        fig.add_trace(
            go.Heatmap(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                colorscale=theme.DIVERGING,
                zmin=0.0,
                zmax=1.0,
                opacity=0.55,
                showlegend=False,
                hoverinfo="skip",
                zsmooth="best",
                colorbar=dict(
                    title=dict(text="クラス1<br>らしさ", font=dict(size=11)),
                    thickness=10,
                    len=0.6,
                    outlinewidth=0,
                    tickfont=dict(size=10, color=theme.TEXT_MUTED),
                ),
            )
        )
        if show_contour:
            # 境界そのもの（スコア 0.5 の等高線）を白で強調する
            fig.add_trace(
                go.Contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=Z,
                    contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                    line=dict(color=theme.TEXT, width=2),
                    showscale=False,
                    hoverinfo="skip",
                    # 等高線トレース自体は凡例に出さない。Plotly が
                    # coloring="none" の等高線に空の見出し（undefined）を
                    # 描いてしまうため、凡例は下のダミー線で用意する。
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name="決定境界",
                    line=dict(color=theme.TEXT, width=2),
                    hoverinfo="skip",
                )
            )

    fig.add_traces(_scatter(X_train, y_train, "訓練", "circle", 8, 0.95, theme.BG))
    if X_test is not None and y_test is not None and len(X_test):
        fig.add_traces(_scatter(X_test, y_test, "テスト", "diamond", 9, 0.9, theme.TEXT))

    fig.update_layout(
        height=height,
        xaxis=dict(title="特徴量 1", showgrid=False, constrain="domain"),
        yaxis=dict(title="特徴量 2", showgrid=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    # title=None を渡すと空の title オブジェクトになり、Plotly が
    # "undefined" という見出しを描いてしまう。中身があるときだけ設定する。
    if title:
        fig.update_layout(title=title)
    return fig


def cluster_figure(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray | None = None,
    title: str = "",
    height: int = 520,
    noise_label: int = -1,
) -> go.Figure:
    """クラスタリング結果の散布図。ラベル -1（ノイズ）はグレーで描く。"""
    fig = go.Figure()
    for cls in sorted(set(np.asarray(labels).tolist())):
        mask = labels == cls
        is_noise = cls == noise_label
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name="ノイズ" if is_noise else f"クラスタ {int(cls)}",
                marker=dict(
                    color=theme.TEXT_MUTED if is_noise else theme.class_color(int(cls)),
                    size=8,
                    symbol="x" if is_noise else "circle",
                    opacity=0.5 if is_noise else 0.95,
                    line=dict(color=theme.BG, width=1.2),
                ),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    if centers is not None and len(centers):
        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers",
                name="重心",
                marker=dict(
                    color=theme.TEXT,
                    size=18,
                    symbol="star",
                    line=dict(color=theme.BG, width=2),
                ),
                hovertemplate="重心<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        xaxis=dict(title="特徴量 1", showgrid=False),
        yaxis=dict(title="特徴量 2", showgrid=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if title:
        fig.update_layout(title=title)
    return fig
