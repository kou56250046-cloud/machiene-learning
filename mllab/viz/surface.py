"""損失曲面と勾配降下の軌跡。

勾配降下法ラボ専用。学習率や手法を変えたときに「軌跡がどう変わるか」を
見せるのが目的なので、最適化器は素朴な NumPy 実装で十分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import plotly.graph_objects as go

from mllab.viz import theme

Point = tuple[float, float]


@dataclass(frozen=True)
class LossSurface:
    """描画・最適化の対象となる 2 変数関数。"""

    key: str
    label: str
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    grad: Callable[[float, float], tuple[float, float]]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    start: Point
    minimum: Point
    log_scale: bool
    note: str
    #: この地形で SGD が発散しない程度の既定学習率
    default_lr: float = 0.05

    def value(self, x: float, y: float) -> float:
        return float(self.func(np.array(x), np.array(y)))


def _bowl(x, y):
    return 0.6 * x**2 + 3.0 * y**2


def _bowl_grad(x, y):
    return 1.2 * x, 6.0 * y


def _rosenbrock(x, y):
    return (1.0 - x) ** 2 + 40.0 * (y - x**2) ** 2


def _rosenbrock_grad(x, y):
    return (
        -2.0 * (1.0 - x) - 160.0 * x * (y - x**2),
        80.0 * (y - x**2),
    )


def _saddle(x, y):
    return x**2 - y**2 + 0.15 * y**4


def _saddle_grad(x, y):
    return 2.0 * x, -2.0 * y + 0.6 * y**3


def _ravine(x, y):
    # 細長い谷。学習率が大きいと谷を挟んで振動する
    return 0.05 * x**2 + 4.0 * y**2


def _ravine_grad(x, y):
    return 0.1 * x, 8.0 * y


def _multi(x, y):
    # 局所解が複数ある地形
    return (
        np.sin(1.2 * x) * np.cos(1.2 * y) * 2.0 + 0.25 * (x**2 + y**2) * 0.5
    )


def _multi_grad(x, y):
    return (
        2.0 * 1.2 * np.cos(1.2 * x) * np.cos(1.2 * y) + 0.25 * x,
        -2.0 * 1.2 * np.sin(1.2 * x) * np.sin(1.2 * y) + 0.25 * y,
    )


SURFACES: dict[str, LossSurface] = {
    "bowl": LossSurface(
        "bowl", "お椀（素直な地形）", _bowl, _bowl_grad,
        (-4, 4), (-2.5, 2.5), (-3.2, 2.0), (0.0, 0.0), False,
        "最も素直な形。どの手法でもまっすぐ最小値へ向かいます。", 0.08,
    ),
    "ravine": LossSurface(
        "ravine", "細長い谷", _ravine, _ravine_grad,
        (-6, 6), (-2, 2), (-5.0, 1.5), (0.0, 0.0), False,
        "縦方向だけ急な地形。SGD は谷を挟んで振動し、なかなか横に進めません。", 0.10,
    ),
    "rosenbrock": LossSurface(
        "rosenbrock", "バナナ谷（Rosenbrock）", _rosenbrock, _rosenbrock_grad,
        (-2, 2), (-1, 3), (-1.6, 2.4), (1.0, 1.0), True,
        "最適化アルゴリズムの定番テスト関数。曲がった谷を辿る必要があります。", 0.002,
    ),
    "saddle": LossSurface(
        "saddle", "鞍点（サドル）", _saddle, _saddle_grad,
        (-2.5, 2.5), (-3, 3), (-1.8, 0.05), (0.0, 1.823), False,
        "原点は最小でも最大でもない鞍点。勾配がほぼ 0 になり停滞します。", 0.05,
    ),
    "multi": LossSurface(
        "multi", "凸凹（局所解あり）", _multi, _multi_grad,
        (-5, 5), (-5, 5), (2.4, 2.4), (-1.2, 0.0), False,
        "谷が複数あり、出発点によって辿り着く場所が変わります。", 0.15,
    ),
}

#: 最適化手法 → 画面表示名
OPTIMIZERS: dict[str, str] = {
    "sgd": "SGD（素の勾配降下）",
    "momentum": "Momentum",
    "rmsprop": "RMSProp",
    "adam": "Adam",
}

OPTIMIZER_NOTES: dict[str, str] = {
    "sgd": "勾配の向きにそのまま進むだけ。単純ですが谷では振動します。",
    "momentum": "前回の移動を慣性として持ち越すので、谷の底を転がるように加速します。",
    "rmsprop": "勾配が大きい方向は歩幅を自動的に縮めるため、振動が収まります。",
    "adam": "Momentum と RMSProp の合わせ技。実務での既定値として最もよく使われます。",
}

#: 発散とみなす損失のしきい値
DIVERGENCE_LIMIT = 1e6


def descend(
    surface: LossSurface,
    optimizer: str = "sgd",
    lr: float = 0.05,
    steps: int = 60,
    start: Point | None = None,
    momentum: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """勾配降下の軌跡を計算する。

    Returns:
        (path, losses, diverged)。path は (steps+1, 2)、losses は (steps+1,)。
        発散した場合はその時点で打ち切り、diverged=True を返す。
    """
    x, y = start if start is not None else surface.start
    v = np.zeros(2)  # Momentum の速度 / Adam の 1 次モーメント
    s = np.zeros(2)  # RMSProp / Adam の 2 次モーメント

    path = [(x, y)]
    losses = [surface.value(x, y)]
    diverged = False

    for t in range(1, int(steps) + 1):
        gx, gy = surface.grad(x, y)
        g = np.array([float(gx), float(gy)])

        if not np.all(np.isfinite(g)):
            diverged = True
            break

        if optimizer == "sgd":
            step = lr * g
        elif optimizer == "momentum":
            v = momentum * v + g
            step = lr * v
        elif optimizer == "rmsprop":
            s = 0.9 * s + 0.1 * g**2
            step = lr * g / (np.sqrt(s) + eps)
        elif optimizer == "adam":
            v = momentum * v + (1 - momentum) * g
            s = beta2 * s + (1 - beta2) * g**2
            v_hat = v / (1 - momentum**t)
            s_hat = s / (1 - beta2**t)
            step = lr * v_hat / (np.sqrt(s_hat) + eps)
        else:
            raise ValueError(f"未知の最適化手法: {optimizer}")

        x, y = float(x - step[0]), float(y - step[1])
        loss = surface.value(x, y)
        path.append((x, y))
        losses.append(loss)

        if not np.isfinite(loss) or abs(loss) > DIVERGENCE_LIMIT:
            diverged = True
            break

    return np.array(path), np.array(losses), diverged


def surface_figure(
    surface: LossSurface,
    path: np.ndarray,
    resolution: int = 140,
    height: int = 520,
) -> go.Figure:
    """等高線マップの上に降下軌跡を重ねる。"""
    # 軌跡が枠外へ飛んでも見えるよう、描画範囲を軌跡に合わせて広げる
    x_lo = min(surface.x_range[0], float(np.nanmin(path[:, 0])) - 0.3)
    x_hi = max(surface.x_range[1], float(np.nanmax(path[:, 0])) + 0.3)
    y_lo = min(surface.y_range[0], float(np.nanmin(path[:, 1])) - 0.3)
    y_hi = max(surface.y_range[1], float(np.nanmax(path[:, 1])) + 0.3)
    # 発散時は無限に広げず、元の範囲の 3 倍までに抑える
    span_x = surface.x_range[1] - surface.x_range[0]
    span_y = surface.y_range[1] - surface.y_range[0]
    x_lo = max(x_lo, surface.x_range[0] - span_x)
    x_hi = min(x_hi, surface.x_range[1] + span_x)
    y_lo = max(y_lo, surface.y_range[0] - span_y)
    y_hi = min(y_hi, surface.y_range[1] + span_y)

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = surface.func(XX, YY)
    if surface.log_scale:
        ZZ = np.log10(ZZ - ZZ.min() + 1.0)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=ZZ,
            colorscale=theme.SEQUENTIAL,
            contours=dict(showlines=True, coloring="heatmap"),
            line=dict(color=theme.rgba(theme.BORDER, 0.6), width=1),
            opacity=0.75,
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    visible = np.isfinite(path).all(axis=1)
    p = path[visible]
    fig.add_trace(
        go.Scatter(
            x=p[:, 0],
            y=p[:, 1],
            mode="lines+markers",
            name="降下の軌跡",
            line=dict(color=theme.TEXT, width=2),
            marker=dict(color=theme.CYAN, size=5, line=dict(color=theme.BG, width=1)),
            hovertemplate="step %{pointNumber}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p[0, 0]], y=[p[0, 1]], mode="markers", name="出発点",
            marker=dict(color=theme.LIME, size=14, symbol="circle",
                        line=dict(color=theme.BG, width=2)),
            hovertemplate="出発点<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p[-1, 0]], y=[p[-1, 1]], mode="markers", name="到達点",
            marker=dict(color=theme.PINK, size=14, symbol="x",
                        line=dict(color=theme.BG, width=1)),
            hovertemplate="到達点<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[surface.minimum[0]], y=[surface.minimum[1]], mode="markers", name="真の最小値",
            marker=dict(color=theme.TEXT, size=16, symbol="star",
                        line=dict(color=theme.BG, width=2)),
            hovertemplate="真の最小値<extra></extra>",
        )
    )

    fig.update_layout(
        height=height,
        xaxis=dict(title="パラメータ w1", range=[x_lo, x_hi], showgrid=False),
        yaxis=dict(title="パラメータ w2", range=[y_lo, y_hi], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def loss_curve_figure(losses: np.ndarray, log_y: bool = True, height: int = 260) -> go.Figure:
    """ステップごとの損失推移。"""
    finite = np.where(np.isfinite(losses), losses, np.nan)
    fig = go.Figure(
        go.Scatter(
            x=np.arange(len(finite)),
            y=finite,
            mode="lines+markers",
            line=dict(color=theme.ORANGE, width=2),
            marker=dict(size=4, color=theme.ORANGE),
            name="損失",
            hovertemplate="step %{x}<br>loss=%{y:.4g}<extra></extra>",
        )
    )
    positive = finite[np.isfinite(finite) & (finite > 0)]
    fig.update_layout(
        height=height,
        showlegend=False,
        xaxis=dict(title="ステップ"),
        yaxis=dict(
            title="損失",
            type="log" if (log_y and len(positive) > 1) else "linear",
        ),
        margin=dict(l=48, r=24, t=16, b=40),
    )
    return fig
