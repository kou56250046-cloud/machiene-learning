"""画像・信号ラボの図。"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mllab.viz import theme

#: 画像はグレースケールで出す。配色を付けると画素値の大小が読み取りにくくなる。
GRAY = [[0.0, "#000000"], [1.0, "#FFFFFF"]]


def image_figure(
    image: np.ndarray,
    title: str = "",
    height: int = 360,
    colorscale=None,
    show_scale: bool = False,
) -> go.Figure:
    """画像 1 枚を描く。カラーなら RGB、グレーなら濃淡で。"""
    if image.ndim == 3:
        fig = go.Figure(
            go.Image(z=np.clip(image * 255, 0, 255).astype(np.uint8), hoverinfo="skip")
        )
    else:
        fig = go.Figure(
            go.Heatmap(
                z=image[::-1],  # Heatmap は下から描くので上下を反転
                colorscale=colorscale or GRAY,
                showscale=show_scale,
                hovertemplate="画素値 %{z:.3f}<extra></extra>",
                colorbar=dict(thickness=8, len=0.7, outlinewidth=0,
                              tickfont=dict(size=9, color=theme.TEXT_MUTED)),
            )
        )
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=32 if title else 8, b=8),
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13)))
    return fig


def pixel_grid_figure(patch: np.ndarray, height: int = 380) -> go.Figure:
    """小さな範囲の画素値を、数字を書き込んだ格子で見せる。

    「画像は数値の行列でしかない」ことを納得するための図。
    """
    text = np.array([[f"{v:.2f}" for v in row] for row in patch])
    fig = go.Figure(
        go.Heatmap(
            z=patch[::-1],
            text=text[::-1],
            texttemplate="%{text}",
            colorscale=GRAY,
            showscale=False,
            zmin=0, zmax=1,
            textfont=dict(size=9, color=theme.CYAN),
            hovertemplate="画素値 %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def channel_figure(image: np.ndarray, height: int = 260) -> go.Figure:
    """カラー画像を R / G / B の 3 枚に分けて見せる。"""
    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.03,
        subplot_titles=("赤 (R)", "緑 (G)", "青 (B)"),
    )
    for index, color in enumerate([theme.PINK, theme.LIME, theme.CYAN]):
        fig.add_trace(
            go.Heatmap(
                z=image[::-1, :, index],
                colorscale=[[0.0, "#000000"], [1.0, color]],
                showscale=False, zmin=0, zmax=1,
                hovertemplate="%{z:.3f}<extra></extra>",
            ),
            row=1, col=index + 1,
        )
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=32, b=8))
    fig.update_annotations(font=dict(size=11, color=theme.TEXT_MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


def histogram_figure(image: np.ndarray, height: int = 240) -> go.Figure:
    """画素値の分布。明るさの偏りが見える。"""
    values = image.ravel() if image.ndim == 2 else image.mean(axis=2).ravel()
    fig = go.Figure(
        go.Histogram(
            x=values, nbinsx=64,
            marker=dict(color=theme.CYAN, line=dict(width=0)),
            hovertemplate="画素値 %{x:.2f}<br>%{y} 画素<extra></extra>",
        )
    )
    fig.update_layout(
        height=height, showlegend=False,
        xaxis=dict(title="画素値（0 = 黒、1 = 白）"),
        yaxis=dict(title="画素数"),
        margin=dict(l=56, r=16, t=16, b=44),
    )
    return fig


def kernel_figure(kernel: np.ndarray, height: int = 240) -> go.Figure:
    """カーネルの中身を数字つきで見せる。"""
    text = np.array([[f"{v:g}" for v in row] for row in kernel])
    limit = float(np.max(np.abs(kernel))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=kernel[::-1],
            text=text[::-1],
            texttemplate="%{text}",
            colorscale=theme.DIVERGING,
            zmid=0, zmin=-limit, zmax=limit,
            showscale=False,
            textfont=dict(size=13, color=theme.TEXT),
            hovertemplate="%{z:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def before_after_figure(
    before: np.ndarray, after: np.ndarray, labels: tuple[str, str], height: int = 340
) -> go.Figure:
    """処理の前後を左右に並べる。"""
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.04, subplot_titles=labels
    )
    for index, image in enumerate([before, after]):
        if image.ndim == 3:
            trace = go.Image(z=np.clip(image * 255, 0, 255).astype(np.uint8), hoverinfo="skip")
        else:
            trace = go.Heatmap(
                z=image[::-1], colorscale=GRAY, showscale=False,
                hovertemplate="%{z:.3f}<extra></extra>",
            )
        fig.add_trace(trace, row=1, col=index + 1)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=34, b=8))
    fig.update_annotations(font=dict(size=12, color=theme.TEXT_MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return fig


# ======================================================================
# 信号
# ======================================================================


def waveform_figure(
    time: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    height: int = 300,
    x_label: str = "時間 (秒)",
) -> go.Figure:
    """波形を重ねて描く。series は (名前, 値, 色) の並び。"""
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(
            go.Scatter(
                x=time, y=values, mode="lines", name=name,
                line=dict(color=color, width=1.6),
                hovertemplate="%{x:.3f} 秒<br>%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        xaxis=dict(title=x_label),
        yaxis=dict(title="振幅"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def spectrum_figure(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    peaks: list[tuple[float, float]] | None = None,
    height: int = 320,
    max_hz: float | None = None,
) -> go.Figure:
    """周波数ごとの強さ。山が立っているところが含まれている成分。"""
    mask = frequencies <= max_hz if max_hz else np.ones_like(frequencies, dtype=bool)
    fig = go.Figure(
        go.Scatter(
            x=frequencies[mask], y=amplitudes[mask], mode="lines",
            name="振幅", line=dict(color=theme.CYAN, width=1.8),
            fill="tozeroy", fillcolor=theme.rgba(theme.CYAN, 0.16),
            hovertemplate="%{x:.1f} Hz<br>振幅 %{y:.3f}<extra></extra>",
        )
    )
    if peaks:
        fig.add_trace(
            go.Scatter(
                x=[f for f, _ in peaks], y=[a for _, a in peaks],
                mode="markers+text", name="検出した山",
                text=[f"{f:.0f}Hz" for f, _ in peaks],
                textposition="top center",
                textfont=dict(size=10, color=theme.PINK),
                marker=dict(color=theme.PINK, size=10, symbol="triangle-down",
                            line=dict(color=theme.BG, width=1)),
                hovertemplate="%{x:.1f} Hz<br>振幅 %{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        xaxis=dict(title="周波数 (Hz)"),
        yaxis=dict(title="振幅"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def spectrogram_figure(
    frequencies: np.ndarray, times: np.ndarray, decibels: np.ndarray, height: int = 380
) -> go.Figure:
    """時間 × 周波数の地図。いつどの高さの成分が出たかが見える。"""
    fig = go.Figure(
        go.Heatmap(
            x=times, y=frequencies, z=decibels,
            colorscale=theme.SEQUENTIAL,
            colorbar=dict(
                title=dict(text="強さ<br>(dB)", font=dict(size=10)),
                thickness=9, len=0.7, outlinewidth=0,
                tickfont=dict(size=9, color=theme.TEXT_MUTED),
            ),
            hovertemplate="%{x:.2f} 秒<br>%{y:.1f} Hz<br>%{z:.1f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        xaxis=dict(title="時間 (秒)"),
        yaxis=dict(title="周波数 (Hz)"),
        margin=dict(l=56, r=16, t=16, b=44),
    )
    return fig


def aliasing_figure(
    dense_time: np.ndarray,
    dense_values: np.ndarray,
    sample_time: np.ndarray,
    sample_values: np.ndarray,
    apparent_hz: float,
    height: int = 320,
) -> go.Figure:
    """標本化が粗いと波が別物に見えることを示す。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dense_time, y=dense_values, mode="lines", name="本当の波",
            line=dict(color=theme.rgba(theme.TEXT_MUTED, 0.7), width=1.4),
            hovertemplate="%{x:.3f} 秒<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_time, y=sample_values, mode="lines+markers",
            name=f"観測した点（見かけ {apparent_hz:.1f} Hz）",
            line=dict(color=theme.PINK, width=2.2),
            marker=dict(color=theme.PINK, size=8, line=dict(color=theme.BG, width=1)),
            hovertemplate="%{x:.3f} 秒<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        xaxis=dict(title="時間 (秒)"),
        yaxis=dict(title="振幅"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
