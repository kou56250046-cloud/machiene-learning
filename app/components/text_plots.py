"""テキストラボの図。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mllab.models import text as TX
from mllab.viz import theme


def frequency_vs_tfidf(
    counts: pd.DataFrame, tfidf: pd.DataFrame, top: int = 20, height: int = 520
) -> go.Figure:
    """単純な頻度と TF-IDF を左右に並べる。

    順位が入れ替わることを見せるのが目的。頻度で上位の語が
    TF-IDF では沈み、その逆も起きる。
    """
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.22,
        subplot_titles=("出現回数が多い語", "TF-IDF が高い語（その文書らしい語）"),
    )
    left = counts.head(top).iloc[::-1]
    right = tfidf.head(top).iloc[::-1]

    fig.add_trace(
        go.Bar(
            x=left["出現回数"], y=left["語"], orientation="h",
            marker=dict(color=theme.CYAN, line=dict(width=0)),
            hovertemplate="%{y}<br>%{x} 回<extra></extra>", showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=right["TF-IDF合計"], y=right["語"], orientation="h",
            marker=dict(color=theme.PINK, line=dict(width=0)),
            hovertemplate="%{y}<br>TF-IDF %{x:.3f}<extra></extra>", showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_layout(height=height, margin=dict(l=90, r=24, t=44, b=44))
    fig.update_annotations(font=dict(size=12, color=theme.TEXT_MUTED))
    return fig


def rank_shift_table(counts: pd.DataFrame, tfidf: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """頻度順位と TF-IDF 順位のずれが大きい語を並べる。"""
    frequency_rank = {w: i + 1 for i, w in enumerate(counts["語"])}
    tfidf_rank = {w: i + 1 for i, w in enumerate(tfidf["語"])}
    shared = set(frequency_rank) & set(tfidf_rank)
    rows = [
        {
            "語": word,
            "頻度順位": frequency_rank[word],
            "TF-IDF順位": tfidf_rank[word],
            "順位の差": frequency_rank[word] - tfidf_rank[word],
        }
        for word in shared
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.reindex(frame["順位の差"].abs().sort_values(ascending=False).index)
        .head(top)
        .reset_index(drop=True)
    )


def cooccurrence_figure(graph: TX.CooccurrenceGraph, height: int = 620) -> go.Figure:
    """語と語のつながりをネットワークで描く。

    線の濃さが結びつきの強さ（Jaccard）、丸の大きさが出現の多さ。
    """
    fig = go.Figure()

    if not graph.edges.empty:
        strongest = float(graph.edges["Jaccard"].max())
        # 線を 1 本ずつ引くと重いので、強さで数段階にまとめる
        for lower, upper, width, alpha in [
            (0.0, 0.34, 0.7, 0.16),
            (0.34, 0.67, 1.6, 0.32),
            (0.67, 1.01, 3.0, 0.6),
        ]:
            xs: list[float | None] = []
            ys: list[float | None] = []
            for row in graph.edges.itertuples():
                strength = row.Jaccard / strongest if strongest else 0.0
                if not (lower <= strength < upper):
                    continue
                start = graph.positions.get(row.語1)
                end = graph.positions.get(row.語2)
                if start is None or end is None:
                    continue
                xs += [start[0], end[0], None]
                ys += [start[1], end[1], None]
            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(color=theme.rgba(theme.CYAN, alpha), width=width),
                    )
                )

    words = [w for w in graph.nodes["語"] if w in graph.positions]
    node_x = [graph.positions[w][0] for w in words]
    node_y = [graph.positions[w][1] for w in words]
    frequency = graph.nodes.set_index("語")["出現文書数"]
    degree = graph.nodes.set_index("語")["つながり数"]
    sizes = frequency.reindex(words).to_numpy(dtype=float)
    spread = float(np.ptp(sizes)) if len(sizes) else 0.0
    scaled = 14 + 30 * (sizes - sizes.min()) / max(spread, 1.0)

    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=words,
            textposition="top center", showlegend=False,
            textfont=dict(size=11, color=theme.TEXT),
            marker=dict(
                size=scaled,
                color=degree.reindex(words).to_numpy(dtype=float),
                colorscale=theme.SEQUENTIAL, showscale=False,
                line=dict(color=theme.BG, width=1.5), opacity=0.9,
            ),
            customdata=np.c_[sizes, degree.reindex(words).to_numpy()],
            hovertemplate="%{text}<br>出現 %{customdata[0]:.0f} 文書<br>"
                          "つながり %{customdata[1]:.0f} 語<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=16, r=16, t=16, b=16),
    )
    return fig


def topic_words_figure(model: TX.TopicModel, top: int = 8, height: int | None = None) -> go.Figure:
    """トピックごとの代表語を並べる。"""
    columns = min(3, model.n_topics)
    rows = int(np.ceil(model.n_topics / columns))
    fig = make_subplots(
        rows=rows, cols=columns,
        subplot_titles=[model.label(t, 2) for t in range(model.n_topics)],
        horizontal_spacing=0.16, vertical_spacing=0.14 / max(rows, 1) + 0.06,
    )
    for topic in range(model.n_topics):
        words = model.top_words(topic, top)[::-1]
        row, column = divmod(topic, columns)
        fig.add_trace(
            go.Bar(
                x=[weight for _, weight in words],
                y=[word for word, _ in words],
                orientation="h", showlegend=False,
                marker=dict(color=theme.class_color(topic), line=dict(width=0)),
                hovertemplate="%{y}<br>重み %{x:.4f}<extra></extra>",
            ),
            row=row + 1, col=column + 1,
        )
    fig.update_layout(
        height=height or (200 * rows + 80),
        margin=dict(l=70, r=24, t=48, b=32),
    )
    fig.update_annotations(font=dict(size=11, color=theme.TEXT_MUTED))
    fig.update_xaxes(showticklabels=False)
    return fig


def topic_share_figure(model: TX.TopicModel, height: int = 300) -> go.Figure:
    """どのトピックがどれだけの文書を占めているか。"""
    dominant = np.argmax(model.document_topics, axis=1)
    counts = [int(np.sum(dominant == t)) for t in range(model.n_topics)]
    fig = go.Figure(
        go.Bar(
            x=[model.label(t, 3) for t in range(model.n_topics)],
            y=counts,
            marker=dict(
                color=[theme.class_color(t) for t in range(model.n_topics)],
                line=dict(width=0),
            ),
            hovertemplate="%{x}<br>%{y} 文書<extra></extra>",
        )
    )
    fig.update_layout(
        height=height, showlegend=False,
        xaxis=dict(title=None, tickangle=-20, tickfont=dict(size=10)),
        yaxis=dict(title="主トピックとする文書数"),
        margin=dict(l=56, r=24, t=16, b=96),
    )
    return fig


def document_map(
    coordinates: np.ndarray,
    labels: np.ndarray | pd.Series,
    hover_texts: list[str],
    legend_title: str = "",
    height: int = 560,
) -> go.Figure:
    """文書を 2 次元に配置した地図。似た文書が近くに来る。"""
    labels = np.asarray(labels)
    fig = go.Figure()
    for i, value in enumerate(sorted(set(labels.tolist()))):
        mask = labels == value
        fig.add_trace(
            go.Scatter(
                x=coordinates[mask, 0], y=coordinates[mask, 1], mode="markers",
                name=str(value),
                marker=dict(
                    color=theme.class_color(i), size=7, opacity=0.85,
                    line=dict(color=theme.BG, width=0.8),
                ),
                customdata=[hover_texts[j] for j in np.where(mask)[0]],
                hovertemplate="%{customdata}<extra>" + str(value) + "</extra>",
            )
        )
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0,
            font=dict(size=10), title=dict(text=legend_title, font=dict(size=10)),
        ),
        margin=dict(l=16, r=16, t=64, b=16),
    )
    return fig


def class_features_figure(
    top_features: dict[str, list[tuple[str, float]]], height: int | None = None
) -> go.Figure:
    """クラスごとに、判断根拠になった語を並べる。"""
    classes = list(top_features)
    columns = min(3, len(classes))
    rows = int(np.ceil(len(classes) / columns))
    fig = make_subplots(
        rows=rows, cols=columns, subplot_titles=classes,
        horizontal_spacing=0.16, vertical_spacing=0.12 / max(rows, 1) + 0.06,
    )
    for i, name in enumerate(classes):
        words = top_features[name][::-1]
        row, column = divmod(i, columns)
        fig.add_trace(
            go.Bar(
                x=[weight for _, weight in words],
                y=[word for word, _ in words],
                orientation="h", showlegend=False,
                marker=dict(color=theme.class_color(i), line=dict(width=0)),
                hovertemplate="%{y}<br>係数 %{x:.3f}<extra></extra>",
            ),
            row=row + 1, col=column + 1,
        )
    fig.update_layout(
        height=height or (210 * rows + 80),
        margin=dict(l=80, r=24, t=48, b=32),
    )
    fig.update_annotations(font=dict(size=11, color=theme.TEXT_MUTED))
    fig.update_xaxes(showticklabels=False)
    return fig


def document_length_figure(lengths: pd.Series, height: int = 280) -> go.Figure:
    """1 文書あたりの語数の分布。"""
    fig = go.Figure(
        go.Histogram(
            x=lengths.to_numpy(), nbinsx=30,
            marker=dict(color=theme.LIME, line=dict(width=0)),
            hovertemplate="%{x} 語<br>%{y} 文書<extra></extra>",
        )
    )
    fig.update_layout(
        height=height, showlegend=False,
        xaxis=dict(title="1 文書あたりの語数（解析対象の語のみ）"),
        yaxis=dict(title="文書数"),
    )
    return fig
