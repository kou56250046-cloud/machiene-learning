"""ラボ 5 — 次元削減。

64 次元の手書き数字を 2 次元の地図に落とし、
PCA と t-SNE で「地図の形」がどう変わるかを見る。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from app.components.cards import Kpi, kpi_row, score_color
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.viz import theme

KEY = "lab5"
MAX_SAMPLES = 1000  # t-SNE は重いので上限を切る

accent = page_header(
    number=5,
    title="次元削減ラボ",
    lede=(
        "手書き数字の画像は 8×8 = 64 個の数字の並び、つまり 64 次元のデータです。"
        "人間には見えないその空間を、情報をできるだけ保ったまま 2 次元の地図に描き直すのが次元削減。"
        "PCA と t-SNE で地図の形はまるで変わります。点をクリックすると元の画像が出ます。"
    ),
)

# ---- サイドバー -------------------------------------------------------
sidebar_section("データ")
digits_choice = st.sidebar.multiselect(
    "使う数字", list(range(10)), default=list(range(10)),
    key=f"{KEY}_digits",
    help="数字を絞ると、地図の重なり方が見やすくなります。",
)
if len(digits_choice) < 2:
    digits_choice = list(range(10))
    st.sidebar.caption("2 種類以上選んでください（全数字に戻しました）。")

n_samples = st.sidebar.slider(
    "サンプル数", 200, MAX_SAMPLES, 600, 50, key=f"{KEY}_n",
    help="t-SNE の計算量はサンプル数の 2 乗に近く効きます。",
)
seed = st.sidebar.number_input("乱数シード", 0, 9999, 0, 1, key=f"{KEY}_seed")

sidebar_section("t-SNE")
perplexity = st.sidebar.slider(
    "perplexity", 5.0, 50.0, 30.0, 1.0, key=f"{KEY}_perp",
    help="1 点あたり「何点を近所とみなすか」の目安。地図の形が大きく変わります。",
)
tsne_iter = st.sidebar.select_slider(
    # sklearn は最初の 250 回を探索フェーズに使うので、それより大きい値のみ許す
    "反復回数", options=[500, 750, 1000, 1500], value=750, key=f"{KEY}_iter",
    help="増やすほど配置が落ち着きますが、計算時間も延びます。",
)


# ---- データ -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load(digits: tuple[int, ...], n_samples: int, seed: int):
    data = load_digits()
    mask = np.isin(data.target, list(digits))
    X, y, images = data.data[mask], data.target[mask], data.images[mask]
    if len(X) > n_samples:
        idx = np.random.default_rng(seed).choice(len(X), n_samples, replace=False)
        X, y, images = X[idx], y[idx], images[idx]
    return X, y, images


X, y, images = load(tuple(sorted(digits_choice)), int(n_samples), int(seed))
X_scaled = StandardScaler().fit_transform(X)


@st.cache_data(show_spinner="PCA を計算中…")
def run_pca(X_scaled, seed):
    pca = PCA(n_components=min(20, X_scaled.shape[1]), random_state=seed)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_, pca.components_


@st.cache_data(show_spinner="t-SNE を計算中… 少し時間がかかります")
def run_tsne(X_scaled, perplexity, tsne_iter, seed):
    # perplexity はサンプル数より十分小さい必要がある
    perp = min(perplexity, (len(X_scaled) - 1) / 3.0)
    tsne = TSNE(
        n_components=2, perplexity=perp, max_iter=int(tsne_iter),
        init="pca", random_state=seed,
    )
    return tsne.fit_transform(X_scaled), tsne.kl_divergence_


pca_coords, evr, components = run_pca(X_scaled, int(seed))
tsne_coords, kl = run_tsne(X_scaled, float(perplexity), int(tsne_iter), int(seed))

kpi_row(
    [
        Kpi("元の次元数", "64", sub="8×8 ピクセル"),
        Kpi("サンプル数", f"{len(X)}", sub=f"数字 {len(set(y.tolist()))} 種類"),
        Kpi("PCA 第1+2主成分", f"{evr[:2].sum():.1%}",
            sub="2 次元でどれだけ情報を保てたか",
            color=score_color(evr[:2].sum(), good=0.5, bad=0.2)),
        Kpi("95% に必要な次元数", f"{int(np.searchsorted(np.cumsum(evr), 0.95) + 1)}",
            unit="次元", sub="情報の 95% を残すのに必要な数", color=theme.PURPLE),
        Kpi("t-SNE の KL", f"{kl:.3f}",
            sub="小さいほど元の近さを再現できている"),
    ],
    accent,
)

if evr[:2].sum() < 0.3:
    note(
        f"PCA の 2 次元では元の情報の {evr[:2].sum():.0%} しか保てていません — "
        "PCA の散布図で数字が重なるのはこのためです",
        tone="warn",
    )

# ---- 2 つの地図を並べる -----------------------------------------------
panel("PCA と t-SNE の地図", "点をクリックすると、右下にその元画像が出ます")


def embedding_figure(coords: np.ndarray, labels: np.ndarray) -> go.Figure:
    """2 次元の埋め込みを散布図にする。見出しは Streamlit 側で出す。"""
    fig = go.Figure()
    for d in sorted(set(labels.tolist())):
        mask = labels == d
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=f"数字 {d}",
                customdata=np.where(mask)[0],
                marker=dict(
                    color=theme.class_color(int(d)), size=7, opacity=0.85,
                    line=dict(color=theme.BG, width=0.8),
                ),
                hovertemplate=f"数字 {d}<extra></extra>",
            )
        )
    fig.update_layout(
        height=480,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False,
                   scaleanchor="x", scaleratio=1),
        # 最大 10 クラスあり、横並びの凡例が 2 段に折り返す。
        # 図の見出しは置かず、その分の余白を凡例に回す。
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=10), itemsizing="constant"),
        margin=dict(l=16, r=16, t=64, b=16),
    )
    return fig


col_pca, col_tsne = st.columns(2)
with col_pca:
    st.markdown("**PCA** — 線形。地図の上の距離に意味があります")
    ev_pca = st.plotly_chart(
        embedding_figure(pca_coords[:, :2], y),
        width="stretch", key=f"{KEY}_pca", on_select="rerun", selection_mode="points",
    )
with col_tsne:
    st.markdown(
        f"**t-SNE** — 非線形（perplexity = {perplexity:.0f}）。"
        "島の間の距離には意味がありません"
    )
    ev_tsne = st.plotly_chart(
        embedding_figure(tsne_coords, y),
        width="stretch", key=f"{KEY}_tsne", on_select="rerun", selection_mode="points",
    )


def selected_indices(event) -> list[int]:
    """plotly の選択イベントから、元データの行番号を取り出す。"""
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return []
    out = []
    for p in points:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd is not None:
            out.append(int(cd))
    return out


picked = selected_indices(ev_pca) + selected_indices(ev_tsne)

col_img, col_evr = st.columns([1, 1.6])

with col_img:
    panel("選んだ点の元画像", "地図の上で点をクリックしてください")
    if picked:
        show = picked[:6]
        img_cols = st.columns(min(len(show), 3))
        for j, idx in enumerate(show):
            with img_cols[j % len(img_cols)]:
                fig_i = go.Figure(
                    go.Heatmap(
                        z=images[idx][::-1], colorscale=theme.SEQUENTIAL,
                        showscale=False, hoverinfo="skip",
                    )
                )
                fig_i.update_layout(
                    height=150, margin=dict(l=4, r=4, t=24, b=4),
                    title=dict(text=f"正解: {y[idx]}", font=dict(size=12)),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
                )
                st.plotly_chart(fig_i, width="stretch", key=f"{KEY}_img_{idx}_{j}")
    else:
        st.caption(
            "まだ何も選ばれていません。上の散布図で点をクリック（またはドラッグで囲む）と、"
            "その元画像がここに出ます。境界付近の点を選ぶと、なぜ混ざっているかが分かります。"
        )

with col_evr:
    panel("PCA の寄与率", "主成分をいくつ使えば、元の情報の何割を保てるか")
    cum = np.cumsum(evr)
    fig_e = go.Figure()
    fig_e.add_trace(
        go.Bar(x=np.arange(1, len(evr) + 1), y=evr, name="各主成分の寄与率",
               marker=dict(color=theme.CYAN, line=dict(width=0)))
    )
    fig_e.add_trace(
        go.Scatter(x=np.arange(1, len(cum) + 1), y=cum, name="累積寄与率",
                   yaxis="y2", mode="lines+markers",
                   line=dict(color=theme.ORANGE, width=2.5))
    )
    fig_e.add_hline(
        y=0.95, yref="y2", line=dict(color=theme.LIME, width=1.5, dash="dot"),
        annotation_text="95%", annotation_font=dict(color=theme.LIME, size=11),
    )
    fig_e.update_layout(
        height=340,
        xaxis=dict(title="第何主成分か", dtick=2),
        yaxis=dict(title="寄与率", tickformat=".0%"),
        yaxis2=dict(title="累積", overlaying="y", side="right", tickformat=".0%",
                    range=[0, 1.02], gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_e, width="stretch")

# ---- 主成分そのものを画像で見る ---------------------------------------
panel("主成分は「絵」として見られる", "PCA が見つけた、数字を区別するための特徴パターンです")

n_show = 8
cols = st.columns(n_show)
for i in range(n_show):
    with cols[i]:
        fig_c = go.Figure(
            go.Heatmap(
                z=components[i].reshape(8, 8)[::-1], colorscale=theme.DIVERGING,
                showscale=False, hoverinfo="skip", zmid=0,
            )
        )
        fig_c.update_layout(
            height=130, margin=dict(l=2, r=2, t=22, b=2),
            title=dict(text=f"PC{i + 1} ({evr[i]:.1%})", font=dict(size=11)),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig_c, width="stretch", key=f"{KEY}_pc_{i}")

st.caption(
    "水色とピンクは符号の違いです。第 1 主成分は「全体的に太いか細いか」のような"
    "大まかな違いを、後ろの主成分ほど細かい違いを捉えています。"
)

explain("dimensionality_reduction")
