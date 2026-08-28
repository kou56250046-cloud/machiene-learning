"""ラボ 4 — クラスタリング。

k-means を 1 ステップずつ進めて重心の動きを見たあと、
他の手法と比べて「どんなクラスタ形状なら取れるのか」を掴む。
"""

from __future__ import annotations

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture

from app.components.cards import Kpi, kpi_row, score_color
from app.components.controls import DataConfig, build_dataset
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.data import toy
from mllab.models.clustering import elbow_curve, kmeans_history
from mllab.viz import theme
from mllab.viz.boundary import cluster_figure

KEY = "lab4"

accent = page_header(
    number=4,
    title="クラスタリングラボ",
    lede=(
        "正解ラベルを一切見ずに、データを似たもの同士でまとめるのがクラスタリングです。"
        "まず k-means を 1 ステップずつ手で進めて、重心が動いて落ち着くまでを目で追ってください。"
        "そのあと他の手法と比べると、k-means が何を苦手にしているかがはっきりします。"
    ),
)

# ---- サイドバー -------------------------------------------------------
sidebar_section("データ")
kinds = list(toy.DATASETS)
kind = st.sidebar.selectbox(
    "データの形", kinds, index=kinds.index("blobs"),
    format_func=lambda k: toy.DATASETS[k], key=f"{KEY}_kind",
    help="moons / circles を選ぶと、k-means の限界がよく分かります。",
)
n_samples = st.sidebar.slider("サンプル数", 50, 1500, 400, 50, key=f"{KEY}_n")
noise = st.sidebar.slider("ノイズ", 0.0, 1.0, 0.15, 0.01, key=f"{KEY}_noise")
true_groups = st.sidebar.slider(
    "本当のグループ数", 2, 5, 3, 1, key=f"{KEY}_tc",
    help="データ生成に使う真のグループ数。blobs 系でのみ効きます。",
)
seed = st.sidebar.number_input("乱数シード", 0, 9999, 0, 1, key=f"{KEY}_seed")

cfg = DataConfig(kind, int(n_samples), float(noise), int(seed), int(true_groups), 0.3)
X, y_true = build_dataset(cfg)

tab_km, tab_cmp, tab_k = st.tabs(
    ["k-means をステップ実行", "手法を比べる", "クラスタ数をどう決めるか"]
)

# ======================================================================
# タブ 1: k-means のステップ実行
# ======================================================================
with tab_km:
    sidebar_section("k-means")
    k = st.sidebar.slider("クラスタ数 k", 2, 8, 3, 1, key=f"{KEY}_k")
    init = st.sidebar.radio(
        "初期化", ["k-means++", "random"], key=f"{KEY}_init",
        help="k-means++ は初期の重心を賢く散らします。random との違いを比べてみてください。",
    )

    @st.cache_data(show_spinner=False)
    def history(_cfg, X, k, init, seed):
        return kmeans_history(X, k, init, seed)

    hist = history(cfg, X, int(k), init, int(seed))

    step = st.sidebar.slider(
        "何回目の反復まで進めるか", 1, len(hist), 1, 1, key=f"{KEY}_step_{k}_{init}",
        help="1 から順に増やすと、重心が動いて落ち着くまでが見られます。",
    )
    snap = hist[step - 1]

    ari = float(adjusted_rand_score(y_true, snap.labels))
    sil = (
        float(silhouette_score(X, snap.labels))
        if len(np.unique(snap.labels)) > 1
        else float("nan")
    )

    kpi_row(
        [
            Kpi("反復", f"{step}", unit=f"／{len(hist)}", sub="収束までの回数"),
            Kpi("クラスタ内平方和", f"{snap.inertia:.1f}",
                sub="小さいほどまとまりが良い（k を増やせば必ず下がる）"),
            Kpi("重心の移動量", f"{snap.shift:.4f}",
                sub="0 に近づいたら収束",
                color=score_color(snap.shift, good=1e-3, bad=0.5)),
            Kpi("シルエット係数", f"{sil:.3f}" if np.isfinite(sil) else "—",
                sub="−1〜1。0.5 以上なら明確に分かれている",
                color=score_color(sil if np.isfinite(sil) else 0, good=0.5, bad=0.25)),
            Kpi("正解との一致 (ARI)", f"{ari:.3f}",
                sub="1 なら真のグループと完全一致",
                color=score_color(ari, good=0.8, bad=0.4)),
        ],
        accent,
    )

    if snap.shift < 1e-6:
        note(f"{step} 回目で収束しました — これ以上重心は動きません", tone="good")
    if kind in ("moons", "circles", "spirals") and ari < 0.5:
        note(
            "k-means は「丸いかたまり」を前提にしているため、"
            "この形状では真のグループを取れません（DBSCAN のタブを見てください）",
            tone="warn",
        )

    col_before, col_after = st.columns(2)
    with col_before:
        panel(f"{step} 回目 — 割り当て", "いまの重心から見て、各点はどのクラスタか")
        fig = cluster_figure(X, snap.labels, snap.centers, height=460)
        st.plotly_chart(fig, width="stretch", key=f"{KEY}_km_before")
    with col_after:
        panel(f"{step} 回目 — 重心の更新", "割り当てた点の平均へ重心が動きます")
        fig2 = cluster_figure(X, snap.labels, snap.moved_centers, height=460)
        # 重心がどこからどこへ動いたかを矢印で示す
        for c0, c1 in zip(snap.centers, snap.moved_centers):
            if np.linalg.norm(c1 - c0) > 1e-9:
                fig2.add_annotation(
                    x=c1[0], y=c1[1], ax=c0[0], ay=c0[1],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2,
                    arrowcolor=theme.TEXT, opacity=0.9,
                )
        st.plotly_chart(fig2, width="stretch", key=f"{KEY}_km_after")

    panel("反復ごとのクラスタ内平方和", "下がりきったところが収束です")
    fig_in = go.Figure(
        go.Scatter(
            x=[h.iteration for h in hist], y=[h.inertia for h in hist],
            mode="lines+markers", line=dict(color=theme.LIME, width=2.5),
            marker=dict(size=8), name="inertia",
        )
    )
    fig_in.add_vline(
        x=step, line=dict(color=theme.ORANGE, width=2),
        annotation_text="いまの反復", annotation_font=dict(color=theme.ORANGE, size=11),
    )
    fig_in.update_layout(
        height=280, showlegend=False,
        xaxis=dict(title="反復回数", dtick=1), yaxis=dict(title="クラスタ内平方和"),
    )
    st.plotly_chart(fig_in, width="stretch")

# ======================================================================
# タブ 2: 手法の比較
# ======================================================================
with tab_cmp:
    panel("4 手法を同じデータに当てる", "同じデータでも、前提が違えば結果はまったく変わります")

    c1, c2, c3 = st.columns(3)
    with c1:
        cmp_k = st.slider("k-means / GMM / 階層のクラスタ数", 2, 8, int(true_groups), 1,
                          key=f"{KEY}_cmpk")
    with c2:
        eps = st.slider("DBSCAN: 近傍の半径 eps", 0.05, 1.5, 0.35, 0.01, key=f"{KEY}_eps",
                        help="この距離以内の点を「近い」とみなします。")
    with c3:
        min_samples = st.slider("DBSCAN: 中心とみなす最小点数", 2, 30, 8, 1,
                                key=f"{KEY}_ms")

    @st.cache_data(show_spinner="クラスタリング中…")
    def compare(_cfg, X, cmp_k, eps, min_samples, seed):
        results = []

        hist = kmeans_history(X, cmp_k, "k-means++", seed)
        results.append(("k-means", hist[-1].labels, hist[-1].moved_centers,
                        "丸いかたまりを仮定。速いが形状の制約が強い。"))

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        results.append(("DBSCAN", db.labels_, None,
                        "密度でつなぐ。任意の形が取れ、外れ値をノイズに落とせる。"))

        Z = linkage(X, method="ward")
        results.append(("階層クラスタリング (Ward)", fcluster(Z, cmp_k, criterion="maxclust") - 1,
                        None, "近いもの同士を順に併合。木構造で粒度を選べる。"))

        gmm = GaussianMixture(n_components=cmp_k, covariance_type="full",
                              random_state=seed).fit(X)
        results.append(("混合ガウス (GMM)", gmm.predict(X), gmm.means_,
                        "楕円のかたまりを仮定。所属を確率で表せる。"))
        return results

    results = compare(cfg, X, int(cmp_k), float(eps), int(min_samples), int(seed))

    cols = st.columns(2)
    for i, (name, labels, centers, desc) in enumerate(results):
        n_found = len(set(labels.tolist()) - {-1})
        n_noise = int(np.sum(labels == -1))
        ari = float(adjusted_rand_score(y_true, labels))
        with cols[i % 2]:
            st.markdown(
                f"**{name}** — クラスタ {n_found} 個 / 正解一致 (ARI) `{ari:.3f}`"
                + (f" / ノイズ {n_noise} 点" if n_noise else "")
            )
            fig = cluster_figure(X, labels, centers, height=360)
            fig.update_layout(showlegend=False, margin=dict(l=28, r=10, t=10, b=28))
            fig.update_xaxes(title=None, showticklabels=False)
            fig.update_yaxes(title=None, showticklabels=False)
            st.plotly_chart(fig, width="stretch", key=f"{KEY}_cmp_{i}")
            st.caption(desc)

    panel("階層クラスタリングの樹形図", "どの高さで横に切るかが、クラスタ数の選択にあたります")

    @st.cache_data(show_spinner="樹形図を作成中…")
    def dendro(_cfg, X, seed):
        # 全点を描くと潰れるので、多い場合は間引く
        sub = X if len(X) <= 120 else X[np.random.default_rng(seed).choice(len(X), 120, replace=False)]
        fig = ff.create_dendrogram(
            sub, linkagefun=lambda d: linkage(d, method="ward"),
            colorscale=theme.ACCENTS * 3,
        )
        fig.update_layout(
            height=340, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.TEXT_MUTED, size=10),
            xaxis=dict(showticklabels=False, title="サンプル（最大 120 点を抽出）",
                       gridcolor=theme.BORDER),
            yaxis=dict(title="併合したときの距離", gridcolor=theme.BORDER),
            margin=dict(l=48, r=24, t=16, b=40),
        )
        return fig

    st.plotly_chart(dendro(cfg, X, int(seed)), width="stretch")

# ======================================================================
# タブ 3: クラスタ数の決め方
# ======================================================================
with tab_k:
    panel("エルボー法", "折れ曲がる（肘の）ところが、増やす価値のある上限の目安です")

    @st.cache_data(show_spinner="k を振って計算中…")
    def elbow(_cfg, X, seed):
        return elbow_curve(X, 10, seed)

    ks, inertias = elbow(cfg, X, int(seed))

    # 「肘」= 両端を結んだ直線から最も離れた点（2 次元の外積 = 行列式で距離を出す）
    p1 = np.array([ks[0], inertias[0]], dtype=float)
    p2 = np.array([ks[-1], inertias[-1]], dtype=float)
    line = p2 - p1
    rel = np.c_[ks, inertias] - p1
    dists = np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / np.linalg.norm(line)
    elbow_k = int(ks[np.argmax(dists)])

    fig_e = go.Figure(
        go.Scatter(x=ks, y=inertias, mode="lines+markers",
                   line=dict(color=theme.LIME, width=2.5), marker=dict(size=9))
    )
    fig_e.add_vline(
        x=elbow_k, line=dict(color=theme.ORANGE, width=2, dash="dot"),
        annotation_text=f"肘: k = {elbow_k}", annotation_font=dict(color=theme.ORANGE, size=12),
    )
    fig_e.update_layout(
        height=320, showlegend=False,
        xaxis=dict(title="クラスタ数 k", dtick=1),
        yaxis=dict(title="クラスタ内平方和"),
    )
    st.plotly_chart(fig_e, width="stretch")

    panel("シルエット図", "各点が「自分のクラスタにどれだけ馴染んでいるか」を並べたもの")

    sil_k = st.slider("シルエットを見るクラスタ数", 2, 8, elbow_k, 1, key=f"{KEY}_silk")

    @st.cache_data(show_spinner="シルエットを計算中…")
    def silhouette(_cfg, X, sil_k, seed):
        labels = kmeans_history(X, sil_k, "k-means++", seed)[-1].labels
        return labels, silhouette_samples(X, labels), float(silhouette_score(X, labels))

    labels, sil_values, sil_avg = silhouette(cfg, X, int(sil_k), int(seed))

    kpi_row(
        [
            Kpi("平均シルエット", f"{sil_avg:.3f}",
                sub="0.5 以上なら明確、0.25 未満なら曖昧",
                color=score_color(sil_avg, good=0.5, bad=0.25)),
            Kpi("エルボーの推奨", f"k = {elbow_k}", sub="平方和の減りが鈍るところ"),
            Kpi("負の値を持つ点", f"{int(np.sum(sil_values < 0))}",
                sub="別のクラスタの方が近い点。多いと分け方が不適切",
                color=score_color(int(np.sum(sil_values < 0)), good=0, bad=len(X) * 0.1)),
        ],
        accent,
    )

    fig_s = go.Figure()
    y_lower = 0
    for c in range(int(sil_k)):
        vals = np.sort(sil_values[labels == c])
        fig_s.add_trace(
            go.Bar(
                x=vals, y=np.arange(y_lower, y_lower + len(vals)),
                orientation="h", name=f"クラスタ {c}",
                marker=dict(color=theme.class_color(c), line=dict(width=0)),
                hovertemplate="シルエット %{x:.3f}<extra></extra>",
            )
        )
        y_lower += len(vals) + 12
    fig_s.add_vline(
        x=sil_avg, line=dict(color=theme.TEXT, width=2, dash="dash"),
        annotation_text=f"平均 {sil_avg:.3f}", annotation_font=dict(color=theme.TEXT, size=11),
    )
    fig_s.update_layout(
        height=420, bargap=0,
        xaxis=dict(title="シルエット係数（1 に近いほど馴染んでいる）"),
        yaxis=dict(title="サンプル（クラスタごとに積み上げ）", showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_s, width="stretch")

explain("clustering")
