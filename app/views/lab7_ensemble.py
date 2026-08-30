"""ラボ 7 — アンサンブル。

木 1 本 → バギング → ランダムフォレスト → 勾配ブースティングと並べ、
弱い学習器を束ねると境界がどう変わるかを見る。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from app.components.cards import Kpi, kpi_row, score_color
from app.components.controls import build_split, dataset_controls, resolution_control
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.models.registry import CLASSIFIERS, ENSEMBLE_STAGES
from mllab.viz import theme
from mllab.viz.boundary import decision_figure

KEY = "lab7"

accent = page_header(
    number=7,
    title="アンサンブルラボ",
    lede=(
        "決定木 1 本の境界はガタガタで、データが少し変わるだけで形が激変します。"
        "ところが同じような木を何本も作って多数決を取ると、境界は驚くほど滑らかで安定します。"
        "木の本数を増やしながら、境界がなめらかになっていく様子を見てください。"
    ),
)

# ---- サイドバー -------------------------------------------------------
cfg = dataset_controls(KEY, default_kind="moons", allow_multiclass=True)

sidebar_section("アンサンブル")
n_estimators = st.sidebar.slider(
    "木の本数", 1, 200, 30, 1, key=f"{KEY}_n_est",
    help="1 から順に増やすと、境界がなめらかになっていくのが分かります。",
)
max_depth = st.sidebar.slider(
    # 既定は浅め。深い木は 1 本でも当たってしまい、束ねる効果が見えなくなる。
    "1 本あたりの深さ", 1, 20, 3, 1, key=f"{KEY}_depth",
    help="浅い木ほど「弱い学習器」。弱くても束ねれば強くなるのがアンサンブルの肝です。"
         "深くすると 1 本でも当たるようになり、束ねる効果は見えにくくなります。",
)
resolution = resolution_control(KEY, default=140)

X_train, X_test, y_train, y_test = build_split(cfg)

tab_stage, tab_grow, tab_imp = st.tabs(
    ["4 段階を並べて比較", "木を増やすとどうなるか", "どの特徴量が効いたか"]
)

# ======================================================================
# タブ 1: 4 段階の比較
# ======================================================================
with tab_stage:

    @st.cache_data(show_spinner="4 手法を学習中…")
    def stages(_cfg, n_estimators, max_depth, resolution, X_tr, y_tr, X_te, y_te):
        out = []
        for key in ENSEMBLE_STAGES:
            spec = CLASSIFIERS[key]
            params = spec.defaults()
            if "n_estimators" in params:
                params["n_estimators"] = n_estimators
            if "max_depth" in params:
                # 勾配ブースティングは浅い木を積むのが前提なので上限を切る
                params["max_depth"] = min(max_depth, 8) if key == "gbdt" else max_depth
            model = spec.create(params)
            model.fit(X_tr, y_tr)
            fig = decision_figure(
                model, X_tr, y_tr, X_te, y_te, resolution=resolution, height=340
            )
            fig.update_layout(showlegend=False, margin=dict(l=28, r=10, t=10, b=28))
            fig.update_xaxes(title=None, showticklabels=False)
            fig.update_yaxes(title=None, showticklabels=False)
            out.append(
                (
                    spec.label,
                    spec.summary,
                    fig,
                    float(accuracy_score(y_tr, model.predict(X_tr))),
                    float(accuracy_score(y_te, model.predict(X_te))),
                )
            )
        return out

    results = stages(
        cfg, int(n_estimators), int(max_depth), resolution,
        X_train, y_train, X_test, y_test,
    )

    best = max(results, key=lambda r: r[4])
    single = results[0]
    kpi_row(
        [
            Kpi("決定木 1 本", f"{single[4]:.1%}", sub="テスト精度の出発点"),
            Kpi("最良の手法", f"{best[4]:.1%}", sub=best[0],
                color=score_color(best[4], good=0.90, bad=0.75)),
            Kpi("束ねた効果", f"{best[4] - single[4]:+.1%}",
                sub="1 本からどれだけ改善したか",
                color=theme.GOOD if best[4] > single[4] else theme.WARN),
            Kpi("木の本数", f"{n_estimators}", sub=f"深さ {max_depth}", color=theme.PURPLE),
        ],
        accent,
    )

    panel("同じデータ・同じ木の本数での比較", "左上から右下へ、束ね方が洗練されていきます")
    cols = st.columns(2)
    for i, (label, summary, fig, tr_acc, te_acc) in enumerate(results):
        with cols[i % 2]:
            st.markdown(
                f"**{label}** — テスト `{te_acc:.1%}` / 訓練 `{tr_acc:.1%}` "
                f"（差 `{tr_acc - te_acc:+.1%}`）"
            )
            st.plotly_chart(fig, width="stretch", key=f"{KEY}_stage_{i}")
            st.caption(summary)

    st.caption(
        "訓練とテストの差に注目してください。決定木 1 本は訓練にぴったり当たるのに"
        "テストで落ちがちですが、束ねるとその差が縮まります。"
        "これがアンサンブルの本当の効きどころです。"
    )

# ======================================================================
# タブ 2: 木の本数を増やす
# ======================================================================
with tab_grow:
    panel("木を 1 本ずつ増やしていく", "何本あたりで頭打ちになるかを見てください")

    @st.cache_data(show_spinner="本数を振って学習中…")
    def growth(_cfg, max_depth, X_tr, y_tr, X_te, y_te):
        counts = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200]
        tr, te = [], []
        for n in counts:
            m = RandomForestClassifier(
                n_estimators=n, max_depth=max_depth, random_state=0, n_jobs=-1
            )
            m.fit(X_tr, y_tr)
            tr.append(accuracy_score(y_tr, m.predict(X_tr)))
            te.append(accuracy_score(y_te, m.predict(X_te)))
        return np.array(counts), np.array(tr), np.array(te)

    counts, tr_curve, te_curve = growth(
        cfg, int(max_depth), X_train, y_train, X_test, y_test
    )

    fig_g = go.Figure()
    fig_g.add_trace(
        go.Scatter(x=counts, y=tr_curve, mode="lines+markers", name="訓練精度",
                   line=dict(color=theme.CYAN, width=2.5))
    )
    fig_g.add_trace(
        go.Scatter(x=counts, y=te_curve, mode="lines+markers", name="テスト精度",
                   line=dict(color=theme.PINK, width=2.5))
    )
    fig_g.add_vline(
        x=n_estimators, line=dict(color=theme.ORANGE, width=2),
        annotation_text="いまの本数", annotation_font=dict(color=theme.ORANGE, size=11),
    )
    fig_g.update_layout(
        height=340,
        xaxis=dict(title="木の本数（対数目盛）", type="log"),
        yaxis=dict(title="精度", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_g, width="stretch")

    plateau = int(counts[np.argmax(te_curve >= te_curve.max() - 0.005)])
    note(
        f"この設定では木を {plateau} 本まで増やせば性能はほぼ頭打ちです — "
        "それ以上増やしても計算時間が延びるだけです",
        tone="good",
    )

    panel("木 1 本 vs 束ねたもの", "同じ深さの木でも、束ねると境界がなめらかになります")

    @st.cache_data(show_spinner="境界を計算中…")
    def single_vs_forest(_cfg, n_estimators, max_depth, resolution, X_tr, y_tr, X_te, y_te):
        figs = []
        for label, model in [
            ("決定木 1 本", DecisionTreeClassifier(max_depth=max_depth, random_state=0)),
            (
                f"ランダムフォレスト {n_estimators} 本",
                RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    random_state=0, n_jobs=-1,
                ),
            ),
        ]:
            model.fit(X_tr, y_tr)
            fig = decision_figure(
                model, X_tr, y_tr, X_te, y_te, resolution=resolution, height=400
            )
            fig.update_layout(showlegend=False, margin=dict(l=28, r=10, t=10, b=28))
            fig.update_xaxes(title=None, showticklabels=False)
            fig.update_yaxes(title=None, showticklabels=False)
            figs.append((label, fig, float(accuracy_score(y_te, model.predict(X_te)))))
        return figs

    pair = single_vs_forest(
        cfg, int(n_estimators), int(max_depth), resolution,
        X_train, y_train, X_test, y_test,
    )
    for col, (label, fig, acc) in zip(st.columns(2), pair):
        with col:
            st.markdown(f"**{label}** — テスト精度 `{acc:.1%}`")
            st.plotly_chart(fig, width="stretch", key=f"{KEY}_pair_{label}")

# ======================================================================
# タブ 3: 特徴量重要度
# ======================================================================
with tab_imp:
    panel(
        "どの特徴量が効いているか",
        "本物の特徴量 2 つに、意味のないノイズ特徴量を混ぜて見分けられるか試します",
    )

    n_noise = st.slider(
        "混ぜるノイズ特徴量の数", 1, 8, 4, 1, key=f"{KEY}_noise_feat",
        help="予測に何の関係もないランダムな列です。重要度が低く出れば正しく見抜けています。",
    )

    @st.cache_data(show_spinner="重要度を計算中…")
    def importances(_cfg, n_noise, n_estimators, max_depth, seed, X_tr, y_tr, X_te, y_te):
        rng = np.random.default_rng(seed)
        Xtr = np.hstack([X_tr, rng.normal(size=(len(X_tr), n_noise))])
        Xte = np.hstack([X_te, rng.normal(size=(len(X_te), n_noise))])
        names = ["特徴量 1（本物）", "特徴量 2（本物）"] + [
            f"ノイズ {i + 1}" for i in range(n_noise)
        ]
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=-1
        )
        model.fit(Xtr, y_tr)
        perm = permutation_importance(
            model, Xte, y_te, n_repeats=10, random_state=seed, n_jobs=-1
        )
        return (
            names,
            model.feature_importances_,
            perm.importances_mean,
            perm.importances_std,
            float(accuracy_score(y_te, model.predict(Xte))),
        )

    names, gini, perm_mean, perm_std, acc_noisy = importances(
        cfg, int(n_noise), int(n_estimators), int(max_depth), cfg.seed,
        X_train, y_train, X_test, y_test,
    )

    real_share = float(np.sum(gini[:2]) / max(np.sum(gini), 1e-12))
    kpi_row(
        [
            Kpi("ノイズ込みのテスト精度", f"{acc_noisy:.1%}",
                sub=f"ノイズ特徴量を {n_noise} 本混ぜた状態"),
            Kpi("本物の特徴量が占める割合", f"{real_share:.1%}",
                sub="重要度のうち本物 2 つに配分された分",
                color=score_color(real_share, good=0.8, bad=0.5)),
            Kpi("最も重要な特徴量", names[int(np.argmax(perm_mean))],
                sub="Permutation Importance で判定", color=theme.PURPLE),
        ],
        accent,
    )

    colors = [theme.LIME, theme.LIME] + [theme.TEXT_MUTED] * n_noise

    col_g, col_p = st.columns(2)
    with col_g:
        st.markdown("**不純度ベースの重要度** — 学習中に自動で得られる、速いが偏りがち")
        fig_gi = go.Figure(
            go.Bar(x=gini, y=names, orientation="h",
                   marker=dict(color=colors, line=dict(width=0)),
                   hovertemplate="%{y}<br>重要度 %{x:.4f}<extra></extra>")
        )
        fig_gi.update_layout(
            height=330, showlegend=False,
            xaxis=dict(title="重要度"), yaxis=dict(autorange="reversed"),
            margin=dict(l=140, r=20, t=16, b=40),
        )
        st.plotly_chart(fig_gi, width="stretch")

    with col_p:
        st.markdown("**Permutation Importance** — 列をシャッフルして精度がどれだけ落ちるかで測る")
        fig_pi = go.Figure(
            go.Bar(x=perm_mean, y=names, orientation="h",
                   error_x=dict(type="data", array=perm_std, color=theme.TEXT_MUTED),
                   marker=dict(color=colors, line=dict(width=0)),
                   hovertemplate="%{y}<br>精度低下 %{x:.4f}<extra></extra>")
        )
        fig_pi.update_layout(
            height=330, showlegend=False,
            xaxis=dict(title="シャッフルによる精度の低下"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=140, r=20, t=16, b=40),
        )
        st.plotly_chart(fig_pi, width="stretch")

    st.caption(
        "ノイズ特徴量（グレー）の重要度が 0 付近に落ちていれば、モデルは正しく見抜けています。"
        "不純度ベースの重要度は「値の種類が多い特徴量」を過大評価する癖があるため、"
        "実務で判断に使うなら Permutation Importance のほうが信頼できます。"
    )

# 解説の数式に、いまの木の本数を差し込む。
explain("ensemble", values={"M": int(n_estimators)})
