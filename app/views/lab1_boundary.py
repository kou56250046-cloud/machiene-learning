"""ラボ 1 — 決定境界。

同じデータに複数のモデルを当てて、引かれる境界の形の違いを見る。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score

from app.components.cards import Kpi, kpi_row, score_color
from app.components.controls import (
    build_split,
    dataset_controls,
    param_controls,
    resolution_control,
)
from app.components.explain import explain
from app.components.layout import note, page_header, panel
from mllab.models.registry import BOUNDARY_MODELS, CLASSIFIERS
from mllab.viz import theme
from mllab.viz.boundary import decision_figure

KEY = "lab1"

accent = page_header(
    number=1,
    title="決定境界ラボ",
    lede=(
        "分類とは、特徴量の空間を「ここから先はクラス 1」と区切ること。"
        "その区切り線を 決定境界 と呼びます。同じデータでもモデルが違えば境界の形は"
        "まるで変わります。データの形とハイパーパラメータを動かして、"
        "どのモデルがどんな線を引くのかを見比べてください。"
    ),
)

# ---- サイドバー -------------------------------------------------------
cfg = dataset_controls(KEY, default_kind="moons", allow_multiclass=True)

st.sidebar.html('<div class="mllab-side-head">モデル</div>')
model_key = st.sidebar.selectbox(
    "分類モデル",
    BOUNDARY_MODELS,
    format_func=lambda k: CLASSIFIERS[k].label,
    key=f"{KEY}_model",
)
spec = CLASSIFIERS[model_key]
st.sidebar.caption(spec.summary)
params = param_controls(spec, KEY)

resolution = resolution_control(KEY, default=160)
compare_mode = st.sidebar.toggle(
    "全モデルを並べて比較", value=False, key=f"{KEY}_cmp",
    help="7 モデルすべての境界を一覧します（少し時間がかかります）。",
)

# ---- 計算 -------------------------------------------------------------
X_train, X_test, y_train, y_test = build_split(cfg)


@st.cache_data(show_spinner=False)
def fit_and_score(
    model_key: str, params: dict, _cfg, X_tr, y_tr, X_te, y_te
) -> tuple[object, float, float]:
    """モデルを学習し、訓練 / テスト精度を返す。"""
    model = CLASSIFIERS[model_key].create(params)
    model.fit(X_tr, y_tr)
    return (
        model,
        float(accuracy_score(y_tr, model.predict(X_tr))),
        float(accuracy_score(y_te, model.predict(X_te))),
    )


model, train_acc, test_acc = fit_and_score(
    model_key, params, cfg, X_train, y_train, X_test, y_test
)
gap = train_acc - test_acc

# ---- KPI --------------------------------------------------------------
kpi_row(
    [
        Kpi("訓練精度", f"{train_acc:.1%}", sub="学習に使ったデータでの正答率"),
        Kpi(
            "テスト精度", f"{test_acc:.1%}",
            sub="学習に使っていないデータでの正答率",
            color=score_color(test_acc, good=0.90, bad=0.70),
        ),
        Kpi(
            "訓練 − テスト", f"{gap:+.1%}",
            sub="開きが大きいほど過学習気味",
            color=score_color(gap, good=0.03, bad=0.12),
        ),
        Kpi("サンプル数", f"{len(X_train)}／{len(X_test)}", sub="訓練／テスト"),
        Kpi("クラス数", f"{len(np.unique(y_train))}", sub="分類先の種類"),
    ],
    accent,
)

if gap > 0.12:
    note(
        f"訓練精度がテスト精度を {gap:.1%} 上回っています — 過学習のサインです",
        tone="bad",
    )

# ---- メイン図 ---------------------------------------------------------
if not compare_mode:
    panel(
        f"{spec.label} の決定境界",
        "白い線が境界そのもの。背景の色は「どちらのクラスらしいか」の強さです。",
    )

    @st.cache_data(show_spinner="境界を計算中…")
    def boundary_fig(model_key, params, _cfg, resolution, X_tr, y_tr, X_te, y_te):
        m = CLASSIFIERS[model_key].create(params)
        m.fit(X_tr, y_tr)
        return decision_figure(m, X_tr, y_tr, X_te, y_te, resolution=resolution, height=560)

    st.plotly_chart(
        boundary_fig(model_key, params, cfg, resolution, X_train, y_train, X_test, y_test),
        width="stretch",
    )
else:
    panel("7 モデルの境界を並べて比較", "同じデータ・同じ既定パラメータでの違いです")

    @st.cache_data(show_spinner="全モデルの境界を計算中…")
    def all_boundaries(_cfg, resolution, X_tr, y_tr, X_te, y_te):
        out = []
        for k in BOUNDARY_MODELS:
            s = CLASSIFIERS[k]
            m = s.create(s.defaults())
            m.fit(X_tr, y_tr)
            fig = decision_figure(
                m, X_tr, y_tr, X_te, y_te, resolution=resolution, height=340
            )
            fig.update_layout(showlegend=False, margin=dict(l=30, r=10, t=10, b=30))
            fig.update_xaxes(title=None, showticklabels=False)
            fig.update_yaxes(title=None, showticklabels=False)
            out.append((s.label, fig, float(accuracy_score(y_te, m.predict(X_te)))))
        return out

    results = all_boundaries(cfg, min(resolution, 140), X_train, y_train, X_test, y_test)
    for row_start in range(0, len(results), 3):
        for col, (label, fig, acc) in zip(
            st.columns(3), results[row_start : row_start + 3]
        ):
            with col:
                st.markdown(f"**{label}** — テスト精度 `{acc:.1%}`")
                st.plotly_chart(fig, width="stretch", key=f"{KEY}_cmp_{label}")

# ---- 補助図: 境界の複雑さ ---------------------------------------------
panel("ハイパーパラメータを振ったときの精度", "1 つのつまみだけを動かした結果です")

sweep_params = [p for p in spec.params if p.kind in ("int", "log")]
if not sweep_params:
    st.caption("このモデルには連続的に振れるハイパーパラメータがありません。")
else:
    target = st.selectbox(
        "振るパラメータ",
        sweep_params,
        format_func=lambda p: p.label,
        key=f"{KEY}_sweep",
    )

    @st.cache_data(show_spinner="スイープ中…")
    def sweep(model_key, params, target_key, _cfg, X_tr, y_tr, X_te, y_te):
        s = CLASSIFIERS[model_key]
        p = next(q for q in s.params if q.key == target_key)
        if p.kind == "int":
            grid = np.unique(np.linspace(p.min, p.max, 18).astype(int))
        else:
            grid = 10.0 ** np.linspace(p.min, p.max, 18)
        tr, te = [], []
        for v in grid:
            m = s.create({**params, target_key: v})
            m.fit(X_tr, y_tr)
            tr.append(accuracy_score(y_tr, m.predict(X_tr)))
            te.append(accuracy_score(y_te, m.predict(X_te)))
        return grid, np.array(tr), np.array(te), p.kind

    grid, tr, te, kind = sweep(
        model_key, params, target.key, cfg, X_train, y_train, X_test, y_test
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=grid, y=tr, mode="lines+markers", name="訓練精度",
                   line=dict(color=theme.CYAN, width=2))
    )
    fig.add_trace(
        go.Scatter(x=grid, y=te, mode="lines+markers", name="テスト精度",
                   line=dict(color=theme.PINK, width=2))
    )
    best = int(np.argmax(te))
    fig.add_vline(
        x=float(grid[best]), line=dict(color=theme.LIME, width=1.5, dash="dot"),
        annotation_text=f"テスト最良 {grid[best]:.4g}",
        annotation_font=dict(color=theme.LIME, size=11),
    )
    fig.update_layout(
        height=320,
        xaxis=dict(title=target.label, type="log" if kind == "log" else "linear"),
        yaxis=dict(title="精度", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "2 本の線が離れていくところが過学習の始まりです。"
        "テスト精度（ピンク）が頭打ちになる手前が、だいたい良い設定になります。"
    )

explain("decision_boundary")
