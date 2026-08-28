"""ラボ 2 — 過学習と正則化。

多項式回帰の次数と正則化の強さを動かして、
訓練誤差と検証誤差が分かれていく様子を見る。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from app.components.cards import Kpi, kpi_row, score_color
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.data.toy import TRUE_FUNCTIONS, make_regression_1d, true_function
from mllab.viz import theme

KEY = "lab2"
MAX_DEGREE = 15

accent = page_header(
    number=2,
    title="過学習と正則化ラボ",
    lede=(
        "モデルを複雑にすれば、訓練データにはいくらでも当てられます。"
        "問題は、それが 未知のデータでも当たるか どうか。"
        "多項式の次数を上げていき、訓練誤差だけが下がり続けて検証誤差が"
        "跳ね上がる瞬間を捉えてください。正則化はその暴走を抑える手綱です。"
    ),
)

# ---- サイドバー -------------------------------------------------------
sidebar_section("データ")
func_kind = st.sidebar.selectbox(
    "真の関数", list(TRUE_FUNCTIONS), format_func=lambda k: TRUE_FUNCTIONS[k],
    key=f"{KEY}_func",
)
n_samples = st.sidebar.slider("サンプル数", 15, 200, 40, 5, key=f"{KEY}_n",
                              help="少ないほど過学習しやすくなります。")
noise = st.sidebar.slider("ノイズ", 0.0, 0.6, 0.20, 0.01, key=f"{KEY}_noise")
seed = st.sidebar.number_input("乱数シード", 0, 9999, 0, 1, key=f"{KEY}_seed")

sidebar_section("モデル")
degree = st.sidebar.slider(
    "多項式の次数", 1, MAX_DEGREE, 3, 1, key=f"{KEY}_deg",
    help="モデルの複雑さそのもの。上げるほど曲線がぐにゃぐにゃになります。",
)
reg_kind = st.sidebar.radio(
    "正則化", ["none", "ridge", "lasso"],
    format_func=lambda k: {"none": "なし（最小二乗）", "ridge": "Ridge (L2)", "lasso": "Lasso (L1)"}[k],
    key=f"{KEY}_reg", horizontal=False,
)
alpha = 0.0
if reg_kind != "none":
    exponent = st.sidebar.slider(
        "正則化の強さ alpha", -6.0, 1.0, -3.0, 0.1, key=f"{KEY}_alpha",
        format="10^%.1f",
        help="大きいほど係数が抑えられ、曲線がおとなしくなります。",
    )
    alpha = float(10.0**exponent)
    st.sidebar.caption(f"　→ alpha = **{alpha:.3g}**")


# ---- モデル構築 -------------------------------------------------------
def build_model(degree: int, reg_kind: str, alpha: float) -> Pipeline:
    """多項式特徴 → 標準化 → 線形モデル のパイプライン。"""
    if reg_kind == "ridge":
        estimator = Ridge(alpha=alpha)
    elif reg_kind == "lasso":
        estimator = Lasso(alpha=alpha, max_iter=20000)
    else:
        estimator = LinearRegression()
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


@st.cache_data(show_spinner=False)
def load_data(func_kind: str, n_samples: int, noise: float, seed: int):
    X, y = make_regression_1d(func_kind, n_samples, noise, seed)
    return train_test_split(X, y, test_size=0.3, random_state=seed)


X_train, X_test, y_train, y_test = load_data(func_kind, n_samples, noise, seed)


@st.cache_data(show_spinner=False)
def fit_curve(func_kind, n_samples, noise, seed, degree, reg_kind, alpha):
    """モデルを学習し、描画用の曲線・誤差・係数を返す。"""
    X_tr, X_te, y_tr, y_te = load_data(func_kind, n_samples, noise, seed)
    model = build_model(degree, reg_kind, alpha)
    model.fit(X_tr, y_tr)
    grid = np.linspace(-0.05, 1.05, 400).reshape(-1, 1)
    coefs = np.ravel(model.named_steps["model"].coef_)
    return (
        grid.ravel(),
        model.predict(grid),
        float(mean_squared_error(y_tr, model.predict(X_tr))),
        float(mean_squared_error(y_te, model.predict(X_te))),
        coefs,
    )


gx, gy, mse_train, mse_test, coefs = fit_curve(
    func_kind, n_samples, noise, seed, degree, reg_kind, alpha
)
n_zero = int(np.sum(np.abs(coefs) < 1e-6))

# ---- KPI --------------------------------------------------------------
kpi_row(
    [
        Kpi("訓練誤差 (MSE)", f"{mse_train:.4f}", sub="学習に使ったデータでの誤差"),
        Kpi(
            "検証誤差 (MSE)", f"{mse_test:.4f}",
            sub="未知データでの誤差 — 本当に見るべき値",
            color=score_color(mse_test, good=mse_train * 1.5, bad=mse_train * 4 + 0.05),
        ),
        Kpi(
            "検証 ÷ 訓練", f"{(mse_test / max(mse_train, 1e-12)):.1f}", unit="倍",
            sub="1 に近いほど健全、大きいほど過学習",
            color=score_color(mse_test / max(mse_train, 1e-12), good=2.0, bad=6.0),
        ),
        Kpi("係数の大きさ", f"{np.abs(coefs).max():.3g}", sub="最大の係数の絶対値"),
        Kpi(
            "0 になった係数", f"{n_zero}／{len(coefs)}",
            sub="Lasso は不要な項を 0 に潰します",
            color=theme.PURPLE,
        ),
    ],
    accent,
)

if mse_test > mse_train * 5 + 0.02:
    note("検証誤差が訓練誤差を大きく上回っています — 典型的な過学習です", tone="bad")
elif degree >= 8 and reg_kind != "none" and mse_test < mse_train * 2:
    note("高い次数でも正則化が効いて検証誤差が抑えられています", tone="good")

# ---- メイン図: フィット曲線 -------------------------------------------
col_fit, col_coef = st.columns([1.55, 1])

with col_fit:
    panel("当てはめた曲線", "白い破線が「真の関数」。これに近いほど良いモデルです。")
    fig = go.Figure()
    tx = np.linspace(0, 1, 400)
    fig.add_trace(
        go.Scatter(x=tx, y=true_function(func_kind, tx), mode="lines", name="真の関数",
                   line=dict(color=theme.TEXT, width=2, dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=gx, y=gy, mode="lines", name=f"{degree} 次の予測",
                   line=dict(color=theme.ORANGE, width=3))
    )
    fig.add_trace(
        go.Scatter(x=X_train.ravel(), y=y_train, mode="markers", name="訓練データ",
                   marker=dict(color=theme.CYAN, size=9,
                               line=dict(color=theme.BG, width=1)))
    )
    fig.add_trace(
        go.Scatter(x=X_test.ravel(), y=y_test, mode="markers", name="検証データ",
                   marker=dict(color=theme.PINK, size=9, symbol="diamond",
                               line=dict(color=theme.BG, width=1)))
    )
    pad = (np.max(y_train) - np.min(y_train)) * 0.6 + 0.5
    fig.update_layout(
        height=440,
        xaxis=dict(title="x", range=[-0.05, 1.05]),
        yaxis=dict(title="y", range=[np.min(y_train) - pad, np.max(y_train) + pad]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")

with col_coef:
    panel("係数の大きさ", "正則化を強めると、この棒が縮みます")
    order = np.arange(1, len(coefs) + 1)
    fig_c = go.Figure(
        go.Bar(
            x=order, y=np.abs(coefs),
            marker=dict(
                color=np.abs(coefs), colorscale=theme.SEQUENTIAL, showscale=False,
                line=dict(color=theme.BG, width=1),
            ),
            hovertemplate="x^%{x} の係数<br>|w| = %{y:.4g}<extra></extra>",
        )
    )
    fig_c.update_layout(
        height=440, showlegend=False,
        xaxis=dict(title="次数（x の何乗か）", dtick=1),
        yaxis=dict(title="係数の絶対値 |w|", type="log"),
    )
    st.plotly_chart(fig_c, width="stretch")

# ---- 学習曲線: 次数を横軸に -------------------------------------------
panel(
    "次数を 1 から 15 まで振ったときの誤差",
    "訓練誤差は下がり続けるのに、検証誤差は途中から上がる — これが過学習です",
)


@st.cache_data(show_spinner="次数をスイープ中…")
def degree_sweep(func_kind, n_samples, noise, seed, reg_kind, alpha):
    X_tr, X_te, y_tr, y_te = load_data(func_kind, n_samples, noise, seed)
    tr, te, cv = [], [], []
    degrees = np.arange(1, MAX_DEGREE + 1)
    for d in degrees:
        m = build_model(int(d), reg_kind, alpha)
        m.fit(X_tr, y_tr)
        tr.append(mean_squared_error(y_tr, m.predict(X_tr)))
        te.append(mean_squared_error(y_te, m.predict(X_te)))
        # 交差検証（分割の運に左右されない、より信頼できる推定）
        scores = cross_val_score(
            build_model(int(d), reg_kind, alpha), X_tr, y_tr,
            cv=min(5, len(X_tr) // 3), scoring="neg_mean_squared_error",
        )
        cv.append(float(-scores.mean()))
    return degrees, np.array(tr), np.array(te), np.array(cv)


degrees, tr_curve, te_curve, cv_curve = degree_sweep(
    func_kind, n_samples, noise, seed, reg_kind, alpha
)

fig_l = go.Figure()
fig_l.add_trace(
    go.Scatter(x=degrees, y=tr_curve, mode="lines+markers", name="訓練誤差",
               line=dict(color=theme.CYAN, width=2.5))
)
fig_l.add_trace(
    go.Scatter(x=degrees, y=te_curve, mode="lines+markers", name="検証誤差",
               line=dict(color=theme.PINK, width=2.5))
)
fig_l.add_trace(
    go.Scatter(x=degrees, y=cv_curve, mode="lines+markers", name="交差検証誤差",
               line=dict(color=theme.PURPLE, width=2, dash="dot"))
)
best_degree = int(degrees[np.argmin(cv_curve)])
fig_l.add_vline(
    x=best_degree, line=dict(color=theme.LIME, width=1.5, dash="dot"),
    annotation_text=f"交差検証で最良: {best_degree} 次",
    annotation_font=dict(color=theme.LIME, size=11),
)
fig_l.add_vline(
    x=degree, line=dict(color=theme.ORANGE, width=2),
    annotation_text="いまの設定", annotation_position="bottom right",
    annotation_font=dict(color=theme.ORANGE, size=11),
)
fig_l.update_layout(
    height=380,
    xaxis=dict(title="多項式の次数（モデルの複雑さ）", dtick=1),
    yaxis=dict(title="平均二乗誤差 (MSE)", type="log"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig_l, width="stretch")

st.caption(
    f"いまの設定（{TRUE_FUNCTIONS[func_kind]} / ノイズ {noise:.2f} / "
    f"{'正則化なし' if reg_kind == 'none' else f'{reg_kind} alpha={alpha:.3g}'}）では、"
    f"交差検証で最も誤差が小さいのは **{best_degree} 次** です。"
)

explain("overfitting")
