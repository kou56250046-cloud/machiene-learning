"""ラボ 6 — 評価指標。

閾値とクラス不均衡を動かして、混同行列・ROC・PR 曲線が連動して動くのを見る。
「精度 98%」がなぜ嘘になりうるかを体感するのが目的。
"""

from __future__ import annotations

import warnings

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.components.cards import Kpi, kpi_row, score_color
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.viz import theme

KEY = "lab6"

accent = page_header(
    number=6,
    title="評価指標ラボ",
    lede=(
        "「精度 98%」と聞くと良いモデルに思えます。でも 100 人に 2 人しか陽性がいない検査なら、"
        "全員に「陰性」と答えるだけで 98% に届いてしまう。"
        "閾値と不均衡のスライダを動かして、どの指標が何を見ているのかを確かめてください。"
    ),
)

# ---- サイドバー -------------------------------------------------------
sidebar_section("データ")
n_samples = st.sidebar.slider("サンプル数", 200, 3000, 1000, 100, key=f"{KEY}_n")
positive_rate = st.sidebar.slider(
    "陽性クラスの割合", 0.02, 0.50, 0.50, 0.01, key=f"{KEY}_pos",
    help="下げるほど不均衡になります。0.02 まで下げると accuracy が無意味になるのが分かります。",
)
separability = st.sidebar.slider(
    "クラスの分けやすさ", 0.3, 3.0, 1.2, 0.1, key=f"{KEY}_sep",
    help="大きいほど 2 クラスがきれいに分かれ、モデルが当てやすくなります。",
)
seed = st.sidebar.number_input("乱数シード", 0, 9999, 0, 1, key=f"{KEY}_seed")

sidebar_section("判定")
threshold = st.sidebar.slider(
    "陽性と判定する確率の閾値", 0.01, 0.99, 0.50, 0.01, key=f"{KEY}_th",
    help="モデルの出力確率がこの値以上なら「陽性」と判定します。ここが全ての指標を左右します。",
)


# ---- データとモデル ---------------------------------------------------
@st.cache_data(show_spinner=False)
def build(n_samples, positive_rate, separability, seed):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[1 - positive_rate, positive_rate],
        class_sep=separability,
        flip_y=0.02,
        random_state=seed,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.35, random_state=seed, stratify=y
    )
    model = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
    )
    model.fit(X_tr, y_tr)
    return y_te, model.predict_proba(X_te)[:, 1]


y_true, y_score = build(
    int(n_samples), float(positive_rate), float(separability), int(seed)
)

y_pred = (y_score >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
specificity = tn / max(tn + fp, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-12)
balanced = (recall + specificity) / 2
roc_auc = float(roc_auc_score(y_true, y_score))
ap = float(average_precision_score(y_true, y_score))
base_rate = float(np.mean(y_true))
majority_acc = max(base_rate, 1 - base_rate)

# ---- KPI --------------------------------------------------------------
kpi_row(
    [
        Kpi("正解率 (Accuracy)", f"{accuracy:.1%}",
            sub=f"全部「陰性」と答えるだけで {majority_acc:.1%}",
            color=theme.BAD if accuracy <= majority_acc + 0.01 else theme.GOOD),
        Kpi("適合率 (Precision)", f"{precision:.1%}",
            sub="陽性と言ったうち、本当に陽性だった割合",
            color=score_color(precision, good=0.8, bad=0.5)),
        Kpi("再現率 (Recall)", f"{recall:.1%}",
            sub="本当の陽性のうち、拾えた割合",
            color=score_color(recall, good=0.8, bad=0.5)),
        Kpi("F1 スコア", f"{f1:.3f}", sub="適合率と再現率の調和平均",
            color=score_color(f1, good=0.8, bad=0.5)),
        Kpi("ROC-AUC", f"{roc_auc:.3f}",
            sub="閾値に依存しない総合力。0.5 = でたらめ",
            color=score_color(roc_auc, good=0.85, bad=0.65)),
        Kpi("PR-AUC", f"{ap:.3f}",
            sub=f"不均衡時の指標。ベースラインは {base_rate:.3f}",
            color=theme.PURPLE),
    ],
    accent,
)

if accuracy <= majority_acc + 0.01:
    note(
        f"正解率 {accuracy:.1%} は「全員を多数派クラスと答える」だけで出せる "
        f"{majority_acc:.1%} を上回っていません — このモデルは実質何も当てていません",
        tone="bad",
    )
elif base_rate < 0.15:
    note(
        f"陽性はわずか {base_rate:.1%}。この不均衡では正解率ではなく "
        "PR-AUC・再現率・F1 を見るべきです",
        tone="warn",
    )

# ---- 混同行列 + ROC/PR ------------------------------------------------
col_cm, col_curve = st.columns([1, 1.35])

with col_cm:
    panel("混同行列", "閾値スライダを動かすと、この 4 つの数が入れ替わります")
    z = np.array([[tn, fp], [fn, tp]])
    text = np.array(
        [
            [f"真陰性<br><b>{tn}</b>", f"偽陽性<br><b>{fp}</b>"],
            [f"偽陰性<br><b>{fn}</b>", f"真陽性<br><b>{tp}</b>"],
        ]
    )
    fig_cm = go.Figure(
        go.Heatmap(
            z=z, text=text, texttemplate="%{text}",
            x=["予測: 陰性", "予測: 陽性"], y=["実際: 陰性", "実際: 陽性"],
            colorscale=theme.SEQUENTIAL, showscale=False,
            textfont=dict(size=15, color=theme.TEXT), hoverinfo="skip",
        )
    )
    fig_cm.update_layout(
        height=330, xaxis=dict(side="top", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
        margin=dict(l=90, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_cm, width="stretch")

    st.markdown(
        f"""
| 指標 | 値 | 意味 |
|---|---|---|
| 適合率 | `{precision:.3f}` | 陽性判定 {tp + fp} 件中 {tp} 件が的中 |
| 再現率 | `{recall:.3f}` | 真の陽性 {tp + fn} 件中 {tp} 件を捕捉 |
| 特異度 | `{specificity:.3f}` | 真の陰性 {tn + fp} 件中 {tn} 件を正しく除外 |
| 平衡精度 | `{balanced:.3f}` | 再現率と特異度の平均（不均衡に強い） |
"""
    )

with col_curve:
    panel("ROC 曲線と PR 曲線", "★ が、いま選んでいる閾値の位置です")

    fpr, tpr, roc_th = roc_curve(y_true, y_score)
    prec_c, rec_c, pr_th = precision_recall_curve(y_true, y_score)

    tab_roc, tab_pr = st.tabs(["ROC 曲線", "PR 曲線"])

    with tab_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="でたらめ (AUC=0.5)",
                       line=dict(color=theme.TEXT_MUTED, width=1.5, dash="dash"))
        )
        fig_roc.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})",
                       line=dict(color=theme.CYAN, width=3),
                       fill="tozeroy", fillcolor=theme.rgba(theme.CYAN, 0.12))
        )
        fig_roc.add_trace(
            go.Scatter(x=[1 - specificity], y=[recall], mode="markers",
                       name=f"閾値 {threshold:.2f}",
                       marker=dict(color=theme.ORANGE, size=18, symbol="star",
                                   line=dict(color=theme.BG, width=2)))
        )
        fig_roc.update_layout(
            height=400,
            xaxis=dict(title="偽陽性率 (1 − 特異度)", range=[-0.02, 1.02]),
            yaxis=dict(title="真陽性率 (再現率)", range=[-0.02, 1.02],
                       scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_roc, width="stretch")
        st.caption(
            "左上に張り付くほど良いモデルです。ROC-AUC は閾値を変えても値が変わらないので、"
            "モデル自体の実力を比べるのに向いています。"
        )

    with tab_pr:
        fig_pr = go.Figure()
        fig_pr.add_hline(
            y=base_rate, line=dict(color=theme.TEXT_MUTED, width=1.5, dash="dash"),
            annotation_text=f"ベースライン {base_rate:.3f}",
            annotation_font=dict(color=theme.TEXT_MUTED, size=11),
        )
        fig_pr.add_trace(
            go.Scatter(x=rec_c, y=prec_c, mode="lines", name=f"PR (AP={ap:.3f})",
                       line=dict(color=theme.PINK, width=3),
                       fill="tozeroy", fillcolor=theme.rgba(theme.PINK, 0.12))
        )
        fig_pr.add_trace(
            go.Scatter(x=[recall], y=[precision], mode="markers",
                       name=f"閾値 {threshold:.2f}",
                       marker=dict(color=theme.ORANGE, size=18, symbol="star",
                                   line=dict(color=theme.BG, width=2)))
        )
        fig_pr.update_layout(
            height=400,
            xaxis=dict(title="再現率", range=[-0.02, 1.02]),
            yaxis=dict(title="適合率", range=[-0.02, 1.02]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig_pr, width="stretch")
        st.caption(
            "陽性が少ないときは、ROC より PR 曲線のほうが厳しく・正直に実力を映します。"
            "破線（陽性の割合）がでたらめなモデルの水準です。"
        )

# ---- 閾値を横軸にした指標の推移 ---------------------------------------
panel("閾値を動かすと各指標はどう動くか", "1 つの閾値ですべてを最良にはできません")


@st.cache_data(show_spinner=False)
def threshold_sweep(y_true, y_score):
    ths = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in ths:
        pred = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm.ravel()
        p = tp_ / max(tp_ + fp_, 1)
        r = tp_ / max(tp_ + fn_, 1)
        rows.append(
            [
                (tp_ + tn_) / len(y_true),
                p,
                r,
                2 * p * r / max(p + r, 1e-12),
            ]
        )
    return ths, np.array(rows)


ths, curves = threshold_sweep(y_true, y_score)
labels = ["正解率", "適合率", "再現率", "F1"]
colors = [theme.TEXT_MUTED, theme.CYAN, theme.PINK, theme.LIME]

fig_t = go.Figure()
for i, (lab, col) in enumerate(zip(labels, colors)):
    fig_t.add_trace(
        go.Scatter(x=ths, y=curves[:, i], mode="lines", name=lab,
                   line=dict(color=col, width=2.5))
    )
best_f1_th = float(ths[np.argmax(curves[:, 3])])
fig_t.add_vline(
    x=best_f1_th, line=dict(color=theme.LIME, width=1.5, dash="dot"),
    annotation_text=f"F1 最良 {best_f1_th:.2f}",
    annotation_font=dict(color=theme.LIME, size=11),
)
fig_t.add_vline(
    x=threshold, line=dict(color=theme.ORANGE, width=2),
    annotation_text="いまの閾値", annotation_position="bottom right",
    annotation_font=dict(color=theme.ORANGE, size=11),
)
fig_t.update_layout(
    height=360,
    xaxis=dict(title="陽性と判定する確率の閾値"),
    yaxis=dict(title="指標の値", range=[0, 1.02], tickformat=".0%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig_t, width="stretch")
st.caption(
    "閾値を下げると再現率は上がりますが適合率は下がります（取りこぼしは減るが誤報が増える）。"
    f"F1 が最大になるのは閾値 {best_f1_th:.2f} 付近です。"
    "ただし本当の最適値は、見逃しと誤報どちらの損失が大きいかという業務上の判断で決まります。"
)

# ---- 交差検証の分割可視化 ---------------------------------------------
panel("交差検証の分け方", "1 回のホールドアウトでは、たまたま簡単な分割を引く危険があります")

col_cv1, col_cv2 = st.columns([1, 3])
with col_cv1:
    n_splits = st.slider("分割数", 2, 8, 5, 1, key=f"{KEY}_folds")
    cv_n = 60

splitters = {
    "KFold": KFold(n_splits=n_splits),
    "StratifiedKFold": StratifiedKFold(n_splits=n_splits),
    "TimeSeriesSplit": TimeSeriesSplit(n_splits=n_splits),
}

y_small = y_true[:cv_n]
X_small = np.zeros((cv_n, 1))

# 陽性が分割数より少ないと StratifiedKFold は各分割に陽性を配れない。
# これは不均衡データで実際に起きる問題なので、警告を握り潰さず画面で伝える。
n_positive = int(np.sum(y_small))
if n_positive < n_splits:
    note(
        f"抜粋した {cv_n} 件のうち陽性はわずか {n_positive} 件 — "
        f"{n_splits} 分割すると陽性が 1 件も入らない分割ができてしまいます。"
        "不均衡データで層化が必要になる、まさにその状況です。",
        tone="warn",
    )

cols_cv = st.columns(3)
for (name, splitter), col in zip(splitters.items(), cols_cv):
    fig_cv = go.Figure()
    with warnings.catch_warnings():
        # 上でユーザーに伝えたので、ターミナルへの重複出力は抑える
        warnings.simplefilter("ignore", UserWarning)
        folds = list(splitter.split(X_small, y_small))
    for i, (tr_idx, te_idx) in enumerate(folds):
        band = np.full(cv_n, np.nan)
        band[tr_idx] = 0
        band[te_idx] = 1
        fig_cv.add_trace(
            go.Heatmap(
                z=[band], y=[f"分割 {i + 1}"], x=np.arange(cv_n),
                colorscale=[[0.0, theme.rgba(theme.CYAN, 0.55)], [1.0, theme.PINK]],
                showscale=False, xgap=0.5, ygap=3, hoverinfo="skip",
            )
        )
    # 一番下に真のクラスの並びを添える
    fig_cv.add_trace(
        go.Heatmap(
            z=[y_small.astype(float)], y=["実際のクラス"], x=np.arange(cv_n),
            colorscale=[[0.0, theme.rgba(theme.TEXT_MUTED, 0.4)], [1.0, theme.ORANGE]],
            showscale=False, xgap=0.5, ygap=3, hoverinfo="skip",
        )
    )
    fig_cv.update_layout(
        height=230, title=dict(text=name, font=dict(size=13)),
        xaxis=dict(title="サンプル（60 件を抜粋）", showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        margin=dict(l=90, r=16, t=40, b=36),
    )
    with col:
        st.plotly_chart(fig_cv, width="stretch", key=f"{KEY}_cv_{name}")

st.caption(
    "水色が訓練、ピンクが検証に回された部分です。"
    "**KFold** は素直に等分するだけ、**StratifiedKFold** は各分割のクラス比を元と揃えます"
    "（不均衡データではこちらが必須）。"
    "**TimeSeriesSplit** は常に過去で学習し未来で検証するので、時系列で未来の情報が漏れるのを防ぎます。"
)

explain("metrics")
