"""ラボ 8 — テーブルデータ。

カタログに溜めた実データ（または同梱データセット）を使い、
前処理 → 学習 → 評価 → 解釈 の一通りを回す。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from app.components import experiment_log as EL
from app.components import tabular_plots as P
from app.components.cards import Kpi, kpi_row, score_color
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.data import store
from mllab.data.connectors import opendata
from mllab.models import tabular as T
from mllab.viz import theme

KEY = "lab8"

accent = page_header(
    number=8,
    title="テーブルデータラボ",
    lede=(
        "ここからは実データです。列の型はばらばら、欠測があり、カテゴリが混ざっている。"
        "実務で時間を取られるのはモデル選びではなく、この前処理の部分です。"
        "欠測の埋め方を変えるとスコアがどう動くか、そしてモデルが何を根拠にしているかを確かめてください。"
    ),
)


# ======================================================================
# データの選択
# ======================================================================


@st.cache_data(show_spinner=False)
def _load_bundled(key: str) -> pd.DataFrame | None:
    result = opendata.fetch(dataset=key)
    return result.frame if result.ok else None


def _table_sources() -> dict[str, tuple[str, str]]:
    """選べるデータ源。カタログ保存済みを先に、同梱データセットを後に並べる。"""
    sources: dict[str, tuple[str, str]] = {}
    for dataset in store.list_datasets():
        if dataset.domain == "table":
            sources[f"catalog:{dataset.name}"] = (
                f"{dataset.name} — {dataset.label}（カタログ / {dataset.rows:,} 行）",
                "catalog",
            )
    for key, (label, _, task, _) in opendata.DATASETS.items():
        sources[f"bundled:{key}"] = (f"{label} — {task}（同梱）", "bundled")
    return sources


sidebar_section("データ")
sources = _table_sources()
source_key = st.sidebar.selectbox(
    "データセット",
    list(sources),
    format_func=lambda k: sources[k][0],
    key=f"{KEY}_source",
    help="カタログで取り込んだテーブル系データと、ライブラリ同梱のデータから選べます。",
)

kind, _, name = source_key.partition(":")
if kind == "catalog":
    raw = store.load(name)
    source_label = f"カタログ / {name}"
else:
    raw = _load_bundled(name)
    source_label = f"同梱 / {opendata.DATASETS[name][0]}"

if raw is None or raw.empty:
    st.error("データを読み込めませんでした。別のデータセットを選んでください。")
    st.stop()

# 行が多いと交差検証と SHAP が重くなるので、体験用に上限を設ける
MAX_ROWS = 8000
truncated = len(raw) > MAX_ROWS
if truncated:
    raw = raw.sample(MAX_ROWS, random_state=0).reset_index(drop=True)

targets = T.suggest_targets(raw)
target = st.sidebar.selectbox(
    "目的変数（当てたい列）",
    targets,
    key=f"{KEY}_target_{source_key}",
    help="ID や日付のように目的変数になりえない列は、候補の後ろへ回してあります。",
)

y = raw[target]
task = T.infer_task(y)
numeric_cols, categorical_cols = T.split_columns(raw, target)

if not numeric_cols and not categorical_cols:
    st.error(f"`{target}` 以外に使える特徴量がありません。別の列を選んでください。")
    st.stop()

st.sidebar.caption(
    f"→ **{T.TASK_LABELS[task]}** と判定（目的変数のユニーク数 {y.nunique():,}）"
)

# ======================================================================
# 前処理の設定
# ======================================================================

sidebar_section("前処理")

missing_rate = st.sidebar.slider(
    "人工的に欠測を作る", 0.0, 0.6, 0.0, 0.05, key=f"{KEY}_missing",
    help="同梱データには欠測がありません。ここを上げると補完の効きめを体感できます。",
)
numeric_impute = st.sidebar.selectbox(
    "数値列の欠測", list(T.NUMERIC_IMPUTE_LABELS),
    format_func=lambda k: T.NUMERIC_IMPUTE_LABELS[k],
    key=f"{KEY}_numimp",
)
categorical_impute = st.sidebar.selectbox(
    "カテゴリ列の欠測", list(T.CATEGORICAL_IMPUTE_LABELS),
    format_func=lambda k: T.CATEGORICAL_IMPUTE_LABELS[k],
    key=f"{KEY}_catimp",
    disabled=not categorical_cols,
)
encode = st.sidebar.selectbox(
    "カテゴリの符号化", list(T.ENCODE_LABELS),
    format_func=lambda k: T.ENCODE_LABELS[k],
    key=f"{KEY}_encode",
    disabled=not categorical_cols,
)

sidebar_section("学習")
model_key = st.sidebar.selectbox(
    "モデル", list(T.MODELS),
    format_func=lambda k: T.MODELS[k].label,
    key=f"{KEY}_model",
)
st.sidebar.caption(T.MODELS[model_key].summary)

n_splits = st.sidebar.slider("交差検証の分割数", 2, 10, 5, 1, key=f"{KEY}_folds")
seed = st.sidebar.number_input("乱数シード", 0, 9999, 0, 1, key=f"{KEY}_seed")

config = T.PreprocessConfig(
    numeric_impute=numeric_impute,
    categorical_impute=categorical_impute,
    scale=True,
    encode=encode,
)

# 欠測の注入は数値列のみ（カテゴリに NaN を入れると型が壊れやすいため）
frame = T.inject_missing(raw, numeric_cols, missing_rate, int(seed))
X = frame.drop(columns=[target])
y = frame[target]

if config.numeric_impute == "drop_rows":
    before = len(X)
    keep = X[numeric_cols].notna().all(axis=1) if numeric_cols else pd.Series(True, index=X.index)
    X, y = X[keep], y[keep]
    if len(X) < 20:
        st.error(
            f"欠測のある行を捨てた結果、{len(X)} 行しか残りませんでした。"
            "補完方法を変えるか、欠測率を下げてください。"
        )
        st.stop()
    if len(X) < before:
        note(
            f"欠測のある行を捨てて {before:,} 行 → {len(X):,} 行になりました"
            f"（{1 - len(X) / before:.0%} を失っています）",
            tone="warn",
        )


# ======================================================================
# 学習と評価
# ======================================================================


@st.cache_data(show_spinner="交差検証中…")
def run_cv(_source: str, _target: str, _config, model_key, task, n_splits, seed, X, y):
    numeric, categorical = T.split_columns(
        pd.concat([X, y.rename(_target)], axis=1), _target
    )
    pipeline = T.build_pipeline(model_key, task, numeric, categorical, _config, seed)
    return T.cross_validate(pipeline, X, y, task, n_splits, seed)


@st.cache_data(show_spinner="ホールドアウトで評価中…")
def run_holdout(_source, _target, _config, model_key, task, seed, X, y):
    """訓練 / テストに分けて 1 回だけ学習し、詳細な図を描くための予測を得る。"""
    stratify = y if task == T.CLASSIFICATION and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=stratify
    )
    numeric, categorical = T.split_columns(
        pd.concat([X, y.rename(_target)], axis=1), _target
    )
    pipeline = T.build_pipeline(model_key, task, numeric, categorical, _config, seed)
    pipeline.fit(X_tr, y_tr)

    y_pred = pipeline.predict(X_te)
    y_proba = pipeline.predict_proba(X_te) if hasattr(pipeline, "predict_proba") else None
    scores = T.score_predictions(task, np.asarray(y_te), np.asarray(y_pred), y_proba)
    return pipeline, X_te, y_te, y_pred, y_proba, scores


cv = run_cv(source_key, target, config, model_key, task, int(n_splits), int(seed), X, y)
pipeline, X_test, y_test, y_pred, y_proba, scores = run_holdout(
    source_key, target, config, model_key, task, int(seed), X, y
)

baseline_cv = run_cv(
    source_key, target, config, "baseline", task, int(n_splits), int(seed), X, y
)

# ---- KPI --------------------------------------------------------------
metrics = T.metrics_for(task)
kpis = [
    Kpi("行数 × 列数", f"{len(X):,} × {len(X.columns)}",
        sub=f"数値 {len(numeric_cols)} / カテゴリ {len(categorical_cols)}"),
]
for metric in metrics:
    value = scores.get(metric.key, float("nan"))
    kpis.append(
        Kpi(
            metric.label,
            metric.fmt.format(value) if np.isfinite(value) else "—",
            sub="ホールドアウト（学習に使っていない 30%）",
            color=theme.PURPLE,
        )
    )
gain = cv.mean - baseline_cv.mean
kpis.append(
    Kpi(
        "ベースライン超え", f"{gain:+.3f}",
        sub=f"交差検証 {cv.metric} の差",
        color=score_color(gain, good=0.05, bad=0.0),
    )
)
kpi_row(kpis, accent)

if truncated:
    note(f"行数が多いため {MAX_ROWS:,} 行を無作為抽出しています", tone="warn")

if gain <= 0.01 and model_key != "baseline":
    note(
        "学習しないベースラインとほぼ同じスコアです — "
        "このモデルは実質何も学習できていません。特徴量を見直してください。",
        tone="bad",
    )
elif cv.std > abs(gain) and model_key != "baseline":
    note(
        f"分割によるばらつき（±{cv.std:.3f}）がベースラインとの差（{gain:+.3f}）より大きい — "
        "この差は偶然の範囲かもしれません。",
        tone="warn",
    )

tab_data, tab_eval, tab_explain = st.tabs(
    ["データを見る", "学習と評価", "何を根拠にしているか"]
)

# ======================================================================
# タブ 1: データを見る
# ======================================================================
with tab_data:
    panel("先頭 10 行", source_label)
    st.dataframe(frame.head(10), width="stretch")

    col_target, col_missing = st.columns(2)

    with col_target:
        panel("目的変数の分布", f"{target} — {T.TASK_LABELS[task]}")
        st.plotly_chart(P.target_figure(y, task), width="stretch", key=f"{KEY}_target")
        if task == T.CLASSIFICATION:
            counts = y.value_counts()
            ratio = counts.min() / counts.max()
            if ratio < 0.3:
                note(
                    f"最少クラスは最多クラスの {ratio:.0%} しかありません — "
                    "不均衡データです。正解率ではなく F1 を見てください。",
                    tone="warn",
                )

    with col_missing:
        panel("欠測の分布", "補完の対象になる列")
        missing_fig = P.missing_figure(frame)
        if missing_fig is None:
            st.caption(
                "欠測はありません。左の「人工的に欠測を作る」を上げると、"
                "補完方法の違いがスコアに出るようになります。"
            )
        else:
            st.plotly_chart(missing_fig, width="stretch", key=f"{KEY}_missing_chart")

    panel("列の内訳", "型と欠測とばらつき")
    profile = pd.DataFrame(
        {
            "列": [str(c) for c in frame.columns],
            "役割": [
                "目的変数" if c == target
                else "数値" if c in numeric_cols
                else "カテゴリ" if c in categorical_cols
                else "使わない（日時など）"
                for c in frame.columns
            ],
            "型": [str(t) for t in frame.dtypes],
            "欠測率": [f"{frame[c].isna().mean():.1%}" for c in frame.columns],
            "ユニーク数": [int(frame[c].nunique(dropna=True)) for c in frame.columns],
        }
    )
    st.dataframe(profile, width="stretch", hide_index=True)

    if numeric_cols:
        panel("数値列の要約統計", "スケールの違いに注目してください")
        st.dataframe(
            frame[numeric_cols].describe().T.round(3), width="stretch"
        )
        st.caption(
            "列によって値の桁が違う場合、線形モデルや距離を使うモデルでは標準化が必須です。"
            "木モデルは大小関係しか見ないので、標準化しなくても結果は変わりません。"
        )

# ======================================================================
# タブ 2: 学習と評価
# ======================================================================
with tab_eval:
    panel(
        "モデルを並べて比較",
        f"同じ前処理・同じ {n_splits} 分割の交差検証",
    )

    @st.cache_data(show_spinner="全モデルを交差検証中…")
    def compare_models(_source, _target, _config, task, n_splits, seed, X, y):
        out: dict[str, np.ndarray] = {}
        numeric, categorical = T.split_columns(
            pd.concat([X, y.rename(_target)], axis=1), _target
        )
        for key, spec in T.MODELS.items():
            pipeline = T.build_pipeline(key, task, numeric, categorical, _config, seed)
            out[spec.label] = T.cross_validate(
                pipeline, X, y, task, n_splits, seed
            ).scores
        return out

    comparison = compare_models(
        source_key, target, config, task, int(n_splits), int(seed), X, y
    )
    st.plotly_chart(
        P.cv_scores_figure(comparison, cv.metric), width="stretch", key=f"{KEY}_cv"
    )

    summary = pd.DataFrame(
        [
            {
                "モデル": label,
                "平均": round(float(np.mean(s)), 4),
                "標準偏差": round(float(np.std(s)), 4),
                "最小": round(float(np.min(s)), 4),
                "最大": round(float(np.max(s)), 4),
            }
            for label, s in comparison.items()
        ]
    ).sort_values("平均", ascending=False)
    st.dataframe(summary, width="stretch", hide_index=True)
    st.caption(
        "箱の高さ（ばらつき）に注目してください。モデル間の平均の差がこのばらつきより小さければ、"
        "その差は分割の運かもしれません。"
    )

    panel(
        f"{T.MODELS[model_key].label} の詳細",
        "ホールドアウト（学習に使っていない 30%）での結果",
    )

    if task == T.CLASSIFICATION:
        class_labels = [str(c) for c in np.unique(y)]
        col_cm, col_roc = st.columns([1, 1.2])
        with col_cm:
            st.markdown("**混同行列** — 各マスは件数と、その行に占める割合")
            st.plotly_chart(
                P.confusion_figure(y_test, y_pred, class_labels),
                width="stretch", key=f"{KEY}_cm",
            )
        with col_roc:
            st.markdown("**ROC 曲線** — 左上に張り付くほど良い")
            roc = P.roc_figure(np.asarray(y_test), y_proba, class_labels)
            if roc is None:
                st.caption("このモデルは確率を出さないため、ROC 曲線は描けません。")
            else:
                st.plotly_chart(roc, width="stretch", key=f"{KEY}_roc")
    else:
        col_scatter, col_resid = st.columns(2)
        with col_scatter:
            st.markdown("**実測 vs 予測** — 破線に乗るほど良い")
            st.plotly_chart(
                P.prediction_scatter(y_test, y_pred), width="stretch", key=f"{KEY}_scatter"
            )
        with col_resid:
            st.markdown("**残差** — 0 の周りに均等に散らばっていれば健全")
            st.plotly_chart(
                P.residual_figure(y_test, y_pred), width="stretch", key=f"{KEY}_resid"
            )
        st.caption(
            "残差が予測値によって偏っている（右へ行くほど散らばりが広がる等）なら、"
            "モデルが捉えきれていない構造が残っています。目的変数の対数を取ると直ることがあります。"
        )

# ======================================================================
# タブ 3: 解釈
# ======================================================================
with tab_explain:
    if model_key == "baseline":
        st.info("ベースラインは学習しないため、解釈するものがありません。別のモデルを選んでください。")
    else:
        panel("特徴量重要度", "2 つの測り方を並べます。食い違うときは Permutation を信じてください")

        col_model, col_perm = st.columns(2)

        with col_model:
            st.markdown(
                "**モデル内部の重要度** — 学習の副産物としてただで手に入るが、"
                "値の種類が多い特徴量を過大評価しがち"
            )
            importance = T.model_importance(pipeline)
            if importance is None:
                st.caption("このモデルからは重要度を取り出せません。")
            else:
                value_column = importance.columns[1]
                st.plotly_chart(
                    P.importance_figure(importance, value_column),
                    width="stretch", key=f"{KEY}_imp",
                )

        with col_perm:
            st.markdown(
                "**Permutation Importance** — 列をシャッフルしてスコアがどれだけ落ちるかで測る。"
                "測りたいことをそのまま測っている"
            )

            @st.cache_data(show_spinner="Permutation Importance を計算中…")
            def perm(_source, _target, _config, model_key, task, seed, X_te, y_te, _pipe):
                return T.permutation_table(_pipe, X_te, y_te, task, n_repeats=5, seed=seed)

            permutation = perm(
                source_key, target, config, model_key, task, int(seed),
                X_test, y_test, pipeline,
            )
            st.plotly_chart(
                P.importance_figure(permutation, "スコア低下", "ばらつき"),
                width="stretch", key=f"{KEY}_perm",
            )

        useless = permutation[permutation["スコア低下"] <= 0]
        if len(useless):
            st.caption(
                f"スコア低下が 0 以下の特徴量が {len(useless)} 個あります"
                f"（{'、'.join(str(c) for c in useless['特徴量'].head(5))}"
                f"{' ほか' if len(useless) > 5 else ''}）。"
                "シャッフルしても精度が落ちない ＝ 使われていない列です。落とせば学習が速くなります。"
            )

        panel("SHAP — 1 件ずつの予測を分解する", "木モデルのみ対応")

        if not T.MODELS[model_key].tree_based:
            st.info(
                "SHAP のこの計算方法（TreeExplainer）は木ベースのモデル専用です。"
                "LightGBM かランダムフォレストを選ぶと表示されます。"
            )
        else:
            class_index, class_label = 0, ""
            if task == T.CLASSIFICATION:
                classes = [str(c) for c in np.unique(y)]
                if len(classes) > 2:
                    class_label = st.selectbox(
                        "どのクラスについての説明か", classes, key=f"{KEY}_shapcls"
                    )
                    class_index = classes.index(class_label)
                else:
                    class_index, class_label = 1, classes[1]
                    st.caption(f"「{class_label}」である確率を押し上げる / 押し下げる向きで示します。")

            @st.cache_data(show_spinner="SHAP を計算中…")
            def shap_for(_source, _target, _config, model_key, task, ci, seed, X_te, _pipe):
                return T.explain_with_shap(
                    _pipe, X_te, task, class_index=ci, seed=seed
                )

            explanation = shap_for(
                source_key, target, config, model_key, task, class_index,
                int(seed), X_test, pipeline,
            )

            if explanation is None:
                st.warning("このモデル構成では SHAP を計算できませんでした。")
            else:
                st.plotly_chart(
                    P.shap_beeswarm(explanation), width="stretch", key=f"{KEY}_bee"
                )
                st.caption(
                    "1 点が 1 件のデータです。右にあるほどその予測を押し上げ、左にあるほど押し下げています。"
                    "点の色はその特徴量の値の大小。**色がきれいに左右に分かれていれば、"
                    "その特徴量は一貫した向きで効いている**と読めます。"
                )

                panel("この 1 件はなぜこう予測されたのか", "行を選ぶと寄与の内訳が出ます")
                row = st.slider(
                    "何行目を見るか", 0, len(explanation.values) - 1, 0, 1,
                    key=f"{KEY}_row",
                )
                st.plotly_chart(
                    P.shap_waterfall(explanation, row), width="stretch", key=f"{KEY}_wf"
                )
                total = float(explanation.values[row].sum())
                st.caption(
                    f"基準値 `{explanation.base_value:.4g}`（全体の平均的な予測）から、"
                    f"各特徴量の寄与を足し引きして `{explanation.base_value + total:.4g}` に至っています。"
                    "ピンクが押し上げ、水色が押し下げです。"
                )

# ======================================================================
# 実験の記録
# ======================================================================

EL.record_panel(
    lab="テーブルデータ",
    params={
        "データ": source_key,
        "目的変数": target,
        "モデル": T.MODELS[model_key].label,
        "欠測の注入率": missing_rate,
        "数値の補完": T.NUMERIC_IMPUTE_LABELS[numeric_impute],
        "カテゴリの符号化": T.ENCODE_LABELS[encode],
        "分割数": int(n_splits),
        "シード": int(seed),
    },
    metrics={
        f"交差検証({cv.metric})": cv.mean,
        "分割のばらつき": cv.std,
        **{m.label: scores.get(m.key, float("nan")) for m in metrics},
    },
    key=KEY,
    default_experiment=f"テーブル/{source_key.split(':')[-1]}",
)

explain("tabular")
