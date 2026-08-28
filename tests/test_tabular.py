"""テーブルデータの前処理・学習・解釈のテスト。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from mllab.models import tabular as T

# SHAP と LightGBM が出す将来変更の予告は、テストの見通しを悪くするだけなので黙らせる
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def wine() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_wine(return_X_y=True, as_frame=True)
    return X, y.rename("品種")


@pytest.fixture(scope="module")
def diabetes() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X, y.rename("進行度")


def mixed_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """数値・カテゴリ・日時・目的変数が混ざった表。"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "日付": pd.date_range("2024-01-01", periods=n, freq="D"),
            "年齢": rng.integers(18, 80, n),
            "年収": rng.normal(5_000_000, 1_500_000, n).round(),
            "職業": rng.choice(["会社員", "自営業", "学生"], n),
            "都道府県": rng.choice([f"県{i}" for i in range(40)], n),
            "契約": rng.integers(0, 2, n),
        }
    )


# ---- 課題の判定 -------------------------------------------------------

@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series(list("abcabc")), T.CLASSIFICATION),
        (pd.Series([0, 1, 1, 0, 1]), T.CLASSIFICATION),
        (pd.Series([1, 2, 3, 4, 5, 3, 2]), T.CLASSIFICATION),
        (pd.Series([True, False, True]), T.CLASSIFICATION),
        (pd.Series(np.linspace(0.1, 9.9, 200)), T.REGRESSION),
        (pd.Series(np.arange(200)), T.REGRESSION),  # 整数だが多水準なら回帰
        (pd.Series([], dtype=float), T.REGRESSION),
    ],
)
def test_infer_task(series: pd.Series, expected: str) -> None:
    assert T.infer_task(series) == expected


def test_infer_task_ignores_missing_values() -> None:
    assert T.infer_task(pd.Series([0, 1, np.nan, 1, 0])) == T.CLASSIFICATION


# ---- 列の分類 ---------------------------------------------------------

def test_split_columns_separates_kinds() -> None:
    numeric, categorical = T.split_columns(mixed_frame(), "契約")
    assert set(numeric) == {"年齢", "年収"}
    assert set(categorical) == {"職業", "都道府県"}
    # 日時はそのままでは特徴量にならないので、どちらにも入れない
    assert "日付" not in numeric and "日付" not in categorical


def test_split_columns_excludes_target_and_drops() -> None:
    numeric, categorical = T.split_columns(mixed_frame(), "契約", drop=["年収"])
    assert "年収" not in numeric
    assert "契約" not in numeric + categorical


def test_suggest_targets_deprioritises_ids_and_dates() -> None:
    frame = mixed_frame(200)
    frame["会員ID"] = [f"M{i:05d}" for i in range(len(frame))]
    order = T.suggest_targets(frame)
    # ID と日付は候補の後ろへ回る
    assert order.index("契約") < order.index("会員ID")
    assert order.index("契約") < order.index("日付")


# ---- 欠測の注入と補完 -------------------------------------------------

def test_inject_missing_hits_expected_rate() -> None:
    frame = mixed_frame(2000)
    out = T.inject_missing(frame, ["年齢"], rate=0.3, seed=0)
    assert 0.25 < out["年齢"].isna().mean() < 0.35
    assert out["年収"].isna().sum() == 0  # 指定した列だけ


def test_inject_missing_is_a_noop_at_zero() -> None:
    frame = mixed_frame(50)
    pd.testing.assert_frame_equal(T.inject_missing(frame, ["年齢"], 0.0), frame)


def test_inject_missing_does_not_mutate_input() -> None:
    frame = mixed_frame(100)
    T.inject_missing(frame, ["年齢"], 0.5, seed=1)
    assert frame["年齢"].isna().sum() == 0


@pytest.mark.parametrize("strategy", ["median", "mean", "constant"])
def test_preprocessor_fills_missing_values(strategy: str) -> None:
    frame = T.inject_missing(mixed_frame(300), ["年齢", "年収"], 0.3, seed=0)
    numeric, categorical = T.split_columns(frame, "契約")
    prep = T.build_preprocessor(
        numeric, categorical, T.PreprocessConfig(numeric_impute=strategy)
    )
    out = prep.fit_transform(frame.drop(columns=["契約"]))
    assert np.isfinite(np.asarray(out, dtype=float)).all(), "補完後に欠測が残っている"


def test_onehot_expands_columns_and_ordinal_does_not() -> None:
    frame = mixed_frame(300)
    X = frame.drop(columns=["契約"])
    numeric, categorical = T.split_columns(frame, "契約")

    onehot = T.build_preprocessor(
        numeric, categorical, T.PreprocessConfig(encode="onehot")
    ).fit_transform(X)
    ordinal = T.build_preprocessor(
        numeric, categorical, T.PreprocessConfig(encode="ordinal")
    ).fit_transform(X)

    assert ordinal.shape[1] == len(numeric) + len(categorical)
    assert onehot.shape[1] > ordinal.shape[1]
    # 水準数の上限を超えても列が無限に増えないこと
    assert onehot.shape[1] < len(numeric) + T.ONEHOT_MAX_CARDINALITY * 2


# ---- パイプライン -----------------------------------------------------

@pytest.mark.parametrize("model_key", list(T.MODELS))
def test_every_model_fits_classification(model_key: str, wine) -> None:
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline(model_key, T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)
    assert set(np.unique(predictions)) <= set(np.unique(y))


@pytest.mark.parametrize("model_key", list(T.MODELS))
def test_every_model_fits_regression(model_key: str, diabetes) -> None:
    X, y = diabetes
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline(model_key, T.REGRESSION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    assert np.isfinite(pipeline.predict(X)).all()


def test_tree_models_skip_scaling(wine) -> None:
    """木モデルは大小関係しか見ないので、標準化を挟まない。"""
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    config = T.PreprocessConfig(scale=True)

    tree = T.build_pipeline("lightgbm", T.CLASSIFICATION, numeric, categorical, config, 0)
    linear = T.build_pipeline("linear", T.CLASSIFICATION, numeric, categorical, config, 0)

    def numeric_steps(pipeline):
        transformer = dict(
            (name, trans) for name, trans, _ in pipeline.named_steps["prep"].transformers
        )["numeric"]
        return [name for name, _ in transformer.steps]

    assert "scale" not in numeric_steps(tree)
    assert "scale" in numeric_steps(linear)


def test_pipeline_handles_categorical_and_missing_end_to_end() -> None:
    frame = T.inject_missing(mixed_frame(400), ["年齢", "年収"], 0.2, seed=0)
    X, y = frame.drop(columns=["契約"]), frame["契約"]
    numeric, categorical = T.split_columns(frame, "契約")
    pipeline = T.build_pipeline("lightgbm", T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    assert len(pipeline.predict(X)) == len(y)


def test_feature_names_match_transformed_width(wine) -> None:
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline("lightgbm", T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    width = pipeline.named_steps["prep"].transform(X).shape[1]
    assert len(T.feature_names(pipeline)) == width


# ---- 評価 -------------------------------------------------------------

def test_cross_validation_reports_spread(wine) -> None:
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline("forest", T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    cv = T.cross_validate(pipeline, X, y, T.CLASSIFICATION, n_splits=5, seed=0)
    assert len(cv.scores) == 5
    assert 0.0 <= cv.mean <= 1.0
    assert cv.std >= 0.0
    assert cv.metric == "accuracy"


def test_real_model_beats_baseline(wine) -> None:
    """ベースラインを超えられないなら、パイプラインのどこかが壊れている。"""
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)

    def mean_score(key: str) -> float:
        pipeline = T.build_pipeline(key, T.CLASSIFICATION, numeric, categorical,
                                    T.PreprocessConfig(), 0)
        return T.cross_validate(pipeline, X, y, T.CLASSIFICATION, 5, 0).mean

    assert mean_score("lightgbm") > mean_score("baseline") + 0.2


def test_score_predictions_classification() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    scores = T.score_predictions(T.CLASSIFICATION, y_true, y_pred, proba)
    assert scores["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= scores["f1"] <= 1.0
    assert scores["roc_auc"] == pytest.approx(1.0)


def test_score_predictions_survives_single_class_proba() -> None:
    """検証分割にクラスが 1 つしか無くても、指標計算で落ちない。"""
    scores = T.score_predictions(
        T.CLASSIFICATION, np.array([1, 1]), np.array([1, 1]),
        np.array([[0.2, 0.8], [0.1, 0.9]]),
    )
    assert scores["accuracy"] == 1.0
    assert np.isnan(scores["roc_auc"])


def test_score_predictions_regression() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    scores = T.score_predictions(T.REGRESSION, y_true, y_true.copy(), None)
    assert scores["r2"] == pytest.approx(1.0)
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["mae"] == pytest.approx(0.0)


def test_metrics_for_each_task() -> None:
    assert {m.key for m in T.metrics_for(T.CLASSIFICATION)} == {"accuracy", "f1", "roc_auc"}
    assert {m.key for m in T.metrics_for(T.REGRESSION)} == {"r2", "rmse", "mae"}


# ---- 解釈 -------------------------------------------------------------

@pytest.mark.parametrize("model_key", ["lightgbm", "forest"])
@pytest.mark.parametrize("class_index", [0, 1, 2])
def test_shap_shape_for_multiclass(model_key: str, class_index: int, wine) -> None:
    """クラス軸の位置は shap のバージョンで変わるので、必ず (n, f) に落ちること。"""
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline(model_key, T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)

    explanation = T.explain_with_shap(pipeline, X, T.CLASSIFICATION, class_index=class_index)
    assert explanation is not None
    assert explanation.values.shape == (len(X), X.shape[1])
    assert explanation.features.shape == explanation.values.shape
    assert len(explanation.names) == X.shape[1]
    assert np.isfinite(explanation.base_value)


def test_shap_binary_and_regression(diabetes) -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    numeric, categorical = T.split_columns(pd.concat([X, y.rename("t")], axis=1), "t")
    clf = T.build_pipeline("lightgbm", T.CLASSIFICATION, numeric, categorical,
                           T.PreprocessConfig(), 0)
    clf.fit(X, y)
    binary = T.explain_with_shap(clf, X, T.CLASSIFICATION, class_index=1)
    assert binary is not None and binary.values.shape[1] == X.shape[1]

    Xr, yr = diabetes
    nr, cr = T.split_columns(pd.concat([Xr, yr], axis=1), yr.name)
    reg = T.build_pipeline("lightgbm", T.REGRESSION, nr, cr, T.PreprocessConfig(), 0)
    reg.fit(Xr, yr)
    regression = T.explain_with_shap(reg, Xr, T.REGRESSION)
    assert regression is not None
    # 行数が上限を超えるデータは抽出されるので、上限側と一致する
    assert regression.values.shape == (
        min(len(Xr), T.SHAP_MAX_SAMPLES),
        Xr.shape[1],
    )


def test_shap_caps_sample_count(wine) -> None:
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline("lightgbm", T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    explanation = T.explain_with_shap(pipeline, X, T.CLASSIFICATION, max_samples=50)
    assert explanation.values.shape[0] == 50


def test_shap_returns_none_for_non_tree_models(wine) -> None:
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline("linear", T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    assert T.explain_with_shap(pipeline, X, T.CLASSIFICATION) is None


def test_explanation_mean_abs_is_sorted(wine) -> None:
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline("lightgbm", T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    table = T.explain_with_shap(pipeline, X, T.CLASSIFICATION, class_index=1).mean_abs()
    values = table["平均|SHAP|"].to_numpy()
    assert np.all(np.diff(values) <= 1e-12)


@pytest.mark.parametrize("model_key", ["lightgbm", "forest", "linear"])
def test_model_importance_matches_feature_names(model_key: str, wine) -> None:
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline(model_key, T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    table = T.model_importance(pipeline)
    assert table is not None
    assert len(table) == len(T.feature_names(pipeline))
    assert (table.iloc[:, 1] >= 0).all()


def test_model_importance_is_none_for_baseline(wine) -> None:
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline("baseline", T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    assert T.model_importance(pipeline) is None


def test_permutation_table_uses_original_columns(wine) -> None:
    """One-Hot で増える前の、人間が理解している列の単位で返ること。"""
    X, y = wine
    numeric, categorical = T.split_columns(pd.concat([X, y], axis=1), y.name)
    pipeline = T.build_pipeline("forest", T.CLASSIFICATION, numeric, categorical,
                                T.PreprocessConfig(), 0)
    pipeline.fit(X, y)
    table = T.permutation_table(pipeline, X, y, T.CLASSIFICATION, n_repeats=3, seed=0)
    assert set(table["特徴量"]) == set(X.columns)
    assert table["スコア低下"].is_monotonic_decreasing


def test_suggest_targets_prefers_last_column() -> None:
    """公開データもCSVも目的変数を最後の列に置く慣習が強い。"""
    frame = pd.DataFrame(
        {"特徴1": [1, 2, 3], "特徴2": [4.0, 5.0, 6.0], "正解": ["A", "B", "A"]}
    )
    assert T.suggest_targets(frame)[0] == "正解"


def test_suggest_targets_on_saved_wine_shape() -> None:
    """カタログに保存した形（特徴量 + 末尾に目的変数）で正しく先頭に来ること。"""
    X, y = load_wine(return_X_y=True, as_frame=True)
    frame = X.copy()
    frame["産地"] = pd.Categorical(y).rename_categories(["A", "B", "C"]).astype(str)
    assert T.suggest_targets(frame)[0] == "産地"
