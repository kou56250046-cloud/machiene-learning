"""テーブルデータの前処理・学習・評価。

決定境界ラボまでは 2 次元の合成データだったが、ここからは
「列の型がばらばらで、欠測があり、カテゴリが混ざった実データ」を扱う。
実務で時間を取られるのはモデル選びではなく、この前処理の部分。

計算はすべてここに置き、`app/views/lab8_tabular.py` は UI の組み立てだけにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CLASSIFICATION = "classification"
REGRESSION = "regression"

TASK_LABELS = {CLASSIFICATION: "分類", REGRESSION: "回帰"}

#: 目的変数が数値でも、ユニーク数がこれ以下なら分類とみなす
CLASSIFICATION_MAX_CLASSES = 20

#: One-Hot にすると列が爆発するので、水準数がこれを超えたら順序符号化に落とす
ONEHOT_MAX_CARDINALITY = 30


# ======================================================================
# 課題の判定と列の分類
# ======================================================================


def infer_task(y: pd.Series) -> str:
    """目的変数から分類か回帰かを推定する。

    文字列・カテゴリなら分類。数値でも整数でユニーク数が少なければ分類
    （0/1 のラベルや 1〜5 の評価など、数値で入っている分類が多いため）。
    """
    y = y.dropna()
    if y.empty:
        return REGRESSION
    if not pd.api.types.is_numeric_dtype(y):
        return CLASSIFICATION
    unique = y.nunique()
    if unique <= 2:
        return CLASSIFICATION
    is_integral = bool(np.all(np.equal(np.mod(y.to_numpy(dtype=float), 1), 0)))
    if is_integral and unique <= CLASSIFICATION_MAX_CLASSES:
        return CLASSIFICATION
    return REGRESSION


def split_columns(
    frame: pd.DataFrame, target: str, drop: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """特徴量の列を数値列とカテゴリ列に分ける。

    日時列は「そのままでは特徴量にならない」ので除外する
    （時系列としての扱いは Phase 4 の時系列ラボで行う）。
    """
    excluded = {target, *(drop or [])}
    numeric: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def suggest_targets(frame: pd.DataFrame) -> list[str]:
    """目的変数の候補を、それらしい順に並べて返す。

    2 つの経験則を使う。
    1. 識別子・自由文（ほぼ全行が別の値）と日時は目的変数になりえないので後ろへ回す。
    2. 残りは末尾の列から順に出す。公開データセットでも CSV でも、
       目的変数を最後の列に置く慣習が強いため。
    """
    positions = {column: i for i, column in enumerate(frame.columns)}

    def score(column: str) -> tuple[int, int]:
        series = frame[column]
        unique = series.nunique(dropna=True)
        # ほぼ全行が別の値 = ID か自由文。目的変数には向かない
        looks_like_id = unique > max(50, len(frame) * 0.9)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        return (int(looks_like_id or is_datetime), -positions[column])

    return sorted(frame.columns, key=score)


# ======================================================================
# 前処理
# ======================================================================


@dataclass(frozen=True)
class PreprocessConfig:
    """前処理の設定。ハッシュ可能なのでキャッシュのキーに使える。"""

    numeric_impute: str = "median"  # median | mean | constant | drop_rows
    categorical_impute: str = "most_frequent"  # most_frequent | constant
    scale: bool = True
    encode: str = "onehot"  # onehot | ordinal


NUMERIC_IMPUTE_LABELS = {
    "median": "中央値で埋める（外れ値に強い）",
    "mean": "平均値で埋める",
    "constant": "0 で埋める",
    "drop_rows": "欠測のある行を捨てる",
}

CATEGORICAL_IMPUTE_LABELS = {
    "most_frequent": "最頻値で埋める",
    "constant": "「欠測」という水準を作る",
}

ENCODE_LABELS = {
    "onehot": "One-Hot（水準ごとに 0/1 の列を作る）",
    "ordinal": "順序符号化（水準に番号を振る）",
}


def build_preprocessor(
    numeric: list[str], categorical: list[str], config: PreprocessConfig
) -> ColumnTransformer:
    """列の型ごとに違う処理を当てる ColumnTransformer を組む。"""
    steps: list[tuple[str, Any, list[str]]] = []

    if numeric:
        numeric_steps: list[tuple[str, Any]] = []
        if config.numeric_impute != "drop_rows":
            strategy = config.numeric_impute
            numeric_steps.append(
                (
                    "impute",
                    SimpleImputer(
                        strategy=strategy if strategy != "constant" else "constant",
                        fill_value=0 if strategy == "constant" else None,
                    ),
                )
            )
        if config.scale:
            numeric_steps.append(("scale", StandardScaler()))
        steps.append(
            ("numeric", Pipeline(numeric_steps) if numeric_steps else "passthrough", numeric)
        )

    if categorical:
        encoder: Any
        if config.encode == "onehot":
            encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                max_categories=ONEHOT_MAX_CARDINALITY,
                sparse_output=False,
                min_frequency=2,
            )
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        steps.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy=config.categorical_impute,
                                fill_value="（欠測）"
                                if config.categorical_impute == "constant"
                                else None,
                            ),
                        ),
                        ("encode", encoder),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


def inject_missing(
    frame: pd.DataFrame, columns: list[str], rate: float, seed: int = 0
) -> pd.DataFrame:
    """指定した列に人工的な欠測を作る。

    同梱データセットには欠測が無いため、補完の効果を体感するには
    自分で欠測を作る必要がある。学習用の仕掛け。
    """
    if rate <= 0 or not columns:
        return frame
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for column in columns:
        mask = rng.random(len(out)) < rate
        out.loc[mask, column] = np.nan
    return out


# ======================================================================
# モデル
# ======================================================================


@dataclass(frozen=True)
class TabularModel:
    """テーブルデータ向けモデル 1 つぶんの定義。"""

    key: str
    label: str
    build: Callable[[str, int], BaseEstimator]
    summary: str
    #: 木ベースか（SHAP の TreeExplainer が使えるか、標準化が要らないか）
    tree_based: bool = False
    #: 比較の基準線として扱うか
    is_baseline: bool = False


def _lightgbm(task: str, seed: int) -> BaseEstimator:
    common = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                  random_state=seed, verbose=-1)
    return lgb.LGBMClassifier(**common) if task == CLASSIFICATION else lgb.LGBMRegressor(**common)


def _forest(task: str, seed: int) -> BaseEstimator:
    common = dict(n_estimators=200, random_state=seed, n_jobs=-1)
    return (
        RandomForestClassifier(**common)
        if task == CLASSIFICATION
        else RandomForestRegressor(**common)
    )


def _linear(task: str, seed: int) -> BaseEstimator:
    if task == CLASSIFICATION:
        return LogisticRegression(max_iter=3000, random_state=seed)
    return Ridge(alpha=1.0, random_state=seed)


def _baseline(task: str, seed: int) -> BaseEstimator:
    if task == CLASSIFICATION:
        return DummyClassifier(strategy="most_frequent")
    return DummyRegressor(strategy="mean")


MODELS: dict[str, TabularModel] = {
    "lightgbm": TabularModel(
        "lightgbm", "LightGBM", _lightgbm,
        "勾配ブースティングの高速な実装。表形式データでは現在も最強クラス。",
        tree_based=True,
    ),
    "forest": TabularModel(
        "forest", "ランダムフォレスト", _forest,
        "既定値のまま回してもそこそこ効く、頼れるベースライン。",
        tree_based=True,
    ),
    "linear": TabularModel(
        "linear", "線形モデル", _linear,
        "分類ならロジスティック回帰、回帰なら Ridge。係数が読めるのが利点。",
    ),
    "baseline": TabularModel(
        "baseline", "ベースライン（学習しない）", _baseline,
        "分類は常に最多クラス、回帰は常に平均を返すだけ。これを超えられないモデルは無意味。",
        is_baseline=True,
    ),
}


def build_pipeline(
    model_key: str,
    task: str,
    numeric: list[str],
    categorical: list[str],
    config: PreprocessConfig,
    seed: int = 0,
) -> Pipeline:
    """前処理とモデルをつないだパイプラインを作る。

    前処理を Pipeline に入れておくと、交差検証のたびに訓練分割だけで
    補完値やスケールを学習し直してくれる。これを分けて書くと、
    検証データの情報が訓練に漏れる（リーク）。
    """
    spec = MODELS[model_key]
    # 木モデルは標準化が不要なので、設定に関わらず切っておく
    effective = config if not spec.tree_based else PreprocessConfig(
        numeric_impute=config.numeric_impute,
        categorical_impute=config.categorical_impute,
        scale=False,
        encode=config.encode,
    )
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric, categorical, effective)),
            ("model", spec.build(task, seed)),
        ]
    )


# ======================================================================
# 評価
# ======================================================================


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    #: 大きいほど良いか
    higher_is_better: bool
    fmt: str


CLASSIFICATION_METRICS = (
    Metric("accuracy", "正解率", True, "{:.1%}"),
    Metric("f1", "F1（マクロ平均）", True, "{:.3f}"),
    Metric("roc_auc", "ROC-AUC", True, "{:.3f}"),
)

REGRESSION_METRICS = (
    Metric("r2", "決定係数 R²", True, "{:.3f}"),
    Metric("rmse", "RMSE", False, "{:.4g}"),
    Metric("mae", "MAE", False, "{:.4g}"),
)


def metrics_for(task: str) -> tuple[Metric, ...]:
    return CLASSIFICATION_METRICS if task == CLASSIFICATION else REGRESSION_METRICS


def score_predictions(
    task: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> dict[str, float]:
    """予測結果から指標をまとめて計算する。"""
    if task == CLASSIFICATION:
        scores = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc": float("nan"),
        }
        if y_proba is not None:
            try:
                classes = np.unique(y_true)
                if len(classes) == 2:
                    scores["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    scores["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    )
            except ValueError:
                # 検証分割にクラスが 1 つしか無いなど。指標を出せないだけで処理は続ける
                pass
        return scores

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


@dataclass
class CrossValidation:
    """交差検証の結果。"""

    metric: str
    scores: np.ndarray
    #: 分割の運によるばらつき。平均だけ見ると見落とす
    mean: float = field(init=False)
    std: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean = float(np.mean(self.scores))
        self.std = float(np.std(self.scores))


#: 交差検証に使う sklearn のスコア名
CV_SCORING = {CLASSIFICATION: "accuracy", REGRESSION: "r2"}


def cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    n_splits: int = 5,
    seed: int = 0,
) -> CrossValidation:
    """交差検証でスコアのばらつきを測る。

    分類は層化分割（各分割のクラス比を揃える）。不均衡データで
    普通の KFold を使うと、陽性が 1 件も入らない分割ができてしまう。
    """
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if task == CLASSIFICATION
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    scoring = CV_SCORING[task]
    scores = cross_val_score(pipeline, X, y, cv=splitter, scoring=scoring, n_jobs=1)
    return CrossValidation(metric=scoring, scores=np.asarray(scores))


def feature_names(pipeline: Pipeline) -> list[str]:
    """前処理後の特徴量名。One-Hot で増えた列も展開された名前で返る。"""
    try:
        return [str(n) for n in pipeline.named_steps["prep"].get_feature_names_out()]
    except (AttributeError, ValueError):
        return []


# ======================================================================
# 解釈
# ======================================================================

#: SHAP は 1 行ずつ寄与を計算するので、行数を切らないと待たされる
SHAP_MAX_SAMPLES = 400


@dataclass
class Explanation:
    """モデルが何を根拠にしているかの説明。"""

    #: (n_samples, n_features) の SHAP 値
    values: np.ndarray
    #: 前処理後の特徴量の実値（色分けに使う）
    features: np.ndarray
    names: list[str]
    #: 予測の出発点（全体の平均的な予測）
    base_value: float
    #: 多クラスのとき、どのクラスについての説明か
    class_label: str = ""

    def mean_abs(self) -> pd.DataFrame:
        """|SHAP| の平均で並べた重要度。"""
        importance = np.abs(self.values).mean(axis=0)
        return (
            pd.DataFrame({"特徴量": self.names, "平均|SHAP|": importance})
            .sort_values("平均|SHAP|", ascending=False)
            .reset_index(drop=True)
        )


def explain_with_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    task: str,
    class_index: int = 1,
    class_label: str = "",
    max_samples: int = SHAP_MAX_SAMPLES,
    seed: int = 0,
) -> Explanation | None:
    """木モデルの予測を SHAP で分解する。

    木ベース以外や、SHAP が対応していない構成では None を返す
    （画面側は None なら SHAP の節を出さない）。
    """
    import shap

    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]

    sample = X
    if len(X) > max_samples:
        sample = X.sample(max_samples, random_state=seed)

    transformed = prep.transform(sample)
    names = feature_names(pipeline) or [f"x{i}" for i in range(transformed.shape[1])]

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        base = explainer.expected_value
    except Exception:  # noqa: BLE001 - 対応外のモデルは説明を諦めるだけ
        return None

    values = np.asarray(values)
    base_array = np.atleast_1d(np.asarray(base, dtype=float))

    # 多クラスは 3 次元で返るが、クラス軸の位置は shap のバージョンで変わる
    # （(n, f, k) のことも、クラスごとの配列を積んだ (k, n, f) のこともある）。
    # サンプル数・特徴量数と一致しない軸をクラス軸とみなして取り出す。
    if values.ndim == 3:
        n_samples, n_features = transformed.shape
        class_axis = next(
            (
                axis
                for axis, size in enumerate(values.shape)
                if not (axis == 0 and size == n_samples)
                and not (axis == 1 and size == n_features)
            ),
            2,
        )
        index = min(class_index, values.shape[class_axis] - 1)
        values = np.take(values, index, axis=class_axis)
        base_value = float(base_array[min(index, len(base_array) - 1)])
    else:
        base_value = float(base_array[0])

    if values.shape[1] != len(names):
        names = [f"x{i}" for i in range(values.shape[1])]

    return Explanation(
        values=values,
        features=np.asarray(transformed, dtype=float),
        names=names,
        base_value=base_value,
        class_label=class_label,
    )


def permutation_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """列をシャッフルしたときのスコア低下を測る（元の列名のまま）。

    不純度ベースの重要度と違い、One-Hot で増える前の「人間が理解している列」
    の単位で効き目が分かるのが利点。
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring=CV_SCORING[task],
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "特徴量": list(X.columns),
                "スコア低下": result.importances_mean,
                "ばらつき": result.importances_std,
            }
        )
        .sort_values("スコア低下", ascending=False)
        .reset_index(drop=True)
    )


def model_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """モデル自身が持つ重要度（木なら分岐への寄与、線形なら係数）。"""
    model = pipeline.named_steps["model"]
    names = feature_names(pipeline)
    if not names:
        return None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        label = "重要度"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        # 多クラスのロジスティック回帰は (k, f) なので大きさに畳む
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        label = "係数の絶対値"
    else:
        return None

    if len(values) != len(names):
        return None

    return (
        pd.DataFrame({"特徴量": names, label: values})
        .sort_values(label, ascending=False)
        .reset_index(drop=True)
    )
