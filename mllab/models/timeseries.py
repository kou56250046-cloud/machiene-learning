"""時系列の分解・自己相関・予測。

テーブルデータと決定的に違うのは「行の順序に意味がある」こと。
だから行をランダムに分けてはいけないし、未来の情報を特徴量に混ぜてもいけない。
ここではその制約を守った形で、分解・周期の発見・予測までを扱う。

計算はすべてここに置き、`app/views/lab9_timeseries.py` は UI の組み立てだけにする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

# ======================================================================
# 系列の取り出しとリサンプリング
# ======================================================================

#: リサンプリングの粒度 → (pandas の頻度, 表示名, 1 年あたりの点数)
FREQUENCIES: dict[str, tuple[str, str, int]] = {
    "D": ("D", "日次", 365),
    "W": ("W", "週次", 52),
    "ME": ("ME", "月次", 12),
}

#: 集約の方法
AGGREGATIONS: dict[str, str] = {
    "mean": "平均",
    "sum": "合計",
    "max": "最大",
    "min": "最小",
    "last": "最後の値",
}


def find_time_columns(frame: pd.DataFrame) -> list[str]:
    """日時として使える列を返す。"""
    return [c for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])]


def find_value_columns(frame: pd.DataFrame, time_column: str) -> list[str]:
    """観測値として使える数値列を返す。"""
    return [
        c
        for c in frame.columns
        if c != time_column
        and pd.api.types.is_numeric_dtype(frame[c])
        and not pd.api.types.is_bool_dtype(frame[c])
    ]


@dataclass(frozen=True)
class SeriesConfig:
    """系列の切り出し方。ハッシュ可能なのでキャッシュのキーに使える。"""

    time_column: str
    value_column: str
    frequency: str = "D"
    aggregation: str = "mean"


def build_series(frame: pd.DataFrame, config: SeriesConfig) -> pd.Series:
    """データフレームから、等間隔の時系列を作る。

    株価のように「取引日しかない」データは、そのままでは日付が飛び飛びで
    自己相関や季節分解が使えない。等間隔に直したうえで、空いた日は前の値で埋める。
    """
    series = (
        frame[[config.time_column, config.value_column]]
        .dropna(subset=[config.time_column])
        .set_index(config.time_column)[config.value_column]
        .sort_index()
    )
    series = pd.to_numeric(series, errors="coerce")

    freq = FREQUENCIES[config.frequency][0]
    resampled = getattr(series.resample(freq), config.aggregation)()

    # 休場日・欠測は直前の値で埋める（時系列では未来の値を使ってはいけないので前方向のみ）
    return resampled.ffill().dropna()


def gaps_in_series(frame: pd.DataFrame, config: SeriesConfig) -> int:
    """等間隔に直したときに、元データに値が無かった点の数。"""
    series = (
        frame[[config.time_column, config.value_column]]
        .dropna(subset=[config.time_column])
        .set_index(config.time_column)[config.value_column]
        .sort_index()
    )
    freq = FREQUENCIES[config.frequency][0]
    resampled = getattr(series.resample(freq), config.aggregation)()
    return int(resampled.isna().sum())


# ======================================================================
# 分解
# ======================================================================


@dataclass
class Decomposition:
    """系列を トレンド + 季節 + 残差 に分けた結果。"""

    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    period: int

    @property
    def seasonal_amplitude(self) -> float:
        """季節成分の振れ幅。季節性の強さの目安。"""
        return float(self.seasonal.max() - self.seasonal.min())

    @property
    def seasonal_strength(self) -> float:
        """季節性の強さ（0〜1）。

        季節成分を取り除いたときに、ばらつきがどれだけ減るかで測る。
        1 に近いほど「季節を知っているだけでかなり説明できる」。
        """
        return _strength(self.residual, self.seasonal)

    @property
    def trend_strength(self) -> float:
        """トレンドの強さ（0〜1）。"""
        return _strength(self.residual, self.trend)


def _strength(residual: pd.Series, component: pd.Series) -> float:
    """成分の強さ = 1 − 分散(残差) / 分散(残差 + 成分)。"""
    combined = (residual + component).var()
    if not np.isfinite(combined) or combined <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - residual.var() / combined)))


def suggest_period(frequency: str) -> int:
    """粒度から、季節周期の既定値を決める（日次なら 365 など）。"""
    return FREQUENCIES[frequency][2]


def decompose(series: pd.Series, period: int, robust: bool = True) -> Decomposition:
    """STL で分解する。

    STL は移動平均を繰り返してトレンドと季節を取り出す手法。
    robust=True にすると外れ値の影響を抑えるので、実データ向き。

    Raises:
        ValueError: 周期の 2 倍より系列が短いとき（分解できない）。
    """
    clean = series.dropna()
    if len(clean) < 2 * period:
        raise ValueError(
            f"周期 {period} を見るには最低 {2 * period} 点必要ですが、{len(clean)} 点しかありません。"
            "期間を延ばすか、周期を短くしてください。"
        )
    result = STL(clean, period=period, robust=robust).fit()
    return Decomposition(
        observed=clean,
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        period=period,
    )


def anomalies(decomposition: Decomposition, threshold: float = 3.0) -> pd.Series:
    """残差が大きく外れた点を返す。

    トレンドと季節を取り除いた残りが、いつもの散らばりから何倍離れているか。
    「7 月なのに寒い」のような、季節を考慮したうえでの異常を拾える。
    """
    residual = decomposition.residual
    # 標準偏差は外れ値自身に引っ張られるので、中央絶対偏差を使う
    median = residual.median()
    mad = (residual - median).abs().median()
    scale = mad * 1.4826 if mad > 0 else residual.std()
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(dtype=float)
    score = (residual - median) / scale
    return residual[score.abs() >= threshold]


# ======================================================================
# 自己相関
# ======================================================================


@dataclass
class Autocorrelation:
    """自己相関と偏自己相関。"""

    lags: np.ndarray
    acf_values: np.ndarray
    acf_low: np.ndarray
    acf_high: np.ndarray
    pacf_values: np.ndarray
    pacf_low: np.ndarray
    pacf_high: np.ndarray

    def significant_lags(self, max_report: int = 5) -> list[int]:
        """信頼区間を超えている（＝偶然とは言えない）ラグ。"""
        significant = [
            int(lag)
            for lag, value, low, high in zip(
                self.lags, self.acf_values, self.acf_low, self.acf_high
            )
            if lag > 0 and (value < low or value > high)
        ]
        return significant[:max_report]


def autocorrelation(series: pd.Series, n_lags: int = 60) -> Autocorrelation:
    """自己相関（ACF）と偏自己相関（PACF）を信頼区間つきで計算する。

    ACF は「k 日前の自分」との相関。PACF は途中のラグの影響を取り除いた分。
    ACF が緩やかに減るならトレンドあり、周期的に山が立つなら季節性あり。
    """
    clean = series.dropna()
    # PACF は系列長の半分未満のラグしか計算できない
    n_lags = int(min(n_lags, len(clean) // 2 - 1))
    if n_lags < 1:
        raise ValueError(f"自己相関を計算するにはデータが短すぎます（{len(clean)} 点）。")

    acf_values, acf_conf = acf(clean, nlags=n_lags, alpha=0.05)
    pacf_values, pacf_conf = pacf(clean, nlags=n_lags, alpha=0.05)

    acf_values = np.asarray(acf_values)
    pacf_values = np.asarray(pacf_values)

    # statsmodels の信頼区間は「推定値を中心とした区間」なので、そのままでは
    # 必ず推定値を含んでしまい有意判定に使えない。推定値を引いて
    # 「相関が無いとしたら値が収まるはずの帯」（0 中心）に直す。
    acf_band = np.asarray(acf_conf) - acf_values[:, None]
    pacf_band = np.asarray(pacf_conf) - pacf_values[:, None]

    return Autocorrelation(
        lags=np.arange(n_lags + 1),
        acf_values=acf_values,
        acf_low=acf_band[:, 0],
        acf_high=acf_band[:, 1],
        pacf_values=pacf_values,
        pacf_low=pacf_band[:, 0],
        pacf_high=pacf_band[:, 1],
    )


# ======================================================================
# 特徴量
# ======================================================================


@dataclass(frozen=True)
class FeatureConfig:
    """ラグ特徴量の設計。"""

    #: 何点前までを特徴量にするか
    n_lags: int = 7
    #: 移動平均・移動標準偏差の窓（空なら作らない）
    windows: tuple[int, ...] = (7, 28)
    #: 月・曜日などのカレンダー特徴を入れるか
    calendar: bool = True
    #: 何点先を予測するか
    horizon: int = 1


def make_features(series: pd.Series, config: FeatureConfig) -> tuple[pd.DataFrame, pd.Series]:
    """ラグ特徴量と目的変数を作る。

    **未来の情報を絶対に混ぜない**のがここでの唯一にして最大の約束。
    移動平均も `shift` してから計算し、「その時点で知りえた情報」だけにする。

    Returns:
        (X, y)。y は horizon 点先の値。特徴量が揃わない先頭は落とす。
    """
    clean = series.dropna()
    frame = pd.DataFrame(index=clean.index)

    for lag in range(1, config.n_lags + 1):
        frame[f"{lag}点前"] = clean.shift(lag)

    for window in config.windows:
        # shift(1) してから窓を取る。当日を含めると未来を覗くことになる
        past = clean.shift(1)
        frame[f"移動平均{window}"] = past.rolling(window).mean()
        frame[f"移動標準偏差{window}"] = past.rolling(window).std()

    if config.calendar:
        index = clean.index
        frame["月"] = index.month
        frame["曜日"] = index.dayofweek
        # 12 月と 1 月が「遠い」と誤解されないよう、周期を円で表す
        day_of_year = index.dayofyear
        frame["年内位置_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
        frame["年内位置_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    target = clean.shift(-config.horizon)

    combined = frame.join(target.rename("__target__")).dropna()
    return combined.drop(columns="__target__"), combined["__target__"]


# ======================================================================
# 予測とベースライン
# ======================================================================


@dataclass(frozen=True)
class Forecaster:
    """予測手法 1 つぶんの定義。"""

    key: str
    label: str
    summary: str
    #: 学習を伴わない単純な基準か
    is_baseline: bool = False
    build: Callable[[int], Any] | None = None


def _lightgbm(seed: int):
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=seed, verbose=-1,
    )


def _ridge(seed: int):
    return Ridge(alpha=1.0, random_state=seed)


FORECASTERS: dict[str, Forecaster] = {
    "naive": Forecaster(
        "naive", "前回と同じ（ナイーブ）",
        "1 点前の値をそのまま予測にする。学習は一切しない。"
        "株価のようなランダムウォークでは、これを超えるのが極めて難しい。",
        is_baseline=True,
    ),
    "seasonal_naive": Forecaster(
        "seasonal_naive", "1 周期前と同じ（季節ナイーブ）",
        "1 年前（周期ぶん前）の同じ時期の値をそのまま使う。季節性が強いデータの基準線。",
        is_baseline=True,
    ),
    "mean": Forecaster(
        "mean", "全期間の平均",
        "常に過去の平均を返すだけ。これを超えられないなら何も学習できていない。",
        is_baseline=True,
    ),
    "ridge": Forecaster(
        "ridge", "線形回帰（Ridge）",
        "ラグ特徴量に線形の重みを当てる。係数が読めるのが利点。",
        build=_ridge,
    ),
    "lightgbm": Forecaster(
        "lightgbm", "LightGBM",
        "ラグ特徴量を勾配ブースティングで学習する。非線形な関係も拾える。",
        build=_lightgbm,
    ),
}


def baseline_prediction(
    key: str, series: pd.Series, index: pd.Index, horizon: int, period: int
) -> np.ndarray:
    """学習しない基準線の予測を返す。

    Args:
        index: 予測したい時点の並び（`make_features` が返す X の index）。
    """
    clean = series.dropna()
    if key == "naive":
        # その時点で最後に観測できた値
        return clean.reindex(index).to_numpy()
    if key == "seasonal_naive":
        shifted = clean.shift(period - horizon)
        return shifted.reindex(index).to_numpy()
    if key == "mean":
        # 未来を覗かないよう、その時点までの累積平均を使う
        expanding = clean.expanding().mean()
        return expanding.reindex(index).to_numpy()
    raise ValueError(f"ベースラインではありません: {key}")


@dataclass
class BacktestResult:
    """時系列交差検証の結果。"""

    key: str
    label: str
    #: 分割ごとの指標
    scores: dict[str, list[float]] = field(default_factory=dict)
    #: 最後の分割での (実測, 予測, 日時)
    actual: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: pd.Index = field(default_factory=lambda: pd.Index([]))

    def mean(self, metric: str) -> float:
        values = [v for v in self.scores.get(metric, []) if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    def std(self, metric: str) -> float:
        values = [v for v in self.scores.get(metric, []) if np.isfinite(v)]
        return float(np.std(values)) if values else float("nan")


METRICS = ("R2", "RMSE", "MAE")


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() < 2:
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan")}
    a, p = actual[mask], predicted[mask]
    return {
        "R2": float(r2_score(a, p)),
        "RMSE": float(root_mean_squared_error(a, p)),
        "MAE": float(mean_absolute_error(a, p)),
    }


def backtest(
    key: str,
    series: pd.Series,
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    horizon: int = 1,
    period: int = 365,
    seed: int = 0,
) -> BacktestResult:
    """時系列交差検証で予測性能を測る。

    `TimeSeriesSplit` は常に「過去で学習し、未来で検証」する。
    普通の KFold を使うと未来のデータで学習して過去を当てることになり、
    実際にはあり得ない好成績が出る（リーク）。
    """
    forecaster = FORECASTERS[key]
    splitter = TimeSeriesSplit(n_splits=n_splits)
    result = BacktestResult(key=key, label=forecaster.label)
    result.scores = {metric: [] for metric in METRICS}

    for train_index, test_index in splitter.split(features):
        test_positions = features.index[test_index]
        actual = target.iloc[test_index].to_numpy(dtype=float)

        if forecaster.is_baseline:
            predicted = baseline_prediction(key, series, test_positions, horizon, period)
        else:
            model = forecaster.build(seed)
            model.fit(features.iloc[train_index], target.iloc[train_index])
            predicted = np.asarray(model.predict(features.iloc[test_index]), dtype=float)

        predicted = np.asarray(predicted, dtype=float)
        for metric, value in _score(actual, predicted).items():
            result.scores[metric].append(value)

        result.actual = actual
        result.predicted = predicted
        result.timestamps = test_positions

    return result


def split_ranges(features: pd.DataFrame, n_splits: int) -> list[dict[str, Any]]:
    """TimeSeriesSplit の各分割が、どの期間を使うかを返す（可視化用）。"""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    ranges = []
    for i, (train_index, test_index) in enumerate(splitter.split(features), start=1):
        ranges.append(
            {
                "分割": i,
                "訓練開始": features.index[train_index[0]],
                "訓練終了": features.index[train_index[-1]],
                "検証開始": features.index[test_index[0]],
                "検証終了": features.index[test_index[-1]],
                "訓練点数": len(train_index),
                "検証点数": len(test_index),
            }
        )
    return ranges


#: 予測期間を振るときの既定の刻み
HORIZON_SWEEP = (1, 3, 7, 14, 30, 60, 90)


def horizon_sweep(
    series: pd.Series,
    feature_config: FeatureConfig,
    horizons: tuple[int, ...] = HORIZON_SWEEP,
    keys: tuple[str, ...] = ("naive", "seasonal_naive", "ridge", "lightgbm"),
    n_splits: int = 5,
    period: int = 365,
    seed: int = 0,
) -> pd.DataFrame:
    """「何点先を当てるか」を変えながら各手法のスコアを測る。

    このラボで最も重要な図の材料。1 点先なら「前回と同じ」が最強だが、
    先を見るほど崩れ、季節性や学習したモデルが逆転する。
    どの手法が良いかは、予測したい期間によって変わる。
    """
    rows = []
    for horizon in horizons:
        config = FeatureConfig(
            n_lags=feature_config.n_lags,
            windows=feature_config.windows,
            calendar=feature_config.calendar,
            horizon=horizon,
        )
        features, target = make_features(series, config)
        if len(features) < n_splits * 2:
            continue
        for key in keys:
            result = backtest(
                key, series, features, target, n_splits, horizon, period, seed
            )
            rows.append(
                {
                    "予測期間": horizon,
                    "手法": FORECASTERS[key].label,
                    "key": key,
                    "R2": result.mean("R2"),
                    "RMSE": result.mean("RMSE"),
                    "MAE": result.mean("MAE"),
                }
            )
    return pd.DataFrame(rows)
