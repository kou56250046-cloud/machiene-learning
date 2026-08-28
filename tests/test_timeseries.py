"""時系列の分解・自己相関・特徴量・予測のテスト。

外部データに依存しないよう、性質の分かっている系列を自分で合成して検証する。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mllab.models import timeseries as TS

warnings.filterwarnings("ignore", category=UserWarning)


# ---- 検証用の系列 -----------------------------------------------------


def seasonal_series(n: int = 1200, period: int = 365, seed: int = 0) -> pd.Series:
    """トレンド + 季節 + ノイズ。気温のような系列。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 15 + 0.002 * t + 10 * np.sin(2 * np.pi * t / period) + rng.normal(0, 1.0, n)
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=n, freq="D"))


def random_walk(n: int = 1200, seed: int = 0) -> pd.Series:
    """ランダムウォーク。株価のような、変化が予測できない系列。"""
    rng = np.random.default_rng(seed)
    return pd.Series(
        100 + np.cumsum(rng.normal(0, 1, n)),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


def white_noise(n: int = 1200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0, 1, n), index=pd.date_range("2020-01-01", periods=n, freq="D")
    )


def frame_from(series: pd.Series, name: str = "値") -> pd.DataFrame:
    return pd.DataFrame({"日付": series.index, name: series.to_numpy()})


# ---- 系列の切り出し ---------------------------------------------------


def test_find_columns() -> None:
    frame = frame_from(seasonal_series(50))
    frame["ラベル"] = "a"
    frame["フラグ"] = True
    assert TS.find_time_columns(frame) == ["日付"]
    assert TS.find_value_columns(frame, "日付") == ["値"]


def test_build_series_is_sorted_and_regular() -> None:
    original = seasonal_series(200)
    shuffled = frame_from(original).sample(frac=1.0, random_state=0)
    series = TS.build_series(shuffled, TS.SeriesConfig("日付", "値", "D", "mean"))
    assert series.index.is_monotonic_increasing
    assert len(series) == len(original)


def test_build_series_forward_fills_gaps() -> None:
    """休場日は直前の値で埋める。未来の値を使わないこと。"""
    original = seasonal_series(60)
    frame = frame_from(original).drop(index=[10, 11, 12])
    series = TS.build_series(frame, TS.SeriesConfig("日付", "値", "D", "mean"))
    assert len(series) == 60  # 等間隔に戻っている
    # 埋めた値は「直前の値」であって、後ろの値ではない
    assert series.iloc[10] == pytest.approx(series.iloc[9])
    assert series.iloc[12] == pytest.approx(series.iloc[9])


def test_gaps_in_series_counts_missing_points() -> None:
    frame = frame_from(seasonal_series(60)).drop(index=[10, 11, 12])
    assert TS.gaps_in_series(frame, TS.SeriesConfig("日付", "値", "D", "mean")) == 3


@pytest.mark.parametrize("frequency", list(TS.FREQUENCIES))
def test_resampling_reduces_points(frequency: str) -> None:
    frame = frame_from(seasonal_series(400))
    series = TS.build_series(frame, TS.SeriesConfig("日付", "値", frequency, "mean"))
    assert len(series) <= 400
    assert series.index.is_monotonic_increasing


# ---- 分解 -------------------------------------------------------------


def test_decompose_recovers_components() -> None:
    """トレンド + 季節 + 残差 を足すと元の系列に戻る。"""
    series = seasonal_series(1200, period=365)
    decomposition = TS.decompose(series, period=365)
    reconstructed = (
        decomposition.trend + decomposition.seasonal + decomposition.residual
    )
    np.testing.assert_allclose(
        reconstructed.to_numpy(), decomposition.observed.to_numpy(), atol=1e-8
    )


def test_seasonal_series_has_strong_seasonality() -> None:
    decomposition = TS.decompose(seasonal_series(1200, period=365), period=365)
    assert decomposition.seasonal_strength > 0.8
    # 振幅は仕込んだ ±10 に近いはず
    assert 15 < decomposition.seasonal_amplitude < 25


def test_random_walk_is_dominated_by_trend_not_season() -> None:
    """ランダムウォークはトレンドで動き、季節性はほとんど無い。

    STL は短い系列だと見かけの季節性を拾うことがあるため、絶対値ではなく
    「本物の季節系列と比べて明確に弱い」ことで判定する。
    """
    seasonal_strength = TS.decompose(
        seasonal_series(1500, period=365), period=365
    ).seasonal_strength

    walks = [TS.decompose(random_walk(1500, seed=s), period=365) for s in range(4)]
    walk_seasonal = float(np.mean([d.seasonal_strength for d in walks]))
    walk_trend = float(np.mean([d.trend_strength for d in walks]))

    assert walk_seasonal < seasonal_strength - 0.4
    assert walk_trend > walk_seasonal


def test_decompose_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="点しかありません"):
        TS.decompose(seasonal_series(100), period=365)


@pytest.mark.parametrize(
    ("frequency", "expected"), [("D", 365), ("W", 52), ("ME", 12)]
)
def test_suggest_period(frequency: str, expected: int) -> None:
    assert TS.suggest_period(frequency) == expected


def test_anomalies_finds_injected_spike() -> None:
    series = seasonal_series(1200, period=365)
    series.iloc[600] += 40  # 明らかな異常値を 1 点入れる
    decomposition = TS.decompose(series, period=365)
    found = TS.anomalies(decomposition, threshold=3.0)
    assert series.index[600] in found.index


def test_anomaly_threshold_controls_count() -> None:
    decomposition = TS.decompose(seasonal_series(1200, period=365), period=365)
    loose = TS.anomalies(decomposition, threshold=2.0)
    strict = TS.anomalies(decomposition, threshold=5.0)
    assert len(loose) >= len(strict)


# ---- 自己相関 ---------------------------------------------------------


def test_acf_band_is_centred_on_zero() -> None:
    """信頼区間は「相関が無いとしたら」の帯。推定値中心のままでは有意判定できない。"""
    series = seasonal_series(1200, period=365)
    ac = TS.autocorrelation(series, n_lags=40)
    # 帯はほぼ 0 対称で、理論値 ±1.96/√n に近い
    assert ac.acf_low[1] < 0 < ac.acf_high[1]
    expected = 1.96 / np.sqrt(len(series))
    assert ac.acf_high[1] == pytest.approx(expected, rel=0.1)


def test_seasonal_series_has_significant_lags() -> None:
    ac = TS.autocorrelation(seasonal_series(1200, period=365), n_lags=40)
    assert ac.significant_lags(5), "季節性のある系列で有意なラグが検出できていない"


def test_white_noise_has_no_significant_lags() -> None:
    ac = TS.autocorrelation(white_noise(2000), n_lags=40)
    # 5% 水準なので 40 ラグ中 2 個程度は偶然出る
    assert len(ac.significant_lags(100)) <= 4


def test_autocorrelation_shapes() -> None:
    ac = TS.autocorrelation(seasonal_series(600), n_lags=30)
    assert len(ac.lags) == 31
    for array in (ac.acf_values, ac.acf_low, ac.acf_high,
                  ac.pacf_values, ac.pacf_low, ac.pacf_high):
        assert len(array) == 31


def test_autocorrelation_caps_lags_to_series_length() -> None:
    ac = TS.autocorrelation(seasonal_series(60), n_lags=500)
    assert len(ac.lags) <= 31  # 系列長の半分未満に切り詰められる


def test_autocorrelation_rejects_tiny_series() -> None:
    with pytest.raises(ValueError, match="短すぎます"):
        TS.autocorrelation(seasonal_series(3), n_lags=10)


# ---- 特徴量 -----------------------------------------------------------


def test_make_features_creates_expected_columns() -> None:
    series = seasonal_series(400)
    X, y = TS.make_features(
        series, TS.FeatureConfig(n_lags=3, windows=(7,), calendar=True, horizon=1)
    )
    assert [c for c in X.columns if c.endswith("点前")] == ["1点前", "2点前", "3点前"]
    assert "移動平均7" in X.columns and "移動標準偏差7" in X.columns
    assert {"月", "曜日", "年内位置_sin", "年内位置_cos"} <= set(X.columns)
    assert len(X) == len(y)
    assert X.notna().all().all()


def test_lag_features_hold_past_values() -> None:
    """1点前の特徴量が、本当に 1 点前の観測値であること。"""
    series = pd.Series(
        np.arange(100, dtype=float),
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )
    X, y = TS.make_features(
        series, TS.FeatureConfig(n_lags=2, windows=(), calendar=False, horizon=1)
    )
    first = X.index[0]
    assert X.loc[first, "1点前"] == pytest.approx(series.loc[first] - 1)
    assert X.loc[first, "2点前"] == pytest.approx(series.loc[first] - 2)


def test_target_is_horizon_steps_ahead() -> None:
    series = pd.Series(
        np.arange(100, dtype=float),
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )
    for horizon in (1, 5, 10):
        X, y = TS.make_features(
            series, TS.FeatureConfig(n_lags=1, windows=(), calendar=False, horizon=horizon)
        )
        timestamp = X.index[0]
        assert y.loc[timestamp] == pytest.approx(series.loc[timestamp] + horizon)


def test_rolling_features_do_not_peek_at_the_present() -> None:
    """移動平均は shift してから計算する。当日を含めると未来を覗くことになる。"""
    series = pd.Series(
        np.arange(1, 101, dtype=float),
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )
    X, _ = TS.make_features(
        series, TS.FeatureConfig(n_lags=1, windows=(3,), calendar=False, horizon=1)
    )
    timestamp = X.index[0]
    current = series.loc[timestamp]
    # 当日を含めない過去 3 点の平均 = current-1, current-2, current-3 の平均
    assert X.loc[timestamp, "移動平均3"] == pytest.approx(current - 2)


def test_calendar_features_wrap_around_the_year() -> None:
    """12/31 と 1/1 が円周上で隣り合うこと。"""
    series = seasonal_series(800)
    X, _ = TS.make_features(
        series, TS.FeatureConfig(n_lags=1, windows=(), calendar=True, horizon=1)
    )
    december = X[(X.index.month == 12) & (X.index.day == 31)]
    january = X[(X.index.month == 1) & (X.index.day == 1)]
    if len(december) and len(january):
        distance = np.hypot(
            december["年内位置_sin"].iloc[0] - january["年内位置_sin"].iloc[0],
            december["年内位置_cos"].iloc[0] - january["年内位置_cos"].iloc[0],
        )
        assert distance < 0.1


# ---- 予測 -------------------------------------------------------------


def test_naive_baseline_repeats_last_value() -> None:
    series = pd.Series(
        np.arange(100, dtype=float),
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )
    X, _ = TS.make_features(
        series, TS.FeatureConfig(n_lags=1, windows=(), calendar=False, horizon=1)
    )
    predicted = TS.baseline_prediction("naive", series, X.index[:5], horizon=1, period=365)
    np.testing.assert_allclose(predicted, series.reindex(X.index[:5]).to_numpy())


def test_seasonal_naive_looks_one_period_back() -> None:
    series = seasonal_series(1200, period=365)
    X, y = TS.make_features(
        series, TS.FeatureConfig(n_lags=1, windows=(), calendar=False, horizon=1)
    )
    predicted = TS.baseline_prediction(
        "seasonal_naive", series, X.index, horizon=1, period=365
    )
    finite = np.isfinite(predicted)
    # 季節性のある系列なら、1 周期前の値はそれなりに当たる
    assert np.corrcoef(predicted[finite], y.to_numpy()[finite])[0, 1] > 0.7


def test_baseline_prediction_rejects_learned_model() -> None:
    with pytest.raises(ValueError, match="ベースラインではありません"):
        TS.baseline_prediction("lightgbm", seasonal_series(100), pd.Index([]), 1, 365)


@pytest.mark.parametrize("key", list(TS.FORECASTERS))
def test_every_forecaster_runs(key: str) -> None:
    series = seasonal_series(900, period=365)
    X, y = TS.make_features(series, TS.FeatureConfig(7, (7,), True, 1))
    result = TS.backtest(key, series, X, y, n_splits=3, horizon=1, period=365, seed=0)
    assert result.key == key
    assert len(result.scores["R2"]) == 3
    assert len(result.actual) == len(result.predicted) == len(result.timestamps)


def test_learned_model_beats_mean_on_seasonal_data() -> None:
    series = seasonal_series(1200, period=365)
    X, y = TS.make_features(series, TS.FeatureConfig(7, (7,), True, 1))
    learned = TS.backtest("ridge", series, X, y, 3, 1, 365, 0)
    mean = TS.backtest("mean", series, X, y, 3, 1, 365, 0)
    assert learned.mean("R2") > mean.mean("R2") + 0.5


def test_nothing_predicts_random_walk_changes() -> None:
    """変化率は原理的に予測できない。どの手法も R² がほぼ 0 になる。"""
    changes = random_walk(1500).diff().dropna()
    X, y = TS.make_features(changes, TS.FeatureConfig(7, (7,), False, 1))
    for key in ("naive", "ridge", "lightgbm"):
        result = TS.backtest(key, changes, X, y, 3, 1, 365, 0)
        assert result.mean("R2") < 0.15, f"{key} が予測できてしまっている"


def test_split_ranges_never_train_on_the_future() -> None:
    """時系列交差検証の要。訓練期間は必ず検証期間より前にある。"""
    series = seasonal_series(1000)
    X, _ = TS.make_features(series, TS.FeatureConfig(3, (), False, 1))
    ranges = TS.split_ranges(X, n_splits=5)
    assert len(ranges) == 5
    for row in ranges:
        assert row["訓練終了"] < row["検証開始"]
    # 分割が進むほど訓練データが増える
    sizes = [row["訓練点数"] for row in ranges]
    assert sizes == sorted(sizes)


def test_backtest_result_aggregates_scores() -> None:
    result = TS.BacktestResult(key="x", label="X")
    result.scores = {"R2": [0.8, 0.9, float("nan")]}
    assert result.mean("R2") == pytest.approx(0.85)
    assert np.isfinite(result.std("R2"))
    assert np.isnan(result.mean("存在しない指標"))


def test_horizon_sweep_shape_and_ordering() -> None:
    series = seasonal_series(1200, period=365)
    sweep = TS.horizon_sweep(
        series, TS.FeatureConfig(7, (7,), True, 1),
        horizons=(1, 7, 30), keys=("naive", "ridge"), n_splits=3, period=365,
    )
    assert set(sweep["予測期間"]) == {1, 7, 30}
    assert set(sweep["key"]) == {"naive", "ridge"}
    assert len(sweep) == 6
    assert {"R2", "RMSE", "MAE"} <= set(sweep.columns)


def test_naive_degrades_as_horizon_grows() -> None:
    """遠い未来ほど「前回と同じ」は当たらなくなる。ラボの核心の主張。"""
    series = seasonal_series(1500, period=365)
    sweep = TS.horizon_sweep(
        series, TS.FeatureConfig(7, (7,), True, 1),
        horizons=(1, 90), keys=("naive",), n_splits=3, period=365,
    )
    short = sweep[sweep["予測期間"] == 1]["R2"].iloc[0]
    long = sweep[sweep["予測期間"] == 90]["R2"].iloc[0]
    assert short > long


def test_seasonal_naive_holds_across_horizons() -> None:
    """1 周期前を見るだけなので、予測期間の長さに影響されない。"""
    series = seasonal_series(1500, period=365)
    sweep = TS.horizon_sweep(
        series, TS.FeatureConfig(7, (7,), True, 1),
        horizons=(1, 90), keys=("seasonal_naive",), n_splits=3, period=365,
    )
    scores = sweep["R2"].to_numpy()
    assert abs(scores[0] - scores[1]) < 0.15
