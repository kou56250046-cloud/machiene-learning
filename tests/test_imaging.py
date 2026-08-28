"""画像処理と信号解析のテスト。

どちらも「入れた通りのものが出てくるか」を、性質の分かっている入力で確かめる。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mllab.models import imaging as IM
from mllab.models import signals as SG

warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# 画像
# ======================================================================


@pytest.mark.parametrize("key", list(IM.IMAGES))
def test_every_sample_image_loads(key: str) -> None:
    image = IM.load_image(key, grayscale=True)
    assert image.ndim == 2
    assert image.dtype == float
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert max(image.shape) <= IM.MAX_SIZE


def test_colour_images_keep_three_channels() -> None:
    colour = [k for k in IM.IMAGES if IM.is_color(k)]
    assert colour, "カラー画像が 1 枚も無い"
    image = IM.load_image(colour[0], grayscale=False)
    assert image.ndim == 3 and image.shape[2] == 3


def test_grayscale_flag_collapses_channels() -> None:
    colour = next(k for k in IM.IMAGES if IM.is_color(k))
    assert IM.load_image(colour, grayscale=True).ndim == 2


def test_load_image_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="未知の画像"):
        IM.load_image("does_not_exist")


def test_large_images_are_shrunk() -> None:
    image = IM.load_image("camera", max_size=64)
    assert max(image.shape[:2]) <= 64


# ---- 畳み込み ---------------------------------------------------------


def test_identity_kernel_leaves_image_unchanged() -> None:
    image = IM.load_image("checkerboard")
    result = IM.apply_kernel(image, IM.KERNELS["identity"].as_array())
    np.testing.assert_allclose(result, image, atol=1e-10)


def test_blur_reduces_variation() -> None:
    """ぼかしは隣との差を小さくする。"""
    image = IM.load_image("checkerboard")
    blurred = IM.apply_kernel(image, IM.KERNELS["blur"].as_array())
    assert blurred.std() < image.std()


def test_edge_kernel_zeroes_flat_regions() -> None:
    """合計 0 のカーネルは、一様な面では 0 を返す。"""
    flat = np.full((32, 32), 0.5)
    result = IM.apply_kernel(flat, IM.KERNELS["edge"].as_array())
    assert np.abs(result).max() < 1e-10


def test_edge_kernel_responds_to_a_step() -> None:
    """段差があるところには値が残る。"""
    step = np.zeros((32, 32))
    step[:, 16:] = 1.0
    result = IM.apply_kernel(step, IM.KERNELS["edge"].as_array())
    assert np.abs(result).max() > 0.5
    # 段差から離れた場所は 0 のまま
    assert np.abs(result[:, :10]).max() < 1e-10


def test_sobel_kernels_pick_different_directions() -> None:
    """縦線用と横線用で、反応する向きが違うこと。"""
    vertical_edge = np.zeros((32, 32))
    vertical_edge[:, 16:] = 1.0

    x_response = np.abs(IM.apply_kernel(vertical_edge, IM.KERNELS["sobel_x"].as_array())).max()
    y_response = np.abs(IM.apply_kernel(vertical_edge, IM.KERNELS["sobel_y"].as_array())).max()
    assert x_response > y_response


def test_kernel_applies_to_every_colour_channel() -> None:
    image = IM.load_image("astronaut", grayscale=False)
    result = IM.apply_kernel(image, IM.KERNELS["blur"].as_array())
    assert result.shape == image.shape


@pytest.mark.parametrize("key", list(IM.KERNELS))
def test_kernel_stats(key: str) -> None:
    stats = IM.kernel_stats(IM.KERNELS[key].as_array())
    assert set(stats) == {"合計", "絶対値の合計", "最小", "最大"}
    assert stats["絶対値の合計"] >= abs(stats["合計"])


def test_blur_kernels_preserve_brightness() -> None:
    for key in ("blur", "gaussian", "identity"):
        assert IM.kernel_stats(IM.KERNELS[key].as_array())["合計"] == pytest.approx(1.0)


def test_edge_kernels_sum_to_zero() -> None:
    for key in ("edge", "sobel_x", "sobel_y"):
        assert IM.kernel_stats(IM.KERNELS[key].as_array())["合計"] == pytest.approx(0.0)


def test_normalise_for_display_maps_to_unit_range() -> None:
    array = np.array([[-3.0, 0.0], [1.0, 5.0]])
    result = IM.normalise_for_display(array)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_normalise_handles_constant_array() -> None:
    result = IM.normalise_for_display(np.full((4, 4), 2.0))
    assert np.all(result == 0.0)


# ---- エッジ検出 -------------------------------------------------------


@pytest.mark.parametrize("method", list(IM.EDGE_METHODS))
def test_edge_detection_shapes(method: str) -> None:
    image = IM.load_image("camera")
    edges = IM.detect_edges(image, method, sigma=2.0)
    assert edges.shape == image.shape
    assert np.isfinite(edges).all()


def test_edge_detection_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="未知のエッジ検出手法"):
        IM.detect_edges(IM.load_image("camera"), "nope")


def test_edges_are_stronger_on_a_step_than_on_flat() -> None:
    flat = np.full((64, 64), 0.5)
    step = np.zeros((64, 64))
    step[:, 32:] = 1.0
    assert IM.detect_edges(step, "sobel").max() > IM.detect_edges(flat, "sobel").max()


def test_threshold_splits_into_two_values() -> None:
    binary, value = IM.threshold_image(IM.load_image("page"))
    assert set(np.unique(binary)) <= {0.0, 1.0}
    assert 0.0 < value < 1.0


# ---- HOG と分類 -------------------------------------------------------


def test_hog_returns_features_and_visualization() -> None:
    image = IM.load_image("camera")
    result = IM.compute_hog(image, orientations=9, pixels_per_cell=16, cells_per_block=2)
    assert result.n_features > 0
    assert result.features.ndim == 1
    assert result.visualization.shape == image.shape


def test_hog_dimension_grows_as_cells_shrink() -> None:
    image = IM.load_image("camera")
    coarse = IM.compute_hog(image, 9, 32, 2).n_features
    fine = IM.compute_hog(image, 9, 16, 2).n_features
    assert fine > coarse


def test_digit_images_shape() -> None:
    images, targets = IM.digit_images()
    assert images.shape[1:] == (8, 8)
    assert len(images) == len(targets)
    assert set(np.unique(targets)) == set(range(10))


@pytest.mark.parametrize("method", list(IM.FEATURE_METHODS))
def test_digit_features_are_two_dimensional(method: str) -> None:
    images, _ = IM.digit_images()
    features = IM.digit_features(images[:100], method)
    assert features.ndim == 2
    assert len(features) == 100
    assert np.isfinite(features).all()


def test_digit_features_rejects_unknown() -> None:
    images, _ = IM.digit_images()
    with pytest.raises(ValueError, match="未知の特徴量"):
        IM.digit_features(images[:10], "nope")


def test_digit_classification_beats_chance() -> None:
    """10 クラスなので、でたらめなら 0.1。それを大きく超えること。"""
    images, targets = IM.digit_images()
    features = IM.digit_features(images, "raw")
    result = IM.classify_digits(features, targets, n_splits=3, seed=0)
    assert result["accuracy"] > 0.85
    assert result["n_features"] == 64


# ======================================================================
# 信号
# ======================================================================


def test_signal_length_matches_configuration() -> None:
    config = SG.SignalConfig(duration=2.0, sample_rate=250)
    time, values = SG.make_signal(config)
    assert len(time) == len(values) == 500
    assert time[-1] == pytest.approx(2.0 - 1 / 250)


def test_fft_recovers_the_configured_frequencies() -> None:
    """入れた周波数と振幅が、そのまま山として出てくること。"""
    components = (SG.Component(5.0, 1.0), SG.Component(20.0, 0.5), SG.Component(60.0, 0.25))
    config = SG.SignalConfig(duration=4.0, sample_rate=500, components=components)
    _, values = SG.make_signal(config)

    peaks = SG.spectrum(values, 500).peaks(top=5)
    found = {round(frequency): amplitude for frequency, amplitude in peaks}

    for component in components:
        key = round(component.frequency)
        assert key in found, f"{component.frequency}Hz の山が見つからない"
        assert found[key] == pytest.approx(component.amplitude, abs=0.05)


def test_spectrum_nyquist_is_half_the_sample_rate() -> None:
    _, values = SG.make_signal(SG.SignalConfig(sample_rate=400))
    spectrum = SG.spectrum(values, 400)
    assert spectrum.nyquist == 200
    assert spectrum.frequencies.max() == pytest.approx(200)


def test_noise_raises_the_floor_between_peaks() -> None:
    clean = SG.SignalConfig(duration=4.0, sample_rate=500, noise=0.0)
    noisy = SG.SignalConfig(duration=4.0, sample_rate=500, noise=0.8, seed=0)
    _, clean_values = SG.make_signal(clean)
    _, noisy_values = SG.make_signal(noisy)

    # 成分が無い高い周波数帯の平均振幅で比べる
    clean_spectrum = SG.spectrum(clean_values, 500)
    noisy_spectrum = SG.spectrum(noisy_values, 500)
    band = clean_spectrum.frequencies > 100
    assert noisy_spectrum.amplitudes[band].mean() > clean_spectrum.amplitudes[band].mean() * 3


# ---- フィルタ ---------------------------------------------------------


def test_lowpass_removes_the_high_component() -> None:
    components = (SG.Component(5.0, 1.0), SG.Component(80.0, 1.0))
    _, values = SG.make_signal(
        SG.SignalConfig(duration=4.0, sample_rate=500, components=components)
    )
    filtered = SG.apply_filter(values, 500, "lowpass", low_hz=20.0)
    spectrum = SG.spectrum(filtered, 500)

    def amplitude_at(hz: float) -> float:
        index = int(np.argmin(np.abs(spectrum.frequencies - hz)))
        return float(spectrum.amplitudes[index])

    assert amplitude_at(5.0) > 0.8
    assert amplitude_at(80.0) < 0.1


def test_highpass_removes_the_low_component() -> None:
    components = (SG.Component(5.0, 1.0), SG.Component(80.0, 1.0))
    _, values = SG.make_signal(
        SG.SignalConfig(duration=4.0, sample_rate=500, components=components)
    )
    filtered = SG.apply_filter(values, 500, "highpass", low_hz=30.0)
    spectrum = SG.spectrum(filtered, 500)
    low = spectrum.amplitudes[np.argmin(np.abs(spectrum.frequencies - 5.0))]
    high = spectrum.amplitudes[np.argmin(np.abs(spectrum.frequencies - 80.0))]
    assert high > low * 5


def test_bandpass_keeps_only_the_middle() -> None:
    components = (SG.Component(3.0, 1.0), SG.Component(30.0, 1.0), SG.Component(120.0, 1.0))
    _, values = SG.make_signal(
        SG.SignalConfig(duration=4.0, sample_rate=500, components=components)
    )
    filtered = SG.apply_filter(values, 500, "bandpass", low_hz=20.0, high_hz=45.0)
    spectrum = SG.spectrum(filtered, 500)

    def amplitude_at(hz: float) -> float:
        return float(spectrum.amplitudes[np.argmin(np.abs(spectrum.frequencies - hz))])

    assert amplitude_at(30.0) > 0.7
    assert amplitude_at(3.0) < 0.15
    assert amplitude_at(120.0) < 0.15


def test_filter_none_is_a_passthrough() -> None:
    _, values = SG.make_signal(SG.SignalConfig())
    np.testing.assert_allclose(SG.apply_filter(values, 500, "none"), values)


def test_filter_rejects_unknown_kind() -> None:
    _, values = SG.make_signal(SG.SignalConfig())
    with pytest.raises(ValueError, match="未知のフィルタ"):
        SG.apply_filter(values, 500, "nope")


# ---- スペクトログラム -------------------------------------------------


def test_spectrogram_tracks_a_rising_chirp() -> None:
    """周波数が上がる波では、ピークも時間とともに上がること。"""
    config = SG.SignalConfig(duration=4.0, sample_rate=500)
    _, values = SG.make_chirp(config, start_hz=5.0, end_hz=100.0)
    frequencies, times, decibels = SG.spectrogram(values, 500, window=128)

    assert decibels.shape == (len(frequencies), len(times))
    start_peak = frequencies[np.argmax(decibels[:, 0])]
    end_peak = frequencies[np.argmax(decibels[:, -1])]
    assert end_peak > start_peak * 3


def test_spectrogram_window_is_clamped_to_series_length() -> None:
    _, values = SG.make_signal(SG.SignalConfig(duration=0.2, sample_rate=200))
    frequencies, times, decibels = SG.spectrogram(values, 200, window=4096)
    assert decibels.shape == (len(frequencies), len(times))
    assert len(times) >= 1


# ---- エイリアシング ---------------------------------------------------


@pytest.mark.parametrize(
    ("true_hz", "sample_rate", "expected"),
    [
        (40.0, 500, 40.0),   # 十分速く測っていれば正しく見える
        (40.0, 100, 40.0),   # ナイキスト 50Hz なのでぎりぎり足りる
        (40.0, 50, 10.0),    # ナイキスト 25Hz を超えると折り返す
        (40.0, 30, 10.0),
    ],
)
def test_aliasing_folds_above_nyquist(true_hz: float, sample_rate: int, expected: float) -> None:
    *_, apparent = SG.aliasing_demo(true_hz, sample_rate)
    assert apparent == pytest.approx(expected, abs=0.01)


def test_aliasing_demo_returns_consistent_arrays() -> None:
    dense_t, dense_v, sample_t, sample_v, _ = SG.aliasing_demo(30.0, 100, duration=0.5)
    assert len(dense_t) == len(dense_v)
    assert len(sample_t) == len(sample_v)
    assert len(sample_t) < len(dense_t)
