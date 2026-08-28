"""信号の周波数解析。

時間の並びを見ているだけでは分からないことが、周波数で見ると一目で分かる。
「どんな高さの音が、どれだけ混ざっているか」を取り出すのがフーリエ変換。

時系列ラボ（ラボ 9）が「過去の値から未来を予測する」話なのに対し、
ここは「いま観測している波が何でできているか」を分解する話。

計算はすべてここに置き、`app/views/lab11_imaging.py` は UI の組み立てだけにする。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

# ======================================================================
# 波を作る
# ======================================================================


@dataclass(frozen=True)
class Component:
    """波の成分 1 つぶん（正弦波）。"""

    frequency: float  # Hz
    amplitude: float


@dataclass(frozen=True)
class SignalConfig:
    """合成する信号の設定。ハッシュ可能なのでキャッシュのキーに使える。"""

    duration: float = 2.0  # 秒
    sample_rate: int = 500  # Hz
    components: tuple[Component, ...] = (
        Component(5.0, 1.0),
        Component(20.0, 0.5),
        Component(60.0, 0.25),
    )
    noise: float = 0.0
    seed: int = 0


def make_signal(config: SignalConfig) -> tuple[np.ndarray, np.ndarray]:
    """設定どおりの波を作る。

    Returns:
        (時刻の配列, 信号の配列)
    """
    n_samples = int(config.duration * config.sample_rate)
    time = np.arange(n_samples) / config.sample_rate
    values = np.zeros(n_samples)
    for component in config.components:
        values += component.amplitude * np.sin(2 * np.pi * component.frequency * time)
    if config.noise > 0:
        rng = np.random.default_rng(config.seed)
        values += rng.normal(0.0, config.noise, n_samples)
    return time, values


def make_chirp(config: SignalConfig, start_hz: float = 2.0, end_hz: float = 80.0):
    """時間とともに周波数が上がっていく波（チャープ）。

    FFT だけでは「いつその周波数が出たか」が分からないことを示すための素材。
    """
    n_samples = int(config.duration * config.sample_rate)
    time = np.arange(n_samples) / config.sample_rate
    values = sp_signal.chirp(time, f0=start_hz, t1=config.duration, f1=end_hz, method="linear")
    if config.noise > 0:
        rng = np.random.default_rng(config.seed)
        values = values + rng.normal(0.0, config.noise, n_samples)
    return time, values


# ======================================================================
# フーリエ変換
# ======================================================================


@dataclass
class Spectrum:
    """周波数ごとの強さ。"""

    frequencies: np.ndarray
    amplitudes: np.ndarray
    sample_rate: int

    @property
    def nyquist(self) -> float:
        """観測できる上限の周波数。標本化周波数の半分。"""
        return self.sample_rate / 2

    def peaks(self, top: int = 5, min_amplitude: float = 0.05) -> list[tuple[float, float]]:
        """目立つ山を返す。(周波数, 強さ) の並び。"""
        indices, _ = sp_signal.find_peaks(self.amplitudes, height=min_amplitude)
        if len(indices) == 0:
            return []
        order = np.argsort(self.amplitudes[indices])[::-1][:top]
        return [
            (float(self.frequencies[indices[i]]), float(self.amplitudes[indices[i]]))
            for i in order
        ]


def spectrum(values: np.ndarray, sample_rate: int) -> Spectrum:
    """信号を周波数成分に分解する（実数信号用の FFT）。

    振幅は「元の波に含まれる正弦波の振幅」と同じ尺度に直してある。
    生の FFT の出力はサンプル数に比例して大きくなるので、そのままでは
    設定した振幅と比べられない。
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    amplitudes = np.abs(np.fft.rfft(values)) * 2.0 / n
    if len(amplitudes):
        amplitudes[0] /= 2.0  # 直流成分は 2 倍しない
    return Spectrum(
        frequencies=np.fft.rfftfreq(n, d=1.0 / sample_rate),
        amplitudes=amplitudes,
        sample_rate=sample_rate,
    )


def spectrogram(
    values: np.ndarray, sample_rate: int, window: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """時間とともに周波数がどう変わるかを見る。

    短い窓を少しずつずらしながら FFT をかける。FFT が「全体で何が鳴っていたか」
    しか答えないのに対し、こちらは「いつ何が鳴ったか」が分かる。

    Returns:
        (周波数, 時刻, 強さの行列)
    """
    window = int(min(window, max(16, len(values) // 4)))
    frequencies, times, power = sp_signal.spectrogram(
        np.asarray(values, dtype=float),
        fs=sample_rate,
        nperseg=window,
        noverlap=window // 2,
        scaling="spectrum",
    )
    # 0 が混ざると対数が -inf になるので下限を置く
    decibels = 10 * np.log10(np.maximum(power, 1e-12))
    return frequencies, times, decibels


# ======================================================================
# フィルタ
# ======================================================================

FILTER_TYPES: dict[str, str] = {
    "none": "かけない",
    "lowpass": "ローパス（低い周波数だけ通す）",
    "highpass": "ハイパス（高い周波数だけ通す）",
    "bandpass": "バンドパス（ある範囲だけ通す）",
}


def apply_filter(
    values: np.ndarray,
    sample_rate: int,
    kind: str,
    low_hz: float = 10.0,
    high_hz: float = 50.0,
    order: int = 4,
) -> np.ndarray:
    """周波数でふるいにかける。

    バターワースフィルタを使い、位相のずれが出ないよう前後 2 回通す
    （`filtfilt`）。1 回だけだと波形が時間方向にずれてしまう。
    """
    if kind == "none":
        return np.asarray(values, dtype=float)

    nyquist = sample_rate / 2
    if kind == "lowpass":
        critical = np.clip(low_hz / nyquist, 1e-4, 0.999)
        b, a = sp_signal.butter(order, critical, btype="low")
    elif kind == "highpass":
        critical = np.clip(low_hz / nyquist, 1e-4, 0.999)
        b, a = sp_signal.butter(order, critical, btype="high")
    elif kind == "bandpass":
        low = np.clip(min(low_hz, high_hz) / nyquist, 1e-4, 0.998)
        high = np.clip(max(low_hz, high_hz) / nyquist, low + 1e-4, 0.999)
        b, a = sp_signal.butter(order, [low, high], btype="band")
    else:
        raise ValueError(f"未知のフィルタ: {kind}")

    return np.asarray(sp_signal.filtfilt(b, a, np.asarray(values, dtype=float)))


def aliasing_demo(
    true_hz: float, sample_rate: int, duration: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """標本化周波数が足りないと、波が別の周波数に化けることを示す。

    Returns:
        (細かい時刻, 真の波, 標本化した時刻, 標本化した値, 見かけの周波数)
    """
    dense_time = np.linspace(0, duration, 2000)
    dense_values = np.sin(2 * np.pi * true_hz * dense_time)

    n_samples = max(2, int(duration * sample_rate))
    sample_time = np.arange(n_samples) / sample_rate
    sample_values = np.sin(2 * np.pi * true_hz * sample_time)

    # ナイキスト周波数で折り返した先が、観測される見かけの周波数
    nyquist = sample_rate / 2
    folded = abs(((true_hz + nyquist) % sample_rate) - nyquist)
    return dense_time, dense_values, sample_time, sample_values, float(folded)
