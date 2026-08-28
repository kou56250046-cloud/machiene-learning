"""ラボで使う合成データの生成。

すべての生成関数は `(X, y)` を返す。X は shape (n_samples, 2) の float 配列、
y は shape (n_samples,) の int 配列。`seed` を固定すれば完全に再現できる。
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

#: データ形状のキー → 画面に出す日本語名
DATASETS: dict[str, str] = {
    "moons": "三日月 (moons)",
    "circles": "同心円 (circles)",
    "blobs": "ガウス塊 (blobs)",
    "xor": "市松 (XOR)",
    "spirals": "らせん (spirals)",
    "aniso": "引き伸ばし (anisotropic)",
    "varied": "分散がばらつく塊 (varied)",
}

#: 線形分離できないデータ（解説で「直線では引けない」と示すのに使う）
NONLINEAR = ("moons", "circles", "xor", "spirals")


def _xor(n_samples: int, noise: float, rng: np.random.Generator):
    X = rng.uniform(-3.0, 3.0, size=(n_samples, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += rng.normal(0.0, noise * 3.0, size=X.shape)
    return X, y


def _spirals(n_samples: int, noise: float, rng: np.random.Generator, n_classes: int):
    per = n_samples // n_classes
    xs, ys = [], []
    for c in range(n_classes):
        t = np.sqrt(rng.uniform(0.06, 1.0, per)) * 2.6 * np.pi
        offset = 2.0 * np.pi * c / n_classes
        r = t
        xs.append(np.c_[r * np.cos(t + offset), r * np.sin(t + offset)])
        ys.append(np.full(per, c))
    X = np.vstack(xs)
    X += rng.normal(0.0, noise * 4.0, size=X.shape)
    return X, np.concatenate(ys)


def _aniso(n_samples: int, noise: float, rng: np.random.Generator, n_classes: int, seed: int):
    X, y = make_blobs(
        n_samples=n_samples, centers=n_classes, cluster_std=0.6 + noise * 2, random_state=seed
    )
    # 一様に引き伸ばして k-means が苦手な形にする
    X = X @ np.array([[0.6, -0.63], [-0.41, 0.85]])
    return X, y


def make_dataset(
    kind: str = "moons",
    n_samples: int = 300,
    noise: float = 0.2,
    seed: int = 0,
    n_classes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """2 次元の合成データセットを作る。

    Args:
        kind: `DATASETS` のキー。
        n_samples: 総サンプル数。
        noise: 0.0〜1.0 のノイズ量。形状ごとに妥当なスケールへ換算する。
        seed: 乱数シード。
        n_classes: blobs / spirals / aniso / varied でのクラス数。

    Returns:
        (X, y)。X は float64 の (n, 2)、y は int の (n,)。
    """
    rng = np.random.default_rng(seed)

    if kind == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise * 0.6, random_state=seed)
    elif kind == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=noise * 0.35, factor=0.5, random_state=seed
        )
    elif kind == "blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=0.6 + noise * 2.0,
            random_state=seed,
        )
    elif kind == "xor":
        X, y = _xor(n_samples, noise, rng)
    elif kind == "spirals":
        X, y = _spirals(n_samples, noise, rng, n_classes)
    elif kind == "aniso":
        X, y = _aniso(n_samples, noise, rng, n_classes, seed)
    elif kind == "varied":
        stds = np.linspace(0.4, 1.6, n_classes) * (0.5 + noise * 1.5)
        X, y = make_blobs(
            n_samples=n_samples, centers=n_classes, cluster_std=stds, random_state=seed
        )
    else:
        raise ValueError(f"未知のデータ形状: {kind}")

    X = np.asarray(X, dtype=float)
    # ラボ間でスケールを揃えると、同じ eps / gamma の感覚が使い回せる
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    return X, np.asarray(y, dtype=int)


# --- 1 次元回帰用（過学習ラボ） --------------------------------------

TRUE_FUNCTIONS: dict[str, str] = {
    "sine": "サイン波",
    "step": "階段",
    "poly": "3 次多項式",
}


def true_function(kind: str, x: np.ndarray) -> np.ndarray:
    """回帰ラボの「真の関数」。ノイズを乗せる前の理想曲線。"""
    if kind == "sine":
        return np.sin(2.0 * np.pi * x)
    if kind == "step":
        return np.where(x < 0.35, -0.6, np.where(x < 0.7, 0.7, -0.2))
    if kind == "poly":
        return 8.0 * (x - 0.2) * (x - 0.55) * (x - 0.9)
    raise ValueError(f"未知の関数: {kind}")


def make_regression_1d(
    kind: str = "sine", n_samples: int = 40, noise: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """1 次元の回帰データ。x は [0, 1] 区間。"""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n_samples))
    y = true_function(kind, x) + rng.normal(0.0, noise, n_samples)
    return x.reshape(-1, 1), y
