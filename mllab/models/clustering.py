"""クラスタリングの補助ロジック。

k-means を 1 イテレーションずつ止められるようにするのが主目的。
sklearn の KMeans は途中経過を返さないので、反復部分だけ自前で書く。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import kmeans_plusplus


@dataclass(frozen=True)
class KMeansStep:
    """k-means の 1 イテレーションのスナップショット。"""

    iteration: int
    centers: np.ndarray  # (k, 2) 割り当て前の重心
    labels: np.ndarray  # (n,) この重心での割り当て
    moved_centers: np.ndarray  # (k, 2) 割り当て後に再計算した重心
    inertia: float  # 各点と担当重心の距離の二乗和
    shift: float  # 重心が動いた距離の合計


def _assign(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(d, axis=1)
    inertia = float(np.sum(np.min(d, axis=1) ** 2))
    return labels, inertia


def kmeans_history(
    X: np.ndarray,
    k: int,
    init: str = "k-means++",
    seed: int = 0,
    max_iter: int = 30,
    tol: float = 1e-6,
) -> list[KMeansStep]:
    """k-means の全イテレーションを記録して返す。

    Args:
        init: "k-means++" か "random"。初期化の良し悪しを比べるために切り替える。

    Returns:
        イテレーションごとの `KMeansStep` のリスト。収束したら打ち切る。
    """
    rng = np.random.default_rng(seed)
    if init == "k-means++":
        centers, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)
    else:
        centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    centers = np.asarray(centers, dtype=float)

    history: list[KMeansStep] = []
    for it in range(max_iter):
        labels, inertia = _assign(X, centers)
        moved = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                moved[c] = X[mask].mean(axis=0)
            # 空クラスタになったら、最も遠い点を新しい重心にする
            else:
                far = np.argmax(np.linalg.norm(X - centers[labels], axis=1))
                moved[c] = X[far]
        shift = float(np.linalg.norm(moved - centers, axis=1).sum())
        history.append(KMeansStep(it + 1, centers.copy(), labels, moved.copy(), inertia, shift))
        centers = moved
        if shift < tol:
            break
    return history


def elbow_curve(
    X: np.ndarray, k_max: int = 10, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """k を 1..k_max まで振ったときの inertia（クラスタ内平方和）。"""
    ks = np.arange(1, k_max + 1)
    values = []
    for k in ks:
        hist = kmeans_history(X, int(k), "k-means++", seed, max_iter=40)
        values.append(hist[-1].inertia)
    return ks, np.array(values)
