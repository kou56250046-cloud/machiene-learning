"""画像の処理と特徴量。

画像は「数値が縦横に並んだ行列」でしかない。そこに小さな行列（カーネル）を
滑らせながら掛け合わせるのが畳み込みで、これが CNN の中身そのもの。
ここでは自分でカーネルを組んで、何が起きるかを直接確かめられるようにする。

計算はすべてここに置き、`app/views/lab11_imaging.py` は UI の組み立てだけにする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.ndimage import convolve
from skimage import color, data, exposure, feature, filters, transform, util

# ======================================================================
# 画像の取得
# ======================================================================


@dataclass(frozen=True)
class SampleImage:
    """使えるサンプル画像 1 枚ぶん。"""

    key: str
    label: str
    loader: Callable[[], np.ndarray]
    note: str


#: scikit-image に同梱されている画像。通信不要でいつでも動く。
IMAGES: dict[str, SampleImage] = {
    "camera": SampleImage(
        "camera", "カメラマン（グレー）", data.camera,
        "定番のテスト画像。輪郭・平坦部・細かい模様が一枚に揃っています。",
    ),
    "coins": SampleImage(
        "coins", "硬貨（グレー）", data.coins,
        "丸い物体が並び、輪郭検出や領域分割の題材になります。",
    ),
    "page": SampleImage(
        "page", "印刷物（グレー）", data.page,
        "文字が並んだ画像。二値化の効果が分かりやすい素材です。",
    ),
    "brick": SampleImage(
        "brick", "レンガ（グレー）", data.brick,
        "規則的な模様。周期的な構造がフィルタでどう出るかを見られます。",
    ),
    "checkerboard": SampleImage(
        "checkerboard", "市松模様", data.checkerboard,
        "境界がはっきりしているので、エッジ検出の挙動が最も見やすい画像です。",
    ),
    "astronaut": SampleImage(
        "astronaut", "宇宙飛行士（カラー）", data.astronaut,
        "カラー画像。RGB の 3 枚が重なってできていることを確かめられます。",
    ),
    "chelsea": SampleImage(
        "chelsea", "猫（カラー）", data.chelsea,
        "毛並みの細かい模様が、ぼかしでどう失われるかが分かります。",
    ),
    "coffee": SampleImage(
        "coffee", "コーヒー（カラー）", data.coffee,
        "色の対比が強く、チャンネルごとの違いがはっきり出ます。",
    ),
}

#: 処理を軽くするための最大辺の長さ
MAX_SIZE = 320


def load_image(key: str, grayscale: bool = True, max_size: int = MAX_SIZE) -> np.ndarray:
    """サンプル画像を読み込み、扱いやすい大きさと形式に揃える。

    Returns:
        float64 の配列。値は 0〜1。グレースケールなら (H, W)、カラーなら (H, W, 3)。
    """
    if key not in IMAGES:
        raise ValueError(f"未知の画像: {key}")

    image = np.asarray(IMAGES[key].loader())
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    image = util.img_as_float(image)

    if grayscale and image.ndim == 3:
        image = color.rgb2gray(image)

    # 大きい画像は縮める。畳み込みは画素数に比例して重くなる
    longest = max(image.shape[:2])
    if longest > max_size:
        scale = max_size / longest
        image = transform.rescale(
            image, scale, anti_aliasing=True,
            channel_axis=2 if image.ndim == 3 else None,
        )
    return np.clip(image, 0.0, 1.0)


def is_color(key: str) -> bool:
    return np.asarray(IMAGES[key].loader()).ndim == 3


# ======================================================================
# 畳み込み
# ======================================================================


@dataclass(frozen=True)
class Kernel:
    """畳み込みカーネル 1 つぶん。"""

    key: str
    label: str
    matrix: tuple[tuple[float, ...], ...]
    note: str

    def as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)


KERNELS: dict[str, Kernel] = {
    "identity": Kernel(
        "identity", "そのまま（恒等）",
        ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
        "中心だけ 1。何も変わりません。比較の出発点です。",
    ),
    "blur": Kernel(
        "blur", "ぼかし（平均）",
        ((1 / 9, 1 / 9, 1 / 9), (1 / 9, 1 / 9, 1 / 9), (1 / 9, 1 / 9, 1 / 9)),
        "周囲 9 画素の平均を取ります。合計が 1 なので明るさは変わりません。",
    ),
    "gaussian": Kernel(
        "gaussian", "ぼかし（ガウス）",
        ((1 / 16, 2 / 16, 1 / 16), (2 / 16, 4 / 16, 2 / 16), (1 / 16, 2 / 16, 1 / 16)),
        "中心を重く見た平均。単純平均より自然にぼけます。",
    ),
    "sharpen": Kernel(
        "sharpen", "シャープ",
        ((0, -1, 0), (-1, 5, -1), (0, -1, 0)),
        "中心を強め、周囲を引きます。輪郭が際立ちます。",
    ),
    "edge": Kernel(
        "edge", "輪郭抽出",
        ((-1, -1, -1), (-1, 8, -1), (-1, -1, -1)),
        "合計が 0。平坦な場所は 0 になり、変化のある場所だけ残ります。",
    ),
    "sobel_x": Kernel(
        "sobel_x", "ソーベル（縦線）",
        ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)),
        "左右方向の変化を見ます。縦の輪郭が強く出ます。",
    ),
    "sobel_y": Kernel(
        "sobel_y", "ソーベル（横線）",
        ((-1, -2, -1), (0, 0, 0), (1, 2, 1)),
        "上下方向の変化を見ます。横の輪郭が強く出ます。",
    ),
    "emboss": Kernel(
        "emboss", "エンボス",
        ((-2, -1, 0), (-1, 1, 1), (0, 1, 2)),
        "斜め方向の差を取ると、浮き彫りのように見えます。",
    ),
}


def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """カーネルを画像に畳み込む。

    カラー画像は各チャンネルに同じカーネルを当てる。
    端は最も近い画素で埋める（黒で埋めると縁に偽の輪郭が出るため）。
    """
    if image.ndim == 3:
        channels = [convolve(image[..., c], kernel, mode="nearest") for c in range(image.shape[2])]
        result = np.stack(channels, axis=2)
    else:
        result = convolve(image, kernel, mode="nearest")
    return result


def normalise_for_display(array: np.ndarray) -> np.ndarray:
    """表示用に 0〜1 へ収める。

    エッジ抽出のように負の値が出るカーネルでは、そのままでは真っ黒になる。
    最小〜最大を 0〜1 に引き伸ばして見えるようにする。
    """
    lowest, highest = float(np.min(array)), float(np.max(array))
    if highest - lowest < 1e-12:
        return np.zeros_like(array)
    return (array - lowest) / (highest - lowest)


def kernel_stats(kernel: np.ndarray) -> dict[str, float]:
    """カーネルの性質。合計が 1 なら明るさ保存、0 なら差分検出。"""
    return {
        "合計": float(np.sum(kernel)),
        "絶対値の合計": float(np.sum(np.abs(kernel))),
        "最小": float(np.min(kernel)),
        "最大": float(np.max(kernel)),
    }


# ======================================================================
# エッジ検出
# ======================================================================

EDGE_METHODS: dict[str, str] = {
    "sobel": "ソーベル（勾配の大きさ）",
    "canny": "キャニー（細線化＋二値化）",
    "laplace": "ラプラシアン（2 階微分）",
}


def detect_edges(image: np.ndarray, method: str, sigma: float = 1.0) -> np.ndarray:
    """輪郭を取り出す。グレースケール画像を前提とする。"""
    gray = color.rgb2gray(image) if image.ndim == 3 else image
    if method == "sobel":
        return filters.sobel(gray)
    if method == "canny":
        return feature.canny(gray, sigma=sigma).astype(float)
    if method == "laplace":
        return np.abs(filters.laplace(gray))
    raise ValueError(f"未知のエッジ検出手法: {method}")


def threshold_image(image: np.ndarray, method: str = "otsu") -> tuple[np.ndarray, float]:
    """二値化する。しきい値も返す。

    Returns:
        (二値画像, しきい値)
    """
    gray = color.rgb2gray(image) if image.ndim == 3 else image
    value = float(filters.threshold_otsu(gray)) if method == "otsu" else float(np.mean(gray))
    return (gray > value).astype(float), value


# ======================================================================
# HOG 特徴量
# ======================================================================


@dataclass
class HogResult:
    """HOG（勾配方向ヒストグラム）の結果。"""

    features: np.ndarray
    visualization: np.ndarray
    n_features: int


def compute_hog(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: int = 16,
    cells_per_block: int = 2,
) -> HogResult:
    """画像を「どの向きの輪郭がどこにどれだけあるか」の数値に変える。

    小さな区画（セル）ごとに勾配の向きを集計する。深層学習が普及する前は、
    画像分類といえばこの手の手作り特徴量を分類器に渡すのが定番だった。
    """
    gray = color.rgb2gray(image) if image.ndim == 3 else image
    features, visualization = feature.hog(
        gray,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=True,
        feature_vector=True,
    )
    # そのままでは暗くて見えないので、コントラストを伸ばす
    visualization = exposure.rescale_intensity(visualization, in_range=(0, np.percentile(visualization, 99.5) or 1))
    return HogResult(
        features=np.asarray(features),
        visualization=np.asarray(visualization),
        n_features=int(np.size(features)),
    )


# ======================================================================
# 手書き数字の分類（古典特徴量 + 分類器）
# ======================================================================


def digit_images() -> tuple[np.ndarray, np.ndarray]:
    """手書き数字（8×8）を返す。ラボ 5 の次元削減と同じデータ。"""
    from sklearn.datasets import load_digits

    digits = load_digits()
    return np.asarray(digits.images), np.asarray(digits.target)


FEATURE_METHODS: dict[str, str] = {
    "raw": "生のピクセル値（64 次元）",
    "hog": "HOG 特徴量",
    "edges": "輪郭の強さ",
}


def digit_features(images: np.ndarray, method: str) -> np.ndarray:
    """数字画像を特徴量ベクトルにする。"""
    if method == "raw":
        return images.reshape(len(images), -1)

    if method == "hog":
        # 8×8 は小さいので、セルを 2×2 にして向きの情報を拾う
        return np.stack(
            [
                feature.hog(
                    img, orientations=8, pixels_per_cell=(2, 2),
                    cells_per_block=(2, 2), feature_vector=True,
                )
                for img in images
            ]
        )

    if method == "edges":
        return np.stack([filters.sobel(img).ravel() for img in images])

    raise ValueError(f"未知の特徴量: {method}")


def classify_digits(
    features: np.ndarray, targets: np.ndarray, n_splits: int = 5, seed: int = 0
) -> dict[str, Any]:
    """特徴量から数字を当てる。交差検証で予測を作る。"""
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    model = Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel="rbf", C=10, gamma="scale"))])
    predicted = cross_val_predict(model, features, targets, cv=n_splits)
    return {
        "predicted": np.asarray(predicted),
        "accuracy": float(np.mean(predicted == targets)),
        "n_features": int(features.shape[1]),
    }
