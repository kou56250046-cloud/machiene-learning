"""分類モデルの定義集。

UI からは「モデル名」と「ハイパーパラメータの辞書」だけを渡す。
どんなスライダを出すかも `PARAM_SPECS` としてここに持たせ、
ページ側はそれを機械的に描画するだけにする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ParamSpec:
    """スライダ 1 本ぶんの定義。"""

    key: str
    label: str
    kind: str  # "int" | "float" | "log" | "select"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[Any, ...] = ()
    help: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """モデル 1 種類ぶんの定義。"""

    key: str
    label: str
    build: Callable[..., BaseEstimator]
    params: tuple[ParamSpec, ...] = ()
    summary: str = ""
    #: 決定関数が滑らかか（解説の出し分けに使う）
    smooth: bool = True

    def create(self, values: dict[str, Any]) -> BaseEstimator:
        """スライダの値からモデルを組み立てる。未知キーは無視する。"""
        allowed = {p.key for p in self.params}
        return self.build(**{k: v for k, v in values.items() if k in allowed})

    def defaults(self) -> dict[str, Any]:
        return {p.key: p.default for p in self.params}


def _scaled(estimator: BaseEstimator) -> Pipeline:
    """距離やマージンを使うモデルは標準化とセットにする。"""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


CLASSIFIERS: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        key="logreg",
        label="ロジスティック回帰",
        build=lambda C=1.0: _scaled(LogisticRegression(C=C, max_iter=2000)),
        params=(
            ParamSpec("C", "正則化の緩さ C", "log", 1.0, -3, 3,
                      help="小さいほど強く正則化され、境界が単純になります。"),
        ),
        summary="直線（超平面）でしか境界を引けない、最もシンプルな分類器。",
    ),
    "knn": ModelSpec(
        key="knn",
        label="k 近傍法 (kNN)",
        build=lambda n_neighbors=5, weights="uniform": _scaled(
            KNeighborsClassifier(n_neighbors=int(n_neighbors), weights=weights)
        ),
        params=(
            ParamSpec("n_neighbors", "近傍数 k", "int", 5, 1, 60, 1,
                      help="k を上げるほど境界が滑らかになり、下げるほどノイズに食いつきます。"),
            ParamSpec("weights", "重み付け", "select", "uniform",
                      options=("uniform", "distance"),
                      help="distance にすると近い点の影響が強くなります。"),
        ),
        summary="学習らしい学習をせず、近くにある点の多数決で決める。",
        smooth=False,
    ),
    "svm_linear": ModelSpec(
        key="svm_linear",
        label="SVM（線形カーネル）",
        build=lambda C=1.0: _scaled(SVC(kernel="linear", C=C)),
        params=(
            ParamSpec("C", "マージンの硬さ C", "log", 1.0, -3, 3,
                      help="大きいほど誤分類を許さず、境界がデータに張り付きます。"),
        ),
        summary="クラス間のマージン（余白）が最大になる直線を引く。",
    ),
    "svm_rbf": ModelSpec(
        key="svm_rbf",
        label="SVM（RBF カーネル）",
        build=lambda C=1.0, gamma=1.0: _scaled(SVC(kernel="rbf", C=C, gamma=gamma)),
        params=(
            ParamSpec("C", "マージンの硬さ C", "log", 1.0, -3, 3),
            ParamSpec("gamma", "カーネル幅 gamma", "log", 1.0, -2, 2,
                      help="大きいほど 1 点の影響範囲が狭まり、境界が細かく波打ちます。"),
        ),
        summary="データを高次元に写して、曲がった境界を引けるようにした SVM。",
    ),
    "tree": ModelSpec(
        key="tree",
        label="決定木",
        build=lambda max_depth=4, min_samples_leaf=1: DecisionTreeClassifier(
            max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf), random_state=0
        ),
        params=(
            ParamSpec("max_depth", "木の深さ", "int", 4, 1, 20, 1,
                      help="深いほど細かく分割でき、過学習しやすくなります。"),
            ParamSpec("min_samples_leaf", "葉の最小サンプル数", "int", 1, 1, 50, 1),
        ),
        summary="軸に平行な直線で領域を切り分けていく。境界が階段状になる。",
        smooth=False,
    ),
    "forest": ModelSpec(
        key="forest",
        label="ランダムフォレスト",
        build=lambda n_estimators=100, max_depth=8: RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=0,
            n_jobs=-1,
        ),
        params=(
            ParamSpec("n_estimators", "木の本数", "int", 100, 1, 300, 1),
            ParamSpec("max_depth", "木の深さ", "int", 8, 1, 20, 1),
        ),
        summary="少しずつ違う決定木を多数作り、多数決を取る。",
        smooth=False,
    ),
    "gbdt": ModelSpec(
        key="gbdt",
        label="勾配ブースティング",
        build=lambda n_estimators=100, learning_rate=0.1, max_depth=3: GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            random_state=0,
        ),
        params=(
            ParamSpec("n_estimators", "段数", "int", 100, 1, 300, 1),
            ParamSpec("learning_rate", "学習率", "log", 0.1, -3, 0),
            ParamSpec("max_depth", "木の深さ", "int", 3, 1, 8, 1),
        ),
        summary="前の木の間違いを次の木が埋める、という積み上げ方式。",
        smooth=False,
    ),
    "bagging": ModelSpec(
        key="bagging",
        label="バギング（決定木）",
        build=lambda n_estimators=50, max_depth=8: BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=int(max_depth), random_state=0),
            n_estimators=int(n_estimators),
            random_state=0,
            n_jobs=-1,
        ),
        params=(
            ParamSpec("n_estimators", "木の本数", "int", 50, 1, 200, 1),
            ParamSpec("max_depth", "木の深さ", "int", 8, 1, 20, 1),
        ),
        summary="データを復元抽出し直して木を作り、平均を取る。",
        smooth=False,
    ),
    "gnb": ModelSpec(
        key="gnb",
        label="ナイーブベイズ",
        build=lambda var_smoothing=1e-9: GaussianNB(var_smoothing=var_smoothing),
        params=(
            ParamSpec("var_smoothing", "分散の下駄", "log", 1e-9, -12, -1),
        ),
        summary="各特徴が独立だと割り切り、確率でクラスを決める。とにかく速い。",
    ),
}

#: 決定境界ラボで出すモデルの並び順
BOUNDARY_MODELS = ("logreg", "knn", "svm_linear", "svm_rbf", "tree", "forest", "gnb")

#: アンサンブルラボで比較する 4 段階
ENSEMBLE_STAGES = ("tree", "bagging", "forest", "gbdt")
