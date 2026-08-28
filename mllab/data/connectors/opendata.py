"""定番の公開データセットを取り込む。

scikit-learn に同梱されているものはネットワーク不要で必ず動く。
外部 API が落ちていてもテーブルデータの練習を始められるよう、
最初の一歩としてここを用意しておく。
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from sklearn import datasets as skds

from mllab.data.connectors.base import Connector, FetchResult, Option

SOURCE = "scikit-learn 同梱データセット"


def _bundled(loader: Callable[..., Any], target_name: str, target_labels: bool):
    """sklearn のローダを、特徴量 + 目的変数の 1 枚の表に変換する。"""

    def build() -> pd.DataFrame:
        data = loader()
        frame = pd.DataFrame(data.data, columns=list(data.feature_names))
        if target_labels and hasattr(data, "target_names"):
            names = list(data.target_names)
            frame[target_name] = [names[int(i)] for i in data.target]
        else:
            frame[target_name] = data.target
        return frame

    return build


#: キー → (表示名, 作る関数, 課題の種類, 説明)
DATASETS: dict[str, tuple[str, Callable[[], pd.DataFrame], str, str]] = {
    "iris": (
        "アヤメ (iris)",
        _bundled(skds.load_iris, "品種", True),
        "分類",
        "150 行 4 列。機械学習で最初に触る定番。3 品種のうち 2 つは重なっています。",
    ),
    "wine": (
        "ワイン (wine)",
        _bundled(skds.load_wine, "産地", True),
        "分類",
        "178 行 13 列。化学成分から産地を当てます。特徴量のスケールがばらばらで、標準化の効果が見えます。",
    ),
    "breast_cancer": (
        "乳がん診断 (breast cancer)",
        _bundled(skds.load_breast_cancer, "診断", True),
        "分類",
        "569 行 30 列。良性・悪性の 2 値分類。医療なので見逃し（再現率）が効く題材です。",
    ),
    "diabetes": (
        "糖尿病の進行度 (diabetes)",
        _bundled(skds.load_diabetes, "1年後の進行度", False),
        "回帰",
        "442 行 10 列。連続値を当てる回帰。特徴量は標準化済みです。",
    ),
    "digits": (
        "手書き数字 (digits)",
        _bundled(skds.load_digits, "数字", False),
        "分類",
        "1,797 行 64 列。8×8 画像を平らに並べたもの。次元削減ラボと同じデータです。",
    ),
    "california_housing": (
        "カリフォルニア住宅価格",
        lambda: _california(),
        "回帰",
        "20,640 行 8 列。実データらしい外れ値と偏りがあります（初回のみダウンロード）。",
    ),
}


def _california() -> pd.DataFrame:
    """カリフォルニア住宅価格。初回だけ sklearn がダウンロードしてキャッシュする。"""
    data = skds.fetch_california_housing()
    frame = pd.DataFrame(data.data, columns=list(data.feature_names))
    frame["住宅価格中央値"] = data.target
    return frame


def fetch(dataset: str = "wine") -> FetchResult:
    """公開データセットを 1 枚の表として返す。"""
    if dataset not in DATASETS:
        return FetchResult.failure(f"未知のデータセットです: {dataset}")

    label, build, task, _ = DATASETS[dataset]
    try:
        frame = build()
    except Exception as exc:  # noqa: BLE001 - ダウンロード失敗もここに来る
        return FetchResult.failure(
            f"{label} の読み込みに失敗しました: {type(exc).__name__}: {exc}"
        )

    return FetchResult.success(
        frame, dataset=dataset, dataset_label=label, task=task,
        n_features=int(frame.shape[1] - 1),
    )


def _name(params: dict[str, Any]) -> str:
    return f"open_{params.get('dataset', 'wine')}"


CONNECTOR = Connector(
    key="opendata",
    label="公開データセット（定番）",
    domain="table",
    source=SOURCE,
    description=(
        "分類・回帰の練習に使われる定番データセット。"
        "ほとんどがライブラリに同梱されているので、ネットワークが無くても取り込めます。"
        "まずここから 1 つ入れて、カタログと SQL の使い方を確かめるのがおすすめです。"
    ),
    fetch=fetch,
    options=(
        Option(
            "dataset", "データセット", "select", "wine",
            options=tuple(DATASETS),
            labels={k: f"{v[0]} — {v[2]}" for k, v in DATASETS.items()},
        ),
    ),
    name_for=_name,
    terms="ライブラリ同梱・通信不要（カリフォルニア住宅価格のみ初回ダウンロード）",
)
