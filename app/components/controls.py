"""サイドバーの共通コントロール。

データ生成のスライダと、モデルのハイパーパラメータ入力をまとめる。
ハイパーパラメータは `ModelSpec.params` の定義から機械的に描くので、
新しいモデルを registry に足せばページを触らなくても UI に出る。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split

from app.components.layout import sidebar_section
from mllab.data import toy
from mllab.models.registry import ModelSpec, ParamSpec


@dataclass(frozen=True)
class DataConfig:
    """データ生成の設定。ハッシュ可能なのでキャッシュのキーに使える。"""

    kind: str
    n_samples: int
    noise: float
    seed: int
    n_classes: int
    test_size: float


def dataset_controls(
    key: str,
    default_kind: str = "moons",
    allow_multiclass: bool = False,
    with_split: bool = True,
    max_samples: int = 2000,
) -> DataConfig:
    """データ生成のサイドバー UI を描き、設定を返す。"""
    sidebar_section("データ")
    kinds = list(toy.DATASETS)
    kind = st.sidebar.selectbox(
        "データの形",
        kinds,
        index=kinds.index(default_kind),
        format_func=lambda k: toy.DATASETS[k],
        key=f"{key}_kind",
    )
    n_samples = st.sidebar.slider(
        "サンプル数", 50, max_samples, min(400, max_samples), 50, key=f"{key}_n"
    )
    noise = st.sidebar.slider(
        "ノイズ", 0.0, 1.0, 0.20, 0.01, key=f"{key}_noise",
        help="大きいほどクラスが混ざり合い、境界を引きにくくなります。",
    )

    n_classes = 2
    if allow_multiclass and kind in ("blobs", "spirals", "aniso", "varied"):
        n_classes = st.sidebar.slider("クラス数", 2, 5, 3, 1, key=f"{key}_c")

    test_size = 0.3
    if with_split:
        test_size = st.sidebar.slider(
            "テストに回す割合", 0.1, 0.5, 0.30, 0.05, key=f"{key}_test",
            help="訓練に使わず、汎化性能の測定だけに使うデータの割合です。",
        )

    seed = st.sidebar.number_input(
        "乱数シード", 0, 9999, 0, 1, key=f"{key}_seed",
        help="同じ値なら毎回まったく同じデータが生成されます。",
    )

    return DataConfig(kind, int(n_samples), float(noise), int(seed), int(n_classes), float(test_size))


@st.cache_data(show_spinner=False)
def build_dataset(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """設定からデータを生成する（キャッシュ付き）。"""
    return toy.make_dataset(cfg.kind, cfg.n_samples, cfg.noise, cfg.seed, cfg.n_classes)


@st.cache_data(show_spinner=False)
def build_split(
    cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """訓練 / テストに分割したデータを返す（キャッシュ付き）。"""
    X, y = build_dataset(cfg)
    return train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )


def param_controls(spec: ModelSpec, key: str, container=None) -> dict[str, Any]:
    """`ModelSpec.params` からスライダ群を描き、値の辞書を返す。"""
    ui = container if container is not None else st.sidebar
    values: dict[str, Any] = {}
    for p in spec.params:
        values[p.key] = _one_control(ui, p, f"{key}_{spec.key}_{p.key}")
    return values


def _one_control(ui, p: ParamSpec, widget_key: str) -> Any:
    if p.kind == "int":
        return int(
            ui.slider(p.label, int(p.min), int(p.max), int(p.default),
                      int(p.step or 1), key=widget_key, help=p.help or None)
        )
    if p.kind == "float":
        return float(
            ui.slider(p.label, float(p.min), float(p.max), float(p.default),
                      float(p.step or 0.01), key=widget_key, help=p.help or None)
        )
    if p.kind == "log":
        # 対数スケール: スライダは指数を動かし、表示は実際の値にする
        exponent = ui.slider(
            p.label, float(p.min), float(p.max), float(np.log10(p.default)), 0.1,
            key=widget_key,
            help=(p.help + " " if p.help else "") + "対数スケールで動きます。",
            format="10^%.1f",
        )
        value = float(10.0**exponent)
        ui.caption(f"　→ {p.label} = **{value:.4g}**")
        return value
    if p.kind == "select":
        return ui.selectbox(
            p.label, list(p.options),
            index=list(p.options).index(p.default),
            key=widget_key, help=p.help or None,
        )
    raise ValueError(f"未知のパラメータ種別: {p.kind}")


def resolution_control(key: str, default: int = 160) -> int:
    """決定境界メッシュの解像度。重いときに下げられるようにしておく。"""
    sidebar_section("描画")
    return int(
        st.sidebar.select_slider(
            "境界の解像度",
            options=[60, 100, 140, 160, 200],
            value=default,
            key=f"{key}_res",
            help="高いほど境界がなめらかに見えますが、計算時間が延びます。",
        )
    )


def spec_controls(
    params: tuple[ParamSpec, ...], key: str, container=None
) -> dict[str, Any]:
    """`ParamSpec` の並びからそのままスライダ群を描く。

    `param_controls` はモデル用に `ModelSpec` を受け取るが、こちらは
    シミュレーションの環境設定のように、モデルに属さないつまみ向け。
    定義側（`mllab/sim/inventory.py` の `WORLD_PARAMS` など）に足せば
    画面を触らなくても増える。
    """
    ui = container if container is not None else st.sidebar
    return {p.key: _one_control(ui, p, f"{key}_{p.key}") for p in params}
