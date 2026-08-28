"""mllab の計算ロジック層のユニットテスト。

UI（Streamlit）は対象外。ここが緑なら、ページ側は組み立てるだけ。
"""

from __future__ import annotations

import numpy as np
import pytest

from mllab.data import toy
from mllab.models.clustering import elbow_curve, kmeans_history
from mllab.models.registry import BOUNDARY_MODELS, CLASSIFIERS, ENSEMBLE_STAGES
from mllab.viz import surface, theme
from mllab.viz.boundary import cluster_figure, decision_figure, decision_scores, make_mesh


# ---- データ生成 -------------------------------------------------------

@pytest.mark.parametrize("kind", list(toy.DATASETS))
def test_make_dataset_shape(kind: str) -> None:
    X, y = toy.make_dataset(kind, n_samples=200, noise=0.2, seed=0, n_classes=3)
    assert X.ndim == 2 and X.shape[1] == 2
    assert len(X) == len(y)
    assert X.dtype == float and y.dtype == int
    assert np.isfinite(X).all()


@pytest.mark.parametrize("kind", list(toy.DATASETS))
def test_make_dataset_is_standardized(kind: str) -> None:
    """ラボ間で感覚を使い回せるよう、各軸は平均 0・分散 1 に揃っている。"""
    X, _ = toy.make_dataset(kind, 400, 0.2, 0, n_classes=3)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-6)


def test_make_dataset_is_reproducible() -> None:
    a, ya = toy.make_dataset("moons", 100, 0.2, seed=7)
    b, yb = toy.make_dataset("moons", 100, 0.2, seed=7)
    assert np.array_equal(a, b) and np.array_equal(ya, yb)


def test_make_dataset_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        toy.make_dataset("does-not-exist")


@pytest.mark.parametrize("kind", list(toy.TRUE_FUNCTIONS))
def test_make_regression_1d(kind: str) -> None:
    X, y = toy.make_regression_1d(kind, 40, 0.2, 0)
    assert X.shape == (40, 1) and y.shape == (40,)
    assert (X >= 0).all() and (X <= 1).all()
    assert np.all(np.diff(X.ravel()) >= 0)  # x はソート済み


# ---- モデルレジストリ -------------------------------------------------

@pytest.mark.parametrize("key", list(CLASSIFIERS))
def test_every_classifier_fits(key: str) -> None:
    spec = CLASSIFIERS[key]
    X, y = toy.make_dataset("moons", 150, 0.2, 0)
    model = spec.create(spec.defaults())
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)) <= set(np.unique(y))


@pytest.mark.parametrize("key", list(CLASSIFIERS))
def test_defaults_are_within_declared_range(key: str) -> None:
    for p in CLASSIFIERS[key].params:
        if p.kind in ("int", "float"):
            assert p.min <= p.default <= p.max
        elif p.kind == "log":
            assert p.min <= np.log10(p.default) <= p.max
        elif p.kind == "select":
            assert p.default in p.options


def test_create_ignores_unknown_params() -> None:
    """UI からモデルを切り替えても、前のモデルのキーが混ざって落ちないこと。"""
    spec = CLASSIFIERS["knn"]
    model = spec.create({**spec.defaults(), "gamma": 3.0, "max_depth": 9})
    model.fit(*toy.make_dataset("blobs", 80, 0.2, 0))


def test_lab_model_lists_exist() -> None:
    for key in BOUNDARY_MODELS + ENSEMBLE_STAGES:
        assert key in CLASSIFIERS


# ---- 決定境界 ---------------------------------------------------------

def test_make_mesh_covers_data() -> None:
    X, _ = toy.make_dataset("blobs", 100, 0.2, 0)
    xx, yy, grid = make_mesh(X, resolution=40)
    assert xx.shape == (40, 40) and grid.shape == (1600, 2)
    assert grid[:, 0].min() < X[:, 0].min() and grid[:, 0].max() > X[:, 0].max()
    assert grid[:, 1].min() < X[:, 1].min() and grid[:, 1].max() > X[:, 1].max()


@pytest.mark.parametrize("key", list(CLASSIFIERS))
def test_decision_scores_modes(key: str) -> None:
    X, y = toy.make_dataset("moons", 120, 0.2, 0)
    spec = CLASSIFIERS[key]
    model = spec.create(spec.defaults())
    model.fit(X, y)
    _, _, grid = make_mesh(X, resolution=20)
    scores, mode = decision_scores(model, grid)
    assert mode in ("proba", "decision", "label")
    assert scores.shape == (400,)
    assert np.isfinite(scores).all()
    if mode in ("proba", "decision"):
        assert scores.min() >= 0.0 and scores.max() <= 1.0


@pytest.mark.parametrize("key", list(BOUNDARY_MODELS))
def test_decision_figure_builds(key: str) -> None:
    X, y = toy.make_dataset("moons", 120, 0.2, 0)
    spec = CLASSIFIERS[key]
    model = spec.create(spec.defaults())
    model.fit(X, y)
    fig = decision_figure(model, X, y, X[:20], y[:20], resolution=30)
    assert len(fig.data) >= 3  # 背景 + 訓練 2 クラス以上


def test_decision_figure_multiclass() -> None:
    X, y = toy.make_dataset("blobs", 150, 0.2, 0, n_classes=4)
    spec = CLASSIFIERS["forest"]
    model = spec.create(spec.defaults())
    model.fit(X, y)
    fig = decision_figure(model, X, y, resolution=30)
    assert len(fig.data) == 1 + 4  # 離散ヒートマップ + 4 クラスの散布


def test_cluster_figure_handles_noise_label() -> None:
    X, _ = toy.make_dataset("blobs", 60, 0.2, 0, n_classes=3)
    labels = np.array([0, 1, -1] * 20)
    fig = cluster_figure(X, labels, centers=np.zeros((2, 2)))
    names = [t.name for t in fig.data]
    assert "ノイズ" in names and "重心" in names


# ---- クラスタリング ---------------------------------------------------

@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_kmeans_history_converges(init: str) -> None:
    X, _ = toy.make_dataset("blobs", 300, 0.2, 0, n_classes=4)
    hist = kmeans_history(X, k=4, init=init, seed=0)
    assert len(hist) >= 1
    # 反復のたびにクラスタ内平方和は単調に減る（増えたらバグ）
    inertias = [h.inertia for h in hist]
    assert all(b <= a + 1e-9 for a, b in zip(inertias, inertias[1:]))
    assert hist[-1].shift < 1e-6  # 最後は収束している
    assert hist[-1].centers.shape == (4, 2)
    assert set(np.unique(hist[-1].labels)) <= set(range(4))


def test_kmeans_history_is_reproducible() -> None:
    X, _ = toy.make_dataset("blobs", 200, 0.2, 0, n_classes=3)
    a = kmeans_history(X, 3, "k-means++", seed=5)
    b = kmeans_history(X, 3, "k-means++", seed=5)
    assert np.array_equal(a[-1].labels, b[-1].labels)


def test_elbow_curve_is_decreasing() -> None:
    X, _ = toy.make_dataset("blobs", 200, 0.2, 0, n_classes=4)
    ks, inertias = elbow_curve(X, k_max=8, seed=0)
    assert ks.tolist() == list(range(1, 9))
    assert all(b <= a + 1e-6 for a, b in zip(inertias, inertias[1:]))


# ---- 勾配降下 ---------------------------------------------------------

@pytest.mark.parametrize("skey", list(surface.SURFACES))
@pytest.mark.parametrize("okey", list(surface.OPTIMIZERS))
def test_descend_with_default_lr_does_not_diverge(skey: str, okey: str) -> None:
    s = surface.SURFACES[skey]
    path, losses, diverged = surface.descend(s, okey, lr=s.default_lr, steps=80)
    assert not diverged, f"{skey}/{okey} が既定学習率で発散した"
    assert path.shape == (81, 2) and losses.shape == (81,)
    assert np.isfinite(losses).all()
    # 出発点より損失が下がっている
    assert losses[-1] <= losses[0] + 1e-9


def test_descend_diverges_with_huge_lr() -> None:
    _, _, diverged = surface.descend(surface.SURFACES["bowl"], "sgd", lr=5.0, steps=50)
    assert diverged


def test_descend_rejects_unknown_optimizer() -> None:
    with pytest.raises(ValueError):
        surface.descend(surface.SURFACES["bowl"], "nope", lr=0.1, steps=5)


def test_surface_minima_are_actual_minima() -> None:
    """宣言している「真の最小値」が、実際に格子上の最小と一致すること。"""
    for s in surface.SURFACES.values():
        xs = np.linspace(*s.x_range, 401)
        ys = np.linspace(*s.y_range, 401)
        XX, YY = np.meshgrid(xs, ys)
        ZZ = s.func(XX, YY)
        assert s.value(*s.minimum) <= ZZ.min() + 1e-2, f"{s.key} の minimum が誤り"


def test_surface_and_loss_figures_build() -> None:
    s = surface.SURFACES["ravine"]
    path, losses, _ = surface.descend(s, "adam", lr=s.default_lr, steps=30)
    assert len(surface.surface_figure(s, path, resolution=40).data) == 5
    assert len(surface.loss_curve_figure(losses).data) == 1


# ---- テーマ -----------------------------------------------------------

def test_theme_has_five_accents() -> None:
    assert len(theme.ACCENTS) == 5
    assert len(set(theme.ACCENTS)) == 5
    for color in theme.ACCENTS:
        assert color.startswith("#") and len(color) == 7


def test_class_color_cycles() -> None:
    assert theme.class_color(0) == theme.class_color(len(theme.CATEGORY))
    assert theme.class_color(1) != theme.class_color(0)


def test_ten_classes_get_distinct_colors() -> None:
    """次元削減ラボは数字 10 クラスを描くので、色が重複してはいけない。"""
    colors = [theme.class_color(i) for i in range(10)]
    assert len(set(colors)) == 10
    assert colors[:5] == theme.ACCENTS  # 少数クラスのラボでは従来の 5 色のまま


def test_rgba_conversion() -> None:
    assert theme.rgba("#4DD8FF", 0.5) == "rgba(77,216,255,0.5)"


def test_apply_registers_template() -> None:
    import plotly.io as pio

    theme.apply()
    assert pio.templates.default == theme.TEMPLATE
    assert list(pio.templates[theme.TEMPLATE].layout.colorway) == theme.ACCENTS


def test_every_lab_has_a_color() -> None:
    for lab in range(12):
        assert theme.LAB_COLORS[lab] in theme.ACCENTS


# ---- Plotly の出力そのものの健全性 ------------------------------------

def _figure_json(fig) -> dict:
    import json

    return json.loads(fig.to_json())


def _all_figures():
    """全ラボが使う共通の図を一通り作って返す。"""
    X, y = toy.make_dataset("moons", 120, 0.2, 0)
    spec = CLASSIFIERS["logreg"]
    model = spec.create(spec.defaults())
    model.fit(X, y)
    yield "decision(binary)", decision_figure(model, X, y, X[:20], y[:20], resolution=30)

    X3, y3 = toy.make_dataset("blobs", 120, 0.2, 0, n_classes=3)
    forest = CLASSIFIERS["forest"]
    m3 = forest.create(forest.defaults())
    m3.fit(X3, y3)
    yield "decision(multiclass)", decision_figure(m3, X3, y3, resolution=30)

    yield "cluster", cluster_figure(X3, y3, centers=np.zeros((3, 2)))

    s = surface.SURFACES["bowl"]
    path, losses, _ = surface.descend(s, "adam", lr=s.default_lr, steps=20)
    yield "surface", surface.surface_figure(s, path, resolution=40)
    yield "loss_curve", surface.loss_curve_figure(losses)


def test_no_figure_emits_an_empty_title() -> None:
    """空の title は Plotly が "undefined" と描いてしまうので、出してはいけない。"""
    for name, fig in _all_figures():
        title = _figure_json(fig)["layout"].get("title")
        assert title in (None, {}) or title.get("text"), f"{name} の title が空"
        assert title != {}, f"{name} が空の title オブジェクトを出している"


def test_only_scatter_traces_appear_in_legends() -> None:
    """ヒートマップ・等高線が凡例に混ざると "undefined" の見出しが出る。"""
    for name, fig in _all_figures():
        for trace in fig.data:
            if trace.type in ("heatmap", "contour"):
                assert trace.showlegend is False, (
                    f"{name} の {trace.type} が凡例から外れていない"
                )


def test_titled_figure_keeps_its_title() -> None:
    X, y = toy.make_dataset("blobs", 60, 0.2, 0, n_classes=2)
    fig = cluster_figure(X, y, title="テスト見出し")
    assert _figure_json(fig)["layout"]["title"]["text"] == "テスト見出し"
