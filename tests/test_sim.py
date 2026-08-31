"""業務シミュレーションの検査。

このラボは「数字が出ること」ではなく「**その数字が信用できること**」が価値なので、
守られていなければ結論そのものが嘘になる 4 点をテストで固定する。

1. 台帳が閉じている（損益の内訳を足すと利益になる）
2. モノが保存されている（期首 + 入荷 − 販売 − 廃棄 = 期末）
3. 方策に未来が渡っていない
4. オラクルが本当に上限である

加えて、このラボの主張そのもの（**損が非対称なら分位点が平均に勝つ**）も
テストにしてある。ここが崩れたら、解説の文章を書き換えなければならない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mllab.sim import core
from mllab.sim import inventory as INV
from mllab.sim.policies import base_stock_order, critical_ratio, z_for_service_level

#: テストは軽い設定で回す。移動平均の予測なら学習が要らない
FAST = INV.ForecastConfig(method="moving_average")


def small(**kwargs) -> INV.InventoryConfig:
    base = dict(days=90, history_days=200, base_demand=60.0)
    base.update(kwargs)
    return INV.InventoryConfig(**base)


def run_all(config: INV.InventoryConfig, spec: INV.ForecastConfig = FAST):
    return INV.compare_policies(config, spec)


# ======================================================================
# 台帳とモノの保存
# ======================================================================


def test_ledger_closes() -> None:
    """損益の内訳を足すと利益に一致する（二重計上の検査）。"""
    _, ledgers, _ = run_all(small())
    for label, ledger in ledgers.items():
        assert ledger.check_closes(), f"{label} の内訳が利益に一致しない"


def test_inventory_is_conserved() -> None:
    """期首在庫 + 入荷 − 販売 − 廃棄 = 期末在庫 が全日で成り立つ。

    どこかで在庫が湧いたり消えたりしていれば、利益もそのぶん嘘になる。
    """
    _, ledgers, _ = run_all(small(shelf_life=4, lead_time=2))
    for label, ledger in ledgers.items():
        frame = ledger.frame()
        residual = (
            frame["期首在庫"] + frame["入荷"] - frame["販売"] - frame["廃棄"]
            - frame["期末在庫"]
        )
        assert np.allclose(residual, 0), f"{label} で在庫が保存されていない"


def test_sales_never_exceed_demand_or_stock() -> None:
    _, ledgers, _ = run_all(small())
    for label, ledger in ledgers.items():
        frame = ledger.frame()
        assert (frame["販売"] <= frame["需要"] + 1e-9).all(), f"{label}: 需要以上に売れている"
        assert (frame["欠品"] >= -1e-9).all(), f"{label}: 欠品が負"
        assert (frame["期末在庫"] >= -1e-9).all(), f"{label}: 在庫が負"


def test_lead_time_zero_arrives_the_same_day() -> None:
    """L=0 は「朝に発注してその日のうちに届く」。純粋な新聞売り子になる。"""
    config = small(lead_time=0, shelf_life=1)
    world = INV.InventoryWorld(config, INV.generate(config))
    world.reset()
    t = world.dates()[0]
    record = world.apply(t, 999)
    assert record["入荷"] == 999, "リードタイム 0 の発注が当日に届いていない"


# ======================================================================
# 未来を見ていないこと
# ======================================================================


def test_drop_truth_removes_outcome_columns() -> None:
    frame = pd.DataFrame({"平均": [1.0], "実績カバー期間需要": [2.0]})
    assert list(core.drop_truth(frame).columns) == ["平均"]


def test_policy_never_receives_the_outcome() -> None:
    """方策が受け取る予測に、実際に起きたことの列が混ざっていないこと。"""
    config = small()
    world = INV.InventoryWorld(config, INV.generate(config))
    signals = INV.forecast(world, config, FAST)
    assert any(c.startswith(core.TRUTH_PREFIX) for c in signals.columns), (
        "前提が崩れている: 予測フレームに実績列が入っているはず"
    )

    seen: list[list[str]] = []

    def spy(state: dict[str, float], signal: pd.Series | None) -> float:
        seen.append(list(signal.index) if signal is not None else [])
        return 0.0

    core.simulate(world, spy, signals, INV.new_ledger())
    assert seen, "方策が 1 度も呼ばれていない"
    for columns in seen:
        assert not [c for c in columns if c.startswith(core.TRUTH_PREFIX)], (
            f"方策に実績列が渡っている: {columns}"
        )


def test_features_do_not_use_the_current_or_future_demand() -> None:
    """当日以降の需要を書き換えても、その日の特徴量が変わらないこと。

    発注は朝、当日の需要を見る前に決める。特徴量に当日の実績が混ざると、
    検証だけ良くて本番で崩れるモデルができる。
    """
    config = small()
    frame = INV.generate(config)
    X_before, _ = INV.make_features(frame, config)

    cut = frame.index[250]
    tampered = frame.copy()
    tampered.loc[cut:, "需要"] = tampered.loc[cut:, "需要"] * 7 + 1000
    X_after, _ = INV.make_features(tampered, config)

    assert cut in X_before.index and cut in X_after.index
    pd.testing.assert_series_equal(X_before.loc[cut], X_after.loc[cut])


# ======================================================================
# オラクルと、このラボの主張
# ======================================================================


def test_oracle_is_the_upper_bound() -> None:
    """真の需要を知っている方策を、どの方策も超えられないこと。

    超えてしまったら、どこかで未来を覗いているか、オラクルの定義が誤っている。
    """
    summary, _, _ = run_all(small())
    oracle = float(summary.loc[summary["方策"] == INV.ORACLE_LABEL, "利益"].iloc[0])
    others = summary[summary["方策"] != INV.ORACLE_LABEL]["利益"].astype(float)
    assert (others <= oracle + 1e-6).all(), (
        f"オラクル {oracle:,.0f} を超えた方策がある:\n{summary}"
    )


def test_oracle_never_runs_out_and_never_wastes() -> None:
    """完全予見なら、欠品も廃棄も起きない。上限として正しく機能している印。"""
    summary, _, _ = run_all(small(lead_time=0, shelf_life=1))
    row = summary[summary["方策"] == INV.ORACLE_LABEL].iloc[0]
    assert row["欠品率"] < 1e-9, "完全予見なのに欠品している"
    assert row["廃棄率"] < 0.01, "完全予見なのに廃棄が多い"


@pytest.mark.parametrize(
    "name,overrides",
    [
        ("欠品のほうが痛い", dict(price=500, cost=200, stockout_penalty=600)),
        ("廃棄のほうが痛い", dict(price=250, cost=200, stockout_penalty=0)),
    ],
)
def test_quantile_beats_mean_when_costs_are_asymmetric(name, overrides) -> None:
    """**このラボの主張そのもの。**

    損が非対称なら、平均を当てにいく方策より、臨界比の分位点を狙う方策のほうが
    利益が大きい。ここが崩れたら解説の文章のほうを直さなければならない。
    """
    config = small(lead_time=0, shelf_life=1, **overrides)
    assert abs(config.critical_ratio - 0.5) > 0.1, "非対称な設定になっていない"

    summary, _, _ = run_all(config)
    profit = summary.set_index("方策")["利益"].astype(float)
    quantile = profit[INV.POLICIES["quantile"].label]
    mean = profit[INV.POLICIES["mean"].label]
    assert quantile > mean, (
        f"{name}（CR={config.critical_ratio:.2f}）で分位点が平均に負けた:\n{summary}"
    )


def test_quantile_orders_more_when_shortage_hurts_more() -> None:
    """臨界比が上がれば、分位点方策は在庫を厚く持つ方向へ動くこと。"""
    def average_stock(penalty: float) -> float:
        config = small(lead_time=0, shelf_life=1, stockout_penalty=penalty)
        world = INV.InventoryWorld(config, INV.generate(config))
        signals = INV.forecast(world, config, FAST)
        return float(INV.run(world, config, signals, "quantile").frame()["発注"].mean())

    assert average_stock(600) > average_stock(0), "欠品ペナルティを上げても発注が増えない"


# ======================================================================
# 損失の分解
# ======================================================================


def test_decomposition_closes() -> None:
    """オラクル利益 − 実現利益 = 予測誤差ぶん + 方策ぶん。"""
    parts = core.decompose(actual=80.0, best_with_forecast=90.0, oracle=100.0)
    assert parts["予測誤差による損失"] == pytest.approx(10.0)
    assert parts["方策の非最適性による損失"] == pytest.approx(10.0)
    total = parts["予測誤差による損失"] + parts["方策の非最適性による損失"]
    assert parts["オラクル利益"] - parts["実現利益"] == pytest.approx(total)


# ======================================================================
# 部品
# ======================================================================


def test_critical_ratio() -> None:
    assert critical_ratio(1.0, 1.0) == pytest.approx(0.5)
    assert critical_ratio(9.0, 1.0) == pytest.approx(0.9)
    # 0 除算でも壊れない
    assert 0.0 < critical_ratio(0.0, 0.0) < 1.0


def test_service_level_z() -> None:
    assert z_for_service_level(0.5) == pytest.approx(0.0, abs=1e-6)
    assert z_for_service_level(0.95) == pytest.approx(1.645, abs=1e-3)


def test_base_stock_order_subtracts_the_pipeline() -> None:
    """発注残を引き忘れると、リードタイムの間ずっと二重発注してしまう。"""
    assert base_stock_order(100, 30) == 70
    assert base_stock_order(100, 120) == 0


def test_forecast_is_reproducible() -> None:
    config = small()
    first, _, _ = run_all(config)
    second, _, _ = run_all(config)
    pd.testing.assert_frame_equal(first, second)


def test_accuracy_vs_profit_covers_every_combination() -> None:
    grid = INV.accuracy_vs_profit(small(), FAST)
    assert not grid.empty
    assert {"予測手法", "方策", "RMSE", "利益"} <= set(grid.columns)
    # 完全予見の行は RMSE 0
    oracle_rows = grid[grid["予測手法"] == INV.ORACLE_LABEL]
    assert (oracle_rows["RMSE"] < 1e-9).all()
