"""シミュレーションの共通骨格 — 世界・方策・台帳。

ここには「どのシナリオでも同じ形になる部分」だけを置く。
在庫なのか与信なのかを知っているのは `inventory.py` のような各シナリオ側で、
このモジュールは中身を知らないまま時間を進める。

## 守っている約束

**方策に未来を渡さない。** `simulate` が方策に渡すのは「いまの状態」と「予測」の
2 つだけで、実際に起きたこと（`実績` で始まる列）は取り除いてから渡す。
シミュレーションで最も起きやすい事故がこれで、しかも起きても
「やたら良い結果」として出てくるだけなので気づきにくい。仕組みで塞いでおく。

**台帳は閉じている。** 損益の内訳を足すと利益に一致する（`Ledger.check_closes`）。
内訳のどれかを二重計上すると、この検査で落ちる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

#: 予測フレームのうち、方策に渡してはいけない列の接頭辞。
#: 評価のために持ち歩くが、意思決定には使わせない。
TRUTH_PREFIX = "実績"


# ======================================================================
# 世界
# ======================================================================


@runtime_checkable
class World(Protocol):
    """時間を進められる世界。状態を持つ。

    各シナリオはこの 4 つのメソッドを備えるだけでよく、
    `simulate` はシナリオの中身を知らずに回せる。
    """

    def dates(self) -> pd.DatetimeIndex:
        """シミュレーションする期間（学習に使う過去は含まない）。"""

    def state(self) -> dict[str, float]:
        """いま方策から見えている状態（在庫・発注残など）。"""

    def apply(self, t: pd.Timestamp, decision: Any) -> dict[str, Any]:
        """意思決定を実行し、その日の記録を返す。状態は更新される。"""

    def reset(self) -> None:
        """状態を初期化する。方策を変えて回し直すときに呼ぶ。"""


#: 方策。状態と予測だけを受け取り、意思決定を返す。
#: 真の値は引数に無い — これが「未来を渡さない」の実体である。
Policy = Callable[[dict[str, float], "pd.Series | None"], Any]


# ======================================================================
# 台帳
# ======================================================================


@dataclass
class Ledger:
    """日々の損益を積む台帳。

    Args:
        breakdown: 利益を構成する列名。**符号込みで足すと利益になる**ように
            費用は負の値で入れる。ウォーターフォール図はこの順に描かれる。
        profit_column: 合計にあたる列。
    """

    breakdown: tuple[str, ...] = ()
    profit_column: str = "利益"
    date_column: str = "日付"
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=[self.date_column, self.profit_column])
        return pd.DataFrame(self.rows)

    def total(self) -> float:
        """期間を通した利益。"""
        frame = self.frame()
        if frame.empty or self.profit_column not in frame:
            return 0.0
        return float(frame[self.profit_column].sum())

    def by_item(self) -> pd.Series:
        """内訳ごとの合計。ウォーターフォール図に使う。"""
        frame = self.frame()
        columns = [c for c in self.breakdown if c in frame]
        if frame.empty or not columns:
            return pd.Series(dtype=float)
        return frame[columns].sum()

    def cumulative(self) -> pd.Series:
        """利益の累積。横軸は日付。"""
        frame = self.frame()
        if frame.empty:
            return pd.Series(dtype=float)
        return pd.Series(
            frame[self.profit_column].cumsum().to_numpy(),
            index=pd.DatetimeIndex(frame[self.date_column]),
        )

    def check_closes(self, tolerance: float = 1e-6) -> bool:
        """内訳の合計が利益に一致するか。二重計上の検査。"""
        frame = self.frame()
        columns = [c for c in self.breakdown if c in frame]
        if frame.empty or not columns:
            return True
        residual = frame[columns].sum(axis=1) - frame[self.profit_column]
        return bool(np.all(np.abs(residual.to_numpy(dtype=float)) < tolerance))


# ======================================================================
# 実行
# ======================================================================


def drop_truth(signals: pd.DataFrame) -> pd.DataFrame:
    """予測フレームから「実際に起きたこと」の列を落とす。

    評価には要るが意思決定に使ってはいけない列を、方策に渡す前にここで外す。
    """
    keep = [c for c in signals.columns if not str(c).startswith(TRUTH_PREFIX)]
    return signals[keep]


def simulate(world: World, policy: Policy, signals: pd.DataFrame, ledger: Ledger) -> Ledger:
    """1 つの方策で、世界を最後まで進める。

    方策に渡るのは `world.state()` と、`実績` 列を落とした予測の 1 行だけ。
    その日に何が起きたかは `world.apply` の中でしか参照されない。

    Args:
        world: 進める世界。呼び出しの先頭で `reset()` される。
        policy: 状態と予測から意思決定を返す関数。
        signals: 日付を index に持つ予測。方策が使わないシナリオでは空でよい。
        ledger: 記録先。`breakdown` を設定済みのものを渡す。

    Returns:
        渡した台帳（行が追加された状態）。
    """
    world.reset()
    usable = drop_truth(signals) if len(signals.columns) else signals

    for t in world.dates():
        signal = usable.loc[t] if t in usable.index else None
        decision = policy(world.state(), signal)
        ledger.add({ledger.date_column: t, **world.apply(t, decision)})

    return ledger


def decompose(actual: float, best_with_forecast: float, oracle: float) -> dict[str, float]:
    """オラクルとの差が、どこから来ているのかを分ける。

        オラクル利益 − 実現利益 = 予測誤差による損失 + 方策の非最適性による損失

    **モデルを磨くべきか、方策を直すべきか**が、この 2 つの大小で決まる。
    予測誤差ぶんが大きいなら特徴量やモデルに手を入れる価値があり、
    方策ぶんが大きいなら、予測をいくら良くしても取り返せない。

    Args:
        actual: いま選んでいる方策の利益。
        best_with_forecast: 同じ予測を使う最良の方策の利益。
        oracle: 真の値を知っていた場合の利益（上限）。
    """
    return {
        "オラクル利益": oracle,
        "予測誤差による損失": oracle - best_with_forecast,
        "方策の非最適性による損失": best_with_forecast - actual,
        "実現利益": actual,
    }
