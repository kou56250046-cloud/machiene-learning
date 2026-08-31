"""方策の共通部品。

方策とは「予測を意思決定に変換する規則」である。ここには
シナリオをまたいで使える変換だけを置く。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def critical_ratio(underage: float, overage: float) -> float:
    """臨界比 CR = Cu / (Cu + Co)。

    `Cu` は 1 個足りなかったときの損、`Co` は 1 個余ったときの損。
    **この比が、当てにいくべき分位点そのものになる。**

    欠品のほうが痛ければ CR > 0.5 となり、平均より多めに持つのが最適。
    逆に廃棄のほうが痛ければ CR < 0.5 で、絞るのが最適になる。
    平均を当てにいく（＝ 0.5 分位点）のが正解なのは、両者が同額のときだけ。
    """
    total = underage + overage
    if total <= 0:
        return 0.5
    return float(np.clip(underage / total, 0.001, 0.999))


def z_for_service_level(level: float) -> float:
    """確率に対応する標準正規の分位点 Φ⁻¹(p)。

    安全在庫 `z σ √(L+1)` の `z` がこれ。渡す確率は 2 通りある。

    - **サービスレベル** — 欠品しない確率として人が決める目標値
    - **臨界比** — コスト構造から導かれる、本来当てるべき分位点

    この 2 つは**別物**で、食い違っているとき在庫は必ず過剰か過少になる。
    臨界比は 0.5 を下回りうる（廃棄のほうが痛い場合）ので、
    ここで 0.5 を下限にしてはいけない。負の z ＝ 平均より絞るのが正解、という
    まっとうな答えを潰してしまう。
    """
    return float(norm.ppf(np.clip(level, 0.001, 0.999)))


def base_stock_order(target_level: float, inventory_position: float) -> float:
    """発注量 = 目標在庫 − 在庫ポジション（手元在庫 + 発注残）。

    発注残を引き忘れると、リードタイムの間ずっと二重発注し続けることになる。
    在庫管理で最も多い実装ミス。
    """
    return float(max(0.0, target_level - inventory_position))
