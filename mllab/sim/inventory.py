"""在庫・発注シミュレーション — 需要予測から発注量へ。

このシナリオが示したいことは 1 つだけである。

> **RMSE を最小にする予測は、利益を最大にしない。**

1 個多く仕入れた損（廃棄）と 1 個少なく仕入れた損（欠品）は同額ではない。
損が非対称なら、当てにいくべきは平均ではなく、臨界比
`CR = Cu / (Cu + Co)` に対応する**分位点**になる。
欠品のほうが痛ければ多めに、廃棄のほうが痛ければ少なめに持つのが正解で、
平均が正解なのは両者が釣り合っているときだけ。

計算はすべてここに置き、`app/views/lab12_inventory.py` は画面の組み立てだけを行う。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mllab.models.registry import ParamSpec
from mllab.sim.core import Ledger, Policy, simulate
from mllab.sim.policies import base_stock_order, critical_ratio, z_for_service_level

#: 損益の内訳。符号込みで足すと利益になる（費用は負で入る）。
BREAKDOWN = ("売上", "売上原価", "廃棄損", "保管費", "欠品損")

#: 需要の移動統計を取る窓。安全在庫の σ もここから作る。
SIGMA_WINDOW = 28


# ======================================================================
# 設定
# ======================================================================


@dataclass(frozen=True)
class InventoryConfig:
    """世界とお金の設定。

    frozen なのは `st.cache_data` のキーにするため。
    """

    # --- 需要の作り方 ---
    days: int = 365
    history_days: int = 730
    base_demand: float = 100.0
    weekend_lift: float = 1.35
    #: 平年より 10℃ 高いときに需要が何倍増えるか（0 なら気温と無関係）
    temp_sensitivity: float = 0.5
    trend_per_year: float = 0.0
    promo_rate: float = 0.06
    promo_lift: float = 1.8
    #: 1.0 ならポアソン。大きいほど過分散（同じ平均でも荒い需要になる）
    dispersion: float = 1.5

    # --- お金 ---
    price: float = 500.0
    cost: float = 200.0
    holding: float = 3.0
    #: 欠品 1 個あたりの、粗利を失う以外の損（機会損失・信用毀損）
    stockout_penalty: float = 150.0

    # --- 運用 ---
    #: 既定は「朝に仕込んで、その日のうちに売り切る」純粋な新聞売り子問題。
    #: どちらかを増やすと売れ残りが翌日へ持ち越され、話は一気に難しくなる
    lead_time: int = 0
    shelf_life: int = 1

    seed: int = 0

    # -- 導かれる値 ------------------------------------------------
    @property
    def margin(self) -> float:
        return self.price - self.cost

    @property
    def underage(self) -> float:
        """1 個足りなかったときの損 Cu = 逃した粗利 + ペナルティ。"""
        return self.margin + self.stockout_penalty

    @property
    def overage(self) -> float:
        """1 個余ったときの損 Co = 原価を回収できない + 保管費。"""
        return self.cost + self.holding

    @property
    def critical_ratio(self) -> float:
        return critical_ratio(self.underage, self.overage)

    @property
    def cover_days(self) -> int:
        """いま発注する量が守らなければならない日数。

        朝に出した発注は `lead_time` 日後に届く。その次の発注が届くのは
        さらに 1 日後なので、いまの在庫ポジションは
        **本日から `lead_time` 日後まで（＝ L + 1 日ぶん）**の需要を賄う必要がある。
        リードタイムが延びるほど、より遠い未来を当てなければならなくなる。
        """
        return self.lead_time + 1


#: 画面のつまみ。`ParamSpec` は決定境界ラボと共通のものを使い回す。
WORLD_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("base_demand", "平均需要（個/日）", "int", 100, 20, 400, 10),
    ParamSpec(
        "dispersion", "需要のばらつき", "float", 1.5, 1.0, 4.0, 0.1,
        help="1.0 でポアソン。上げるほど同じ平均でも荒れ、予測が難しくなります。",
    ),
    ParamSpec(
        "temp_sensitivity", "気温への反応", "float", 0.5, 0.0, 1.5, 0.1,
        help="平年より 10℃ 高い日に需要が何倍増えるか。0 にすると気温は無関係になります。",
    ),
    ParamSpec("promo_rate", "販促の頻度", "float", 0.06, 0.0, 0.3, 0.02),
)

MONEY_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("price", "売価（円）", "int", 500, 120, 2000, 10),
    ParamSpec("cost", "仕入原価（円）", "int", 200, 50, 1500, 10),
    ParamSpec(
        "stockout_penalty", "欠品ペナルティ（円/個）", "int", 150, 0, 1000, 10,
        help="売り逃した粗利とは別に、欠品 1 個あたりで失うもの。"
        "上げるほど「多めに持つ」が正解に近づきます。",
    ),
    ParamSpec("holding", "保管費（円/個/日）", "int", 3, 0, 50, 1),
)

OPERATION_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "lead_time", "リードタイム（日）", "int", 0, 0, 7, 1,
        help="発注してから届くまでの日数。長いほど、遠い未来を当てなければならなくなります。",
    ),
    ParamSpec(
        "shelf_life", "賞味期限（日）", "int", 1, 1, 30, 1,
        help="1 日なら売れ残りは廃棄。延ばすと翌日に持ち越せるので、"
        "「余った損 Co」が小さくなり、臨界比の前提が変わります。",
    ),
)


# ======================================================================
# 世界の生成
# ======================================================================


def generate(config: InventoryConfig) -> pd.DataFrame:
    """需要と外生変数を、決めた仕組みどおりに生成する。

        需要の平均 λ_t = 基準 × 曜日 × 気温 × 販促 × トレンド

    合成データなので「正解の仕組み」を我々は知っている。だからこそ
    オラクル（真の需要を知っていた場合）が計算でき、
    **予測をこれ以上頑張っても届く上限**が分かる。実データではこれができない。

    Returns:
        日付を index に持つ、`history_days + days` 行のフレーム。
    """
    rng = np.random.default_rng(config.seed)
    total = config.history_days + config.days
    index = pd.date_range("2021-01-01", periods=total, freq="D")

    day_of_year = index.dayofyear.to_numpy()
    # 気温は 8 月にピークが来る年周期 + 日々のゆらぎ
    temperature = (
        16.0
        + 11.0 * np.sin(2 * np.pi * (day_of_year - 110) / 365.25)
        + rng.normal(0, 2.5, total)
    )

    weekday = index.dayofweek.to_numpy()
    weekend = np.isin(weekday, (5, 6))
    promo = rng.random(total) < config.promo_rate

    years = np.arange(total) / 365.25
    level = (
        config.base_demand
        * np.where(weekend, config.weekend_lift, 1.0)
        * (1.0 + config.temp_sensitivity * (temperature - 16.0) / 10.0)
        * np.where(promo, config.promo_lift, 1.0)
        * (1.0 + config.trend_per_year * years)
    )
    level = np.clip(level, 1.0, None)

    if config.dispersion <= 1.0:
        demand = rng.poisson(level)
    else:
        # 負の二項。分散 = λ × dispersion になるよう r を決める
        r = level / (config.dispersion - 1.0)
        demand = rng.negative_binomial(r, r / (r + level))

    return pd.DataFrame(
        {
            "需要": demand.astype(float),
            "気温": temperature.round(1),
            "販促": promo,
            "週末": weekend,
        },
        index=index,
    )


# ======================================================================
# 世界
# ======================================================================


class InventoryWorld:
    """在庫を持ち、時間を進める世界。

    1 日の進み方は、実際の店と同じ順序にしてある。

        朝いちばんに発注 → 入荷 → 需要を捌く → 期限切れを捨てる → 保管費

    **発注はその日の需要を見る前に決める。** 見てから決められるなら予測は要らない。
    方策が見ているのは前日終業時点の在庫と、その日届く予定の発注残だけである。

    在庫は「残り日数つきの箱」の列で持ち、古いものから売る（先入れ先出し）。
    これをやらないと賞味期限が意味を持たなくなる。
    """

    def __init__(self, config: InventoryConfig, frame: pd.DataFrame) -> None:
        self.config = config
        self.frame = frame
        self._sim_index = frame.index[-config.days :]
        #: 学習に使ってよい過去。方策の影響を受けていない期間。
        self.history = frame.iloc[: -config.days]
        self.reset()

    # -- World プロトコル ------------------------------------------
    def dates(self) -> pd.DatetimeIndex:
        return self._sim_index

    def state(self) -> dict[str, float]:
        on_hand = self._on_hand()
        pipeline = float(sum(self._pipeline.values()))
        return {
            "在庫": on_hand,
            "発注残": pipeline,
            # 発注残を足したものが「実質いま持っている量」。引き忘れると二重発注になる
            "在庫ポジション": on_hand + pipeline,
        }

    def reset(self) -> None:
        config = self.config
        #: [残り日数, 個数] の列。先頭が最も古い
        self._batches: deque[list[float]] = deque()
        self._pipeline: dict[pd.Timestamp, float] = {}
        initial = config.base_demand * config.cover_days
        if initial > 0:
            self._batches.append([float(config.shelf_life), float(initial)])

    def apply(self, t: pd.Timestamp, decision: Any) -> dict[str, Any]:
        config = self.config
        opening = self._on_hand()

        # 発注が先。リードタイム 0 なら、いま出した発注がこの直後に届く
        order = float(max(0.0, round(float(decision or 0.0))))
        if order > 0:
            arrival = t + pd.Timedelta(days=config.lead_time)
            self._pipeline[arrival] = self._pipeline.get(arrival, 0.0) + order

        arrived = float(self._pipeline.pop(t, 0.0))
        if arrived > 0:
            self._batches.append([float(config.shelf_life), arrived])

        demand = float(self.frame.at[t, "需要"])
        sold = self._consume(demand)
        shortage = demand - sold

        expired = self._age_one_day()
        if t == self._sim_index[-1]:
            # 期末に残った在庫は費用に落とす。落とさないと「最後に大量発注して
            # 原価を計上しない」方策が得をしてしまい、比較が公平でなくなる
            expired += self._flush()

        closing = self._on_hand()

        revenue = config.price * sold
        cogs = -config.cost * sold
        waste = -config.cost * expired
        holding = -config.holding * closing
        stockout = -config.stockout_penalty * shortage

        return {
            "需要": demand,
            "販売": sold,
            "欠品": shortage,
            "廃棄": expired,
            "発注": order,
            "入荷": arrived,
            "期首在庫": opening,
            "期末在庫": closing,
            "売上": revenue,
            "売上原価": cogs,
            "廃棄損": waste,
            "保管費": holding,
            "欠品損": stockout,
            "利益": revenue + cogs + waste + holding + stockout,
        }

    # -- 在庫の出し入れ --------------------------------------------
    def _on_hand(self) -> float:
        return float(sum(batch[1] for batch in self._batches))

    def _consume(self, demand: float) -> float:
        """古い箱から順に取り出す（先入れ先出し）。"""
        remaining = demand
        sold = 0.0
        while remaining > 0 and self._batches:
            batch = self._batches[0]
            take = min(batch[1], remaining)
            batch[1] -= take
            sold += take
            remaining -= take
            if batch[1] <= 1e-9:
                self._batches.popleft()
        return sold

    def _age_one_day(self) -> float:
        """1 日ぶん古くし、期限切れを捨てる。"""
        expired = 0.0
        keep: deque[list[float]] = deque()
        for batch in self._batches:
            batch[0] -= 1
            if batch[0] <= 0:
                expired += batch[1]
            else:
                keep.append(batch)
        self._batches = keep
        return expired

    def _flush(self) -> float:
        left = self._on_hand()
        self._batches.clear()
        return left


# ======================================================================
# 特徴量
# ======================================================================


def make_features(
    frame: pd.DataFrame, config: InventoryConfig, n_lags: int = 7
) -> tuple[pd.DataFrame, pd.Series]:
    """発注日を index にした特徴量と、リードタイム需要を作る。

    時系列ラボの `mllab/models/timeseries.py` にも `make_features` があるが、
    こちらは目的が違うので別に用意している。違いは 2 点。

    1. **目的変数がカバー期間の需要合計** — 1 点先の値ではなく
       `Σ_{i=0..L} D_{t+i}`。発注が守らなければならないのはこの期間ぶんだから。
    2. **外生変数は予測時点で分かるものだけ** — 販促は事前に決まっており、
       数日先の気温は天気予報で分かる。だからカバー期間の販促日数と平均気温は
       特徴量に入れてよい。**その期間に実際に売れた数は入れてはいけない。**

    発注は朝、その日の需要を見る前に決める。したがって
    **行 `t` の需要ラグは `t-1` 日以前だけ**で、当日の需要は入らない。
    ここに 1 日ぶんのずれを作ると、検証だけ良くて本番で崩れるモデルができる。
    """
    cover = config.cover_days
    demand = frame["需要"]
    #: 当日の需要はまだ観測できていない。すべて前日までで作る
    past = demand.shift(1)

    features = pd.DataFrame(index=frame.index)
    for lag in range(1, n_lags + 1):
        features[f"{lag}日前"] = demand.shift(lag)

    features["移動平均7"] = past.rolling(7).mean()
    features[f"移動平均{SIGMA_WINDOW}"] = past.rolling(SIGMA_WINDOW).mean()
    features[f"移動標準偏差{SIGMA_WINDOW}"] = past.rolling(SIGMA_WINDOW).std()

    # カバー期間（t 〜 t+L）について、事前に分かっていること
    features["カバー期間の週末日数"] = _window_sum(frame["週末"].astype(float), cover)
    features["カバー期間の販促日数"] = _window_sum(frame["販促"].astype(float), cover)
    features["カバー期間の平均気温"] = _window_sum(frame["気温"], cover) / cover

    features["曜日"] = frame.index.dayofweek
    day_of_year = frame.index.dayofyear
    features["年内位置_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["年内位置_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    target = _window_sum(demand, cover).rename("カバー期間需要")

    combined = features.join(target).dropna()
    return combined.drop(columns="カバー期間需要"), combined["カバー期間需要"]


def _window_sum(series: pd.Series, window: int) -> pd.Series:
    """`t` から `t+window-1` までの合計（当日を含む）。

    未来を見てよいのは目的変数と、予測時点で分かっている外生変数だけ。
    """
    return series.iloc[::-1].rolling(window).sum().iloc[::-1]


# ======================================================================
# 予測
# ======================================================================


@dataclass(frozen=True)
class ForecastSpec:
    """予測手法 1 つぶんの定義。"""

    key: str
    label: str
    summary: str
    #: 分位点を直接推定できるか。できない手法は正規分布を仮定して平均 + zσ で代用する
    native_quantile: bool = False


FORECASTS: dict[str, ForecastSpec] = {
    "lightgbm": ForecastSpec(
        "lightgbm", "LightGBM",
        "非線形も交互作用も拾える。分位点回帰（pinball loss）で分位点を直接学習できる。",
        native_quantile=True,
    ),
    "ridge": ForecastSpec(
        "ridge", "線形モデル（Ridge）",
        "係数が読める。分位点は直接出せないので、正規分布を仮定して平均 + zσ で代用する。",
    ),
    "moving_average": ForecastSpec(
        "moving_average", f"移動平均{SIGMA_WINDOW}日",
        "学習しない基準線。直近の平均をそのまま使う。これを超えられない予測に価値はない。",
    ),
}

#: 予測誤差ゼロの世界。方策の上限を測るために使う（手法ではない）
ORACLE = "oracle"


@dataclass(frozen=True)
class ForecastConfig:
    method: str = "lightgbm"
    #: 何日ごとにモデルを学習し直すか。毎日やる現場は無い
    refit_every: int = 30
    n_lags: int = 7
    seed: int = 0


def forecast(
    world: InventoryWorld, config: InventoryConfig, spec: ForecastConfig
) -> pd.DataFrame:
    """発注日ごとの予測を、時間を守って作る。

    **学習に使ってよいのは、その時点で目的変数まで確定している行だけ。**
    発注日 `s` の目的変数は `s + cover` 日目に確定するので、
    `t` 日に学習できるのは `s ≤ t − cover` の行に限られる。
    ここを緩めると、検証だけ良くて本番で崩れるモデルができる。

    Returns:
        シミュレーション期間を index に持つフレーム。
        `平均` / `標準偏差` / `分位点` と、評価用の `実績カバー期間需要`。
        `実績` で始まる列は方策に渡らない（`mllab/sim/core.py` の `drop_truth`）。
    """
    sim_dates = world.dates()
    cover = config.cover_days
    ratio = config.critical_ratio
    z = z_for_service_level(ratio)
    truth = _window_sum(world.frame["需要"], cover).reindex(sim_dates)

    if spec.method == ORACLE:
        return pd.DataFrame(
            {
                "平均": truth,
                "標準偏差": 0.0,
                "分位点": truth,
                "実績カバー期間需要": truth,
            }
        )

    # σ は需要そのものの直近のばらつきから作る。モデル残差から作ると、
    # 学習データに当てはまりすぎたモデルほど σ を小さく見積もってしまう
    past = world.frame["需要"].shift(1)
    sigma_daily = past.rolling(SIGMA_WINDOW).std()
    sigma = (sigma_daily * np.sqrt(cover)).reindex(sim_dates).ffill().fillna(0.0)
    fallback = config.base_demand * cover

    if spec.method == "moving_average":
        base = past.rolling(SIGMA_WINDOW).mean() * cover
        mean = base.reindex(sim_dates).ffill().fillna(fallback)
        quantile = mean + z * sigma
    else:
        X, y = make_features(world.frame, config, spec.n_lags)
        mean = pd.Series(np.nan, index=sim_dates, dtype=float)
        quantile = pd.Series(np.nan, index=sim_dates, dtype=float)

        for start in range(0, len(sim_dates), max(1, spec.refit_every)):
            block = sim_dates[start : start + spec.refit_every]
            cutoff = block[0] - pd.Timedelta(days=cover)
            train = X.index <= cutoff
            if int(train.sum()) < 60:
                continue
            X_block = X.reindex(block).dropna()
            if X_block.empty:
                continue

            mean_model, quantile_model = _fit(spec, ratio, X[train], y[train])
            mean.loc[X_block.index] = mean_model.predict(X_block)
            if quantile_model is None:
                # 分位点を出せない手法は、正規分布を仮定して平均から作る
                quantile.loc[X_block.index] = (
                    mean.loc[X_block.index] + z * sigma.loc[X_block.index]
                )
            else:
                quantile.loc[X_block.index] = quantile_model.predict(X_block)

        mean = mean.ffill().fillna(fallback)
        quantile = quantile.ffill().fillna(fallback)

    return pd.DataFrame(
        {
            "平均": mean.clip(lower=0),
            "標準偏差": sigma,
            "分位点": quantile.clip(lower=0),
            "実績カバー期間需要": truth,
        }
    )


def _fit(spec: ForecastConfig, ratio: float, X: pd.DataFrame, y: pd.Series):
    """平均を当てるモデルと、分位点を当てるモデルを作る。"""
    if spec.method == "ridge":
        return Ridge(alpha=1.0).fit(X, y), None

    common = dict(
        n_estimators=250, learning_rate=0.05, num_leaves=31,
        random_state=spec.seed, verbose=-1,
    )
    mean_model = lgb.LGBMRegressor(**common).fit(X, y)
    # pinball loss。alpha を臨界比にすると、まさに欲しい分位点が直接手に入る
    quantile_model = lgb.LGBMRegressor(objective="quantile", alpha=ratio, **common).fit(X, y)
    return mean_model, quantile_model


def accuracy(signals: pd.DataFrame) -> dict[str, float]:
    """予測の当たり具合。事業指標とは別に、必ず並べて出す。"""
    truth = signals["実績カバー期間需要"]
    valid = truth.notna()
    error = signals.loc[valid, "平均"] - truth[valid]
    if error.empty:
        return {"RMSE": float("nan"), "MAE": float("nan"), "偏り": float("nan")}
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "偏り": float(np.mean(error)),
    }


# ======================================================================
# 方策
# ======================================================================


@dataclass(frozen=True)
class PolicySpec:
    """方策 1 つぶんの定義。"""

    key: str
    label: str
    summary: str
    #: 「予測を活かしきった上限」として、損失の分解で基準になる方策か
    is_reference: bool = False


POLICIES: dict[str, PolicySpec] = {
    "fixed": PolicySpec(
        "fixed", "固定量",
        "毎日同じ量まで補充する。予測をまったく使わない下限。",
    ),
    "mean": PolicySpec(
        "mean", "平均予測",
        "予測された平均需要ぶんだけ持つ。RMSE を最小にする予測をそのまま使う、最も素直な方策。",
    ),
    "safety": PolicySpec(
        "safety", "安全在庫",
        "平均 + zσ。サービスレベルを人が決める、在庫管理の教科書どおりのやり方。",
    ),
    "quantile": PolicySpec(
        "quantile", "分位点（臨界比）",
        "CR = Cu/(Cu+Co) の分位点を目標にする。コスト構造から目標を決める唯一の方策。",
        is_reference=True,
    ),
}

#: 損失の分解で「予測を活かしきった上限」として使う方策
REFERENCE_POLICY = next(k for k, v in POLICIES.items() if v.is_reference)

ORACLE_LABEL = "オラクル（完全予見）"


def make_policy(
    key: str,
    config: InventoryConfig,
    service_level: float = 0.95,
    fixed_quantity: float | None = None,
) -> Policy:
    """方策を作る。返る関数は、状態と予測しか受け取れない。"""
    cover = config.cover_days
    z = z_for_service_level(service_level)
    fixed = fixed_quantity if fixed_quantity is not None else config.base_demand * cover

    def target(signal: pd.Series | None) -> float:
        if key == "fixed" or signal is None:
            return float(fixed)
        if key == "mean":
            return float(signal["平均"])
        if key == "safety":
            return float(signal["平均"] + z * signal["標準偏差"])
        if key == "quantile":
            return float(signal["分位点"])
        raise ValueError(f"未知の方策: {key}")

    def policy(state: dict[str, float], signal: pd.Series | None) -> float:
        return base_stock_order(target(signal), state["在庫ポジション"])

    return policy


# ======================================================================
# まとめて回す
# ======================================================================


def new_ledger() -> Ledger:
    return Ledger(breakdown=BREAKDOWN)


def run(
    world: InventoryWorld,
    config: InventoryConfig,
    signals: pd.DataFrame,
    policy_key: str,
    service_level: float = 0.95,
    fixed_quantity: float | None = None,
) -> Ledger:
    """1 つの方策で 1 回シミュレーションする。"""
    policy = make_policy(policy_key, config, service_level, fixed_quantity)
    return simulate(world, policy, signals, new_ledger())


def summarize(ledger: Ledger) -> dict[str, float]:
    """事業の側から見た成績。ML 指標と必ず並べて出す。"""
    frame = ledger.frame()
    if frame.empty:
        return {}
    demand = float(frame["需要"].sum())
    received = float(frame["入荷"].sum())
    return {
        "利益": float(frame["利益"].sum()),
        "売上": float(frame["売上"].sum()),
        "欠品率": float(frame["欠品"].sum() / demand) if demand else 0.0,
        "廃棄率": float(frame["廃棄"].sum() / received) if received else 0.0,
        "平均在庫": float(frame["期末在庫"].mean()),
        "欠品日数": float((frame["欠品"] > 0).sum()),
    }


def compare_policies(
    config: InventoryConfig,
    spec: ForecastConfig,
    service_level: float = 0.95,
    fixed_quantity: float | None = None,
    policy_keys: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Ledger], pd.DataFrame]:
    """同じ世界・同じ予測で、方策を横並びに比べる。

    予測は方策に影響されないので **1 回だけ計算して使い回す**。
    こうしないと、方策の差なのか予測の揺れなのかが分からなくなる。

    Returns:
        (成績の表, 方策ごとの台帳, 予測フレーム)。
        表にはオラクル（予測誤差ゼロ）の行も含む。
    """
    frame = generate(config)
    world = InventoryWorld(config, frame)
    signals = forecast(world, config, spec)

    keys = policy_keys or tuple(POLICIES)
    ledgers: dict[str, Ledger] = {}
    rows: list[dict[str, Any]] = []

    for key in keys:
        ledger = run(world, config, signals, key, service_level, fixed_quantity)
        ledgers[POLICIES[key].label] = ledger
        rows.append({"方策": POLICIES[key].label, **summarize(ledger)})

    # オラクル = 予測誤差ゼロの世界で回したもの。予測が完璧なら分位点も安全在庫も
    # 同じ答えになるので、どの方策で回しても結果は変わらない
    oracle_signals = forecast(world, config, ForecastConfig(method=ORACLE))
    oracle_ledger = run(world, config, oracle_signals, "mean")
    ledgers[ORACLE_LABEL] = oracle_ledger
    rows.append({"方策": ORACLE_LABEL, **summarize(oracle_ledger)})

    return pd.DataFrame(rows), ledgers, signals


def accuracy_vs_profit(
    config: InventoryConfig,
    spec: ForecastConfig,
    service_level: float = 0.95,
    fixed_quantity: float | None = None,
) -> pd.DataFrame:
    """予測手法 × 方策の総当たり。**このラボの結論が出る表。**

    横軸に予測精度、縦軸に利益を取って散布図にすると、点が右下がりの線に
    ならないことが見える。同じ RMSE でも方策次第で利益は大きく変わり、
    RMSE を下げても利益がついてこない領域がある。

    Returns:
        予測手法・方策ごとに 1 行。`RMSE` と `利益` を持つ。
    """
    frame = generate(config)
    world = InventoryWorld(config, frame)

    methods = [(key, FORECASTS[key].label) for key in FORECASTS]
    methods.append((ORACLE, ORACLE_LABEL))

    rows: list[dict[str, Any]] = []
    for method, method_label in methods:
        signals = forecast(
            world, config,
            ForecastConfig(method, spec.refit_every, spec.n_lags, spec.seed),
        )
        scores = accuracy(signals)
        for policy_key, policy in POLICIES.items():
            if method == ORACLE and policy_key == "fixed":
                # 予測を使わない方策は、予測が完璧でも何も変わらない。
                # 散布図に同じ点が重なるだけなので省く
                continue
            ledger = run(world, config, signals, policy_key, service_level, fixed_quantity)
            rows.append(
                {
                    "予測手法": method_label,
                    "方策": policy.label,
                    "RMSE": scores["RMSE"],
                    **summarize(ledger),
                }
            )
    return pd.DataFrame(rows)
