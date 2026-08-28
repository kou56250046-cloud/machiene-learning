"""Yahoo Finance の公開チャート API から株価・指数の日次データを取る。

yfinance ライブラリが内部で叩いているのと同じ公開エンドポイントを
httpx で直接呼ぶ。依存を 1 つ減らせるうえ、pandas のバージョンに
振り回されない。

出典: https://query1.finance.yahoo.com/v8/finance/chart/<symbol>
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from mllab.data.connectors.base import (
    Connector,
    FetchResult,
    Option,
    explain_http_error,
    get_json,
)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
SOURCE = "Yahoo Finance"

#: よく使う銘柄・指数。`.T` が東証、`^` 始まりが指数。
SYMBOLS: dict[str, str] = {
    "^N225": "日経平均株価",
    "^TPX": "TOPIX",
    "USDJPY=X": "ドル円",
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "9984.T": "ソフトバンクグループ",
    "^GSPC": "S&P 500",
    "AAPL": "Apple",
    "BTC-JPY": "ビットコイン（円）",
}

RANGES: dict[str, str] = {
    "1y": "1 年",
    "2y": "2 年",
    "5y": "5 年",
    "10y": "10 年",
    "max": "取得できる全期間",
}


def fetch(symbol: str = "^N225", range: str = "5y") -> FetchResult:  # noqa: A002
    """日次の始値・高値・安値・終値・出来高を取得する。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return FetchResult.failure("銘柄コードを入力してください。")

    try:
        payload = get_json(
            CHART_URL.format(symbol=symbol), {"range": range, "interval": "1d"}
        )
    except Exception as exc:  # noqa: BLE001
        return FetchResult.failure(explain_http_error(exc, SOURCE))

    chart = payload.get("chart") or {}
    if chart.get("error"):
        return FetchResult.failure(
            f"{SOURCE}: 銘柄 {symbol} が見つかりません。"
            "東証なら `7203.T`、指数なら `^N225` のように入力してください。"
        )

    results = chart.get("result") or []
    if not results:
        return FetchResult.failure(f"{SOURCE}: {symbol} のデータが空でした。")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps:
        return FetchResult.failure(
            f"{SOURCE}: {symbol} に該当期間のデータがありません。期間を延ばしてみてください。"
        )

    meta = result.get("meta", {})
    frame = pd.DataFrame(
        {
            "日付": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(
                meta.get("exchangeTimezoneName") or "UTC"
            ).tz_localize(None).normalize(),
            "始値": quote.get("open"),
            "高値": quote.get("high"),
            "安値": quote.get("low"),
            "終値": quote.get("close"),
            "出来高": quote.get("volume"),
        }
    )
    frame.insert(1, "銘柄", SYMBOLS.get(symbol, symbol))

    # 休場日は null が並ぶので落とす。残すと差分やリターンの計算が壊れる。
    frame = frame.dropna(subset=["終値"]).reset_index(drop=True)
    if frame.empty:
        return FetchResult.failure(f"{SOURCE}: {symbol} の有効な終値がありませんでした。")

    # 前日比リターンは時系列ラボで必ず使うので、ここで付けておく
    frame["前日比(%)"] = frame["終値"].pct_change() * 100

    return FetchResult.success(
        frame,
        symbol=symbol,
        symbol_label=SYMBOLS.get(symbol, symbol),
        range=range,
        currency=meta.get("currency"),
    )


def _name(params: dict[str, Any]) -> str:
    """`^N225` のような記号を SQL のテーブル名にできる形へ落とす。"""
    raw = str(params.get("symbol", "asset")).lower()
    safe = "".join(c if c.isalnum() else "_" for c in raw).strip("_")
    return f"market_{safe or 'asset'}"


CONNECTOR = Connector(
    key="markets",
    label="株価・指数（日次）",
    domain="timeseries",
    source=f"{SOURCE} (query1.finance.yahoo.com)",
    description=(
        "株価指数・個別銘柄・為替の日次 OHLCV と前日比リターン。"
        "気象と違って季節性がほぼ無く、ノイズが支配的なデータの例として対照的です。"
    ),
    fetch=fetch,
    options=(
        Option(
            "symbol", "銘柄コード", "text", "^N225",
            help="一覧にないものも直接入力できます。東証は `7203.T`、指数は `^N225` の形式です。",
        ),
        Option(
            "range", "期間", "select", "5y",
            options=tuple(RANGES), labels=RANGES,
        ),
    ),
    name_for=_name,
    terms="個人利用の範囲で無料・APIキー不要",
)
