"""Open-Meteo から気象の過去データを取る。

API キー不要・登録不要で、1940 年以降の再解析データを日次で返してくれる。
時系列ラボ（季節性・トレンド・予測）の主力データ源。

出典: https://open-meteo.com/  (非商用利用は無料)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from mllab.data.connectors.base import (
    Connector,
    FetchResult,
    Option,
    explain_http_error,
    get_json,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SOURCE = "Open-Meteo"

#: 主要都市の緯度経度。自分で座標を打たなくても試せるようにしておく。
CITIES: dict[str, tuple[str, float, float]] = {
    "tokyo": ("東京", 35.6895, 139.6917),
    "sapporo": ("札幌", 43.0621, 141.3544),
    "sendai": ("仙台", 38.2682, 140.8694),
    "nagoya": ("名古屋", 35.1815, 136.9066),
    "osaka": ("大阪", 34.6937, 135.5023),
    "fukuoka": ("福岡", 33.5904, 130.4017),
    "naha": ("那覇", 26.2124, 127.6809),
}

#: 取得する日次変数 → 日本語の列名
DAILY_VARIABLES: dict[str, str] = {
    "temperature_2m_max": "最高気温",
    "temperature_2m_min": "最低気温",
    "temperature_2m_mean": "平均気温",
    "precipitation_sum": "降水量",
    "windspeed_10m_max": "最大風速",
    "sunshine_duration": "日照時間",
}


def fetch(
    city: str = "tokyo",
    start: str | date = "2015-01-01",
    end: str | date | None = None,
) -> FetchResult:
    """指定都市の日次気象データを取得する。

    Args:
        city: `CITIES` のキー。
        start: 取得開始日 (YYYY-MM-DD)。
        end: 取得終了日。None なら「3 日前」。再解析データは直近数日ぶん
            まだ確定していないため、余裕を持たせる。
    """
    if city not in CITIES:
        return FetchResult.failure(f"未知の都市です: {city}")

    label, lat, lon = CITIES[city]
    end = end or (date.today() - timedelta(days=3)).isoformat()
    start, end = str(start), str(end)
    if start >= end:
        return FetchResult.failure("開始日は終了日より前にしてください。")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Tokyo",
    }

    try:
        payload = get_json(ARCHIVE_URL, params)
    except Exception as exc:  # noqa: BLE001 - 通信失敗はすべて画面に返す
        return FetchResult.failure(explain_http_error(exc, SOURCE))

    daily = payload.get("daily")
    if not daily or not daily.get("time"):
        reason = payload.get("reason", "日次データが空でした。")
        return FetchResult.failure(f"{SOURCE}: {reason}")

    frame = pd.DataFrame({"日付": pd.to_datetime(daily["time"])})
    for key, name in DAILY_VARIABLES.items():
        if key in daily:
            frame[name] = pd.to_numeric(pd.Series(daily[key]), errors="coerce")

    # 日照時間は秒で返るので時間に直す。単位が混ざると解析でつまずくため。
    if "日照時間" in frame:
        frame["日照時間"] = frame["日照時間"] / 3600.0

    frame.insert(1, "都市", label)
    frame = frame.dropna(subset=["日付"]).reset_index(drop=True)

    return FetchResult.success(
        frame, city=city, city_label=label, start=start, end=end, latitude=lat, longitude=lon
    )


def _name(params: dict[str, Any]) -> str:
    return f"weather_{params.get('city', 'tokyo')}"


CONNECTOR = Connector(
    key="weather",
    label="気象データ（日次・過去）",
    domain="timeseries",
    source=f"{SOURCE} (archive-api.open-meteo.com)",
    description=(
        "指定した都市の日次の気温・降水量・風速・日照時間。1940 年以降を遡れます。"
        "季節性がはっきり出るので、トレンド分解や時系列予測の練習に向いています。"
    ),
    fetch=fetch,
    options=(
        Option(
            "city", "都市", "select", "tokyo",
            options=tuple(CITIES),
            labels={k: v[0] for k, v in CITIES.items()},
            help="緯度差が大きい札幌と那覇を比べると、季節性の違いが分かりやすいです。",
        ),
        Option(
            "start", "開始日", "date", "2015-01-01",
            help="長く取るほど季節性の推定が安定します（10 年で約 3,650 行）。",
        ),
        Option("end", "終了日", "date", None,
               help="空なら 3 日前まで。再解析データは直近数日が未確定です。"),
    ),
    name_for=_name,
    terms="非商用利用は無料・APIキー不要",
)
