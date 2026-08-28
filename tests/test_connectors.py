"""コネクタのテスト。

通信は原則モックする。取得元が落ちていても CI やローカルのテストが赤くならず、
かつ「返ってきた JSON をどう表に直すか」という自分たちのロジックを検証できる。

実際に外部へ繋ぐテストは `-m network` を付けたときだけ動く。
"""

from __future__ import annotations

import httpx
import pandas as pd
import pytest

from mllab.data.connectors import CONNECTORS, base, estat, markets, news, opendata, weather


# ---- レジストリ全体の約束ごと -----------------------------------------

def test_registry_has_all_domains() -> None:
    domains = {c.domain for c in CONNECTORS.values()}
    assert domains == {"table", "timeseries", "text"}


@pytest.mark.parametrize("key", list(CONNECTORS))
def test_connector_is_well_formed(key: str) -> None:
    c = CONNECTORS[key]
    assert c.key == key
    assert c.label and c.source and c.description and c.terms
    assert c.domain in base.DOMAIN_LABELS
    # 既定値だけで保存名を作れること（UI が起動時に呼ぶ）
    name = c.name_for(c.defaults())
    from mllab.data.store import NAME_PATTERN

    assert NAME_PATTERN.match(name), f"{key} の既定の保存名が不正: {name}"


@pytest.mark.parametrize("key", list(CONNECTORS))
def test_option_defaults_are_consistent(key: str) -> None:
    for option in CONNECTORS[key].options:
        assert option.kind in ("text", "int", "date", "select", "multiselect")
        if option.kind == "select":
            assert option.default in option.options
        if option.kind == "multiselect":
            assert set(option.default or ()) <= set(option.options)
        if option.kind == "int":
            assert option.min <= option.default <= option.max


# ---- 公開データセット（通信不要なので実物を使う） ---------------------

@pytest.mark.parametrize("key", [k for k in opendata.DATASETS if k != "california_housing"])
def test_opendata_bundled_datasets(key: str) -> None:
    result = opendata.fetch(dataset=key)
    assert result.ok, result.error
    assert len(result.frame) > 0
    assert len(result.frame.columns) >= 2
    assert result.params["dataset"] == key


def test_opendata_rejects_unknown() -> None:
    result = opendata.fetch(dataset="nope")
    assert not result.ok and "未知" in result.error


# ---- 気象（Open-Meteo をモック） --------------------------------------

def fake_daily_payload() -> dict:
    return {
        "daily": {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "temperature_2m_max": [8.1, 9.2, 7.0],
            "temperature_2m_min": [1.0, 2.5, 0.4],
            "temperature_2m_mean": [4.5, 5.8, 3.7],
            "precipitation_sum": [0.0, 3.2, 0.5],
            "windspeed_10m_max": [12.0, 15.5, 9.8],
            "sunshine_duration": [36000.0, 18000.0, 27000.0],
        }
    }


def test_weather_converts_payload_to_frame(monkeypatch) -> None:
    monkeypatch.setattr(weather, "get_json", lambda url, params: fake_daily_payload())
    result = weather.fetch(city="tokyo", start="2024-01-01", end="2024-01-03")

    assert result.ok, result.error
    frame = result.frame
    assert list(frame.columns)[:3] == ["日付", "都市", "最高気温"]
    assert len(frame) == 3
    assert frame["都市"].unique().tolist() == ["東京"]
    assert pd.api.types.is_datetime64_any_dtype(frame["日付"])
    # 日照時間は秒で来るので時間に直している（36000 秒 = 10 時間）
    assert frame.loc[0, "日照時間"] == pytest.approx(10.0)


def test_weather_rejects_unknown_city() -> None:
    result = weather.fetch(city="atlantis")
    assert not result.ok and "未知の都市" in result.error


def test_weather_rejects_reversed_dates() -> None:
    result = weather.fetch(city="tokyo", start="2024-05-01", end="2024-01-01")
    assert not result.ok and "開始日" in result.error


def test_weather_reports_empty_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        weather, "get_json", lambda url, params: {"reason": "範囲外です"}
    )
    result = weather.fetch(city="tokyo", start="1900-01-01", end="1900-01-02")
    assert not result.ok and "範囲外です" in result.error


def test_weather_surfaces_timeout_in_japanese(monkeypatch) -> None:
    def boom(url, params):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(weather, "get_json", boom)
    result = weather.fetch()
    assert not result.ok
    assert "タイムアウト" in result.error


def test_weather_name_is_per_city() -> None:
    assert weather.CONNECTOR.name_for({"city": "osaka"}) == "weather_osaka"


# ---- 株価（Yahoo Finance をモック） -----------------------------------

def fake_chart_payload(n: int = 4) -> dict:
    base_ts = 1_704_067_200  # 2024-01-01
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"currency": "JPY", "exchangeTimezoneName": "Asia/Tokyo"},
                    "timestamp": [base_ts + i * 86400 for i in range(n)],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 102.0, None, 105.0][:n],
                                "high": [103.0, 104.0, None, 108.0][:n],
                                "low": [99.0, 101.0, None, 104.0][:n],
                                "close": [102.0, 103.0, None, 106.0][:n],
                                "volume": [1000, 1200, None, 1500][:n],
                            }
                        ]
                    },
                }
            ],
        }
    }


def test_markets_converts_payload_and_adds_return(monkeypatch) -> None:
    monkeypatch.setattr(markets, "get_json", lambda url, params: fake_chart_payload())
    result = markets.fetch(symbol="^N225", range="1y")

    assert result.ok, result.error
    frame = result.frame
    assert "前日比(%)" in frame.columns
    # 終値が null の行（休場日）は落としている
    assert len(frame) == 3
    assert frame["終値"].notna().all()
    assert frame.loc[1, "前日比(%)"] == pytest.approx((103.0 - 102.0) / 102.0 * 100)
    assert frame["銘柄"].unique().tolist() == ["日経平均株価"]
    assert result.params["currency"] == "JPY"


def test_markets_rejects_blank_symbol() -> None:
    result = markets.fetch(symbol="  ")
    assert not result.ok and "銘柄コード" in result.error


def test_markets_reports_unknown_symbol(monkeypatch) -> None:
    monkeypatch.setattr(
        markets, "get_json", lambda url, params: {"chart": {"error": {"code": "Not Found"}}}
    )
    result = markets.fetch(symbol="NOPE")
    assert not result.ok and "見つかりません" in result.error


def test_markets_reports_all_null_closes(monkeypatch) -> None:
    payload = fake_chart_payload()
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"] = [None] * 4
    monkeypatch.setattr(markets, "get_json", lambda url, params: payload)
    result = markets.fetch(symbol="^N225")
    assert not result.ok and "終値" in result.error


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("^N225", "market_n225"),   # 記号は落として前後のアンダースコアも削る
        ("7203.T", "market_7203_t"),
        ("USDJPY=X", "market_usdjpy_x"),
        ("BTC-JPY", "market_btc_jpy"),
    ],
)
def test_markets_name_is_sql_safe(symbol: str, expected: str) -> None:
    from mllab.data.store import NAME_PATTERN

    name = markets.CONNECTOR.name_for({"symbol": symbol})
    assert name == expected
    assert NAME_PATTERN.match(name)


# ---- ニュース（RSS をモック） -----------------------------------------

RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>test</title>
<item>
  <title>テスト見出し A</title>
  <link>https://example.com/a</link>
  <description>&lt;p&gt;要約 A&lt;/p&gt;</description>
  <pubDate>Mon, 01 Jan 2024 09:00:00 +0000</pubDate>
</item>
<item>
  <title>テスト見出し B</title>
  <link>https://example.com/b</link>
  <description>要約 B</description>
  <pubDate>Tue, 02 Jan 2024 09:00:00 +0000</pubDate>
</item>
</channel></rss>""".encode("utf-8")


class FakeClient:
    """httpx.Client の代わり。常に同じ RSS を返す。"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def get(self, url):
        return httpx.Response(200, content=RSS_XML, request=httpx.Request("GET", url))


def test_news_parses_feed(monkeypatch) -> None:
    monkeypatch.setattr(news.httpx, "Client", FakeClient)
    result = news.fetch(feeds=["nhk_main"], merge=False)

    assert result.ok, result.error
    frame = result.frame
    assert list(frame.columns) == ["配信元", "配信日時", "見出し", "要約", "URL", "取得日時"]
    assert len(frame) == 2
    # HTML タグは要約から落としている
    assert "<p>" not in frame["要約"].iloc[0]
    assert frame["配信元"].unique().tolist() == ["NHK 主要ニュース"]


def test_news_merge_accumulates_without_duplicates(monkeypatch, tmp_path) -> None:
    """2 回取得しても、同じ記事は URL で 1 件にまとまる。

    RSS は最新の数十件しか返さないため、繰り返し取って溜める前提の作りになっている。
    """
    from mllab.data import store

    monkeypatch.setattr(store, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(news.httpx, "Client", FakeClient)
    store.ensure_dirs()

    first = news.fetch(feeds=["nhk_main"], merge=True)
    assert first.ok and len(first.frame) == 2
    store.save(
        "news_nhk_main", first.frame,
        label="ニュース", source="テスト", domain="text",
    )

    # 同じ内容をもう一度取得しても増えない
    second = news.fetch(feeds=["nhk_main"], merge=True)
    assert second.ok
    assert len(second.frame) == 2
    assert second.params["fetched_now"] == 2
    assert second.params["total_after_merge"] == 2


def test_news_requires_at_least_one_feed() -> None:
    result = news.fetch(feeds=[], merge=False)
    assert not result.ok and "フィード" in result.error


def test_news_reports_failure_per_feed(monkeypatch) -> None:
    class FailingClient(FakeClient):
        def get(self, url):
            raise httpx.ConnectError("no route")

    monkeypatch.setattr(news.httpx, "Client", FailingClient)
    result = news.fetch(feeds=["nhk_main"], merge=False)
    assert not result.ok
    assert "接続できませんでした" in result.error


def test_news_clean_strips_markup() -> None:
    assert news._clean("<p>あ&amp;い</p>  \n う") == "あ&い う"


# ---- e-Stat（鍵が無い前提の分岐） -------------------------------------

def test_estat_without_key_explains_how_to_get_one(monkeypatch) -> None:
    monkeypatch.setattr(estat, "app_id", lambda: "")
    result = estat.fetch(stats_data_id="0003448233")
    assert not result.ok
    assert "利用登録" in result.error
    assert "ESTAT_APP_ID" in result.error


def test_estat_requires_table_id(monkeypatch) -> None:
    monkeypatch.setattr(estat, "app_id", lambda: "dummy-key")
    result = estat.fetch(stats_data_id="")
    assert not result.ok and "統計表 ID" in result.error


def test_estat_reports_api_error(monkeypatch) -> None:
    monkeypatch.setattr(estat, "app_id", lambda: "dummy-key")
    monkeypatch.setattr(
        estat,
        "get_json",
        lambda url, params: {
            "GET_STATS_DATA": {"RESULT": {"STATUS": 100, "ERROR_MSG": "IDが不正です"}}
        },
    )
    result = estat.fetch(stats_data_id="bad")
    assert not result.ok and "IDが不正です" in result.error


def test_estat_maps_codes_to_labels(monkeypatch) -> None:
    """コードのままでは何のデータか分からないので、ラベルに変換している。"""
    monkeypatch.setattr(estat, "app_id", lambda: "dummy-key")
    monkeypatch.setattr(
        estat,
        "get_json",
        lambda url, params: {
            "GET_STATS_DATA": {
                "RESULT": {"STATUS": 0},
                "STATISTICAL_DATA": {
                    "TABLE_INF": {"TITLE": {"$": "人口推計"}},
                    "CLASS_INF": {
                        "CLASS_OBJ": [
                            {
                                "@id": "area",
                                "@name": "地域",
                                "CLASS": [
                                    {"@code": "13", "@name": "東京都"},
                                    {"@code": "27", "@name": "大阪府"},
                                ],
                            }
                        ]
                    },
                    "DATA_INF": {
                        "VALUE": [
                            {"@area": "13", "@unit": "人", "$": "14000000"},
                            {"@area": "27", "@unit": "人", "$": "8800000"},
                        ]
                    },
                },
            }
        },
    )
    result = estat.fetch(stats_data_id="0003448233")
    assert result.ok, result.error
    frame = result.frame
    assert set(frame.columns) == {"地域", "単位", "値"}
    assert frame["地域"].tolist() == ["東京都", "大阪府"]
    assert frame["値"].tolist() == [14000000, 8800000]
    assert result.params["title"] == "人口推計"


def test_estat_load_env_reads_file(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env.local"
    env.write_text("# コメント\nESTAT_APP_ID = abc123\nOTHER=x\n", encoding="utf-8")
    monkeypatch.setattr(estat, "ENV_FILE", env)
    monkeypatch.delenv("ESTAT_APP_ID", raising=False)
    assert estat.app_id() == "abc123"


def test_estat_env_does_not_override_existing(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env.local"
    env.write_text("ESTAT_APP_ID=from_file\n", encoding="utf-8")
    monkeypatch.setattr(estat, "ENV_FILE", env)
    monkeypatch.setenv("ESTAT_APP_ID", "from_environment")
    assert estat.app_id() == "from_environment"


# ---- エラーメッセージの日本語化 ---------------------------------------

@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.TimeoutException("x"), "タイムアウト"),
        (httpx.ConnectError("x"), "接続できませんでした"),
        (
            httpx.HTTPStatusError(
                "x",
                request=httpx.Request("GET", "https://e.com"),
                response=httpx.Response(429),
            ),
            "レート制限",
        ),
        (
            httpx.HTTPStatusError(
                "x",
                request=httpx.Request("GET", "https://e.com"),
                response=httpx.Response(404),
            ),
            "該当データがありません",
        ),
        (ValueError("boom"), "予期しないエラー"),
    ],
)
def test_explain_http_error(exc: Exception, expected: str) -> None:
    assert expected in base.explain_http_error(exc, "テスト取得元")


# ---- 実通信（既定では動かさない） -------------------------------------

@pytest.mark.network
def test_live_weather() -> None:
    result = weather.fetch(city="tokyo", start="2024-01-01", end="2024-01-07")
    assert result.ok, result.error
    assert len(result.frame) == 7


@pytest.mark.network
def test_live_markets() -> None:
    result = markets.fetch(symbol="^N225", range="1y")
    assert result.ok, result.error
    assert len(result.frame) > 100


@pytest.mark.network
def test_live_news() -> None:
    result = news.fetch(feeds=["nhk_main"], merge=False)
    assert result.ok, result.error
    assert len(result.frame) > 0
