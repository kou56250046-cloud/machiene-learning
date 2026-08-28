"""RSS フィードから記事の見出しと要約を集める。

テキストラボ（形態素解析・TF-IDF・トピックモデル）の材料。
RSS は各社が公開している配信用のフォーマットなので、スクレイピングと違い
取得してよいことがはっきりしている。

1 回の取得で数十件しか取れないため、日を置いて何度か取ると溜まっていく。
同じ記事は URL で重複排除する。
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx
import pandas as pd

from mllab.data.connectors.base import (
    TIMEOUT,
    USER_AGENT,
    Connector,
    FetchResult,
    Option,
    explain_http_error,
)

SOURCE = "RSS"

#: フィードのキー → (表示名, URL)
FEEDS: dict[str, tuple[str, str]] = {
    "nhk_main": ("NHK 主要ニュース", "https://www.nhk.or.jp/rss/news/cat0.xml"),
    "nhk_society": ("NHK 社会", "https://www.nhk.or.jp/rss/news/cat1.xml"),
    "nhk_science": ("NHK 科学・文化", "https://www.nhk.or.jp/rss/news/cat3.xml"),
    "nhk_economy": ("NHK 経済", "https://www.nhk.or.jp/rss/news/cat5.xml"),
    "nhk_intl": ("NHK 国際", "https://www.nhk.or.jp/rss/news/cat6.xml"),
    "nhk_sports": ("NHK スポーツ", "https://www.nhk.or.jp/rss/news/cat7.xml"),
    "itmedia": ("ITmedia", "https://rss.itmedia.co.jp/rss/2.0/itmedia_all.xml"),
    "hatena": ("はてなブックマーク 人気", "https://b.hatena.ne.jp/hotentry.rss"),
}

_TAG = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    """要約に混ざる HTML タグと実体参照を落とす。"""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", text or ""))).strip()


def _published(entry: Any) -> pd.Timestamp | None:
    parsed = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if not parsed:
        return None
    try:
        return pd.Timestamp(datetime(*parsed[:6], tzinfo=timezone.utc)).tz_convert(
            "Asia/Tokyo"
        ).tz_localize(None)
    except (ValueError, TypeError):
        return None


def fetch(feeds: tuple[str, ...] | list[str] = ("nhk_main",), merge: bool = True) -> FetchResult:
    """選んだフィードから記事を集めて 1 つの表にする。

    Args:
        feeds: `FEEDS` のキーの並び。
        merge: 既に同名で保存済みのデータがあれば結合して重複を除く。
            RSS は最新数十件しか返さないので、繰り返し取って蓄積する前提。
    """
    keys = [k for k in (feeds or []) if k in FEEDS]
    if not keys:
        return FetchResult.failure("フィードを 1 つ以上選んでください。")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    with httpx.Client(
        timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    ) as client:
        for key in keys:
            label, url = FEEDS[key]
            try:
                response = client.get(url)
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{label}: {explain_http_error(exc, SOURCE)}")
                continue

            parsed = feedparser.parse(response.content)
            for entry in parsed.entries:
                title = _clean(getattr(entry, "title", ""))
                if not title:
                    continue
                rows.append(
                    {
                        "配信元": label,
                        "配信日時": _published(entry),
                        "見出し": title,
                        "要約": _clean(getattr(entry, "summary", "")),
                        "URL": getattr(entry, "link", ""),
                    }
                )

    if not rows:
        detail = " / ".join(failures) if failures else "記事が 0 件でした。"
        return FetchResult.failure(f"記事を取得できませんでした。{detail}")

    frame = pd.DataFrame(rows)
    frame["取得日時"] = pd.Timestamp.now().floor("s")

    fetched_now = len(frame)
    if merge:
        frame = _merge_with_saved(_name({"feeds": keys}), frame)

    frame = frame.sort_values("配信日時", ascending=False, na_position="last")
    frame = frame.reset_index(drop=True)

    return FetchResult.success(
        frame,
        feeds=keys,
        feed_labels=[FEEDS[k][0] for k in keys],
        fetched_now=fetched_now,
        total_after_merge=len(frame),
        partial_failures=failures,
    )


def _merge_with_saved(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """保存済みがあれば結合し、URL で重複を落とす。"""
    # 循環 import を避けるため、使うときに読み込む
    from mllab.data import store

    try:
        previous = store.load(name)
    except store.StoreError:
        return frame
    combined = pd.concat([previous, frame], ignore_index=True)
    return combined.drop_duplicates(subset=["URL"], keep="first")


def _name(params: dict[str, Any]) -> str:
    keys = list(params.get("feeds") or [])
    if len(keys) == 1:
        return f"news_{keys[0]}"
    return "news_mixed"


CONNECTOR = Connector(
    key="news",
    label="ニュース記事（RSS）",
    domain="text",
    source=f"{SOURCE} (NHK / ITmedia / はてなブックマーク)",
    description=(
        "各社が公開している RSS から見出しと要約を集めます。"
        "1 回で数十件しか取れませんが、日を空けて何度か取ると蓄積されます"
        "（同じ記事は URL で重複排除）。テキスト解析の材料に使います。"
    ),
    fetch=fetch,
    options=(
        Option(
            "feeds", "フィード", "multiselect", ("nhk_main",),
            options=tuple(FEEDS),
            labels={k: v[0] for k, v in FEEDS.items()},
            help="複数選ぶと 1 つの表にまとまります。ジャンル違いを混ぜると分類の練習になります。",
        ),
        Option(
            "merge", "既存データに追記する", "select", True,
            options=(True, False),
            labels={True: "追記する（推奨）", False: "毎回入れ替える"},
            help="追記なら取得のたびに記事が溜まります。",
        ),
    ),
    name_for=_name,
    terms="各社が公開している配信フォーマット・APIキー不要",
)
