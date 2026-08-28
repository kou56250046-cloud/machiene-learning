"""データ取得元のレジストリ。

新しい取得元を足すときは、このパッケージにモジュールを 1 つ作って
`CONNECTOR` を定義し、下の `CONNECTORS` に並べるだけでよい。
カタログ画面はこの辞書を見て入力欄を組み立てるので、UI を触る必要はない。
"""

from __future__ import annotations

from mllab.data.connectors import estat, markets, news, opendata, weather
from mllab.data.connectors.base import (
    DOMAIN_LABELS,
    Connector,
    FetchResult,
    Option,
)

#: 画面に出す順。通信不要のものを先頭に置き、鍵が要るものを最後にする。
CONNECTORS: dict[str, Connector] = {
    c.key: c
    for c in (
        opendata.CONNECTOR,
        weather.CONNECTOR,
        markets.CONNECTOR,
        news.CONNECTOR,
        estat.CONNECTOR,
    )
}

__all__ = [
    "CONNECTORS",
    "DOMAIN_LABELS",
    "Connector",
    "FetchResult",
    "Option",
    "estat",
    "markets",
    "news",
    "opendata",
    "weather",
]
