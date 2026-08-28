"""コネクタの共通の型と HTTP まわり。

ネットワーク取得は失敗するのが普通なので、例外を投げっぱなしにせず
`FetchResult` に包んで返す（プロジェクトの Result 型パターン）。
画面側は `ok` を見るだけでよく、通信エラーでページが落ちない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import httpx
import pandas as pd

#: 相手のサーバーに迷惑をかけないための既定タイムアウト（秒）
TIMEOUT = 20.0

#: 自分が何者かを名乗る。公開 API を叩くときの最低限の礼儀。
USER_AGENT = "ML-Lab/0.1 (personal learning project)"


@dataclass(frozen=True)
class FetchResult:
    """取得の結果。成功なら frame、失敗なら error にだけ中身が入る。"""

    ok: bool
    frame: pd.DataFrame | None = None
    error: str = ""
    #: 保存時にメタ情報として残したい取得条件
    params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def success(frame: pd.DataFrame, **params: Any) -> "FetchResult":
        return FetchResult(ok=True, frame=frame, params=params)

    @staticmethod
    def failure(error: str, **params: Any) -> "FetchResult":
        return FetchResult(ok=False, error=error, params=params)


@dataclass(frozen=True)
class Option:
    """コネクタが画面に出す入力欄 1 つぶんの定義。

    `ParamSpec`（モデルのハイパラ）と同じ考え方で、UI はこの定義から
    機械的にウィジェットを描く。コネクタを足せば画面は勝手に増える。
    """

    key: str
    label: str
    kind: str  # "text" | "int" | "date" | "select" | "multiselect"
    default: Any = None
    options: tuple[Any, ...] = ()
    #: select / multiselect の表示名（値 → ラベル）
    labels: dict[Any, str] = field(default_factory=dict)
    min: int | None = None
    max: int | None = None
    help: str = ""


class Fetcher(Protocol):
    def __call__(self, **kwargs: Any) -> FetchResult: ...


@dataclass(frozen=True)
class Connector:
    """データ取得元 1 つぶんの定義。"""

    key: str
    label: str
    #: timeseries / table / text のいずれか。カタログの絞り込みに使う。
    domain: str
    source: str
    description: str
    fetch: Fetcher
    options: tuple[Option, ...] = ()
    #: 保存名の候補を作る関数。同じ取得元でも条件ごとに別データセットにする。
    name_for: Callable[[dict[str, Any]], str] = lambda _: "dataset"
    #: API キーなどが要る場合の案内。不要なら空。
    requires_key: str = ""
    #: 無料で使えるかどうかの但し書き
    terms: str = ""

    def defaults(self) -> dict[str, Any]:
        return {o.key: o.default for o in self.options}


DOMAIN_LABELS: dict[str, str] = {
    "timeseries": "時系列",
    "table": "テーブル",
    "text": "テキスト",
}


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """JSON を取る。失敗は httpx の例外として上に投げる。"""
    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(url, params=params, follow_redirects=True)
        response.raise_for_status()
        return response.json()


def get_text(url: str, params: dict[str, Any] | None = None) -> str:
    """テキスト（CSV など）を取る。"""
    with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(url, params=params, follow_redirects=True)
        response.raise_for_status()
        return response.text


def explain_http_error(exc: Exception, source: str) -> str:
    """通信の例外を、画面にそのまま出せる日本語にする。"""
    if isinstance(exc, httpx.TimeoutException):
        return f"{source} への接続がタイムアウトしました（{TIMEOUT:.0f} 秒）。時間をおいて試してください。"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 429:
            return f"{source} からのレート制限です（429）。しばらく待ってから取得してください。"
        if code == 404:
            return f"{source} に該当データがありません（404）。条件を見直してください。"
        return f"{source} がエラーを返しました（HTTP {code}）。"
    if isinstance(exc, httpx.ConnectError):
        return f"{source} に接続できませんでした。ネットワーク接続を確認してください。"
    if isinstance(exc, httpx.HTTPError):
        return f"{source} との通信に失敗しました: {exc}"
    return f"{source} の取得中に予期しないエラー: {type(exc).__name__}: {exc}"
