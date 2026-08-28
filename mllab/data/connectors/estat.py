"""e-Stat（政府統計の総合窓口）の API から統計表を取る。

利用は無料だが、アプリケーション ID（appId）の登録が要る唯一のコネクタ。
ID は環境変数 `ESTAT_APP_ID` から読む。`.env.local` に書いておけば
`load_env()` が拾う。ID をコードに書かないための作り。

出典: https://www.e-stat.go.jp/api/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from mllab.data.connectors.base import (
    Connector,
    FetchResult,
    Option,
    explain_http_error,
    get_json,
)

SOURCE = "e-Stat"
STATS_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
SEARCH_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsList"

ENV_KEY = "ESTAT_APP_ID"
ENV_FILE = Path(__file__).resolve().parents[3] / ".env.local"

HOW_TO_GET_KEY = (
    "e-Stat だけは無料の利用登録が必要です。\n"
    "1. https://www.e-stat.go.jp/mypage/user/preregister でユーザー登録\n"
    "2. マイページ → API 機能（アプリケーション ID 発行）で appId を取得\n"
    "3. プロジェクト直下の `.env.local` に `ESTAT_APP_ID=取得したID` と書く\n"
    "4. このページを再読み込み"
)


def load_env() -> None:
    """`.env.local` を読んで環境変数に載せる。

    依存を増やしたくないので `KEY=VALUE` だけの素朴なパーサ。
    既に環境変数がある場合は上書きしない。
    """
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def app_id() -> str:
    """設定済みの appId を返す。未設定なら空文字。"""
    load_env()
    return os.environ.get(ENV_KEY, "").strip()


def has_key() -> bool:
    return bool(app_id())


def search(keyword: str, limit: int = 30) -> FetchResult:
    """統計表をキーワードで検索し、統計表 ID の一覧を返す。"""
    key = app_id()
    if not key:
        return FetchResult.failure(HOW_TO_GET_KEY)
    if not keyword.strip():
        return FetchResult.failure("検索キーワードを入力してください。")

    try:
        payload = get_json(
            SEARCH_URL,
            {"appId": key, "searchWord": keyword, "limit": int(limit), "lang": "J"},
        )
    except Exception as exc:  # noqa: BLE001
        return FetchResult.failure(explain_http_error(exc, SOURCE))

    root = payload.get("GET_STATS_LIST", {})
    status, message = _api_status(root)
    if status != 0:
        return FetchResult.failure(f"{SOURCE}: {message}")

    tables = root.get("DATALIST_INF", {}).get("TABLE_INF", [])
    if isinstance(tables, dict):
        tables = [tables]
    if not tables:
        return FetchResult.failure(f"「{keyword}」に該当する統計表がありませんでした。")

    frame = pd.DataFrame(
        [
            {
                "統計表ID": t.get("@id", ""),
                "統計名": _text(t.get("STAT_NAME")),
                "表題": _text(t.get("TITLE")),
                "調査年月": _text(t.get("SURVEY_DATE")),
                "行数": t.get("OVERALL_TOTAL_NUMBER", ""),
            }
            for t in tables
        ]
    )
    return FetchResult.success(frame, keyword=keyword, hits=len(frame))


def fetch(stats_data_id: str = "", limit: int = 10000) -> FetchResult:
    """統計表 ID を指定して数値データを取得する。"""
    key = app_id()
    if not key:
        return FetchResult.failure(HOW_TO_GET_KEY)

    stats_data_id = (stats_data_id or "").strip()
    if not stats_data_id:
        return FetchResult.failure(
            "統計表 ID を入力してください。下の検索欄でキーワードから探せます。"
        )

    try:
        payload = get_json(
            STATS_URL,
            {
                "appId": key,
                "statsDataId": stats_data_id,
                "limit": int(limit),
                "lang": "J",
                "metaGetFlg": "Y",
            },
        )
    except Exception as exc:  # noqa: BLE001
        return FetchResult.failure(explain_http_error(exc, SOURCE))

    root = payload.get("GET_STATS_DATA", {})
    status, message = _api_status(root)
    if status != 0:
        return FetchResult.failure(f"{SOURCE}: {message}")

    statistical_data = root.get("STATISTICAL_DATA", {})
    values = statistical_data.get("DATA_INF", {}).get("VALUE", [])
    if isinstance(values, dict):
        values = [values]
    if not values:
        return FetchResult.failure(f"統計表 {stats_data_id} にデータがありませんでした。")

    frame = pd.DataFrame(values)
    frame = _apply_labels(frame, statistical_data)

    title = _text(
        statistical_data.get("TABLE_INF", {}).get("TITLE")
    ) or stats_data_id

    return FetchResult.success(
        frame, stats_data_id=stats_data_id, title=title, rows=len(frame)
    )


def _api_status(root: dict[str, Any]) -> tuple[int, str]:
    result = root.get("RESULT", {})
    return int(result.get("STATUS", -1)), str(result.get("ERROR_MSG", "不明なエラー"))


def _text(node: Any) -> str:
    """e-Stat は文字列だったり `{"$": "..."}` だったりするので均す。"""
    if isinstance(node, dict):
        return str(node.get("$", ""))
    return "" if node is None else str(node)


def _apply_labels(frame: pd.DataFrame, statistical_data: dict[str, Any]) -> pd.DataFrame:
    """コード列を人間が読めるラベルに置き換える。

    e-Stat の値は `@cat01` などのコードで返ってくる。メタ情報の対応表を使って
    ラベル列を足さないと、取り込んでも何のデータか分からない。
    """
    class_objects = (
        statistical_data.get("CLASS_INF", {}).get("CLASS_OBJ", []) or []
    )
    if isinstance(class_objects, dict):
        class_objects = [class_objects]

    renames: dict[str, str] = {}
    for obj in class_objects:
        code = f"@{obj.get('@id', '')}"
        if code not in frame.columns:
            continue
        name = _text(obj.get("@name")) or code.lstrip("@")
        classes = obj.get("CLASS", [])
        if isinstance(classes, dict):
            classes = [classes]
        mapping = {str(c.get("@code")): _text(c.get("@name")) for c in classes}
        frame[code] = frame[code].astype(str).map(mapping).fillna(frame[code])
        renames[code] = name

    # 値そのものの列。`$` のままでは扱いにくいので日本語にして数値化する。
    if "$" in frame.columns:
        frame["$"] = pd.to_numeric(frame["$"], errors="coerce")
        renames["$"] = "値"
    if "@unit" in frame.columns:
        renames["@unit"] = "単位"

    frame = frame.rename(columns=renames)
    # 使い道のない内部列は落とす
    return frame.drop(columns=[c for c in frame.columns if c.startswith("@")], errors="ignore")


def _name(params: dict[str, Any]) -> str:
    raw = str(params.get("stats_data_id", "table")).lower()
    safe = "".join(c if c.isalnum() else "_" for c in raw).strip("_")
    return f"estat_{safe or 'table'}"


CONNECTOR = Connector(
    key="estat",
    label="政府統計 e-Stat",
    domain="table",
    source=f"{SOURCE} (api.e-stat.go.jp)",
    description=(
        "国勢調査・家計調査・人口推計など、日本の公的統計を統計表 ID 指定で取得します。"
        "コードはメタ情報を使って日本語ラベルに変換済み。"
        "実データらしい欠測や表記ゆれがあり、前処理の練習になります。"
    ),
    fetch=fetch,
    options=(
        Option(
            "stats_data_id", "統計表 ID", "text", "",
            help="下の検索欄でキーワードから探して、ID をここに貼り付けます。",
        ),
        Option(
            "limit", "最大取得行数", "int", 10000, min=100, max=100000,
            help="大きな表は数十万行あります。まずは 1 万行程度で様子を見てください。",
        ),
    ),
    name_for=_name,
    requires_key=HOW_TO_GET_KEY,
    terms="利用は無料（要・無料のアプリケーションID登録）",
)
