"""データの置き場所と、その上での SQL 検索。

方針:
- 取得したデータは `data/raw/<name>.parquet` に 1 ファイル 1 データセットで置く。
- 由来（いつ・どこから・どんな条件で取ったか）は `<name>.meta.json` に併置する。
  データだけ残って出所が分からなくなるのを防ぐため、必ずセットで書く。
- 読み出しは DuckDB。Parquet を直接 SQL で引けるので、DB へ取り込む手順が要らない。

サーバープロセスは一切立てない。すべてローカルのファイルで完結する。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

#: SQL のテーブル名として使える名前だけ許可する
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class StoreError(RuntimeError):
    """保存・読み出しの失敗。"""


@dataclass(frozen=True)
class StoredDataset:
    """保存済みデータセット 1 件のメタ情報。"""

    name: str
    label: str
    source: str
    domain: str
    description: str
    fetched_at: str
    rows: int
    columns: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / 1024**2 if self.path and self.path.exists() else 0.0

    @property
    def fetched_at_local(self) -> str:
        """保存時刻を見やすい文字列にする。"""
        try:
            dt = datetime.fromisoformat(self.fetched_at)
        except ValueError:
            return self.fetched_at
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def _validate_name(name: str) -> str:
    if not NAME_PATTERN.match(name):
        raise StoreError(
            f"データセット名が不正です: {name!r}"
            "（英小文字で始まり、英小文字・数字・アンダースコアのみ、63 文字以内）"
        )
    return name


def parquet_path(name: str) -> Path:
    return RAW_DIR / f"{_validate_name(name)}.parquet"


def meta_path(name: str) -> Path:
    return RAW_DIR / f"{_validate_name(name)}.meta.json"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save(
    name: str,
    frame: pd.DataFrame,
    *,
    label: str,
    source: str,
    domain: str,
    description: str = "",
    params: dict[str, Any] | None = None,
) -> StoredDataset:
    """データフレームを Parquet で保存し、メタ情報を併置する。

    同名のデータセットがあれば上書きする（取り直しを想定した挙動）。
    """
    _validate_name(name)
    if frame.empty:
        raise StoreError(f"{label}: 取得結果が 0 行でした。保存しません。")

    ensure_dirs()
    path = parquet_path(name)
    frame.to_parquet(path, index=False)

    meta = {
        "name": name,
        "label": label,
        "source": source,
        "domain": domain,
        "description": description,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "params": params or {},
    }
    meta_path(name).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return StoredDataset(**meta, path=path)


def load(name: str) -> pd.DataFrame:
    """保存済みデータセットを読み出す。"""
    path = parquet_path(name)
    if not path.exists():
        raise StoreError(f"データセットが見つかりません: {name}")
    return pd.read_parquet(path)


def describe(name: str) -> StoredDataset:
    """1 件のメタ情報を返す。"""
    mpath = meta_path(name)
    if not mpath.exists():
        raise StoreError(f"メタ情報が見つかりません: {name}")
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    return StoredDataset(**meta, path=parquet_path(name))


def list_datasets() -> list[StoredDataset]:
    """保存済みデータセットを新しい順に返す。

    メタ情報が壊れている・Parquet だけ残っているものは黙って飛ばさず除外する
    （カタログ画面が落ちないように）。
    """
    if not RAW_DIR.exists():
        return []
    out: list[StoredDataset] = []
    for mpath in RAW_DIR.glob("*.meta.json"):
        name = mpath.name.removesuffix(".meta.json")
        if not parquet_path(name).exists():
            continue
        try:
            out.append(describe(name))
        except (StoreError, json.JSONDecodeError, TypeError):
            continue
    return sorted(out, key=lambda d: d.fetched_at, reverse=True)


def delete(name: str) -> None:
    """データセットとそのメタ情報を消す。"""
    parquet_path(name).unlink(missing_ok=True)
    meta_path(name).unlink(missing_ok=True)


# --- DuckDB ------------------------------------------------------------


def connect(names: list[str] | None = None) -> duckdb.DuckDBPyConnection:
    """保存済み Parquet をビューとして登録した DuckDB 接続を返す。

    ビュー名はデータセット名そのもの。DB ファイルは作らず、毎回メモリ上に
    ビューを張り直すだけなので、Parquet を消せば次回から消える。
    """
    con = duckdb.connect(database=":memory:")
    targets = names if names is not None else [d.name for d in list_datasets()]
    for name in targets:
        path = parquet_path(name)
        if path.exists():
            con.execute(
                f'CREATE OR REPLACE VIEW "{name}" AS '
                f"SELECT * FROM read_parquet('{path.as_posix()}')"
            )
    return con


def query(sql: str, names: list[str] | None = None) -> pd.DataFrame:
    """保存済みデータセットに対して SQL を実行する。

    書き込み系は受け付けない。カタログ画面から任意の SQL を打てるようにするため、
    元データを壊しうる文を弾いておく。
    """
    guard_read_only(sql)
    con = connect(names)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


#: 読み取り専用にするために弾く先頭キーワード
_WRITE_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "copy",
    "attach", "detach", "install", "load", "export", "import", "call", "pragma",
)


def guard_read_only(sql: str) -> None:
    """SELECT / WITH 以外の文を拒否する。"""
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if not statements:
        raise StoreError("SQL が空です。")
    if len(statements) > 1:
        raise StoreError("一度に実行できるのは 1 文だけです。")
    head = statements[0].lstrip("( \t\n").split(None, 1)[0].lower()
    if head in _WRITE_KEYWORDS:
        raise StoreError(
            f"読み取り専用です。`{head.upper()}` は実行できません。"
            "SELECT または WITH で始まる文を書いてください。"
        )
    if head not in ("select", "with", "from", "describe", "summarize", "show", "table"):
        raise StoreError(
            f"`{head.upper()}` で始まる文には対応していません。SELECT / WITH を使ってください。"
        )
