"""データ蓄積層（Parquet + DuckDB）のテスト。

ネットワークには一切触らない。`store` の保存先は tmp_path に差し替える。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mllab.data import store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """本物の data/ を汚さないよう、保存先をテスト用ディレクトリへ向ける。"""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(store, "PROCESSED_DIR", tmp_path / "processed")
    store.ensure_dirs()
    return tmp_path


def make_frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "日付": pd.date_range("2024-01-01", periods=n, freq="D"),
            "都市": ["東京"] * n,
            "平均気温": rng.normal(15, 8, n).round(1),
            "降水量": rng.gamma(1.0, 3.0, n).round(1),
        }
    )


def save_sample(name: str = "weather_tokyo", n: int = 50) -> store.StoredDataset:
    return store.save(
        name,
        make_frame(n),
        label="気象データ",
        source="テスト",
        domain="timeseries",
        description="テスト用",
        params={"city": "tokyo"},
    )


# ---- 保存と読み出し ---------------------------------------------------

def test_save_then_load_roundtrip() -> None:
    original = make_frame(30)
    saved = save_sample(n=30)
    assert saved.rows == 30
    assert saved.columns == list(original.columns)

    loaded = store.load("weather_tokyo")
    pd.testing.assert_frame_equal(loaded, original)


def test_save_writes_metadata_sidecar() -> None:
    save_sample()
    meta = json.loads(store.meta_path("weather_tokyo").read_text(encoding="utf-8"))
    # 「いつ・どこから・どんな条件で」が残っていないと後から使えない
    assert meta["source"] == "テスト"
    assert meta["params"] == {"city": "tokyo"}
    assert meta["rows"] == 50
    assert meta["fetched_at"].endswith("+00:00")


def test_save_rejects_empty_frame() -> None:
    with pytest.raises(store.StoreError, match="0 行"):
        store.save("empty_set", pd.DataFrame(), label="空", source="x", domain="table")


@pytest.mark.parametrize(
    "name", ["Bad-Name", "1starts_with_digit", "has space", "", "x" * 70, "UPPER"]
)
def test_save_rejects_invalid_names(name: str) -> None:
    with pytest.raises(store.StoreError):
        store.save(name, make_frame(5), label="x", source="y", domain="table")


def test_load_missing_dataset_raises() -> None:
    with pytest.raises(store.StoreError, match="見つかりません"):
        store.load("does_not_exist")


def test_save_overwrites_existing() -> None:
    save_sample(n=10)
    saved = save_sample(n=25)
    assert saved.rows == 25
    assert len(store.load("weather_tokyo")) == 25
    assert len(store.list_datasets()) == 1


# ---- 一覧 -------------------------------------------------------------

def test_list_datasets_is_newest_first() -> None:
    save_sample("first_set")
    save_sample("second_set")
    names = [d.name for d in store.list_datasets()]
    assert set(names) == {"first_set", "second_set"}
    times = [d.fetched_at for d in store.list_datasets()]
    assert times == sorted(times, reverse=True)


def test_list_skips_orphan_parquet() -> None:
    """メタ情報のない Parquet は一覧に出さない（出所不明のデータを載せない）。"""
    save_sample()
    make_frame(5).to_parquet(store.parquet_path("orphan_set"), index=False)
    assert [d.name for d in store.list_datasets()] == ["weather_tokyo"]


def test_list_survives_corrupted_metadata() -> None:
    """メタ情報が壊れていてもカタログ画面が落ちないこと。"""
    save_sample()
    save_sample("broken_set")
    store.meta_path("broken_set").write_text("{ これは JSON ではない", encoding="utf-8")
    assert [d.name for d in store.list_datasets()] == ["weather_tokyo"]


def test_delete_removes_both_files() -> None:
    save_sample()
    store.delete("weather_tokyo")
    assert not store.parquet_path("weather_tokyo").exists()
    assert not store.meta_path("weather_tokyo").exists()
    assert store.list_datasets() == []


def test_delete_is_idempotent() -> None:
    store.delete("never_existed")  # 例外を出さない


def test_size_mb_reflects_file() -> None:
    saved = save_sample(n=1000)
    assert saved.size_mb > 0
    assert saved.size_mb < 1.0  # Parquet は圧縮が効くので小さいはず


# ---- SQL --------------------------------------------------------------

def test_query_over_parquet() -> None:
    save_sample(n=40)
    result = store.query("SELECT count(*) AS n FROM weather_tokyo")
    assert int(result.loc[0, "n"]) == 40


def test_query_handles_japanese_column_names() -> None:
    save_sample(n=40)
    result = store.query('SELECT avg("平均気温") AS mean FROM weather_tokyo')
    assert np.isfinite(result.loc[0, "mean"])


def test_query_can_join_two_datasets() -> None:
    save_sample("weather_tokyo", n=30)
    save_sample("weather_osaka", n=30)
    result = store.query(
        'SELECT count(*) AS n FROM weather_tokyo a '
        'JOIN weather_osaka b ON a."日付" = b."日付"'
    )
    assert int(result.loc[0, "n"]) == 30


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE weather_tokyo",
        "DELETE FROM weather_tokyo",
        "UPDATE weather_tokyo SET 都市 = 'x'",
        "INSERT INTO weather_tokyo VALUES (1)",
        "CREATE TABLE x AS SELECT 1",
        "COPY weather_tokyo TO 'out.csv'",
        "ATTACH 'other.db'",
        "PRAGMA database_list",
    ],
)
def test_query_rejects_write_statements(sql: str) -> None:
    save_sample()
    with pytest.raises(store.StoreError):
        store.query(sql)


def test_query_rejects_multiple_statements() -> None:
    save_sample()
    with pytest.raises(store.StoreError, match="1 文だけ"):
        store.query("SELECT 1; SELECT 2")


def test_query_rejects_empty_sql() -> None:
    with pytest.raises(store.StoreError, match="空"):
        store.query("   ")


def test_query_allows_with_and_leading_paren() -> None:
    save_sample(n=20)
    result = store.query(
        'WITH t AS (SELECT "平均気温" AS v FROM weather_tokyo) SELECT count(*) AS n FROM t'
    )
    assert int(result.loc[0, "n"]) == 20


def test_connect_registers_only_existing_datasets() -> None:
    save_sample()
    con = store.connect()
    try:
        views = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert views == {"weather_tokyo"}
