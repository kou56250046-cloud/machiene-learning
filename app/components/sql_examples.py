"""カタログの SQL 欄に出す例文。

保存済みのデータに合わせて、実在するテーブル名・列名で組み立てる。
テーブル名を覚えていなくても SQL を打ち始められるようにするのが狙い。
列名が日本語なので、DuckDB では二重引用符で囲む必要がある点も例文で示す。
"""

from __future__ import annotations

from mllab.data.store import StoredDataset

#: 気象・株価などで「数値ではない」列。要約統計の例文から外す。
_NON_NUMERIC = {"日付", "都市", "銘柄", "配信元", "見出し", "要約", "URL", "取得日時", "配信日時"}


def build(stored: list[StoredDataset]) -> dict[str, str]:
    """保存済みデータセットに合わせた例文を返す（表示名 → SQL）。"""
    if not stored:
        return {}

    first = stored[0]
    examples: dict[str, str] = {
        "先頭 20 行を見る": f"SELECT *\nFROM {first.name}\nLIMIT 20;",
        "行数を数える": f"SELECT count(*) AS 行数\nFROM {first.name};",
        "列ごとの要約統計": f"SUMMARIZE SELECT * FROM {first.name};",
    }

    numeric = [c for c in first.columns if c not in _NON_NUMERIC][:3]
    if numeric:
        aggregates = ",\n".join(
            f'  round(avg("{c}"), 2) AS "{c}_平均",'
            f' min("{c}") AS "{c}_最小",'
            f' max("{c}") AS "{c}_最大"'
            for c in numeric
        )
        examples["数値列の平均・最小・最大"] = (
            f"SELECT\n{aggregates}\nFROM {first.name};"
        )

    weather = _first_named(stored, "weather_")
    if weather:
        examples["気象: 月ごとに集計する"] = (
            "SELECT\n"
            "  strftime(\"日付\", '%Y-%m') AS 年月,\n"
            '  round(avg("平均気温"), 1) AS 平均気温,\n'
            '  round(sum("降水量"), 1) AS 降水量合計\n'
            f"FROM {weather.name}\n"
            "GROUP BY 年月\n"
            "ORDER BY 年月;"
        )
        examples["気象: 年ごとの猛暑日を数える"] = (
            "SELECT\n"
            '  year("日付") AS 年,\n'
            '  count(*) FILTER (WHERE "最高気温" >= 35) AS 猛暑日,\n'
            '  count(*) FILTER (WHERE "最高気温" >= 30) AS 真夏日\n'
            f"FROM {weather.name}\n"
            "GROUP BY 年\n"
            "ORDER BY 年;"
        )

    market = _first_named(stored, "market_")
    if market:
        examples["株価: 値動きが大きかった日"] = (
            'SELECT "日付", "終値", round("前日比(%)", 2) AS 前日比\n'
            f"FROM {market.name}\n"
            'WHERE "前日比(%)" IS NOT NULL\n'
            'ORDER BY abs("前日比(%)") DESC\n'
            "LIMIT 20;"
        )
        examples["株価: 20 日移動平均を出す"] = (
            'SELECT\n  "日付",\n  "終値",\n'
            '  round(avg("終値") OVER (ORDER BY "日付" ROWS 19 PRECEDING), 1)'
            " AS 移動平均20日\n"
            f"FROM {market.name}\n"
            'ORDER BY "日付" DESC\n'
            "LIMIT 100;"
        )

    news = _first_named(stored, "news_")
    if news:
        examples["ニュース: 配信元ごとの件数"] = (
            'SELECT "配信元", count(*) AS 件数\n'
            f"FROM {news.name}\n"
            'GROUP BY "配信元"\n'
            "ORDER BY 件数 DESC;"
        )
        examples["ニュース: 見出しにキーワードを含む記事"] = (
            'SELECT "配信日時", "配信元", "見出し"\n'
            f"FROM {news.name}\n"
            "WHERE \"見出し\" LIKE '%AI%'\n"
            'ORDER BY "配信日時" DESC;'
        )

    if weather and market:
        examples["気象 × 株価を日付で結合する"] = (
            "SELECT\n"
            '  w."日付", w."平均気温", m."終値", round(m."前日比(%)", 2) AS 前日比\n'
            f"FROM {weather.name} AS w\n"
            f'JOIN {market.name} AS m ON w."日付" = m."日付"\n'
            'ORDER BY w."日付" DESC\n'
            "LIMIT 50;"
        )

    return examples


def _first_named(stored: list[StoredDataset], prefix: str) -> StoredDataset | None:
    return next((d for d in stored if d.name.startswith(prefix)), None)
