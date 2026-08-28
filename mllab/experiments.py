"""実験の記録。

「何を試して、どうだったか」を残す仕組み。これが無いと、
半年後に「あのとき一番良かった設定は何だったか」が分からなくなる。

MLflow を使う手もあるが、この構成では採らなかった。理由は 2 つ。
- 本体は pandas を 2.x へ下げてしまう（このプロジェクトは 3.x で動いている）
- skinny 版でも fastapi / databricks-sdk / opentelemetry など 28 個の依存が付く

このプロジェクトは「サーバーを立てない・ローカルのファイルだけ」を通してきたので、
既にある Parquet + DuckDB の仕組みに乗せる。記録の形が見えるぶん、
実験管理が何をしているかも分かりやすい。
"""

from __future__ import annotations

import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mllab.data.store import DATA_DIR

#: 実験ログの置き場所。データセットとは分けておく
EXPERIMENTS_DIR = DATA_DIR / "experiments"
RUNS_FILE = EXPERIMENTS_DIR / "runs.parquet"

#: 記録する列のうち、必ず入っているもの
CORE_COLUMNS = ("run_id", "実験", "記録日時", "ラボ", "メモ")


class ExperimentError(RuntimeError):
    """記録・読み出しの失敗。"""


@dataclass(frozen=True)
class Run:
    """1 回の試行の記録。

    params（何を設定したか）と metrics（どうだったか）を分けて持つ。
    後から「この設定のときスコアはどうだったか」を表で並べられる。
    """

    experiment: str
    lab: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    note: str = ""
    run_id: str = ""
    recorded_at: str = ""

    def to_row(self) -> dict[str, Any]:
        """1 行の辞書にする。列名の衝突を避けるため接頭辞を付ける。"""
        row: dict[str, Any] = {
            "run_id": self.run_id or uuid.uuid4().hex[:12],
            "実験": self.experiment,
            "記録日時": self.recorded_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ラボ": self.lab,
            "メモ": self.note,
        }
        for key, value in self.params.items():
            row[f"設定:{key}"] = _stringify(value)
        for key, value in self.metrics.items():
            row[f"結果:{key}"] = _as_float(value)
        return row


def _stringify(value: Any) -> str:
    """設定値を、表に並べても読める文字列にする。

    設定はモデルによって型がばらばら（数値・文字列・タプル）なので、
    列の型を揃えるために文字列で持つ。比較して眺めるのが目的なので困らない。
    """
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "、".join(_stringify(v) for v in value)
    if isinstance(value, bool):
        return "はい" if value else "いいえ"
    return str(value)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def environment() -> dict[str, str]:
    """実行環境。同じコードでも環境が変われば結果は変わりうる。"""
    import numpy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "os": f"{platform.system()} {platform.release()}",
    }


def log(
    experiment: str,
    lab: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    note: str = "",
) -> str:
    """1 回の試行を記録し、run_id を返す。

    同じ実験名で何度呼んでも追記されていく。列が増えたら既存の行は空欄になる
    （設定項目はラボによって違うので、それを許す作りにしてある）。
    """
    if not experiment.strip():
        raise ExperimentError("実験名を入力してください。")

    run = Run(
        experiment=experiment.strip(),
        lab=lab,
        params=params or {},
        metrics=metrics or {},
        note=note.strip(),
    )
    row = run.to_row()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if RUNS_FILE.exists():
        existing = pd.read_parquet(RUNS_FILE)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_parquet(RUNS_FILE, index=False)

    return str(row["run_id"])


def load_runs() -> pd.DataFrame:
    """記録された全ての試行を、新しい順に返す。"""
    if not RUNS_FILE.exists():
        return pd.DataFrame(columns=list(CORE_COLUMNS))
    frame = pd.read_parquet(RUNS_FILE)
    if frame.empty:
        return frame
    return frame.sort_values("記録日時", ascending=False).reset_index(drop=True)


def experiments() -> list[str]:
    """記録がある実験名の一覧。"""
    frame = load_runs()
    if frame.empty or "実験" not in frame:
        return []
    return sorted(frame["実験"].dropna().unique().tolist())


def runs_of(experiment: str) -> pd.DataFrame:
    """1 つの実験に属する試行だけを返す。全て空の列は落とす。"""
    frame = load_runs()
    if frame.empty:
        return frame
    selected = frame[frame["実験"] == experiment].copy()
    return selected.dropna(axis=1, how="all").reset_index(drop=True)


def param_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if str(c).startswith("設定:")]


def metric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if str(c).startswith("結果:")]


def delete_experiment(experiment: str) -> int:
    """1 つの実験の記録をまとめて消す。消した件数を返す。"""
    frame = load_runs()
    if frame.empty:
        return 0
    keep = frame[frame["実験"] != experiment]
    removed = len(frame) - len(keep)
    if removed:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        keep.to_parquet(RUNS_FILE, index=False)
    return int(removed)


def delete_run(run_id: str) -> bool:
    """試行を 1 件だけ消す。"""
    frame = load_runs()
    if frame.empty:
        return False
    keep = frame[frame["run_id"] != run_id]
    if len(keep) == len(frame):
        return False
    keep.to_parquet(RUNS_FILE, index=False)
    return True


def clear_all() -> None:
    """全ての記録を消す。"""
    RUNS_FILE.unlink(missing_ok=True)


# ======================================================================
# 分析
# ======================================================================


def best_run(frame: pd.DataFrame, metric: str, higher_is_better: bool = True):
    """指定した指標が最も良かった試行を返す。"""
    if frame.empty or metric not in frame:
        return None
    values = pd.to_numeric(frame[metric], errors="coerce")
    if values.isna().all():
        return None
    index = values.idxmax() if higher_is_better else values.idxmin()
    return frame.loc[index]


def which_settings_matter(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """設定ごとに、その値を変えたときスコアがどれだけ動いたかを見る。

    群ごとの平均の幅（最大 − 最小）で並べる。厳密な要因分析ではないが、
    「どのつまみが効いているか」の当たりを付けるには十分。
    """
    if frame.empty or metric not in frame:
        return pd.DataFrame(columns=["設定", "試した値の数", "スコアの幅", "最良の値"])

    values = pd.to_numeric(frame[metric], errors="coerce")
    rows = []
    for column in param_columns(frame):
        distinct = frame[column].nunique(dropna=True)
        if distinct < 2:
            continue  # 1 通りしか試していない設定は比較できない
        grouped = values.groupby(frame[column]).mean().dropna()
        if len(grouped) < 2:
            continue
        rows.append(
            {
                "設定": str(column).removeprefix("設定:"),
                "試した値の数": int(distinct),
                "スコアの幅": float(grouped.max() - grouped.min()),
                "最良の値": str(grouped.idxmax()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["設定", "試した値の数", "スコアの幅", "最良の値"])
    return (
        pd.DataFrame(rows)
        .sort_values("スコアの幅", ascending=False)
        .reset_index(drop=True)
    )


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """指標ごとの、試行全体を通した統計。"""
    rows = []
    for column in metric_columns(frame):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "指標": str(column).removeprefix("結果:"),
                "試行数": int(len(values)),
                "最良": float(values.max()),
                "最悪": float(values.min()),
                "平均": float(values.mean()),
                "ばらつき": float(values.std()) if len(values) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def export_csv(frame: pd.DataFrame) -> bytes:
    """記録を CSV にする（他のツールへ持ち出す用）。"""
    return frame.to_csv(index=False).encode("utf-8-sig")


def storage_info() -> dict[str, Any]:
    """記録の置き場所と大きさ。"""
    exists = RUNS_FILE.exists()
    return {
        "path": str(RUNS_FILE),
        "exists": exists,
        "size_kb": RUNS_FILE.stat().st_size / 1024 if exists else 0.0,
    }
