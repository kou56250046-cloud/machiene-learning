"""実験ログのテスト。

本物の data/experiments を汚さないよう、保存先を tmp_path へ差し替える。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mllab import experiments as EX


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(EX, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(EX, "RUNS_FILE", tmp_path / "experiments" / "runs.parquet")
    return tmp_path


def seed_runs(experiment: str = "テスト実験") -> None:
    """モデル 3 種 × 深さ 3 通りの試行を記録する。"""
    rng = np.random.default_rng(0)
    base = {"lightgbm": 0.90, "forest": 0.86, "linear": 0.78}
    for model, score in base.items():
        for depth in (3, 6, 12):
            EX.log(
                experiment,
                "テーブルデータ",
                params={"モデル": model, "深さ": depth, "標準化": True},
                metrics={"正解率": score + depth * 0.004 + rng.normal(0, 0.005)},
                note=f"{model} 深さ{depth}",
            )


# ---- 記録と読み出し ---------------------------------------------------


def test_log_creates_file_and_returns_run_id() -> None:
    run_id = EX.log("実験A", "テーブルデータ", {"モデル": "lightgbm"}, {"正解率": 0.9})
    assert len(run_id) == 12
    assert EX.RUNS_FILE.exists()
    assert len(EX.load_runs()) == 1


def test_log_requires_experiment_name() -> None:
    with pytest.raises(EX.ExperimentError, match="実験名"):
        EX.log("   ", "ラボ", {}, {"正解率": 0.5})


def test_columns_are_prefixed_by_kind() -> None:
    EX.log("実験A", "テーブルデータ", {"モデル": "svm"}, {"正解率": 0.8}, note="メモ")
    frame = EX.load_runs()
    assert "設定:モデル" in frame.columns
    assert "結果:正解率" in frame.columns
    for column in EX.CORE_COLUMNS:
        assert column in frame.columns


def test_runs_accumulate_under_the_same_name() -> None:
    for i in range(4):
        EX.log("同じ実験", "ラボ", {"n": i}, {"score": i / 10})
    assert len(EX.runs_of("同じ実験")) == 4


def test_experiments_are_kept_separate() -> None:
    EX.log("実験A", "ラボ", {"x": 1}, {"score": 0.5})
    EX.log("実験B", "ラボ", {"x": 2}, {"score": 0.6})
    assert EX.experiments() == ["実験A", "実験B"]
    assert len(EX.runs_of("実験A")) == 1


def test_new_settings_do_not_break_old_rows() -> None:
    """ラボごとに設定項目が違っても、同じファイルに追記できること。"""
    EX.log("混在", "テーブルデータ", {"モデル": "svm"}, {"score": 0.8})
    EX.log("混在", "時系列", {"周期": 365, "予測期間": 7}, {"score": 0.9})

    frame = EX.runs_of("混在")
    assert len(frame) == 2
    assert {"設定:モデル", "設定:周期", "設定:予測期間"} <= set(frame.columns)
    # 片方にしか無い設定は、もう片方では欠測になる
    assert frame["設定:周期"].isna().sum() == 1


def test_runs_are_newest_first() -> None:
    seed_runs()
    times = EX.load_runs()["記録日時"].tolist()
    assert times == sorted(times, reverse=True)


def test_load_runs_on_empty_store() -> None:
    frame = EX.load_runs()
    assert frame.empty
    assert list(frame.columns) == list(EX.CORE_COLUMNS)


# ---- 値の変換 ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "はい"),
        (False, "いいえ"),
        (3, "3"),
        (0.123456789, "0.123457"),
        ("文字列", "文字列"),
        ([1, 2, 3], "1、2、3"),
        (("a", "b"), "a、b"),
    ],
)
def test_settings_are_stored_as_readable_text(value, expected: str) -> None:
    EX.log("実験", "ラボ", {"設定値": value}, {"score": 1.0})
    assert EX.load_runs()["設定:設定値"].iloc[0] == expected


def test_non_numeric_metrics_become_nan() -> None:
    EX.log("実験", "ラボ", {}, {"score": "計測できず"})
    assert np.isnan(EX.load_runs()["結果:score"].iloc[0])


# ---- 分析 -------------------------------------------------------------


def test_best_run_picks_the_maximum() -> None:
    seed_runs()
    frame = EX.runs_of("テスト実験")
    best = EX.best_run(frame, "結果:正解率", higher_is_better=True)
    assert best is not None
    assert best["結果:正解率"] == pytest.approx(
        pd.to_numeric(frame["結果:正解率"]).max()
    )
    assert best["設定:モデル"] == "lightgbm"


def test_best_run_can_minimise() -> None:
    for value in (0.5, 0.2, 0.9):
        EX.log("誤差実験", "ラボ", {"x": value}, {"RMSE": value})
    frame = EX.runs_of("誤差実験")
    best = EX.best_run(frame, "結果:RMSE", higher_is_better=False)
    assert best["結果:RMSE"] == pytest.approx(0.2)


def test_best_run_returns_none_for_missing_metric() -> None:
    seed_runs()
    assert EX.best_run(EX.runs_of("テスト実験"), "結果:存在しない") is None


def test_which_settings_matter_ranks_by_spread() -> None:
    """モデルの違いのほうが深さより効いているデータを作ってあるので、上に来ること。"""
    seed_runs()
    effects = EX.which_settings_matter(EX.runs_of("テスト実験"), "結果:正解率")
    assert not effects.empty
    assert effects.iloc[0]["設定"] == "モデル"
    assert effects.iloc[0]["最良の値"] == "lightgbm"
    assert effects["スコアの幅"].is_monotonic_decreasing


def test_which_settings_matter_skips_constant_settings() -> None:
    """1 通りしか試していない設定は比較できないので出さない。"""
    seed_runs()
    effects = EX.which_settings_matter(EX.runs_of("テスト実験"), "結果:正解率")
    assert "標準化" not in effects["設定"].tolist()


def test_which_settings_matter_on_empty_frame() -> None:
    effects = EX.which_settings_matter(pd.DataFrame(), "結果:正解率")
    assert effects.empty
    assert list(effects.columns) == ["設定", "試した値の数", "スコアの幅", "最良の値"]


def test_summary_covers_every_metric() -> None:
    seed_runs()
    EX.log("テスト実験", "ラボ", {"モデル": "x"}, {"正解率": 0.5, "F1": 0.4})
    summary = EX.summary(EX.runs_of("テスト実験"))
    assert set(summary["指標"]) == {"正解率", "F1"}
    assert (summary["最良"] >= summary["最悪"]).all()


def test_param_and_metric_column_helpers() -> None:
    seed_runs()
    frame = EX.runs_of("テスト実験")
    assert EX.param_columns(frame) == ["設定:モデル", "設定:深さ", "設定:標準化"]
    assert EX.metric_columns(frame) == ["結果:正解率"]


# ---- 削除と書き出し ---------------------------------------------------


def test_delete_run_removes_one() -> None:
    seed_runs()
    frame = EX.load_runs()
    assert EX.delete_run(frame["run_id"].iloc[0]) is True
    assert len(EX.load_runs()) == len(frame) - 1


def test_delete_run_returns_false_for_unknown_id() -> None:
    seed_runs()
    assert EX.delete_run("存在しないID") is False


def test_delete_experiment_removes_only_that_one() -> None:
    seed_runs("実験A")
    EX.log("実験B", "ラボ", {"x": 1}, {"score": 0.5})
    removed = EX.delete_experiment("実験A")
    assert removed == 9
    assert EX.experiments() == ["実験B"]


def test_clear_all_removes_the_file() -> None:
    seed_runs()
    EX.clear_all()
    assert not EX.RUNS_FILE.exists()
    assert EX.load_runs().empty


def test_export_csv_round_trips() -> None:
    seed_runs()
    frame = EX.runs_of("テスト実験")
    data = EX.export_csv(frame)
    assert data.startswith(b"\xef\xbb\xbf")  # Excel 向けに BOM 付き
    import io

    restored = pd.read_csv(io.BytesIO(data))
    assert len(restored) == len(frame)


def test_storage_info_reports_size() -> None:
    assert EX.storage_info()["exists"] is False
    seed_runs()
    info = EX.storage_info()
    assert info["exists"] is True
    assert info["size_kb"] > 0


def test_environment_lists_versions() -> None:
    environment = EX.environment()
    assert {"python", "numpy", "sklearn", "os"} <= set(environment)
    assert environment["python"].startswith("3.")
