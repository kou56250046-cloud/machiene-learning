"""解説 Markdown の変換テスト。

依存を増やさないための自前パーサなので、対応している記法と
「変換し損ねて生の記号が画面に出ない」ことを押さえておく。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.components.explain import markdown_to_html

CONTENT_DIR = Path(__file__).resolve().parents[1] / "content"


def test_headings() -> None:
    html = markdown_to_html("# 大見出し\n\n## 中見出し\n\n### 小見出し")
    assert "<h2>大見出し</h2>" in html
    assert "<h2>中見出し</h2>" in html
    assert "<h3>小見出し</h3>" in html


def test_paragraph_joins_wrapped_lines() -> None:
    html = markdown_to_html("これは\n1 つの段落です。\n\n次の段落。")
    assert html.count("<p>") == 2
    assert "これは 1 つの段落です。" in html


def test_unordered_and_ordered_lists() -> None:
    html = markdown_to_html("- あ\n- い\n\n1. 一\n2. 二")
    assert "<ul><li>あ</li><li>い</li></ul>" in html
    assert "<ol><li>一</li><li>二</li></ol>" in html


def test_inline_bold_and_code() -> None:
    html = markdown_to_html("**太字** と `コード` です。")
    assert "<strong>太字</strong>" in html
    assert "<code>コード</code>" in html


def test_blockquote() -> None:
    html = markdown_to_html("> 注意書きです。\n> 続きます。")
    assert "<blockquote><p>注意書きです。 続きます。</p></blockquote>" in html


def test_table() -> None:
    html = markdown_to_html("| A | B |\n|---|---|\n| 1 | 2 |")
    assert "<th>A</th>" in html and "<th>B</th>" in html
    assert "<td>1</td>" in html and "<td>2</td>" in html
    # 区切り行はセルとして出さない
    assert "<td>---</td>" not in html


def test_fenced_code_block() -> None:
    html = markdown_to_html("説明。\n\n```sql\nSELECT * FROM t;\n```\n\n続き。")
    assert '<pre><code class="language-sql">SELECT * FROM t;</code></pre>' in html
    # コードの前後の段落も残る
    assert "<p>説明。</p>" in html and "<p>続き。</p>" in html


def test_fenced_code_block_without_language() -> None:
    html = markdown_to_html("```\nplain text\n```")
    assert "<pre><code>plain text</code></pre>" in html


def test_fenced_code_keeps_markdown_literal() -> None:
    """コードブロックの中身は変換しない（**や`をそのまま見せる）。"""
    html = markdown_to_html("```\n**not bold** and `not code`\n```")
    assert "<strong>" not in html
    assert "**not bold**" in html


def test_html_is_escaped() -> None:
    html = markdown_to_html("<script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ---- 実際の解説ファイル -----------------------------------------------

CONTENT_FILES = sorted(CONTENT_DIR.glob("*.md"))


def test_content_directory_is_not_empty() -> None:
    assert CONTENT_FILES, "content/*.md が 1 つも無い"


@pytest.mark.parametrize("path", CONTENT_FILES, ids=lambda p: p.stem)
def test_content_renders_without_leftover_markup(path: Path) -> None:
    """変換し損ねた記法が画面にそのまま出ていないこと。"""
    html = markdown_to_html(path.read_text(encoding="utf-8"))
    assert html.strip(), f"{path.name} の変換結果が空"

    leftovers = {
        "太字の **": "**",
        "コードフェンス ```": "```",
        "表の区切り行 |---": "|---",
    }
    for name, token in leftovers.items():
        assert token not in html, f"{path.name} に未変換の {name} が残っている"


@pytest.mark.parametrize("path", CONTENT_FILES, ids=lambda p: p.stem)
def test_content_starts_with_a_heading(path: Path) -> None:
    """各解説は見出しから始める（本文がいきなり出ると読みにくいため）。"""
    first = next(
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    assert first.startswith("#"), f"{path.name} が見出しで始まっていない"
