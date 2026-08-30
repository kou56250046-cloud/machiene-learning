"""解説 Markdown の変換テスト。

依存を増やさないための自前パーサなので、対応している記法と
「変換し損ねて生の記号が画面に出ない」ことを押さえておく。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.components.explain import (
    fill_blocks,
    format_value,
    markdown_to_html,
    parse_blocks,
)

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


# ---- 数式（```math フェンス） ------------------------------------------


def test_math_fence_becomes_a_latex_block() -> None:
    """数式は HTML に混ぜず、st.latex へ渡すブロックとして切り出す。"""
    blocks = parse_blocks("前。\n\n```math\nE = mc^2\n```\n\n後。")
    kinds = [kind for kind, _ in blocks]
    assert kinds == ["html", "latex", "html"]
    assert blocks[1][1] == "E = mc^2"
    assert "<p>前。</p>" in blocks[0][1]
    assert "<p>後。</p>" in blocks[2][1]


def test_math_fence_keeps_tex_untouched() -> None:
    """TeX の記号を HTML エスケープや強調変換で壊さない。"""
    tex = r"\lVert w \rVert^{2} + \sum_{i=1}^{n} x_i"
    blocks = parse_blocks("```math\n" + tex + "\n```")
    assert blocks == [("latex", tex)]


def test_math_falls_back_to_pre_in_plain_html() -> None:
    """KaTeX を使えない文脈では <pre> に落ちる（記号がそのまま出ない）。"""
    html = markdown_to_html("```math\na < b\n```")
    assert 'class="mllab-math"' in html
    assert "a &lt; b" in html


# ---- タブ 3 枚の構成 ---------------------------------------------------

BASE_FILES = [p for p in CONTENT_FILES if "." not in p.stem]
EXTRA_FILES = [p for p in CONTENT_FILES if "." in p.stem]


def test_every_extra_doc_has_a_base_doc() -> None:
    """`<topic>.math.md` / `<topic>.usecase.md` は解説本体とセットで置く。"""
    topics = {p.stem for p in BASE_FILES}
    for path in EXTRA_FILES:
        topic, suffix = path.stem.rsplit(".", 1)
        assert suffix in ("math", "usecase"), f"{path.name} は未対応の接尾辞"
        assert topic in topics, f"{path.name} に対応する {topic}.md が無い"


@pytest.mark.parametrize(
    "path", [p for p in EXTRA_FILES if p.stem.endswith(".math")], ids=lambda p: p.stem
)
def test_math_docs_contain_a_formula(path: Path) -> None:
    """「数式で見る」タブに数式が 1 つも無いことは無いはず。"""
    blocks = parse_blocks(path.read_text(encoding="utf-8"))
    assert any(kind == "latex" for kind, _ in blocks), f"{path.name} に数式が無い"


# ---- スライダ値の差し込み ----------------------------------------------


def test_slot_falls_back_when_no_value_is_given() -> None:
    """値を渡さなければ代替 TeX が出る（解説は単体でも読める）。"""
    blocks, filled = fill_blocks([("latex", "-@gamma:\\gamma@ x")], None)
    assert blocks == [("latex", "-\\gamma x")]
    assert filled == 0


def test_slot_without_fallback_uses_its_name() -> None:
    blocks, _ = fill_blocks([("latex", "@C@ \\sum x")], None)
    assert blocks == [("latex", "C \\sum x")]


def test_slot_is_filled_and_highlighted() -> None:
    """数式は KaTeX の色指定、本文は span で「いまの値」だと分かるようにする。"""
    blocks, filled = fill_blocks(
        [("latex", "-@gamma:\\gamma@ x"), ("html", "<p>k = @k@</p>")],
        {"gamma": 3.5, "k": 7},
    )
    assert blocks[0][1] == "-\\textcolor{#a8f04b}{3.5} x"
    assert blocks[1][1] == '<p>k = <span class="mllab-live">7</span></p>'
    assert filled == 2


def test_unknown_slot_keeps_its_fallback() -> None:
    """そのラボに無いパラメータは記号のまま残す（モデルを切り替えたとき）。"""
    blocks, filled = fill_blocks([("latex", "@C:C@ + @gamma:\\gamma@")], {"C": 0.5})
    assert blocks[0][1] == "\\textcolor{#a8f04b}{0.5} + \\gamma"
    assert filled == 1


@pytest.mark.parametrize(
    "value, expected",
    [
        (7, "7"),
        (12000, "12{,}000"),
        (0.08, "0.08"),
        (1.0 / 3, "0.3333"),
        (0.0001, "1 \\times 10^{-4}"),
    ],
)
def test_format_value(value: object, expected: str) -> None:
    assert format_value(value) == expected


@pytest.mark.parametrize("path", CONTENT_FILES, ids=lambda p: p.stem)
def test_content_has_no_broken_slot(path: Path) -> None:
    """閉じ忘れなどで `@名前:` が本文に出てしまわないこと。"""
    html = markdown_to_html(path.read_text(encoding="utf-8"))
    assert not re.search(r"@[A-Za-z_][A-Za-z0-9_]*:", html), f"{path.name} の差し込み口が壊れている"
