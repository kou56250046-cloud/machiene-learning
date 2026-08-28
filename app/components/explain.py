"""解説ブロック。

文章は `content/<topic>.md` に置き、コードから分離する。
Markdown → HTML の変換は依存を増やさないよう最小限の自前パーサで行う
（見出し・段落・箇条書き・表・引用・強調・インラインコードのみ対応）。
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

_CONTENT_DIR = Path(__file__).resolve().parents[2] / "content"

_INLINE = (
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
)


def _inline(text: str) -> str:
    out = html.escape(text)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def _table(rows: list[str]) -> str:
    """`| a | b |` 形式の行を HTML テーブルにする。区切り行は読み飛ばす。"""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    body = [r for r in cells if not all(set(c) <= set("-: ") for c in r)]
    if not body:
        return ""
    head, rest = body[0], body[1:]
    thead = "".join(f"<th>{_inline(c)}</th>" for c in head)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>" for row in rest
    )
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def markdown_to_html(text: str) -> str:
    """解説用の簡易 Markdown レンダラ。"""
    out: list[str] = []
    buffer: list[str] = []
    mode: str | None = None  # "ul" | "ol" | "table" | "quote" | "p"

    def flush() -> None:
        nonlocal mode, buffer
        if not buffer:
            mode = None
            return
        if mode == "ul":
            out.append("<ul>" + "".join(f"<li>{_inline(b)}</li>" for b in buffer) + "</ul>")
        elif mode == "ol":
            out.append("<ol>" + "".join(f"<li>{_inline(b)}</li>" for b in buffer) + "</ol>")
        elif mode == "table":
            out.append(_table(buffer))
        elif mode == "quote":
            out.append(f"<blockquote><p>{_inline(' '.join(buffer))}</p></blockquote>")
        elif mode == "p":
            out.append(f"<p>{_inline(' '.join(buffer))}</p>")
        buffer = []
        mode = None

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        index += 1
        line = raw.rstrip()
        stripped = line.strip()

        # ``` で囲まれたコードブロック。中身はそのまま出す。
        if stripped.startswith("```"):
            flush()
            language = stripped[3:].strip()
            block: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1  # 閉じの ``` を読み飛ばす
            css_class = f' class="language-{html.escape(language)}"' if language else ""
            out.append(
                f"<pre><code{css_class}>{html.escape(chr(10).join(block))}</code></pre>"
            )
            continue

        if not stripped:
            flush()
            continue
        if stripped.startswith("### "):
            flush()
            out.append(f"<h3>{_inline(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            flush()
            out.append(f"<h2>{_inline(stripped[3:])}</h2>")
            continue
        if stripped.startswith("# "):
            flush()
            out.append(f"<h2>{_inline(stripped[2:])}</h2>")
            continue
        if stripped.startswith("|"):
            if mode != "table":
                flush()
                mode = "table"
            buffer.append(stripped)
            continue
        if stripped.startswith("> "):
            if mode != "quote":
                flush()
                mode = "quote"
            buffer.append(stripped[2:])
            continue
        if stripped.startswith(("- ", "* ")):
            if mode != "ul":
                flush()
                mode = "ul"
            buffer.append(stripped[2:])
            continue
        if re.match(r"^\d+\.\s", stripped):
            if mode != "ol":
                flush()
                mode = "ol"
            buffer.append(re.sub(r"^\d+\.\s", "", stripped))
            continue

        if mode not in (None, "p"):
            flush()
        mode = "p"
        buffer.append(stripped)

    flush()
    return "".join(out)


@st.cache_data(show_spinner=False)
def _load(topic: str) -> str:
    path = _CONTENT_DIR / f"{topic}.md"
    if not path.exists():
        return f"<p>解説ファイルが見つかりません: <code>content/{topic}.md</code></p>"
    return markdown_to_html(path.read_text(encoding="utf-8"))


def explain(topic: str) -> None:
    """`content/<topic>.md` の解説をページ下部に表示する。"""
    st.html(f'<div class="mllab-explain">{_load(topic)}</div>')
