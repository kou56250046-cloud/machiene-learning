"""解説ブロック。

文章は `content/` に置き、コードから分離する。1 トピックにつき最大 3 枚:

| ファイル | タブ | 内容 |
|---|---|---|
| `<topic>.md` | 解説 | 図の読み方。必須 |
| `<topic>.math.md` | 数式で見る | 何を計算しているか。任意 |
| `<topic>.usecase.md` | 使いどころ | いつ効くか・実例。任意 |

Markdown → HTML の変換は依存を増やさないよう最小限の自前パーサで行う
（見出し・段落・箇条書き・表・引用・強調・インラインコードのみ対応）。
数式だけは自前で描けないので、` ```math ` フェンスに TeX を書き、
Streamlit に同梱の KaTeX へ渡す（`st.latex`）。
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

_CONTENT_DIR = Path(__file__).resolve().parents[2] / "content"

# (ファイル接尾辞, タブ名)。解説は常に先頭。
_DOCS = (
    ("", "解説"),
    (".math", "数式で見る"),
    (".usecase", "使いどころ"),
)

# ("html", HTML断片) または ("latex", TeX)
Block = tuple[str, str]

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


def parse_blocks(text: str) -> list[Block]:
    """解説 Markdown を、HTML 断片と数式の並びに変換する。

    ` ```math ` フェンスだけを `("latex", TeX)` として切り出し、
    それ以外はまとめて `("html", ...)` にする。
    """
    blocks: list[Block] = []
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

    def cut_html() -> None:
        """ここまでの HTML を 1 ブロックとして確定する。"""
        if out:
            blocks.append(("html", "".join(out)))
            out.clear()

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
            body = chr(10).join(block)
            if language == "math":
                cut_html()
                blocks.append(("latex", body.strip()))
                continue
            css_class = f' class="language-{html.escape(language)}"' if language else ""
            out.append(f"<pre><code{css_class}>{html.escape(body)}</code></pre>")
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
    cut_html()
    return blocks


def markdown_to_html(text: str) -> str:
    """解説用の簡易 Markdown レンダラ。

    数式は KaTeX を使えない文脈（テストなど）向けに `<pre>` へ落とす。
    画面表示では `parse_blocks` を使い、`st.latex` に描かせる。
    """
    parts = []
    for kind, payload in parse_blocks(text):
        if kind == "latex":
            parts.append(f'<pre class="mllab-math"><code>{html.escape(payload)}</code></pre>')
        else:
            parts.append(payload)
    return "".join(parts)


@st.cache_data(show_spinner=False)
def _load(topic: str, suffix: str) -> list[Block]:
    path = _CONTENT_DIR / f"{topic}{suffix}.md"
    if not path.exists():
        return [("html", f"<p>解説ファイルが見つかりません: <code>{path.name}</code></p>")]
    return parse_blocks(path.read_text(encoding="utf-8"))


def _render(blocks: list[Block]) -> None:
    """HTML は 1 枚のカードに、数式は KaTeX に描かせる。"""
    pending: list[str] = []

    def flush() -> None:
        if pending:
            st.html(f'<div class="mllab-explain">{"".join(pending)}</div>')
            pending.clear()

    for kind, payload in blocks:
        if kind == "latex":
            flush()
            st.latex(payload)
        else:
            pending.append(payload)
    flush()


def explain(topic: str) -> None:
    """`content/<topic>*.md` の解説をページ下部に表示する。

    `.math` / `.usecase` が存在すればタブで束ねる。無ければ解説 1 枚だけ。
    """
    available = [
        (label, suffix)
        for suffix, label in _DOCS
        if (_CONTENT_DIR / f"{topic}{suffix}.md").exists()
    ]
    if not available:
        available = [("解説", "")]  # 見つからない旨を _load が出す

    with st.container(key=f"mllab-explain-{topic}"):
        if len(available) == 1:
            _render(_load(topic, available[0][1]))
            return
        for tab, (_, suffix) in zip(st.tabs([label for label, _ in available]), available):
            with tab:
                _render(_load(topic, suffix))
