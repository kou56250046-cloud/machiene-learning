"""ラボ 10 — テキスト。

カタログに溜めたニュース記事を使い、分かち書き → 重要語 → 話題の発見 → 分類 を回す。
数値でないデータを、どうやって機械学習が扱える形にするかがテーマ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import tabular_plots as TabP
from app.components import experiment_log as EL
from app.components import text_plots as P
from app.components.cards import Kpi, kpi_row, score_color
from app.components.explain import explain
from app.components.layout import note, page_header, panel, sidebar_section
from mllab.data import store
from mllab.models import text as TX
from mllab.viz import theme

KEY = "lab10"

accent = page_header(
    number=10,
    title="テキストラボ",
    lede=(
        "日本語は単語が空白で区切られていないので、まず分かち書きが要ります。"
        "そこから「その文書らしい語」を見つけ、語のつながりを描き、"
        "正解を与えずに話題を取り出し、最後に文書を分類するところまで通します。"
    ),
)

# ======================================================================
# データの選択
# ======================================================================

sidebar_section("データ")

candidates = [d for d in store.list_datasets() if d.domain == "text"]

if not candidates:
    st.warning(
        "テキストのデータがカタログにありません。"
        "**データカタログ**で「ニュース記事（RSS）」を取得してください。"
        "フィードを複数選ぶと、話題の違いが見えて面白くなります。",
        icon="📭",
    )
    explain("text")
    st.stop()

dataset_name = st.sidebar.selectbox(
    "データセット",
    [d.name for d in candidates],
    format_func=lambda n: f"{n}（{next(d.rows for d in candidates if d.name == n):,} 件）",
    key=f"{KEY}_dataset",
)
frame = store.load(dataset_name)

text_columns = [
    c for c in frame.columns if frame[c].dtype == object or pd.api.types.is_string_dtype(frame[c])
]
if not text_columns:
    st.error(f"`{dataset_name}` に文字列の列がありません。")
    st.stop()

default_columns = [c for c in ("見出し", "要約") if c in text_columns] or text_columns[:1]
used_columns = st.sidebar.multiselect(
    "本文にする列", text_columns, default=default_columns, key=f"{KEY}_cols",
    help="複数選ぶと連結して 1 つの文書として扱います。",
)
if not used_columns:
    st.warning("本文にする列を 1 つ以上選んでください。")
    st.stop()

label_candidates = [
    c for c in text_columns
    if c not in used_columns and 2 <= frame[c].nunique() <= 30
]
label_column = st.sidebar.selectbox(
    "ラベルの列（分類で当てる対象）",
    label_candidates or ["（なし）"],
    key=f"{KEY}_label_{dataset_name}",
    help="配信元など。分類タブと地図の色分けに使います。",
) if label_candidates else "（なし）"

documents = (
    frame[used_columns].fillna("").agg(" ".join, axis=1).str.strip()
)
keep = documents.str.len() > 0
documents, frame = documents[keep].reset_index(drop=True), frame[keep].reset_index(drop=True)

if len(documents) < 10:
    st.error(f"文書が {len(documents)} 件しかありません。カタログでもう少し取得してください。")
    st.stop()

# ======================================================================
# 解析の設定
# ======================================================================

sidebar_section("分かち書き")
selected_pos = st.sidebar.multiselect(
    "使う品詞", list(TX.POS_LABELS), default=list(TX.DEFAULT_POS),
    format_func=lambda p: TX.POS_LABELS[p], key=f"{KEY}_pos",
    help="助詞や助動詞はどの文書にも出るので、普通は外します。",
)
use_base_form = st.sidebar.toggle(
    "活用を基本形に揃える", value=True, key=f"{KEY}_base",
    help="「走っ」「走ら」を「走る」にまとめます。",
)
remove_stopwords = st.sidebar.toggle(
    "意味の薄い語を除く", value=True, key=f"{KEY}_stop",
    help="「する」「こと」「ため」など。オフにすると上位を埋め尽くします。",
)
min_length = st.sidebar.slider("何文字以上の語を使うか", 1, 4, 2, 1, key=f"{KEY}_minlen")

tokenize_config = TX.TokenizeConfig(
    pos=tuple(selected_pos) or TX.DEFAULT_POS,
    use_base_form=use_base_form,
    remove_stopwords=remove_stopwords,
    min_length=int(min_length),
)

sidebar_section("ベクトル化")
min_df = st.sidebar.slider(
    "最低何件に出る語を使うか", 1, 10, 2, 1, key=f"{KEY}_mindf",
    help="1 件にしか出ない語は、たいてい固有名詞か誤字です。",
)
max_df = st.sidebar.slider(
    "何割を超えたら捨てるか", 0.2, 1.0, 0.5, 0.05, key=f"{KEY}_maxdf",
    help="ほとんどの文書に出る語は、区別に使えません。",
)
vectorize_config = TX.VectorizeConfig(min_df=int(min_df), max_df=float(max_df))

sidebar_section("話題")
n_topics = st.sidebar.slider("トピック数", 2, 12, 6, 1, key=f"{KEY}_ntopics")
topic_method = st.sidebar.selectbox(
    "手法", list(TX.TOPIC_METHODS),
    format_func=lambda k: TX.TOPIC_METHODS[k], key=f"{KEY}_method",
)


# ======================================================================
# 計算
# ======================================================================


@st.cache_data(show_spinner="形態素解析中…")
def run_tokenize(_name, _config, documents):
    return TX.tokenize_all(documents, _config)


@st.cache_data(show_spinner="TF-IDF を計算中…")
def run_vectorize(_name, _tok_config, _vec_config, tokenized, documents):
    return TX.vectorize(tokenized, documents, _vec_config)


tokenized = run_tokenize(dataset_name, tokenize_config, documents.tolist())
lengths = pd.Series([len(t) for t in tokenized])

if lengths.sum() == 0:
    st.error("解析対象の語が 1 つも残りませんでした。品詞や文字数の設定を緩めてください。")
    st.stop()

try:
    vectorized = run_vectorize(
        dataset_name, tokenize_config, vectorize_config, tokenized, documents.tolist()
    )
except ValueError as exc:
    st.error(
        f"語彙が作れませんでした（{exc}）。"
        "「最低何件に出る語を使うか」を下げるか、「何割を超えたら捨てるか」を上げてください。"
    )
    st.stop()

vocabulary_size = vectorized.shape[1]

kpi_row(
    [
        Kpi("文書数", f"{len(documents):,}", sub="／".join(used_columns)),
        Kpi("延べ語数", f"{int(lengths.sum()):,}", sub=f"1 文書あたり平均 {lengths.mean():.1f} 語"),
        Kpi("語彙数", f"{vocabulary_size:,}",
            sub="TF-IDF で残った異なり語数", color=theme.PURPLE),
        Kpi("行列の密度", f"{vectorized.matrix.nnz / max(np.prod(vectorized.shape), 1):.2%}",
            sub="ほとんどが 0 の疎行列です", color=theme.ORANGE),
    ],
    accent,
)

if vocabulary_size < 20:
    note(
        f"語彙が {vocabulary_size} 語しかありません — 設定が厳しすぎるか、文書数が足りません。",
        tone="warn",
    )

# 分類タブで求めた成績を、ページ末尾の記録欄でも使うため外に置いておく
classification_accuracy: float | None = None
classification_baseline: float | None = None

tabs = st.tabs(["言葉を数える", "語のつながり", "話題を見つける", "分類してみる"])

# ======================================================================
# タブ 1: 言葉を数える
# ======================================================================
with tabs[0]:
    panel("分かち書きしてみる", "1 文がどう切られ、どの品詞が付くか")

    sample_index = st.slider(
        "何件目の文書を見るか", 0, len(documents) - 1, 0, 1, key=f"{KEY}_sample"
    )
    sample_text = documents.iloc[sample_index]
    st.markdown(f"> {sample_text}")

    col_tokens, col_pos = st.columns([1.6, 1])
    with col_tokens:
        st.markdown("**形態素解析の結果**")
        analyzed = TX.analyze(sample_text)
        st.dataframe(
            pd.DataFrame(analyzed, columns=["表層形", "品詞", "基本形"]),
            width="stretch", hide_index=True, height=300,
        )
    with col_pos:
        st.markdown("**品詞の内訳**")
        st.dataframe(
            TX.pos_distribution(sample_text), width="stretch", hide_index=True, height=300
        )

    kept = tokenized[sample_index]
    st.caption(
        f"このうち、いまの設定で残ったのは **{len(kept)} 語** です：`{' / '.join(kept)}`"
    )

    panel("頻度と TF-IDF はどう違うか", "同じ文書集合でも、順位はかなり入れ替わります")

    counts = TX.word_counts(tokenized, 40)
    tfidf_top = vectorized.top_words(40)
    st.plotly_chart(
        P.frequency_vs_tfidf(counts, tfidf_top), width="stretch", key=f"{KEY}_freq"
    )

    shift = P.rank_shift_table(counts, tfidf_top)
    if not shift.empty:
        col_shift, col_note = st.columns([1, 1])
        with col_shift:
            st.markdown("**順位が大きく動いた語**")
            st.dataframe(shift, width="stretch", hide_index=True, height=300)
        with col_note:
            st.markdown(
                """
**順位の差がプラス** の語は、頻度では下位なのに TF-IDF で上がってきた語です。
特定の文書にだけ集中して出るため、「その記事らしさ」を強く表します。

**順位の差がマイナス** の語は逆で、たくさん出るけれど広く薄く分布しているため、
文書の区別には役立ちません。

TF-IDF はこの「集中しているか、散らばっているか」を数式にしたものです。
                """
            )

    panel("1 文書あたりの語数", "短すぎる文書は、分析でうまく扱えません")
    st.plotly_chart(
        P.document_length_figure(lengths), width="stretch", key=f"{KEY}_len"
    )
    st.caption(
        f"最小 {int(lengths.min())} 語、中央値 {int(lengths.median())} 語、"
        f"最大 {int(lengths.max())} 語。RSS の要約は短いので、"
        "1 文書あたりの語数はどうしても少なくなります。"
    )

# ======================================================================
# タブ 2: 語のつながり
# ======================================================================
with tabs[1]:
    panel("共起ネットワーク", "同じ文書に一緒に出てくる語を線で結びます")

    col_a, col_b = st.columns(2)
    with col_a:
        network_words = st.slider(
            "描く語の数", 10, 80, 35, 5, key=f"{KEY}_networds",
            help="出現文書数の多い順に、この数だけ描きます。",
        )
    with col_b:
        min_cooccurrence = st.slider(
            "何回以上一緒に出たら線を引くか", 2, 15, 3, 1, key=f"{KEY}_mincooc"
        )

    @st.cache_data(show_spinner="共起を数えています…")
    def run_cooccurrence(_name, _config, tokenized, words, minimum):
        return TX.cooccurrence(tokenized, words, minimum, seed=0)

    graph = run_cooccurrence(
        dataset_name, tokenize_config, tokenized, int(network_words), int(min_cooccurrence)
    )

    if graph.edges.empty:
        note(
            "線が 1 本も引けませんでした。「何回以上一緒に出たら線を引くか」を下げてください。",
            tone="warn",
        )
    else:
        st.plotly_chart(
            P.cooccurrence_figure(graph), width="stretch", key=f"{KEY}_network"
        )
        st.caption(
            "丸の大きさが出現の多さ、色の明るさがつながりの多さ、線の濃さが結びつきの強さです。"
            "近くに固まっている語のかたまりが、そのまま話題のまとまりになります。"
        )

        col_edges, col_nodes = st.columns(2)
        with col_edges:
            st.markdown("**結びつきの強い組み合わせ**")
            st.dataframe(
                graph.edges.head(15), width="stretch", hide_index=True, height=320
            )
            st.caption(
                "Jaccard は「どちらかに出た文書のうち、両方に出た割合」です。"
                "単に共起回数で並べると、頻出語同士が上位を占めてしまいます。"
            )
        with col_nodes:
            st.markdown("**つながりの多い語**")
            st.dataframe(
                graph.nodes.sort_values("つながり数", ascending=False).head(15),
                width="stretch", hide_index=True, height=320,
            )
            st.caption(
                "多くの語と結びつく語は、話題をまたいで使われる汎用語であることが多いです。"
                "分析の邪魔になるようなら、ストップワードに足すことを検討します。"
            )

# ======================================================================
# タブ 3: 話題を見つける
# ======================================================================
with tabs[2]:
    panel(
        f"トピックモデル（{TX.TOPIC_METHODS[topic_method]}）",
        "正解を与えずに、文書集合から話題を取り出します",
    )

    @st.cache_data(show_spinner="トピックを抽出中…")
    def run_topics(_name, _tok, _vec, _vectorized, n_topics, method):
        return TX.fit_topics(_vectorized, n_topics, method, seed=0)

    try:
        topics = run_topics(
            dataset_name, tokenize_config, vectorize_config,
            vectorized, int(n_topics), topic_method,
        )
    except ValueError as exc:
        st.error(f"トピックを抽出できませんでした: {exc}")
        topics = None

    if topics is not None:
        st.plotly_chart(
            P.topic_words_figure(topics), width="stretch", key=f"{KEY}_topics"
        )
        st.caption(
            "各トピックは「よく一緒に出る語のまとまり」です。番号に意味はありません。"
            "代表語を見て、人間が名前を付けるところまでがトピックモデルの使い方です。"
        )

        col_share, col_table = st.columns([1.2, 1])
        with col_share:
            st.plotly_chart(
                P.topic_share_figure(topics), width="stretch", key=f"{KEY}_share"
            )
        with col_table:
            st.markdown("**トピックの一覧**")
            st.dataframe(
                topics.summary(6), width="stretch", hide_index=True, height=300
            )

        panel("文書の地図", "似た内容の文書が近くに来るよう 2 次元に配置します")

        color_by = st.radio(
            "色分け",
            ["トピック", "クラスタ"] + (["ラベル"] if label_column != "（なし）" else []),
            horizontal=True, key=f"{KEY}_colorby",
        )

        @st.cache_data(show_spinner="文書を配置中… 少し時間がかかります")
        def run_embedding(_name, _tok, _vec, _vectorized):
            return TX.embed_documents(_vectorized, seed=0)

        coordinates = run_embedding(
            dataset_name, tokenize_config, vectorize_config, vectorized
        )

        if color_by == "トピック":
            dominant = np.argmax(topics.document_topics, axis=1)
            colors = np.array([topics.label(t, 2) for t in dominant])
            legend = "主トピック"
        elif color_by == "クラスタ":
            n_clusters = st.slider(
                "クラスタ数", 2, 12, min(6, int(n_topics)), 1, key=f"{KEY}_nclusters"
            )

            @st.cache_data(show_spinner="クラスタリング中…")
            def run_clusters(_name, _tok, _vec, _vectorized, k):
                return TX.cluster_documents(_vectorized, k, seed=0)

            labels_array, representatives = run_clusters(
                dataset_name, tokenize_config, vectorize_config, vectorized, int(n_clusters)
            )
            colors = np.array(
                [f"C{c + 1}: " + "・".join(representatives[c][:2]) for c in labels_array]
            )
            legend = "クラスタ"
        else:
            colors = frame[label_column].astype(str).to_numpy()
            legend = label_column

        hover = [d[:70] + ("…" if len(d) > 70 else "") for d in documents]
        st.plotly_chart(
            P.document_map(coordinates, colors, hover, legend),
            width="stretch", key=f"{KEY}_map",
        )
        st.caption(
            "点にカーソルを合わせると本文の冒頭が出ます。"
            "色分けを「トピック」と「ラベル」で切り替えてみてください。"
            "**正解を与えずに見つけた話題が、実際の分類とどれだけ一致しているか**が分かります。"
        )

# ======================================================================
# タブ 4: 分類してみる
# ======================================================================
with tabs[3]:
    if label_column == "（なし）":
        st.info(
            "分類に使えるラベルの列がありません。"
            "ニュース記事なら「配信元」の列があるデータを選んでください。"
        )
    else:
        panel(
            f"{label_column} を当てる",
            "TF-IDF ベクトルにしてしまえば、テキストも普通の分類問題になります",
        )

        @st.cache_data(show_spinner="交差検証で分類中…")
        def run_classify(_name, _tok, _vec, _vectorized, targets):
            return TX.classify(_vectorized, targets, n_splits=5, seed=0)

        result = run_classify(
            dataset_name, tokenize_config, vectorize_config,
            vectorized, frame[label_column],
        )
        baseline = TX.majority_baseline(frame[label_column])
        classification_accuracy = result.accuracy
        classification_baseline = baseline

        kpi_row(
            [
                Kpi("正解率", f"{result.accuracy:.1%}", sub="交差検証での予測",
                    color=score_color(result.accuracy, good=baseline + 0.25, bad=baseline)),
                Kpi("ベースライン", f"{baseline:.1%}",
                    sub="常に最多クラスと答えた場合"),
                Kpi("超えた分", f"{result.accuracy - baseline:+.1%}",
                    sub="学習で得た分",
                    color=score_color(result.accuracy - baseline, good=0.2, bad=0.0)),
                Kpi("クラス数", f"{len(result.labels)}", sub=label_column,
                    color=theme.PURPLE),
            ],
            accent,
        )

        weakest = result.report["F1"].idxmin()
        if result.report.loc[weakest, "F1"] < 0.3:
            note(
                f"「{weakest}」の F1 が {result.report.loc[weakest, 'F1']:.2f} と低いままです — "
                f"件数が {int(result.report.loc[weakest, '件数'])} 件と少なく、"
                "モデルがこのクラスをほぼ無視しています。不均衡データの典型です。",
                tone="warn",
            )

        col_matrix, col_report = st.columns([1.1, 1])
        with col_matrix:
            st.markdown("**混同行列** — どこと取り違えているか")
            st.plotly_chart(
                TabP.confusion_figure(result.y_true, result.y_predicted, result.labels),
                width="stretch", key=f"{KEY}_cm",
            )
        with col_report:
            st.markdown("**クラスごとの成績**")
            st.dataframe(result.report, width="stretch", height=340)
            st.caption(
                "全体の正解率だけ見ていると、件数の少ないクラスが"
                "まったく当たっていないことに気づけません。"
            )

        panel("何を根拠にしているか", "クラスごとに、係数が大きかった語")
        st.plotly_chart(
            P.class_features_figure(result.top_features),
            width="stretch", key=f"{KEY}_features",
        )
        st.caption(
            "「その語が出たら、このクラスらしい」と学習された語です。"
            "納得できる語が並んでいれば、モデルは妥当な根拠で判断しています。"
            "無関係な語が上位なら、データの偏りを拾っている可能性を疑ってください。"
        )

# ======================================================================
# 実験の記録
# ======================================================================

EL.record_panel(
    lab="テキスト",
    params={
        "データ": dataset_name,
        "本文の列": used_columns,
        "品詞": [TX.POS_LABELS[p] for p in tokenize_config.pos],
        "基本形に揃える": use_base_form,
        "ストップワード除去": remove_stopwords,
        "min_df": int(min_df),
        "max_df": float(max_df),
        "トピック数": int(n_topics),
        "トピック手法": TX.TOPIC_METHODS[topic_method],
    },
    metrics={
        "語彙数": float(vocabulary_size),
        "平均語数": float(lengths.mean()),
        **(
            {
                "分類の正解率": classification_accuracy,
                "ベースライン超え": classification_accuracy - classification_baseline,
            }
            if classification_accuracy is not None
            else {}
        ),
    },
    key=KEY,
    default_experiment=f"テキスト/{dataset_name}",
)

# 解説の数式に、いまの文書数とトピック数を差し込む。
explain("text", values={"n": int(len(tokenized)), "K": int(n_topics)})
