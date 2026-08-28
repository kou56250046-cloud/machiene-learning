"""日本語テキスト解析のテスト。

外部データに依存しないよう、話題の分かっている小さな文書集合を自分で作って検証する。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mllab.models import text as TX

warnings.filterwarnings("ignore", category=UserWarning)


# ---- 検証用の文書 -----------------------------------------------------

SPORTS = [
    "選手がホームランを打ってチームが勝利しました",
    "サッカーの試合で選手が得点しチームが逆転しました",
    "野球のチームが優勝し選手たちが喜びました",
    "選手の活躍でチームは決勝に進出しました",
    "監督が選手を起用しチームは連勝しています",
    "大会で選手が記録を更新しチームが表彰されました",
]

ECONOMY = [
    "企業の業績が改善し株価が上昇しました",
    "政府が経済対策を発表し市場が反応しました",
    "円安が進み輸出企業の利益が拡大しています",
    "日銀が金利を据え置き市場は落ち着いています",
    "物価の上昇で家計の負担が増えています",
    "企業の投資が拡大し経済成長が続いています",
]


def documents() -> list[str]:
    return SPORTS + ECONOMY


def labels() -> pd.Series:
    return pd.Series(["スポーツ"] * len(SPORTS) + ["経済"] * len(ECONOMY))


@pytest.fixture(scope="module")
def tokenized() -> list[list[str]]:
    return TX.tokenize_all(documents())


@pytest.fixture(scope="module")
def vectorized(tokenized) -> TX.Vectorized:
    return TX.vectorize(tokenized, documents(), TX.VectorizeConfig(min_df=1, max_df=1.0))


# ---- 形態素解析 -------------------------------------------------------


def test_analyze_returns_surface_pos_base() -> None:
    result = TX.analyze("選手が走った")
    surfaces = [surface for surface, _, _ in result]
    assert "選手" in surfaces
    parts = {pos for _, pos, _ in result}
    assert "名詞" in parts and "助詞" in parts
    # 活用した動詞は基本形が取れる
    bases = {base for _, _, base in result}
    assert "走る" in bases


def test_tokenize_keeps_only_selected_pos() -> None:
    words = TX.tokenize("選手が速く走った", TX.TokenizeConfig(pos=("名詞",)))
    assert "選手" in words
    assert "走る" not in words  # 動詞は除外されている


def test_tokenize_normalises_conjugation() -> None:
    inflected = TX.tokenize("選手が走った", TX.TokenizeConfig(use_base_form=True))
    plain = TX.tokenize("選手が走る", TX.TokenizeConfig(use_base_form=True))
    assert "走る" in inflected and "走る" in plain


def test_tokenize_without_base_form_keeps_surface() -> None:
    words = TX.tokenize("選手が走った", TX.TokenizeConfig(use_base_form=False))
    assert "走る" not in words


def test_stopwords_are_removed() -> None:
    text = "これをすることが大切です"
    with_stopwords = TX.tokenize(text, TX.TokenizeConfig(remove_stopwords=False))
    without = TX.tokenize(text, TX.TokenizeConfig(remove_stopwords=True))
    assert len(without) < len(with_stopwords)
    assert not (set(without) & TX.STOPWORDS)


def test_min_length_filters_short_words() -> None:
    long_only = TX.tokenize("選手が本を読む", TX.TokenizeConfig(min_length=2))
    assert all(len(w) >= 2 for w in long_only)


def test_tokenize_drops_symbols_and_single_kana() -> None:
    words = TX.tokenize("選手、が。走った！", TX.TokenizeConfig(min_length=1))
    assert "、" not in words and "。" not in words and "！" not in words


def test_tokenize_handles_empty_and_symbols_only() -> None:
    assert TX.tokenize("") == []
    assert TX.tokenize("！？＃＄") == []


def test_pos_distribution_counts_parts() -> None:
    frame = TX.pos_distribution("選手が速く走った")
    assert set(frame.columns) == {"品詞", "件数"}
    assert frame["件数"].sum() > 0
    assert frame["件数"].is_monotonic_decreasing


# ---- 頻度と TF-IDF ----------------------------------------------------


def test_word_counts_reports_both_totals(tokenized) -> None:
    counts = TX.word_counts(tokenized, top=10)
    assert list(counts.columns) == ["語", "出現回数", "出現文書数"]
    assert counts["出現回数"].is_monotonic_decreasing
    # 出現回数は文書数以上（同じ文書に 2 回出ることがある）
    assert (counts["出現回数"] >= counts["出現文書数"]).all()


def test_vectorize_shape_matches_documents(tokenized, vectorized) -> None:
    assert vectorized.shape[0] == len(documents())
    assert vectorized.shape[1] == len(vectorized.vocabulary)
    assert vectorized.shape[1] > 5


def test_tfidf_ranks_distinctive_words_above_common_ones(tokenized) -> None:
    """全文書に出る語より、片方の話題にだけ出る語が上に来ること。"""
    vector = TX.vectorize(
        tokenized, documents(), TX.VectorizeConfig(min_df=1, max_df=1.0)
    )
    scores = dict(zip(vector.top_words(200)["語"], vector.top_words(200)["TF-IDF合計"]))
    # 「チーム」はスポーツ 6 件すべてに出る一方、「企業」は経済側だけ
    assert "企業" in scores
    assert scores["企業"] > 0


def test_document_top_words_are_from_that_document(vectorized) -> None:
    top = vectorized.document_top_words(0, top=5)
    assert not top.empty
    assert (top["TF-IDF"] > 0).all()
    words_in_document = set(TX.tokenize(documents()[0]))
    assert set(top["語"]) <= words_in_document


def test_min_df_shrinks_vocabulary(tokenized) -> None:
    loose = TX.vectorize(tokenized, documents(), TX.VectorizeConfig(min_df=1, max_df=1.0))
    strict = TX.vectorize(tokenized, documents(), TX.VectorizeConfig(min_df=3, max_df=1.0))
    assert strict.shape[1] < loose.shape[1]


def test_max_df_removes_ubiquitous_words(tokenized) -> None:
    everywhere = ["共通語"] * len(tokenized)
    with_common = [words + [w] for words, w in zip(tokenized, everywhere)]
    kept = TX.vectorize(with_common, documents(), TX.VectorizeConfig(min_df=1, max_df=1.0))
    dropped = TX.vectorize(with_common, documents(), TX.VectorizeConfig(min_df=1, max_df=0.5))
    assert "共通語" in kept.vocabulary
    assert "共通語" not in dropped.vocabulary


# ---- 共起 -------------------------------------------------------------


def test_cooccurrence_builds_graph(tokenized) -> None:
    graph = TX.cooccurrence(tokenized, top_words=20, min_cooccurrence=2)
    assert set(graph.nodes.columns) == {"語", "出現文書数", "つながり数"}
    assert set(graph.edges.columns) == {"語1", "語2", "共起回数", "Jaccard"}
    assert not graph.edges.empty
    assert graph.edges["Jaccard"].is_monotonic_decreasing
    assert (graph.edges["Jaccard"] <= 1.0).all()
    # すべてのノードに座標が付いている
    assert set(graph.nodes["語"]) <= set(graph.positions)


def test_cooccurrence_links_words_from_the_same_topic(tokenized) -> None:
    """同じ話題の語同士が、違う話題の語より強く結びつくこと。"""
    graph = TX.cooccurrence(tokenized, top_words=30, min_cooccurrence=2)
    pairs = {
        frozenset((row.語1, row.語2)): row.Jaccard for row in graph.edges.itertuples()
    }
    same_topic = pairs.get(frozenset(("選手", "チーム")), 0.0)
    cross_topic = pairs.get(frozenset(("選手", "企業")), 0.0)
    assert same_topic > cross_topic


def test_cooccurrence_threshold_reduces_edges(tokenized) -> None:
    loose = TX.cooccurrence(tokenized, top_words=20, min_cooccurrence=1)
    strict = TX.cooccurrence(tokenized, top_words=20, min_cooccurrence=4)
    assert len(strict.edges) <= len(loose.edges)


def test_cooccurrence_is_reproducible(tokenized) -> None:
    a = TX.cooccurrence(tokenized, 20, 2, seed=7)
    b = TX.cooccurrence(tokenized, 20, 2, seed=7)
    assert a.positions == b.positions


# ---- トピックモデル ---------------------------------------------------


@pytest.mark.parametrize("method", list(TX.TOPIC_METHODS))
def test_topic_model_shapes(method: str, vectorized) -> None:
    model = TX.fit_topics(vectorized, n_topics=2, method=method, seed=0)
    assert model.components.shape == (2, vectorized.shape[1])
    assert model.document_topics.shape == (vectorized.shape[0], 2)
    assert len(model.top_words(0, 5)) == 5
    assert model.label(0).startswith("トピック1:")


def test_topic_model_rejects_unknown_method(vectorized) -> None:
    with pytest.raises(ValueError, match="未知のトピックモデル"):
        TX.fit_topics(vectorized, 2, "unknown")


def test_topics_separate_the_two_subjects(vectorized) -> None:
    """スポーツ 6 件と経済 6 件が、別々のトピックに割り当てられること。"""
    model = TX.fit_topics(vectorized, n_topics=2, method="nmf", seed=0)
    dominant = np.argmax(model.document_topics, axis=1)
    sports_topic = set(dominant[: len(SPORTS)])
    economy_topic = set(dominant[len(SPORTS) :])
    assert len(sports_topic) == 1 and len(economy_topic) == 1
    assert sports_topic != economy_topic


def test_topic_summary_columns(vectorized) -> None:
    summary = TX.fit_topics(vectorized, 2, "nmf").summary()
    assert list(summary.columns) == ["トピック", "代表語", "主トピックの割合"]
    assert len(summary) == 2
    assert summary["主トピックの割合"].sum() == pytest.approx(1.0, abs=1e-6)


# ---- 配置とクラスタリング ---------------------------------------------


def test_embed_documents_returns_2d(vectorized) -> None:
    coordinates = TX.embed_documents(vectorized, seed=0)
    assert coordinates.shape == (vectorized.shape[0], 2)
    assert np.isfinite(coordinates).all()


def test_cluster_documents_labels_and_representatives(vectorized) -> None:
    labels_array, representatives = TX.cluster_documents(vectorized, n_clusters=2, seed=0)
    assert len(labels_array) == vectorized.shape[0]
    assert set(labels_array.tolist()) == {0, 1}
    assert set(representatives) == {0, 1}
    assert all(len(words) > 0 for words in representatives.values())


def test_clusters_match_the_two_subjects(vectorized) -> None:
    labels_array, _ = TX.cluster_documents(vectorized, n_clusters=2, seed=0)
    sports = set(labels_array[: len(SPORTS)])
    economy = set(labels_array[len(SPORTS) :])
    assert len(sports) == 1 and len(economy) == 1
    assert sports != economy


# ---- 分類 -------------------------------------------------------------


def test_classification_beats_the_baseline(vectorized) -> None:
    result = TX.classify(vectorized, labels(), n_splits=3, seed=0)
    assert result.accuracy > TX.majority_baseline(labels())
    assert set(result.labels) == {"スポーツ", "経済"}
    assert len(result.y_true) == len(result.y_predicted) == vectorized.shape[0]


def test_classification_report_columns(vectorized) -> None:
    result = TX.classify(vectorized, labels(), n_splits=3, seed=0)
    assert {"適合率", "再現率", "F1", "件数"} <= set(result.report.columns)
    assert list(result.report.index) == result.labels


def test_confusion_matrix_shape(vectorized) -> None:
    result = TX.classify(vectorized, labels(), n_splits=3, seed=0)
    matrix = result.confusion()
    assert matrix.shape == (2, 2)
    assert matrix.sum() == vectorized.shape[0]


def test_top_features_are_real_vocabulary(vectorized) -> None:
    result = TX.classify(vectorized, labels(), n_splits=3, seed=0)
    assert set(result.top_features) == {"スポーツ", "経済"}
    for words in result.top_features.values():
        assert all(word in vectorized.vocabulary for word, _ in words)


def test_classification_caps_splits_to_smallest_class(vectorized) -> None:
    """少数クラスの件数より多い分割は要求できない。落ちずに切り詰めること。"""
    skewed = pd.Series(["多数"] * 10 + ["少数"] * 2)
    result = TX.classify(vectorized, skewed, n_splits=5, seed=0)
    assert len(result.y_predicted) == vectorized.shape[0]


def test_majority_baseline() -> None:
    assert TX.majority_baseline(pd.Series(["a"] * 7 + ["b"] * 3)) == pytest.approx(0.7)
    assert TX.majority_baseline(labels()) == pytest.approx(0.5)
