"""日本語テキストの解析。

英語と違い、日本語は単語が空白で区切られていない。まず「分かち書き」が要る。
ここでは janome（純 Python の形態素解析器）を使う。MeCab のような外部インストールが
不要なので、Windows でも `uv sync` だけで動く。

計算はすべてここに置き、`app/views/lab10_text.py` は UI の組み立てだけにする。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict

# ======================================================================
# 形態素解析
# ======================================================================

#: 解析に使う品詞。名詞・動詞・形容詞だけ残すのが定石。
#: 助詞（「の」「を」）や助動詞は、どの文書にも出るので区別に役立たない。
POS_LABELS: dict[str, str] = {
    "名詞": "名詞",
    "動詞": "動詞",
    "形容詞": "形容詞",
    "副詞": "副詞",
    "連体詞": "連体詞",
    "感動詞": "感動詞",
    "接続詞": "接続詞",
    "助詞": "助詞",
    "助動詞": "助動詞",
    "記号": "記号",
    "フィラー": "フィラー",
}

DEFAULT_POS: tuple[str, ...] = ("名詞", "動詞", "形容詞")

#: 内容を持たない語。頻度は高いが話題の区別に使えない。
STOPWORDS: frozenset[str] = frozenset(
    """
    する ある なる いる れる られる こと もの これ それ あれ どれ ここ そこ どこ
    ため よう そう とき ところ など まま うち ほう つもり わけ はず 一 人 的 性 化
    さん 氏 これら それら 今 今回 今年 昨年 来年 今月 先月 来月 今日 昨日 明日
    年 月 日 時 分 秒 円 人 回 目 中 上 下 前 後 内 外 他 等 なに 何 私 僕 彼 彼女
    いう 言う 思う 見る 行く 来る 出る 入る 持つ 使う 対する 関する 続く 続ける
    できる わかる 分かる 知る 考える 含む 示す 受ける 与える 取る 取れる
    """.split()
)

#: 1 文字のひらがな・カタカナだけの語も、単独では意味を持たない
_KANA_ONLY = re.compile(r"^[ぁ-んァ-ヶー]{1}$")
_SYMBOLS = re.compile(r"^[\W_]+$")


@lru_cache(maxsize=1)
def _tokenizer():
    """janome の Tokenizer は生成が重いので使い回す。"""
    from janome.tokenizer import Tokenizer

    return Tokenizer()


@dataclass(frozen=True)
class TokenizeConfig:
    """分かち書きの設定。ハッシュ可能なのでキャッシュのキーに使える。"""

    pos: tuple[str, ...] = DEFAULT_POS
    #: 活用を基本形に揃えるか（「走っ た」→「走る」）
    use_base_form: bool = True
    #: ストップワードを除くか
    remove_stopwords: bool = True
    #: この文字数未満の語を捨てる
    min_length: int = 2


def analyze(text: str) -> list[tuple[str, str, str]]:
    """1 つの文を (表層形, 品詞, 基本形) の並びにする。"""
    return [
        (token.surface, token.part_of_speech.split(",")[0], token.base_form)
        for token in _tokenizer().tokenize(str(text))
    ]


def tokenize(text: str, config: TokenizeConfig = TokenizeConfig()) -> list[str]:
    """文を、解析に使う語の並びにする。

    品詞で絞り、活用を基本形に揃え、意味を持たない語を落とす。
    ここで何を残すかが、この後の全ての分析の土台になる。
    """
    words: list[str] = []
    for surface, pos, base in analyze(text):
        if pos not in config.pos:
            continue
        word = base if (config.use_base_form and base != "*") else surface
        if len(word) < config.min_length:
            continue
        if _KANA_ONLY.match(word) or _SYMBOLS.match(word):
            continue
        if config.remove_stopwords and word in STOPWORDS:
            continue
        words.append(word)
    return words


def tokenize_all(
    documents: Iterable[str], config: TokenizeConfig = TokenizeConfig()
) -> list[list[str]]:
    return [tokenize(document, config) for document in documents]


def pos_distribution(text: str) -> pd.DataFrame:
    """1 つの文の品詞の内訳。分かち書きが何をしているかを見せるため。"""
    counts = Counter(pos for _, pos, _ in analyze(text))
    return (
        pd.DataFrame(
            {"品詞": list(counts), "件数": [counts[p] for p in counts]}
        )
        .sort_values("件数", ascending=False)
        .reset_index(drop=True)
    )


# ======================================================================
# 語の重要度
# ======================================================================


def word_counts(tokenized: list[list[str]], top: int = 40) -> pd.DataFrame:
    """単純な出現回数。文書数（何件に出たか）も併記する。"""
    total = Counter()
    document_frequency = Counter()
    for words in tokenized:
        total.update(words)
        document_frequency.update(set(words))

    rows = [
        {"語": word, "出現回数": count, "出現文書数": document_frequency[word]}
        for word, count in total.most_common(top)
    ]
    return pd.DataFrame(rows)


@dataclass
class Vectorized:
    """TF-IDF でベクトルにした結果。"""

    matrix: Any  # scipy の疎行列
    vocabulary: list[str]
    #: 各文書の元テキスト（結果の確認用）
    documents: list[str]

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def top_words(self, top: int = 40) -> pd.DataFrame:
        """TF-IDF の合計が大きい語。単純な頻度とは順位が変わる。"""
        scores = np.asarray(self.matrix.sum(axis=0)).ravel()
        order = np.argsort(scores)[::-1][:top]
        return pd.DataFrame(
            {
                "語": [self.vocabulary[i] for i in order],
                "TF-IDF合計": scores[order].round(4),
            }
        )

    def document_top_words(self, index: int, top: int = 10) -> pd.DataFrame:
        """1 文書の中で TF-IDF が高い語。その文書らしさを表す語。"""
        row = np.asarray(self.matrix[index].todense()).ravel()
        order = np.argsort(row)[::-1][:top]
        return pd.DataFrame(
            {
                "語": [self.vocabulary[i] for i in order if row[i] > 0],
                "TF-IDF": [round(float(row[i]), 4) for i in order if row[i] > 0],
            }
        )


@dataclass(frozen=True)
class VectorizeConfig:
    """TF-IDF の設定。"""

    #: この件数未満にしか出ない語を捨てる（誤字・固有名詞のノイズ対策）
    min_df: int = 2
    #: この割合を超える文書に出る語を捨てる（どこにでもある語）
    max_df: float = 0.5
    #: 語彙の上限
    max_features: int = 3000
    #: True なら出現の有無だけを見る（回数を無視）
    binary: bool = False


def vectorize(
    tokenized: list[list[str]], documents: list[str], config: VectorizeConfig
) -> Vectorized:
    """分かち書き済みの文書を TF-IDF 行列にする。

    TF-IDF は「その文書でよく出て、他の文書にはあまり出ない語」を高く評価する。
    単純な頻度だと「日本」「発表」のような、どこにでも出る語が上位を占めてしまう。
    """
    vectorizer = TfidfVectorizer(
        analyzer=lambda words: words,  # 既に分かち書き済みなので素通し
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        binary=config.binary,
    )
    matrix = vectorizer.fit_transform(tokenized)
    return Vectorized(
        matrix=matrix,
        vocabulary=list(vectorizer.get_feature_names_out()),
        documents=documents,
    )


# ======================================================================
# 共起
# ======================================================================


@dataclass
class CooccurrenceGraph:
    """語と語のつながり。"""

    nodes: pd.DataFrame  # 語, 出現回数
    edges: pd.DataFrame  # 語1, 語2, 共起回数, Jaccard
    positions: dict[str, tuple[float, float]]


def cooccurrence(
    tokenized: list[list[str]],
    top_words: int = 40,
    min_cooccurrence: int = 3,
    seed: int = 0,
) -> CooccurrenceGraph:
    """同じ文書に一緒に出てくる語のペアを数え、ネットワークにする。

    「一緒に出る」だけだと、単に頻出な語同士が繋がってしまう。
    Jaccard 係数（どちらかに出た文書のうち、両方に出た割合）で正規化して、
    「この語が出たらあの語も出やすい」という結びつきの強さを測る。
    """
    import networkx as nx

    document_frequency = Counter()
    for words in tokenized:
        document_frequency.update(set(words))

    vocabulary = [word for word, _ in document_frequency.most_common(top_words)]
    allowed = set(vocabulary)

    pair_counts = Counter()
    for words in tokenized:
        present = sorted(set(words) & allowed)
        pair_counts.update(combinations(present, 2))

    rows = []
    for (left, right), count in pair_counts.items():
        if count < min_cooccurrence:
            continue
        union = document_frequency[left] + document_frequency[right] - count
        rows.append(
            {
                "語1": left,
                "語2": right,
                "共起回数": count,
                "Jaccard": round(count / union, 4) if union else 0.0,
            }
        )
    edges = pd.DataFrame(rows, columns=["語1", "語2", "共起回数", "Jaccard"])

    graph = nx.Graph()
    graph.add_nodes_from(vocabulary)
    for row in rows:
        graph.add_edge(row["語1"], row["語2"], weight=row["Jaccard"])

    # 疎なグラフでも見やすいよう、バネモデルで配置する
    positions = nx.spring_layout(graph, seed=seed, k=None, weight="weight")

    nodes = pd.DataFrame(
        {
            "語": vocabulary,
            "出現文書数": [document_frequency[w] for w in vocabulary],
            "つながり数": [graph.degree(w) for w in vocabulary],
        }
    )
    return CooccurrenceGraph(
        nodes=nodes,
        edges=edges.sort_values("Jaccard", ascending=False).reset_index(drop=True),
        positions={k: (float(v[0]), float(v[1])) for k, v in positions.items()},
    )


# ======================================================================
# トピックモデル
# ======================================================================

TOPIC_METHODS: dict[str, str] = {
    "nmf": "NMF（行列を分解する）",
    "lda": "LDA（確率モデル）",
}


@dataclass
class TopicModel:
    """トピックモデルの結果。"""

    method: str
    n_topics: int
    #: (n_topics, n_words) トピックごとの語の重み
    components: np.ndarray
    #: (n_documents, n_topics) 文書ごとのトピック構成比
    document_topics: np.ndarray
    vocabulary: list[str]

    def top_words(self, topic: int, top: int = 10) -> list[tuple[str, float]]:
        weights = self.components[topic]
        order = np.argsort(weights)[::-1][:top]
        return [(self.vocabulary[i], float(weights[i])) for i in order]

    def label(self, topic: int, n_words: int = 3) -> str:
        """トピックを代表語で呼ぶ。番号だけでは何の話題か分からないため。"""
        words = [word for word, _ in self.top_words(topic, n_words)]
        return f"トピック{topic + 1}: " + "・".join(words)

    def summary(self, top: int = 8) -> pd.DataFrame:
        rows = []
        for topic in range(self.n_topics):
            share = float(np.mean(np.argmax(self.document_topics, axis=1) == topic))
            rows.append(
                {
                    "トピック": topic + 1,
                    "代表語": "、".join(w for w, _ in self.top_words(topic, top)),
                    "主トピックの割合": round(share, 3),
                }
            )
        return pd.DataFrame(rows)


def fit_topics(
    vectorized: Vectorized, n_topics: int = 6, method: str = "nmf", seed: int = 0
) -> TopicModel:
    """文書集合から話題を取り出す。

    NMF は行列を「トピック×語」と「文書×トピック」に分解する。
    LDA は「各文書はいくつかの話題の混ざりもの」という確率モデルを当てはめる。
    どちらも教師なし — 正解ラベルを与えずに話題を見つける。
    """
    if method == "lda":
        model = LatentDirichletAllocation(
            n_components=n_topics, random_state=seed, learning_method="batch",
            max_iter=30,
        )
    elif method == "nmf":
        model = NMF(
            n_components=n_topics, random_state=seed, init="nndsvda", max_iter=400
        )
    else:
        raise ValueError(f"未知のトピックモデル: {method}")

    document_topics = model.fit_transform(vectorized.matrix)
    return TopicModel(
        method=method,
        n_topics=n_topics,
        components=np.asarray(model.components_),
        document_topics=np.asarray(document_topics),
        vocabulary=vectorized.vocabulary,
    )


# ======================================================================
# 文書の配置とクラスタリング
# ======================================================================


def embed_documents(vectorized: Vectorized, seed: int = 0) -> np.ndarray:
    """TF-IDF 行列を 2 次元に落とす。

    TF-IDF は数千次元の疎行列なので、まず SVD（疎行列版の PCA）で
    50 次元程度に減らしてから t-SNE にかける。いきなり t-SNE に渡すと
    距離の計算が不安定になり、時間もかかる。
    """
    from sklearn.manifold import TSNE

    n_documents = vectorized.matrix.shape[0]
    n_components = int(min(50, n_documents - 1, vectorized.matrix.shape[1] - 1))
    reduced = TruncatedSVD(n_components=max(2, n_components), random_state=seed)
    dense = reduced.fit_transform(vectorized.matrix)

    perplexity = float(min(30, max(5, (n_documents - 1) / 3)))
    tsne = TSNE(
        n_components=2, perplexity=perplexity, max_iter=500,
        init="pca", random_state=seed,
    )
    return np.asarray(tsne.fit_transform(dense))


def cluster_documents(vectorized: Vectorized, n_clusters: int = 5, seed: int = 0):
    """TF-IDF ベクトルを k-means でまとめる。

    Returns:
        (labels, 各クラスタの代表語)
    """
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(vectorized.matrix)

    representatives: dict[int, list[str]] = {}
    for cluster in range(n_clusters):
        centre = model.cluster_centers_[cluster]
        order = np.argsort(centre)[::-1][:6]
        representatives[cluster] = [vectorized.vocabulary[i] for i in order]
    return np.asarray(labels), representatives


# ======================================================================
# 分類
# ======================================================================


@dataclass
class ClassificationResult:
    """テキスト分類の結果。"""

    labels: list[str]
    y_true: np.ndarray
    y_predicted: np.ndarray
    accuracy: float
    report: pd.DataFrame
    #: クラスごとの、判断根拠になった語
    top_features: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_predicted, labels=self.labels)


def classify(
    vectorized: Vectorized,
    targets: pd.Series,
    n_splits: int = 5,
    seed: int = 0,
) -> ClassificationResult:
    """TF-IDF ベクトルから文書のラベルを当てる。

    交差検証で予測を作るので、どの文書も「自分を学習に使っていないモデル」に
    予測される。テキストでも、ベクトルにしてしまえば普通の分類問題になる。
    """
    y = targets.astype(str).to_numpy()
    labels = sorted(set(y.tolist()))

    # 最少クラスの件数より分割数を大きくできない
    minimum = min(Counter(y).values())
    splits = int(max(2, min(n_splits, minimum)))

    model = LogisticRegression(max_iter=2000, random_state=seed)
    predicted = cross_val_predict(model, vectorized.matrix, y, cv=splits)

    report = pd.DataFrame(
        classification_report(y, predicted, output_dict=True, zero_division=0)
    ).T
    report = report.loc[labels]
    report = report.rename(
        columns={
            "precision": "適合率",
            "recall": "再現率",
            "f1-score": "F1",
            "support": "件数",
        }
    ).round(3)

    # 判断根拠を見るために、全データで一度学習し直して係数を取る
    model.fit(vectorized.matrix, y)
    coefficients = np.atleast_2d(model.coef_)
    top_features: dict[str, list[tuple[str, float]]] = {}
    for i, label in enumerate(model.classes_):
        row = coefficients[i] if coefficients.shape[0] > 1 else coefficients[0]
        order = np.argsort(row)[::-1][:8]
        top_features[str(label)] = [
            (vectorized.vocabulary[j], float(row[j])) for j in order
        ]

    return ClassificationResult(
        labels=labels,
        y_true=y,
        y_predicted=np.asarray(predicted),
        accuracy=float(np.mean(y == predicted)),
        report=report,
        top_features=top_features,
    )


def majority_baseline(targets: pd.Series) -> float:
    """常に最多クラスと答えたときの正解率。分類の基準線。"""
    counts = targets.astype(str).value_counts()
    return float(counts.iloc[0] / counts.sum())
