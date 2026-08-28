# ML Lab — 体験型 機械学習・パターン認識 実践システム

スライダを動かすと決定境界・損失曲面・クラスタが**その場で変わる**、触って学ぶための
ローカル完結型ダッシュボードです。完全無料・個人運用・サーバー不要。

## 起動

```powershell
uv sync
uv run streamlit run app/Home.py
```

→ http://localhost:8501

## 基礎ラボ（Phase 1）

| # | ラボ | 体感できること |
|---|---|---|
| 1 | 決定境界ラボ | モデルごとに境界の形が全く違うこと |
| 2 | 過学習と正則化ラボ | 訓練誤差は下がるのに検証誤差が U 字を描くこと |
| 3 | 勾配降下法ラボ | 学習率が大きすぎると発散、小さすぎると進まないこと |
| 4 | クラスタリングラボ | k-means は球状しか取れず、DBSCAN なら三日月も取れること |
| 5 | 次元削減ラボ | t-SNE の見た目が perplexity で激変すること |
| 6 | 評価指標ラボ | 「精度 98%」がなぜ嘘になりうるか |
| 7 | アンサンブルラボ | 弱い学習器の足し合わせが強い境界を作ること |
| 8 | テーブルデータラボ | 実データの前処理・評価・解釈（SHAP）の一通り |

## データ基盤（Phase 2）

**データカタログ**ページで、公開データを取り込んで Parquet に溜め、DuckDB で SQL を打てます。
サーバーは 1 つも立てません。

| 取得元 | 種類 | APIキー | 内容 |
|---|---|---|---|
| 公開データセット | テーブル | 不要 | iris / wine / 乳がん / 糖尿病 / digits / 住宅価格 |
| 気象データ | 時系列 | 不要 | Open-Meteo。全国 7 都市の日次気温・降水量ほか（1940 年以降） |
| 株価・指数 | 時系列 | 不要 | Yahoo Finance。日経平均・個別株・為替の日次 OHLCV |
| ニュース記事 | テキスト | 不要 | NHK / ITmedia / はてブの RSS。取得のたびに蓄積 |
| 政府統計 e-Stat | テーブル | **要** | 統計表 ID 指定で取得。コードは日本語ラベルに変換済み |

- 保存は `data/raw/<name>.parquet` ＋ 取得条件を残す `<name>.meta.json`
- SQL は読み取り専用（`SELECT` / `WITH` の 1 文のみ）
- 取得したデータは git 管理外。**取り直せるものはリポジトリに入れない**方針

e-Stat のみ無料の利用登録が必要です。取得した appId は `.env.local` に
`ESTAT_APP_ID=...` と書いてください（`.gitignore` 済み）。

## 構成

```
app/
  Home.py           エントリ（ページ登録）
  views/            各ラボ・データカタログ
  components/       共通 UI 部品（カード・解説・入力欄）
  assets/style.css  ダッシュボードの CSS
mllab/
  data/toy.py       合成データ生成
  data/store.py     Parquet 保存 + DuckDB 検索
  data/connectors/  外部データ取得元
  viz/              配色・決定境界・損失曲面
  models/           モデル定義・クラスタリング
content/            各ページの解説 Markdown
tests/              ユニットテスト + 全ページのスモークテスト
data/               取得データの置き場（git 管理外）
```

計算は `mllab/` に集約し、`app/views/*.py` は UI の組み立てのみ。
ノートブックや CLI からも同じロジックを再利用できます。

## 開発

```powershell
uv run pytest -q          # 通常のテスト（通信しない）
uv run pytest -m network  # 外部 API に実際に繋ぐテスト
```

外部サービスが落ちていてもテストが赤くならないよう、通信はモックしています。

## 今後

- **Phase 4** 時系列ラボ（季節分解・自己相関・ラグ特徴量・TimeSeriesSplit）
- **Phase 5** テキストラボ（形態素解析・TF-IDF・トピックモデル）
- **Phase 6** 画像・信号ラボ（フィルタ・HOG・FFT）
- **Phase 7** 実験管理（MLflow）

重い深層学習は Google Colab の無料 GPU に逃がす方針です。
