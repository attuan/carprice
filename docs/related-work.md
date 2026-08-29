# 関連研究メモ

CLAUDE.md の3方式（①LLM直接予測 / ②LLMでモデル選択 / ③LLMで特徴量生成）に対応させて整理。
2026-08-28 時点。アブストラクトのみ確認、本文未読のものが多いので数値は原典で要確認。

## 0. 出発点（伊藤さん共有）

- **Why Large Language Models Fail at Tabular Prediction** (2026-08-03)
  https://arxiv.org/abs/2608.02412 — Garnelo & Czarnecki
  31ベンチマークで5仮説を検証。ノイズ耐性・CSV形式・数値トークン化・テスト点数は
  いずれも原因ではなく、**決定的なのは入力次元数**。9手法中 LLM だけが次元増加で精度が落ちる。
  2次元では距離ベース手法のように振る舞うが、高次元でそのパターンが崩壊する。
  → 方式①に対する最も強い反証。ただし裏返せば「低次元なら戦える」とも読める。

## 1. LLM で直接価格を予測する（方式①）

- **TabLLM: Few-shot Classification of Tabular Data with LLMs** (AISTATS 2023)
  https://arxiv.org/abs/2210.10723 — 行を自然文にシリアライズしてLLMに入力。
  zero-shot でも非自明な精度、**超少数サンプル域では GBDT に競合**。分類タスク。
- **Quantile Regression with LLMs for Price Prediction** (Findings of ACL 2025)
  https://arxiv.org/abs/2506.06657 — Mistral-7B に分位点回帰ヘッドを付け、
  非構造テキストから**予測分布**を出す。点推定・分布推定の両方で従来手法を上回った。
  価格予測3データセット使用。**本プロジェクトに最も近い設計**。
- **LLMs on Tabular Data with Limited Semantics: Industrial Car Retrofit Prediction** (2026-06-13)
  https://arxiv.org/abs/2606.15314 — 車両登録28万件。決定木アンサンブル vs LLM埋め込み
  vs 直接プロンプト分類(Claude Sonnet 4) vs ML+LLMスタッキングを比較。
  結論は **「単体では木系アンサンブルが最強、LLM は補完として有効」**。
  埋め込み特徴は効くが、意味情報が失われると直接プロンプトの精度は落ちる。
- **LLMs on Tabular Data: Prediction, Generation, Understanding — A Survey**
  https://arxiv.org/abs/2402.17944 — この分野の全体像。最初に読むならこれ。
  論文リスト: https://github.com/tanfiona/LLM-on-Tabular-Data-Prediction-Table-Understanding-Data-Generation

## 2. LLM でモデル選択を自動化する（方式②）

- **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML**
  https://arxiv.org/abs/2410.02958 — データ取得からモデル探索・HPOまで全工程をエージェント化。
- **SELA: Tree-Search Enhanced LLM Agents for Automated ML**
  https://arxiv.org/abs/2410.17238 — MCTS で探索。ベースラインに 65〜80% の勝率。
- **Pre-Hoc Predictions in AutoML: LLMs for Model Selection on Tabular Datasets**
  https://arxiv.org/html/2510.01842 — 学習前にLLMがモデルを選ぶ。OpenML で検証。
  「モデル選択だけLLM」という一番軽い形。方式②の最小構成として参考になる。
- 論文まとめ: https://github.com/t-harden/LLM4AutoML

## 3. LLM で特徴量生成を自動化する（方式③）

- **CAAFE: Context-Aware Automated Feature Engineering** (NeurIPS 2023)
  https://arxiv.org/abs/2305.03403 — Hollmann, Müller, Hutter。
  データセットの説明文からLLMが特徴量生成コード + その根拠説明を反復生成。
  14データセット中11で改善、平均 ROC AUC 0.798 → 0.822。
  実装: https://github.com/noahho/CAAFE
  → **方式③のリファレンス実装。まずこれを シエンタデータで動かすのが最短。**
- **LLM-Select: Feature Selection with LLMs** (TMLR 2025)
  https://arxiv.org/abs/2407.02694 — 列名とタスク説明だけで重要特徴量を選ばせる。
  訓練データを一切見ずに LASSO 等に匹敵。「どの情報を集めるべきか」の判断にも使える。

## 4. 比較対象になるベースライン（重要）

LLMを使わない側の到達点を押さえておかないと「改善した」と言えない。

- **TabPFN (Nature 2025)** https://www.nature.com/articles/s41586-024-08328-6
  1万サンプル以下の表形式データで既存手法を大差で上回る基盤モデル。
  **シエンタデータ（数千行規模）はまさにこの射程**。
  TabPFN-2.5: https://arxiv.org/abs/2511.08667 （5万行 / 2000特徴量まで拡張）
- **ProbSAINT: Probabilistic Tabular Regression for Used Car Pricing** (2024-03)
  https://arxiv.org/abs/2403.03812 — 中古車価格そのもの。掲載日数に応じた動的価格付けと
  不確実性の定量化。点推定はブースティング同等。**ドメインが完全一致するので必読。**
- **How much is my car worth? Random Forest による中古車価格予測**
  https://arxiv.org/abs/1711.06970 — 古典的ベースライン。

## 示唆

現時点の文献から読み取れること:

1. **「LLM単体で表形式回帰」は分が悪い。** 2608.02412 と 2606.15314 が独立に同じ結論。
   方式①をメインに据えるのはリスクが高い。
2. **勝ち筋は「LLM = 特徴量生成器 / 埋め込み器」、予測器は木系または TabPFN。**
   2606.15314 のスタッキング構成、CAAFE がこの型。CLAUDE.md の
   「定性情報をダミー変数でしか扱えなかった」という問題意識にも直接刺さる。
3. **ベースラインは GBDT だけでなく TabPFN も置くべき。** データ規模が射程内で、
   これを超えられないと「LLMで改善」の主張が弱くなる。
4. 価格は分布で出す価値がある（ProbSAINT / 2506.06657）。点推定のRMSEだけでなく
   予測区間の評価も指標に入れると差別化しやすい。

---

# 追補: 特徴量エンジニアリング自動化（AutoFE）の先行事例

2026-08-29 追記。方式③（＝伊藤さん仕様書の機能A）に絞って掘り下げた結果。
**この分野は2015年から続く独立した研究領域で、LLM版もすでに4世代ある。**
「先行事例が無い」という前提でゼロから作ると、確実に車輪の再発明になる。

## A. LLM 以前の系譜（2015〜2022）— 「演算子を総当たりして選別する」

いずれも「既存の列に四則演算・log・集約などの変換を機械的に大量適用し、
効いたものだけ残す（expand-and-reduce）」という同じ骨格を持つ。
LLM が来る前の到達点であり、**LLM版の比較対象は常にこれら**。

- **Deep Feature Synthesis / Featuretools** (DSAA 2015, MIT)
  https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf
  複数テーブルのリレーションを辿って集約特徴量を自動生成。この分野の元祖。
  Kaggle 的なコンペで906チーム中615チームに勝った、というのが当時の宣伝文句。
- **Cognito** (ICDMW 2016, IBM) — 変換を階層的・貪欲に探索。
- **ExploreKit** (ICDM 2016) — 候補生成 + 学習済みランカーで選別。
- **Feature Engineering for Predictive Modeling using RL** (AAAI 2018, Khurana ら)
  変換の適用順を強化学習で決める路線の代表。
- **SAFE** (2020) https://arxiv.org/abs/2003.02556 — 産業スケール向けの高速版。
- **autofeat** — 非線形変換を大量生成 → 線形モデルで選別。
- **OpenFE** (ICML 2023) https://arxiv.org/abs/2211.12507
  LightGBM ベースの選別。**「専門家を超える」を標榜する現行最強クラスの非LLM手法。**
  → 機能Aを評価するなら、LLM無しのこれを必ず並べること。

## B. LLM 以後の系譜（2023〜）— 「列名と背景知識から意味のある特徴量を書く」

Aとの違いは、列名やデータセット説明文という**意味情報**を使う点。
「価格 ÷ 走行距離」のような、人間なら思いつくが総当たりでは出にくい特徴量を狙う。

- **CAAFE** (NeurIPS 2023) https://arxiv.org/abs/2305.03403 ※上記1章にも記載
  第1世代。説明文 → LLM が pandas コードを生成 → 検証スコアで採否。
- **FeatLLM** (ICML 2024) https://arxiv.org/abs/2404.09491
  「Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning」
  LLM にルール（条件分岐）を書かせて**二値特徴量**に落とし、線形回帰など軽いモデルで予測。
  少数サンプル（few-shot）設定に特化。TabLLM・STUNT を上回る。
- **OCTree** (NeurIPS 2024) https://arxiv.org/abs/2406.08527
  実装: https://github.com/jaehyun513/OCTree
  CAAFE の弱点＝「検証スコアという数字しかLLMに返さない」を改善。
  **決定木を自然言語に書き下してLLMにフィードバック**し、反復で規則を改善する。
- **LLM-FE** (TMLR 2026) https://arxiv.org/abs/2503.14434
  実装: https://github.com/nikhilsab/llmfe
  特徴量生成を**プログラム探索**とみなし、LLM を進化的アルゴリズムの変異操作として使う。
  複数の系統（island）を並列に進化させ、高性能プログラムを外部メモリに蓄積。
  CAAFE / OCTree が単一経路の逐次改善なのに対し、こちらは並列探索。現時点の最新到達点。
- **LATTEArena** (2026-06) https://arxiv.org/abs/2606.09004
  **この分野の統一ベンチマーク。** 15手法を部品に分解して24構成を比較し、
  精度だけでなく**トークン消費量と実行の頑健性**まで測っている。
  4000件超の実行ログ公開。→ 評価設計を考えるときの下敷きにできる。

## C. 逆風・注意すべき否定的結果（重要）

「LLMで特徴量生成すれば改善する」は無条件には成り立たない。

- **LLMs Engineer Too Many Simple Features For Tabular Data** (2024-10)
  https://arxiv.org/abs/2410.17787
  4モデル×27データセット。**LLM は加算のような単純な演算子に偏り、
  groupby+集約のような複雑な操作を使えない。** この偏りは精度をむしろ下げうる。
  → 生成された特徴量の演算子分布を集計して偏りを見る、という評価軸が要る。
- **TabPrep** (2026-06) https://arxiv.org/abs/2606.02384
  実装: https://github.com/atschalz/tabprep
  **LLMを一切使わず**、データの構造的パターン3種を狙った前処理・特徴量生成だけで
  TabArena ベンチマークの木系・NN系・基盤モデル全部の性能を底上げし、
  モデル側の工夫による改善幅をしばしば上回った。
  → 「LLMで改善した」と言う前に、**地味な前処理でどこまで行くか**を潰しておく必要がある。

## D. 機能A（埋め込み→近傍分類）に一番近い先行事例

注意: 伊藤さん仕様書の機能Aは「LLMに特徴量生成コードを書かせる」CAAFE系とは**別物**。
非構造列（画像・自由記述）を埋め込んで既存ラベルの近傍で型付き列にする、という設計なので、
系譜としては **AutoFE ではなくマルチモーダル表形式学習**に属する。こちらの先行事例:

- **Benchmarking Multimodal AutoML for Tabular Data with Text Fields** (NeurIPS 2021 D&B)
  https://arxiv.org/abs/2111.02705 — テキスト列を含む表データのベンチマーク。
- **AutoGluon-Multimodal (AutoMM)** (2024) https://arxiv.org/abs/2404.16233
  画像・テキスト・表を統合する AutoML。**機能Aの機能的な競合そのもの。**
- **Bag of Tricks for Multimodal AutoML** (2024) https://arxiv.org/abs/2412.16243
  実務的知見が詰まっている。曰く、テキスト埋め込みは PCA で次元を落として GBDT に入れる、
  ユニーク値50未満のテキスト列はカテゴリ扱いにする、
  ドメイン適応した埋め込みは汎用埋め込みに勝る、集約は Stack-Ensemble が最良。
  → **機能Aの「埋め込みを特徴量にする」部分は、ここに書かれた既知手法とほぼ同じ。**
- **CARTE** / **PORTAL** (2024) https://arxiv.org/abs/2410.13516
  文字列を含む表データ向けの基盤モデル。CatBoost+埋め込みと同程度、という報告。

### 中古車ドメインでのマルチモーダル価格予測

- **AI Blue Book: Vehicle Price Prediction using Visual Features** (2018)
  https://arxiv.org/abs/1803.11227 — 車両画像から価格を予測。画像列を使う場合の先行事例。
- **Multi-modal ML for Vehicle Rating Predictions** (2023) https://arxiv.org/abs/2305.15218
  画像・テキスト・パラメータの3モーダル統合。
- **Deep end-to-end learning for price prediction of second-hand items** (KIS 2020)
- 小売価格推定のマルチモーダル枠組み (ScienceDirect 2025)
  EfficientNet(画像) + GloVe/BiLSTM(テキスト) + 埋め込み(カテゴリ) を late fusion。

## 示唆（追補分）

1. **「特徴量生成の自動化」自体は完全に既存研究。** 新規性を主張するなら生成手法ではなく、
   (a) 中古車という具体ドメインでの検証、(b) 来歴管理・信頼度ルーティングという
   **運用側の設計**、(c) scikit-learn 互換 API としての使いやすさ、のどれかになる。
2. **比較対象は最低3本立て。** ①素の GBDT、②OpenFE / TabPrep（LLM無しAutoFE）、
   ③CAAFE または LLM-FE（LLM有りAutoFE）。②を置かないと「LLMのおかげ」が言えない。
3. **回帰タスクでの検証が手薄い。** CAAFE・FeatLLM・OCTree は分類ベンチマーク中心。
   中古車価格＝回帰で体系的に測ること自体に一定の価値がある。
4. **評価軸に「コスト」と「安定性」を入れる。** LATTEArena がまさにこれをやっている。
   精度だけ見ると、トークンを大量に燃やして0.5%改善、のような結論になりやすい。
5. **演算子の偏りを監視する。** 2410.17787 の指摘どおり、LLM生成特徴量は単純操作に偏る。
   生成物の内訳を集計する仕組みを最初から入れておくと、そのまま考察材料になる。
