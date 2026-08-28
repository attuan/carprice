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
