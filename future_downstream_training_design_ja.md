# 将来設計案: Frozen V-JEPA 表現上での下流タスク学習

## 1. 目的

この資料は、現在の V-JEPA ベースの動画リスク解析プロトタイプに対して、教師データが用意できた場合に追加する下流タスク学習の将来設計案を整理したものです。

現在の MVP では、入力動画を短い clip に分割し、各 clip を V-JEPA で embedding 化し、clip 間の表現差分から状態変化を検出し、ルールベースでリスク説明を生成しています。

教師データが用意できると、この構成を単なる heuristic な変化検出から、教師ありの下流タスク学習へ拡張できます。

基本方針は以下です。

```text
V-JEPA は frozen のまま固定する。
V-JEPA は動画特徴抽出器としてのみ使う。
抽出された embedding の上に軽量な downstream probe を学習する。
```

これにより、大規模な動画 backbone を fine-tuning せずに、業務ドメイン固有のリスク分類、イベント分類、近未来リスク予測を追加できます。

---

## 2. 現在の MVP 構成

現在の処理は以下のように整理できます。

```text
入力動画
  ↓
フレーム抽出
  ↓
clip 分割
  ↓
Frozen V-JEPA encoder
  ↓
clip-level embedding
  ↓
表現差分による変化検出
  ↓
ルールベースの risk scoring
  ↓
テンプレートベースの説明生成
```

現在の構成で分かるのは、主に「何かが変化した」ということです。

一方で、以下のような業務的・意味的ラベルはまだ学習していません。

- ニアミス
- 作業者の危険エリア侵入
- フォークリフトの接近
- 物体落下
- 群衆の異常移動
- 交通ボトルネック
- インフラ上の危険兆候

これらの意味付けには、下流タスク用の教師信号が必要です。

---

## 3. 教師データ追加後の将来アーキテクチャ

教師データが用意できた後は、以下の構成に発展させるのが自然です。

```text
入力動画
  ↓
フレーム抽出
  ↓
clip 分割
  ↓
Frozen V-JEPA encoder
  ↓
clip-level embedding
  ├─ 表現差分 detector
  ├─ risk classification probe
  ├─ event-type classification probe
  └─ temporal risk prediction probe
  ↓
risk / event fusion
  ↓
ドメイン文脈に基づく説明生成
  ↓
frontend / report / audit log
```

責務分離は以下です。

```text
V-JEPA:
  汎用的な動画表現を提供する。

Downstream probe:
  embedding から業務ラベル・リスクラベルへの写像を学習する。

Rule / explanation layer:
  モデル出力を業務上理解可能な説明に変換する。
```

---

## 4. 下流タスク候補

### 4.1 Clip-Level Risk Classification

目的:

```text
clip embedding → low / medium / high risk
```

ラベル例:

```text
low
medium
high
```

これは最も単純な教師あり拡張であり、最初に実装するべきタスクです。

推奨モデル:

- Logistic Regression
- Linear SVM
- Random Forest
- LightGBM
- データが増えた場合は小規模 MLP

初期実装としては以下が妥当です。

```text
StandardScaler + LogisticRegression(class_weight="balanced")
```

理由:

- 学習が簡単
- 挙動を説明しやすい
- 少量データでも比較的安定する
- linear probe として妥当

---

### 4.2 Event-Type Classification

目的:

```text
clip embedding → event type
```

ラベル例:

```text
normal
sudden_motion
approach
crossing
occlusion
object_fall
near_collision
congestion
unknown
```

この probe を追加すると、説明文の質が大きく上がります。

event-type probe がない場合、説明は以下のように抽象的になります。

```text
表現差分が大きい区間が検出されました。
```

event-type probe がある場合、以下のように意味を付与できます。

```text
この区間は near-collision pattern に近いと推定されます。
```

---

### 4.3 Temporal Risk Prediction

目的:

```text
連続する clip embedding → 近未来リスクスコア
```

clip を単独で分類するのではなく、前後の embedding sequence を使います。

```text
[clip_t-2, clip_t-1, clip_t, clip_t+1, clip_t+2]
```

出力例:

```text
risk_score_next_2s
risk_score_next_5s
probability_of_high_risk
```

リスクは単一フレームの見た目ではなく、時間的な遷移によって生じることが多いため、このタスクは実務上重要です。

最初の実装:

```text
前後の embedding を連結 → Logistic Regression / MLP
```

発展形:

```text
Embedding sequence → GRU / Transformer / temporal attention probe
```

---

### 4.4 Normal Embedding に基づく Anomaly Detection

目的:

```text
clip embedding → anomaly score
```

正常データしか用意できない場合に有効です。

候補手法:

- One-Class SVM
- Isolation Forest
- Gaussian Mixture Model
- Mahalanobis distance
- 正常 embedding bank からの kNN distance

これは現在の heuristic な変化検出とは異なります。

現在の MVP:

```text
隣接 clip との差分が大きい → 状態変化候補
```

将来の anomaly detection:

```text
学習済み正常分布から遠い → 異常状態候補
```

---

## 5. 教師データ設計

### 5.1 アノテーション単位

推奨するアノテーション単位は、短い動画区間です。

```text
video_path, start_sec, end_sec, risk_label, event_type
```

例:

```csv
video_path,start_sec,end_sec,risk_label,event_type
data/train/warehouse_001.mp4,0,2,low,normal
data/train/warehouse_001.mp4,2,4,medium,sudden_motion
data/train/warehouse_001.mp4,4,6,high,near_collision
```

フレーム単位の精密アノテーションよりも簡単で、最初の downstream probe には十分です。

---

### 5.2 ラベル設計

初期の risk label は以下で十分です。

```text
low
medium
high
```

初期の event label は以下程度に抑えるのが良いです。

```text
normal
sudden_motion
approach
crossing
occlusion
object_fall
near_collision
unknown
```

最初からラベルを増やしすぎると、各クラスのサンプルが薄くなり、学習が不安定になります。

---

### 5.3 Human-in-the-Loop によるデータ作成

現在の heuristic system は、教師データ作成コストを下げるために使えます。

推奨ワークフロー:

```text
1. 現在の V-JEPA change detector を未ラベル動画に適用する。
2. 表現変化が大きい候補区間を抽出する。
3. 人間が候補区間だけ確認してラベルを付ける。
4. downstream probe を学習する。
5. probe の予測結果を使って新しい候補ラベルを提案する。
6. 人間が予測を修正する。
7. 修正済みデータを training set に追加する。
8. 定期的に probe を再学習する。
```

これにより、現在の MVP は単なるデモではなく、教師データ構築の bootstrap tool として機能します。

---

## 6. Feature Dataset 生成

将来の dataset generation script では、以下のファイルを生成する想定です。

```text
data/probe_features/
  X.npy
  y_risk.npy
  y_event.npy
  meta.jsonl
```

各ファイルの意味:

```text
X.npy:
  shape = [num_samples, feature_dim]

y_risk.npy:
  clip-level または interval-level の risk label

y_event.npy:
  event-type label

meta.jsonl:
  video path, time interval, warnings, source metadata
```

特徴抽出には既存の V-JEPA adapter を使います。

```text
frames_rgb
  ↓
model.encode_clips(frames_rgb, clip_size)
  ↓
clip embeddings
  ↓
アノテーション区間内で平均または temporal pooling
  ↓
training sample
```

---

## 7. Probe 学習パイプライン

初期の学習パイプラインは以下で十分です。

```text
X.npy
  ↓
train / validation split
  ↓
StandardScaler
  ↓
LogisticRegression / LinearSVM / RandomForest
  ↓
evaluation
  ↓
probe model 保存
```

保存成果物例:

```text
probes/
  risk_probe.joblib
  event_probe.joblib
  temporal_risk_probe.joblib
```

最低限見るべき評価指標:

```text
accuracy
macro F1
class-wise precision / recall
confusion matrix
probability を使う場合は calibration curve
```

不均衡データでは、単純な accuracy よりも macro F1 と class-wise recall を重視するべきです。

---

## 8. Probe を用いた推論パイプライン

推論時には、heuristic output と supervised output を組み合わせます。

```text
入力動画
  ↓
V-JEPA embeddings
  ↓
heuristic change score
  ↓
risk_probe prediction
  ↓
event_probe prediction
  ↓
score fusion
  ↓
説明生成
```

融合例:

```text
final_risk_score =
  0.4 × representation_change_score
+ 0.6 × supervised_high_risk_probability
```

重みは設定値として外出しするべきです。

初期段階では、heuristic score を捨てない方がよいです。probe のデータ量が少ない間は、heuristic score が fallback として機能します。

---

## 9. 説明生成設計

説明生成層は、モデル出力とドメインルールの両方を使います。

説明生成層への入力例:

```json
{
  "clip_index": 12,
  "time_range": [24.0, 26.0],
  "change_score": 0.78,
  "risk_label": "high",
  "risk_confidence": 0.83,
  "event_type": "near_collision",
  "event_confidence": 0.74,
  "domain_mode": "warehouse"
}
```

説明例:

```text
この区間は high risk と分類されました。event-type probe は near-collision pattern を示しており、表現差分スコアも高いです。倉庫ドメインでは、この時間帯の移動経路、分離ルール、死角条件を確認する必要があります。
```

説明では、以下を明確に分けるべきです。

```text
model prediction
heuristic signal
domain-specific interpretation
recommended operational action
```

これにより、「V-JEPA自体が意味を理解している」と過大に表現することを避けられます。

---

## 10. 推奨追加モジュール

将来的に追加するファイル案:

```text
backend/
  probe_dataset.py
    ラベル付き動画区間から feature dataset を構築する。

  probe_train.py
    risk / event / temporal probe を学習する。

  probe_model.py
    sklearn または PyTorch ベースの probe class を定義する。

  probe_inference.py
    学習済み probe を読み込み、clip embedding に対して推論する。

  probe_registry.py
    利用可能な probe と metadata を管理する。

  explanation_builder.py
    モデル出力をドメイン文脈に応じた説明文へ変換する。
```

既存ファイルの責務は維持します。

```text
model_loader.py:
  frozen V-JEPA のロードと encoding

analyzer.py:
  動画解析全体の orchestration

schemas.py:
  API response model

main.py:
  FastAPI endpoint
```

---

## 11. Deployment 上の考慮点

### 11.1 Offline-First Model Loading

Frozen V-JEPA model と学習済み probe はローカルに mount します。

```text
/models/
  vjepa2_1_vitb_dist_vitG_384.pt
  vjepa2_repo/
    hubconf.py

/probes/
  risk_probe.joblib
  event_probe.joblib
```

インターネット接続なしで動作することを前提にします。

---

### 11.2 Versioning

各 probe には metadata を持たせます。

```json
{
  "probe_name": "risk_probe",
  "version": "0.1.0",
  "trained_at": "2026-05-12",
  "vjepa_backend": "vjepa2_1_vit_base_384",
  "feature_dim": 768,
  "label_set": ["low", "medium", "high"],
  "training_samples": 1200,
  "macro_f1": 0.71
}
```

この metadata は audit log や analysis output に含めるべきです。

---

### 11.3 Failure Mode

明示的に扱うべき failure mode:

```text
V-JEPA model unavailable
probe file unavailable
feature dimension mismatch
unknown label
low confidence prediction
out-of-distribution embedding
```

probe が失敗した場合は、現在の heuristic change detection に fallback します。

---

## 12. Roadmap

### Phase 1: Data Preparation

- label taxonomy を定義する。
- annotation manifest format を作る。
- 現在の MVP で candidate interval を抽出する。
- 小規模なラベル付き dataset を作る。

### Phase 2: First Probe

- clip-level risk classifier を学習する。
- 最初は Logistic Regression を linear probe として使う。
- API response に probe prediction を追加する。

### Phase 3: Event Probe

- event-type classifier を学習する。
- event-based explanation template を追加する。
- frontend の表示を改善する。

### Phase 4: Temporal Probe

- 前後 clip embedding を追加する。
- near-future risk classifier を学習する。
- risk scoring に temporal context を入れる。

### Phase 5: Productionization

- probe registry を追加する。
- model / probe versioning を追加する。
- monitoring と active learning loop を追加する。
- prediction / explanation の audit log を追加する。

---

## 13. まとめ

将来の教師あり拡張では、まず V-JEPA 自体を fine-tuning するべきではありません。

推奨構成は以下です。

```text
Frozen V-JEPA encoder
  ↓
Clip embeddings
  ↓
Lightweight downstream probes
  ↓
Risk / event prediction
  ↓
Domain-aware explanation
```

この設計は、実装がモジュール化しやすく、少量の教師データから始めやすく、現在の MVP から自然に発展できます。

現在の change detector は、将来的には教師データ候補区間を抽出する bootstrap tool としても機能します。heuristic な変化検出を起点に、人間がラベルを付け、そのラベルから downstream probe を学習し、徐々にドメイン固有の意味理解へ近づける構成が現実的です。
