# V-JEPA Video Risk Lens

`V-JEPA Video Risk Lens` は、**動画の表現変化を検出し、状況・リスク・都市空間上の変化点を説明するための Docker 化 MVP** です。

このプロジェクトは、一般的な動画キャプション生成システムではありません。V-JEPA 2.1 を frozen video encoder として使い、動画クリップ間の embedding 変化を検出し、簡易的な motion feature と組み合わせたうえで、人間が読める説明に変換します。

現状の実装では、主に2つの説明モードを扱います。

1. **Physical / operational risk mode**  
   スマホ動画、ドライブレコーダー、倉庫・工場・店舗の短い監視映像、ロボット視点動画、作業員の手元動画などを想定します。

2. **Geo / Urban Intelligence mode**  
   地価マップ、エリア価値マップ、人口動態マップ、空き家確率マップ、施設到達性マップ、アクセシビリティマップなど、動画化された地理空間・都市データの可視化を想定します。

後者は、動画化された地理空間データを **area intelligence**、**urban-state transition detection**、**意思決定支援テキスト** に変換するデモとして利用できます。

---

## コンセプト

```text
動画をアップロード
  ↓
ブラウザ再生可能な preview を生成
  ↓
フレーム / クリップをサンプリング
  ↓
ローカル checkpoint が配置されていれば V-JEPA 2.1 clip embeddings を抽出
  ↓
時間方向の representation delta + classical motion energy を計算
  ↓
state-change / transition window を検出
  ↓
説明モードを適用
    - physical-risk explanation
    - geo / urban-intelligence explanation
  ↓
React UI に結果を表示
```

この MVP は、V-JEPA が未来動画を直接生成・予測している、あるいは物体名・区名・自治体名を直接特定している、という主張はしません。

ここでの使い方は、V-JEPA の clip representation が急激に変化した時間窓を **state-transition candidate** として扱い、それを operational risk または geo-spatial explanation layer に変換するものです。

---

## 主な機能

- Docker Compose による frontend / backend 起動
- React / Vite frontend
- FastAPI backend
- PyTorch / OpenCV による動画処理
- V-JEPA 2.1 checkpoint loading
- `/api/analyze` 実行時の lazy model loading
- 明示的な no-model demo mode
- ブラウザ再生互換 video preview endpoint
- state-change / transition window の timeline 表示
- risk / transition score 表示
- Geo / Urban Intelligence explanation card
- Geo / Urban Intelligence card と Explanation card の英語 / 日本語切り替え
- Explanation の一文ごとの箇条書き表示
- Diagnostics JSON の保持
- optional Rust / Axum gateway profile

---

## リポジトリ構成

```text
vjepa-risk-lens/
  docker-compose.yml
  .env.example

  backend/
    app/
      main.py                FastAPI routes: health, preview, analyze
      analyzer.py            Video analysis pipeline
      model_loader.py        V-JEPA / V-JEPA 2.1 model loading
      domain_interpreter.py  Geo / Urban Intelligence explanation layer
      schemas.py             API response schemas
      video_io.py            Video sampling utilities
      storage.py             Upload/result storage

  frontend/
    src/
      main.tsx               React UI
      styles.css             UI styles

  rust_gateway/
    optional Rust/Axum gateway profile

  models/
    vjepa2_1_vitb_dist_vitG_384.pt
    vjepa2_repo/             optional local V-JEPA2 source repo for offline use

  data/
    uploads/
    results/
    previews/

  scripts/
    make_synthetic_video.py
```

---

## 起動方法

```bash
cp .env.example .env
docker compose up --build
```

UI:

```text
http://localhost:5173
```

backend health:

```text
http://localhost:8000/api/health
```

frontend 経由の proxied health route:

```text
http://localhost:5173/api/health
```

frontend は same-origin の `/api/*` を呼び出します。Vite dev server が Docker Compose 内部の backend container、つまり `http://backend:8000` に proxy します。

---

## Docker の注意点: 固定 container name

compose file が以下のような固定 container name を使っている場合があります。

```text
vjepa-risk-lens-backend
vjepa-risk-lens-frontend
vjepa-risk-lens-rust-gateway
```

別ディレクトリ・別プロジェクト名で複数コピーを起動すると、Docker 側で同名 container conflict が起きることがあります。

古い container を削除します。

```bash
docker rm -f vjepa-risk-lens-backend vjepa-risk-lens-frontend vjepa-risk-lens-rust-gateway
```

その後、再起動します。

```bash
docker compose up --build
```

または、前回起動したプロジェクトディレクトリで以下を実行します。

```bash
docker compose down --remove-orphans
```

---

## V-JEPA 2.1 checkpoint の配置

想定する checkpoint は以下です。

```text
vjepa2_1_vitb_dist_vitG_384.pt
```

host 側では、以下のどちらかに配置できます。

```text
./models/vjepa2_1_vitb_dist_vitG_384.pt
```

または:

```text
./models/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

推奨される `.env` のデフォルト値は以下です。

```env
MODEL_DIR=/models
MODEL_MODE=auto
MODEL_NAME=vjepa2_1_vit_base_384
VJEPA_CHECKPOINT_PATH=/models/vjepa2_1_vitb_dist_vitG_384.pt
VJEPA_CHECKPOINT_KEY=ema_encoder
ALLOW_ONLINE_HUB_DOWNLOAD=true
LOAD_MODEL_ON_STARTUP=false
```

公式の `vjepa2_1_vitb_dist_vitG_384.pt` は TorchScript module ではなく checkpoint です。そのため backend は以下の architecture を構築します。

```text
vjepa2_1_vit_base_384(pretrained=False)
```

そのうえで、checkpoint 内の `ema_encoder` weights を encoder に load します。

正常に load された場合、UI と API には以下のような backend 名が表示されます。

```text
vjepa_checkpoint:vjepa2_1_vitb_dist_vitG_384.pt:vjepa2_1_vit_base_384
```

---

## offline model loading

完全 offline で動かす場合、checkpoint だけでは architecture 定義が不足します。

公式 V-JEPA2 repository の local clone を以下に置いてください。

```text
models/
  vjepa2_repo/
    hubconf.py
    src/
    ...
```

そのうえで `.env` を以下のように設定します。

```env
ALLOW_ONLINE_HUB_DOWNLOAD=false
VJEPA_REPO_DIR=/models/vjepa2_repo
```

`/api/health` の `discovered_model_files` が空の場合、backend container から model file が見えていません。

---

## lazy model loading

backend はデフォルトでは process startup 時に V-JEPA を load しません。これにより、startup crash を避け、`/api/health` を軽量に返せるようにしています。

デフォルト:

```env
LOAD_MODEL_ON_STARTUP=false
```

モデルは `/api/analyze` が呼ばれたタイミングで load されます。

startup 時に model loading 完了まで block したい場合のみ、以下を使います。

```env
LOAD_MODEL_ON_STARTUP=true
```

---

## 明示的 no-model demo mode

このアプリは random-initialized model を黙って作成しません。

V-JEPA model が mount されていない場合、通常の analyze は明示的な error を返します。no-model classical demo は、以下の2条件を満たした場合のみ利用できます。

1. backend environment:

```bash
ALLOW_DEMO_FALLBACK=true docker compose up --build
```

2. UI checkbox:

```text
Use explicit classical demo mode if no V-JEPA model is mounted
```

この mode では、backend は color histogram と motion feature を使います。結果は classical demo として label されるため、V-JEPA output と誤認されません。

---

## ブラウザ動画 preview

UI では、動画を upload すると preview が表示されます。

OpenCV で生成した MP4 などは、`mp4v` のような codec を使っている場合があります。この場合、OpenCV では読めても、ブラウザの `<video>` element では再生できないことがあります。

そのため、backend は以下の endpoint を提供します。

```text
POST /api/preview
```

この endpoint は upload 動画を以下の形式に変換します。

```text
H.264 / yuv420p / faststart MP4
```

frontend はまず local preview URL を作成し、その後 backend で transcode された preview に置き換えます。

UI には preview conversion status が表示されます。

```text
Preparing browser-compatible preview...
Preview converted for browser playback.
```

---

## UI workflow

1. `http://localhost:5173` を開く
2. 動画を選択する
3. preview が表示・再生されることを確認する
4. explanation mode を選択する
   - `Auto detect`
   - `Geo / urban value map`
   - `Area intelligence`
   - `Physical-risk only`
5. language を選択する
   - `English`
   - `日本語`
6. `Analyze` をクリックする

結果画面には以下が表示されます。

- model / backend status
- video preview
- risk または urban-transition score
- timeline
- optional Geo / Urban Intelligence interpretation
- current state
- predicted near-future change
- detected state-change windows
- explanation
- diagnostics JSON

---

## explanation mode

### Auto detect

`Auto detect` は、filename や context が map-like / geo-spatial content を示している場合に、Geo / Urban Intelligence layer を選択します。

例:

```text
land price
value map
地価
東京23区
23区
map
geo
urban
vacancy
population
area value
```

それ以外の場合は、physical / operational risk explanation を維持します。

### Physical-risk only

以下のような動画向けです。

- smartphone videos
- dashcam clips
- warehouse / factory monitoring clips
- robot-view videos
- hand-work videos

説明は以下に焦点を当てます。

- state change
- motion discontinuity
- possible contact instability
- operational risk windows

### Geo / Urban Intelligence

以下のような動画向けです。

- land-price maps
- area-value maps
- population change maps
- vacancy probability maps
- accessibility maps
- station-area or corridor animations
- animated GIS dashboards

この mode では、V-JEPA の representation change を以下として解釈します。

- map-state transition candidates
- hotspot emergence
- value-gradient shifts
- corridor effects
- peripheral weakening
- spatial regime-shift cues

この結果は、物理的な物体リスク予測として読むべきではありません。これは **urban-state transition explanation layer** です。

---

## 英語 / 日本語表示

UI には `Language` selector があります。

現時点で、以下の section が英語 / 日本語切り替えに対応しています。

- `Geo / Urban Intelligence interpretation`
- `Explanation`

backend output は可能な限り deterministic かつ language-neutral に維持し、frontend 側で選択言語に応じて表示します。

`Explanation` section は読みやすさのため、一文ごとの bullet list として表示します。

---

## API

### `GET /api/health`

backend と model loader の状態を返します。

例:

```json
{
  "ok": true,
  "model_loaded": true,
  "model_backend": "vjepa_checkpoint:vjepa2_1_vitb_dist_vitG_384.pt:vjepa2_1_vit_base_384",
  "discovered_model_files": [
    "/models/vjepa2_1_vitb_dist_vitG_384.pt"
  ]
}
```

### `POST /api/preview`

multipart form:

```text
video: video file
```

戻り値:

```text
video/mp4
```

目的:

```text
upload 動画をブラウザ再生可能な preview MP4 に変換する
```

### `POST /api/analyze`

multipart form:

```text
video: video file
demo_mode: true | false
domain_mode: auto | geo_urban | area_intelligence | off
```

response には以下のような field が含まれます。

```json
{
  "risk_level": "High",
  "risk_score": 0.83,
  "backend": "vjepa_checkpoint:vjepa2_1_vitb_dist_vitG_384.pt:vjepa2_1_vit_base_384",
  "frames": 64,
  "duration": 41.9,
  "current_state": [
    "The uploaded video is treated as an animated urban value map rather than a physical interaction scene."
  ],
  "predicted_near_future_change": [
    "The next useful step is mapping these transition windows to named wards, corridors, station areas, or mesh cells."
  ],
  "timeline": [
    {
      "start": 4.8,
      "end": 5.9,
      "label": "risk",
      "score": 0.83
    }
  ],
  "detected_state_change": [
    {
      "time": 5.3,
      "label": "risk window detected",
      "score": 0.83
    }
  ],
  "explanation": "...",
  "domain_insight": {
    "domain": "geo_urban_intelligence",
    "title": "Geo / Urban Intelligence interpretation for the target urban area",
    "observations": ["..."],
    "interpretation": ["..."],
    "microbase_angle": ["..."],
    "recommended_next_steps": ["..."],
    "caveat": "..."
  },
  "diagnostics": {
    "native_fps": 15,
    "sample_fps": 4,
    "frame_size": 384,
    "clip_size": 8,
    "embedding_shape": [11, 1769472],
    "delta": [0.0, 0.2],
    "risk_curve": [0.07, 0.83]
  }
}
```

注記: `microbase_angle` という internal field name が backward compatibility のため残っている場合があります。ただし UI 上では `Area intelligence angle` として匿名化表示されます。

---

## synthetic test video の生成

containers 起動後:

```bash
docker compose exec backend python /scripts/make_synthetic_video.py /data/uploads/synthetic_warehouse_risk.mp4
```

OpenCV が local に install 済みの場合:

```bash
python scripts/make_synthetic_video.py data/uploads/synthetic_warehouse_risk.mp4
```

生成された MP4 は以下から UI に upload できます。

```text
data/uploads/
```

---

## optional LLM rewrite

この app は LLM なしで動作します。

deterministic explanation を local Ollama 経由で rewrite したい場合:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1
```

LLM prompt では、metadata、OCR、GIS sidecar data に基づかない object identity、正確な地名、区単位の findings を捏造しないようにする必要があります。

---

## optional Rust gateway

Rust gateway は、将来の orchestration layer 用の extension point として含まれています。

```bash
docker compose --profile gateway up --build
```

gateway health:

```text
http://localhost:8080/health
http://localhost:8080/api/health
```

---

## GPU について

この ZIP は CPU PyTorch を使うため、通常の Docker 環境で build できます。

GPU を使う場合:

1. backend Dockerfile の PyTorch install を対応する CUDA wheel / index に差し替える
2. Docker を GPU access 付きで起動する
3. 以下を設定する

```env
DEVICE=cuda
```

Python code は CUDA が利用可能な場合に CUDA を使います。

---

## troubleshooting

### `Failed to fetch`

frontend は `http://localhost:8000` を直接呼ばず、same-origin の `/api/*` を呼びます。

確認:

```bash
curl http://localhost:5173/api/health
curl http://localhost:8000/api/health
```

logs:

```bash
docker compose logs -f frontend
docker compose logs -f backend
```

### `502 Bad Gateway`

Vite proxy は動いているが、backend に到達できない、または backend が crash している状態です。

確認:

```bash
curl http://localhost:8000/api/health
docker compose logs backend
```

### `No V-JEPA model was loaded`

model file が container 内から見えているか確認します。

期待される host path:

```text
models/vjepa2_1_vitb_dist_vitG_384.pt
```

期待される container path:

```text
/models/vjepa2_1_vitb_dist_vitG_384.pt
```

health を確認します。

```bash
curl http://localhost:8000/api/health
```

以下があるか確認します。

```json
"discovered_model_files": [
  "/models/vjepa2_1_vitb_dist_vitG_384.pt"
]
```

### `container name is already in use`

古い container を削除します。

```bash
docker rm -f vjepa-risk-lens-backend vjepa-risk-lens-frontend vjepa-risk-lens-rust-gateway
```

その後:

```bash
docker compose up --build
```

### preview 枠は出るが動画が再生されない

source video codec がブラウザ非対応の可能性があります。

frontend は現在、以下を呼びます。

```text
POST /api/preview
```

正常時は UI に以下が表示されます。

```text
Preview converted for browser playback.
```

変換に失敗する場合は backend logs を確認します。

```bash
docker compose logs backend
```

### Geo / Urban Intelligence mode なのに physical-risk 表現が残る

以下のいずれかを選択します。

```text
Explanation mode: Geo / urban value map
Explanation mode: Area intelligence
```

または `Auto detect` を使い、filename に map / geo / land-price content が明確に分かる語を入れます。

---

## 現状の制約

- object segmentation は未実装
- legend / label の OCR は未実装
- pixel を正確な区・駅・mesh cell に georeference していない
- 地価や空き家 risk に対する直接的な causal inference は未実装
- downstream action-anticipation head は未学習
- hotspot / corridor / peripheral weakening label 用の trained classifier は未実装
- Geo / Urban Intelligence explanation は、V-JEPA representation change と簡易 visual statistics に基づく rule-based layer
- 正確な都市解釈には、GeoJSON、mesh statistics、land-price tables、population data、vacancy data、station data、facility reachability features などの structured data sidecar が必要

---

## 次の開発ステップ

### 1. metadata sidecar の追加

map video に以下のような sidecar file を付与します。

```json
{
  "title": "Tokyo 23 wards land price value map",
  "year_range": "2015-2024",
  "unit": "JPY / square meter",
  "legend": {
    "warm": "higher value",
    "cool": "lower value"
  },
  "source": "your data source",
  "geography": "Tokyo 23 wards"
}
```

### 2. GIS sidecar の追加

以下を接続します。

```text
GeoJSON
ward polygons
mesh polygons
station coordinates
facility locations
road / rail network
```

その後、transition window を named area に対応付けます。

### 3. generic label の置換

現在の label:

```text
normal
state change
risk
```

domain-specific label:

```text
hotspot emergence
value-gradient shift
corridor effect
peripheral weakening
transition plateau
```

### 4. structured model head の追加

V-JEPA embedding delta を trigger とし、structured data 上に小さな head を学習します。

```text
embedding delta + GIS features + demographic features + accessibility features
  -> hotspot / weakening / corridor-effect classifier
```

### 5. analyst export の追加

以下を export します。

```text
JSON
Markdown report
CSV of transition windows
PNG timeline
GeoJSON features
```

---

## one-sentence positioning

`V-JEPA Video Risk Lens` は、V-JEPA 2.1 を frozen representation encoder として使い、短期的な視覚状態遷移を検出し、その遷移を physical operational risk explanation または geo-spatial area-intelligence explanation に変換する MVP です。
