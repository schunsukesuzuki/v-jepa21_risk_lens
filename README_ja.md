# V-JEPA Video Risk Lens

短い動画から「状態遷移」と「近未来リスク」を読むための Docker 化 MVP です。

今回の修正版では、`vjepa2_1_vitb_dist_vitG_384.pt` をそのまま `./models` に置いた場合も探索します。以前の版のように `./models/vjepa2` だけを見る実装ではありません。

## 起動

```bash
cp .env.example .env
docker compose up --build
```

UI:

```text
http://localhost:5173
```

API health:

```text
http://localhost:8000/api/health
```

## V-JEPA 2.1 ViT-B checkpoint の配置

以下のどちらでも動くようにしています。

```text
./models/vjepa2_1_vitb_dist_vitG_384.pt
```

または:

```text
./models/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

`.env` のデフォルトは以下です。

```env
MODEL_DIR=/models
MODEL_MODE=auto
MODEL_NAME=vjepa2_1_vit_base_384
VJEPA_CHECKPOINT_PATH=/models/vjepa2_1_vitb_dist_vitG_384.pt
VJEPA_CHECKPOINT_KEY=ema_encoder
ALLOW_ONLINE_HUB_DOWNLOAD=true
```

公式の `vjepa2_1_vitb_dist_vitG_384.pt` は TorchScript ではなく checkpoint 形式です。そのため、このアプリは PyTorch Hub の `facebookresearch/vjepa2` から `vjepa2_1_vit_base_384(pretrained=False)` を作成し、ローカル checkpoint の `ema_encoder` を encoder に読み込みます。

## 完全オフラインで動かす場合

Docker 実行時に GitHub から公式 V-JEPA2 repo を取得したくない場合は、公式 repo をローカルに clone して以下へ置いてください。

```text
./models/vjepa2_repo/hubconf.py
./models/vjepa2_repo/src/...
./models/vjepa2_repo/app/...
```

そのうえで `.env` を以下にします。

```env
ALLOW_ONLINE_HUB_DOWNLOAD=false
VJEPA_REPO_DIR=/models/vjepa2_repo
```

## モデル探索の確認

health endpoint の JSON に以下が出ます。

```json
{
  "model_loaded": true,
  "model_backend": "vjepa_checkpoint:vjepa2_1_vitb_dist_vitG_384.pt:vjepa2_1_vit_base_384",
  "discovered_model_files": ["/models/vjepa2_1_vitb_dist_vitG_384.pt"]
}
```

`discovered_model_files` が空なら、Docker container からモデルファイルが見えていません。まず以下を確認してください。

```bash
docker compose exec backend ls -lah /models
docker compose exec backend find /models -maxdepth 3 -type f
```

## 明示的な no-model demo

モデル配置前の疎通確認だけ行いたい場合:

```bash
ALLOW_DEMO_FALLBACK=true docker compose up --build
```

UI 側で `Use explicit classical demo mode` をチェックして分析します。この場合、出力には `classical-demo-explicit` と表示され、V-JEPA 出力とは区別されます。

## 合成テスト動画

```bash
docker compose exec backend python /scripts/make_synthetic_video.py /data/uploads/synthetic_warehouse_risk.mp4
```

生成された `data/uploads/synthetic_warehouse_risk.mp4` を UI からアップロードしてください。

## 構成

```text
backend/      FastAPI + PyTorch + OpenCV
frontend/     React + Vite
rust_gateway/ 任意の Rust/Axum gateway profile
models/       ダウンロード済みモデルの mount 先
data/         upload/result 永続化
```

## GPU

この ZIP は通常の Docker で動かしやすい CPU PyTorch 版です。CUDA を使う場合は backend Dockerfile の PyTorch install 行を CUDA 対応 wheel/index に差し替え、Docker 側で GPU runtime を有効化してください。

## v3: `Failed to fetch` 対策

v2 では frontend が browser から `http://localhost:8000` に直接 fetch していました。環境によっては backend container が起動していない、CORS 以前に port が見えていない、または `localhost` 解決がずれて `Failed to fetch` だけが表示されます。

v3 では frontend からは同一 origin の `/api/*` を呼び、Vite dev server が Docker Compose 内部の `http://backend:8000` に proxy します。

確認コマンド:

```bash
curl http://localhost:5173/api/health
curl http://localhost:8000/api/health
```

前者が通って後者が通らない場合でも UI は動作します。両方通らない場合は backend container が落ちています。

ログ確認:

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

## v4 note: 502 Bad Gateway 対策

v3 で `http://localhost:5173/api/health` が `502 Bad Gateway` になる場合、frontend の Vite proxy は起動していますが、proxy 先の backend が起動完了していない、または起動時のモデルロードで落ちています。

v4 では backend 起動時に V-JEPA checkpoint を読み込まず、`/api/health` は常に軽量に返す設計へ変更しました。モデルロードは `/api/analyze` 実行時に遅延実行されます。

確認順序:

```bash
curl http://localhost:8000/api/health
curl http://localhost:5173/api/health
```

前者が失敗する場合は backend の起動失敗です。

```bash
docker compose logs backend
```

後者のみ失敗する場合は frontend proxy の問題です。

```bash
docker compose logs frontend
```

モデル配置例:

```text
models/
  vjepa2_1_vitb_dist_vitG_384.pt
```

公式 checkpoint をローカルで復元するには、checkpoint だけでなく V-JEPA2 の `hubconf.py` を含む公式ソースも必要です。オンライン取得を許可する場合は `ALLOW_ONLINE_HUB_DOWNLOAD=true`、完全オフラインにする場合は公式 repo を以下に配置してください。

```text
models/
  vjepa2_repo/
    hubconf.py
    src/
    ...
```

起動時にあえてモデルロードまで確認したい場合のみ、`.env` で次を指定してください。

```env
LOAD_MODEL_ON_STARTUP=true
```
