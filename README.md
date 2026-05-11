# V-JEPA Video Risk Lens

`V-JEPA Video Risk Lens` is a Dockerized MVP for a **Video Situation / Risk Analyzer**.

It is intentionally built as a representation-change demo, not as a video-captioning system:

```text
Upload video
  -> sample frames / clips
  -> extract V-JEPA / V-JEPA 2.1 clip embeddings if a local model is mounted
  -> compute temporal representation deltas + motion energy
  -> detect state-change / risk windows
  -> generate an operational explanation
  -> show result in React UI
```

The default behavior avoids the previous failure mode: **the app does not silently create a random-init model**. If no model is mounted, the API returns a clear 503 error. A no-model classical demo exists, but it is available only when you explicitly set `ALLOW_DEMO_FALLBACK=true` and check demo mode in the UI.

## Structure

```text
vjepa-risk-lens/
  docker-compose.yml
  .env.example
  backend/          FastAPI + PyTorch/OpenCV video analyzer
  frontend/         React/Vite UI
  rust_gateway/     optional Rust/Axum gateway profile
  models/          mount downloaded V-JEPA/V-JEPA 2.1 files here
  data/             uploaded videos and JSON analysis results
  scripts/          synthetic test video generator
```

## Start

```bash
cp .env.example .env
docker compose up --build
```

Open:

```text
http://localhost:5173
```

Backend health:

```text
http://localhost:8000/api/health
```

## V-JEPA 2.1 ViT-B checkpoint placement

This revision searches both of these host paths:

```text
./models/vjepa2_1_vitb_dist_vitG_384.pt
```

or:

```text
./models/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

Default `.env` values are:

```env
MODEL_DIR=/models
MODEL_MODE=auto
MODEL_NAME=vjepa2_1_vit_base_384
VJEPA_CHECKPOINT_PATH=/models/vjepa2_1_vitb_dist_vitG_384.pt
VJEPA_CHECKPOINT_KEY=ema_encoder
ALLOW_ONLINE_HUB_DOWNLOAD=true
```

The official `vjepa2_1_vitb_dist_vitG_384.pt` file is a checkpoint, not a TorchScript module. The backend therefore builds `vjepa2_1_vit_base_384(pretrained=False)` through the official PyTorch Hub entry and loads the local checkpoint's `ema_encoder` weights into the encoder.

For fully offline runtime, put a local clone of the official repo under `./models/vjepa2_repo` so `./models/vjepa2_repo/hubconf.py` exists, then set:

```env
ALLOW_ONLINE_HUB_DOWNLOAD=false
VJEPA_REPO_DIR=/models/vjepa2_repo
```

The health endpoint includes `discovered_model_files`; if it is empty, the container cannot see the mounted model file.

## Explicit no-model demo mode

For smoke testing before model placement:

```bash
ALLOW_DEMO_FALLBACK=true docker compose up --build
```

Then check **Use explicit classical demo mode** in the UI. The backend will use color histogram + motion features. The result is labeled `classical-demo-explicit`, so it cannot be mistaken for V-JEPA output.

## Generate a synthetic test video

After containers are running:

```bash
docker compose exec backend python /scripts/make_synthetic_video.py /data/uploads/synthetic_warehouse_risk.mp4
```

Alternatively, if OpenCV is installed locally:

```bash
python scripts/make_synthetic_video.py data/uploads/synthetic_warehouse_risk.mp4
```

Upload the generated MP4 from `data/uploads/` in the UI.

## API

### `GET /api/health`

Returns model loader status and diagnostics.

### `POST /api/analyze`

Multipart form:

```text
video: video file
demo_mode: true | false
```

Response contains:

```json
{
  "risk_level": "Medium",
  "risk_score": 0.55,
  "current_state": ["..."],
  "predicted_near_future_change": ["..."],
  "timeline": [{"start": 4.0, "end": 5.0, "label": "state change", "score": 0.62}],
  "detected_state_change": [{"timestamp": 4.2, "title": "motion discontinuity detected", "score": 0.62}],
  "reason": "..."
}
```

## Optional LLM rewrite

The app works without an LLM. To rewrite the deterministic explanation through local Ollama:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1
```

The LLM prompt explicitly says not to invent object identities.

## Optional Rust gateway

The Rust gateway is included as a clean extension point for a future Rust API gateway / orchestration layer.

```bash
docker compose --profile gateway up --build
```

Gateway health:

```text
http://localhost:8080/health
http://localhost:8080/api/health
```

## Notes on GPU

This ZIP uses CPU PyTorch so it builds on ordinary Docker installations. For GPU, replace the backend Dockerfile's PyTorch installation with the matching CUDA wheel/index and run Docker with GPU access. The Python code will use `DEVICE=cuda` when CUDA is available.

## v3: `Failed to fetch` fix

v2 called `http://localhost:8000` directly from the browser. Depending on Docker/browser/network setup, this can fail before the request reaches FastAPI, resulting in a generic `Failed to fetch` message.

v3 calls same-origin `/api/*` from the frontend. The Vite dev server proxies those requests to `http://backend:8000` inside Docker Compose.

Check:

```bash
curl http://localhost:5173/api/health
curl http://localhost:8000/api/health
```

The UI uses the first route. If both fail, inspect backend logs:

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

## v4 note: 502 Bad Gateway fix

If v3 returned `502 Bad Gateway` from `http://localhost:5173/api/health`, the Vite proxy was running but the backend was not ready or crashed while loading the V-JEPA checkpoint at process startup.

v4 no longer loads V-JEPA at backend startup. `/api/health` is lightweight and should return even when the model is not yet loaded. The model is loaded lazily when `/api/analyze` is called.

Check in this order:

```bash
curl http://localhost:8000/api/health
curl http://localhost:5173/api/health
```

If the first command fails, inspect backend logs:

```bash
docker compose logs backend
```

If only the second command fails, inspect frontend proxy logs:

```bash
docker compose logs frontend
```

Expected model placement:

```text
models/
  vjepa2_1_vitb_dist_vitG_384.pt
```

For official checkpoints, the checkpoint alone is not enough for strict offline reconstruction; the V-JEPA2 source code with `hubconf.py` is also needed. Either allow online hub loading with `ALLOW_ONLINE_HUB_DOWNLOAD=true`, or place the official repository here:

```text
models/
  vjepa2_repo/
    hubconf.py
    src/
    ...
```

Only set this when you intentionally want startup to block until the model is loaded:

```env
LOAD_MODEL_ON_STARTUP=true
```
