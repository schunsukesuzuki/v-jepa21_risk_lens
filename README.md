# V-JEPA Video Risk Lens

Note: no supervised probe implemented

`V-JEPA Video Risk Lens` is a Dockerized MVP for a **video representation-change analyzer**.

The project is intentionally designed as a **V-JEPA 2.1 representation-change demo**, not as a generic video-captioning system. It uses V-JEPA 2.1 as a frozen video encoder, detects temporal changes in clip embeddings, combines those changes with simple motion features, and turns the result into a human-readable explanation.

This service is designed to analyze videos such as the following:

- [1 Tokyo 23 Wards Land Price Value Map](https://www.youtube.com/watch?v=poJX_WaP_oM)

The current implementation supports two explanation styles:

1. **Physical / operational risk mode**  
   For short smartphone, dashcam, warehouse, robot-view, or hand-work videos.

2. **Geo / Urban Intelligence mode**  
   For animated maps such as land-price maps, area-value maps, population maps, vacancy maps, accessibility maps, and other geo-spatial decision-support visualizations.

The second mode is useful for demonstrating how animated geo-spatial evidence can be transformed into **area intelligence**, **urban-state transition detection**, and **decision-support text**.

---

## Concept

```text
Upload video
  ↓
Browser-compatible preview is generated
  ↓
Sample frames / clips
  ↓
Extract V-JEPA 2.1 clip embeddings when a local checkpoint is mounted
  ↓
Compute temporal representation deltas + classical motion energy
  ↓
Detect state-change / transition windows
  ↓
Apply explanation mode:
    - physical-risk explanation
    - geo / urban-intelligence explanation
  ↓
Render results in React UI
```

This MVP does **not** claim that V-JEPA directly predicts a future video or directly identifies exact objects, wards, or municipalities. Instead, it treats sharp changes in V-JEPA clip representations as **state-transition candidates** and maps them into an operational or geo-spatial explanation layer.

---

## Key features

- Docker Compose setup for frontend and backend
- React / Vite frontend
- FastAPI backend
- PyTorch / OpenCV video processing
- V-JEPA 2.1 checkpoint loading
- Lazy model loading on `/api/analyze`
- Explicit no-model demo mode
- Browser-compatible video preview endpoint
- Timeline of state-change / transition windows
- Risk / transition score display
- Geo / Urban Intelligence explanation card
- English / Japanese toggle for the Geo / Urban Intelligence card and Explanation card
- Explanation text rendered as sentence-level bullet points
- Diagnostics JSON retained for debugging and inspection
- Optional Rust/Axum gateway profile

---

## Repository structure

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

## Start

```bash
cp .env.example .env
docker compose up --build
```

Open the UI:

```text
http://localhost:5173
```

Backend health:

```text
http://localhost:8000/api/health
```

Frontend proxied health route:

```text
http://localhost:5173/api/health
```

The frontend calls same-origin `/api/*`. Vite proxies those requests to the backend container at `http://backend:8000`.

---

## Important Docker note: fixed container names

The compose file may use fixed container names such as:

```text
vjepa-risk-lens-backend
vjepa-risk-lens-frontend
vjepa-risk-lens-rust-gateway
```

If you run multiple copies of the project from different directories, Docker can report a name conflict.

Remove old containers:

```bash
docker rm -f vjepa-risk-lens-backend vjepa-risk-lens-frontend vjepa-risk-lens-rust-gateway
```

Then start again:

```bash
docker compose up --build
```

Or, from the previous project directory:

```bash
docker compose down --remove-orphans
```

---

## V-JEPA 2.1 checkpoint placement

The expected checkpoint is:

```text
vjepa2_1_vitb_dist_vitG_384.pt
```

Supported host-side placements:

```text
./models/vjepa2_1_vitb_dist_vitG_384.pt
```

or:

```text
./models/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

Recommended default `.env` values:

```env
MODEL_DIR=/models
MODEL_MODE=auto
MODEL_NAME=vjepa2_1_vit_base_384
VJEPA_CHECKPOINT_PATH=/models/vjepa2_1_vitb_dist_vitG_384.pt
VJEPA_CHECKPOINT_KEY=ema_encoder
ALLOW_ONLINE_HUB_DOWNLOAD=true
LOAD_MODEL_ON_STARTUP=false
```

The official `vjepa2_1_vitb_dist_vitG_384.pt` file is a checkpoint, not a TorchScript module. The backend builds the architecture using:

```text
vjepa2_1_vit_base_384(pretrained=False)
```

and loads the checkpoint's `ema_encoder` weights.

When loaded successfully, the UI and API should show a backend name similar to:

```text
vjepa_checkpoint:vjepa2_1_vitb_dist_vitG_384.pt:vjepa2_1_vit_base_384
```

---

## Offline model loading

For strict offline runtime, the checkpoint alone is not enough because the architecture definition is also required.

Place a local clone of the official V-JEPA2 repository here:

```text
models/
  vjepa2_repo/
    hubconf.py
    src/
    ...
```

Then set:

```env
ALLOW_ONLINE_HUB_DOWNLOAD=false
VJEPA_REPO_DIR=/models/vjepa2_repo
```

If `discovered_model_files` is empty in `/api/health`, the backend container cannot see your mounted model file.

---

## Lazy model loading

The backend does **not** load V-JEPA at process startup by default. This avoids startup crashes and keeps `/api/health` lightweight.

Default:

```env
LOAD_MODEL_ON_STARTUP=false
```

The model is loaded when `/api/analyze` is called.

Only use this if you intentionally want startup to block until model loading completes:

```env
LOAD_MODEL_ON_STARTUP=true
```

---

## Explicit no-model demo mode

The app does not silently create a random-initialized model.

If no V-JEPA model is mounted, normal analysis returns a clear error. A no-model classical demo exists only when both conditions are met:

1. Backend environment:

```bash
ALLOW_DEMO_FALLBACK=true docker compose up --build
```

2. UI checkbox:

```text
Use explicit classical demo mode if no V-JEPA model is mounted
```

In that mode, the backend uses classical color histogram and motion features. The result is labeled as a classical demo so it cannot be confused with V-JEPA output.

---

## Browser video preview

The UI shows a video preview after upload.

Some MP4 files generated by OpenCV use codecs such as `mp4v`, which may be readable by OpenCV but not by the browser `<video>` element. To make preview reliable, the backend provides:

```text
POST /api/preview
```

This endpoint transcodes the uploaded video to browser-compatible MP4:

```text
H.264 / yuv420p / faststart MP4
```

The frontend first creates a local preview URL, then replaces it with the backend-transcoded preview when ready.

Preview conversion status is shown in the UI:

```text
Preparing browser-compatible preview...
Preview converted for browser playback.
```

---

## UI workflow

1. Open `http://localhost:5173`
2. Choose a video
3. Confirm that the preview appears and plays
4. Select explanation mode:
   - `Auto detect`
   - `Geo / urban value map`
   - `Area intelligence`
   - `Physical-risk only`
5. Select language:
   - `English`
   - `日本語`
6. Click `Analyze`

The result view includes:

- model/backend status
- video preview
- risk or urban-transition score
- timeline
- optional Geo / Urban Intelligence interpretation
- current state
- predicted near-future change
- detected state-change windows
- explanation
- diagnostics JSON

---

## Explanation modes

### Auto detect

`Auto detect` chooses the Geo / Urban Intelligence layer when the filename or context suggests map-like or geo-spatial content.

Examples:

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

Otherwise, the app keeps the physical / operational risk explanation.

### Physical-risk only

Use this for videos such as:

- smartphone videos
- dashcam clips
- warehouse / factory monitoring clips
- robot-view videos
- hand-work videos

The explanation focuses on:

- state change
- motion discontinuity
- possible contact instability
- operational risk windows

### Geo / Urban Intelligence

Use this for videos such as:

- land-price maps
- area-value maps
- population change maps
- vacancy probability maps
- accessibility maps
- station-area or corridor animations
- animated GIS dashboards

In this mode, the app interprets V-JEPA representation changes as:

- map-state transition candidates
- hotspot emergence
- value-gradient shifts
- corridor effects
- peripheral weakening
- spatial regime-shift cues

The result should not be read as a physical object-risk prediction. It is an **urban-state transition explanation layer**.

---

## English / Japanese display

The UI includes a `Language` selector.

Currently, the following sections support English / Japanese switching:

- `Geo / Urban Intelligence interpretation`
- `Explanation`

The backend output remains deterministic and language-neutral where possible. The frontend renders the selected language.

The Explanation section is displayed as sentence-level bullet points for readability.

---

## API

### `GET /api/health`

Returns backend and model-loader status.

Example fields:

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

Multipart form:

```text
video: video file
```

Returns:

```text
video/mp4
```

Purpose:

```text
Convert uploaded video to browser-compatible preview MP4.
```

### `POST /api/analyze`

Multipart form:

```text
video: video file
demo_mode: true | false
domain_mode: auto | geo_urban | area_intelligence | off
```

Response contains fields such as:

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

Note: the internal field name `microbase_angle` may remain for backward compatibility, but the UI anonymizes it as `Area intelligence angle`.

---

## Generate a synthetic test video

After containers are running:

```bash
docker compose exec backend python /scripts/make_synthetic_video.py /data/uploads/synthetic_warehouse_risk.mp4
```

Alternatively, if OpenCV is installed locally:

```bash
python scripts/make_synthetic_video.py data/uploads/synthetic_warehouse_risk.mp4
```

Upload the generated MP4 from:

```text
data/uploads/
```

---

## Optional LLM rewrite

The app works without an LLM.

If you want to rewrite the deterministic explanation through local Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1
```

The LLM prompt should not invent object identities, exact place names, or ward-level findings unless those are grounded by metadata, OCR, or GIS sidecar data.

---

## Optional Rust gateway

The Rust gateway is included as an extension point for future orchestration.

```bash
docker compose --profile gateway up --build
```

Gateway health:

```text
http://localhost:8080/health
http://localhost:8080/api/health
```

---

## Notes on GPU

This ZIP uses CPU PyTorch so it builds on ordinary Docker installations.

For GPU:

1. Replace the backend Dockerfile's PyTorch installation with the matching CUDA wheel/index.
2. Run Docker with GPU access.
3. Set:

```env
DEVICE=cuda
```

The Python code will use CUDA when available.

---

## Troubleshooting

### `Failed to fetch`

The frontend should call same-origin `/api/*`, not `http://localhost:8000` directly.

Check:

```bash
curl http://localhost:5173/api/health
curl http://localhost:8000/api/health
```

Logs:

```bash
docker compose logs -f frontend
docker compose logs -f backend
```

### `502 Bad Gateway`

The Vite proxy is running, but the backend is not reachable or has crashed.

Check:

```bash
curl http://localhost:8000/api/health
docker compose logs backend
```

### `No V-JEPA model was loaded`

Check that the model file is visible inside the container.

Expected host path:

```text
models/vjepa2_1_vitb_dist_vitG_384.pt
```

Expected container path:

```text
/models/vjepa2_1_vitb_dist_vitG_384.pt
```

Check health:

```bash
curl http://localhost:8000/api/health
```

Look for:

```json
"discovered_model_files": [
  "/models/vjepa2_1_vitb_dist_vitG_384.pt"
]
```

### `container name is already in use`

Remove old containers:

```bash
docker rm -f vjepa-risk-lens-backend vjepa-risk-lens-frontend vjepa-risk-lens-rust-gateway
```

Then:

```bash
docker compose up --build
```

### Preview frame appears but video does not play

The source video codec may not be browser-compatible.

The frontend now calls:

```text
POST /api/preview
```

and should show:

```text
Preview converted for browser playback.
```

If conversion fails, inspect backend logs:

```bash
docker compose logs backend
```

### Geo / Urban Intelligence mode still says physical-risk terms

Use one of:

```text
Explanation mode: Geo / urban value map
Explanation mode: Area intelligence
```

or choose `Auto detect` and use a filename that clearly indicates map / geo / land-price content.

---

## Current limitations

- No object segmentation
- No OCR for legends or labels
- No georeferencing of pixels to exact wards, stations, or mesh cells
- No direct causal inference over land price or vacancy risk
- No learned downstream action-anticipation head
- No trained classifier for hotspot / corridor / peripheral weakening labels
- Geo / Urban Intelligence explanations are rule-based over V-JEPA representation changes and simple visual statistics
- Exact urban interpretation requires structured data sidecars such as GeoJSON, mesh statistics, land-price tables, population data, vacancy data, station data, and facility reachability features

---

## Suggested next development steps

### 1. Add metadata sidecar

For map videos, add a sidecar file:

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

### 2. Add GIS sidecar

Attach:

```text
GeoJSON
ward polygons
mesh polygons
station coordinates
facility locations
road / rail network
```

Then map transition windows to named areas.

### 3. Replace generic labels

Current labels:

```text
normal
state change
risk
```

Domain-specific labels:

```text
hotspot emergence
value-gradient shift
corridor effect
peripheral weakening
transition plateau
```

### 4. Add structured model heads

Use V-JEPA embedding deltas as triggers, then train small heads over structured data:

```text
embedding delta + GIS features + demographic features + accessibility features
  -> hotspot / weakening / corridor-effect classifier
```

### 5. Add analyst export

Export:

```text
JSON
Markdown report
CSV of transition windows
PNG timeline
GeoJSON features
```

---

## One-sentence positioning

`V-JEPA Video Risk Lens` uses V-JEPA 2.1 as a frozen representation encoder to detect short-term visual state transitions, then translates those transitions into either physical operational risk explanations or geo-spatial area-intelligence explanations.
