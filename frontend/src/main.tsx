import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type AnyRecord = Record<string, any>;

const API_HEALTH = "/api/health";
const API_ANALYZE = "/api/analyze";
const API_PREVIEW = "/api/preview";

function isRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function safeText(value: unknown, fallback = "not available"): string {
  if (value === undefined || value === null || value === "") return fallback;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return fallback;
  }
}

function getPath(obj: unknown, path: string): unknown {
  if (!isRecord(obj)) return undefined;
  const parts = path.split(".");
  let cur: any = obj;
  for (const part of parts) {
    if (!isRecord(cur) && !Array.isArray(cur)) return undefined;
    cur = cur?.[part];
    if (cur === undefined || cur === null) return undefined;
  }
  return cur;
}

function firstDefined(obj: unknown, paths: string[]): unknown {
  for (const path of paths) {
    const value = getPath(obj, path);
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

function asNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function formatSeconds(value: unknown): string {
  const n = asNumber(value);
  if (n === undefined) return "unavailable";
  return `${n.toFixed(1)}s`;
}

function formatFrameCount(result: unknown): string {
  const direct = firstDefined(result, [
    "frame_count",
    "frames",
    "num_frames",
    "total_frames",
    "n_frames",
    "sampled_frames",
    "video_frame_count",
    "metadata.frame_count",
    "metadata.frames",
    "metadata.num_frames",
    "video.frame_count",
    "video.frames",
    "video.num_frames",
    "video_info.frame_count",
    "video_info.frames",
    "video_info.num_frames",
    "diagnostics.frame_count",
    "diagnostics.frames",
    "diagnostics.num_frames",
    "diagnostics.sampled_frames",
  ]);

  const n = asNumber(direct);
  if (n !== undefined) return `${Math.round(n)} frames`;

  // Last-resort estimate: timeline windows usually correspond to sampled clip windows.
  const timeline = getTimeline(result);
  const clipSize = asNumber(firstDefined(result, ["diagnostics.clip_size", "clip_size"]));
  if (timeline.length > 0 && clipSize !== undefined) {
    return `approx. ${Math.round(timeline.length * clipSize)} sampled frames`;
  }

  return "frames unavailable";
}

function formatDuration(result: unknown): string {
  const direct = firstDefined(result, [
    "duration",
    "duration_s",
    "duration_sec",
    "duration_seconds",
    "video_duration",
    "video_duration_s",
    "metadata.duration",
    "metadata.duration_s",
    "metadata.duration_sec",
    "metadata.duration_seconds",
    "video.duration",
    "video.duration_s",
    "video.duration_sec",
    "video.duration_seconds",
    "video_info.duration",
    "video_info.duration_s",
    "video_info.duration_sec",
    "video_info.duration_seconds",
    "diagnostics.duration",
    "diagnostics.duration_s",
    "diagnostics.duration_sec",
    "diagnostics.duration_seconds",
  ]);

  const n = asNumber(direct);
  if (n !== undefined) return `${n.toFixed(1)}s duration`;

  // Fallback from timeline end.
  const timeline = getTimeline(result);
  const maxEnd = Math.max(
    ...timeline.map((x) => asNumber(firstDefined(x, ["end", "end_s", "end_time", "end_sec"])) ?? -1),
  );
  if (Number.isFinite(maxEnd) && maxEnd > 0) return `approx. ${maxEnd.toFixed(1)}s duration`;

  return "duration unavailable";
}

function getBackendName(result: unknown, health?: unknown): string {
  const raw = firstDefined(result, [
    "backend",
    "model_backend",
    "model_name",
    "encoder_backend",
    "diagnostics.backend",
    "diagnostics.model_backend",
    "diagnostics.model_name",
  ]) ?? firstDefined(health, [
    "backend",
    "model_backend",
    "model_name",
    "encoder_backend",
  ]);

  return safeText(raw, "not available");
}

function getRiskLabel(result: unknown): string {
  return safeText(firstDefined(result, ["risk", "risk_level", "level", "summary.risk", "summary.risk_level"]), "not available");
}

function getMaxScore(result: unknown): string {
  const raw = firstDefined(result, ["max_score", "risk_score", "score", "summary.max_score", "summary.risk_score"]);
  const n = asNumber(raw);
  return n === undefined ? "not available" : n.toFixed(2);
}

function getTimeline(result: unknown): AnyRecord[] {
  const raw = firstDefined(result, ["timeline", "timeline_items", "windows", "segments"]);
  return Array.isArray(raw) ? raw.filter(isRecord) : [];
}

function getEvents(result: unknown): AnyRecord[] {
  const raw = firstDefined(result, [
    "detected_state_change",
    "detected_state_changes",
    "state_changes",
    "events",
    "detections",
  ]);
  return Array.isArray(raw) ? raw.filter(isRecord) : [];
}


function getDomainInsight(result: unknown): AnyRecord | null {
  const raw = firstDefined(result, ["domain_insight", "domainInsights", "domain"]);
  return isRecord(raw) ? raw : null;
}

function getStringArray(result: unknown, paths: string[]): string[] {
  const raw = firstDefined(result, paths);
  if (Array.isArray(raw)) return raw.map((x) => safeText(x)).filter(Boolean);
  if (typeof raw === "string" && raw.trim()) return [raw];
  return [];
}

function getExplanation(result: unknown, backend?: string): string {
  const raw = firstDefined(result, ["explanation", "reason", "summary.explanation", "summary.reason"]);
  let text = safeText(raw, "");

  if (!text) return "";

  // Backend is rendered below as a dedicated block. Remove the duplicate inline
  // backend suffix so long checkpoint names do not break the explanation paragraph.
  if (backend && backend !== "not available") {
    text = text.replace(` Backend: ${backend}.`, " ");
    text = text.replace(`Backend: ${backend}.`, " ");
  }

  // Defensive cleanup for checkpoint-style backend strings.
  text = text.replace(/\s*Backend:\s*vjepa_checkpoint:[^\n]+?(?=\s+This MVP|\s*$)/, " ");
  return text.replace(/\s{2,}/g, " ").trim();
}

function BackendBlock({ value }: { value: unknown }) {
  return <code className="backend-block">{safeText(value, "not available")}</code>;
}

function KV({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="kv">
      <div className="kv-label">{label}</div>
      <div className="kv-value">{children}</div>
    </div>
  );
}

function StatusCard({ health, result }: { health: unknown; result: unknown }) {
  const backend = getBackendName(result, health);
  const modelLoaded = Boolean(firstDefined(health, ["model_loaded"])) || backend.includes("vjepa_checkpoint");
  const analyzedWithVjepa = Boolean(result) && backend.includes("vjepa");
  const title = analyzedWithVjepa
    ? "Analyzed with V-JEPA 2.1"
    : modelLoaded
      ? "Backend ready"
      : "Backend ready";
  const subtitle = result
    ? `${formatFrameCount(result)} / ${formatDuration(result)}`
    : "Model will be loaded on Analyze";

  return (
    <section className="status-card card">
      <div>
        <div className="status-title">{title}</div>
        <div className="status-subtitle">{subtitle}</div>
      </div>
      <KV label="Backend">
        <BackendBlock value={backend} />
      </KV>
    </section>
  );
}

function RiskCard({ result, health }: { result: unknown; health: unknown }) {
  if (!result) return null;
  const backend = getBackendName(result, health);
  return (
    <section className="card">
      <h2>Risk</h2>
      <div className="risk-grid">
        <KV label="Level">
          <span className="risk-level">{getRiskLabel(result)}</span>
        </KV>
        <KV label="Max score">
          <span className="score">{getMaxScore(result)}</span>
        </KV>
        <KV label="Backend">
          <BackendBlock value={backend} />
        </KV>
        <KV label="Video">
          <span>{formatFrameCount(result)} / {formatDuration(result)}</span>
        </KV>
      </div>
    </section>
  );
}

function Timeline({ result }: { result: unknown }) {
  const timeline = getTimeline(result);
  if (!timeline.length) return null;

  return (
    <section className="card">
      <div className="section-header">
        <h2>Timeline</h2>
        <span className="muted">scroll horizontally</span>
      </div>
      <div className="timeline-scroll" tabIndex={0}>
        <div className="timeline-track">
          {timeline.map((item, idx) => {
            const label = safeText(firstDefined(item, ["label", "state", "type", "name"]), "window");
            const start = formatSeconds(firstDefined(item, ["start", "start_s", "start_time", "start_sec"]));
            const end = formatSeconds(firstDefined(item, ["end", "end_s", "end_time", "end_sec"]));
            const score = asNumber(firstDefined(item, ["score", "risk_score", "value"]));
            return (
              <div key={idx} className={`timeline-item ${label.replace(/\s+/g, "-").toLowerCase()}`}>
                <div className="timeline-label">{label}</div>
                <div className="timeline-time">{start}–{end}</div>
                {score !== undefined && <div className="timeline-score">{score.toFixed(2)}</div>}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function TextListCard({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <section className="card">
      <h2>{title}</h2>
      <ul className="plain-list">
        {items.map((item, idx) => <li key={idx}>{item}</li>)}
      </ul>
    </section>
  );
}

function EventsCard({ result }: { result: unknown }) {
  const events = getEvents(result);
  if (!events.length) return null;
  return (
    <section className="card">
      <h2>Detected state change</h2>
      <div className="event-list">
        {events.map((event, idx) => {
          const t = firstDefined(event, ["time", "time_s", "timestamp", "timestamp_s", "center", "center_s"]);
          const label = safeText(firstDefined(event, ["label", "type", "title", "name"]), "state change");
          const reason = safeText(firstDefined(event, ["reason", "description", "explanation"]), "");
          const score = asNumber(firstDefined(event, ["score", "risk_score"]));
          return (
            <div className="event" key={idx}>
              <div className="event-time">{asNumber(t) !== undefined ? formatSeconds(t) : "-"}</div>
              <div className="event-body">
                <div className="event-title">{label}</div>
                {reason && <p>{reason}</p>}
                {score !== undefined && <div className="event-score">score {score.toFixed(2)}</div>}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}


function DomainInsightCard({ result }: { result: unknown }) {
  const insight = getDomainInsight(result);
  if (!insight) return null;

  const observations = getStringArray(insight, ["observations"]);
  const interpretation = getStringArray(insight, ["interpretation"]);
  const microbase = getStringArray(insight, ["microbase_angle", "microbaseAngle"]);
  const nextSteps = getStringArray(insight, ["recommended_next_steps", "recommendedNextSteps"]);
  const caveat = safeText(firstDefined(insight, ["caveat"]), "");

  return (
    <section className="card domain-card">
      <div className="section-header">
        <h2>{safeText(firstDefined(insight, ["title"]), "Geo / Microbase interpretation")}</h2>
        <span className="domain-pill">{safeText(firstDefined(insight, ["domain"]), "geo")}</span>
      </div>

      <div className="domain-grid">
        <div>
          <h3>Video-specific observations</h3>
          <ul className="plain-list">
            {observations.map((item, idx) => <li key={idx}>{item}</li>)}
          </ul>
        </div>

        <div>
          <h3>Urban interpretation</h3>
          <ul className="plain-list">
            {interpretation.map((item, idx) => <li key={idx}>{item}</li>)}
          </ul>
        </div>

        <div>
          <h3>Microbase angle</h3>
          <ul className="plain-list">
            {microbase.map((item, idx) => <li key={idx}>{item}</li>)}
          </ul>
        </div>

        <div>
          <h3>Next implementation steps</h3>
          <ul className="plain-list">
            {nextSteps.map((item, idx) => <li key={idx}>{item}</li>)}
          </ul>
        </div>
      </div>

      {caveat && <p className="caveat"><strong>Caveat:</strong> {caveat}</p>}
    </section>
  );
}

function ExplanationCard({ result, health }: { result: unknown; health: unknown }) {
  if (!result) return null;
  const backend = getBackendName(result, health);
  const explanation = getExplanation(result, backend);
  return (
    <section className="card explanation-card">
      <h2>Explanation</h2>
      {explanation ? <p className="explanation-text">{explanation}</p> : <p className="muted">No explanation returned.</p>}
      <KV label="Backend used">
        <BackendBlock value={backend} />
      </KV>
    </section>
  );
}

function Diagnostics({ result }: { result: unknown }) {
  const diagnostics = firstDefined(result, ["diagnostics"]);
  if (!diagnostics) return null;
  return (
    <section className="card diagnostics-card">
      <details>
        <summary>Diagnostics</summary>
        <pre className="diagnostics">{safeText(diagnostics, "{}")}</pre>
      </details>
    </section>
  );
}


class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { error: string }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: "" };
  }

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error.message : String(error) };
  }

  componentDidCatch(error: unknown) {
    console.error("UI render error:", error);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="app">
          <section className="card error-card">
            <h2>UI render error</h2>
            <pre className="error-box">{this.state.error}</pre>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}

function App() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [health, setHealth] = useState<unknown>(null);
  const [file, setFile] = useState<File | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [domainMode, setDomainMode] = useState("auto");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    fetch(API_HEALTH)
      .then((r) => r.json())
      .then(setHealth)
      .catch((e) => setHealth({ error: String(e) }));
  }, []);

  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [previewStatus, setPreviewStatus] = useState<string>("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      setPreviewStatus("");
      return;
    }

    const controller = new AbortController();
    let localUrl = URL.createObjectURL(file);
    let convertedUrl = "";

    // Show something immediately, then replace it with a browser-compatible
    // backend-transcoded preview when available. This avoids blank previews for
    // OpenCV/mp4v MP4 files that Chrome/Safari often cannot decode.
    setPreviewUrl(localUrl);
    setPreviewStatus("Preparing browser-compatible preview...");

    const form = new FormData();
    form.append("video", file, file.name);

    fetch(API_PREVIEW, {
      method: "POST",
      body: form,
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || `Preview conversion failed: HTTP ${response.status}`);
        }
        return response.blob();
      })
      .then((blob) => {
        convertedUrl = URL.createObjectURL(blob);
        setPreviewUrl(convertedUrl);
        setPreviewStatus("Preview converted for browser playback.");
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        console.warn("Preview conversion failed; using original file URL.", e);
        setPreviewStatus("Using original file preview. If it does not play, the video codec is not browser-compatible.");
      });

    return () => {
      controller.abort();
      URL.revokeObjectURL(localUrl);
      if (convertedUrl) URL.revokeObjectURL(convertedUrl);
    };
  }, [file]);

  const currentState = useMemo(
    () => getStringArray(result, ["current_state", "state.current", "summary.current_state"]),
    [result],
  );

  const prediction = useMemo(
    () => getStringArray(result, ["predicted_near_future_change", "prediction", "predictions", "near_future", "summary.prediction"]),
    [result],
  );

  async function analyze() {
    if (!file) {
      setError("Please choose a video file first.");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("video", file, file.name);
    form.append("demo_mode", String(demoMode));
    form.append("domain_mode", domainMode);

    try {
      const response = await fetch(API_ANALYZE, { method: "POST", body: form });
      const text = await response.text();
      let payload: unknown;
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        payload = { error: text };
      }

      if (!response.ok) {
        setError(safeText(payload, `HTTP ${response.status}`));
      } else {
        setResult(payload);
        // Refresh health after lazy model loading.
        fetch(API_HEALTH).then((r) => r.json()).then(setHealth).catch(() => {});
      }
    } catch (e) {
      setError(`Failed to fetch: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <header className="hero">
        <div className="eyebrow">V-JEPA 2.1 / representation-change MVP</div>
        <h1>V-JEPA Video Risk Lens</h1>
        <p>
          Upload a short video. The backend samples clips, extracts V-JEPA embeddings when a local model is mounted,
          detects state-transition windows, and turns them into an operational risk explanation.
        </p>
      </header>

      <StatusCard health={health} result={result} />

      <section className="card input-card">
        <h2>Input video</h2>
        <input
          ref={inputRef}
          className="hidden-input"
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button className="choose-button" onClick={() => inputRef.current?.click()} type="button">
          {file ? file.name : "Choose a smartphone, dashcam, warehouse, robot-view, or hand-work video"}
        </button>

        {previewUrl && (
          <div className="video-preview">
            <div className="preview-header">
              <span>Preview</span>
              <span className="muted">{file?.name}</span>
            </div>
            <video
              className="preview-video"
              src={previewUrl}
              controls
              muted
              playsInline
              preload="metadata"
              onCanPlay={() => {
                if (previewStatus.startsWith("Preparing")) {
                  setPreviewStatus("Preview ready.");
                }
              }}
              onError={() => setPreviewStatus("Preview playback failed. The original codec may be unsupported; backend conversion may still be running or failed.")}
            />
            {previewStatus && <div className="preview-status">{previewStatus}</div>}
          </div>
        )}

        <label className="checkbox-row">
          <input type="checkbox" checked={demoMode} onChange={(e) => setDemoMode(e.target.checked)} />
          <span>Use explicit classical demo mode if no V-JEPA model is mounted</span>
        </label>

        <label className="field-row">
          <span>Explanation mode</span>
          <select value={domainMode} onChange={(e) => setDomainMode(e.target.value)}>
            <option value="auto">Auto detect</option>
            <option value="geo_urban">Geo / urban value map</option>
            <option value="microbase">Microbase-style area intelligence</option>
            <option value="off">Physical-risk only</option>
          </select>
        </label>

        <button className="analyze-button" onClick={analyze} disabled={busy || !file} type="button">
          {busy ? "Analyzing..." : "Analyze"}
        </button>
        {error && <pre className="error-box">{error}</pre>}
      </section>

      <RiskCard result={result} health={health} />
      <Timeline result={result} />
      <DomainInsightCard result={result} />
      <TextListCard title="Current state" items={currentState} />
      <TextListCard title="Predicted near-future change" items={prediction} />
      <EventsCard result={result} />
      <ExplanationCard result={result} health={health} />
      <Diagnostics result={result} />
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<ErrorBoundary><App /></ErrorBoundary>);
