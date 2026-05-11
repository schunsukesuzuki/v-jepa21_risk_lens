export type RiskLevel = 'Low' | 'Medium' | 'High';

export interface TimelineSegment {
  start: number;
  end: number;
  label: string;
  score: number;
}

export interface DetectedEvent {
  timestamp: number;
  title: string;
  detail: string;
  score: number;
}

export interface AnalysisResult {
  analysis_id: string;
  video_filename: string;
  duration_sec: number;
  sampled_frames: number;
  model_backend: string;
  model_loaded: boolean;
  risk_level: RiskLevel;
  risk_score: number;
  current_state: string[];
  predicted_near_future_change: string[];
  reason: string;
  timeline: TimelineSegment[];
  detected_state_change: DetectedEvent[];
  diagnostics: Record<string, unknown>;
}

export interface HealthResponse {
  ok: boolean;
  model_backend: string;
  model_loaded: boolean;
  device: string;
  model_dir: string;
  allow_demo_fallback: boolean;
  errors: string[];
  discovered_model_files: string[];
  load_attempted: boolean;
}

// Default to same-origin /api. In Docker this is served by Vite and proxied to http://backend:8000.
// This avoids browser-side `localhost` / CORS / container-network mismatch failures.
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = 120_000): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (e) {
    const raw = e instanceof Error ? e.message : String(e);
    throw new Error(
      `Network request failed: ${raw}. ` +
      `Open ${apiUrl('/api/health')} in your browser, or run: curl http://localhost:5173/api/health`
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetchWithTimeout(apiUrl('/api/health'), {}, 15_000);
  if (!res.ok) throw new Error(`Health check failed: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function analyzeVideo(file: File, demoMode: boolean): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('video', file);
  form.append('demo_mode', String(demoMode));
  const res = await fetchWithTimeout(apiUrl('/api/analyze'), { method: 'POST', body: form }, 180_000);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const payload = await res.json();
      detail = payload.detail ?? JSON.stringify(payload);
    } catch {
      try {
        detail = await res.text();
      } catch {
        // ignore
      }
    }
    throw new Error(detail);
  }
  return res.json();
}
