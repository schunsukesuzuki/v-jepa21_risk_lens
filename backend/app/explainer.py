from __future__ import annotations

import httpx

from .schemas import DetectedEvent


def build_explanation(
    *,
    risk_level: str,
    risk_score: float,
    backend: str,
    model_loaded: bool,
    events: list[DetectedEvent],
    center_bias: float,
    direction: str,
    llm_provider: str,
    ollama_base_url: str,
    ollama_model: str,
) -> str:
    base = deterministic_explanation(
        risk_level=risk_level,
        risk_score=risk_score,
        backend=backend,
        model_loaded=model_loaded,
        events=events,
        center_bias=center_bias,
        direction=direction,
    )
    if llm_provider.lower() != "ollama":
        return base
    try:
        return ollama_rewrite(base, ollama_base_url, ollama_model)
    except Exception as exc:  # noqa: BLE001
        return f"{base}\n\nLLM rewrite skipped because Ollama was unavailable: {type(exc).__name__}: {exc}"


def deterministic_explanation(
    *,
    risk_level: str,
    risk_score: float,
    backend: str,
    model_loaded: bool,
    events: list[DetectedEvent],
    center_bias: float,
    direction: str,
) -> str:
    event_text = "; ".join([f"{e.timestamp:.2f}s: {e.title}" for e in events]) or "no sharp event peak"
    representation = (
        "V-JEPA clip representations"
        if model_loaded
        else "classical visual features because explicit demo mode was enabled"
    )
    return (
        f"Risk is {risk_level} (score {risk_score:.2f}). The analyzer compared consecutive {representation} "
        f"and combined the representation delta with frame-level motion energy. The strongest windows were: {event_text}. "
        f"Dominant motion is {direction}, and the central-motion ratio is {center_bias:.2f}. "
        f"Backend: {backend}. This MVP treats sharp representation changes as state-transition candidates, "
        "then maps those candidates to an operational risk explanation rather than hallucinating a full future video."
    )


def ollama_rewrite(base: str, base_url: str, model: str) -> str:
    prompt = (
        "Rewrite the following machine-analysis note as a concise operational video-risk explanation. "
        "Do not invent object identities. Keep the uncertainty explicit.\n\n"
        f"{base}"
    )
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
    text = str(data.get("response", "")).strip()
    return text or base
