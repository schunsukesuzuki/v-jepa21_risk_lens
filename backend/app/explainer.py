from __future__ import annotations

import httpx

from .schemas import DetectedEvent


# Builds the final human-readable explanation for the video-risk analysis result.
#
# This is the main entry point of this module. It first creates a deterministic,
# rule-based explanation from the computed analysis values. That base explanation
# is always available and does not depend on an LLM.
#
# If llm_provider is set to "ollama", the function optionally asks the local LLM
# rewrite function to make the explanation more natural. If that rewrite fails,
# the function still returns the deterministic explanation and appends a short
# failure reason instead of breaking the whole analysis flow.
#
# Parameters:
# - risk_level: A categorical risk label such as "Low", "Medium", or "High".
# - risk_score: The numeric risk score used to support the risk label.
# - backend: The name of the analysis backend that produced the result.
# - model_loaded: Whether the V-JEPA-style model was actually loaded.
# - events: Detected state-change or motion-peak events.
# - center_bias: Ratio indicating how much motion is concentrated near the center.
# - direction: The dominant motion direction inferred by the analyzer.
# - llm_provider: Selects whether an LLM rewrite should be attempted.
# - ollama_base_url: Base URL used by the optional rewrite function.
# - ollama_model: Model name used by the optional rewrite function.
#
# Returns:
# A final explanation string. The function is designed to return a usable
# explanation even when the optional LLM rewrite path is unavailable.
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


# Creates a deterministic explanation from the already-computed analysis result.
#
# This function does not call any LLM. It formats the analysis outputs into a
# stable, template-based explanation. This makes the output predictable and keeps
# the explanation grounded in the actual computed values.
#
# The function summarizes detected events by timestamp and title. If no event is
# present, it explicitly says that no sharp event peak was found. It also chooses
# the representation description based on whether the model was actually loaded:
# either V-JEPA clip representations or classical visual features in demo mode.
#
# The generated text explains:
# - the risk level and numeric score,
# - whether the analyzer compared learned clip representations or classical features,
# - that representation deltas were combined with frame-level motion energy,
# - the strongest detected event windows,
# - the dominant motion direction,
# - the central-motion ratio,
# - the backend used for the analysis,
# - and the MVP interpretation that sharp representation changes are treated as
#   state-transition candidates rather than as a generated prediction of the full
#   future video.
#
# Returns:
# A deterministic, human-readable explanation string.
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


# Rewrites the deterministic base explanation into a more concise operational note.
#
# This function is intentionally used as a rewrite step, not as the source of the
# analysis itself. The input `base` already contains the grounded explanation. The
# prompt asks the model to preserve uncertainty and avoid inventing object
# identities, which helps reduce hallucinated details in the final explanation.
#
# The function sends a non-streaming generation request, checks the HTTP status,
# reads the JSON response, and extracts the `response` field. If the response text
# is empty, it falls back to the original deterministic explanation.
#
# Parameters:
# - base: The deterministic explanation to rewrite.
# - base_url: Base URL for the generation endpoint.
# - model: Model name passed to the generation request.
#
# Returns:
# The rewritten explanation if available; otherwise, the original base text.
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
