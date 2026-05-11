from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Settings


class ModelUnavailable(RuntimeError):
    pass


class EncoderOnly(torch.nn.Module):
    """Wrap official hub return values so downstream code sees a normal Module."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> Any:  # noqa: D401
        return self.encoder(x)


class VJEPAModelAdapter:
    """Offline-first V-JEPA/V-JEPA 2.1 adapter.

    Supported layouts on the host side:

        ./models/vjepa2_1_vitb_dist_vitG_384.pt
        ./models/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
        ./models/vjepa2_repo/hubconf.py + checkpoint file

    The important fix is that we no longer assume ./models/vjepa2 only. The
    Docker mount is ./models:/models, so a file directly under ./models is valid.

    For official V-JEPA 2.1 checkpoints, this loader creates the architecture via
    PyTorch Hub entry `vjepa2_1_vit_base_384(pretrained=False)`, then loads the
    local checkpoint key, usually `ema_encoder`, into the encoder. It does not use
    random initialization as a fallback.
    """

    KNOWN_VJEPA_CHECKPOINT_NAMES = (
        "vjepa2_1_vitb_dist_vitG_384.pt",
        "vjepa2_1_vitl_dist_vitG_384.pt",
        "vjepa2_1_vitg_384.pt",
        "vjepa2_1_vitG_384.pt",
        "vitg-384.pt",
        "vitg.pt",
        "vith.pt",
        "vitl.pt",
    )

    def __init__(self, settings: Settings, *, autoload: bool = False):
        self.settings = settings
        requested = settings.device.strip().lower()
        self.device = torch.device(requested if requested == "cpu" or torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.backend = "not_loaded_yet"
        self.errors: list[str] = []
        self.discovered_files: list[str] = []
        self.load_attempted = False
        # Always discover files so /api/health is useful, but do not load the
        # large model at process startup. Eager loading caused Vite proxy 502s
        # while torch.hub / checkpoint loading was still running or failed.
        self.refresh_discovery()
        if autoload:
            self.ensure_loaded()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def refresh_discovery(self) -> list[str]:
        self._discover_model_files()
        return self.discovered_files

    def ensure_loaded(self) -> bool:
        if self.model is not None:
            return True
        if self.load_attempted:
            return False
        self.load_attempted = True
        self._load_once()
        return self.model is not None

    def _load_once(self) -> None:
        if self.settings.model_mode == "off":
            self.backend = "off"
            return

        # Reset diagnostics for this load attempt, but keep fresh discovery.
        self.errors = []
        self.refresh_discovery()

        # Keep torch hub cache inside the mounted models directory so it persists.
        os.environ.setdefault("TORCH_HOME", "/models/.torchhub")

        attempts: list[tuple[str, Path | str]] = []

        explicit_ts = self.settings.vjepa_torchscript_path.strip()
        if explicit_ts:
            attempts.append(("torchscript", Path(explicit_ts)))

        explicit_ckpt = self.settings.vjepa_checkpoint_path.strip()
        if explicit_ckpt:
            attempts.append(("vjepa_checkpoint", Path(explicit_ckpt)))

        for p in self._discover_model_files():
            # Try official checkpoint loading first for known V-JEPA checkpoint names.
            if p.name in self.KNOWN_VJEPA_CHECKPOINT_NAMES or p.name.startswith("vjepa2_1_"):
                attempts.append(("vjepa_checkpoint", p))
            attempts.append(("file", p))

        repo_dir = self._discover_repo_dir()
        if repo_dir is not None:
            attempts.append(("hub_local", repo_dir))
        elif self.settings.allow_online_hub_download:
            attempts.append(("hub_online", self.settings.vjepa_repo))

        seen: set[str] = set()
        unique_attempts = []
        for kind, path in attempts:
            key = f"{kind}:{path}"
            if key not in seen:
                seen.add(key)
                unique_attempts.append((kind, path))

        if not unique_attempts:
            self.backend = "unavailable"
            self.errors.append(
                "No model candidates found. Expected e.g. /models/vjepa2_1_vitb_dist_vitG_384.pt "
                "or /models/vjepa2_repo/hubconf.py."
            )
            return

        for kind, path in unique_attempts:
            try:
                if kind == "vjepa_checkpoint":
                    self.model = self._load_official_vjepa_checkpoint(Path(path))
                    self.backend = f"vjepa_checkpoint:{Path(path).name}:{self.settings.model_name}"
                elif kind in {"torchscript", "file"}:
                    self.model = self._load_file(Path(path))
                    self.backend = f"{kind}:{Path(path).name}"
                elif kind == "hub_local":
                    self.model = self._load_hub_model(repo=str(path), source="local", pretrained=True)
                    self.backend = f"hub_local:{self.settings.model_name}"
                elif kind == "hub_online":
                    self.model = self._load_hub_model(repo=str(path), source="github", pretrained=True)
                    self.backend = f"hub_online:{self.settings.model_name}"
                if self.model is not None:
                    self.model.to(self.device)
                    self.model.eval()
                    return
            except Exception as exc:  # noqa: BLE001 - aggregate diagnostics for UI
                self.errors.append(f"{kind} {path}: {type(exc).__name__}: {exc}")
        self.backend = "unavailable"

    def _discover_model_files(self) -> list[Path]:
        roots: list[Path] = []
        model_dir = Path(self.settings.model_dir)
        roots.append(model_dir)
        roots.append(Path("/models"))
        roots.append(Path("/models/vjepa2"))
        roots.append(Path("/models/checkpoints"))

        files: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            for pattern in ("*.torchscript", "*.jit", "*.pt", "*.pth", "**/*.torchscript", "**/*.jit", "**/*.pt", "**/*.pth"):
                files.extend(root.glob(pattern))
        unique = sorted({p.resolve() for p in files if p.is_file()})
        self.discovered_files = [str(p) for p in unique]
        return unique

    def _discover_repo_dir(self) -> Path | None:
        candidates: list[Path] = []
        if self.settings.vjepa_repo_dir.strip():
            candidates.append(Path(self.settings.vjepa_repo_dir.strip()))
        candidates.extend([Path("/models/vjepa2_repo"), Path("/models/vjepa2"), Path("/models")])
        for c in candidates:
            if c.joinpath("hubconf.py").exists():
                return c
        return None

    def _load_hub_model(self, *, repo: str, source: str, pretrained: bool) -> torch.nn.Module:
        obj = torch.hub.load(repo, self.settings.model_name, source=source, pretrained=pretrained, trust_repo=True)
        return self._normalize_hub_object(obj)

    def _normalize_hub_object(self, obj: Any) -> torch.nn.Module:
        if isinstance(obj, torch.nn.Module):
            return obj
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], torch.nn.Module):
            return EncoderOnly(obj[0])
        raise ModelUnavailable(f"PyTorch Hub returned unsupported object: {type(obj)}")

    def _load_official_vjepa_checkpoint(self, path: Path) -> torch.nn.Module:
        if not path.exists():
            raise ModelUnavailable(f"Checkpoint path does not exist: {path}")

        repo_dir = self._discover_repo_dir()
        if repo_dir is not None:
            obj = torch.hub.load(str(repo_dir), self.settings.model_name, source="local", pretrained=False, trust_repo=True)
        elif self.settings.allow_online_hub_download:
            obj = torch.hub.load(self.settings.vjepa_repo, self.settings.model_name, source="github", pretrained=False, trust_repo=True)
        else:
            raise ModelUnavailable(
                "Official V-JEPA checkpoint requires the V-JEPA2 source code. "
                "Either put a local clone with hubconf.py under ./models/vjepa2_repo, "
                "or set ALLOW_ONLINE_HUB_DOWNLOAD=true."
            )

        encoder: torch.nn.Module
        if isinstance(obj, torch.nn.Module):
            encoder = obj
        elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], torch.nn.Module):
            encoder = obj[0]
        else:
            raise ModelUnavailable(f"Hub model returned unsupported object: {type(obj)}")

        ckpt = torch.load(str(path), map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ModelUnavailable(f"Checkpoint is not a dict: {type(ckpt)}")

        state = None
        for key in (self.settings.vjepa_checkpoint_key, "ema_encoder", "target_encoder", "encoder", "backbone", "model", "state_dict"):
            candidate = ckpt.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
        if state is None:
            # Some checkpoints are themselves a raw state_dict.
            tensor_values = [v for v in ckpt.values() if torch.is_tensor(v)]
            if tensor_values:
                state = ckpt
            else:
                raise ModelUnavailable(f"Could not find encoder state_dict. Available keys: {list(ckpt.keys())[:30]}")

        cleaned = clean_backbone_key(state)
        missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
        if len(cleaned) == 0:
            raise ModelUnavailable("Loaded checkpoint state_dict is empty.")
        self.errors.append(
            "checkpoint_load_info: "
            f"loaded_keys={len(cleaned)}, missing={len(missing)}, unexpected={len(unexpected)}, "
            f"first_missing={list(missing)[:5]}, first_unexpected={list(unexpected)[:5]}"
        )
        return EncoderOnly(encoder)

    def _load_file(self, path: Path) -> torch.nn.Module:
        try:
            model = torch.jit.load(str(path), map_location=self.device)
            return model  # type: ignore[return-value]
        except Exception as jit_exc:
            obj: Any = torch.load(str(path), map_location=self.device)
            if isinstance(obj, torch.nn.Module):
                return obj
            if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], torch.nn.Module):
                return obj["model"]
            raise ModelUnavailable(
                f"{path.name} is neither TorchScript nor serialized nn.Module. "
                "If this is an official V-JEPA checkpoint, use MODEL_MODE=auto or MODEL_MODE=vjepa_checkpoint. "
                "First TorchScript error: "
                f"{jit_exc}"
            )

    def encode_clips(self, frames_rgb: np.ndarray, clip_size: int) -> tuple[np.ndarray, list[str]]:
        if self.model is None:
            raise ModelUnavailable("V-JEPA model is not loaded. Mount model files under ./models or enable explicit demo mode.")

        clips = make_clips(frames_rgb, clip_size)
        outputs: list[np.ndarray] = []
        warnings: list[str] = []
        with torch.no_grad():
            for clip in clips:
                tensor_cthw = frames_to_tensor(clip).to(self.device)  # [1,C,T,H,W]
                try:
                    out = self.model(tensor_cthw)
                except Exception as first_exc:
                    tensor_tchw = tensor_cthw.permute(0, 2, 1, 3, 4).contiguous()  # [1,T,C,H,W]
                    try:
                        out = self.model(tensor_tchw)
                        warnings.append("Model accepted [B,T,C,H,W]; first [B,C,T,H,W] call failed.")
                    except Exception as second_exc:
                        raise ModelUnavailable(
                            "Model forward failed for both [B,C,T,H,W] and [B,T,C,H,W]. "
                            f"First: {first_exc}; second: {second_exc}"
                        ) from second_exc
                feat = flatten_output(out)
                outputs.append(feat)
        return np.stack(outputs).astype(np.float32), warnings


def clean_backbone_key(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, val in state_dict.items():
        if not torch.is_tensor(val):
            continue
        k = str(key)
        for prefix in ("module.", "backbone.", "encoder.", "target_encoder.", "ema_encoder."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        cleaned[k] = val
    return cleaned


def make_clips(frames_rgb: np.ndarray, clip_size: int) -> list[np.ndarray]:
    n = len(frames_rgb)
    if n <= clip_size:
        return [frames_rgb]
    stride = max(1, clip_size // 2)
    clips = [frames_rgb[i : i + clip_size] for i in range(0, n - clip_size + 1, stride)]
    if (n - clip_size) % stride != 0:
        clips.append(frames_rgb[-clip_size:])
    return clips


def frames_to_tensor(frames_rgb: np.ndarray) -> torch.Tensor:
    arr = frames_rgb.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return tensor


def flatten_output(out: Any) -> np.ndarray:
    if isinstance(out, dict):
        for key in ("x", "embeddings", "features", "last_hidden_state"):
            if key in out:
                out = out[key]
                break
        else:
            out = next(iter(out.values()))
    if isinstance(out, (list, tuple)):
        out = out[0]
    if not torch.is_tensor(out):
        raise ModelUnavailable(f"Model output is not tensor-like: {type(out)}")
    feat = out.detach().float().cpu()
    if feat.ndim >= 3:
        feat = feat.mean(dim=tuple(range(1, feat.ndim - 1))) if feat.shape[-1] >= feat.shape[1] else feat.flatten(1)
    feat = feat.reshape(feat.shape[0], -1).mean(dim=0)
    feat = torch.nn.functional.normalize(feat, dim=0)
    return feat.numpy()
