from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    device: str = "cpu"

    # Docker compose mounts ./models on the host to /models in the container.
    # The app now searches both /models and /models/vjepa2 recursively.
    model_dir: str = "/models"
    model_mode: str = "auto"  # auto | vjepa_checkpoint | torchscript | torch_module | hub | off
    model_name: str = "vjepa2_1_vit_base_384"
    vjepa_torchscript_path: str = ""
    vjepa_checkpoint_path: str = ""
    vjepa_repo: str = "facebookresearch/vjepa2"
    vjepa_repo_dir: str = ""  # optional local clone containing hubconf.py
    vjepa_checkpoint_key: str = "ema_encoder"
    allow_online_hub_download: bool = True
    allow_demo_fallback: bool = False
    load_model_on_startup: bool = False

    clip_size: int = 8
    sample_fps: float = 4.0
    frame_size: int = 384
    max_frames: int = 64

    data_dir: str = "/data"
    llm_provider: str = "none"  # none | ollama
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.clip_size < 2:
        raise ValueError("CLIP_SIZE must be >= 2")
    if settings.max_frames < settings.clip_size:
        raise ValueError("MAX_FRAMES must be >= CLIP_SIZE")
    return settings
