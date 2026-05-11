import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile


class Storage:
    def __init__(self, data_dir: str):
        self.root = Path(data_dir)
        self.uploads = self.root / "uploads"
        self.results = self.root / "results"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, upload: UploadFile) -> tuple[str, Path]:
        suffix = Path(upload.filename or "video.mp4").suffix.lower() or ".mp4"
        analysis_id = uuid.uuid4().hex[:12]
        safe_name = f"{analysis_id}{suffix}"
        path = self.uploads / safe_name
        with path.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return analysis_id, path

    def save_result(self, analysis_id: str, payload: dict[str, Any]) -> Path:
        path = self.results / f"{analysis_id}.json"
        tmp = self.results / f"{analysis_id}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return path
