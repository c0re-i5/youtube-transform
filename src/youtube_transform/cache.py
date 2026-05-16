"""Filesystem cache helpers. Everything keyed by video_id so re-runs are cheap."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cache:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for sub in ("audio", "transcripts", "clips", "meta"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # --- paths ---
    def audio_path(self, video_id: str) -> Path:
        return self.root / "audio" / f"{video_id}.m4a"

    def transcript_path(self, video_id: str, model: str) -> Path:
        return self.root / "transcripts" / f"{video_id}.{model}.json"

    def clip_path(self, video_id: str, start: float, end: float, fmt_tag: str) -> Path:
        key = f"{video_id}_{start:.3f}_{end:.3f}_{fmt_tag}"
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        return self.root / "clips" / f"{video_id}_{h}.mp4"

    def meta_path(self, video_id: str) -> Path:
        return self.root / "meta" / f"{video_id}.json"

    # --- json helpers ---
    @staticmethod
    def read_json(p: Path) -> dict | None:
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    @staticmethod
    def write_json(p: Path, data: dict) -> None:
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
