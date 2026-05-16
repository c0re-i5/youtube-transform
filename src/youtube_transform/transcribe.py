"""Whisper transcription with word-level timestamps via WhisperX forced alignment.

Output schema (cached as JSON):
{
  "video_id": "...",
  "model": "small",
  "language": "en",
  "words": [
    {"w": "hello", "start": 1.23, "end": 1.41, "score": 0.98},
    ...
  ]
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache import Cache


def transcribe(
    audio_path: Path,
    video_id: str,
    cache: Cache,
    *,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Returns the transcript dict, using cache if present (unless `force`)."""
    cached_path = cache.transcript_path(video_id, model_name)
    if not force:
        cached = cache.read_json(cached_path)
        if cached is not None:
            return cached

    # Lazy import so that `--help` and source-only commands don't pay the torch import cost.
    import torch  # type: ignore
    import whisperx  # type: ignore

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    asr = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    audio = whisperx.load_audio(str(audio_path))
    asr_result = asr.transcribe(audio, batch_size=16)
    detected_lang = asr_result.get("language", language or "en")

    align_model, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
    aligned = whisperx.align(
        asr_result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    words: list[dict[str, Any]] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            # whisperx may omit timestamps for some tokens (numbers, punct). Skip those.
            if "start" not in w or "end" not in w:
                continue
            words.append(
                {
                    "w": w.get("word", "").strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "score": float(w.get("score", 0.0)),
                }
            )

    out = {
        "video_id": video_id,
        "model": model_name,
        "language": detected_lang,
        "words": words,
    }
    cache.write_json(cached_path, out)
    return out


def transcript_stats(transcript: dict, audio_path: Path | None = None) -> dict[str, Any]:
    """Compute coverage stats useful for diagnosing 'why no hits'.

    Returns:
      {
        "n_words": int,
        "first_ts": float,
        "last_ts": float,
        "covered_seconds": float,   # last - first
        "audio_seconds": float | None,
        "coverage_ratio": float | None,   # covered / audio
        "median_score": float,
      }
    """
    import statistics
    import subprocess

    words = transcript.get("words", [])
    n = len(words)
    first = float(words[0]["start"]) if n else 0.0
    last = float(words[-1]["end"]) if n else 0.0
    audio_seconds: float | None = None
    if audio_path is not None and audio_path.exists():
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            try:
                audio_seconds = float(proc.stdout.strip())
            except ValueError:
                audio_seconds = None
    covered = last - first
    ratio = (covered / audio_seconds) if (audio_seconds and audio_seconds > 0) else None
    scores = [w.get("score", 0.0) for w in words if w.get("score")]
    return {
        "n_words": n,
        "first_ts": first,
        "last_ts": last,
        "covered_seconds": covered,
        "audio_seconds": audio_seconds,
        "coverage_ratio": ratio,
        "median_score": statistics.median(scores) if scores else 0.0,
    }
