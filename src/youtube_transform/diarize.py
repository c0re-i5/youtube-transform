"""Speaker diarization via WhisperX's pyannote pipeline.

Produces a list of speaker turns: [{"start": float, "end": float, "speaker": "SPK_00"}, ...].
Caches per video_id. Requires HUGGING_FACE_HUB_TOKEN env var (pyannote model is gated).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .cache import Cache


def diarize(
    audio_path: Path,
    video_id: str,
    cache: Cache,
    *,
    device: str = "auto",
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    """Returns list of speaker turns. Cached on disk."""
    cache_path = cache.root / "diarization" / f"{video_id}.json"
    cache_path.parent.mkdir(exist_ok=True)
    cached = cache.read_json(cache_path)
    if cached is not None:
        return cached.get("turns", [])

    token = hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Diarization requires a Hugging Face token (set HUGGING_FACE_HUB_TOKEN). "
            "You also need to accept the pyannote/speaker-diarization-3.1 license on HF."
        )

    import torch  # type: ignore
    import whisperx  # type: ignore

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = whisperx.DiarizationPipeline(use_auth_token=token, device=device)
    df = pipe(str(audio_path))
    # df is a pandas DataFrame with columns: start, end, speaker
    turns = [
        {"start": float(r.start), "end": float(r.end), "speaker": str(r.speaker)}
        for r in df.itertuples()
    ]
    cache.write_json(cache_path, {"video_id": video_id, "turns": turns})
    return turns


def assign_speakers(transcript: dict, turns: list[dict]) -> dict:
    """Annotate each transcript word with the speaker label of its covering turn.

    Words outside any turn (or in a tiny gap) get speaker = None.
    """
    if not turns:
        return transcript
    # Sort turns once, then linear-scan with a pointer per word (words are ordered too).
    turns_sorted = sorted(turns, key=lambda t: t["start"])
    new_words = []
    ti = 0
    for w in transcript["words"]:
        mid = (w["start"] + w["end"]) / 2
        # advance pointer while turn ends before mid
        while ti < len(turns_sorted) and turns_sorted[ti]["end"] < mid:
            ti += 1
        spk = None
        if ti < len(turns_sorted) and turns_sorted[ti]["start"] <= mid <= turns_sorted[ti]["end"]:
            spk = turns_sorted[ti]["speaker"]
        new_words.append({**w, "speaker": spk})
    return {**transcript, "words": new_words}


def list_speakers(transcript: dict) -> dict[str, int]:
    """Return {speaker: word_count} from a speaker-annotated transcript."""
    counts: dict[str, int] = {}
    for w in transcript.get("words", []):
        spk = w.get("speaker")
        if spk:
            counts[spk] = counts.get(spk, 0) + 1
    return counts
