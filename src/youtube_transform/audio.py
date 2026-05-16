"""Audio-only download (small) for transcription."""

from __future__ import annotations

from pathlib import Path

from yt_dlp import YoutubeDL

from .cache import Cache


def fetch_audio(url: str, video_id: str, cache: Cache) -> Path:
    """Download bestaudio as m4a if not already cached. Returns path."""
    out = cache.audio_path(video_id)
    if out.exists() and out.stat().st_size > 0:
        return out

    # yt-dlp will pick the extension; we ask for m4a/aac when available.
    tmpl = str(cache.root / "audio" / f"{video_id}.%(ext)s")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": tmpl,
        "noprogress": True,
        # Be polite.
        "retries": 3,
        "fragment_retries": 3,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])

    # Find whatever extension we got and normalize to .m4a path symlink-or-rename.
    candidates = sorted((cache.root / "audio").glob(f"{video_id}.*"))
    if not candidates:
        raise RuntimeError(f"audio download failed for {video_id}")
    actual = candidates[0]
    if actual.suffix != ".m4a":
        # leave as-is; whisper/ffmpeg can read most containers. Just return actual.
        return actual
    return actual
