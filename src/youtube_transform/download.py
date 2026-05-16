"""Download just the bytes for a single time window using yt-dlp's --download-sections.

Under the hood yt-dlp uses HTTP Range requests against the CDN, so bandwidth is
proportional to clip length, not video length.
"""

from __future__ import annotations

from pathlib import Path

from yt_dlp import YoutubeDL

from .cache import Cache
from .search import Hit


def fetch_clip(
    url: str,
    hit: Hit,
    cache: Cache,
    *,
    max_height: int = 720,
    fmt_tag: str | None = None,
) -> Path:
    """Download a single time window. Returns the on-disk path (cached if present)."""
    tag = fmt_tag or f"h{max_height}"
    out = cache.clip_path(hit.video_id, hit.start, hit.end, tag)
    if out.exists() and out.stat().st_size > 0:
        return out

    # Use a template that yt-dlp will fill; we rename to `out` after.
    tmp_tmpl = str(cache.root / "clips" / f"{out.stem}.%(ext)s")

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "merge_output_format": "mp4",
        "outtmpl": tmp_tmpl,
        # The magic: only download the byte ranges for this section.
        "download_ranges": _make_range(hit.start, hit.end),
        # Re-mux/encode just the boundary GOPs so cuts are clean.
        "force_keyframes_at_cuts": True,
        "retries": 3,
        "fragment_retries": 3,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])

    # find produced file
    candidates = sorted((cache.root / "clips").glob(f"{out.stem}.*"))
    candidates = [c for c in candidates if c.suffix in {".mp4", ".mkv", ".webm"}]
    if not candidates:
        raise RuntimeError(
            f"clip download failed for {hit.video_id} @ {hit.start:.2f}-{hit.end:.2f}"
        )
    produced = candidates[0]
    if produced != out:
        produced.rename(out)
    return out


def _make_range(start: float, end: float):
    """Build the callable yt-dlp expects for download_ranges."""

    def _ranges(_info_dict, _ydl):
        return [{"start_time": float(start), "end_time": float(end)}]

    return _ranges
