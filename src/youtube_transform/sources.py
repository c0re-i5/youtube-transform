"""Resolve a user input (single video URL, playlist, channel) to a flat list of video URLs.

Uses yt-dlp's `extract_flat` which only fetches the index, no media.
"""

from __future__ import annotations

from dataclasses import dataclass

from yt_dlp import YoutubeDL


@dataclass
class VideoRef:
    video_id: str
    url: str
    title: str | None = None


def resolve(inputs: list[str], limit: int | None = None) -> list[VideoRef]:
    """Expand each input into one or more VideoRef. Cheap — no media download."""
    refs: list[VideoRef] = []
    opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
    }
    with YoutubeDL(opts) as ydl:
        for src in inputs:
            info = ydl.extract_info(src, download=False)
            if info is None:
                continue
            entries = info.get("entries")
            if entries is None:
                # single video
                refs.append(_to_ref(info))
            else:
                for e in entries:
                    if e is None:
                        continue
                    refs.append(_to_ref(e))
    # dedupe preserving order
    seen: set[str] = set()
    deduped: list[VideoRef] = []
    for r in refs:
        if r.video_id in seen:
            continue
        seen.add(r.video_id)
        deduped.append(r)
        if limit and len(deduped) >= limit:
            break
    return deduped


def _to_ref(info: dict) -> VideoRef:
    vid = info.get("id") or ""
    url = info.get("webpage_url") or info.get("url") or f"https://www.youtube.com/watch?v={vid}"
    return VideoRef(video_id=vid, url=url, title=info.get("title"))
