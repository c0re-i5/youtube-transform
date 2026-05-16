"""Search a transcript for word matches → list of clip windows."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


@dataclass
class Hit:
    video_id: str
    word: str
    start: float
    end: float


def _normalize(s: str) -> str:
    # lowercase, strip punctuation around the token
    return re.sub(r"[^\w']", "", s).lower()


def find_word(
    transcript: dict,
    target: str,
    *,
    whole_word: bool = True,
    context_words: int = 0,
    speaker: str | None = None,
) -> list[Hit]:
    """Return one Hit per occurrence of `target` in the transcript words.

    `target` may be a single word or a multi-word phrase (whitespace-separated).
    For phrases, all tokens must match consecutively; the Hit spans
    the first token's start to the last token's end.

    If `context_words > 0`, the Hit window is extended to include up to that
    many transcript words on each side of the match.

    If `speaker` is provided, only matches whose first token belongs to that
    speaker label (per `assign_speakers`) are returned.
    """
    target_tokens = [t for t in (_normalize(t) for t in target.split()) if t]
    if not target_tokens:
        return []

    # Build a parallel list of (normalized_token, original_word_dict) for non-empty tokens.
    items: list[tuple[str, dict]] = []
    for w in transcript["words"]:
        token = _normalize(w["w"])
        if token:
            items.append((token, w))

    n = len(target_tokens)
    hits: list[Hit] = []

    def _match(a: str, b: str) -> bool:
        return (a == b) if whole_word else (b in a)

    i = 0
    while i <= len(items) - n:
        if all(_match(items[i + k][0], target_tokens[k]) for k in range(n)):
            if speaker is not None and items[i][1].get("speaker") != speaker:
                i += 1
                continue
            ctx_lo = max(0, i - context_words)
            ctx_hi = min(len(items) - 1, i + n - 1 + context_words)
            first = items[ctx_lo][1]
            last = items[ctx_hi][1]
            hits.append(
                Hit(
                    video_id=transcript["video_id"],
                    word=" ".join(items[i + k][1]["w"] for k in range(n)),
                    start=float(first["start"]),
                    end=float(last["end"]),
                )
            )
            i += n  # don't double-count overlapping matches of the same phrase
        else:
            i += 1
    return hits


def windows_from_hits(
    hits: list[Hit],
    *,
    pad_before: float = 0.25,
    pad_after: float = 0.35,
    merge_gap: float = 0.0,
    min_duration: float = 0.0,
    max_duration: float | None = None,
) -> list[Hit]:
    """Pad, clamp, and optionally merge overlapping windows.

    - `pad_before/after`: seconds added to each side.
    - `min_duration`: extend symmetrically until the clip reaches this length.
    - `max_duration`: trim symmetrically (centered on midpoint) if longer.
    - `merge_gap`: merge windows in the same video whose gap is <= this many seconds.
    """
    if not hits:
        return []
    out: list[Hit] = []
    for h in hits:
        s = max(0.0, h.start - pad_before)
        e = h.end + pad_after
        # min duration: extend symmetrically
        if e - s < min_duration:
            extra = (min_duration - (e - s)) / 2
            s = max(0.0, s - extra)
            e = e + extra
        # max duration: trim symmetrically about the midpoint
        if max_duration is not None and e - s > max_duration:
            mid = (s + e) / 2
            s = max(0.0, mid - max_duration / 2)
            e = s + max_duration
        out.append(Hit(h.video_id, h.word, s, e))

    out.sort(key=lambda h: (h.video_id, h.start))
    merged: list[Hit] = []
    for h in out:
        if merged and merged[-1].video_id == h.video_id and h.start <= merged[-1].end + merge_gap:
            prev = merged[-1]
            merged[-1] = Hit(prev.video_id, prev.word, prev.start, max(prev.end, h.end))
        else:
            merged.append(h)
    return merged


def beat_snap(
    hits: list[Hit],
    *,
    bpm: float,
    min_beats: int = 1,
    speed: float = 1.0,
) -> list[Hit]:
    """Snap each window's source duration so the OUTPUT lasts an integer number of beats.

    output_duration = beats * (60 / bpm)
    source_duration = output_duration * speed   (since output = source / speed)

    Each clip is centered on its current midpoint and trimmed/extended.
    """
    if bpm <= 0 or not hits:
        return hits
    beat_out = 60.0 / bpm
    snapped: list[Hit] = []
    for h in hits:
        cur_out = (h.end - h.start) / max(speed, 1e-6)  # ignored for bookkeeping; we want SOURCE
        target_beats = max(min_beats, round(cur_out / beat_out))
        target_source = target_beats * beat_out * speed
        mid = (h.start + h.end) / 2
        s = max(0.0, mid - target_source / 2)
        e = s + target_source
        snapped.append(Hit(h.video_id, h.word, s, e))
    return snapped


def suggest_similar(transcript: dict, target: str, n: int = 8, cutoff: float = 0.6) -> list[str]:
    """Return up to `n` distinct vocabulary tokens most similar to `target`.

    Uses difflib ratio. Useful when a search returns 0 hits — surfaces typos,
    Whisper mishearings, or words the model rendered differently than expected.
    """
    target_norm = _normalize(target.split()[0]) if target.strip() else ""
    if not target_norm:
        return []
    vocab = {_normalize(w["w"]) for w in transcript.get("words", [])}
    vocab.discard("")
    return difflib.get_close_matches(target_norm, list(vocab), n=n, cutoff=cutoff)


def grep_transcript(transcript: dict, pattern: str, *, context: int = 4) -> list[tuple[float, str]]:
    """Return `(timestamp, snippet)` lines where the regex matches the joined word stream.

    Snippet shows `context` words on each side of the matched word for sanity-checking.
    """
    rx = re.compile(pattern, re.IGNORECASE)
    words = transcript.get("words", [])
    out: list[tuple[float, str]] = []
    for i, w in enumerate(words):
        if rx.search(w["w"]):
            lo = max(0, i - context)
            hi = min(len(words), i + context + 1)
            snippet = " ".join(
                f"[{ww['w']}]" if j == i else ww["w"] for j, ww in enumerate(words[lo:hi], start=lo)
            )
            out.append((float(w["start"]), snippet))
    return out
