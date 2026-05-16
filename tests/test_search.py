"""Pure-logic tests for `youtube_transform.search`.

These tests intentionally exercise only the dependency-free portion of the
codebase (no torch, whisperx, yt-dlp, or ffmpeg required), so they run fast
in CI on a clean Python install.
"""

from __future__ import annotations

import math

from youtube_transform.search import (
    Hit,
    beat_snap,
    find_word,
    grep_transcript,
    suggest_similar,
    windows_from_hits,
)


def _t(video_id: str, words: list[tuple[str, float, float]], **extra) -> dict:
    """Build a transcript dict in the shape `search` expects."""
    return {
        "video_id": video_id,
        "words": [{"w": w, "start": s, "end": e, **(extra.get(w, {}))} for (w, s, e) in words],
    }


# ---------------------------------------------------------------------------
# find_word
# ---------------------------------------------------------------------------


def test_find_word_single_match():
    tr = _t("v1", [("hello", 0.0, 0.4), ("world", 0.4, 0.9)])
    hits = find_word(tr, "world")
    assert len(hits) == 1
    assert hits[0].video_id == "v1"
    assert hits[0].word == "world"
    assert math.isclose(hits[0].start, 0.4)
    assert math.isclose(hits[0].end, 0.9)


def test_find_word_normalizes_punctuation_and_case():
    tr = _t("v1", [("Hello,", 0.0, 0.4), ("World!", 0.4, 0.9)])
    assert len(find_word(tr, "hello")) == 1
    assert len(find_word(tr, "WORLD")) == 1


def test_find_word_multi_word_phrase():
    tr = _t(
        "v1",
        [
            ("my", 0.0, 0.2),
            ("baby", 0.2, 0.6),
            ("mama", 0.6, 1.0),
            ("said", 1.0, 1.3),
        ],
    )
    hits = find_word(tr, "baby mama")
    assert len(hits) == 1
    assert hits[0].word == "baby mama"
    assert math.isclose(hits[0].start, 0.2)
    assert math.isclose(hits[0].end, 1.0)


def test_find_word_phrase_no_match_when_not_consecutive():
    tr = _t(
        "v1",
        [
            ("baby", 0.0, 0.4),
            ("loves", 0.4, 0.8),
            ("mama", 0.8, 1.2),
        ],
    )
    assert find_word(tr, "baby mama") == []


def test_find_word_no_overlapping_phrase_matches():
    # "ha ha" appearing 3 times consecutively should yield 1 non-overlapping match
    # at positions 0-1, then advance past, so a second match at 2-3 if present.
    tr = _t(
        "v1",
        [
            ("ha", 0.0, 0.2),
            ("ha", 0.2, 0.4),
            ("ha", 0.4, 0.6),
            ("ha", 0.6, 0.8),
        ],
    )
    hits = find_word(tr, "ha ha")
    assert len(hits) == 2


def test_find_word_substring_mode():
    tr = _t("v1", [("running", 0.0, 0.5)])
    assert find_word(tr, "run", whole_word=True) == []
    assert len(find_word(tr, "run", whole_word=False)) == 1


def test_find_word_context_expansion():
    tr = _t(
        "v1",
        [
            ("a", 0.0, 0.1),
            ("b", 0.1, 0.2),
            ("target", 0.2, 0.4),
            ("c", 0.4, 0.5),
            ("d", 0.5, 0.6),
        ],
    )
    hits = find_word(tr, "target", context_words=2)
    assert len(hits) == 1
    assert math.isclose(hits[0].start, 0.0)
    assert math.isclose(hits[0].end, 0.6)


def test_find_word_speaker_filter():
    tr = _t(
        "v1",
        [("hi", 0.0, 0.3), ("hi", 1.0, 1.3)],
        hi={"speaker": "SPEAKER_00"},
    )
    # Both 'hi' tokens share the extras dict from helper; tweak to differentiate.
    tr["words"][0]["speaker"] = "SPEAKER_00"
    tr["words"][1]["speaker"] = "SPEAKER_01"

    only_00 = find_word(tr, "hi", speaker="SPEAKER_00")
    assert len(only_00) == 1
    assert math.isclose(only_00[0].start, 0.0)


# ---------------------------------------------------------------------------
# windows_from_hits
# ---------------------------------------------------------------------------


def test_windows_padding_applied():
    hits = [Hit("v1", "x", 1.0, 1.5)]
    out = windows_from_hits(hits, pad_before=0.5, pad_after=0.25)
    assert math.isclose(out[0].start, 0.5)
    assert math.isclose(out[0].end, 1.75)


def test_windows_padding_clamped_to_zero():
    hits = [Hit("v1", "x", 0.1, 0.2)]
    out = windows_from_hits(hits, pad_before=10.0, pad_after=0.0)
    assert out[0].start == 0.0


def test_windows_min_duration_extends_symmetrically():
    hits = [Hit("v1", "x", 5.0, 5.2)]
    out = windows_from_hits(hits, pad_before=0.0, pad_after=0.0, min_duration=1.0)
    assert math.isclose(out[0].end - out[0].start, 1.0)
    # Centered on original midpoint 5.1
    assert math.isclose((out[0].start + out[0].end) / 2, 5.1)


def test_windows_max_duration_trims_symmetrically():
    hits = [Hit("v1", "x", 10.0, 20.0)]
    out = windows_from_hits(hits, pad_before=0.0, pad_after=0.0, max_duration=2.0)
    assert math.isclose(out[0].end - out[0].start, 2.0)
    assert math.isclose((out[0].start + out[0].end) / 2, 15.0)


def test_windows_merge_overlapping_in_same_video():
    hits = [Hit("v1", "x", 0.0, 1.0), Hit("v1", "y", 0.9, 2.0)]
    out = windows_from_hits(hits, pad_before=0.0, pad_after=0.0, merge_gap=0.0)
    assert len(out) == 1
    assert math.isclose(out[0].end, 2.0)


def test_windows_merge_respects_gap_threshold():
    hits = [Hit("v1", "x", 0.0, 1.0), Hit("v1", "y", 1.4, 2.0)]
    out = windows_from_hits(hits, pad_before=0.0, pad_after=0.0, merge_gap=0.5)
    assert len(out) == 1


def test_windows_does_not_merge_across_videos():
    hits = [Hit("v1", "x", 0.0, 1.0), Hit("v2", "y", 0.5, 1.5)]
    out = windows_from_hits(hits, pad_before=0.0, pad_after=0.0, merge_gap=10.0)
    assert len(out) == 2


def test_windows_empty_input():
    assert windows_from_hits([]) == []


# ---------------------------------------------------------------------------
# beat_snap
# ---------------------------------------------------------------------------


def test_beat_snap_120bpm_one_beat():
    # 120 bpm → 0.5s per beat. A ~0.5s clip should snap to exactly 0.5s source.
    hits = [Hit("v1", "x", 10.0, 10.4)]
    out = beat_snap(hits, bpm=120.0, min_beats=1, speed=1.0)
    assert math.isclose(out[0].end - out[0].start, 0.5, abs_tol=1e-9)


def test_beat_snap_respects_min_beats():
    hits = [Hit("v1", "x", 10.0, 10.1)]  # very short
    out = beat_snap(hits, bpm=120.0, min_beats=2, speed=1.0)
    # 2 beats at 120 bpm = 1.0s output. speed=1 → source=1.0s.
    assert math.isclose(out[0].end - out[0].start, 1.0)


def test_beat_snap_speed_doubles_source_duration():
    hits = [Hit("v1", "x", 10.0, 10.4)]
    out = beat_snap(hits, bpm=120.0, min_beats=1, speed=2.0)
    # 1 beat output (0.5s) at 2x speed → source must be 1.0s.
    assert math.isclose(out[0].end - out[0].start, 1.0)


def test_beat_snap_noop_on_zero_bpm():
    hits = [Hit("v1", "x", 1.0, 2.0)]
    assert beat_snap(hits, bpm=0.0) == hits


def test_beat_snap_noop_on_empty():
    assert beat_snap([], bpm=120.0) == []


# ---------------------------------------------------------------------------
# suggest_similar
# ---------------------------------------------------------------------------


def test_suggest_similar_finds_typos():
    tr = _t(
        "v1",
        [
            ("welcome", 0.0, 0.4),
            ("assume", 0.4, 0.8),
            ("apple", 0.8, 1.2),
        ],
    )
    out = suggest_similar(tr, "awesome", cutoff=0.5)
    assert any(w in out for w in ("welcome", "assume"))


def test_suggest_similar_handles_empty_target():
    tr = _t("v1", [("hello", 0.0, 0.4)])
    assert suggest_similar(tr, "") == []
    assert suggest_similar(tr, "   ") == []


def test_suggest_similar_handles_empty_vocab():
    tr = _t("v1", [])
    assert suggest_similar(tr, "anything") == []


# ---------------------------------------------------------------------------
# grep_transcript
# ---------------------------------------------------------------------------


def test_grep_returns_matches_with_timestamps():
    tr = _t(
        "v1",
        [
            ("hello", 0.0, 0.4),
            ("world", 0.4, 0.9),
            ("hello", 1.0, 1.4),
        ],
    )
    out = grep_transcript(tr, r"^hello$", context=1)
    assert len(out) == 2
    assert math.isclose(out[0][0], 0.0)
    assert math.isclose(out[1][0], 1.0)


def test_grep_is_case_insensitive():
    tr = _t("v1", [("Hello", 0.0, 0.4)])
    assert len(grep_transcript(tr, "hello")) == 1


def test_grep_brackets_the_matched_word():
    tr = _t("v1", [("alpha", 0.0, 0.4), ("beta", 0.4, 0.8), ("gamma", 0.8, 1.2)])
    out = grep_transcript(tr, "beta", context=1)
    assert len(out) == 1
    _, snippet = out[0]
    assert "[beta]" in snippet
    assert "alpha" in snippet and "gamma" in snippet
