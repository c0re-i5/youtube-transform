# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-16

Initial public release.

### Added

- Source resolution for single videos, playlists, and channels via
  `yt-dlp` flat extraction (no media download).
- Audio-only fetch + WhisperX word-level transcription with on-disk caching.
- Multi-word phrase search with whole-word / substring modes and optional
  context-word window expansion.
- HTTP-Range clip download via `yt-dlp` `download_ranges` with
  `force_keyframes_at_cuts`.
- Per-clip normalize → ffmpeg concat (with optional `xfade` crossfade and
  EBU R128 loudness normalization).
- Effect controls: caption overlay, speed ramp (`atempo`-corrected audio),
  BPM beat-snap, min/max duration clamps.
- Optional pyannote speaker diarization (`--diarize`, `--speaker`).
- Diagnostics: `--dump-transcript`, `--grep`, `--retranscribe`, coverage
  stats, "did you mean?" fuzzy suggestions on zero-hit videos.
- Clip ordering: `source` / `shuffle` / `interleave`.
- Auto-generated output filename: `<slug>_<video_id?>_<timestamp>.mp4`.

[Unreleased]: https://github.com/c0re-i5/youtube-transform/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/c0re-i5/youtube-transform/releases/tag/v0.1.0
