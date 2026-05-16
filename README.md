# youtube-transform

<p>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href=".github/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/badge/ci-ruff%20%2B%20pytest-success"></a>
  <a href="CHANGELOG.md"><img alt="Status: Alpha" src="https://img.shields.io/badge/status-alpha-orange.svg"></a>
</p>

> **Research project.** Bandwidth-efficient word-supercut generation from
> YouTube videos using Whisper forced-alignment timestamps and HTTP Range
> downloads.

`youtube-transform` is a CLI that, given a word or short phrase and one or
more YouTube source URLs, produces a single stitched video containing every
utterance of that target — **without ever downloading the full source
videos**.

It exists to explore three problems together:

1. Efficient retrieval of media segments without bulk transfer.
2. Practical use of forced-alignment ASR for sub-second word localization in
   long-form audio.
3. Stage-cached pipeline design for reproducible media composition.

---

## ⚠️ Research-use notice

This is a **personal research and portfolio project**. It is **not** a
content-harvesting tool, scraping framework, or platform-abuse utility.

- ✅ Personal, non-commercial study of ASR pipelines and video tooling.
- ✅ Linguistic / accessibility work on material you have rights to use.
- ❌ Bulk scraping of channels or catalogues.
- ❌ Re-uploading or monetizing derivative supercuts of third-party content.
- ❌ Harassment supercuts; misrepresentation; ToS or DRM evasion.

Read **[ETHICS.md](ETHICS.md)** before using or contributing. Downloading
from YouTube may violate that platform's Terms of Service; users are solely
responsible for compliance with applicable laws and platform terms.

---

## Table of contents

- [Why it's interesting](#why-its-interesting)
- [Architecture](#architecture)
- [Install](#install)
- [Quickstart](#quickstart)
- [Configuration reference](#configuration-reference)
- [Recipes](#recipes)
- [Caching & performance](#caching--performance)
- [Diagnostics](#diagnostics)
- [Project layout](#project-layout)
- [Development](#development)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Why it's interesting

A naive implementation downloads every source video in full, runs ASR, cuts
clips, and concatenates them. That wastes bandwidth, time, and storage on
material that will never appear in the final output. This project is built
around a different ordering:

| Stage         | What it touches                                                      | Why it's cheap                                                  |
| ------------- | -------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Resolve**   | Playlist / channel index via `yt-dlp` flat extraction                | Tens of KB; no media touched.                                   |
| **Audio**     | `bestaudio[m4a]` per video                                           | ~10× smaller than equivalent video.                             |
| **Transcribe**| WhisperX → JSON with per-word timestamps                             | One-time per `video_id`; reused for every future query.         |
| **Search**    | Pure-Python pass over the cached word stream                         | Microseconds.                                                   |
| **Download**  | HTTP Range fetch of the matched windows via `yt-dlp` `download_ranges` | Only the bytes you'll actually use, snapped to nearest keyframe. |
| **Normalize** | `ffmpeg` re-encode to common spec (size, fps, sar, audio)             | Cached per clip × variant (speed / caption / dimensions).       |
| **Concat**    | `ffmpeg` concat demuxer, or `xfade`+`acrossfade` filter graph         | Stream copy when no crossfade; one re-encode otherwise.         |

Every cacheable artefact is keyed by `video_id` (and, for variants, by a
hash of the rendering parameters) so re-running with a different target word
or different stylistic options reuses everything it possibly can.

## Architecture

```
                                    yt-dlp flat extract
   YouTube URLs ─────────────────►  ────────────────────► VideoRef[]
   (video / playlist / channel)         (sources.py)
                                                │
                          ┌─────────────────────┼──────────────────────┐
                          ▼                     ▼                      ▼
                   bestaudio[m4a]      cached transcripts       optional pyannote
                    (audio.py)         (transcribe.py)            (diarize.py)
                          │                     │                      │
                          └────────┬────────────┘                      │
                                   ▼                                   │
                          word-timestamped JSON ◄────────────── speaker turns
                                   │
                                   ▼
                          find_word + windows_from_hits + beat_snap
                                   │           (search.py)
                                   ▼
                          [(video_id, start, end)]
                                   │
                                   ▼
                          HTTP Range clip fetch
                          (download.py + yt-dlp download_ranges,
                           force_keyframes_at_cuts)
                                   │
                                   ▼
                          ffmpeg normalize per clip
                          (concat.py: scale/pad/fps/setsar,
                           atempo speed, drawtext caption)
                                   │
                                   ▼
                    ffmpeg concat (demuxer) | xfade + acrossfade
                                   │
                                   ▼
                          optional EBU R128 loudnorm
                                   │
                                   ▼
                          out/<slug>_<video_id?>_<timestamp>.mp4
```

## Install

System requirement: **`ffmpeg`** on `PATH`. Install via your OS package
manager (`apt install ffmpeg`, `brew install ffmpeg`, etc.).

```bash
git clone https://github.com/c0re-i5/youtube-transform.git
cd youtube-transform
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

WhisperX needs a matching `torch` build. CPU works for short videos; GPU is
much faster for anything longer than a few minutes:

```bash
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Diarization is opt-in and requires accepting the
[`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
license on Hugging Face and setting:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxx
```

## Quickstart

```bash
# Find every "actually" in a single video
yt-transform "actually" https://www.youtube.com/watch?v=VIDEO_ID

# Multi-word phrase across mixed sources
yt-transform "baby mama" \
  https://www.youtube.com/watch?v=A \
  https://www.youtube.com/playlist?list=PL... \
  --model small \
  --max-height 480 \
  --pad-before 0.3 --pad-after 0.4 \
  --merge-gap 0.1

# Inspect matches without downloading any video
yt-transform "ok" https://... --dry-run
```

Output filename pattern (auto-generated under `out/`):

```
<phrase-slug>_<video_id?>_<YYYYMMDD-HHMMSS>.mp4
```

The `video_id` is included only when the run targets a single source URL.

## Configuration reference

### Search & windowing

| Flag                       | Default | Description                                                          |
| -------------------------- | ------- | -------------------------------------------------------------------- |
| `--whole-word/--substring` | whole   | Whole-word vs. substring match per token.                            |
| `--context-words N`        | `0`     | Extend each window by N transcript words on each side.               |
| `--pad-before SEC`         | `0.4`   | Seconds of audio before each match.                                  |
| `--pad-after SEC`          | `0.6`   | Seconds of audio after each match.                                   |
| `--min-duration SEC`       | `0`     | Symmetrically extend short clips up to this length.                  |
| `--max-duration SEC`       | none    | Symmetrically trim long clips down to this length.                   |
| `--merge-gap SEC`          | `0`     | Merge near-adjacent windows within the same video.                   |
| `--max-hits-per-video N`   | none    | Cap matches taken from a single video.                               |
| `--max-clips N`            | none    | Cap total clips in the final output.                                 |
| `--limit-videos N`         | none    | Cap how many source videos to process.                               |

### Ordering & composition

| Flag                                | Default     | Description                                                       |
| ----------------------------------- | ----------- | ----------------------------------------------------------------- |
| `--order source\|shuffle\|interleave` | `source`    | Clip ordering strategy in the final cut.                          |
| `--seed N`                          | none        | Deterministic seed for `shuffle` / `interleave`.                  |
| `--crossfade SEC`                   | `0`         | Crossfade between clips (forces a re-encode pass).                |
| `--audio-normalize`                 | off         | EBU R128 loudnorm pass on the mixed output (`I=-16 TP=-1.5 LRA=11`). |
| `--max-height PX`                   | `720`       | Cap source resolution; lower = faster downloads.                  |

### Effects

| Flag                         | Default | Description                                                            |
| ---------------------------- | ------- | ---------------------------------------------------------------------- |
| `--caption`                  | off     | Burn the matched phrase as on-screen text.                             |
| `--caption-text "TEXT"`      | match   | Override the burned-in text.                                           |
| `--font PATH`                | auto    | Path to a `.ttf`. Auto-detects DejaVu / Liberation / Arial.            |
| `--speed FLOAT`              | `1.0`   | Per-clip playback speed; audio pitch-corrected via chained `atempo`.   |
| `--bpm FLOAT`                | none    | Snap each clip's *output* duration to an integer number of beats.       |
| `--min-beats N`              | `1`     | Floor on the beat count per clip when `--bpm` is set.                  |

### Diarization

| Flag                         | Default | Description                                                            |
| ---------------------------- | ------- | ---------------------------------------------------------------------- |
| `--diarize`                  | off     | Run pyannote and attach `speaker` labels to each transcript word.       |
| `--speaker LABEL\|auto`      | none    | Filter matches by speaker (`SPEAKER_00`, `auto` = most-talkative).      |

### Diagnostics

| Flag                         | Description                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| `--dry-run`                  | List every match across every video without downloading clips.                       |
| `--dump-transcript`          | Print the cached transcript for each video; useful when matches seem wrong.          |
| `--grep PATTERN`             | Regex-grep the joined word stream and print snippets with timestamps.                |
| `--retranscribe`             | Force re-running ASR (bypass the transcript cache). Combine with a bigger `--model`. |
| `--model SIZE`               | `tiny` / `base` / `small` / `medium` / `large-v3` (`small` is the sweet spot).        |

## Recipes

**Slow-motion emphasis with captions**

```bash
yt-transform "actually" URL \
  --speed 0.5 --caption \
  --pad-before 0.3 --pad-after 0.4
```

**Beat-synced music-video pacing**

```bash
yt-transform "yeah" URL \
  --bpm 120 --min-beats 1 \
  --crossfade 0.12 --audio-normalize
```

**Single-speaker supercut**

```bash
yt-transform "actually" URL --diarize --speaker auto
```

**Reproducible interleaved cut from multiple sources**

```bash
yt-transform "literally" URL_A URL_B URL_C \
  --order interleave --seed 42 \
  --max-hits-per-video 8
```

## Caching & performance

```
cache/
  audio/<video_id>.m4a            # one-time per source
  transcripts/<video_id>.json     # WhisperX word-timestamps
  diarization/<video_id>.json     # pyannote turns (optional)
  clips/<video_id>_<start>_<end>.mp4
  normalized/<video_id>_<start>_<end>_<param-hash>.mp4
  meta/                           # bookkeeping
```

- Transcripts are reusable across **every future search** against that video.
- Normalized clips are keyed by `(width, height, fps, speed, caption text)`
  so changing a stylistic flag invalidates only what it must.
- A first cold run downloads Whisper + alignment model weights (hundreds of
  MB); subsequent runs reuse them from `~/.cache/`.

## Diagnostics

When a search returns zero hits even though you *know* the word is in the
audio, the typical culprit is Whisper mishearing it. The CLI has three tools
for this:

1. **"Did you mean?"** — on zero-hit videos, the CLI runs a fuzzy-vocabulary
   match against the cached transcript and surfaces near-spellings.
2. **`--grep`** — regex-search the raw transcript word stream with bracketed
   snippet context, to confirm what the model actually heard.
3. **`--retranscribe --model medium`** (or larger) — re-run ASR with a more
   capable model when alignment quality matters.

Run with `--dump-transcript` to inspect coverage statistics (how much of the
audio actually produced words, and the median alignment score).

## Project layout

```
src/youtube_transform/
  cli.py          # click entry point, orchestration, output filename
  sources.py      # URL / playlist / channel → VideoRef list (flat extract)
  audio.py        # bestaudio[m4a] download
  transcribe.py   # WhisperX wrapper, cache, coverage stats
  search.py       # pure-logic word/phrase match + window math + beat-snap
  download.py     # byte-range clip fetch via yt-dlp download_ranges
  diarize.py      # pyannote wrapper + speaker assignment
  concat.py       # ffmpeg normalize/concat/xfade/loudnorm; font lookup
  cache.py        # filesystem cache layout helpers

tests/
  test_search.py  # pure-logic coverage; runs in CI without torch/ffmpeg

.github/
  workflows/ci.yml          # ruff + pytest matrix (3.10 / 3.11 / 3.12)
  ISSUE_TEMPLATE/*.yml      # structured bug & feature templates
  PULL_REQUEST_TEMPLATE.md  # PR checklist incl. ethics review
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check src tests
ruff format src tests
pytest
```

CI runs the same on every push and PR against `main` for Python 3.10–3.12.

The pure-logic tests in [tests/test_search.py](tests/test_search.py) cover
phrase matching, window padding/merging, beat-snap math, fuzzy suggestion,
and regex grep — all without requiring `torch`, `whisperx`, `ffmpeg`, or the
network.

See [CONTRIBUTING.md](CONTRIBUTING.md) for code-style and PR conventions,
and [ETHICS.md](ETHICS.md) for the scope statement contributions must fit
within.

## Limitations

- **ASR quality.** Whisper-class models mishear unusual words, names, and
  audio with music or heavy background noise. Larger models help but are
  slower and need more RAM/VRAM.
- **Live / age-gated / member-only content** may fail to download. Errors
  are logged per-video and the run continues with the rest.
- **Diarization is best-effort.** pyannote labels are synthetic and stable
  *within* a single audio file, not across files.
- **Keyframe snapping** means the actual clip start can shift up to a few
  hundred milliseconds earlier than the requested window. `--pad-before`
  compensates.
- **No bulk / unattended mode.** This is intentional; see [ETHICS.md](ETHICS.md).

## Acknowledgements

This project would not exist without the work of:

- [**WhisperX**](https://github.com/m-bain/whisperX) — forced alignment on
  top of OpenAI Whisper, producing the per-word timestamps that the whole
  pipeline pivots on.
- [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) — flat playlist extraction
  and the `download_ranges` + `force_keyframes_at_cuts` machinery that makes
  byte-range fetches viable.
- [**FFmpeg**](https://ffmpeg.org/) — every cut, normalize, crossfade,
  caption, and loudnorm pass.
- [**pyannote.audio**](https://github.com/pyannote/pyannote-audio) — the
  optional speaker-diarization stage.
- [**Click**](https://click.palletsprojects.com/) and
  [**Rich**](https://rich.readthedocs.io/) — the CLI surface and progress UI.

## License

[MIT](LICENSE) © 2026 c0re-i5.

The MIT license covers this source code only. It does **not** grant any
rights to content downloaded through the tool, which remains the property of
its respective copyright holders.
