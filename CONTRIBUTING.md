# Contributing

Thanks for your interest. A few ground rules before you open a PR.

## Scope

This is a personal research project (see [ETHICS.md](ETHICS.md)). PRs that
expand the project toward bulk scraping, automation at scale, or
content-redistribution use cases will be declined regardless of code quality.

Good PR territory:

- Bug fixes in the pipeline (download, transcription, alignment, normalize,
  concat, diarization).
- New analysis/visualization options that operate on cached transcripts.
- Performance improvements (caching, parallelism within a single run).
- Documentation, examples, tests.
- New CLI flags that compose with existing ones rather than replacing them.

Out-of-scope (will be closed):

- Daemon / background-worker modes.
- Built-in distribution / upload to platforms.
- Account login, cookie injection, age-gate or geo bypass.
- "Trending word detector" or other auto-targeting features.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Install `ffmpeg` from your system package manager.

## Quality bar

Before pushing:

```bash
ruff check src tests
ruff format src tests
pytest
```

CI runs the same on every PR.

## Style

- Type-hint all public functions.
- Prefer pure functions where possible (most logic in
  [src/youtube_transform/search.py](src/youtube_transform/search.py) is pure
  and testable without network or GPU).
- Keep heavyweight imports (`torch`, `whisperx`, `pyannote`) lazy inside
  function bodies so `--help` and inspection-only commands stay snappy.
- Cache everything that's expensive to recompute, keyed by inputs.
- Fail loud on misconfiguration; fail soft (skip + log) on per-video errors
  so a long run doesn't die on one bad URL.

## Commits

Conventional Commits are encouraged but not enforced:

```
feat(search): add fuzzy phrase matching
fix(diarize): handle empty pyannote turns
docs: clarify HF token setup
```

## Reporting bugs

Open an issue with:

- Command you ran (redact URLs if you'd rather).
- Full stderr, including the `[yellow]skip ...` warnings.
- Output of `yt-transform --version` and `ffmpeg -version | head -n1`.
- Whether the bug is reproducible from cold cache.
