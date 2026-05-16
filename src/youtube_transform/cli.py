"""CLI entry point: yt-transform <word> <urls...> [options]"""

from __future__ import annotations

import hashlib
import random
import re
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .audio import fetch_audio
from .cache import Cache
from .concat import concat as ffmpeg_concat
from .concat import concat_xfade, ensure_ffmpeg, find_font, loudnorm, normalize
from .diarize import assign_speakers, diarize, list_speakers
from .download import fetch_clip
from .search import beat_snap, find_word, grep_transcript, suggest_similar, windows_from_hits
from .sources import resolve
from .transcribe import transcribe, transcript_stats

console = Console()


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-")[:60]


@click.command()
@click.argument("query")
@click.argument("urls", nargs=-1, required=True)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output supercut path. Default: out/<phrase>_<timestamp>.mp4",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("cache"),
    show_default=True,
)
@click.option(
    "--model",
    default="small",
    show_default=True,
    help="Whisper model: tiny|base|small|medium|large-v3",
)
@click.option(
    "--language", default=None, help="Force language code (e.g. 'en'). Auto-detect if omitted."
)
@click.option(
    "--max-height", type=int, default=720, show_default=True, help="Max video height to download."
)
@click.option(
    "--pad-before",
    type=float,
    default=0.4,
    show_default=True,
    help="Seconds of audio/video to keep BEFORE each match.",
)
@click.option(
    "--pad-after",
    type=float,
    default=0.6,
    show_default=True,
    help="Seconds of audio/video to keep AFTER each match.",
)
@click.option(
    "--context-words",
    type=int,
    default=0,
    show_default=True,
    help="Include this many transcript words on each side of the match (semantic context).",
)
@click.option(
    "--min-duration",
    type=float,
    default=0.0,
    show_default=True,
    help="Extend each clip symmetrically until at least this many seconds long.",
)
@click.option(
    "--max-duration",
    type=float,
    default=None,
    help="Trim each clip symmetrically if longer than this many seconds.",
)
@click.option(
    "--merge-gap",
    type=float,
    default=0.1,
    show_default=True,
    help="Merge windows whose gap is <= this many seconds.",
)
@click.option(
    "--limit-videos",
    type=int,
    default=None,
    help="Cap number of videos resolved from playlists/channels.",
)
@click.option(
    "--max-hits-per-video", type=int, default=None, help="Cap clips taken from any single video."
)
@click.option(
    "--max-clips",
    type=int,
    default=None,
    help="Cap total number of clips in the final supercut (applied after ordering).",
)
@click.option(
    "--order",
    type=click.Choice(["source", "shuffle", "interleave"]),
    default="source",
    show_default=True,
    help="Clip ordering: source=original per-video order; shuffle=random; interleave=round-robin across videos.",
)
@click.option("--seed", type=int, default=None, help="Seed for --order shuffle (reproducibility).")
@click.option(
    "--crossfade",
    type=float,
    default=0.0,
    show_default=True,
    help="Crossfade between clips (seconds). 0 = hard cuts (fastest).",
)
@click.option(
    "--audio-normalize",
    is_flag=True,
    help="Apply EBU R128 loudness normalization to the final supercut.",
)
@click.option(
    "--diarize",
    "do_diarize",
    is_flag=True,
    help="Run speaker diarization (requires HUGGING_FACE_HUB_TOKEN env var).",
)
@click.option(
    "--speaker",
    default=None,
    help="Filter to a single speaker label (e.g. 'SPEAKER_00') or 'auto' for the most-talkative.",
)
@click.option(
    "--caption/--no-caption",
    default=False,
    show_default=True,
    help="Burn the matched phrase as text on every clip.",
)
@click.option(
    "--caption-text",
    default=None,
    help="Override the caption text (defaults to QUERY when --caption is set).",
)
@click.option(
    "--font",
    "font_path",
    default=None,
    help="Path to a TTF font for captions. Auto-detects common system fonts.",
)
@click.option(
    "--speed",
    type=float,
    default=1.0,
    show_default=True,
    help="Playback speed for each clip. <1 = slow-mo emphasis, >1 = chipmunk.",
)
@click.option(
    "--bpm",
    type=float,
    default=None,
    help="Snap each clip's output duration to a beat grid at this BPM.",
)
@click.option(
    "--min-beats",
    type=int,
    default=1,
    show_default=True,
    help="Minimum beats per clip when --bpm is set.",
)
@click.option("--whole-word/--substring", default=True, show_default=True)
@click.option("--width", type=int, default=1280, show_default=True)
@click.option("--height", type=int, default=720, show_default=True)
@click.option("--fps", type=int, default=30, show_default=True)
@click.option("--dry-run", is_flag=True, help="Stop after listing hits; don't download clips.")
@click.option(
    "--grep",
    "grep_pattern",
    default=None,
    help="Inspection mode: show transcript lines matching this regex (with context) and exit.",
)
@click.option(
    "--retranscribe",
    is_flag=True,
    help="Ignore the cached transcript and re-run Whisper for each video.",
)
@click.option(
    "--dump-transcript",
    is_flag=True,
    help="Print every transcribed word with timestamps and exit (no clip building).",
)
def main(
    query: str,
    urls: tuple[str, ...],
    out_path: Path | None,
    cache_dir: Path,
    model: str,
    language: str | None,
    max_height: int,
    pad_before: float,
    pad_after: float,
    context_words: int,
    min_duration: float,
    max_duration: float | None,
    merge_gap: float,
    limit_videos: int | None,
    max_hits_per_video: int | None,
    max_clips: int | None,
    order: str,
    seed: int | None,
    crossfade: float,
    audio_normalize: bool,
    do_diarize: bool,
    speaker: str | None,
    caption: bool,
    caption_text: str | None,
    font_path: str | None,
    speed: float,
    bpm: float | None,
    min_beats: int,
    whole_word: bool,
    width: int,
    height: int,
    fps: int,
    dry_run: bool,
    grep_pattern: str | None,
    retranscribe: bool,
    dump_transcript: bool,
) -> None:
    """Build a supercut of every utterance of QUERY across the given YouTube URLS.

    QUERY may be a single word ("actually") or a multi-word phrase ("baby mama").
    """
    ensure_ffmpeg()
    if caption and find_font(font_path) is None:
        raise SystemExit(
            "--caption requested but no font found. Install a TTF (e.g. fonts-dejavu) or pass --font PATH."
        )
    cache = Cache(cache_dir)

    console.rule("[bold]resolving sources")
    refs = resolve(list(urls), limit=limit_videos)
    console.print(f"resolved [cyan]{len(refs)}[/] video(s)")
    if not refs:
        raise SystemExit("no videos resolved")

    if out_path is None:
        slug = _slugify(query) or "supercut"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Include video_id when single source so the filename is self-describing.
        vid_part = f"_{refs[0].video_id}" if len(refs) == 1 else ""
        out_path = Path("out") / f"{slug}{vid_part}_{ts}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_windows: list[tuple[str, str, float, float]] = []  # (video_id, url, start, end)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("transcribing", total=len(refs))
        for ref in refs:
            progress.update(task, description=f"audio: {ref.video_id}")
            try:
                audio = fetch_audio(ref.url, ref.video_id, cache)
            except Exception as e:
                console.print(f"[yellow]skip {ref.video_id} (audio): {e}")
                progress.advance(task)
                continue

            progress.update(task, description=f"whisper: {ref.video_id}")
            try:
                transcript = transcribe(
                    audio,
                    ref.video_id,
                    cache,
                    model_name=model,
                    language=language,
                    force=retranscribe,
                )
            except Exception as e:
                console.print(f"[yellow]skip {ref.video_id} (transcribe): {e}")
                progress.advance(task)
                continue

            # Coverage stats — reveals VAD eating audio, low confidence, etc.
            try:
                stats = transcript_stats(transcript, audio)
                ratio_s = (
                    f"{stats['coverage_ratio']:.0%}" if stats["coverage_ratio"] is not None else "?"
                )
                audio_s = (
                    f"{stats['audio_seconds']:.1f}s" if stats["audio_seconds"] is not None else "?"
                )
                console.print(
                    f"  [dim]{ref.video_id}[/]: {stats['n_words']} words, "
                    f"covers {stats['covered_seconds']:.1f}s of {audio_s} ({ratio_s}), "
                    f"median score {stats['median_score']:.2f}"
                )
            except Exception:
                pass

            if dump_transcript:
                for w in transcript["words"]:
                    console.print(f"  [{float(w['start']):8.2f}-{float(w['end']):8.2f}] {w['w']}")
                progress.advance(task)
                continue

            # Diarization: assign speaker label to each transcript word.
            chosen_speaker: str | None = None
            if do_diarize:
                progress.update(task, description=f"diarize: {ref.video_id}")
                try:
                    turns = diarize(audio, ref.video_id, cache)
                    transcript = assign_speakers(transcript, turns)
                    counts = list_speakers(transcript)
                    if counts:
                        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                        console.print(f"  [magenta]{ref.video_id}[/] speakers: {summary}")
                    if speaker:
                        if speaker == "auto":
                            chosen_speaker = max(counts, key=counts.get) if counts else None
                        else:
                            chosen_speaker = speaker
                except Exception as e:
                    console.print(f"[yellow]diarize skipped for {ref.video_id}: {e}")

            hits = find_word(
                transcript,
                query,
                whole_word=whole_word,
                context_words=context_words,
                speaker=chosen_speaker,
            )
            if grep_pattern:
                matches = grep_transcript(transcript, grep_pattern)
                console.print(f"  [cyan]{ref.video_id}[/]: {len(matches)} grep match(es)")
                for ts, snippet in matches[:50]:
                    console.print(f"    [{ts:8.2f}s] {snippet}")
            elif not hits:
                # "Did you mean?" suggestions when this video had nothing.
                sugg = suggest_similar(transcript, query)
                if sugg:
                    console.print(
                        f"  [yellow]{ref.video_id}[/]: 0 hits. Did you mean: [cyan]{', '.join(sugg)}[/]?"
                    )
            if max_hits_per_video:
                hits = hits[:max_hits_per_video]
            windows = windows_from_hits(
                hits,
                pad_before=pad_before,
                pad_after=pad_after,
                merge_gap=merge_gap,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            if bpm:
                windows = beat_snap(windows, bpm=bpm, min_beats=min_beats, speed=speed)
            console.print(
                f"  [green]{ref.video_id}[/]: {len(hits)} hit(s) → {len(windows)} window(s)"
            )
            for w in windows:
                all_windows.append((ref.video_id, ref.url, w.start, w.end))
            progress.advance(task)

    if grep_pattern:
        # Inspection mode — don't proceed to clip building.
        return
    if dump_transcript:
        return

    console.rule(f"[bold]found {len(all_windows)} window(s)")
    if not all_windows:
        console.print(
            "[yellow]Tips:[/] try [cyan]--substring[/] (e.g. matches 'awesomely' too), "
            "a different [cyan]--model[/] (e.g. medium/large-v3), "
            "or [cyan]--grep PATTERN[/] to inspect what Whisper actually heard."
        )
        raise SystemExit("no matches; nothing to do")

    all_windows = _order_windows(all_windows, order=order, seed=seed)
    if max_clips is not None:
        all_windows = all_windows[:max_clips]
        console.print(f"capped to [cyan]{len(all_windows)}[/] clip(s)")

    if dry_run:
        for vid, _url, s, e in all_windows:
            console.print(f"  {vid}  {s:8.2f} → {e:8.2f}  ({e - s:.2f}s)")
        return

    # Download clips
    raw_clips: list[Path] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]downloading clip {task.completed}/{task.total}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("clips", total=len(all_windows))
        for vid, url, s, e in all_windows:
            from .search import Hit

            hit = Hit(video_id=vid, word=query, start=s, end=e)
            try:
                p = fetch_clip(url, hit, cache, max_height=max_height)
                raw_clips.append(p)
            except Exception as ex:
                console.print(f"[yellow]skip clip {vid} {s:.2f}-{e:.2f}: {ex}")
            progress.advance(task)

    if not raw_clips:
        raise SystemExit("no clips downloaded")

    # Normalize all clips to common spec
    console.rule("[bold]normalizing clips")
    norm_dir = cache.root / "normalized"
    norm_dir.mkdir(exist_ok=True)
    cap_text = (caption_text if caption_text is not None else query) if caption else None
    # Cache key for normalization variants.
    norm_key = hashlib.sha1(
        f"{width}x{height}@{fps}|sp={speed}|cap={cap_text}".encode()
    ).hexdigest()[:8]
    norm_clips: list[Path] = []
    with Progress(console=console) as progress:
        task = progress.add_task("normalize", total=len(raw_clips))
        for src in raw_clips:
            dst = norm_dir / f"{src.stem}_{norm_key}.mp4"
            if not dst.exists():
                try:
                    normalize(
                        src,
                        dst,
                        width=width,
                        height=height,
                        fps=fps,
                        speed=speed,
                        caption=cap_text,
                        font_path=font_path,
                    )
                except Exception as ex:
                    console.print(f"[yellow]skip normalize {src.name}: {ex}")
                    progress.advance(task)
                    continue
            norm_clips.append(dst)
            progress.advance(task)

    if not norm_clips:
        raise SystemExit("no normalized clips")

    console.rule("[bold]concatenating")
    if crossfade > 0 and len(norm_clips) > 1:
        if audio_normalize:
            tmp = out_path.with_suffix(".raw.mp4")
            concat_xfade(norm_clips, tmp, crossfade=crossfade)
            loudnorm(tmp, out_path)
            tmp.unlink(missing_ok=True)
        else:
            concat_xfade(norm_clips, out_path, crossfade=crossfade)
    else:
        if audio_normalize:
            tmp = out_path.with_suffix(".raw.mp4")
            ffmpeg_concat(norm_clips, tmp)
            loudnorm(tmp, out_path)
            tmp.unlink(missing_ok=True)
        else:
            ffmpeg_concat(norm_clips, out_path)
    console.print(f"[bold green]done →[/] {out_path}  ({len(norm_clips)} clips)")


def _order_windows(
    windows: list[tuple[str, str, float, float]],
    *,
    order: str,
    seed: int | None,
) -> list[tuple[str, str, float, float]]:
    if order == "source":
        return windows
    if order == "shuffle":
        rng = random.Random(seed)
        out = list(windows)
        rng.shuffle(out)
        return out
    if order == "interleave":
        # Group by video_id preserving first-seen video order.
        buckets: dict[str, list[tuple[str, str, float, float]]] = {}
        for w in windows:
            buckets.setdefault(w[0], []).append(w)
        rounds = zip_longest(*buckets.values())
        out: list[tuple[str, str, float, float]] = []
        for r in rounds:
            for item in r:
                if item is not None:
                    out.append(item)
        return out
    return windows


if __name__ == "__main__":
    main()
