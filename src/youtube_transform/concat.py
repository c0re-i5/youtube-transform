"""Normalize each clip to a common spec then concatenate with ffmpeg.

We always re-encode to a target spec because clips often come from videos with
different resolutions, fps, sample rates — concat -c copy fails in that case.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def find_font(override: str | None = None) -> str | None:
    if override:
        return override if Path(override).exists() else None
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def normalize(
    src: Path,
    dst: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    sample_rate: int = 48000,
    speed: float = 1.0,
    caption: str | None = None,
    font_path: str | None = None,
    target_duration: float | None = None,
) -> None:
    """Re-encode src to a canonical spec.

    - `speed`: 1.0 = unchanged, 0.5 = half-speed (slow-mo), 2.0 = double.
    - `caption`: text burned at the bottom of every frame.
    - `target_duration`: trim OUTPUT to exactly this many seconds (post-speed).
    """
    vfilters = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        f"fps={fps}",
        "setsar=1",
    ]
    afilters: list[str] = []

    if speed != 1.0:
        # Video: setpts multiplies presentation time. >1 = slower playback.
        vfilters.append(f"setpts={1.0 / speed:.6f}*PTS")
        # Audio: atempo factor; equals playback-rate, so SLOWER playback = atempo<1.
        # atempo accepts 0.5..100, chain if outside range.
        afilters += _atempo_chain(speed)

    if caption:
        font = find_font(font_path)
        if font is None:
            raise RuntimeError(
                "No font found for caption overlay. Install a TTF font or pass --font PATH."
            )
        # Use textfile to avoid ffmpeg's special-char escaping hell.
        cap_path = dst.with_suffix(".caption.txt")
        cap_path.write_text(caption)
        # Escape backslashes/colons in the textfile path.
        font_esc = font.replace("\\", "\\\\").replace(":", "\\:")
        path_esc = str(cap_path).replace("\\", "\\\\").replace(":", "\\:")
        vfilters.append(
            f"drawtext=fontfile='{font_esc}':textfile='{path_esc}'"
            ":fontcolor=white:fontsize=48:borderw=3:bordercolor=black"
            ":x=(w-text_w)/2:y=h-text_h-40"
        )

    vf = ",".join(vfilters)
    af = ",".join(afilters) if afilters else "anull"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        "-af",
        af,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
    ]
    if target_duration is not None and target_duration > 0:
        cmd += ["-t", f"{target_duration:.3f}"]
    cmd.append(str(dst))
    try:
        _run(cmd)
    finally:
        if caption:
            dst.with_suffix(".caption.txt").unlink(missing_ok=True)


def _atempo_chain(speed: float) -> list[str]:
    """Build an atempo filter chain. atempo only accepts 0.5..100, so chain when needed."""
    s = float(speed)
    parts: list[str] = []
    # speed > 1 means faster playback → atempo > 1
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    if abs(s - 1.0) > 1e-6:
        parts.append(f"atempo={s:.6f}")
    return parts or ["anull"]


def concat(clips: list[Path], out: Path) -> None:
    """Concat normalized clips with the ffmpeg concat demuxer (-c copy, fast)."""
    if not clips:
        raise ValueError("no clips to concat")
    list_file = out.with_suffix(".txt")
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in clips))
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(out),
    ]
    try:
        _run(cmd)
    finally:
        list_file.unlink(missing_ok=True)


def _probe_duration(p: Path) -> float:
    """Get clip duration in seconds via ffprobe."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(p),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {p}: {proc.stderr}")
    return float(proc.stdout.strip())


def concat_xfade(clips: list[Path], out: Path, *, crossfade: float = 0.15) -> None:
    """Concat with video xfade + audio crossfade. Re-encodes (slower than copy)."""
    if not clips:
        raise ValueError("no clips to concat")
    if len(clips) == 1:
        # nothing to crossfade; just copy
        concat(clips, out)
        return

    durations = [_probe_duration(c) for c in clips]
    # Build filter graph: chain xfade/acrossfade across all clips.
    # cumulative offset for next xfade = sum of previous output durations - crossfade
    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]

    parts: list[str] = []
    prev_v = "0:v"
    prev_a = "0:a"
    cum = durations[0]
    for i in range(1, len(clips)):
        offset = cum - crossfade
        if offset < 0:
            offset = 0
        out_v = f"v{i}"
        out_a = f"a{i}"
        parts.append(
            f"[{prev_v}][{i}:v]xfade=transition=fade:duration={crossfade}:offset={offset:.3f}[{out_v}]"
        )
        parts.append(f"[{prev_a}][{i}:a]acrossfade=d={crossfade}[{out_a}]")
        prev_v, prev_a = out_v, out_a
        cum += durations[i] - crossfade

    filter_complex = ";".join(parts)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{prev_v}]",
        "-map",
        f"[{prev_a}]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(out),
    ]
    _run(cmd)


def loudnorm(src: Path, dst: Path) -> None:
    """Apply EBU R128 loudness normalization to the audio track."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    _run(cmd)
