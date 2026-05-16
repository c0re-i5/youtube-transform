# Ethical Use & Research Stance

`youtube-transform` is a **personal research project** exploring three
intersecting problems:

1. Efficient retrieval of media segments without bulk transfer (HTTP Range
   download + keyframe-aware cutting).
2. Practical use of forced-alignment ASR (WhisperX) for sub-second word
   localization in long-form audio.
3. Pipeline design for caching, normalizing, and composing video clips at
   scale.

It is published as a portfolio artefact to demonstrate engineering choices,
not as a turnkey content-harvesting tool.

---

## Acceptable use

This software is intended for:

- **Personal, non-commercial research** into ASR, audio alignment, and video
  pipeline design.
- **Linguistic analysis** of speech patterns where you have rights to the
  source material.
- **Educational exploration** of multi-stage media-processing pipelines.
- **Accessibility** — generating word-level transcripts for personal use.

## Out of scope / not supported

The following uses are explicitly **not** supported. Pull requests aimed at
enabling them will be declined:

- **Bulk scraping** of channels, large playlists, or entire creator
  catalogues.
- **Re-uploading or monetizing** clips derived from third-party content.
- **Harassment supercuts** — assembling clips to misrepresent, ridicule, or
  defame a specific person.
- **Evading rate limits, age gates, or geographic restrictions** on platforms
  you do not have authorization to bypass.
- **Automated, unattended operation** at scale (no daemon mode, no queue
  workers, no scheduling are provided).

## Platform terms of service

This project uses [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) to access
public YouTube content. **Downloading content from YouTube may violate
YouTube's Terms of Service.** Users are solely responsible for ensuring their
use complies with applicable laws and the terms of any platform they access.

The author of this software:

- Does **not** host, redistribute, or mirror any third-party media.
- Does **not** circumvent DRM or paywalls.
- Does **not** provide any service, API, or hosted version of this tool.

## Copyright

Captions, audio, and video accessed through this tool remain the property of
their respective copyright holders. Generating a "supercut" does not transfer
or waive those rights. **Do not redistribute outputs derived from material you
do not own or have explicit permission to use.**

Fair-use, fair-dealing, and similar exceptions exist in many jurisdictions
for purposes such as criticism, commentary, parody, scholarship, and
research, but the scope of those exceptions varies. If you intend to publish
derivative work, consult a lawyer in your jurisdiction.

## Privacy

The optional `--diarize` flag uses
[`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)
to identify distinct speakers in an audio track. Speaker labels are
synthetic (`SPEAKER_00`, `SPEAKER_01`, ...) and do **not** perform speaker
identification against any database. Do not use diarization output to track
or profile individuals without consent.

## Reporting misuse

If you discover this tool being used in ways that violate the above, please
open an issue with the `misuse` label. The author will not assist with
takedowns of derivative work but will document patterns of misuse in this
file to help downstream users make informed decisions.
