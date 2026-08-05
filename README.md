# فصل الصوت — Vocals & Def (Drums) Extractor

Separate audio into AI stems, keep **only the Vocals and the Drums (Def /
percussion)** stems, merge them with balanced, clip-free levels, and export a
single file.

Comes with **two interfaces**:

- `separator.py` — a command-line tool.
- `webapp.py` — a local web GUI (paste YouTube URLs, drag & drop files, watch
  live progress, download the result).

## Quick start (web GUI)

```bash
source .venv/bin/activate
python webapp.py
# open http://127.0.0.1:8000
```

## Quick start (CLI)

```bash
python separator.py "https://www.youtube.com/watch?v=..."
python separator.py track.mp3
python separator.py ./songs_folder --format wav
```

## Requirements

- Python 3.8+ (3.11 recommended)
- `ffmpeg` on your PATH (required for downloads, decoding and MP3 export):
  `brew install ffmpeg`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# YouTube / Web URLs
python separator.py "https://www.youtube.com/watch?v=..."

# Direct audio links
python separator.py https://example.com/song.mp3

# A single local file
python separator.py track.mp3

# A whole folder of audio/video files
python separator.py ./songs

# Mix of everything, WAV output, multiple URLs
python separator.py url1 "url2" local.mp3 ./folder --format wav
```

Outputs are written to `output/` with the naming scheme:

```
[Original_Name]_Vocals_and_Def.mp3
```

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `-o, --output-dir` | `output` | Output folder |
| `-f, --format` | `mp3` | Output format: `mp3` or `wav` |
| `-b, --bitrate` | `320k` | MP3 bitrate |
| `-m, --model` | `htdemucs` | Separation model (`htdemucs`, `htdemucs_ft`, `BS-RoFormer`, ...) |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` (device is auto-detected; `cpu` forces software mode) |
| `--cache-dir` | `download_cache/` | Folder where downloaded audio is cached for reuse (or `$VOCAL_DEF_CACHE_DIR`) |
| `--no-cache` | off | Do not use the download cache; always download from the network |
| `--clear-cache` | off | Delete the download cache and exit |
| `--keep-temp` | off | Keep temporary downloads/stems |
| `--temp-dir` | system tmp | Base directory for temp files |
| `-v, --verbose` | off | Debug logging |

## Download cache

Downloads are cached on disk (default: `download_cache/` next to the script),
keyed by the video ID / URL. Requesting the same URL again reuses the cached
WAV instead of downloading it again, so re-running a failed job or re-submitting
the same song costs almost nothing. Clearing the cache (`--clear-cache` CLI flag
or the "Clear download cache" button in the web GUI) removes all cached files so
they are fetched fresh next time.

## Web GUI

```bash
python webapp.py                  # http://127.0.0.1:8000
python webapp.py --port 9000      # custom port
python webapp.py --host 0.0.0.0   # expose to the local network (be careful)
```

Features:

- Paste YouTube/Web URLs (one per line) **and/or** drag & drop local files.
- Pick the model (`htdemucs`, `htdemucs_ft`, `BS-RoFormer`), output format and
  bitrate.
- Live job progress with an indeterminate animation during AI inference, a
  scrollable log console, and per-job download buttons.
- Jobs run serially through a single worker; the model stays loaded in memory
  between jobs for speed.

API for scripted use:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/jobs` | Create a job (multipart: `urls`, `files`, `format`, `model`, `bitrate`) |
| `GET` | `/api/jobs/<id>` | Job status + logs |
| `GET` | `/api/jobs` | All jobs |
| `GET` | `/output/<file>` | Download a result |
| `GET` | `/api/cache` | Download cache info (file count, size) |
| `DELETE` | `/api/cache` | Clear the download cache |

## How it works

1. **Input** — URLs (YouTube or direct links) are downloaded as high-quality
   audio with `yt-dlp` and converted to WAV via ffmpeg. Local files/folders are
   decoded to WAV.
2. **Separation** — `audio-separator` (demucs `htdemucs`) splits the track into
   `vocals`, `drums`, `bass` and `other`.
3. **Merge** — `bass` and `other` are discarded. Vocals and drums are normalized
   to −9 dBFS for mixing headroom, overlaid, then peak-normalized to −1 dBFS so
   the result never clips.
4. **Output** — merged file is exported to `output/` and all temporary stem and
   download files are cleaned up automatically.

The model is downloaded automatically on first run (~80 MB for `htdemucs`).

## Disk space handling

Separation writes several large WAVs (download + stems + output), so jobs need
roughly 500 MB–1 GB of free space on both the temp volume and the output
folder. The tools check this **before** starting:

- If free space is below 1 GB the job fails immediately with a clear message
  instead of producing cryptic ffmpeg decode errors ("Invalid data found when
  processing input") from files truncated by a full disk.
- Truncated/corrupt WAV files are detected after download and separation, and a
  corrupt cached download is deleted automatically so the next run re-fetches it.
- On startup, the web GUI removes leftover job temp dirs from crashed runs to
  reclaim space.

Tune the threshold with `VOCAL_DEF_MIN_FREE_MB` (e.g. `512` for small volumes).

## Notes

- CUDA is used automatically when available, with an automatic CPU fallback.
- Broken links, unsupported files and decode failures are reported and skipped
  without aborting the whole batch.
- Set `--keep-temp` to inspect the intermediate stems for debugging.
