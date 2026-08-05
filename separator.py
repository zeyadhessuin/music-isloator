#!/usr/bin/env python3
"""Vocals & Def (Drums) Extractor CLI + pipeline library.

Accepts YouTube/Web URLs, direct audio links, single local files, or a folder
of audio files. Separates audio into stems with an AI model (htdemucs by
default), keeps only the vocals and drums (percussion / "Def") stems, merges
them with normalized levels, and exports a single file to output/.

The pipeline functions (process_items, SeparationEngine, collect_inputs) are
importable so other frontends (e.g. webapp.py) can reuse them.

Usage:
    python separator.py "https://www.youtube.com/watch?v=..."
    python separator.py https://example.com/song.mp3
    python separator.py track.mp3
    python separator.py ./songs_folder
    python separator.py url1 url2 file.mp3 --format wav
"""

import argparse
import hashlib
import logging
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus",
    ".wma", ".aiff", ".aif", ".amr", ".mka", ".alac",
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts", ".flv"}

SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

URL_RE = re.compile(r"^(https?|ftp)://", re.IGNORECASE)

DEFAULT_MODEL = "htdemucs"

DEFAULT_CACHE_DIR = Path(
    os.environ.get("VOCAL_DEF_CACHE_DIR", str(Path(__file__).resolve().parent / "download_cache"))
)

MODEL_ALIASES = {
    "htdemucs": "htdemucs.yaml",
    "htdemucs_ft": "htdemucs_ft.yaml",
    "htdemucs_6s": "htdemucs_6s.yaml",
    "BS-RoFormer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "bs-roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
}

# Performance presets
PERFORMANCE_PRESETS = {
    "fast": {
        "segment_size": 128,
        "shifts": 1,
        "overlap": 0.15,
        "description": "Fastest processing, lower quality"
    },
    "balanced": {
        "segment_size": 256,
        "shifts": 2,
        "overlap": 0.25,
        "description": "Balanced speed and quality (default)"
    },
    "quality": {
        "segment_size": 512,
        "shifts": 4,
        "overlap": 0.35,
        "description": "Best quality, slower processing"
    }
}

log = logging.getLogger("vocal_def")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="separator.py",
        description=(
            "Separate audio into stems, keep only Vocals and Drums (Def), "
            "merge them, and export a single file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="YouTube/Web URLs, direct audio links, a local file, or a folder of audio files.",
    )
    parser.add_argument(
        "-o", "--output-dir", default="output", help="Folder where the merged files are saved."
    )
    parser.add_argument(
        "-f", "--format", dest="output_format", choices=["mp3", "wav"], default="mp3",
        help="Output format of the merged file.",
    )
    parser.add_argument("-b", "--bitrate", default="320k", help="Bitrate for MP3 export.")
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help="Separation model (htdemucs, htdemucs_ft, BS-RoFormer, ...).",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="Compute device. 'auto' detects GPU automatically, 'cuda' forces NVIDIA GPU, 'cpu' forces software mode.",
    )
    parser.add_argument(
        "--segment-size", type=int, default=None,
        help="Segment size in seconds for Demucs model (default: 256). Lower values are faster but may reduce quality.",
    )
    parser.add_argument(
        "--shifts", type=int, default=None,
        help="Number of shifts for Demucs model (default: 2). Lower values are faster but may reduce quality.",
    )
    parser.add_argument(
        "--overlap", type=float, default=None,
        help="Overlap between segments for Demucs model (default: 0.25). Lower values are faster but may reduce quality.",
    )
    parser.add_argument(
        "--preset", choices=["fast", "balanced", "quality"], default=None,
        help="Performance preset (fast/balanced/quality). Overrides individual segment-size/shifts/overlap settings.",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary downloads/stems instead of cleaning them up.",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Folder where downloaded audio is cached for reuse "
        "(default: download_cache/ next to this script, or $VOCAL_DEF_CACHE_DIR).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Do not use the download cache; always download from the network.",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Delete the download cache and exit.",
    )
    parser.add_argument("--temp-dir", default=None, help="Base directory used for temporary files.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose (debug) logging.")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root = logging.getLogger("vocal_def")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def sanitize_filename(name):
    cleaned = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    return cleaned or "output"


def apply_device_override(device):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import torch
            if hasattr(torch.backends, "mps"):
                torch.backends.mps.is_available = lambda: False
        except Exception:
            pass
    elif device == "cuda":
        # Force CUDA usage if available
        try:
            import torch
            if torch.cuda.is_available():
                # Enable CUDA memory optimizations
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                log.info("CUDA acceleration enabled")
            else:
                log.warning("CUDA requested but not available, falling back to CPU")
        except Exception:
            log.warning("PyTorch not available for CUDA check")
    elif device == "auto":
        # Auto-detect and log available devices
        try:
            import torch
            if torch.cuda.is_available():
                log.info("CUDA detected and will be used for acceleration")
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                log.info("Apple Silicon (MPS) detected and will be used for acceleration")
            else:
                log.info("No GPU detected, using CPU")
        except Exception:
            log.info("Unable to detect GPU availability, using default")


def resolve_model_filename(name):
    name = name.strip()
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    if name.endswith((".yaml", ".ckpt", ".pth")):
        return name
    return name


def collect_inputs(raw_inputs):
    items = []
    for raw in raw_inputs:
        if URL_RE.match(raw):
            items.append({"kind": "url", "source": raw})
            continue
        path = Path(raw)
        if not path.exists():
            log.warning("Input not found, skipping: %s", raw)
            continue
        if path.is_dir():
            found = sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not found:
                log.warning("No supported audio/video files in folder: %s", path)
            for f in found:
                items.append({"kind": "file", "source": str(f)})
        elif path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                items.append({"kind": "file", "source": str(path)})
            else:
                log.warning("Unsupported file type, skipping: %s", path)
    return items


# --------------------------------------------------------------------------- #
# Downloading (yt-dlp)
# --------------------------------------------------------------------------- #

def _youtube_id(url):
    patterns = (
        r"(?:youtube\.com/(?:watch\?[^#]*v=|shorts/|embed/|live/|v/))([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def url_cache_key(url):
    """Stable cache key for a URL: YouTube video ID when known, else URL hash."""
    vid = _youtube_id(url)
    return vid if vid else hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def cached_download_path(url, cache_dir=None):
    return Path(cache_dir or DEFAULT_CACHE_DIR) / f"{url_cache_key(url)}.wav"


def _read_title_cache(cache_dir, key):
    path = Path(cache_dir) / f"{key}.title"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _write_title_cache(cache_dir, key, title):
    try:
        (Path(cache_dir) / f"{key}.title").write_text(title, encoding="utf-8")
    except OSError:
        pass


def clear_cache(cache_dir=None):
    """Delete all cached downloads, returning how many files were removed."""
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    removed = 0
    if cache_dir.exists():
        for path in cache_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def cache_info(cache_dir=None):
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    count = 0
    size = 0
    if cache_dir.exists():
        for path in cache_dir.iterdir():
            if path.is_file():
                count += 1
                size += path.stat().st_size
    return {"count": count, "size_bytes": size, "dir": str(cache_dir)}


MIN_FREE_BYTES = int(os.environ.get("VOCAL_DEF_MIN_FREE_MB", "1024")) * 1024 * 1024


def free_space(path):
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def check_disk_space(paths, minimum=MIN_FREE_BYTES):
    """Raise RuntimeError if any path has less than `minimum` free bytes."""
    for raw in paths:
        path = Path(raw)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        free = free_space(path)
        if free is not None and free < minimum:
            raise RuntimeError(
                "Not enough free disk space on "
                f"'{path}': {free // (1024 * 1024)} MB free, need at least "
                f"{minimum // (1024 * 1024)} MB. Delete old outputs or clear the "
                "download cache, then retry."
            )


def _is_valid_wav(path):
    """Cheap check that a WAV file is complete and not truncated.

    Returns False for empty/garbage files or files whose RIFF header declares
    more bytes than actually exist (the signature of a write that was cut short
    by a full disk).
    """
    try:
        size = path.stat().st_size
        if size < 12:
            return False
        with open(path, "rb") as fh:
            head = fh.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return False
        declared = struct.unpack("<I", head[4:8])[0] + 8
        return declared <= size
    except (OSError, struct.error, ValueError):
        return False


def download_audio(url, out_dir=None, report=None, cache_dir=None):
    """Download the best audio stream and convert it to WAV via ffmpeg.

    When cache_dir is set, the converted WAV is saved there under a stable key
    derived from the URL. Repeating the same URL reuses the cached file instead
    of downloading it again. When cache_dir is None, downloads go to out_dir
    (used for one-off / --no-cache runs).
    """
    import yt_dlp

    key = url_cache_key(url) if cache_dir is not None else None
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    if cache_dir is not None:
        cached = cache_dir / f"{key}.wav"
        if cached.exists() and _is_valid_wav(cached):
            if report is not None:
                report(0.04, "converting", "Using cached download...")
            return cached, _read_title_cache(cache_dir, key) or key
        if cached.exists():
            cached.unlink(missing_ok=True)
            log.warning("Removed corrupt cached download: %s", cached)
        cache_dir.mkdir(parents=True, exist_ok=True)

    download_dir = cache_dir if cache_dir is not None else Path(out_dir)
    outtmpl = (
        str(download_dir / f"{key}.%(ext)s")
        if key is not None
        else str(download_dir / "%(title).80B [%(id)s].%(ext)s")
    )

    def progress_hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if report is not None and total:
                report(0.04 + 0.14 * downloaded / total, "downloading", "Downloading...")
        elif d["status"] == "finished":
            if report is not None:
                report(0.18, "converting", "Converting to WAV...")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "continuedl": True,
        "progress_hooks": [progress_hook],
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}
        ],
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "referer": "https://www.youtube.com/",
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
        "socket_timeout": 60,
        "retries": 5,
        "fragment_retries": 5,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as exc:
            raise RuntimeError(f"Download failed for {url}: {exc}") from exc
        if not info:
            raise RuntimeError(f"Nothing downloadable at: {url}")

        title = info.get("title") or info.get("id") or "download"
        base_path = Path(ydl.prepare_filename(info))
        wav_path = base_path.with_suffix(".wav")
        if not wav_path.exists() and key is not None:
            wav_path = download_dir / f"{key}.wav"
        if wav_path.exists():
            if key is not None:
                _write_title_cache(cache_dir, key, title)
            return wav_path, title
        if base_path.exists():
            return base_path, title
        candidates = sorted(
            download_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            if key is not None:
                _write_title_cache(cache_dir, key, title)
            return candidates[0], title
        raise RuntimeError(f"Downloaded file not found for {url}")


# --------------------------------------------------------------------------- #
# Separation engine (audio-separator / demucs)
# --------------------------------------------------------------------------- #

class SeparationEngine:
    """Lazy wrapper around audio_separator.Separator.

    Device selection is auto-detected by audio-separator (CUDA -> MPS -> CPU).
    """

    def __init__(self, model_name=DEFAULT_MODEL, device="auto", model_dir=None, logger=None, 
                 segment_size=None, shifts=None, overlap=None):
        self.model_name = model_name
        self.device = device
        self.logger = logger or log
        self.model_dir = Path(model_dir or (Path(tempfile.gettempdir()) / "audio-separator-models"))
        self.separator = None
        # Performance parameters
        self.segment_size = segment_size
        self.shifts = shifts
        self.overlap = overlap

    def ensure_loaded(self):
        if self.separator is not None:
            return
        apply_device_override(self.device)
        from audio_separator.separator import Separator

        self.model_dir.mkdir(parents=True, exist_ok=True)
        filename = resolve_model_filename(self.model_name)
        self.logger.info("Loading model '%s' (%s) ...", self.model_name, filename)
        
        # Build separator parameters with performance optimizations
        separator_kwargs = {
            "model_file_dir": str(self.model_dir),
            "output_format": "wav",
            "normalization_threshold": 0.9,
            "amplification_threshold": 0.0,
            "log_level": logging.DEBUG if self.logger.isEnabledFor(logging.DEBUG) else logging.INFO,
        }
        
        # Build demucs_params dictionary for architecture-specific parameters
        demucs_params = {}
        
        # Add performance parameters if specified
        if self.segment_size is not None:
            demucs_params["segment_size"] = self.segment_size
            self.logger.info("Using segment_size: %s", self.segment_size)
        if self.shifts is not None:
            demucs_params["shifts"] = self.shifts
            self.logger.info("Using shifts: %s", self.shifts)
        if self.overlap is not None:
            demucs_params["overlap"] = self.overlap
            self.logger.info("Using overlap: %s", self.overlap)
        
        # Detect if GPU is available for additional optimizations
        try:
            import torch
            if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                # Use batch processing for GPU acceleration
                demucs_params["batch_size"] = 4
                self.logger.info("GPU detected - using batch_size: 4 for faster processing")
        except Exception:
            pass
        
        # Add demucs_params to separator_kwargs if any parameters were set
        if demucs_params:
            separator_kwargs["demucs_params"] = demucs_params
        
        self.separator = Separator(**separator_kwargs)
        self.separator.load_model(model_filename=filename)

    def separate(self, input_path, stems_dir):
        self.ensure_loaded()
        stems_dir = Path(stems_dir)
        stems_dir.mkdir(parents=True, exist_ok=True)
        inner = getattr(self.separator, "model_instance", None)
        if inner is not None:
            inner.output_dir = str(stems_dir)
        else:
            self.separator.output_dir = str(stems_dir)
        self.logger.info("Separating: %s", input_path)
        return self.separator.separate(str(input_path))


# --------------------------------------------------------------------------- #
# Stems / mixing
# --------------------------------------------------------------------------- #

def _stem_matches(name, stem):
    return (
        name == stem
        or name.endswith(f"_{stem}")
        or f"({stem})" in name
        or f"_{stem}_" in name
    )


def find_stems(stems_dir):
    vocals = None
    drums = None
    for f in sorted(Path(stems_dir).rglob("*")):
        if f.suffix.lower() != ".wav" or not f.is_file():
            continue
        name = f.stem.lower()
        if vocals is None and _stem_matches(name, "vocals"):
            vocals = f
        elif drums is None and (
            _stem_matches(name, "drums")
            or _stem_matches(name, "def")
            or _stem_matches(name, "percussion")
        ):
            drums = f
    return vocals, drums


def mix_stems(vocals_path, drums_path, output_path, output_format="mp3", bitrate="320k"):
    from pydub import AudioSegment

    try:
        vocals = AudioSegment.from_file(str(vocals_path))
        drums = AudioSegment.from_file(str(drums_path))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read stems '{Path(vocals_path).name}' / '{Path(drums_path).name}': {exc}. "
            "The stem files may be corrupt or truncated, usually because the disk ran out "
            "of space. Free up disk space and retry."
        ) from exc

    vocals = vocals.set_channels(2).normalize(headroom=9.0)
    drums = drums.set_channels(2).normalize(headroom=9.0)

    target = max(len(vocals), len(drums))
    if len(vocals) < target:
        vocals += AudioSegment.silent(duration=target - len(vocals))
    if len(drums) < target:
        drums += AudioSegment.silent(duration=target - len(drums))

    mixed = vocals.overlay(drums)
    mixed = mixed.normalize(headroom=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "wav":
        mixed.export(str(output_path), format="wav", parameters=["-ar", "44100"])
    else:
        mixed.export(str(output_path), format="mp3", bitrate=bitrate)


# --------------------------------------------------------------------------- #
# Input preparation
# --------------------------------------------------------------------------- #

def prepare_local_input(src, dst):
    from pydub import AudioSegment

    try:
        audio = AudioSegment.from_file(str(src))
    except Exception as exc:
        raise RuntimeError(f"Cannot decode {src}: {exc}") from exc
    if audio.channels > 2:
        audio = audio.set_channels(2)
    audio.export(str(dst), format="wav", parameters=["-ar", "44100"])
    return dst


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def process_item(item, index, engine, options, temp_root, report=None):
    """Process a single input item. report(frac, stage, message, indeterminate)."""

    def rep(frac, stage, message, indeterminate=False):
        if report is not None:
            report(frac, stage, message, indeterminate)

    item_dir = temp_root / f"item_{index:03d}"
    item_dir.mkdir(parents=True)

    if item["kind"] == "url":
        dl_dir = item_dir / "download"
        dl_dir.mkdir(parents=True)
        rep(0.02, "downloading", f"Downloading: {item['source']}")
        audio_path, title = download_audio(
            item["source"], dl_dir, report=rep, cache_dir=options.get("cache_dir")
        )
        label = sanitize_filename(title or Path(audio_path).stem)
        working_audio = audio_path
        rep(0.18, "converting", f"Downloaded: {label}")
    else:
        src = Path(item["source"])
        label = sanitize_filename(src.stem)
        conv_dir = item_dir / "input"
        conv_dir.mkdir(parents=True)
        rep(0.02, "converting", f"Decoding: {src.name}")
        working_audio = prepare_local_input(src, conv_dir / f"{label}.wav")
        rep(0.18, "converting", f"Ready: {label}")

    stems_dir = item_dir / "stems"
    stems_dir.mkdir(parents=True)
    working_path = Path(working_audio)
    if working_path.suffix.lower() == ".wav" and not _is_valid_wav(working_path):
        if options.get("cache_dir") is not None:
            working_path.unlink(missing_ok=True)
            log.warning("Removed corrupt cached download: %s", working_path)
        raise RuntimeError(
            f"Audio is corrupt or truncated: {working_path.name}. "
            "This is usually caused by a full disk; free up space and retry."
        )
    rep(0.20, "separating", f"Separating: {label}", indeterminate=True)
    engine.separate(working_audio, stems_dir)
    rep(0.85, "mixing", f"Merging Vocals + Drums: {label}")

    vocals, drums = find_stems(stems_dir)
    if vocals is None or drums is None:
        raise RuntimeError(
            f"Stem extraction incomplete for '{label}' "
            f"(vocals={'found' if vocals else 'missing'}, "
            f"drums={'found' if drums else 'missing'})"
        )
    for stem in (vocals, drums):
        if not _is_valid_wav(stem):
            raise RuntimeError(
                f"Stem '{stem.name}' is corrupt or truncated. "
                "This is usually caused by a full disk; free up space and retry."
            )

    out_name = f"{label}_Vocals_and_Def.{options['output_format']}"
    out_path = Path(options["output_dir"]) / out_name
    mix_stems(vocals, drums, out_path, options["output_format"], options["bitrate"])
    rep(1.0, "saving", f"Saved: {out_name}")
    return out_path


def process_items(items, engine, options, temp_root, report=None):
    """Process all items sequentially.

    One failing item does not abort the batch; errors are logged and the next
    item is processed. Returns (results, failures) where failures is a list of
    (source, exception) tuples.
    """
    results = []
    failures = []
    total = len(items)
    for index, item in enumerate(items):
        base = index / total

        def item_report(frac, stage, message, indeterminate=False):
            if report is not None:
                report(base + frac / total, stage, message, indeterminate)

        try:
            out = process_item(item, index, engine, options, temp_root, report=item_report)
            results.append(out)
        except Exception as exc:
            failures.append((item["source"], exc))
            log.error("Failed (%s): %s", item["source"], exc)
            if report is not None:
                report(base + 0.99, "error", f"Failed: {exc}")
    return results, failures


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose)

    if shutil.which("ffmpeg") is None:
        log.error(
            "ffmpeg was not found on PATH. Install it (e.g. 'brew install ffmpeg'). "
            "It is required for downloads, decoding and MP3 export."
        )
        return 1

    if args.clear_cache:
        removed = clear_cache(args.cache_dir)
        log.info("Removed %d cached download file(s).", removed)
        return 0

    if not args.inputs:
        log.error("No inputs provided. Pass URLs, files or a folder (or use --clear-cache).")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = collect_inputs(args.inputs)
    if not items:
        log.error("No valid inputs found. Check your URLs and paths.")
        return 1
    log.info("Queued %d input(s).", len(items))

    temp_root = Path(tempfile.mkdtemp(prefix="vocal_def_", dir=args.temp_dir))
    log.info("Temporary working directory: %s", temp_root)

    try:
        check_disk_space([temp_root, out_dir])
    except RuntimeError as exc:
        log.error("%s", exc)
        shutil.rmtree(temp_root, ignore_errors=True)
        return 1

    # Determine performance parameters
    segment_size = args.segment_size
    shifts = args.shifts
    overlap = args.overlap
    
    if args.preset:
        preset = PERFORMANCE_PRESETS[args.preset]
        segment_size = preset["segment_size"]
        shifts = preset["shifts"]
        overlap = preset["overlap"]
        log.info("Using performance preset '%s': segment_size=%s, shifts=%s, overlap=%s", 
                 args.preset, segment_size, shifts, overlap)
    
    engine = SeparationEngine(
        model_name=args.model, 
        device=args.device,
        segment_size=segment_size,
        shifts=shifts,
        overlap=overlap
    )
    options = {
        "output_dir": str(out_dir),
        "output_format": args.output_format,
        "bitrate": args.bitrate,
        "cache_dir": None if args.no_cache else Path(args.cache_dir or DEFAULT_CACHE_DIR),
    }
    if options["cache_dir"] is not None:
        options["cache_dir"].mkdir(parents=True, exist_ok=True)
        log.info("Download cache: %s", options["cache_dir"])

    bar = tqdm(total=100, desc="Preparing", unit="%")

    def report(frac, stage, message, indeterminate=False):
        bar.n = int(frac * 100)
        bar.set_description(message)
        bar.refresh()

    results = []
    failures = []
    try:
        results, failures = process_items(items, engine, options, temp_root, report=report)
    except Exception as exc:
        failures.append(("pipeline", exc))
        log.error("Pipeline failed: %s", exc)
        if args.verbose:
            log.exception("Traceback")
    finally:
        bar.close()
        if args.keep_temp:
            log.info("Keeping temporary files in %s", temp_root)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
            log.info("Cleaned up temporary files.")

    log.info("Finished: %d succeeded, %d failed.", len(results), len(failures))
    for r in results:
        log.info("  -> %s", r)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
