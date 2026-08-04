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
import logging
import os
import re
import shutil
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

MODEL_ALIASES = {
    "htdemucs": "htdemucs.yaml",
    "htdemucs_ft": "htdemucs_ft.yaml",
    "htdemucs_6s": "htdemucs_6s.yaml",
    "BS-RoFormer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "bs-roformer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
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
        nargs="+",
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
        help="Compute device. The engine auto-detects CUDA; use 'cpu' to force software mode.",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary downloads/stems instead of cleaning them up.",
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

def download_audio(url, out_dir, report=None):
    """Download the best audio stream and convert it to WAV via ffmpeg."""
    import yt_dlp

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
        "outtmpl": str(out_dir / "%(title).80B [%(id)s].%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "continuedl": True,
        "progress_hooks": [progress_hook],
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}
        ],
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
        if wav_path.exists():
            return wav_path, title
        if base_path.exists():
            return base_path, title
        candidates = sorted(
            out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            return candidates[0], title
        raise RuntimeError(f"Downloaded file not found for {url}")


# --------------------------------------------------------------------------- #
# Separation engine (audio-separator / demucs)
# --------------------------------------------------------------------------- #

class SeparationEngine:
    """Lazy wrapper around audio_separator.Separator.

    Device selection is auto-detected by audio-separator (CUDA -> MPS -> CPU).
    """

    def __init__(self, model_name=DEFAULT_MODEL, device="auto", model_dir=None, logger=None):
        self.model_name = model_name
        self.device = device
        self.logger = logger or log
        self.model_dir = Path(model_dir or (Path(tempfile.gettempdir()) / "audio-separator-models"))
        self.separator = None

    def ensure_loaded(self):
        if self.separator is not None:
            return
        apply_device_override(self.device)
        from audio_separator.separator import Separator

        self.model_dir.mkdir(parents=True, exist_ok=True)
        filename = resolve_model_filename(self.model_name)
        self.logger.info("Loading model '%s' (%s) ...", self.model_name, filename)
        self.separator = Separator(
            model_file_dir=str(self.model_dir),
            output_format="wav",
            normalization_threshold=0.9,
            amplification_threshold=0.0,
            log_level=logging.DEBUG if self.logger.isEnabledFor(logging.DEBUG) else logging.INFO,
        )
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

    vocals = AudioSegment.from_file(str(vocals_path))
    drums = AudioSegment.from_file(str(drums_path))

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
        audio_path, title = download_audio(item["source"], dl_dir, report=rep)
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = collect_inputs(args.inputs)
    if not items:
        log.error("No valid inputs found. Check your URLs and paths.")
        return 1
    log.info("Queued %d input(s).", len(items))

    temp_root = Path(tempfile.mkdtemp(prefix="vocal_def_", dir=args.temp_dir))
    log.info("Temporary working directory: %s", temp_root)

    engine = SeparationEngine(model_name=args.model, device=args.device)
    options = {
        "output_dir": str(out_dir),
        "output_format": args.output_format,
        "bitrate": args.bitrate,
    }

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
