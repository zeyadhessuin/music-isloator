#!/usr/bin/env python3
"""Web GUI for the Vocals & Def (Drums) separator.

Run:
    python webapp.py                 # http://127.0.0.1:8000
    python webapp.py --port 9000
"""

import argparse
import logging
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from flask import Flask, jsonify, render_template, request, send_from_directory

import separator
from separator import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL,
    cache_info,
    check_disk_space,
    clear_cache,
    process_items,
    sanitize_filename,
    SeparationEngine,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = DEFAULT_CACHE_DIR
TEMP_BASE = Path(tempfile.gettempdir()) / "vocal_def_web"
UPLOAD_DIR = TEMP_BASE / "uploads"
QUEUE = queue.Queue()
JOBS = {}
JOBS_LOCK = threading.Lock()
ENGINE_CACHE = {}
ACTIVE_JOB = {"job": None}
STOP = threading.Event()

app = Flask(__name__, static_folder="static", template_folder="templates")


# --------------------------------------------------------------------------- #
# Logging bridge: capture package logs into the active job
# --------------------------------------------------------------------------- #

class JobLogHandler(logging.Handler):
    def emit(self, record):
        job = ACTIVE_JOB["job"]
        if job is not None:
            job.add_log(f"{record.levelname}: {record.getMessage()}")


def setup_logging():
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    for name in ("vocal_def", "audio_separator"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(JobLogHandler())
        logger.addHandler(stream)


# --------------------------------------------------------------------------- #
# Job model
# --------------------------------------------------------------------------- #

class Job:
    def __init__(self, job_id, items, options):
        self.id = job_id
        self.items = items
        self.options = options
        self.state = "queued"
        self.progress = 0.0
        self.stage = "queued"
        self.message = "Waiting for the worker..."
        self.logs = []
        self.results = []
        self.error = None
        self.lock = threading.Lock()
        self.created = time.time()
        self.started_at = None

    def add_log(self, message):
        with self.lock:
            self.logs.append(message)
            if len(self.logs) > 400:
                self.logs = self.logs[-400:]

    def report(self, frac, stage, message, indeterminate=False):
        with self.lock:
            self.progress = round(frac * 100, 1)
            self.stage = stage
            self.message = message
            if indeterminate:
                self.stage = "separating"

    def to_dict(self, full=False):
        with self.lock:
            data = {
                "id": self.id,
                "state": self.state,
                "progress": self.progress,
                "stage": self.stage,
                "message": self.message,
                "results": self.results,
                "error": self.error,
            }
            if full:
                data["logs"] = self.logs[-200:]
            return data


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def get_engine(model_name, device, segment_size=None, shifts=None, overlap=None):
    key = f"{model_name}|{device}|{segment_size}|{shifts}|{overlap}"
    if key not in ENGINE_CACHE:
        ENGINE_CACHE[key] = SeparationEngine(
            model_name=model_name, device=device, logger=logging.getLogger("vocal_def"),
            segment_size=segment_size, shifts=shifts, overlap=overlap
        )
    return ENGINE_CACHE[key]


def _gather_results(job):
    since = job.started_at or 0.0
    pattern = "*_Vocals_and_Def.*"
    for path in sorted(OUTPUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.stat().st_mtime < since - 2:
            continue
        result = {"name": path.name, "url": "/output/" + quote(path.name)}
        if result not in job.results:
            job.results.append(result)


def run_job(job):
    job.state = "running"
    job.started_at = time.time()
    job.add_log("Job started.")
    try:
        check_disk_space([Path(tempfile.gettempdir()), OUTPUT_DIR, CACHE_DIR])
    except RuntimeError as exc:
        job.state = "error"
        job.error = str(exc)
        job.add_log(f"ERROR: {exc}")
        job.add_log("Free up space (delete old outputs or clear the download cache) and retry.")
        return
    temp_root = Path(tempfile.mkdtemp(prefix="vocal_def_web_"))
    try:
        job.report(0.0, "starting", "Preparing engine...")
        engine = get_engine(
            job.options["model"], 
            job.options.get("device", "auto"),
            segment_size=job.options.get("segment_size"),
            shifts=job.options.get("shifts"),
            overlap=job.options.get("overlap")
        )
        options = {
            "output_dir": str(OUTPUT_DIR),
            "output_format": job.options["output_format"],
            "bitrate": job.options.get("bitrate", "320k"),
            "cache_dir": job.options.get("cache_dir"),
        }
        result_paths, failures = process_items(
            job.items, engine, options, temp_root, report=job.report
        )
        _gather_results(job)
        job.add_log(f"Finished {len(result_paths)} file(s).")
        if failures:
            job.add_log(f"{len(failures)} item(s) failed.")
            if not result_paths:
                raise RuntimeError("All input items failed.")
            job.error = f"{len(failures)} item(s) failed, {len(result_paths)} succeeded."
    except Exception as exc:
        job.state = "error"
        job.error = str(exc)
        job.add_log(f"ERROR: {exc}")
        logging.getLogger("vocal_def").exception("Job %s failed", job.id)
        _gather_results(job)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        for item in job.items:
            if item["kind"] == "file":
                source = Path(item["source"])
                if str(source).startswith(str(UPLOAD_DIR)):
                    source.unlink(missing_ok=True)
        job.add_log("Temporary files cleaned up.")
        if job.state != "error":
            job.state = "done"
            job.add_log("Job finished.")


def worker_loop():
    while not STOP.is_set():
        try:
            job = QUEUE.get(timeout=1.0)
        except queue.Empty:
            continue
        if job is None:
            break
        ACTIVE_JOB["job"] = job
        try:
            run_job(job)
        finally:
            ACTIVE_JOB["job"] = None
            QUEUE.task_done()


def cleanup_stale_temp():
    """Remove leftover job temp dirs from crashed runs to reclaim disk space."""
    base = Path(tempfile.gettempdir())
    removed = 0
    freed = 0
    for path in base.glob("vocal_def_web_*"):
        if not path.is_dir():
            continue
        freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    if removed:
        logging.getLogger("vocal_def").info(
            "Removed %d stale temp dir(s), freeing ~%.1f MB.", removed, freed / (1024 * 1024)
        )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/jobs")
def create_job():
    form = request.form
    urls = [u.strip() for u in form.get("urls", "").replace("\r", "").splitlines() if u.strip()]
    files = request.files.getlist("files")

    output_format = form.get("format", "mp3")
    if output_format not in ("mp3", "wav"):
        output_format = "mp3"
    model = form.get("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    bitrate = form.get("bitrate", "320k").strip() or "320k"
    device = form.get("device", "auto").strip() or "auto"
    
    # Performance parameters
    preset = form.get("preset", "balanced").strip() if form.get("preset") else "balanced"
    segment_size = None
    shifts = None
    overlap = None
    
    if preset in ("fast", "balanced", "quality"):
        from separator import PERFORMANCE_PRESETS
        preset_config = PERFORMANCE_PRESETS[preset]
        segment_size = preset_config["demucs_segment_size"]
        shifts = preset_config["demucs_shifts"]
        overlap = preset_config["demucs_overlap"]
    
    # Allow custom values to override preset if provided
    try:
        if form.get("segment_size"):
            segment_size = int(form.get("segment_size"))
        if form.get("shifts"):
            shifts = int(form.get("shifts"))
        if form.get("overlap"):
            overlap = float(form.get("overlap"))
    except (ValueError, TypeError):
        # Fall back to balanced preset if invalid values
        from separator import PERFORMANCE_PRESETS
        preset_config = PERFORMANCE_PRESETS["balanced"]
        segment_size = preset_config["demucs_segment_size"]
        shifts = preset_config["demucs_shifts"]
        overlap = preset_config["demucs_overlap"]

    items = [{"kind": "url", "source": u} for u in urls]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        base = sanitize_filename(Path(f.filename).stem)
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{base}{ext}"
        f.save(str(dest))
        items.append({"kind": "file", "source": str(dest)})

    if not items:
        return jsonify({"error": "No inputs provided. Add URLs and/or files."}), 400

    job = Job(
        job_id=uuid.uuid4().hex[:8],
        items=items,
        options={
            "output_format": output_format,
            "model": model,
            "bitrate": bitrate,
            "device": device,
            "cache_dir": CACHE_DIR,
            "segment_size": segment_size,
            "shifts": shifts,
            "overlap": overlap,
        },
    )
    with JOBS_LOCK:
        JOBS[job.id] = job
    QUEUE.put(job)
    return jsonify({"job_id": job.id})


@app.get("/api/jobs")
def list_jobs():
    with JOBS_LOCK:
        return jsonify([job.to_dict() for job in reversed(list(JOBS.values()))])


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(job.to_dict(full=True))


@app.get("/output/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.get("/api/cache")
def cache_status():
    info = cache_info(CACHE_DIR)
    info["enabled"] = True
    return jsonify(info)


@app.delete("/api/cache")
def cache_delete():
    removed = clear_cache(CACHE_DIR)
    return jsonify({"removed": removed, **cache_info(CACHE_DIR)})


@app.get("/api/health")
def health():
    from separator import PERFORMANCE_PRESETS
    return jsonify(
        {
            "ok": True, 
            "model_default": DEFAULT_MODEL, 
            "cache": cache_info(CACHE_DIR),
            "presets": PERFORMANCE_PRESETS,
        }
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(description="Vocals & Def separator web GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args(argv)

    setup_logging()
    cleanup_stale_temp()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        logging.getLogger("vocal_def").error(
            "ffmpeg was not found on PATH. Install it (e.g. 'brew install ffmpeg')."
        )
        return 1

    threading.Thread(target=worker_loop, daemon=True, name="worker").start()
    logging.getLogger("vocal_def").info(
        "Web GUI running at http://%s:%d  (output dir: %s)", args.host, args.port, OUTPUT_DIR
    )
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    STOP.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
