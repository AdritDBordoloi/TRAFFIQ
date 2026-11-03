"""
TRAFFIQ Web Dashboard Flask Server & REST API.
Provides video streaming (/video_feed) and polling endpoints for dashboard telemetry.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Generator, List, Optional, Union

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler
from werkzeug.utils import secure_filename

from traffiq.database.connection import get_connection, register_sqlite_adapters
from traffiq.web.engine import DashboardEngine, evidence_url

MAX_CHALLAN_LIMIT = 500
MAX_DEMO_EMERGENCY_SECONDS = 120.0
MAX_UPLOAD_BYTES = 2 * 1024 ** 3
NO_ENGINE_STREAM_DELAY_SECONDS = 0.5
ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"})


class LowLatencyRequestHandler(WSGIRequestHandler):
    """Sends each video frame immediately; TCP's default batching makes a stream stutter."""

    disable_nagle_algorithm = True


def list_videos(uploads_dir: str) -> List[str]:
    """Video files available in the uploads folder, newest first."""
    if not os.path.isdir(uploads_dir):
        return []
    names = [
        name for name in os.listdir(uploads_dir)
        if os.path.splitext(name)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
        and os.path.isfile(os.path.join(uploads_dir, name))
    ]
    return sorted(names, key=lambda n: os.path.getmtime(os.path.join(uploads_dir, n)), reverse=True)


def is_readable_video(path: str) -> bool:
    """True when OpenCV can decode at least one frame from the file."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok) and frame is not None
    finally:
        cap.release()


def unique_upload_path(uploads_dir: str, filename: str) -> str:
    """Pick a sanitised, unused path so an upload can never overwrite or escape the uploads folder."""
    stem, ext = os.path.splitext(secure_filename(filename))
    # secure_filename drops non-ASCII characters, which can leave no usable name
    if not ext:
        stem, ext = "", os.path.splitext(filename)[1].lower()
    stem = stem or "video"
    candidate = os.path.join(uploads_dir, stem + ext)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(uploads_dir, f"{stem}_{counter}{ext}")
        counter += 1
    return candidate


def create_app(
    engine: Optional[DashboardEngine] = None,
    frontend_dir: Optional[str] = None,
    violations_dir: str = "violations",
    db_path: str = "traffiq_enforcement.db",
    uploads_dir: str = "uploads",
) -> Flask:
    """
    Application factory creating and configuring the Flask dashboard backend.
    """
    register_sqlite_adapters()

    if frontend_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        frontend_dir = os.path.join(base_dir, "frontend")

    app = Flask(__name__, static_folder=frontend_dir)
    app.json.sort_keys = False
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    # Store references in app context
    app.engine = engine  # type: ignore[attr-defined]
    app.violations_dir = os.path.realpath(violations_dir)  # type: ignore[attr-defined]
    app.db_path = db_path  # type: ignore[attr-defined]
    app.uploads_dir = os.path.realpath(uploads_dir)  # type: ignore[attr-defined]

    @app.errorhandler(413)
    def upload_too_large(_error):
        limit_gb = MAX_UPLOAD_BYTES // 1024 ** 3
        return jsonify({"success": False, "message": f"Video is larger than the {limit_gb} GB limit."}), 413

    # -------------------------------------------------------------------------
    # Static & Evidence Routes
    # -------------------------------------------------------------------------
    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/<path:path>")
    def static_files(path: str):
        if os.path.exists(os.path.join(frontend_dir, path)):
            return send_from_directory(frontend_dir, path)
        return send_from_directory(frontend_dir, "index.html")

    @app.route("/evidence/<path:filename>")
    def serve_evidence(filename: str):
        return send_from_directory(app.violations_dir, filename)

    # -------------------------------------------------------------------------
    # MJPEG Video Streaming
    # -------------------------------------------------------------------------
    def generate_mjpeg_frames() -> Generator[bytes, None, None]:
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        seq = 0
        while True:
            if eng is None:
                frame_bytes = b""
                time.sleep(NO_ENGINE_STREAM_DELAY_SECONDS)
            else:
                # Sends each frame as soon as it exists; a slow viewer simply gets the newest one
                seq, frame_bytes = eng.wait_for_frame(seq)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" + frame_bytes + b"\r\n"
            )

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    # -------------------------------------------------------------------------
    # REST API Telemetry Routes
    # -------------------------------------------------------------------------
    @app.route("/api/state", methods=["GET"])
    def get_state():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        if eng is not None:
            return jsonify(eng.get_state())
        return jsonify({
            "stats": {"vehicles": 0, "violations": 0, "emergencies": 0, "total_fines": 0, "fps": 0.0},
            "signal": {"state": "GREEN", "time_remaining": 8.0, "is_emergency_locked": False},
            "emergency_active": False,
        })

    @app.route("/api/events", methods=["GET"])
    def get_events():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        try:
            since_id = int(request.args.get("since", 0))
        except (ValueError, TypeError):
            since_id = 0

        if eng is not None:
            return jsonify(eng.get_events(since_id=since_id))
        return jsonify([])

    @app.route("/api/challans", methods=["GET"])
    def get_challans():
        try:
            limit = int(request.args.get("limit", 20))
        except (ValueError, TypeError):
            limit = 20
        limit = max(1, min(limit, MAX_CHALLAN_LIMIT))

        challans: List[Dict[str, Any]] = []
        conn = None
        try:
            conn = get_connection(app.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    c.challan_id,
                    c.timestamp,
                    c.track_id,
                    c.plate_number,
                    c.owner_name,
                    c.vehicle_model,
                    c.violation_type,
                    c.fine_amount,
                    c.full_evidence_path,
                    c.crop_evidence_path,
                    c.evidence_path
                FROM challan_records c
                ORDER BY c.challan_id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

            for r in rows:
                full_path = r["full_evidence_path"] or r["evidence_path"] or ""
                crop_path = r["crop_evidence_path"] or ""


                challans.append({
                    "challan_id": int(r["challan_id"]),
                    "timestamp": str(r["timestamp"]),
                    "track_id": int(r["track_id"]),
                    "plate_number": str(r["plate_number"]),
                    "owner_name": str(r["owner_name"]),
                    "vehicle_model": str(r["vehicle_model"] or ""),
                    "violation_type": str(r["violation_type"]),
                    "fine_amount": int(r["fine_amount"]),
                    "image_url": evidence_url(full_path, app.violations_dir),
                    "plate_url": evidence_url(crop_path, app.violations_dir),
                })
        except Exception as exc:
            print(f"[WARN] /api/challans query failed: {exc}")
        finally:
            if conn is not None:
                conn.close()

        return jsonify(challans)

    # -------------------------------------------------------------------------
    # Video Source Selection & Upload
    # -------------------------------------------------------------------------
    @app.route("/api/source", methods=["GET"])
    def get_source():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        return jsonify({
            "current": eng.get_source() if eng is not None else None,
            "videos": list_videos(app.uploads_dir),
        })

    @app.route("/api/source", methods=["POST"])
    def set_source():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        if eng is None:
            return jsonify({"success": False, "message": "No engine active"}), 503

        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        source: Union[int, str]
        if "video" in data:
            # Only names from the listing are accepted, so a request cannot reach other files.
            name = data["video"]
            if name not in list_videos(app.uploads_dir):
                return jsonify({"success": False, "message": "Unknown video."}), 404
            source = os.path.join(app.uploads_dir, name)
        elif isinstance(data.get("camera"), int) and not isinstance(data["camera"], bool) and data["camera"] >= 0:
            source = data["camera"]
        else:
            return jsonify({"success": False, "message": "Provide 'video' or 'camera'."}), 400
        return jsonify({"success": True, "source": eng.set_source(source)})

    @app.route("/api/upload", methods=["POST"])
    def upload_video():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        if eng is None:
            return jsonify({"success": False, "message": "No engine active"}), 503

        file = request.files.get("video")
        if file is None or not file.filename:
            return jsonify({"success": False, "message": "No video file provided."}), 400
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
            return jsonify({"success": False, "message": f"Unsupported file type. Use one of: {allowed}."}), 400

        os.makedirs(app.uploads_dir, exist_ok=True)
        path = unique_upload_path(app.uploads_dir, file.filename)
        file.save(path)
        if not is_readable_video(path):
            os.remove(path)
            return jsonify({"success": False, "message": "That file could not be read as a video."}), 400

        return jsonify({"success": True, "video": os.path.basename(path), "source": eng.set_source(path)})

    @app.route("/api/signal/toggle", methods=["POST"])
    def toggle_signal():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        if eng is not None:
            return jsonify(eng.toggle_signal())
        return jsonify({"success": False, "message": "No engine active"})

    @app.route("/api/demo/violation", methods=["POST"])
    def demo_violation():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        if eng is not None:
            ev = eng.trigger_demo_violation()
            return jsonify({"success": True, "event": ev})
        return jsonify({"success": False, "message": "No engine active"})

    @app.route("/api/demo/ambulance", methods=["POST"])
    def demo_ambulance():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        data = request.get_json(silent=True) or {}
        try:
            duration = float(data.get("duration", 15.0))
        except (ValueError, TypeError):
            duration = 15.0
        if not math.isfinite(duration):
            duration = 15.0
        duration = max(1.0, min(duration, MAX_DEMO_EMERGENCY_SECONDS))
        if eng is not None:
            ev = eng.trigger_demo_ambulance(duration=duration)
            return jsonify({"success": True, "event": ev})
        return jsonify({"success": False, "message": "No engine active"})

    @app.route("/api/reset", methods=["POST"])
    def reset_stats():
        eng: Optional[DashboardEngine] = getattr(app, "engine", None)
        if eng is not None:
            return jsonify(eng.reset_stats())
        return jsonify({"success": True})

    return app
