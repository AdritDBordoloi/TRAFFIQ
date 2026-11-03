"""
TRAFFIQ Web Dashboard Flask Server & REST API.
Provides video streaming (/video_feed) and polling endpoints for dashboard telemetry.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Generator, List, Optional
from flask import Flask, Response, jsonify, request, send_from_directory

from traffiq.database.connection import get_connection, register_sqlite_adapters
from traffiq.web.engine import DashboardEngine, evidence_url

MAX_CHALLAN_LIMIT = 500
MAX_DEMO_EMERGENCY_SECONDS = 120.0


def create_app(
    engine: Optional[DashboardEngine] = None,
    frontend_dir: Optional[str] = None,
    violations_dir: str = "violations",
    db_path: str = "traffiq_enforcement.db",
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

    # Store references in app context
    app.engine = engine  # type: ignore[attr-defined]
    app.violations_dir = os.path.realpath(violations_dir)  # type: ignore[attr-defined]
    app.db_path = db_path  # type: ignore[attr-defined]

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
        while True:
            frame_bytes = eng.get_latest_frame() if eng else b""
            if not frame_bytes:
                # 1x1 empty JPEG if engine not ready
                frame_bytes = b""
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.04)  # ~25 FPS stream delivery

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
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
