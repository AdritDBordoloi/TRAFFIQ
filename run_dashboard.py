#!/usr/bin/env python3
"""
run_dashboard.py
================
Unified launcher for the TRAFFIQ Exhibition Web Dashboard.
Starts the background vision and priority engine, mounts the Flask REST backend,
and serves the single-screen dashboard at http://localhost:8000.
"""

from __future__ import annotations

import argparse
import sys

from traffiq.web.engine import DashboardEngine
from traffiq.web.server import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="TRAFFIQ Exhibition Web Dashboard")
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="Video capture source: camera index (0) or video file path (default: 0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="HTTP server port (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP server host binding (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--device", "-d",
        default="0",
        help="Target inference compute device ('0' for CUDA GPU, 'cpu') (default: 0)",
    )
    parser.add_argument(
        "--signal-duration",
        type=float,
        default=8.0,
        help="Traffic signal cycle duration in seconds (default: 8.0)",
    )
    parser.add_argument(
        "--fine",
        type=int,
        default=1000,
        help="Fine penalty in INR for red-light infractions (default: 1000)",
    )
    parser.add_argument(
        "--auto-demo",
        action="store_true",
        help="Automatically inject simulated violations on idle timeout (60s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("      TRAFFIQ SMART TRAFFIC ENFORCEMENT EXHIBITION DASHBOARD     ")
    print("=" * 65)
    print(f" Web Dashboard URL : http://localhost:{args.port}")
    print(f" Video Source Feed : {args.source}")
    print(f" Compute Device    : {args.device}")
    print(f" Auto-Demo Mode    : {'Enabled (60s idle trigger)' if args.auto_demo else 'Disabled'}")
    print("=" * 65 + "\n")

    # 1. Initialize & Start Vision Streaming Engine
    engine = DashboardEngine(
        source=args.source,
        device=args.device,
        signal_duration=args.signal_duration,
        fine_amount=args.fine,
        auto_demo=args.auto_demo,
        start_thread=True,
    )

    # 2. Build Flask Web Application
    app = create_app(engine=engine)

    try:
        # 3. Serve HTTP & Streaming Endpoints
        app.run(
            host=args.host,
            port=args.port,
            threaded=True,
            debug=False,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Stopping TRAFFIQ Dashboard Server...")
    finally:
        engine.stop()
        print("[INFO] Engine stopped cleanly.")


if __name__ == "__main__":
    sys.exit(main() or 0)
