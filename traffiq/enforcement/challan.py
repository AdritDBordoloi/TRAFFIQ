"""
E-Challan generation, database logging, evidence storage, and HUD banner alerts for TRAFFIQ.
Coordinates vehicle registry lookup, persistent SQLite record insertion with strict integer casting,
violation snapshot archiving, and real-time on-screen banner notifications.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from traffiq.database.repository import DatabaseManager


class ChallanIssuer:
    """
    Issues official e-challan violations, records entries into the SQLite database,
    saves vehicle evidence snapshots to disk, and manages real-time HUD notification banners.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        db_path: str = "traffiq_enforcement.db",
        output_dir: str = "violations",
        fine_amount: int = 1000,
        alert_duration: float = 5.0,
    ) -> None:
        """
        Initialize ChallanIssuer.

        Args:
            db_manager: Optional existing DatabaseManager instance.
            db_path: Path to SQLite database if db_manager is not supplied.
            output_dir: Destination folder for violation JPEG captures.
            fine_amount: Standard fine penalty amount in INR.
            alert_duration: Seconds to display on-screen HUD banner following an infraction.
        """
        self.db_manager = db_manager or DatabaseManager(db_path=db_path)
        self.output_dir = str(output_dir)
        self.default_fine = int(fine_amount)
        self.alert_duration = float(alert_duration)

        os.makedirs(self.output_dir, exist_ok=True)

        # On-screen HUD alert state
        self.recent_alert: Optional[Dict[str, Any]] = None
        self.alert_display_timer: float = 0.0

    def issue_challan(
        self,
        track_id: Union[int, np.integer, Any],
        vehicle_crop: Optional[np.ndarray],
        detected_plate: str = "UNKNOWN",
        violation_type: str = "Red Light Jump",
        fine_amount: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Issue an automated e-challan for a detected violation.
        Guarantees strict native Python int casting for track_id to eliminate SQLite BLOB corruption.

        Args:
            track_id: Vehicle tracking identifier (cast strictly to native int).
            vehicle_crop: Cropped vehicle image array representing visual evidence.
            detected_plate: Cleaned license plate alphanumeric string.
            violation_type: Description of the infraction.
            fine_amount: Penalty fine in local currency.
            timestamp: Formatted violation timestamp.

        Returns:
            Dictionary containing issued challan metadata and primary key ID.
        """
        # Strict native Python integer casting
        int_track_id = int(track_id)
        effective_fine = int(fine_amount) if fine_amount is not None else self.default_fine
        clean_plate = str(detected_plate).strip().upper()

        # Generate timestamps
        now_dt = datetime.now()
        ts_str = timestamp or now_dt.strftime("%Y-%m-%d %H:%M:%S")
        ts_file = now_dt.strftime("%Y%m%d_%H%M%S")

        # Save evidence image snapshot
        evidence_filename = f"violation_ID{int_track_id}_{ts_file}.jpg"
        evidence_path = os.path.join(self.output_dir, evidence_filename)

        if vehicle_crop is not None and vehicle_crop.size > 0:
            cv2.imwrite(evidence_path, vehicle_crop)
        else:
            evidence_path = ""

        # Query vehicle registry for owner details
        owner_details = self.db_manager.lookup_vehicle(clean_plate)
        if owner_details:
            owner_name = owner_details.get("owner_name") or owner_details.get("owner", "UNKNOWN")
            phone = owner_details.get("contact_number") or owner_details.get("phone", "N/A")
            model = owner_details.get("vehicle_model") or owner_details.get("model", "Unregistered")
        else:
            owner_name = "UNKNOWN / OUT-OF-STATE"
            phone = "N/A"
            model = "Unregistered"

        # Record into database with strict native int
        challan_id = self.db_manager.record_challan(
            track_id=int_track_id,
            plate_number=clean_plate,
            owner_name=owner_name,
            violation_type=str(violation_type),
            fine_amount=effective_fine,
            evidence_path=evidence_path,
            timestamp=ts_str,
        )

        challan_record: Dict[str, Any] = {
            "challan_id": challan_id,
            "timestamp": ts_str,
            "track_id": int_track_id,
            "plate_number": clean_plate,
            "owner_name": owner_name,
            "contact_number": phone,
            "vehicle_model": model,
            "violation_type": str(violation_type),
            "fine_amount": effective_fine,
            "evidence_path": evidence_path,
        }

        # Activate on-screen HUD banner alert
        self.recent_alert = {
            "challan_id": challan_id,
            "track_id": int_track_id,
            "plate": clean_plate,
            "owner": owner_name,
            "fine": effective_fine,
            "issued_at": time.time(),
        }
        self.alert_display_timer = time.time()

        return challan_record

    def draw_hud_banner(
        self,
        frame: np.ndarray,
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draw real-time pop-up notification banner across bottom of frame
        with countdown lifecycle if an alert is currently active.

        Args:
            frame: 3-channel BGR video frame.
            current_time: Optional epoch timestamp.

        Returns:
            Frame with HUD banner rendered if active.
        """
        if self.recent_alert is None:
            return frame

        now = time.time() if current_time is None else float(current_time)
        elapsed = now - self.alert_display_timer

        if elapsed >= self.alert_duration:
            self.recent_alert = None
            return frame

        h, w = frame.shape[:2]
        banner_y = max(0, h - 100)

        # Draw red warning container with white border
        cv2.rectangle(frame, (20, banner_y), (w - 20, h - 20), (0, 0, 180), -1)
        cv2.rectangle(frame, (20, banner_y), (w - 20, h - 20), (255, 255, 255), 2)

        # Banner title
        cv2.putText(
            frame,
            "AUTOMATED E-CHALLAN GENERATED",
            (40, banner_y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # Banner details
        info_text = (
            f"Plate: {self.recent_alert['plate']}  |  "
            f"Owner: {self.recent_alert['owner']}  |  "
            f"Fine: INR {self.recent_alert['fine']}"
        )
        cv2.putText(
            frame,
            info_text,
            (40, banner_y + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 255),
            1,
        )

        return frame

    def draw_telemetry_hud(
        self,
        frame: np.ndarray,
        signal_state: str,
        fps: float,
        total_passed: int,
        violations_count: int,
        line_color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Draw upper telemetry HUD box in top-left corner.

        Args:
            frame: 3-channel BGR video frame.
            signal_state: 'GREEN' or 'RED'.
            fps: Current measured frames per second.
            total_passed: Cumulative traffic count.
            violations_count: Cumulative violations count.
            line_color: BGR border color matching signal state.

        Returns:
            Frame with telemetry HUD.
        """
        col = line_color or ((0, 0, 255) if signal_state == "RED" else (0, 255, 0))

        cv2.rectangle(frame, (10, 10), (370, 115), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (370, 115), col, 2)
        cv2.putText(frame, f"SIGNAL: {signal_state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Total Passed: {total_passed}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"E-Challans Issued: {violations_count}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        return frame
