"""
Centralized configuration management for the TRAFFIQ system.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class TraffiqConfig:
    """Centralized configuration dataclass for TRAFFIQ system parameters."""

    camera_index: Union[int, str] = 0
    frame_width: int = 1280
    frame_height: int = 720
    device: str = "0"
    model_path: str = "yolov8s.pt"
    confidence_threshold: float = 0.25
    tripwire_ratio: float = 0.55
    fine_amount: int = 1000
    signal_duration: float = 8.0
    output_dir: str = "violations"
    db_path: str = "traffiq_enforcement.db"
    no_gui: bool = False
    max_frames: Optional[int] = None
    tracker: str = "bytetrack.yaml"
    imgsz: int = 640

    def __post_init__(self) -> None:
        """Validate and coerce configuration parameters."""
        # Handle camera_index conversion if string is purely numeric
        if isinstance(self.camera_index, str):
            stripped = self.camera_index.strip()
            if stripped.isdigit():
                self.camera_index = int(stripped)
            else:
                self.camera_index = stripped

        self.frame_width = int(self.frame_width)
        self.frame_height = int(self.frame_height)
        self.device = str(self.device).strip()
        self.model_path = str(self.model_path).strip()

        self.confidence_threshold = float(self.confidence_threshold)
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )

        self.tripwire_ratio = float(self.tripwire_ratio)
        if not (0.0 <= self.tripwire_ratio <= 1.0):
            raise ValueError(
                f"tripwire_ratio must be between 0.0 and 1.0, got {self.tripwire_ratio}"
            )

        self.fine_amount = int(self.fine_amount)
        if self.fine_amount < 0:
            raise ValueError(
                f"fine_amount must be non-negative, got {self.fine_amount}"
            )

        self.signal_duration = float(self.signal_duration)
        if self.signal_duration <= 0.0:
            raise ValueError(
                f"signal_duration must be positive, got {self.signal_duration}"
            )

        self.output_dir = str(self.output_dir).strip()
        self.db_path = str(self.db_path).strip()
        self.no_gui = bool(self.no_gui)

        if self.max_frames is not None:
            self.max_frames = int(self.max_frames)
            if self.max_frames <= 0:
                raise ValueError(
                    f"max_frames must be positive integer or None, got {self.max_frames}"
                )

        self.tracker = str(self.tracker).strip()
        self.imgsz = int(self.imgsz)

    def validate(self) -> None:
        """Explicitly validate configuration parameters."""
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def copy_with(self, **kwargs: Any) -> TraffiqConfig:
        """Create a new TraffiqConfig instance with updated values."""
        current = self.to_dict()
        current.update(kwargs)
        return TraffiqConfig(**current)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraffiqConfig:
        """Construct TraffiqConfig from dictionary, filtering out unknown keys."""
        valid_keys = {
            "camera_index",
            "frame_width",
            "frame_height",
            "device",
            "model_path",
            "confidence_threshold",
            "tripwire_ratio",
            "fine_amount",
            "signal_duration",
            "output_dir",
            "db_path",
            "no_gui",
            "max_frames",
            "tracker",
            "imgsz",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}
        return cls(**filtered)

    @classmethod
    def from_args(cls, args: Any) -> TraffiqConfig:
        """
        Construct TraffiqConfig from an argparse.Namespace or any object with attributes.
        Maps CLI flag names (source, width, conf, line_ratio, fine, etc.) to config attributes.
        """
        if hasattr(args, "__dict__"):
            data = vars(args).copy()
        elif isinstance(args, dict):
            data = args.copy()
        else:
            data = {}

        # Handle CLI argument aliases
        alias_map = {
            "source": "camera_index",
            "width": "frame_width",
            "height": "frame_height",
            "model": "model_path",
            "conf": "confidence_threshold",
            "line_ratio": "tripwire_ratio",
            "fine": "fine_amount",
        }

        for cli_key, config_key in alias_map.items():
            if cli_key in data and data[cli_key] is not None:
                if config_key not in data or data[config_key] is None:
                    data[config_key] = data[cli_key]

        return cls.from_dict(data)

    @classmethod
    def from_env(cls, prefix: str = "TRAFFIQ_") -> TraffiqConfig:
        """
        Construct TraffiqConfig with environment variable overrides.
        Checks for variables like TRAFFIQ_CAMERA_INDEX, TRAFFIQ_FINE_AMOUNT, etc.
        """
        env_mappings: Dict[str, tuple[str, type]] = {
            f"{prefix}CAMERA_INDEX": ("camera_index", str),
            f"{prefix}FRAME_WIDTH": ("frame_width", int),
            f"{prefix}FRAME_HEIGHT": ("frame_height", int),
            f"{prefix}DEVICE": ("device", str),
            f"{prefix}MODEL_PATH": ("model_path", str),
            f"{prefix}CONFIDENCE_THRESHOLD": ("confidence_threshold", float),
            f"{prefix}TRIPWIRE_RATIO": ("tripwire_ratio", float),
            f"{prefix}FINE_AMOUNT": ("fine_amount", int),
            f"{prefix}SIGNAL_DURATION": ("signal_duration", float),
            f"{prefix}OUTPUT_DIR": ("output_dir", str),
            f"{prefix}DB_PATH": ("db_path", str),
            f"{prefix}NO_GUI": ("no_gui", lambda val: val.lower() in ("1", "true", "yes")),
            f"{prefix}MAX_FRAMES": ("max_frames", int),
            f"{prefix}TRACKER": ("tracker", str),
            f"{prefix}IMGSZ": ("imgsz", int),
        }

        overrides: Dict[str, Any] = {}
        for env_var, (config_key, converter) in env_mappings.items():
            if env_var in os.environ:
                raw_val = os.environ[env_var]
                try:
                    overrides[config_key] = converter(raw_val)
                except Exception:
                    pass

        return cls.from_dict(overrides)
