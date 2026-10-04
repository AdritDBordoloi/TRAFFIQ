"""
Traffic signal simulation and state machine for TRAFFIQ.
Features automatic timer-based state transitions between GREEN and RED,
manual override toggles, and elapsed duration calculations.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional, Tuple, Union


class SignalState(str, Enum):
    """Enumeration of traffic signal illumination states."""

    GREEN = "GREEN"
    RED = "RED"

    def __str__(self) -> str:
        return self.value


class TrafficSignal:
    """
    Traffic signal state machine controller.
    Automatically cycles between GREEN and RED based on configured cycle duration,
    and supports keyboard/manual state overrides.
    """

    def __init__(
        self,
        cycle_duration: float = 8.0,
        initial_state: Union[SignalState, str] = SignalState.GREEN,
    ) -> None:
        """
        Initialize TrafficSignal controller.

        Args:
            cycle_duration: Duration in seconds before auto-toggling state (default 8.0s).
            initial_state: Starting signal state ('GREEN' or 'RED').
        """
        self.cycle_duration = float(cycle_duration)
        if isinstance(initial_state, SignalState):
            self.state: str = initial_state.value
        else:
            self.state = str(initial_state).strip().upper()
        self.last_switch: float = time.time()

    def update(self, current_time: Optional[float] = None) -> str:
        """
        Inspect timer and transition signal state if cycle duration has elapsed.

        Args:
            current_time: Optional epoch timestamp (defaults to time.time()).

        Returns:
            Current signal state string ('GREEN' or 'RED').
        """
        now = time.time() if current_time is None else float(current_time)
        if now - self.last_switch >= self.cycle_duration:
            self.toggle(now)
        return self.state

    def toggle(self, current_time: Optional[float] = None) -> str:
        """
        Manually flip signal state ('GREEN' <-> 'RED') and reset cycle timer.

        Args:
            current_time: Optional epoch timestamp.

        Returns:
            The newly active signal state.
        """
        now = time.time() if current_time is None else float(current_time)
        self.state = "RED" if self.state == "GREEN" else "GREEN"
        self.last_switch = now
        return self.state

    def elapsed_time(self, current_time: Optional[float] = None) -> float:
        """
        Calculate elapsed seconds since the most recent signal state transition.

        Args:
            current_time: Optional epoch timestamp.

        Returns:
            Elapsed seconds as float.
        """
        now = time.time() if current_time is None else float(current_time)
        return max(0.0, now - self.last_switch)

    def time_remaining(self, current_time: Optional[float] = None) -> float:
        """Calculate remaining seconds before next automatic state transition."""
        return max(0.0, self.cycle_duration - self.elapsed_time(current_time))

    def is_red(self) -> bool:
        """Return True if current state is RED."""
        return self.state == "RED"

    def is_green(self) -> bool:
        """Return True if current state is GREEN."""
        return self.state == "GREEN"

    @property
    def bgr_color(self) -> Tuple[int, int, int]:
        """Return BGR color tuple for rendering overlays."""
        return (0, 0, 255) if self.is_red() else (0, 255, 0)


# Backward-compatible alias matching test expectations
TrafficSignalController = TrafficSignal
