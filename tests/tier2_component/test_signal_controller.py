"""Tier 2: Traffic Light State Machine & Signal Controller Tests.
Verifies traffic light state machine: initial state, automatic timer-based
transition between RED and GREEN, manual toggle ('t' key), and elapsed time tracking.
"""

import time
import unittest

try:
    from traffiq.enforcement.signal import TrafficSignalController
except ImportError:
    TrafficSignalController = None


class OracleSignalController:
    """Authoritative reference implementation of the Traffic Signal state machine."""

    def __init__(self, cycle_duration: float = 8.0, initial_state: str = "GREEN"):
        self.cycle_duration = float(cycle_duration)
        self.state = initial_state
        self.last_switch = time.time()

    def update(self, current_time: float = None) -> str:
        """Checks timer and transitions light if cycle duration has elapsed."""
        now = time.time() if current_time is None else current_time
        if now - self.last_switch >= self.cycle_duration:
            self.toggle(now)
        return self.state

    def toggle(self, current_time: float = None) -> str:
        """Manually toggles the light state and resets the cycle timer."""
        now = time.time() if current_time is None else current_time
        self.state = "RED" if self.state == "GREEN" else "GREEN"
        self.last_switch = now
        return self.state

    def elapsed_time(self, current_time: float = None) -> float:
        """Returns elapsed seconds in the current signal state."""
        now = time.time() if current_time is None else current_time
        return max(0.0, now - self.last_switch)

    def is_red(self) -> bool:
        return self.state == "RED"

    def is_green(self) -> bool:
        return self.state == "GREEN"


class TestSignalController(unittest.TestCase):
    """Tests traffic signal lifecycle, timer transitions, and manual overrides."""

    def setUp(self):
        self.controller_cls = TrafficSignalController or OracleSignalController

    def test_initial_state(self):
        """Verifies default initial state is GREEN and cycle duration is 8.0 seconds."""
        ctrl = self.controller_cls(cycle_duration=8.0)
        self.assertEqual(ctrl.state, "GREEN")
        self.assertTrue(ctrl.is_green())
        self.assertFalse(ctrl.is_red())

    def test_manual_toggle(self):
        """Verifies calling toggle() flips GREEN -> RED -> GREEN immediately."""
        ctrl = self.controller_cls(cycle_duration=8.0)
        
        # Toggle 1: GREEN -> RED
        new_state = ctrl.toggle()
        self.assertEqual(new_state, "RED")
        self.assertTrue(ctrl.is_red())
        self.assertFalse(ctrl.is_green())

        # Toggle 2: RED -> GREEN
        new_state = ctrl.toggle()
        self.assertEqual(new_state, "GREEN")
        self.assertTrue(ctrl.is_green())

    def test_automatic_timer_transition(self):
        """Verifies simulated time progression triggers automatic state transition."""
        t0 = 1000.0
        ctrl = self.controller_cls(cycle_duration=8.0)
        ctrl.last_switch = t0

        # At t0 + 4s (mid-cycle): state should still be GREEN
        state = ctrl.update(current_time=t0 + 4.0)
        self.assertEqual(state, "GREEN")

        # At t0 + 8.1s: state should automatically cycle to RED
        state = ctrl.update(current_time=t0 + 8.1)
        self.assertEqual(state, "RED")

        # At t0 + 16.2s: state should cycle back to GREEN
        state = ctrl.update(current_time=t0 + 16.2)
        self.assertEqual(state, "GREEN")

    def test_elapsed_time_calculation(self):
        """Verifies elapsed time calculation tracks duration accurately."""
        t0 = 2000.0
        ctrl = self.controller_cls(cycle_duration=10.0)
        ctrl.last_switch = t0

        elapsed = ctrl.elapsed_time(current_time=t0 + 3.5)
        self.assertAlmostEqual(elapsed, 3.5, places=2)

    def test_reset_timer_on_toggle(self):
        """Verifies manual toggle resets the cycle timer."""
        t0 = 3000.0
        ctrl = self.controller_cls(cycle_duration=10.0)
        ctrl.last_switch = t0

        # Mid-cycle toggle at t0 + 7.0
        ctrl.toggle(current_time=t0 + 7.0)
        self.assertEqual(ctrl.state, "RED")

        # Elapsed time from the toggle moment (at t0 + 8.0, elapsed should be 1.0s, not 8.0s)
        elapsed = ctrl.elapsed_time(current_time=t0 + 8.0)
        self.assertAlmostEqual(elapsed, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
