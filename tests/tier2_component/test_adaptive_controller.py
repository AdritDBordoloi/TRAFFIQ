"""Tier 2: Adaptive 4-Way Traffic Signal Controller Component Tests.
Verifies dynamic green allocation, starvation prevention, emergency preemption,
safety transition invariants (Yellow / All-Red clearance), and early gap-out.
"""

import json
import unittest

try:
    from traffiq.enforcement.adaptive_signal import (
        AdaptiveIntersectionController,
        Approach,
        PhaseState,
        SignalColor,
    )
except (ImportError, ModuleNotFoundError, AttributeError):
    AdaptiveIntersectionController = None
    Approach = None
    PhaseState = None
    SignalColor = None


class TestAdaptiveIntersectionController(unittest.TestCase):
    """Test suite for Adaptive 4-Way Traffic Signal Controller."""

    def setUp(self):
        if AdaptiveIntersectionController is None:
            self.skipTest('adaptive_signal not yet implemented')
        self.controller = AdaptiveIntersectionController(
            min_green=10.0,
            max_green=60.0,
            yellow_time=3.0,
            all_red_time=1.5,
            density_weight=2.0,
            wait_weight=0.5,
            gap_out_time=3.0,
        )

    def test_initial_state_invariants(self):
        """Verifies initial 4-way intersection state."""
        state = self.controller.get_state()
        self.assertEqual(state["phase_state"], PhaseState.GREEN.value)
        self.assertEqual(state["active_approach"], Approach.NORTH.value)
        self.assertEqual(state["approaches"]["NORTH"]["state"], SignalColor.GREEN.value)
        self.assertEqual(state["approaches"]["SOUTH"]["state"], SignalColor.RED.value)
        self.assertEqual(state["approaches"]["EAST"]["state"], SignalColor.RED.value)
        self.assertEqual(state["approaches"]["WEST"]["state"], SignalColor.RED.value)

    def test_dynamic_green_duration_calculation(self):
        """Dynamic green scales with vehicle queue and respects min/max boundaries."""
        # 0 vehicles -> min_green
        self.assertEqual(self.controller.calculate_dynamic_green(0), 10.0)
        # 5 vehicles -> 10 + 2*5 = 20.0s
        self.assertEqual(self.controller.calculate_dynamic_green(5), 20.0)
        # 20 vehicles -> 10 + 2*20 = 50.0s
        self.assertEqual(self.controller.calculate_dynamic_green(20), 50.0)
        # 40 vehicles -> capped at max_green (60.0s)
        self.assertEqual(self.controller.calculate_dynamic_green(40), 60.0)

    def test_starvation_prevention_aging(self):
        """Wait time aging ensures low-density lanes never starve."""
        # Approach EAST has 1 vehicle waiting for 60 seconds
        self.controller.approaches["EAST"].vehicle_count = 1
        self.controller.approaches["EAST"].wait_time = 60.0
        # Approach SOUTH has 4 vehicles that just arrived (wait time 0s)
        self.controller.approaches["SOUTH"].vehicle_count = 4
        self.controller.approaches["SOUTH"].wait_time = 0.0

        score_east = self.controller.calculate_priority("EAST")
        score_south = self.controller.calculate_priority("SOUTH")

        # EAST: (2.0 * 1) + (0.5 * 60) = 2.0 + 30.0 = 32.0
        # SOUTH: (2.0 * 4) + (0.5 * 0) = 8.0
        self.assertGreater(score_east, score_south)
        self.assertEqual(score_east, 32.0)
        self.assertEqual(score_south, 8.0)

    def test_emergency_preemption_highest_priority(self):
        """Emergency vehicle flag immediately grants top priority over any queue."""
        self.controller.approaches["WEST"].vehicle_count = 20
        self.controller.approaches["WEST"].wait_time = 50.0

        self.controller.approaches["SOUTH"].vehicle_count = 1
        self.controller.trigger_emergency("SOUTH")

        score_west = self.controller.calculate_priority("WEST")
        score_south = self.controller.calculate_priority("SOUTH")

        self.assertGreater(score_south, score_west)
        self.assertGreater(score_south, 100000.0)

        # Arbiter must select SOUTH next
        next_app = self.controller._select_next_approach()
        self.assertEqual(next_app, "SOUTH")

    def test_safety_transitions_mutual_exclusion(self):
        """Tests GREEN -> YELLOW -> ALL_RED -> next GREEN transition invariants."""
        t0 = 1000.0
        self.controller.last_update_time = t0
        self.controller.phase_start_time = t0
        self.controller.current_green_allocated = 10.0
        self.controller.set_vehicle_count("EAST", 10)

        # 1. During GREEN
        self.controller.update(current_time=t0 + 5.0)
        self.assertEqual(self.controller.phase_state, PhaseState.GREEN.value)
        self.assertEqual(self.controller.approaches["NORTH"].state, SignalColor.GREEN.value)

        # 2. GREEN expires (t0 + 10.1) -> transitions to YELLOW
        self.controller.update(current_time=t0 + 10.1)
        self.assertEqual(self.controller.phase_state, PhaseState.YELLOW.value)
        self.assertEqual(self.controller.approaches["NORTH"].state, SignalColor.YELLOW.value)
        # All other approaches must be RED
        self.assertEqual(self.controller.approaches["SOUTH"].state, SignalColor.RED.value)
        self.assertEqual(self.controller.approaches["EAST"].state, SignalColor.RED.value)
        self.assertEqual(self.controller.approaches["WEST"].state, SignalColor.RED.value)

        # 3. YELLOW expires (t0 + 10.1 + 3.1) -> transitions to ALL_RED
        self.controller.update(current_time=t0 + 13.2)
        self.assertEqual(self.controller.phase_state, PhaseState.ALL_RED.value)
        # All 4 approaches must be strictly RED during clearance interval
        for app in Approach.all_approaches():
            self.assertEqual(self.controller.approaches[app].state, SignalColor.RED.value)

        # 4. ALL_RED expires (t0 + 13.2 + 1.6) -> next approach (EAST) turns GREEN
        self.controller.update(current_time=t0 + 14.8)
        self.assertEqual(self.controller.phase_state, PhaseState.GREEN.value)
        self.assertEqual(self.controller.active_approach, "EAST")
        self.assertEqual(self.controller.approaches["EAST"].state, SignalColor.GREEN.value)
        self.assertEqual(self.controller.approaches["NORTH"].state, SignalColor.RED.value)

    def test_gap_out_early_termination(self):
        """Queue clearing to 0 triggers early gap-out termination instead of wasting time."""
        t0 = 2000.0
        self.controller.last_update_time = t0
        self.controller.phase_start_time = t0
        # Allocated 40s of green because 15 cars were initially present
        self.controller.current_green_allocated = 40.0
        self.controller.set_vehicle_count("NORTH", 0)

        # At t0 + 10.5 (past min_green 10s), queue has been empty for > gap_out_time (3s)
        self.controller._empty_queue_since = t0 + 7.0
        self.controller.update(current_time=t0 + 10.5)

        # Must gap-out into YELLOW ahead of the 40s timer
        self.assertEqual(self.controller.phase_state, PhaseState.YELLOW.value)
        self.assertEqual(self.controller.approaches["NORTH"].state, SignalColor.YELLOW.value)

    def test_queue_discharge_is_frame_rate_independent(self):
        """Discharge follows vehicle_clearance_rate (veh/s), not how often update() is called.

        Regression: the queue used to drop by 1 on *every* update() call, so a 30 FPS loop
        drained queues ~50x faster than configured and inflated served_count / efficiency stats.
        """
        served = []
        for step in (1.0 / 30.0, 0.5):
            c = AdaptiveIntersectionController(
                min_green=100.0, max_green=200.0, vehicle_clearance_rate=0.6
            )
            t0 = 5000.0
            c.last_update_time = t0
            c.phase_start_time = t0
            c.current_green_allocated = 150.0
            c.set_vehicle_count("NORTH", 50)
            t = t0
            while t < t0 + 20.0:
                t += step
                c.update(current_time=t)
            served.append(c.total_vehicles_served)
            self.assertEqual(c.approaches["NORTH"].served_count, c.total_vehicles_served)

        # ~ (20s - 1.5s start-up lag) * 0.6 veh/s = 11 vehicles at either update rate
        for s in served:
            self.assertTrue(10 <= s <= 12, f"served={served}")
        self.assertLessEqual(abs(served[0] - served[1]), 1, f"served={served}")

    def test_dual_ring_mode(self):
        """Dual-ring mode alternates between North/South and East/West pairs."""
        controller = AdaptiveIntersectionController(mode="dual_ring")
        controller.active_approach = "NORTH"
        next_app = controller._select_next_approach()
        self.assertIn(next_app, ["EAST", "WEST"])

        controller.active_approach = "EAST"
        next_app = controller._select_next_approach()
        self.assertIn(next_app, ["NORTH", "SOUTH"])

    def test_json_serializability(self):
        """get_state() returns strictly valid JSON with native types."""
        state = self.controller.get_state()
        encoded = json.dumps(state)
        decoded = json.loads(encoded)
        self.assertIn("approaches", decoded)
        self.assertIn("stats", decoded)
        self.assertEqual(decoded["active_approach"], "NORTH")

    def test_reset_behavior(self):
        """Reset clears all approaches and restores starting defaults."""
        self.controller.set_vehicle_count("EAST", 25)
        self.controller.trigger_emergency("WEST")
        self.controller.reset()

        state = self.controller.get_state()
        self.assertEqual(state["active_approach"], "NORTH")
        self.assertEqual(state["approaches"]["NORTH"]["state"], "GREEN")
        self.assertEqual(state["approaches"]["EAST"]["vehicle_count"], 0)
        self.assertFalse(state["approaches"]["WEST"]["emergency"])


if __name__ == "__main__":
    unittest.main()
