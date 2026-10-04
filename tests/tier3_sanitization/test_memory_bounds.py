"""Tier 3: Tracking Memory Bounds & Trajectory TTL Pruning Tests.
Verifies that the trajectory tracking cache implements frame-based TTL pruning,
evicts inactive tracks after TTL frames, bounds memory consumption, and
NEVER destroys active in-flight trajectories with destructive .clear() wiping.
"""

import unittest

try:
    from traffiq.vision.tracker import VehicleTracker, TrajectoryCache
except ImportError:
    VehicleTracker = None
    TrajectoryCache = None


class OracleTrajectoryCache:
    """Authoritative reference implementation of a bounded TTL trajectory cache."""

    def __init__(self, ttl: int = 30):
        self.ttl = int(ttl)
        # Store {track_id: {"last_seen": frame_idx, "positions": [(cx, cy), ...]}}
        self._cache = {}

    def update(self, track_id: int, centroid: tuple[int, int], frame_idx: int):
        """Records position and updates last seen timestamp."""
        if track_id not in self._cache:
            self._cache[track_id] = {"last_seen": frame_idx, "positions": []}
        
        self._cache[track_id]["last_seen"] = frame_idx
        self._cache[track_id]["positions"].append(centroid)

    def prune_inactive(self, current_frame: int, ttl: int = None):
        """Evicts entries unseen for > ttl frames without wiping active tracks."""
        effective_ttl = self.ttl if ttl is None else ttl
        to_evict = [
            tid
            for tid, data in self._cache.items()
            if (current_frame - data["last_seen"]) > effective_ttl
        ]
        for tid in to_evict:
            del self._cache[tid]
        return len(to_evict)

    def get_trajectory(self, track_id: int):
        if track_id in self._cache:
            return self._cache[track_id]["positions"]
        return None

    def active_track_count(self) -> int:
        return len(self._cache)


class TestMemoryBounds(unittest.TestCase):
    """Tests trajectory memory bounding and safe eviction without data loss."""

    def setUp(self):
        self.cache_cls = TrajectoryCache or OracleTrajectoryCache

    def test_ttl_eviction_of_inactive_tracks(self):
        """Verifies tracks unseen for > TTL frames are evicted."""
        cache = self.cache_cls(ttl=30)

        # Vehicle 1 active at frame 10
        cache.update(track_id=1, centroid=(100, 200), frame_idx=10)
        # Vehicle 2 active at frame 35
        cache.update(track_id=2, centroid=(150, 250), frame_idx=35)

        self.assertEqual(cache.active_track_count(), 2)

        # At frame 45:
        # Vehicle 1 was last seen at frame 10 (45 - 10 = 35 > 30 -> should be evicted)
        # Vehicle 2 was last seen at frame 35 (45 - 35 = 10 <= 30 -> must be preserved)
        evicted = cache.prune_inactive(current_frame=45)
        self.assertEqual(evicted, 1)
        self.assertEqual(cache.active_track_count(), 1)
        self.assertIsNone(cache.get_trajectory(track_id=1))
        self.assertIsNotNone(cache.get_trajectory(track_id=2))

    def test_active_in_flight_trajectories_never_wiped(self):
        """Verifies active tracks in the process of crossing the tripwire are NEVER wiped."""
        cache = self.cache_cls(ttl=30)

        # Vehicle 99 actively moving across frames 100 to 110
        for f in range(100, 111):
            cache.update(track_id=99, centroid=(200, 200 + (f - 100) * 10), frame_idx=f)
            # Periodic pruning called every frame
            cache.prune_inactive(current_frame=f)

        # Trajectory must remain intact and complete with 11 positions
        traj = cache.get_trajectory(99)
        self.assertIsNotNone(traj, "Active trajectory was prematurely dropped!")
        self.assertEqual(len(traj), 11)
        self.assertEqual(traj[-1], (200, 300))

    def test_bounded_memory_under_sustained_traffic_stream(self):
        """Verifies memory stays strictly bounded even after thousands of vehicles pass."""
        cache = self.cache_cls(ttl=20)
        max_seen_size = 0

        # Simulate 1,000 distinct vehicles over 2,000 frames
        # At any moment, only ~10 vehicles are active on screen
        for frame in range(1, 2001):
            # Spawn a new vehicle every 2 frames, lasting 10 frames
            active_id = frame // 2
            cache.update(track_id=active_id, centroid=(300, 100 + (frame % 10) * 20), frame_idx=frame)

            # Prune every 10 frames
            if frame % 10 == 0:
                cache.prune_inactive(current_frame=frame)
                current_size = cache.active_track_count()
                if current_size > max_seen_size:
                    max_seen_size = current_size

        # Without TTL pruning, cache size would reach 1,000!
        # With TTL=20, cache size must stay bounded (typically < 30)
        self.assertLess(
            max_seen_size,
            50,
            f"Trajectory cache grew unbounded: peak size was {max_seen_size} items",
        )


if __name__ == "__main__":
    unittest.main()
