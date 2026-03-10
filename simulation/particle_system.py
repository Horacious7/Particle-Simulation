"""Manages the collection of particles and all physics (integration,
wall collisions, inter-particle elastic collisions).

Performance notes
-----------------
- Integration and wall collisions are **vectorised** with NumPy so the hot
  path runs in compiled C rather than Python loops.
- Particle–particle collisions use **spatial hashing** (uniform grid) to
  reduce the complexity from O(n²) to ~O(n) on average.
- Visual sync is performed every *k*-th step to cut WebSocket overhead.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict

import numpy as np

from .particle import Particle
from .constants import CUBE_HALF, PARTICLE_RADIUS

# How often (in physics sub-steps) we push positions to vpython.
_VISUAL_SYNC_INTERVAL = 2


class ParticleSystem:
    """Owns the list of :class:`Particle` objects and handles physics.

    A threading lock protects the particle list so that the vpython UI
    callback (slider) cannot mutate it while the physics loop is iterating.
    """

    def __init__(self) -> None:
        self._particles: list[Particle] = []
        self._lock = threading.Lock()
        self._step_count: int = 0

    # ── access ──────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._particles)

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    def positions_array(self) -> np.ndarray:
        """Return an (N, 3) array of all current positions."""
        with self._lock:
            if not self._particles:
                return np.empty((0, 3))
            return np.array([p.position for p in self._particles])

    # ── add / remove ────────────────────────────────────────────────────────

    def add(self, n: int = 1) -> None:
        """Create *n* new particles at random positions / velocities."""
        with self._lock:
            for _ in range(n):
                p = Particle(Particle.random_position(),
                             Particle.random_velocity())
                self._particles.append(p)

    def remove(self, n: int = 1) -> None:
        """Remove the last *n* particles from the system."""
        with self._lock:
            for _ in range(min(n, len(self._particles))):
                p = self._particles.pop()
                p.destroy()

    # ── physics step ────────────────────────────────────────────────────────

    def step(self, dt: float) -> None:
        """Advance the simulation by *dt*: integrate, collide, sync visuals.

        The entire step is performed while holding the lock so the slider
        callback cannot add / remove particles mid-iteration.
        """
        with self._lock:
            n = len(self._particles)
            if n == 0:
                return

            self._integrate_vectorised(dt, n)
            self._wall_collisions_vectorised(n)
            self._particle_collisions_spatial()

            self._step_count += 1
            if self._step_count % _VISUAL_SYNC_INTERVAL == 0:
                self._sync_visuals()

    # ── vectorised integration ──────────────────────────────────────────────

    def _integrate_vectorised(self, dt: float, n: int) -> None:
        """Euler step using bulk NumPy operations (no Python per-particle loop)."""
        positions = np.array([p.position for p in self._particles])
        velocities = np.array([p.velocity for p in self._particles])
        positions += velocities * dt
        for i in range(n):
            self._particles[i].position = positions[i]

    # ── vectorised wall collisions ──────────────────────────────────────────

    def _wall_collisions_vectorised(self, n: int) -> None:
        """Reflect particles off bounding-box walls using NumPy masking."""
        lim = CUBE_HALF - PARTICLE_RADIUS
        positions = np.array([p.position for p in self._particles])
        velocities = np.array([p.velocity for p in self._particles])

        # upper wall
        mask_hi = positions > lim
        positions[mask_hi] = lim
        velocities[mask_hi] = -np.abs(velocities[mask_hi])

        # lower wall
        mask_lo = positions < -lim
        positions[mask_lo] = -lim
        velocities[mask_lo] = np.abs(velocities[mask_lo])

        for i in range(n):
            self._particles[i].position = positions[i]
            self._particles[i].velocity = velocities[i]

    # ── spatial-hash particle collisions (~O(n)) ────────────────────────────

    def _particle_collisions_spatial(self) -> None:
        """Elastic collisions using a uniform spatial grid so only
        neighbouring cells are checked (~O(n) average instead of O(n²))."""
        particles = self._particles
        n = len(particles)
        if n < 2:
            return

        min_dist = 2 * PARTICLE_RADIUS
        min_dist_sq = min_dist * min_dist
        cell_size = min_dist          # cell width ≥ interaction range

        inv_cell = 1.0 / cell_size

        # ── build grid ──────────────────────────────────────────────────
        grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i, p in enumerate(particles):
            cx = int(math.floor(p.position[0] * inv_cell))
            cy = int(math.floor(p.position[1] * inv_cell))
            cz = int(math.floor(p.position[2] * inv_cell))
            grid[(cx, cy, cz)].append(i)

        # ── neighbour offsets (27 cells including self) ─────────────────
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        # ── check only nearby pairs ────────────────────────────────────
        checked: set[tuple[int, int]] = set()

        for (cx, cy, cz), indices in grid.items():
            for dx, dy, dz in offsets:
                neighbour_key = (cx + dx, cy + dy, cz + dz)
                neighbour = grid.get(neighbour_key)
                if neighbour is None:
                    continue
                for i in indices:
                    for j in neighbour:
                        if i >= j:
                            continue
                        pair = (i, j)
                        if pair in checked:
                            continue
                        checked.add(pair)

                        dp = particles[i].position - particles[j].position
                        dist_sq = float(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2])
                        if dist_sq >= min_dist_sq or dist_sq < 1e-12:
                            continue

                        dist = math.sqrt(dist_sq)
                        n_vec = dp / dist
                        dv = particles[i].velocity - particles[j].velocity
                        dvn = float(dv[0] * n_vec[0] + dv[1] * n_vec[1] + dv[2] * n_vec[2])
                        if dvn > 0:            # already separating
                            continue

                        # equal-mass elastic: swap normal velocity components
                        impulse = dvn * n_vec
                        particles[i].velocity = particles[i].velocity - impulse
                        particles[j].velocity = particles[j].velocity + impulse

                        # push apart to resolve overlap
                        overlap = min_dist - dist
                        shift = (overlap * 0.5 + 0.001) * n_vec
                        particles[i].position = particles[i].position + shift
                        particles[j].position = particles[j].position - shift

    # ── visual sync ─────────────────────────────────────────────────────────

    def _sync_visuals(self) -> None:
        for p in self._particles:
            p.update_visual()

