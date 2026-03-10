"""Single-particle data object with vpython sphere rendering."""

from __future__ import annotations

import random

import numpy as np
from vpython import sphere, vector, color

from .constants import CUBE_HALF, PARTICLE_RADIUS, MAX_SPEED


class Particle:
    """Represents a single 3-D particle with position, velocity, and a
    vpython sphere for rendering."""

    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self._position = position.copy()
        self._velocity = velocity.copy()
        self._sphere = sphere(
            pos=vector(*self._position),
            radius=PARTICLE_RADIUS,
            color=color.white,
            make_trail=False,
            emissive=True,
        )

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        self._position = value

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self._velocity = value

    # ── public helpers ──────────────────────────────────────────────────────

    def update_visual(self) -> None:
        """Sync the vpython sphere position with the internal numpy array."""
        self._sphere.pos = vector(*self._position)

    def destroy(self) -> None:
        """Remove the visual sphere from the scene."""
        self._sphere.visible = False
        del self._sphere

    # ── factory helpers ─────────────────────────────────────────────────────

    @staticmethod
    def random_position() -> np.ndarray:
        lim = CUBE_HALF - PARTICLE_RADIUS
        return np.array([random.uniform(-lim, lim) for _ in range(3)])

    @staticmethod
    def random_velocity() -> np.ndarray:
        return np.array([random.uniform(-MAX_SPEED, MAX_SPEED) for _ in range(3)])

