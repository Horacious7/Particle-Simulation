"""Subcube grid – divides the bounding cube into n×n×n cells, computes
Shannon entropy from particle density, and colours each cell in real time."""

from __future__ import annotations

import numpy as np
from vpython import box, vector, color


class SubCubeGrid:
    """Divides the bounding cube into *n*×*n*×*n* sub-cubes, computes
    Shannon entropy from particle density, and colours each cell on a
    blue → cyan → green → yellow → red gradient."""

    def __init__(self, n_sub: int, cube_half: float) -> None:
        self._n = n_sub
        self._cube_half = cube_half
        self._side = 2 * cube_half / n_sub      # side length of one sub-cube
        self._total = n_sub ** 3
        self._boxes: list[box] = []
        self._create_boxes()

    # ── initialisation ──────────────────────────────────────────────────────

    def _create_boxes(self) -> None:
        s = self._side
        half = self._cube_half
        for i in range(self._n):
            for j in range(self._n):
                for k in range(self._n):
                    cx = -half + s * (i + 0.5)
                    cy = -half + s * (j + 0.5)
                    cz = -half + s * (k + 0.5)
                    b = box(
                        pos=vector(cx, cy, cz),
                        size=vector(s, s, s) * 0.98,
                        color=color.blue,
                        opacity=0.12,
                    )
                    self._boxes.append(b)

    # ── public interface ────────────────────────────────────────────────────

    def update(self, positions: np.ndarray, n_particles: int) -> None:
        """Recompute entropy from current particle positions and recolour."""
        if n_particles == 0:
            return
        ids = self._assign(positions)
        h_vals, counts = self._shannon_entropy(ids)
        colours = self._entropy_to_colour(h_vals)

        for idx, b in enumerate(self._boxes):
            r, g, bl = colours[idx]
            b.color = vector(r, g, bl)
            b.opacity = 0.05 + 0.20 * (counts[idx] / max(n_particles, 1))

    # ── private helpers ─────────────────────────────────────────────────────

    def _assign(self, positions: np.ndarray) -> np.ndarray:
        """Return the linear subcube index for every particle."""
        idx = ((positions + self._cube_half) / self._side).astype(int)
        idx = np.clip(idx, 0, self._n - 1)
        return idx[:, 0] * (self._n ** 2) + idx[:, 1] * self._n + idx[:, 2]

    def _shannon_entropy(self, subcube_ids: np.ndarray):
        """Compute per-subcube Shannon entropy contribution.

        H_i = −p_i · log₂(p_i)  where  p_i = count_i / total_particles.
        """
        total = max(len(subcube_ids), 1)
        counts = np.bincount(subcube_ids, minlength=self._total).astype(float)
        probs = counts / total
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(probs > 0, -probs * np.log2(probs), 0.0)
        return h, counts

    @staticmethod
    def _entropy_to_colour(
        h_values: np.ndarray,
    ) -> list[tuple[float, float, float]]:
        """Map entropy values to a blue → cyan → green → yellow → red gradient."""
        h_max = h_values.max() if h_values.max() > 0 else 1.0
        normed = h_values / h_max
        colours: list[tuple[float, float, float]] = []
        for t in normed:
            if t < 0.5:
                r, g, b = 0.0, 2.0 * t, 1.0 - 2.0 * t
            else:
                r, g, b = 2.0 * (t - 0.5), 1.0 - 2.0 * (t - 0.5), 0.0
            colours.append((r, g, b))
        return colours

