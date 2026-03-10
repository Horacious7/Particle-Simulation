"""Top-level orchestrator – scene, UI widgets, and the main loop."""

from __future__ import annotations

from vpython import canvas, box, vector, color, rate, slider, wtext

from .constants import CUBE_HALF, DT, N_SUB, INITIAL_N, MAX_PARTICLES, FPS
from .particle_system import ParticleSystem
from .subcube_grid import SubCubeGrid


class Simulation:
    """Sets up the vpython scene, creates the UI, owns the
    :class:`ParticleSystem` and :class:`SubCubeGrid`, and runs the main loop."""

    def __init__(self) -> None:
        # ── scene ───────────────────────────────────────────────────────────
        self._scene = canvas(
            title="<b>3-D Particle Entropy Simulation</b>",
            width=900,
            height=700,
            center=vector(0, 0, 0),
            background=vector(0.15, 0.15, 0.2),
        )
        self._scene.caption = "\n"

        # bounding cube (wireframe look via low opacity)
        self._bounding_box = box(
            pos=vector(0, 0, 0),
            size=vector(2 * CUBE_HALF, 2 * CUBE_HALF, 2 * CUBE_HALF),
            color=color.white,
            opacity=0.08,
        )

        # ── domain objects ──────────────────────────────────────────────────
        self._system = ParticleSystem()
        self._grid = SubCubeGrid(N_SUB, CUBE_HALF)

        # ── initial particles ───────────────────────────────────────────────
        self._system.add(INITIAL_N)

        # ── UI widgets ──────────────────────────────────────────────────────
        self._label = wtext(text=f"  Particles: {self._system.count}  ")
        self._scene.append_to_caption("\nNumber of particles: ")
        self._slider = slider(
            min=1,
            max=MAX_PARTICLES,
            value=INITIAL_N,
            step=1,
            length=350,
            bind=self._on_slider_change,
        )
        self._scene.append_to_caption("\n\n")

        # Pending slider target – processed safely in the main loop to avoid
        # mutating the particle list while the physics step is running.
        self._pending_target: int | None = None

    # ── slider callback ─────────────────────────────────────────────────────

    def _on_slider_change(self, s) -> None:
        """Called from the vpython UI thread.  We only record the desired
        count here; the actual add/remove happens in the main loop between
        physics steps, so no concurrent-modification crash can occur."""
        self._pending_target = int(s.value)

    def _apply_pending_target(self) -> None:
        """Add or remove particles to match the slider, called once per
        frame at a safe point (before or after the physics step)."""
        if self._pending_target is None:
            return
        target = self._pending_target
        self._pending_target = None
        diff = target - self._system.count
        if diff > 0:
            self._system.add(diff)
        elif diff < 0:
            self._system.remove(-diff)
        self._label.text = f"  Particles: {self._system.count}  "

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the infinite simulation loop."""
        while True:
            rate(FPS)

            # safely apply any slider change between frames
            self._apply_pending_target()

            if self._system.count == 0:
                continue

            # physics
            self._system.step(DT)

            # entropy visualisation
            positions = self._system.positions_array()
            self._grid.update(positions, self._system.count)

