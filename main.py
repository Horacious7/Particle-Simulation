"""
Task 0 – 3D Particle Simulation inside a Cube with Entropy Visualisation
=========================================================================
Entry point.  Run with:   python main.py

Project structure
-----------------
main.py                         ← you are here
simulation/
    __init__.py                 ← package façade
    constants.py                ← global configuration values
    particle.py                 ← Particle class (data + vpython sphere)
    particle_system.py          ← ParticleSystem (physics, collisions)
    subcube_grid.py             ← SubCubeGrid (entropy computation + colour)
    simulation.py               ← Simulation (scene, UI, main loop)
"""

from simulation import Simulation

if __name__ == "__main__":
    sim = Simulation()
    sim.run()

