# Task 0 – Theory & Design Notes

## 1. What the Simulation Does

We simulate **N rigid spheres** bouncing inside a closed cube. The cube is
subdivided into an **n × n × n grid** of smaller cells. For each cell we
compute and visualise the **Shannon entropy** of the particle distribution in
real time. The result is a colour-coded 3-D map that shows how "spread out"
the particles are across the volume.

---

## 2. Physics

### 2.1 Equations of Motion (Euler Integration)

Each particle carries a position **r** and velocity **v**.  
At every time-step Δt we perform a first-order Euler update:

```
r(t + Δt) = r(t) + v(t) · Δt
```

This is the simplest integrator; it introduces a small energy drift over time
but is perfectly adequate for a visualisation that does not need long-term
conservation guarantees.

### 2.2 Wall Collisions

When a particle reaches a face of the bounding cube its velocity component
normal to that face is **reflected** (sign-flipped):

```
if  rₐ > L − R  then  rₐ = L − R,  vₐ = −|vₐ|
if  rₐ < −L + R  then  rₐ = −L + R,  vₐ = |vₐ|
```

where *L* is the cube half-length, *R* the particle radius, and *a ∈ {x,y,z}*.

### 2.3 Particle–Particle Elastic Collisions

Two equal-mass spheres *i* and *j* collide when their centres are closer than
2*R*.  For equal masses the elastic collision formula reduces to swapping the
**normal components** of velocity:

```
n̂ = (rᵢ − rⱼ) / ‖rᵢ − rⱼ‖          (collision normal)
Δvₙ = (vᵢ − vⱼ) · n̂                 (relative approach speed)

vᵢ ← vᵢ − Δvₙ · n̂
vⱼ ← vⱼ + Δvₙ · n̂
```

If Δvₙ > 0, the particles are already separating and we skip them.  
We also push overlapping spheres apart to prevent "sticking".

---

## 3. Entropy

### 3.1 Shannon Entropy

We treat each subcube as a bin.  Let *N* be the total particle count and
*nₖ* the count in cell *k*.  The probability that a randomly chosen particle
is in cell *k* is:

```
pₖ = nₖ / N
```

The **Shannon entropy** of the whole distribution is:

```
H = − Σₖ pₖ · log₂(pₖ)       (bits)
```

Empty cells contribute 0 (by convention 0 · log 0 = 0).

| Scenario | Entropy |
|----------|---------|
| All particles in one cell | H = 0 (minimum – perfect order) |
| Particles spread uniformly | H = log₂(n³) (maximum – most disordered) |

In the visualisation each **individual cell** is coloured by its own term
`hₖ = −pₖ · log₂(pₖ)`, which peaks when a cell holds a "fair share" of
particles.

### 3.2 Why Shannon and not Boltzmann?

Boltzmann entropy (*S = k_B · ln W*) counts the number of **microstates**
compatible with a macrostate.  For a discrete particle-in-cells model the two
are essentially equivalent: the Shannon entropy (in nats) multiplied by *k_B*
gives Boltzmann entropy.  We chose Shannon because:

* it works directly with the probability distribution we already compute,
* it does not require defining energy levels or temperature,
* it maps naturally to an information-theoretic colour scale.

---

## 4. Performance Optimisations

### 4.1 NumPy Vectorisation

Integration and wall-collision are the most frequently called routines.
Instead of a Python `for` loop over every particle, we pack all positions
and velocities into **(N, 3) NumPy arrays** and operate on them with a single
vectorised expression.  NumPy delegates the actual arithmetic to compiled C /
BLAS, which is **2–5× faster** than pure Python loops.

### 4.2 Spatial Hashing (Uniform Grid)

A brute-force collision check tests every pair → **O(n²)**.  
With spatial hashing:

1. Divide space into cells whose side length equals the interaction distance
   (2 × particle radius).
2. Insert each particle into its cell → **O(n)**.
3. For each cell, check only the **27 neighbouring cells** (3³ = 27).
4. Because each cell contains very few particles on average the total work
   is **~O(n)**.

This is the same technique used by real-time game engines for broad-phase
collision detection.

### 4.3 Reduced Visual Sync

vpython pushes every attribute change over a WebSocket to the browser.
Updating all sphere positions every single sub-step floods this channel.
We only call `_sync_visuals()` every **k-th step** (default k = 2), cutting
rendering overhead roughly in half with no visible difference.

---

## 5. Architecture (OOP Design)

```
main.py                  ← entry point
simulation/
├── __init__.py          ← public package API
├── constants.py         ← all tuneable parameters in one place
├── particle.py          ← Particle class  (SRP: one sphere's data + visual)
├── particle_system.py   ← ParticleSystem  (SRP: physics & collection management)
├── subcube_grid.py      ← SubCubeGrid     (SRP: entropy computation + colouring)
└── simulation.py        ← Simulation       (SRP: scene setup, UI, main loop)
```

### Principles applied

| Principle | How |
|-----------|-----|
| **Single Responsibility** | Each class has exactly one reason to change. |
| **Open / Closed** | New entropy methods (e.g. Boltzmann) can be added by extending `SubCubeGrid` without modifying existing code. |
| **Encapsulation** | Internal state (`_particles`, `_lock`) is private; interaction goes through methods and properties. |
| **Thread Safety** | A `threading.Lock` prevents the UI slider callback from mutating the particle list during a physics step (which was the cause of the original `IndexError` crash). |
| **Separation of Concerns** | Rendering (vpython) is isolated in `Particle.update_visual()` and `SubCubeGrid`; physics logic knows nothing about the display. |

---

## 6. Colour Mapping

Entropy values are normalised to [0, 1] and mapped to a five-stop gradient:

```
0.0  → blue    (low entropy – few or no particles)
0.25 → cyan
0.50 → green
0.75 → yellow
1.0  → red     (high entropy – many particles / "fair share")
```

Cell **opacity** also scales with particle count so empty cells stay nearly
transparent while populated cells become more visible.

