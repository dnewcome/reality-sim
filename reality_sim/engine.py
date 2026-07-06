"""Engines: the numpy machinery that evolves a LawSet forward in time.

Each *engine family* is a different kind of physics. An engine owns the grid
(the universe's state) and knows how to advance it one generation. The grid is
always a 2-D ``uint8`` array of cell states so the rest of the stack (streaming,
rendering, probes) can stay generic.

Design notes
------------
* Neighbor counting is done with ``scipy.ndimage.convolve`` using ``mode="wrap"``
  so every universe is a torus (no special-case edges). This is also what gives
  each CA its hard causal light-cone: information moves at most one cell per
  generation, which is the discrete analogue of a speed-of-light limit. That
  bound is exactly what the (future) FTL / signal-speed probes will measure and
  try to beat.
* Everything is vectorized — no Python-level per-cell loops — so a 512x512
  universe steps in well under a millisecond and the same code scales to the
  large grids you'd run on a bigger machine.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .lawset import LawSet

# Moore neighborhood (the 8 cells around a cell; center excluded).
MOORE = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=np.uint8)


class Engine:
    """Base class: owns the grid and the generation counter."""

    family = "base"

    def __init__(self, lawset: LawSet, shape: tuple[int, int], rng: np.random.Generator):
        self.lawset = lawset
        self.h, self.w = shape
        self.rng = rng
        self.generation = 0
        self.grid = np.zeros(shape, dtype=np.uint8)
        self.seed()

    # -- lifecycle -------------------------------------------------------
    def seed(self) -> None:
        """(Re)initialize the grid from the LawSet's seed recipe."""
        recipe = self.lawset.seed
        kind = recipe.get("kind", "random")
        if kind == "clear":
            self.grid[:] = 0
        elif kind == "random":
            density = float(recipe.get("density", 0.25))
            live = self.rng.random((self.h, self.w)) < density
            self.grid = live.astype(np.uint8)
        else:
            raise ValueError(f"unknown seed kind: {kind!r}")
        self.generation = 0

    def step(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def reconfigure(self, lawset: LawSet) -> None:
        """Adopt new rule parameters *in place*, keeping the current grid — so a
        running pattern can react to a changed law instead of being reseeded.
        Subclasses recompute their derived rule state; the base just swaps the
        LawSet (which carries params, states, and palette)."""
        self.lawset = lawset

    def clear(self) -> None:
        self.grid[:] = 0
        self.generation = 0

    # -- introspection ---------------------------------------------------
    def stats(self) -> dict:
        live = int(np.count_nonzero(self.grid))
        total = self.h * self.w
        return {
            "generation": self.generation,
            "live": live,
            "density": round(live / total, 4),
        }

    def resize(self, shape: tuple[int, int]) -> None:
        self.h, self.w = shape
        self.grid = np.zeros(shape, dtype=np.uint8)
        self.seed()

    def paint(self, r: int, c: int, value: int, radius: int = 0) -> None:
        """Set a cell (and optionally a small disk around it) — used by the
        browser brush so you can hand-draw structures into a running universe."""
        value = int(value) % self.lawset.states
        if radius <= 0:
            if 0 <= r < self.h and 0 <= c < self.w:
                self.grid[r, c] = value
            return
        rr, cc = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = rr * rr + cc * cc <= radius * radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if disk[dr + radius, dc + radius]:
                    self.grid[(r + dr) % self.h, (c + dc) % self.w] = value


class LifeEngine(Engine):
    """Two-state outer-totalistic "life-like" CA (Conway's Life and kin).

    The law is a birth set B and a survival set S over the count of live Moore
    neighbors: a dead cell is born if its live-neighbor count is in B; a live
    cell survives if its count is in S. Conway is B3/S23.
    """

    family = "life"

    def __init__(self, lawset, shape, rng):
        self._build_luts(lawset)
        super().__init__(lawset, shape, rng)

    def _build_luts(self, lawset: LawSet) -> None:
        birth = set(int(x) for x in lawset.params.get("birth", [3]))
        survival = set(int(x) for x in lawset.params.get("survival", [2, 3]))
        # Lookup tables indexed by neighbor count 0..8 — turns the rule into a
        # single vectorized fancy-index instead of per-cell membership tests.
        self._birth_lut = np.array([n in birth for n in range(9)], dtype=bool)
        self._survival_lut = np.array([n in survival for n in range(9)], dtype=bool)

    def reconfigure(self, lawset: LawSet) -> None:
        super().reconfigure(lawset)
        self._build_luts(lawset)

    def step(self) -> None:
        alive = self.grid.astype(bool)
        counts = ndimage.convolve(self.grid, MOORE, mode="wrap")
        born = ~alive & self._birth_lut[counts]
        survive = alive & self._survival_lut[counts]
        self.grid = (born | survive).astype(np.uint8)
        self.generation += 1


class ExcitableEngine(Engine):
    """Greenberg-Hastings excitable medium — a genuinely different physics.

    States: 0 = resting, 1 = excited, 2..n-1 = refractory (recovering). Rules:
      * a resting cell fires (-> 1) if it has >= ``threshold`` excited neighbors;
      * any cell in state k >= 1 advances k -> k+1, wrapping n-1 -> 0 back to rest.

    The result is self-organizing spiral / target waves — traveling signals with
    a well-defined propagation speed. This is the natural first testbed for the
    "how fast can a signal move?" question, and its waves look nothing like the
    static gliders of life-like rules, which is the point: different laws, different
    universe.
    """

    family = "excitable"

    def __init__(self, lawset, shape, rng):
        self.n = int(lawset.states)
        self.threshold = int(lawset.params.get("threshold", 2))
        super().__init__(lawset, shape, rng)

    def reconfigure(self, lawset: LawSet) -> None:
        super().reconfigure(lawset)
        self.threshold = int(lawset.params.get("threshold", self.threshold))
        new_n = int(lawset.states)
        if new_n != self.n:
            # Keep the current wave pattern, but fold any now-out-of-range states
            # back to resting so nothing points past the (new) palette.
            self.grid = np.where(self.grid >= new_n, 0, self.grid).astype(np.uint8)
            self.n = new_n

    def seed(self) -> None:
        recipe = self.lawset.seed
        kind = recipe.get("kind", "random")
        if kind == "clear":
            self.grid[:] = 0
        elif kind == "random":
            # `density` = fraction of non-resting cells; the rest start at rest.
            density = float(recipe.get("density", 1.0))
            states = self.rng.integers(0, self.n, size=(self.h, self.w), dtype=np.uint8)
            mask = self.rng.random((self.h, self.w)) < density
            self.grid = np.where(mask, states, 0).astype(np.uint8)
        else:
            raise ValueError(f"unknown seed kind: {kind!r}")
        self.generation = 0

    def step(self) -> None:
        excited = (self.grid == 1).astype(np.uint8)
        excited_neighbors = ndimage.convolve(excited, MOORE, mode="wrap")

        new = self.grid.copy()
        resting = self.grid == 0
        refractory = self.grid >= 1

        # Resting -> excited where enough neighbors are firing.
        new[resting & (excited_neighbors >= self.threshold)] = 1
        # Excited / refractory advance one step; last state wraps back to rest.
        advanced = self.grid.astype(np.int16) + 1
        advanced[advanced >= self.n] = 0
        new[refractory] = advanced[refractory].astype(np.uint8)

        self.grid = new
        self.generation += 1

    def stats(self) -> dict:
        base = super().stats()
        base["excited"] = int(np.count_nonzero(self.grid == 1))
        return base


class ForestFireEngine(Engine):
    """Drossel-Schwabl forest-fire model — our first *stochastic* universe.

    Three states: 0 empty, 1 tree, 2 burning. Synchronous update each step:
      * a burning cell -> empty;
      * a tree -> burning if any Moore neighbor is burning, or spontaneously with
        probability ``f`` (lightning);
      * an empty cell -> tree with probability ``p`` (growth).

    With slow growth and rare lightning (``f`` << ``p``) it self-organizes to a
    critical state with power-law-distributed fire sizes — a universe that finds
    the edge of chaos on its own, no tuning required. It is also the excitable
    medium's stochastic cousin (tree = resting, fire = excited, empty = refractory),
    and its fire fronts advance at a definite, watchable speed.
    """

    family = "forestfire"
    EMPTY, TREE, FIRE = 0, 1, 2

    def __init__(self, lawset, shape, rng):
        self.p = float(lawset.params.get("p", 0.03))
        self.f = float(lawset.params.get("f", 0.0006))
        super().__init__(lawset, shape, rng)

    def reconfigure(self, lawset: LawSet) -> None:
        super().reconfigure(lawset)
        self.p = float(lawset.params.get("p", self.p))
        self.f = float(lawset.params.get("f", self.f))

    def seed(self) -> None:
        recipe = self.lawset.seed
        kind = recipe.get("kind", "random")
        if kind == "clear":
            self.grid[:] = self.EMPTY
        elif kind == "random":
            density = float(recipe.get("density", 0.4))
            trees = self.rng.random((self.h, self.w)) < density
            self.grid = np.where(trees, self.TREE, self.EMPTY).astype(np.uint8)
        else:
            raise ValueError(f"unknown seed kind: {kind!r}")
        self.generation = 0

    def step(self) -> None:
        g = self.grid
        fire = g == self.FIRE
        tree = g == self.TREE
        empty = g == self.EMPTY
        burning_neighbors = ndimage.convolve(fire.astype(np.uint8), MOORE, mode="wrap")
        r = self.rng.random(g.shape)

        new = np.where(fire, self.EMPTY, g).astype(np.uint8)
        new[tree & ((burning_neighbors > 0) | (r < self.f))] = self.FIRE
        new[empty & (r < self.p)] = self.TREE
        self.grid = new
        self.generation += 1

    def stats(self) -> dict:
        base = super().stats()  # 'live' = non-empty cells (trees + fire)
        base["trees"] = int(np.count_nonzero(self.grid == self.TREE))
        base["fire"] = int(np.count_nonzero(self.grid == self.FIRE))
        return base


class TotalisticCA(Engine):
    """A *generative* family: a multi-state outer-totalistic CA whose entire rule
    is a random transition table. This is how we generate new universe **types**
    rather than new parameters of a fixed type.

    The rule is `next = T[current_state, neighbor_sum]`, where `neighbor_sum` is
    the sum of the (radius-r Moore) neighbors' states and `T` is a table of shape
    `(states, max_sum + 1)` with entries in `0..states-1`. Randomizing `T`,
    `states`, and `r` spans a vast space of qualitatively different automata —
    crystalline, cyclic/wave-like, life-like, chaotic — and it contains the other
    families as special cases (which is the sign it's the right generalization).
    Fully vectorized: one convolution for the neighbor sum, one fancy-index for
    the whole grid's next state.
    """

    family = "totalistic"

    def __init__(self, lawset, shape, rng):
        self._configure(lawset)
        super().__init__(lawset, shape, rng)

    def _configure(self, lawset: LawSet) -> None:
        self.n = int(lawset.states)
        self.radius = int(lawset.params.get("radius", 1))
        self.table = np.asarray(lawset.params["table"], dtype=np.uint8)
        r = self.radius
        kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=np.uint8)
        kernel[r, r] = 0
        self.kernel = kernel
        self.max_sum = int(kernel.sum()) * (self.n - 1)

    def reconfigure(self, lawset: LawSet) -> None:
        super().reconfigure(lawset)
        self._configure(lawset)

    def seed(self) -> None:
        recipe = self.lawset.seed
        kind = recipe.get("kind", "random")
        if kind == "clear":
            self.grid[:] = 0
        elif kind == "random":
            density = float(recipe.get("density", 0.5))
            states = self.rng.integers(0, self.n, size=(self.h, self.w), dtype=np.uint8)
            mask = self.rng.random((self.h, self.w)) < density
            self.grid = np.where(mask, states, 0).astype(np.uint8)
        else:
            raise ValueError(f"unknown seed kind: {kind!r}")
        self.generation = 0

    def step(self) -> None:
        s = ndimage.convolve(self.grid.astype(np.int32), self.kernel, mode="wrap")
        np.clip(s, 0, self.max_sum, out=s)
        self.grid = self.table[self.grid, s].astype(np.uint8)
        self.generation += 1


class LeniaEngine(Engine):
    """Lenia — a *continuous* cellular automaton, a genuinely different kind of
    universe from the discrete families.

    The state is a real field A(x) in [0, 1], not a handful of discrete values.
    Each step convolves the field with a smooth radial **kernel** K (concentric
    rings), maps the result through a **growth function** G (a Gaussian bump
    centered at mu, width sigma), and integrates: A += dt * G(K * A), clipped to
    [0, 1]. Out of this fall the famous glider-like "creatures". The rule *is* the
    (kernel, growth) pair, so randomizing it explores a continuous space of
    automata — the first continuous primitive of the type-grammar.

    For streaming/rendering the float field is quantized to uint8 (states=256 with
    a gradient palette), so the rest of the stack is unchanged.
    """

    family = "lenia"

    def __init__(self, lawset, shape, rng):
        self._configure(lawset)
        super().__init__(lawset, shape, rng)

    def _configure(self, lawset: LawSet) -> None:
        p = lawset.params
        self.R = int(p.get("R", 13))
        self.mu = float(p.get("mu", 0.15))
        self.sigma = float(p.get("sigma", 0.015))
        self.dt = float(p.get("dt", 0.1))
        self.beta = [float(b) for b in p.get("beta", [1.0])]
        self.kernel = self._make_kernel()

    def _make_kernel(self) -> np.ndarray:
        R = self.R
        y, x = np.ogrid[-R:R + 1, -R:R + 1]
        dist = np.sqrt(x * x + y * y) / R          # normalized radius, 0 at center
        nb = len(self.beta)
        r = dist * nb
        shell = np.minimum(r.astype(int), nb - 1)
        core = np.exp(-((r % 1.0 - 0.5) ** 2) / (2 * 0.15 ** 2))  # bump in each ring
        beta = np.array(self.beta)
        K = np.where(dist < 1.0, beta[shell] * core, 0.0)
        total = K.sum()
        return K / total if total > 0 else K

    def reconfigure(self, lawset: LawSet) -> None:
        super().reconfigure(lawset)
        self._configure(lawset)                    # keeps self.field; morphs dynamics

    def _quantize(self) -> None:
        self.grid = np.clip(self.field * 255.0, 0, 255).astype(np.uint8)

    def seed(self) -> None:
        recipe = self.lawset.seed
        kind = recipe.get("kind", "random")
        self.field = np.zeros((self.h, self.w), dtype=np.float64)
        if kind == "random":
            density = float(recipe.get("density", 0.5))
            rad = max(4, int(min(self.h, self.w) * 0.22))
            patch = self.rng.random((2 * rad, 2 * rad))
            patch = ndimage.gaussian_filter(patch, sigma=2.0)   # smooth blob of life
            lo, hi = patch.min(), patch.max()
            patch = (patch - lo) / (hi - lo + 1e-9) * density
            y0, x0 = self.h // 2 - rad, self.w // 2 - rad
            self.field[y0:y0 + 2 * rad, x0:x0 + 2 * rad] = patch
        elif kind != "clear":
            raise ValueError(f"unknown seed kind: {kind!r}")
        self.generation = 0
        self._quantize()

    def clear(self) -> None:
        self.field[:] = 0.0
        self.generation = 0
        self._quantize()

    def step(self) -> None:
        u = ndimage.convolve(self.field, self.kernel, mode="wrap")
        g = 2.0 * np.exp(-((u - self.mu) ** 2) / (2.0 * self.sigma ** 2)) - 1.0
        self.field = np.clip(self.field + self.dt * g, 0.0, 1.0)
        self.generation += 1
        self._quantize()

    def paint(self, r: int, c: int, value: int, radius: int = 0) -> None:
        v = 1.0 if int(value) > 0 else 0.0
        radius = max(radius, 1)
        rr, cc = int(r), int(c)
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = xx * xx + yy * yy <= radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if disk[dy + radius, dx + radius] and 0 <= rr + dy < self.h and 0 <= cc + dx < self.w:
                    self.field[rr + dy, cc + dx] = v
        self._quantize()

    def stats(self) -> dict:
        return {
            "generation": self.generation,
            "live": int(np.count_nonzero(self.grid)),
            "mass": round(float(self.field.sum()), 1),
        }


class LevelSetEngine(Engine):
    """A signed-distance-field / level-set automaton — a universe of *shapes*.

    The state is a signed distance field phi: negative inside a shape, positive
    outside, zero on the surface. The "matter" is the region phi < 0. Each step
    evolves the *interface* (the zero level set) by two geometric operators:

      * **grow / erode** — shift phi by a constant; since phi is a true distance,
        subtracting `grow` moves every surface outward by exactly that many cells
        (negative `grow` erodes).
      * **surface tension** — motion by mean curvature `kappa = div(grad phi/|grad phi|)`:
        convex bumps shrink, concave dents fill, thin necks pinch and split. This
        is what rounds shapes and makes them merge/relax like droplets.

    After evolving, phi is **re-distanced** — recomputed as the exact signed
    distance to the new interface via `scipy`'s Euclidean distance transform — so it
    stays a true SDF and the operators keep their geometric meaning. Grow vs. tension
    balance into stable rounded blobs that merge and pinch off.
    """

    family = "levelset"

    def __init__(self, lawset, shape, rng):
        self._configure(lawset)
        super().__init__(lawset, shape, rng)

    def _configure(self, lawset: LawSet) -> None:
        p = lawset.params
        self.grow = float(p.get("grow", 0.3))
        self.tension = float(p.get("tension", 0.5))
        self.reinit = int(p.get("reinit", 5))       # re-distance every N steps
        self.quant = float(p.get("quant", 10.0))    # cells-per-palette-step near the surface
        self._rc = 0

    def reconfigure(self, lawset: LawSet) -> None:
        super().reconfigure(lawset)
        self._configure(lawset)

    def _redistance(self, interior: np.ndarray) -> None:
        edt = ndimage.distance_transform_edt
        if not interior.any():
            self.phi = edt(np.ones_like(interior))
        elif interior.all():
            self.phi = -edt(interior)
        else:
            self.phi = (edt(~interior) - edt(interior)).astype(np.float64)

    def _quantize(self) -> None:
        # far exterior -> 0 (background), surface -> 128, deep interior -> 255
        q = np.clip(128.0 - self.phi * self.quant, 0, 255)
        self.grid = q.astype(np.uint8)

    def seed(self) -> None:
        recipe = self.lawset.seed
        kind = recipe.get("kind", "random")
        interior = np.zeros((self.h, self.w), dtype=bool)
        if kind == "random":
            density = float(recipe.get("density", 0.4))
            n = max(1, int(round(density * 10)))
            yy, xx = np.ogrid[:self.h, :self.w]
            lo, hi = min(self.h, self.w) // 12, max(min(self.h, self.w) // 5, min(self.h, self.w) // 12 + 1)
            for _ in range(n):
                cy = int(self.rng.integers(0, self.h))
                cx = int(self.rng.integers(0, self.w))
                rad = int(self.rng.integers(lo, hi))
                interior |= (yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad
        elif kind != "clear":
            raise ValueError(f"unknown seed kind: {kind!r}")
        self._redistance(interior)
        self.generation = 0
        self._quantize()

    def clear(self) -> None:
        self.phi = np.full((self.h, self.w), 20.0)   # all exterior (empty)
        self.generation = 0
        self._quantize()

    def step(self) -> None:
        phi = self.phi
        gy, gx = np.gradient(phi)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-6
        nx, ny = gx / mag, gy / mag
        kappa = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)   # divergence of normal
        np.clip(kappa, -1.0, 1.0, out=kappa)
        # Evolve the field continuously so sub-cell motion accumulates; re-distance
        # only every `reinit` steps (else a hard phi<0 threshold would freeze motion).
        self.phi = phi - self.grow + self.tension * kappa
        self._rc += 1
        if self._rc >= self.reinit:
            self._redistance(self.phi < 0)
            self._rc = 0
        self.generation += 1
        self._quantize()

    def paint(self, r: int, c: int, value: int, radius: int = 0) -> None:
        radius = max(radius, 2)
        yy, xx = np.ogrid[:self.h, :self.w]
        disk = (yy - int(r)) ** 2 + (xx - int(c)) ** 2 <= radius * radius
        interior = self.phi < 0
        interior = (interior | disk) if int(value) > 0 else (interior & ~disk)
        self._redistance(interior)
        self._quantize()

    def stats(self) -> dict:
        area = int(np.count_nonzero(self.phi < 0))
        return {"generation": self.generation, "live": area,
                "area": round(area / (self.h * self.w), 4)}


ENGINES: dict[str, type[Engine]] = {
    "life": LifeEngine,
    "excitable": ExcitableEngine,
    "forestfire": ForestFireEngine,
    "totalistic": TotalisticCA,
    "lenia": LeniaEngine,
    "levelset": LevelSetEngine,
}


def make_engine(lawset: LawSet, shape: tuple[int, int], rng: np.random.Generator | None = None) -> Engine:
    """Build the right engine for a LawSet's family."""
    if rng is None:
        rng = np.random.default_rng()
    try:
        cls = ENGINES[lawset.family]
    except KeyError:
        raise ValueError(
            f"no engine registered for family {lawset.family!r}; "
            f"known families: {sorted(ENGINES)}"
        )
    return cls(lawset, shape, rng)
