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


ENGINES: dict[str, type[Engine]] = {
    "life": LifeEngine,
    "excitable": ExcitableEngine,
    "forestfire": ForestFireEngine,
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
