"""Dynamical metrics: read numbers off a *freely evolving* universe.

This is deliberately **not** probing. We never perturb a cell and watch a front
(that is the deferred signal-speed probe). We just let a universe run from a
random soup and measure the shape of its free evolution — population, activity,
spatial structure, periodicity. Those numbers are the feature vector the ML
sweep clusters and learns to predict from a rule.

Why these features capture "what kind of universe is this":
  * **population / density** — does matter persist, vanish, or fill everything?
  * **activity** (fraction of cells that change per step) — the classic
    order/chaos dial: ~0 means frozen, ~0.5 means boiling. Sustained *moderate*
    activity is the fingerprint of the interesting edge-of-chaos regime.
  * **spatial entropy** (over 2x2 blocks) — is the final state structured or
    random-looking?
  * **period** — did it settle into a short cycle (ordered) or not (complex/chaotic)?
"""

from __future__ import annotations

import numpy as np

from .engine import make_engine
from .lawset import LawSet

FEATURE_NAMES = [
    "final_density", "mean_density", "std_density", "max_density",
    "mean_activity", "activity_last", "activity_early", "activity_drop",
    "growth", "spatial_entropy", "period", "alive",
]


def block_entropy(grid: np.ndarray) -> float:
    """Shannon entropy (bits, 0..4) of the distribution of 2x2 cell blocks in the
    final state. High = spatially disordered; low = uniform / simple."""
    b = (grid != 0).astype(np.uint8)
    a = b[:-1, :-1]; c = b[:-1, 1:]; d = b[1:, :-1]; e = b[1:, 1:]
    code = (a << 3) | (c << 2) | (d << 1) | e
    counts = np.bincount(code.ravel(), minlength=16).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _detect_period(engine, max_p: int = 16) -> float:
    """Steps the engine up to ``max_p`` times looking for an exact recurrence.
    Returns the period (1 = still life, 2 = blinker, ...) or 0 if none is found
    within the window (a long transient — chaotic or complex). Mutates the engine,
    so call this only after all other features are collected."""
    ref = engine.grid.copy()
    for p in range(1, max_p + 1):
        engine.step()
        if np.array_equal(engine.grid, ref):
            return float(p)
    return 0.0


def measure_run(lawset: LawSet, size: int = 64, steps: int = 200, seed: int = 0,
                max_period: int = 16) -> dict:
    """Evolve ``lawset`` from a random soup and return its feature vector."""
    rng = np.random.default_rng(seed)
    eng = make_engine(lawset, (size, size), rng)
    n = size * size

    prev = eng.grid.copy()
    pop = np.empty(steps + 1, dtype=np.float64)
    act = np.empty(steps, dtype=np.float64)
    pop[0] = np.count_nonzero(eng.grid)

    for t in range(steps):
        eng.step()
        g = eng.grid
        act[t] = np.count_nonzero(g != prev) / n
        pop[t + 1] = np.count_nonzero(g)
        prev = g.copy()

    dens = pop / n
    half = steps // 2
    late = dens[half:]
    early_act = act[:half].mean() if half else act.mean()
    late_act = act[half:].mean() if half else act.mean()

    feats = {
        "final_density": float(dens[-1]),
        "mean_density": float(late.mean()),
        "std_density": float(late.std()),
        "max_density": float(dens.max()),
        "mean_activity": float(late_act),
        "activity_last": float(act[-1]),
        "activity_early": float(early_act),
        "activity_drop": float(early_act - late_act),
        "growth": float(np.log((pop[-1] + 1.0) / (pop[0] + 1.0))),
        "alive": float(pop[-1] > 0),
        "spatial_entropy": block_entropy(eng.grid),
        "period": _detect_period(eng, max_p=max_period),
    }
    return feats


def measure_mean(lawset: LawSet, size: int = 64, steps: int = 200, reps: int = 2,
                 seed: int = 0) -> dict:
    """Average the feature vector over ``reps`` random initial conditions, so a
    rule is characterized by its typical behavior rather than one lucky soup."""
    acc: dict[str, float] = {}
    for r in range(reps):
        f = measure_run(lawset, size=size, steps=steps, seed=seed + r)
        for k, v in f.items():
            acc[k] = acc.get(k, 0.0) + v
    return {k: v / reps for k, v in acc.items()}
