"""reality-sim: simulate toy universes with pluggable laws of physics.

Quick start (headless)::

    import numpy as np
    from reality_sim import make_engine, lawsets

    eng = make_engine(lawsets.get("conway"), shape=(128, 128),
                      rng=np.random.default_rng(0))
    for _ in range(100):
        eng.step()
    print(eng.stats())

Or launch the live web viewer::

    python -m reality_sim.server
"""

from __future__ import annotations

from .lawset import LawSet
from .engine import Engine, LifeEngine, ExcitableEngine, ForestFireEngine, ENGINES, make_engine
from . import lawsets

__all__ = [
    "LawSet",
    "Engine",
    "LifeEngine",
    "ExcitableEngine",
    "ForestFireEngine",
    "ENGINES",
    "make_engine",
    "lawsets",
]

__version__ = "0.1.0"
