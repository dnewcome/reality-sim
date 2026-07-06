"""LawSet: the portable specification of one toy universe's *physics*.

The organizing idea of this whole package: **inventing new physics == writing a
new LawSet.** A LawSet carries just enough to (a) build a numpy engine that
evolves that universe forward in time, and (b) tell the viewer how to draw it.
Everything downstream (engines, server, frontend, probes) is generic over it.

A LawSet is intentionally *data*, not code — it can be serialized, logged,
mutated, or searched over (that last one is where the ML-driven rule search will
eventually plug in: a rule is just a point in LawSet-space).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class LawSet:
    """One universe's laws.

    Attributes
    ----------
    id : short stable identifier, used in URLs / commands.
    name : human label.
    description : what this universe *is* / what emerges in it.
    family : which engine evolves it. Must be a key in ``engine.ENGINES``
             (currently "life" or "excitable"). This is the pluggability seam:
             new physics families register a new engine + a matching family tag.
    states : number of distinct cell states (0 .. states-1).
    params : family-specific parameters (e.g. birth/survival sets for "life",
             excitation threshold for "excitable").
    palette : one "#rrggbb" per state, index == state value. The viewer maps
              raw state -> color with this, so the universe's *appearance* is
              part of its spec.
    seed : default initial-condition recipe, e.g. {"kind": "random",
           "density": 0.22} or {"kind": "clear"}.
    controls : UI spec for the *live-tunable* knobs of this universe. Each entry
               is a dict {key, label, type, ...} the viewer turns into a widget,
               and whose changes are sent back as `set_param` commands. Types:
               "set9" (a subset of {0..8} as toggle chips, e.g. life's B/S),
               "int"/"float" (a slider with min/max/step). The special key
               "density" tunes the seed recipe (applied on the next reseed).
    """

    id: str
    name: str
    description: str
    family: str
    states: int
    params: dict[str, Any] = field(default_factory=dict)
    palette: list[str] = field(default_factory=list)
    seed: dict[str, Any] = field(default_factory=lambda: {"kind": "random", "density": 0.25})
    controls: list[dict] = field(default_factory=list)

    def to_public(self) -> dict:
        """JSON-serializable view sent to the browser."""
        return asdict(self)
