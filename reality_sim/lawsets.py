"""The library of concrete universes — the actual "laws of physics" you can run.

Each entry is a :class:`~reality_sim.lawset.LawSet`. Adding a universe is adding
an entry here (or building one at runtime and registering it). The four shipped
here span two engine families on purpose, to prove the pluggable-substrate
architecture works from day one.
"""

from __future__ import annotations

from .lawset import LawSet

LIBRARY: dict[str, LawSet] = {}
ORDER: list[str] = []


def register(ls: LawSet) -> LawSet:
    LIBRARY[ls.id] = ls
    ORDER.append(ls.id)
    return ls


# Shared two-state palette: deep space + electric white.
_LIFE_PALETTE = ["#0b0f1a", "#e8f0ff"]


def _excitable_palette(n: int) -> list[str]:
    """Resting = near-black indigo; excited = hot cyan-white; refractory fades
    back down to deep indigo. Makes traveling waves read as glowing ripples."""
    pal = []
    for k in range(n):
        if k == 0:
            pal.append("#060814")
            continue
        t = (k - 1) / max(n - 2, 1)  # 0 at the excited front .. 1 at last refractory
        r = int(220 * (1 - t) + 18 * t)
        g = int(245 * (1 - t) + 26 * t)
        b = int(255 * (1 - t) + 90 * t)
        pal.append(f"#{r:02x}{g:02x}{b:02x}")
    return pal


# --- life-like universes (2-state) -------------------------------------------

register(LawSet(
    id="conway",
    name="Conway's Life",
    description=(
        "B3/S23 — the reference universe. Gliders, oscillators, and structures "
        "capable of universal computation. If a toy cosmos can 'compute', it can here."
    ),
    family="life",
    states=2,
    params={"birth": [3], "survival": [2, 3]},
    palette=_LIFE_PALETTE,
    seed={"kind": "random", "density": 0.22},
))

register(LawSet(
    id="highlife",
    name="HighLife",
    description=(
        "B36/S23 — almost Conway, but the extra birth-on-6 gives it a genuine "
        "self-replicator. The natural first universe for asking 'how fast does "
        "replication (a proxy for a civilization bootstrapping) arise?'"
    ),
    family="life",
    states=2,
    params={"birth": [3, 6], "survival": [2, 3]},
    palette=["#0b0f1a", "#ffd76a"],
    seed={"kind": "random", "density": 0.22},
))

register(LawSet(
    id="daynight",
    name="Day & Night",
    description=(
        "B3678/S34678 — symmetric under swapping live<->dead, so 'matter' and "
        "'vacuum' obey the same law. Grows blobby domains with their own physics-feel."
    ),
    family="life",
    states=2,
    params={"birth": [3, 6, 7, 8], "survival": [3, 4, 6, 7, 8]},
    palette=["#0b0f1a", "#8ef0c0"],
    seed={"kind": "random", "density": 0.5},
))


# --- excitable-medium universe (multi-state) ---------------------------------

_EXCITABLE_STATES = 16
register(LawSet(
    id="excitable",
    name="Excitable Medium",
    description=(
        "Greenberg-Hastings, 16 states, fire on >=1 excited neighbor. A totally "
        "different physics: self-organizing spiral waves with a definite signal "
        "speed. The first testbed for 'how fast can information travel here?'"
    ),
    family="excitable",
    states=_EXCITABLE_STATES,
    params={"threshold": 1},
    palette=_excitable_palette(_EXCITABLE_STATES),
    seed={"kind": "random", "density": 1.0},
))


def get(lawset_id: str) -> LawSet:
    return LIBRARY[lawset_id]


def catalog() -> list[dict]:
    """Ordered, JSON-serializable list of every universe, for the picker UI."""
    return [LIBRARY[i].to_public() for i in ORDER]


DEFAULT_ID = "conway"
