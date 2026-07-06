"""The library of concrete universes — the actual "laws of physics" you can run.

Each entry is a :class:`~reality_sim.lawset.LawSet`, and now each also declares
its **live-tunable controls**: the knobs the viewer renders as widgets so you can
edit a universe's law while it runs (toggle Conway's birth rules, stretch the
excitable medium's refractory tail, tip the forest between smoldering and
firestorm). Adding a universe is adding an entry here.
"""

from __future__ import annotations

from .lawset import LawSet

LIBRARY: dict[str, LawSet] = {}
ORDER: list[str] = []


def register(ls: LawSet) -> LawSet:
    LIBRARY[ls.id] = ls
    ORDER.append(ls.id)
    return ls


# --- control-spec builders (the UI knobs each family exposes) -----------------

def life_controls() -> list[dict]:
    return [
        {"key": "birth", "label": "birth  B", "type": "set9"},
        {"key": "survival", "label": "survival  S", "type": "set9"},
        {"key": "density", "label": "seed density", "type": "float", "min": 0.05, "max": 0.95, "step": 0.01},
    ]


def excitable_controls() -> list[dict]:
    return [
        {"key": "threshold", "label": "ignition threshold", "type": "int", "min": 1, "max": 8, "step": 1},
        {"key": "states", "label": "refractory length", "type": "int", "min": 3, "max": 32, "step": 1},
        {"key": "density", "label": "seed density", "type": "float", "min": 0.05, "max": 1.0, "step": 0.05},
    ]


def forestfire_controls() -> list[dict]:
    return [
        {"key": "p", "label": "tree growth  p", "type": "float", "min": 0.0, "max": 0.3, "step": 0.002},
        {"key": "f", "label": "lightning  f", "type": "float", "min": 0.0, "max": 0.02, "step": 0.0002},
        {"key": "density", "label": "initial forest", "type": "float", "min": 0.0, "max": 1.0, "step": 0.02},
    ]


# --- palettes -----------------------------------------------------------------

_LIFE_PALETTE = ["#0b0f1a", "#e8f0ff"]


def excitable_palette(n: int) -> list[str]:
    """Resting = near-black indigo; excited = hot cyan-white; refractory fades
    back down to deep indigo. Public because the live `states` knob regenerates it
    when the refractory length changes."""
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
        "capable of universal computation. Toggle a birth/survival bit to feel the "
        "edge of chaos move."
    ),
    family="life",
    states=2,
    params={"birth": [3], "survival": [2, 3]},
    palette=_LIFE_PALETTE,
    seed={"kind": "random", "density": 0.22},
    controls=life_controls(),
))

register(LawSet(
    id="highlife",
    name="HighLife",
    description=(
        "B36/S23 — almost Conway, but birth-on-6 gives it a genuine self-replicator. "
        "Turn B6 off live and watch it become Conway."
    ),
    family="life",
    states=2,
    params={"birth": [3, 6], "survival": [2, 3]},
    palette=["#0b0f1a", "#ffd76a"],
    seed={"kind": "random", "density": 0.22},
    controls=life_controls(),
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
    controls=life_controls(),
))


# --- excitable-medium universe (multi-state) ---------------------------------

_EXCITABLE_STATES = 16
register(LawSet(
    id="excitable",
    name="Excitable Medium",
    description=(
        "Greenberg-Hastings — self-organizing spiral waves with a definite signal "
        "speed. Stretch the refractory length or raise the ignition threshold to "
        "reshape the waves."
    ),
    family="excitable",
    states=_EXCITABLE_STATES,
    params={"threshold": 1},
    palette=excitable_palette(_EXCITABLE_STATES),
    seed={"kind": "random", "density": 1.0},
    controls=excitable_controls(),
))


# --- forest-fire universe (stochastic, self-organized criticality) -----------

register(LawSet(
    id="forestfire",
    name="Forest Fire",
    description=(
        "Drossel-Schwabl — the first stochastic universe. Empty grows trees (rate p); "
        "lightning (rate f) or a burning neighbor ignites them; fire dies to ash. With "
        "f << p it self-tunes to the critical point: a universe that finds the edge of "
        "chaos on its own."
    ),
    family="forestfire",
    states=3,
    params={"p": 0.03, "f": 0.0006},
    palette=["#0d130d", "#2f9e44", "#ff7038"],  # empty, tree, fire
    seed={"kind": "random", "density": 0.4},
    controls=forestfire_controls(),
))


def get(lawset_id: str) -> LawSet:
    return LIBRARY[lawset_id]


def catalog() -> list[dict]:
    """Ordered, JSON-serializable list of every universe, for the picker UI."""
    return [LIBRARY[i].to_public() for i in ORDER]


DEFAULT_ID = "conway"
