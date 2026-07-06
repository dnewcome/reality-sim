# Adding a Universe

There are two very different kinds of "add a universe," and it's worth being
clear about which one you're doing:

1. **A new rule in an existing family** (a new life-like rule, a new
   excitable variant) — this is a pure data change: one new `LawSet` entry in
   `reality_sim/lawsets.py`. No engine, server, or frontend code changes.
2. **A whole new kind of physics** (a new engine family — continuum fields,
   N-body, anything that isn't outer-totalistic-on-a-Moore-neighborhood or
   Greenberg-Hastings) — this means writing a new `Engine` subclass and
   registering it.

Most of the time you want #1. This doc covers both, in that order.

## 1. A new universe in an existing family

### The `LawSet` fields you control

Every entry in `reality_sim/lawsets.py` is built with `register(LawSet(...))`.
The fields:

| field | meaning |
|---|---|
| `id` | short stable identifier — used in URLs / the `set_lawset` command |
| `name` | human label shown in the picker |
| `description` | what this universe *is* / what emerges — shown under the picker |
| `family` | which engine evolves it; a key in `engine.ENGINES` (`"life"`, `"excitable"`, `"forestfire"`, `"totalistic"`, `"lenia"`, or `"levelset"` today) |
| `states` | number of distinct cell states, `0..states-1` |
| `params` | family-specific rule parameters (see below) |
| `palette` | one `"#rrggbb"` string per state, index == state value |
| `seed` | initial-condition recipe: `{"kind": "random", "density": ...}` or `{"kind": "clear"}` |
| `controls` | UI spec for the live-tunable knobs — the widgets the viewer renders and whose changes come back as `set_param` (see "Live-tunable controls" below) |

For the **`"life"`** family, `params` is `{"birth": [...], "survival": [...]}`
— lists of Moore-neighbor counts (`0..8`). `LifeEngine` turns each list into a
boolean lookup table indexed by neighbor count: a dead cell is born if its
live-neighbor count is in `birth`; a live cell survives if its count is in
`survival`. Anything not listed in `survival` dies; anything not listed in
`birth` stays dead. Nothing enforces it in code, but every shipped `"life"`
LawSet uses `states=2` (dead/alive) — that's the natural fit for a
birth/survival rule over a two-state grid.

For the **`"excitable"`** family, `params` is `{"threshold": n}` — the number
of currently-excited (state `1`) Moore neighbors a resting (state `0`) cell
needs to fire. `states` sets how many refractory steps a cell spends
recovering (state `k` always advances to `k+1`, wrapping `states-1` back to
`0`). See the gotcha below before picking a threshold.

For the **`"forestfire"`** family (stochastic, `states=3`: empty/tree/fire),
`params` is `{"p": ..., "f": ...}` — the per-cell probabilities that an empty
cell grows a tree (`p`) and that a tree spontaneously ignites by lightning
(`f`). A tree also ignites if any Moore neighbor is burning; fire always dies to
empty next step. With `f` << `p` the model self-organizes to criticality.

The **`"totalistic"`** family is different in kind: it is a *generative* family
whose `params` are `{"radius": r, "table": [[...]]}`, where `table` has shape
`(states, max_sum + 1)` and the rule is `next = table[state, neighbor_sum]`
(`neighbor_sum` = the summed states of the radius-`r` Moore neighbors). Because
the whole rule *is* a table, randomizing it generates new automaton **types**,
not just new parameters — this is what backs the "🎲 New random universe" button's
generated types (`lawsets.random_type()`), which rolls several and keeps the most
interesting by the [sweep metrics](metrics.md). You normally generate these rather
than hand-write them.

The **`"lenia"`** family is *continuous* — a different kind of automaton entirely.
Cells are real values in `[0, 1]` (not discrete states); the rule is a smooth
radial **kernel** plus a **growth function** `G(u) = 2·exp(-((u-μ)²)/(2σ²)) - 1`,
integrated as `A += dt·G(K*A)`. `params` are `{R, mu, sigma, dt, beta}` (kernel
radius, growth center/width, timestep, and ring weights). For streaming/rendering
the float field is quantized to `uint8` with `states=256` and a 256-entry gradient
palette (`gradient_palette(...)`), so the rest of the stack is unchanged. `mu`/`sigma`
are live-tunable and morph the "creatures" in place. `random_lenia()` generates and
curates continuous types the same way `random_type()` does for discrete ones. This
is the first *continuous* primitive of the eventual rule-grammar.

The **`"levelset"`** family is *geometric*: the state is a signed distance field
(negative inside a shape, positive outside), and the rule evolves the *interface*.
`params` are `{grow, tension, reinit}` — the constant normal speed (grow/erode),
the surface-tension (curvature-flow) weight, and how often the field is re-distanced.
Each step shifts the field by `grow`, adds `tension·κ` (mean curvature), and every
`reinit` steps recomputes the exact SDF from the shape via
`scipy.ndimage.distance_transform_edt` (the field must evolve *continuously* between
re-distancings or a hard `phi<0` threshold would freeze sub-cell motion). Rendered
by quantizing the field to `uint8` (states=256) with a diverging palette. `grow` is
allowed to go negative (erosion); `random_levelset_lawset()` samples it for the dice.

### Live-tunable controls

The `controls` field is what makes a universe's knobs appear in the viewer. Each
entry is a dict the frontend renders as a widget, and each change is sent back as
a `{"cmd": "set_param", "key": ..., "value": ...}` message. Three types:

| type | widget | value | used by |
|---|---|---|---|
| `set9` | nine toggle chips for a subset of `{0..8}` | a list of ints | life `birth` / `survival` |
| `int` | integer slider (`min`/`max`/`step`) | an int | excitable `threshold` / `states` |
| `float` | float slider (`min`/`max`/`step`) | a float | forestfire `p` / `f`, seed `density` |

The special key `"density"` tunes the `seed` recipe (it takes effect on the next
reseed, not immediately). Build the list with the helpers in `lawsets.py`
(`life_controls()`, `excitable_controls()`, `forestfire_controls()`) or hand-write
dicts. When a knob changes, the server mutates a per-session copy of the LawSet
(`dataclasses.replace`) and calls **`engine.reconfigure(new_lawset)`** — which
updates the rule *in place, keeping the current grid*, so a running pattern
reacts to the changed law instead of being reset.

### Worked example: Seeds (a life-like rule)

Seeds (`B2/S`) is a good second example precisely because it's *not* like
Conway: no cell ever survives — every live cell dies every generation — so
the only thing keeping the universe going is birth on exactly 2 neighbors.
The result is explosive, non-settling growth rather than Conway's mix of
stable/oscillating/traveling structures.

```python
register(LawSet(
    id="seeds",
    name="Seeds",
    description=(
        "B2/S — every live cell dies every generation; the universe only "
        "persists by birth on exactly 2 neighbors. No stable structures, "
        "just relentless sparking growth."
    ),
    family="life",
    states=2,
    params={"birth": [2], "survival": []},
    palette=["#0b0f1a", "#ff6b6b"],
    seed={"kind": "random", "density": 0.1},
))
```

Notes on the choices:

- `"survival": []` is deliberate, not an oversight — it's what makes Seeds
  *Seeds*. `LifeEngine` builds `_survival_lut` from `set(int(x) for x in
  params.get("survival", [2, 3]))`; an empty list gives an all-`False` LUT,
  so no live cell ever survives regardless of neighbor count.
- The seed density is low (`0.1` vs. Conway's `0.22`) because Seeds grows
  explosively from almost any live cell — starting dense just floods the
  grid in a few generations.
- Add the entry anywhere in the "life-like universes" section of
  `lawsets.py`; registration order is display order (`ORDER` is a plain
  list, appended to by `register()`).

### A second one-liner: Maze

Maze (`B3/S12345`) is another life-like rule worth knowing about: it births
like Conway (on exactly 3 neighbors) but survives across a much wider band
(`1` through `5`), which makes it fill space with stable corridor-like
structures instead of gliders:

```python
params={"birth": [3], "survival": [1, 2, 3, 4, 5]}
```

Same family, same engine, same server and frontend code — only the two lists
change.

## 2. A whole new engine family

When a new physics genuinely doesn't fit "outer-totalistic rule over a Moore
neighborhood" or "Greenberg-Hastings excitable medium," write a new `Engine`
subclass in `reality_sim/engine.py` (or your own module, since `ENGINES` is
just a plain module-level dict you can add to):

1. **Subclass `Engine`** and set a class-level `family` string — this is the
   key `make_engine()` will look up.
2. **Implement `seed(self)`** — build `self.grid` from `self.lawset.seed`
   (`recipe.get("kind")`, typically `"random"` or `"clear"`), and reset
   `self.generation = 0`. Follow `LifeEngine`/`ExcitableEngine`'s pattern of
   raising `ValueError` on an unrecognized `kind`.
3. **Implement `step(self)`** — advance `self.grid` by exactly one generation
   using vectorized numpy/scipy (no per-cell Python loops — that's what keeps
   a step under a millisecond even at large grid sizes), and increment
   `self.generation`.
4. **Optionally override `stats(self)`** to report family-specific numbers —
   `ExcitableEngine` adds an `"excited"` count on top of the base
   `generation`/`live`/`density`.
5. **Keep `self.grid` a 2-D `uint8` array with values in `0..lawset.states-1`**
   — this is the one invariant `server.py` and `frontend/app.js` depend on;
   everything downstream of the engine is generic *because* they never see
   anything else.
6. **Register it**: add `ENGINES["myfamily"] = MyEngine` (or add the class
   directly into the `ENGINES` dict literal in `engine.py` if you're editing
   the package itself).
7. **Tag matching `LawSet`s** with `family="myfamily"` in `lawsets.py`
   (or wherever you register universes) — `make_engine()` picks up the new
   family automatically, with no other code path to touch.
8. **(Optional) Support live tuning** — override `reconfigure(self, lawset)`
   (call `super().reconfigure(lawset)` first) to adopt changed `params` in place
   without reseeding, keeping `self.grid`; and give your `LawSet`s a `controls`
   list so the knobs appear in the viewer. Skip this and the universe still runs
   fine — it just won't be live-editable. `ForestFireEngine` is the smallest
   worked example of a stochastic family with both.

That's the entire seam: `LawSet.family` plus the `ENGINES` dict is what makes
the system pluggable, and it's intentionally the *only* place new physics has
to be wired in.

### (Optional) React to the environment field

The base `Engine` carries an optional **environment field** — a spatial gradient
`F(x, y) ∈ [-1, 1]` a universe can be dropped into (see the "Fields" section of the
README). It's universe-agnostic: the field is the *same* array whatever the family,
and each engine decides what it *means*. The base class does the plumbing (building
the field, its gradient, and the strength scaling); your `step()` just reads whichever
of these is convenient and skips the block when there's no field:

- `self.env_bias` — the per-cell push `strength · F` in `[-1, 1]`, or `None` when no
  field is set. This is what the *rate-based / continuous* families use: forest-fire
  scales its growth `p` by `1 + 3·env_bias`, level-set adds `0.6·env_bias` to its
  normal speed, excitable turns firing into a coin-flip with probability `0.5 + env_bias`.
- `self.env_gx`, `self.env_gy` — the field's gradient, for **drift**: Lenia advects its
  field up-gradient (`A -= speed · (∇F/|∇F|)·∇A`) so structures migrate toward high `F`.
- `self._env_habitable()` — a boolean mask (or `None`) for the *discrete* families:
  cells may only live where `F` clears a strength-scaled threshold. Life and the
  totalistic type both `&=` (or zero) their next grid with it to confine a pattern.

The one rule: **guard on `None`** (`if self.env_bias is not None:`) so a universe with no
field runs exactly as before. Everything else — building the field from the viewer's
shape/strength/angle, re-applying it after a resize or a universe switch, and streaming
a thumbnail for the overlay — is handled in `server.py` (`_apply_field`) with no
per-family code. If your new family has an obvious "how things move / grow / survive"
knob, coupling it to the field is usually two or three lines.

## Gotcha: the excitable engine needs `threshold=1` to sustain waves

The shipped `"excitable"` LawSet uses `params={"threshold": 1}` with `states =
16`, and that's not an arbitrary default — it was found by sweeping the
parameter. `ExcitableEngine`'s rule is: a resting cell fires when its count of
*excited* (state `1`, not refractory) Moore neighbors meets `threshold`. With
16 states, a traveling wavefront is thin — only a one-cell-wide ring of cells
is actually in state `1` at any given tick, with everything behind it already
advanced into refractory states `2..15`.

That thinness is exactly why `threshold >= 2` kills the dynamics once `states
>= 8`: a resting cell sitting just ahead of a thin, mostly-convex wavefront
typically only borders *one* excited cell at a time, not two. Demanding two
simultaneously-excited neighbors means most of the front simply fails to
propagate past its first tick or two — the wave induces briefly, then dies
out, rather than sustaining the self-organizing spiral/target waves the rule
is supposed to produce. `threshold=1` is what actually lets those waves
survive and propagate indefinitely. If you add a new excitable-family
universe with a different `states` count or a different intended threshold,
treat this as a real constraint to re-check, not a knob to tune blind —
verify waves actually sustain past the first few dozen generations before
registering it.
