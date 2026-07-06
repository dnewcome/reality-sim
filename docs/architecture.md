# Architecture

reality-sim is four small layers stacked on one idea: **a law of physics is
data, not code.** Everything else in the system — the numpy engines, the
websocket server, the canvas renderer — exists to turn that data into motion
and pixels without ever needing to know what specific universe it's looking
at.

## The core idea: `LawSet` is a value, not a program

`reality_sim/lawset.py` defines the whole vocabulary of "a universe" as one
frozen dataclass:

```python
@dataclass(frozen=True)
class LawSet:
    id: str
    name: str
    description: str
    family: str
    states: int
    params: dict[str, Any] = field(default_factory=dict)
    palette: list[str] = field(default_factory=list)
    seed: dict[str, Any] = field(default_factory=lambda: {"kind": "random", "density": 0.25})
```

No methods that do anything, no imports of numpy, no reference to any engine
class. `family` is a string key, `params` is a plain dict, `palette` is a list
of `"#rrggbb"` strings, `seed` is a recipe dict. That's deliberate: a `LawSet`
"can be serialized, logged, mutated, or searched over" (its own docstring) —
because it's inert data, the exact same object that builds the numpy engine
(`make_engine`) is also the exact object JSON-serialized to the browser
(`LawSet.to_public()` → `asdict(self)`) for the universe picker. There's no
second description of a universe to keep in sync.

This is also why the palette lives *inside* the LawSet rather than being a
frontend concern: "the universe's appearance is part of its spec" (again, the
docstring). A universe's look is as much a law of physics as its update rule.

## Data flow

```
 ┌────────────────────────────────────────────────────────────────┐
 │ LawSet                     reality_sim/lawset.py               │
 │   id, name, description, family, states, params, palette, seed │
 │   — frozen dataclass, no behavior                              │
 └──────────────────────────────────┬─────────────────────────────┘
                                    │ make_engine(lawset, shape, rng)
                                    │ cls = ENGINES[lawset.family]
                                    ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ Engine family              reality_sim/engine.py               │
 │   Engine: grid uint8[h,w], generation, seed(), step()          │
 │     ├─ LifeEngine       family="life"      birth/survival LUTs │
 │     └─ ExcitableEngine  family="excitable" Greenberg-Hastings  │
 └──────────────────────────────────┬─────────────────────────────┘
                                    │ .grid / .generation / .step() / .stats()
                                    ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ Session                     reality_sim/server.py              │
 │   one per websocket connection                                 │
 │   sim loop: drain queued cmds → engine.step() → send_frame()   │
 │   — the ONLY coroutine that touches engine or writes the socket│
 └──────────────────────────────────┬─────────────────────────────┘
              ▲                     │ binary frame + JSON status/catalog
   JSON cmds                         ▼
 ┌────────────────────────────────────────────────────────────────┐
 │ frontend                     frontend/app.js                   │
 │   onBinary() → palette lookup → putImageData → drawImage       │
 │   UI controls → send({cmd: ...}) back over the same socket     │
 └────────────────────────────────────────────────────────────────┘
```

### 1. `LawSet` → `Engine` (`make_engine`, the `ENGINES` registry)

`make_engine()` in `reality_sim/engine.py` is the entire wiring:

```python
def make_engine(lawset, shape, rng=None):
    cls = ENGINES[lawset.family]
    return cls(lawset, shape, rng)
```

`ENGINES` is a plain `dict[str, type[Engine]]` — currently
`{"life": LifeEngine, "excitable": ExcitableEngine}`. If `lawset.family` isn't
a key, it raises a `ValueError` naming the known families instead of failing
silently later. This lookup is the single point where "which physics" gets
resolved into "which numpy code runs."

`Engine` (the base class) owns the universal parts every family needs
regardless of its rule: a `uint8` grid of shape `(h, w)`, a `generation`
counter, an `rng`, and the lifecycle methods `seed()`, `step()`, `stats()`,
`resize()`, and `paint()` (the brush). Subclasses only have to supply the
physics:

- **`LifeEngine`** (`family = "life"`) implements two-state outer-totalistic
  "life-like" rules: a birth set `B` and survival set `S` over the Moore
  neighbor count. It turns `lawset.params["birth"]` / `["survival"]` into
  boolean lookup tables indexed `0..8` (`_birth_lut`, `_survival_lut`) so
  applying the rule to the whole grid is one fancy-index, `born =
  ~alive & self._birth_lut[counts]`, with no per-cell branching.
- **`ExcitableEngine`** (`family = "excitable"`) implements a Greenberg-Hastings
  excitable medium: state `0` is resting, `1` is excited, `2..n-1` are
  refractory. A resting cell fires when its count of excited neighbors meets
  `lawset.params["threshold"]`; every non-resting cell just advances one step,
  wrapping back to rest. It's a genuinely different kind of physics from Life
  — traveling spiral waves instead of static gliders — built from the same
  `Engine` contract.

Both keep `self.grid` as a strict 2-D `uint8` array with values `0..states-1`.
That single invariant is what lets `server.py` and `app.js` stay completely
generic over the family: they never branch on which physics produced the
bytes.

### 2. `Engine` → `server.py` (`Session`, the single-writer sim loop)

`server.py` is explicit about its scope: "one universe per browser connection
— this is a local research tool, not a multi-tenant service." Each websocket
connection gets a `Session` with two coroutines:

- **`read_commands()`** — a reader task that does nothing but drain incoming
  JSON text frames onto an `asyncio.Queue`.
- **`run()`** — the sim loop, and the *only* coroutine that ever touches
  `self.engine` or calls `self.ws.send_*`. Each tick it: drains everything
  queued since last tick through `handle()`, steps the engine if `playing`,
  sends a binary frame if `frame_dirty`, sends a JSON status if
  `status_dirty`, then sleeps `1/fps`.

Making the sim loop the sole writer is the whole point: "no interleaved
websocket sends, no locks." Commands arrive concurrently from the reader task,
but they only ever get *applied* — never acted on directly — inside the one
loop that owns the engine.

The two dirty flags (`frame_dirty`, `status_dirty`) are a bandwidth choice:
a new binary frame goes out whenever the grid actually changed (a step, a
paint stroke, a reseed), but the small JSON `status` message only goes out on
*structural* change — play/pause, fps, size, or lawset swapped — not every
tick. See `docs/protocol.md` for the exact message shapes.

### 3. `server.py` → frontend (canvas render)

`frontend/app.js`'s `onBinary()` parses the 12-byte header with a `DataView`
and wraps the remaining bytes as a zero-copy `Uint8Array` view directly over
the incoming `ArrayBuffer` — no JSON, no per-cell parsing. `render()` then:

1. Looks up each cell's state in a `Uint32Array` palette (`palette32`,
   built by `setPalette()` from the LawSet's `palette` hex strings) to get a
   packed little-endian RGBA value — one table lookup per cell, no string
   work in the hot path.
2. Writes those uint32s directly into an offscreen canvas's `ImageData`
   buffer (`pix32`, a `Uint32Array` view over `imgData.data.buffer`) at the
   grid's native resolution.
3. `drawImage`s that offscreen canvas onto the visible one, scaled up by an
   integer factor with `imageSmoothingEnabled = false` — nearest-neighbor
   upscaling so cells stay crisp squares instead of blurry gradients.

## Key design choices

| Choice | Where | Why |
|---|---|---|
| Vectorized numpy + `scipy.ndimage.convolve(mode="wrap")` | `engine.py`, `MOORE` kernel | No Python-level per-cell loops — a step is a couple of array ops, fast enough that a 512×512 grid steps in well under a millisecond. `mode="wrap"` makes every universe a torus (no edge special-casing) and is *also* what gives each CA its hard causal light cone: a convolution with a 3×3 kernel can only pull information from 1 cell away, so nothing can influence a cell faster than 1 cell per generation — the discrete analogue of a speed-of-light limit (see `docs/concepts.md`). |
| `uint8` grids everywhere | `Engine.grid` | Caps any universe at 256 states, but keeps the frame format trivial (one byte per cell, no encoding) and lets the wire format be literally `grid.tobytes()`. |
| One sim loop per connection, sole websocket writer | `Session.run()` | Eliminates a whole class of bugs (interleaved/out-of-order sends) without needing a lock — there's only ever one coroutine that could contend for one. |
| Palette lives in the `LawSet` | `LawSet.palette`, `_LIFE_PALETTE`, `_excitable_palette()` | A universe's appearance ships with its physics; the frontend never hardcodes "state 1 is white" — it always asks the current LawSet. |

## Extension seams

The pluggability point for the whole system is exactly two things working
together:

1. `LawSet.family` — a string tag on the data.
2. `ENGINES` — the `dict[str, type[Engine]]` registry `make_engine()`
   consults.

Adding a **new universe in an existing family** (a new life-like rule, a new
excitable variant) touches zero engine or server code — it's a new `LawSet`
entry in `reality_sim/lawsets.py`. Adding a **new kind of physics** means
writing a new `Engine` subclass and adding it to `ENGINES` under a new family
key; every `LawSet` that sets `family` to that key is automatically routed to
it, and the server/frontend need no changes at all since they only ever see
`grid`, `generation`, `stats()`, and `palette`. Worked examples of both are in
`docs/adding-a-universe.md`.
