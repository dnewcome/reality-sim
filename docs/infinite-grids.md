# Infinite grids

Every other engine lives on a fixed `w × h` **torus** (wrap-around edges). The
life family also has a second boundary mode: an **unbounded plane**, where a
pattern can grow or fly away from the origin forever.

![Gosper glider gun on the infinite plane](img/infinite.png)

*A Gosper glider gun emitting an endless diagonal stream — population grows
without bound (36 → 61 → 136 …) while the engine holds just a handful of tiles.*

## How it works — a hash map of tiles

`reality_sim/chunked.py`'s `ChunkedLifeEngine` stores the world as a dict of
small fixed-size tiles, `{(tile_x, tile_y): uint8[64, 64]}`, and **only tiles that
contain (or border) live cells exist**. Memory scales with the live *population*,
not the area of the world.

Each step:

1. **Candidate tiles** = every occupied tile plus its 8 neighbors (new cells can
   only appear within one cell of an existing live cell).
2. For each candidate, assemble a `(64+2)²` block: the tile's cells plus a
   one-cell **halo** stitched from its eight neighbor tiles, so a single
   convolution gives correct neighbor counts across tile seams.
3. Apply the life-like birth/survival rule; keep the result only if it's non-empty
   (empty tiles are dropped, so the world stays sparse).

A glider tracked for 4,000 generations travels ~1,000 cells diagonally while the
engine holds **exactly one tile** — O(population), not O(area). That's the whole
point.

## The honest caveat: only *localized* universes can be infinite

An infinite grid only helps when activity sits on an **empty background**. Two
consequences:

- **No B0.** A rule that births on 0 neighbors (`B0`) would turn the infinite
  vacuum live *everywhere* at once — no finite representation. Conway and its
  usual kin all have `B0 = 0`, so this is exactly the "finite pattern on empty
  space" family. `ChunkedLifeEngine` exposes `has_b0` for callers that want to
  guard.
- **Space-filling universes stay toroidal.** Forest fire grows trees *everywhere*
  (rate `p` acts on all empty cells) and Day&Night fills space too — their
  populations are genuinely infinite on an infinite plane. So the boundary toggle
  is **life-family only**, and the server auto-reverts to the torus if you switch
  to a space-filling universe. "Infinite" is a per-universe capability, not a
  global mode.

## In the viewer

Flip **boundary → ∞ infinite** (enabled only for life universes). You then get a
**camera** over the endless plane:

- **pan** — the `pan` brush tool (drag to grab-and-pull the world), the arrow
  keys, or… just watch a gun and hit **⊙ center** to recenter on the pattern.
- **zoom** — the `+ in` / `− out` buttons or the scroll wheel; zooming out
  OR-reduces blocks so a huge pattern fits on screen.
- the readout swaps live/density for **population**, live **tile** count, and the
  camera position + zoom.

## The wire protocol addition

The binary frame format is unchanged — the server just streams a **viewport**
(`engine.viewport(cx, cy, w, h, zoom)`) instead of the whole grid, so the same
`w × h` uint8 frame now represents a *window* onto the plane. Because a fixed-size
frame can't carry the camera or the total population, infinite mode adds one small
JSON message per tick:

```json
{"type": "view", "cx": 50, "cy": 30, "zoom": 4, "generation": 812, "population": 214, "tiles": 9}
```

and the client sends `set_boundary`, `pan {dx,dy}`, `zoom {zoom}`, and `recenter`.
Everything else — rule editing via `set_param`, play/pause, painting — works
exactly as in toroidal mode (painting maps the viewport pixel to a world cell
first). See [protocol.md](protocol.md) for the base protocol.
