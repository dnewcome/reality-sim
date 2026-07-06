"""Unbounded ("infinite") life-like universes via a hash map of tiles.

The toroidal engines in :mod:`reality_sim.engine` live on a fixed w x h grid. This
is the other option: an *unbounded* plane. The world is stored as a dict of small
fixed-size tiles ``{(tx, ty): uint8[T, T]}``, and **only tiles that contain (or
border) live cells exist** — so memory scales with the live *population*, not the
area. A glider can fly away from the origin forever while the engine holds just a
handful of tiles.

This works for any life-like rule *without* B0 (birth on an empty neighborhood):
B0 would turn the infinite vacuum live everywhere at once, which has no finite
representation. Conway and its usual kin all have B0 = 0, so this is exactly the
family of "a finite pattern on an empty background" universes.

The engine renders any window of the plane on demand via :meth:`viewport`, which
is how the server streams a movable camera over an endless world.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .engine import MOORE
from .lawset import LawSet

TILE = 64


class ChunkedLifeEngine:
    """A life-like CA on an unbounded plane, stored as sparse tiles."""

    family = "life"

    def __init__(self, lawset: LawSet, shape: tuple[int, int], rng: np.random.Generator, tile: int = TILE):
        self.T = int(tile)
        self.rng = rng
        self.view_h, self.view_w = shape  # size of the default seed patch
        self.lawset = lawset
        self._build_luts(lawset)
        self.tiles: dict[tuple[int, int], np.ndarray] = {}
        self.generation = 0
        self.seed()

    # -- rule ------------------------------------------------------------
    def _build_luts(self, lawset: LawSet) -> None:
        birth = set(int(x) for x in lawset.params.get("birth", [3]))
        survival = set(int(x) for x in lawset.params.get("survival", [2, 3]))
        self._birth_lut = np.array([n in birth for n in range(9)], dtype=bool)
        self._survival_lut = np.array([n in survival for n in range(9)], dtype=bool)
        self.has_b0 = 0 in birth  # infinite mode is only well-defined without this

    def reconfigure(self, lawset: LawSet) -> None:
        self.lawset = lawset
        self._build_luts(lawset)

    def clear(self) -> None:
        self.tiles.clear()
        self.generation = 0

    # -- tile region I/O -------------------------------------------------
    def _get_region(self, wx: int, wy: int, W: int, H: int) -> np.ndarray:
        """Render the world rectangle [wx, wx+W) x [wy, wy+H) to a dense array."""
        out = np.zeros((H, W), dtype=np.uint8)
        T = self.T
        for ty in range(math.floor(wy / T), math.floor((wy + H - 1) / T) + 1):
            for tx in range(math.floor(wx / T), math.floor((wx + W - 1) / T) + 1):
                tile = self.tiles.get((tx, ty))
                if tile is None:
                    continue
                ox0, ox1 = max(wx, tx * T), min(wx + W, (tx + 1) * T)
                oy0, oy1 = max(wy, ty * T), min(wy + H, (ty + 1) * T)
                out[oy0 - wy:oy1 - wy, ox0 - wx:ox1 - wx] = \
                    tile[oy0 - ty * T:oy1 - ty * T, ox0 - tx * T:ox1 - tx * T]
        return out

    def _set_cell(self, x: int, y: int, value: int) -> None:
        T = self.T
        tx, ty = math.floor(x / T), math.floor(y / T)
        tile = self.tiles.get((tx, ty))
        if tile is None:
            if value == 0:
                return
            tile = np.zeros((T, T), dtype=np.uint8)
            self.tiles[(tx, ty)] = tile
        tile[y - ty * T, x - tx * T] = value

    def _stamp(self, wx: int, wy: int, arr: np.ndarray) -> None:
        ys, xs = np.nonzero(arr)
        for y, x in zip(ys.tolist(), xs.tolist()):
            self._set_cell(wx + x, wy + y, int(arr[y, x]))

    # -- lifecycle -------------------------------------------------------
    def seed(self) -> None:
        self.tiles.clear()
        self.generation = 0
        recipe = self.lawset.seed
        kind = recipe.get("kind", "random")
        if kind == "clear":
            return
        if kind == "random":
            density = float(recipe.get("density", 0.3))
            h, w = self.view_h, self.view_w
            patch = (self.rng.random((h, w)) < density).astype(np.uint8)
            self._stamp(-w // 2, -h // 2, patch)  # a finite soup centered on origin
        else:
            raise ValueError(f"unknown seed kind: {kind!r}")

    def _padded(self, tx: int, ty: int) -> np.ndarray:
        """A (T+2)x(T+2) block: this tile's cells plus a one-cell halo stitched
        from the eight neighbor tiles, so a single convolution gives correct
        neighbor counts across tile seams."""
        T = self.T
        p = np.zeros((T + 2, T + 2), dtype=np.uint8)
        g = self.tiles.get
        c = g((tx, ty))
        if c is not None:
            p[1:-1, 1:-1] = c
        top, bot = g((tx, ty - 1)), g((tx, ty + 1))
        left, right = g((tx - 1, ty)), g((tx + 1, ty))
        if top is not None:
            p[0, 1:-1] = top[-1, :]
        if bot is not None:
            p[-1, 1:-1] = bot[0, :]
        if left is not None:
            p[1:-1, 0] = left[:, -1]
        if right is not None:
            p[1:-1, -1] = right[:, 0]
        tl, tr = g((tx - 1, ty - 1)), g((tx + 1, ty - 1))
        bl, br = g((tx - 1, ty + 1)), g((tx + 1, ty + 1))
        if tl is not None:
            p[0, 0] = tl[-1, -1]
        if tr is not None:
            p[0, -1] = tr[-1, 0]
        if bl is not None:
            p[-1, 0] = bl[0, -1]
        if br is not None:
            p[-1, -1] = br[0, 0]
        return p

    def step(self) -> None:
        # New cells can only appear within one cell of an existing live cell, so
        # only currently-occupied tiles and their 8 neighbors can be non-empty next.
        candidates: set[tuple[int, int]] = set()
        for (tx, ty) in self.tiles:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    candidates.add((tx + dx, ty + dy))

        new_tiles: dict[tuple[int, int], np.ndarray] = {}
        for (tx, ty) in candidates:
            padded = self._padded(tx, ty)
            counts = ndimage.convolve(padded, MOORE, mode="constant")[1:-1, 1:-1]
            center = padded[1:-1, 1:-1].astype(bool)
            nxt = (~center & self._birth_lut[counts]) | (center & self._survival_lut[counts])
            if nxt.any():
                new_tiles[(tx, ty)] = nxt.astype(np.uint8)
        self.tiles = new_tiles
        self.generation += 1

    def paint(self, x: int, y: int, value: int, radius: int = 0) -> None:
        """Paint a world cell (or a disk of world cells) — coordinates are world,
        not viewport, so the server translates before calling."""
        value = int(value) % self.lawset.states
        if radius <= 0:
            self._set_cell(int(x), int(y), value)
            return
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    self._set_cell(int(x) + dx, int(y) + dy, value)

    # -- views / introspection ------------------------------------------
    def viewport(self, cx: int, cy: int, vw: int, vh: int, zoom: int = 1):
        """Render a vw x vh camera window centered on world (cx, cy). At zoom > 1
        each pixel OR-reduces a zoom x zoom block, so you can pull back and watch a
        huge pattern. Returns (frame, world_x0, world_y0)."""
        zoom = max(1, int(zoom))
        W, H = vw * zoom, vh * zoom
        wx, wy = cx - W // 2, cy - H // 2
        region = self._get_region(wx, wy, W, H)
        if zoom > 1:
            region = region.reshape(vh, zoom, vw, zoom).max(axis=(1, 3))
        return region.astype(np.uint8), wx, wy

    def population(self) -> int:
        return int(sum(int(np.count_nonzero(t)) for t in self.tiles.values()))

    def bbox(self):
        """World-coordinate bounding box of live cells, or None if empty."""
        if not self.tiles:
            return None
        xs0 = ys0 = 10**18
        xs1 = ys1 = -10**18
        T = self.T
        for (tx, ty), tile in self.tiles.items():
            ys, xs = np.nonzero(tile)
            if not len(xs):
                continue
            xs0 = min(xs0, tx * T + int(xs.min())); xs1 = max(xs1, tx * T + int(xs.max()))
            ys0 = min(ys0, ty * T + int(ys.min())); ys1 = max(ys1, ty * T + int(ys.max()))
        if xs1 < xs0:
            return None
        return (xs0, ys0, xs1, ys1)

    def center_of_mass(self):
        b = self.bbox()
        if b is None:
            return (0, 0)
        x0, y0, x1, y1 = b
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    def stats(self) -> dict:
        return {
            "generation": self.generation,
            "population": self.population(),
            "tiles": len(self.tiles),
        }
