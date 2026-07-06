"""Live web viewer: an aiohttp server that streams a running universe to the
browser and takes control commands back.

Architecture (one universe per browser connection — this is a local research
tool, not a multi-tenant service):

  * A **reader** task drains incoming JSON commands onto a queue.
  * A single **sim loop** is the *only* coroutine that touches the engine or
    writes to the socket. Each tick it applies queued commands, optionally steps
    the universe, and sends one binary frame. Single-writer => no interleaved
    websocket sends, no locks.

Wire protocol
-------------
Client -> server (JSON text): {"cmd": "...", ...}
    play | pause | step | reset | clear
    set_lawset {id}      set_fps {fps}      set_size {w,h}
    paint {r,c,value,radius}

Server -> client:
    JSON  {"type":"catalog", "lawsets":[...], "current":id}   (once, on connect)
    JSON  {"type":"status", lawset, playing, fps, w, h, states}  (on change)
    BINARY frame:  <uint32 w><uint32 h><uint32 generation> then h*w uint8 states
                   (little-endian header; grid is row-major)

The frontend derives generation from the frame header and computes live/density
itself, so per-frame JSON is unnecessary — the hot path is pure binary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from . import lawsets
from .chunked import ChunkedLifeEngine
from .engine import make_engine

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Guardrails so a stray command can't ask for a 100k x 100k universe.
MIN_DIM, MAX_DIM = 16, 1024
MIN_FPS, MAX_FPS = 1, 120


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


class Session:
    """All mutable state for one connected viewer."""

    def __init__(self, ws: web.WebSocketResponse):
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue()
        self.closed = False

        self.rng = np.random.default_rng()
        self.w = 240
        self.h = 240
        self.fps = 20
        self.playing = True
        self.lawset_id = lawsets.DEFAULT_ID
        # A live, per-session copy of the law-set that `set_param` mutates.
        self.lawset = lawsets.get(self.lawset_id)

        # Boundary mode: fixed torus (default) or an unbounded "infinite" plane
        # (life family only). In infinite mode we stream a movable camera window.
        self.infinite = False
        self.cam_x = 0
        self.cam_y = 0
        self.zoom = 1

        # Color mode: "state" (each cell's value) or "age" (how many consecutive
        # generations it has been alive). Age is tracked here, not in the engine.
        self.color_mode = "state"
        self.age = np.zeros((self.h, self.w), dtype=np.uint16)

        self.engine = self._make_engine()

        self.frame_dirty = True   # a new frame needs sending this tick
        self.status_dirty = True  # structural state changed; resend status

    def _reset_age(self) -> None:
        self.age = np.zeros((self.h, self.w), dtype=np.uint16)

    def _age_tick(self) -> None:
        """After a step: increment age where alive, reset to 0 where dead."""
        if self.infinite:
            return
        g = self.engine.grid
        if self.age.shape != g.shape:
            self._reset_age()
            return
        alive = g != 0
        self.age = np.where(alive, np.minimum(self.age.astype(np.int32) + 1, 65535), 0).astype(np.uint16)

    def _step(self) -> None:
        """Advance one generation and update the age buffer (single choke point)."""
        self.engine.step()
        self._age_tick()

    def _make_engine(self):
        if self.infinite:
            return ChunkedLifeEngine(self.lawset, (self.h, self.w), self.rng)
        return make_engine(self.lawset, (self.h, self.w), self.rng)

    # -- reader ----------------------------------------------------------
    async def read_commands(self) -> None:
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        await self.queue.put(json.loads(msg.data))
                    except (ValueError, TypeError):
                        continue
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                    break
        finally:
            self.closed = True

    # -- command handling ------------------------------------------------
    def handle(self, cmd: dict) -> None:
        c = cmd.get("cmd")
        if c == "play":
            self.playing = True
            self.status_dirty = True
        elif c == "pause":
            self.playing = False
            self.status_dirty = True
        elif c == "step":
            if not self.playing:
                self._step()
                self.frame_dirty = True
        elif c == "reset":
            self.engine.seed()
            self._reset_age()
            self.frame_dirty = self.status_dirty = True
        elif c == "clear":
            self.engine.clear()
            self._reset_age()
            self.frame_dirty = self.status_dirty = True
        elif c == "set_lawset":
            self._set_lawset(cmd.get("id"))
        elif c == "set_fps":
            self.fps = _clamp(cmd.get("fps", 20), MIN_FPS, MAX_FPS)
            self.status_dirty = True
        elif c == "set_size":
            self.w = _clamp(cmd.get("w", self.w), MIN_DIM, MAX_DIM)
            self.h = _clamp(cmd.get("h", self.h), MIN_DIM, MAX_DIM)
            if self.infinite:
                # In infinite mode the "size" is the camera window, not the world.
                self.engine.view_w, self.engine.view_h = self.w, self.h
            else:
                self.engine.resize((self.h, self.w))
            self._reset_age()
            self.frame_dirty = self.status_dirty = True
        elif c == "paint":
            self._paint(cmd)
        elif c == "set_param":
            self._set_param(cmd.get("key"), cmd.get("value"))
        elif c == "set_boundary":
            self._set_boundary(cmd.get("mode"))
        elif c == "pan":
            if self.infinite:
                self.cam_x += int(cmd.get("dx", 0)) * self.zoom
                self.cam_y += int(cmd.get("dy", 0)) * self.zoom
                self.frame_dirty = True
        elif c == "zoom":
            if self.infinite:
                self.zoom = _clamp(cmd.get("zoom", 1), 1, 32)
                self.frame_dirty = self.status_dirty = True
        elif c == "recenter":
            if self.infinite:
                self.cam_x, self.cam_y = self.engine.center_of_mass()
                self.frame_dirty = True
        elif c == "random":
            self._random_universe()
        elif c == "set_color_mode":
            self.color_mode = "age" if cmd.get("mode") == "age" else "state"
            self.frame_dirty = self.status_dirty = True

    def _paint(self, cmd: dict) -> None:
        try:
            r, col = int(cmd["r"]), int(cmd["c"])
            value = int(cmd.get("value", 1))
            radius = int(cmd.get("radius", 1))
        except (KeyError, ValueError, TypeError):
            return
        if self.infinite:
            # Map a viewport pixel to a world cell (same origin viewport() uses).
            wx0 = self.cam_x - (self.w * self.zoom) // 2
            wy0 = self.cam_y - (self.h * self.zoom) // 2
            self.engine.paint(wx0 + col * self.zoom, wy0 + r * self.zoom, value, radius * self.zoom)
        else:
            self.engine.paint(r, col, value, radius)
        self.frame_dirty = True

    def _set_boundary(self, mode) -> None:
        want = (mode == "infinite") and self.lawset.family == "life"
        self.infinite = want
        self.cam_x = self.cam_y = 0
        self.zoom = 1
        self.engine = self._make_engine()
        self._reset_age()
        self.frame_dirty = self.status_dirty = True

    def _random_universe(self) -> None:
        """Invent and load a brand-new random universe (a random law of physics)."""
        self.lawset = lawsets.random_lawset(self.rng)
        self.lawset_id = self.lawset.id
        self.infinite = False           # a random universe may be any family
        self.cam_x = self.cam_y = 0
        self.zoom = 1
        self.engine = self._make_engine()
        self._reset_age()
        self.frame_dirty = self.status_dirty = True

    def _set_lawset(self, lid) -> None:
        if lid not in lawsets.LIBRARY:
            return
        self.lawset_id = lid
        # New physics, same canvas: reset to the library default and rebuild.
        self.lawset = lawsets.get(lid)
        if self.infinite and self.lawset.family != "life":
            # Infinite mode is life-only; fall back to the torus for other families.
            self.infinite = False
            self.cam_x = self.cam_y = 0
            self.zoom = 1
        self.engine = self._make_engine()
        self._reset_age()
        self.frame_dirty = self.status_dirty = True

    def _set_param(self, key, value) -> None:
        """Tune one knob of the *current* universe live, keeping the grid. Mutates
        the session's LawSet (via dataclasses.replace) and reconfigures the engine."""
        ls = self.lawset
        try:
            if key in ("birth", "survival"):
                vals = sorted({int(x) for x in value if 0 <= int(x) <= 8})
                self.lawset = replace(ls, params={**ls.params, key: vals})
            elif key == "threshold":
                self.lawset = replace(ls, params={**ls.params, "threshold": _clamp(value, 1, 8)})
            elif key == "states":
                n = _clamp(value, 2, 64)
                self.lawset = replace(ls, states=n, palette=lawsets.excitable_palette(n))
            elif key in ("p", "f", "mu", "sigma"):
                v = max(0.0, min(1.0, float(value)))
                self.lawset = replace(ls, params={**ls.params, key: v})
            elif key == "grow":                       # level-set: can be negative (erode)
                self.lawset = replace(ls, params={**ls.params, "grow": max(-1.0, min(1.0, float(value)))})
            elif key == "tension":
                self.lawset = replace(ls, params={**ls.params, "tension": max(0.0, min(2.0, float(value)))})
            elif key == "density":
                d = max(0.0, min(1.0, float(value)))
                self.lawset = replace(ls, seed={**ls.seed, "density": d})
            else:
                return
        except (ValueError, TypeError):
            return
        # reconfigure keeps the grid; density only takes effect on the next reseed.
        self.engine.reconfigure(self.lawset)
        self.frame_dirty = self.status_dirty = True

    # -- outbound --------------------------------------------------------
    async def send_json(self, obj: dict) -> None:
        if self.ws.closed:
            self.closed = True
            return
        try:
            await self.ws.send_str(json.dumps(obj))
        except (ConnectionError, RuntimeError):
            self.closed = True

    async def send_frame(self) -> None:
        if self.ws.closed:
            self.closed = True
            return
        if self.infinite:
            # Stream the camera window over the unbounded plane; the frame is the
            # same w x h uint8 format, just a *view* rather than the whole world.
            arr, _, _ = self.engine.viewport(self.cam_x, self.cam_y, self.w, self.h, self.zoom)
            g = np.ascontiguousarray(arr, dtype=np.uint8)
        elif self.color_mode == "age":
            # Stream per-cell age (capped to 255); the client renders it with the
            # age-gradient palette we send in the status. Same binary format.
            g = np.ascontiguousarray(np.minimum(self.age, 255), dtype=np.uint8)
        else:
            g = np.ascontiguousarray(self.engine.grid, dtype=np.uint8)
        h, w = g.shape
        header = struct.pack("<III", w, h, self.engine.generation)
        try:
            await self.ws.send_bytes(header + g.tobytes())
        except (ConnectionError, RuntimeError):
            self.closed = True

    async def send_view(self) -> None:
        """Per-tick JSON for infinite mode: camera + world stats the fixed-size
        binary frame can't carry (total population, live tile count, zoom)."""
        st = self.engine.stats()
        await self.send_json({
            "type": "view",
            "cx": self.cam_x, "cy": self.cam_y, "zoom": self.zoom,
            "generation": st.get("generation", 0),
            "population": st.get("population", 0),
            "tiles": st.get("tiles", 0),
        })

    async def send_status(self) -> None:
        ls = self.lawset
        params = dict(ls.params)
        params["density"] = ls.seed.get("density")
        # In age mode we swap in the age-gradient palette (256 entries); the client's
        # render path is unchanged — it just colors the age bytes with this palette.
        age_mode = self.color_mode == "age" and not self.infinite
        palette = lawsets.AGE_PALETTE if age_mode else ls.palette
        states = 256 if age_mode else ls.states
        await self.send_json({
            "type": "status",
            "lawset": self.lawset_id,
            "name": ls.name,             # so one-off random universes can show a label
            "description": ls.description,
            "family": ls.family,         # which engine/kind of automaton this is
            "color_mode": self.color_mode,
            "playing": self.playing,
            "fps": self.fps,
            "w": self.w,
            "h": self.h,
            "states": states,
            "palette": palette,          # live: regenerated when e.g. `states` changes
            "controls": ls.controls,     # the tunable-knob spec for this universe
            "params": params,            # current knob values (incl. seed density)
            "infinite": self.infinite,   # is the world an unbounded plane right now
            "can_infinite": ls.family == "life",  # is infinite even available here
            "zoom": self.zoom,
        })

    # -- main loop -------------------------------------------------------
    async def run(self) -> None:
        await self.send_json({
            "type": "catalog",
            "lawsets": lawsets.catalog(),
            "current": self.lawset_id,
        })
        await self.send_frame()
        await self.send_status()
        if self.infinite:
            await self.send_view()
        # The initial sends above already flushed current state; don't let the
        # first loop tick re-send a duplicate frame/status.
        self.frame_dirty = self.status_dirty = False

        while not self.closed and not self.ws.closed:
            # Apply everything queued since last tick.
            while not self.queue.empty():
                self.handle(self.queue.get_nowait())

            if self.playing:
                self._step()
                self.frame_dirty = True

            if self.frame_dirty:
                await self.send_frame()
                # Infinite mode: the camera/population change every tick, so the
                # view message rides along with each frame.
                if self.infinite:
                    await self.send_view()
                self.frame_dirty = False
            if self.status_dirty:
                await self.send_status()
                self.status_dirty = False

            await asyncio.sleep(1.0 / max(self.fps, 1))


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    session = Session(ws)
    reader = asyncio.create_task(session.read_commands())
    try:
        await session.run()
    finally:
        reader.cancel()
        if not ws.closed:
            await ws.close()
    return ws


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(FRONTEND_DIR / "index.html")


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static/", FRONTEND_DIR)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="reality-sim live universe viewer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()

    app = build_app()
    url = f"http://{args.host}:{args.port}"
    print(f"reality-sim  ·  open {url}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
