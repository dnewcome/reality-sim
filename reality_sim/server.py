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
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from . import lawsets
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
        self.engine = make_engine(lawsets.get(self.lawset_id), (self.h, self.w), self.rng)

        self.frame_dirty = True   # a new frame needs sending this tick
        self.status_dirty = True  # structural state changed; resend status

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
                self.engine.step()
                self.frame_dirty = True
        elif c == "reset":
            self.engine.seed()
            self.frame_dirty = self.status_dirty = True
        elif c == "clear":
            self.engine.grid[:] = 0
            self.engine.generation = 0
            self.frame_dirty = self.status_dirty = True
        elif c == "set_lawset":
            self._set_lawset(cmd.get("id"))
        elif c == "set_fps":
            self.fps = _clamp(cmd.get("fps", 20), MIN_FPS, MAX_FPS)
            self.status_dirty = True
        elif c == "set_size":
            self.w = _clamp(cmd.get("w", self.w), MIN_DIM, MAX_DIM)
            self.h = _clamp(cmd.get("h", self.h), MIN_DIM, MAX_DIM)
            self.engine.resize((self.h, self.w))
            self.frame_dirty = self.status_dirty = True
        elif c == "paint":
            try:
                self.engine.paint(
                    int(cmd["r"]), int(cmd["c"]),
                    int(cmd.get("value", 1)), int(cmd.get("radius", 1)),
                )
                self.frame_dirty = True
            except (KeyError, ValueError, TypeError):
                pass

    def _set_lawset(self, lid) -> None:
        if lid not in lawsets.LIBRARY:
            return
        self.lawset_id = lid
        # New physics, same canvas: rebuild the engine at the current size.
        self.engine = make_engine(lawsets.get(lid), (self.h, self.w), self.rng)
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
        g = np.ascontiguousarray(self.engine.grid, dtype=np.uint8)
        h, w = g.shape
        header = struct.pack("<III", w, h, self.engine.generation)
        try:
            await self.ws.send_bytes(header + g.tobytes())
        except (ConnectionError, RuntimeError):
            self.closed = True

    async def send_status(self) -> None:
        await self.send_json({
            "type": "status",
            "lawset": self.lawset_id,
            "playing": self.playing,
            "fps": self.fps,
            "w": self.w,
            "h": self.h,
            "states": self.engine.lawset.states,
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

        while not self.closed and not self.ws.closed:
            # Apply everything queued since last tick.
            while not self.queue.empty():
                self.handle(self.queue.get_nowait())

            if self.playing:
                self.engine.step()
                self.frame_dirty = True

            if self.frame_dirty:
                await self.send_frame()
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
