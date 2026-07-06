# Wire Protocol

One websocket connection (`/ws`, handled by `ws_handler` in
`reality_sim/server.py`) carries everything for one viewer: JSON control
messages in both directions, plus one binary grid frame per tick from server
to client. There is no REST API and no per-frame JSON — the design is that the
hot path (a new generation, ~20–120 times a second) is pure binary, and JSON
is reserved for state that changes rarely.

## Client → server: commands

Every command is a JSON text frame shaped `{"cmd": "...", ...}`, read by
`Session.read_commands()` and queued for the sim loop to apply in
`Session.handle()`. Unknown or malformed JSON is silently dropped (the reader
catches `ValueError`/`TypeError` on parse).

| `cmd` | fields | effect | guardrails |
|---|---|---|---|
| `play` | — | `playing = True` | |
| `pause` | — | `playing = False` | |
| `step` | — | `engine.step()` once — **only if not already playing** | no-op while `playing` is `True` |
| `reset` | — | `engine.seed()` — reseeds from the LawSet's `seed` recipe, generation → 0 | |
| `clear` | — | `grid[:] = 0`, `generation = 0` | |
| `set_lawset` | `id` | rebuilds the engine for the new LawSet at the *current* `(h, w)` via `make_engine()` | `id` must exist in `lawsets.LIBRARY`; unknown ids are silently ignored |
| `set_fps` | `fps` | sets the sim loop's tick rate | clamped to `MIN_FPS=1 .. MAX_FPS=120` |
| `set_size` | `w`, `h` | `engine.resize((h, w))` — new blank grid, reseeded | each dimension clamped to `MIN_DIM=16 .. MAX_DIM=1024` |
| `paint` | `r`, `c`, `value` (default `1`), `radius` (default `1`) | `engine.paint(r, c, value, radius)` — sets a cell, or a filled disk of `radius` around it, to `value` | `value` is taken mod `lawset.states`; missing/bad fields (`KeyError`/`ValueError`/`TypeError`) are caught and the command is dropped |

Every command sets `status_dirty` (so the next tick re-sends `status`)
**except** `step` and `paint` — those two only touch the grid, so they set
just `frame_dirty` (so the next tick re-sends a binary frame) and leave the
cheap JSON status message alone.

## Server → client: messages

### `catalog` (JSON, sent once per connection)

Sent immediately on connect, before the first frame:

```json
{"type": "catalog", "lawsets": [ /* LawSet.to_public() for every registered LawSet, in registration order */ ], "current": "conway"}
```

Each entry in `lawsets` is `asdict()` of a `LawSet` — `id`, `name`,
`description`, `family`, `states`, `params`, `palette`, `seed` — exactly the
same object used to build the engine server-side. `current` is the
session's active `lawset_id` at connect time (`lawsets.DEFAULT_ID`, i.e.
`"conway"`, unless already changed).

### `status` (JSON, sent on structural change)

Sent once right after `catalog`/the first frame, and again any tick where
`status_dirty` is set (play/pause, fps change, resize, lawset swap):

```json
{"type": "status", "lawset": "conway", "playing": true, "fps": 20, "w": 240, "h": 240, "states": 2}
```

| field | meaning |
|---|---|
| `lawset` | current `lawset_id` |
| `playing` | whether the sim loop is auto-stepping |
| `fps` | current tick rate |
| `w`, `h` | current grid dimensions |
| `states` | `engine.lawset.states` — how many distinct cell values this universe uses |

### Binary frame (every tick)

One binary websocket frame per tick, built in `Session.send_frame()`:

```python
header = struct.pack("<III", w, h, self.engine.generation)
await self.ws.send_bytes(header + g.tobytes())
```

| bytes | type | field |
|---|---|---|
| 0–3 | `uint32`, little-endian | `w` — grid width |
| 4–7 | `uint32`, little-endian | `h` — grid height |
| 8–11 | `uint32`, little-endian | `generation` — the engine's generation counter |
| 12 … `12 + w*h - 1` | `uint8` × `w*h` | the grid, row-major, one byte per cell, value in `0..states-1` |

`frontend/app.js`'s `onBinary()` reads this back with a `DataView` for the
header and a zero-copy `Uint8Array` view for the body:

```js
const dv = new DataView(buf);
gridW = dv.getUint32(0, true);
gridH = dv.getUint32(4, true);
generation = dv.getUint32(8, true);
latestGrid = new Uint8Array(buf, 12, gridW * gridH);
```

### Why there's no per-frame JSON

The frontend never receives generation, live-cell count, or density as
fields — it derives all of them itself from the binary frame: `generation`
comes straight out of the header, and `updateStats()` computes `live` and
`excited` counts (and the density percentage) by scanning `latestGrid` client
-side. This is a deliberate split: the only thing that has to travel every
tick is the raw grid, so that channel stays pure binary with a fixed 12-byte
header, and every JSON message is reserved for state that changes rarely
enough to afford to serialize.
