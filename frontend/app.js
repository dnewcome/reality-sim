"use strict";

// ---- connection ------------------------------------------------------------
const proto = location.protocol === "https:" ? "wss" : "ws";
let ws;
let reconnectTimer = null;

// ---- state -----------------------------------------------------------------
const catalog = {};           // id -> lawset meta (name/description for the picker)
let currentId = null;
let builtForId = null;        // which universe the controls panel was built for
let palette32 = new Uint32Array([0, 0xffffffff]); // state -> packed RGBA
let statesCount = 2;
let gridW = 0, gridH = 0, generation = 0;
let latestGrid = null;        // Uint8Array of last frame
let playing = true;

let brushRadius = 2;
let tool = "paint";           // "paint" | "erase" | "pan"

// infinite-grid mode
let infinite = false;
let canInfinite = false;
let zoom = 1;
let camX = 0, camY = 0;
const PAN_KEY_STEP = 24;      // pixels moved per arrow-key press

// ---- canvas ----------------------------------------------------------------
const view = document.getElementById("view");
const ctx = view.getContext("2d", { alpha: false });
ctx.imageSmoothingEnabled = false;
let off = document.createElement("canvas");   // offscreen at native grid size
let offCtx = off.getContext("2d");
let imgData = null;
let pix32 = null;             // Uint32 view over imgData

// ---- dom -------------------------------------------------------------------
const el = (id) => document.getElementById(id);
const conn = el("conn");

function packHex(hex) {
  // "#rrggbb" -> little-endian RGBA uint32 (a=255)
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
}

function setPalette(hexList) {
  palette32 = new Uint32Array(hexList.length);
  for (let i = 0; i < hexList.length; i++) palette32[i] = packHex(hexList[i]);
  statesCount = hexList.length;
}

function ensureBuffers() {
  if (off.width !== gridW || off.height !== gridH || !imgData) {
    off.width = gridW;
    off.height = gridH;
    imgData = offCtx.createImageData(gridW, gridH);
    pix32 = new Uint32Array(imgData.data.buffer);
    // Keep the visible canvas backing store an integer multiple of the grid so
    // nearest-neighbor upscaling stays crisp.
    const scale = Math.max(1, Math.floor(700 / Math.max(gridW, gridH)));
    view.width = gridW * scale;
    view.height = gridH * scale;
    ctx.imageSmoothingEnabled = false;
  }
}

function render() {
  if (!latestGrid) return;
  ensureBuffers();
  const n = gridW * gridH;
  const g = latestGrid;
  const pal = palette32;
  const pmax = pal.length - 1;
  for (let i = 0; i < n; i++) {
    let s = g[i];
    if (s > pmax) s = pmax;
    pix32[i] = pal[s];
  }
  offCtx.putImageData(imgData, 0, 0);
  ctx.drawImage(off, 0, 0, view.width, view.height);
  updateStats();
}

function updateStats() {
  if (!latestGrid) return;
  const n = gridW * gridH;
  let live = 0, excited = 0;
  const g = latestGrid;
  for (let i = 0; i < n; i++) {
    const s = g[i];
    if (s !== 0) live++;
    if (s === 1) excited++;
  }
  el("stat-gen").textContent = generation.toLocaleString();
  el("stat-live").textContent = live.toLocaleString();
  el("stat-density").textContent = (100 * live / n).toFixed(1) + "%";
  const excitedWrap = el("stat-excited-wrap");
  if (statesCount > 2) {
    excitedWrap.classList.add("show");
    el("stat-excited").textContent = excited.toLocaleString();
  } else {
    excitedWrap.classList.remove("show");
  }
}

// ---- protocol --------------------------------------------------------------
function onBinary(buf) {
  const dv = new DataView(buf);
  gridW = dv.getUint32(0, true);
  gridH = dv.getUint32(4, true);
  generation = dv.getUint32(8, true);
  latestGrid = new Uint8Array(buf, 12, gridW * gridH);
  render();
}

function onJson(msg) {
  if (msg.type === "catalog") {
    for (const ls of msg.lawsets) catalog[ls.id] = ls;
    buildLawsetList(msg.lawsets);
    selectLawsetUI(msg.current);
  } else if (msg.type === "status") {
    currentId = msg.lawset;
    if (msg.palette) setPalette(msg.palette);   // live palette (may change with `states`)
    selectLawsetUI(currentId);
    playing = msg.playing;
    el("btn-play").innerHTML = playing ? "&#10073;&#10073; Pause" : "&#9654; Play";
    el("fps").value = msg.fps;
    el("fps-val").textContent = msg.fps;
    el("size").value = msg.w;
    el("size-val").textContent = msg.w;
    infinite = !!msg.infinite;
    canInfinite = !!msg.can_infinite;
    if (msg.zoom) zoom = msg.zoom;
    // (Re)build the parameter panel only when the universe changes, so tuning a
    // knob (which echoes a status) never yanks a slider out from under the mouse.
    if (msg.controls && msg.lawset !== builtForId) {
      buildControls(msg.controls, msg.params || {});
      builtForId = msg.lawset;
    }
    updateBoundaryUI();
    render();
  } else if (msg.type === "view") {
    // per-tick infinite-mode readout (camera + world stats)
    camX = msg.cx; camY = msg.cy; zoom = msg.zoom;
    generation = msg.generation;
    el("stat-gen").textContent = generation.toLocaleString();
    el("stat-pop").textContent = msg.population.toLocaleString();
    el("stat-tiles").textContent = msg.tiles.toLocaleString();
    el("stat-cam").textContent = `${msg.cx},${msg.cy}`;
    el("stat-zoom").textContent = msg.zoom;
  }
}

function updateBoundaryUI() {
  el("btn-infinite").classList.toggle("disabled", !canInfinite);
  el("btn-torus").classList.toggle("on", !infinite);
  el("btn-infinite").classList.toggle("on", infinite);
  el("infinite-controls").style.display = infinite ? "block" : "none";
  el("size-hint").textContent = infinite
    ? "the window onto an endless plane"
    : "smaller = faster + chunkier cells";
  // swap which stats are visible: torus shows live/density, infinite shows pop/tiles/cam
  for (const id of ["stat-live-wrap", "stat-density-wrap"])
    el(id).style.display = infinite ? "none" : "inline";
  for (const id of ["stat-pop-wrap", "stat-tiles-wrap", "stat-cam-wrap"])
    el(id).classList.toggle("show", infinite);
  el("btn-pan").style.display = infinite ? "" : "none";
  if (!infinite && tool === "pan") setTool("paint");
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

// ---- lawset picker ---------------------------------------------------------
function buildLawsetList(list) {
  const box = el("lawset-list");
  box.innerHTML = "";
  for (const ls of list) {
    const b = document.createElement("button");
    b.className = "lawset-btn";
    b.dataset.id = ls.id;
    b.textContent = ls.name;
    b.onclick = () => send({ cmd: "set_lawset", id: ls.id });
    box.appendChild(b);
  }
}

function selectLawsetUI(id) {
  for (const b of document.querySelectorAll(".lawset-btn")) {
    b.classList.toggle("active", b.dataset.id === id);
  }
  if (catalog[id]) el("lawset-desc").textContent = catalog[id].description;
}

// ---- live parameter controls ----------------------------------------------
function fmtNum(v, step) {
  if (Number.isInteger(step) && Number.isInteger(v)) return String(v);
  const s = String(step), dot = s.indexOf(".");
  const dec = dot >= 0 ? s.length - dot - 1 : 2;
  return Number(v).toFixed(Math.min(dec, 5));
}

function buildControls(controls, params) {
  const box = el("controls");
  box.innerHTML = "";
  for (const c of controls) {
    box.appendChild(c.type === "set9"
      ? buildSet9(c, params[c.key] || [])
      : buildSlider(c, params[c.key]));
  }
}

// A subset of {0..8} as nine toggle chips — e.g. life's birth/survival sets.
function buildSet9(c, current) {
  const wrap = document.createElement("div");
  wrap.className = "control";
  const label = document.createElement("div");
  label.className = "control-label";
  label.innerHTML = `<span>${c.label}</span>`;
  wrap.appendChild(label);

  const chips = document.createElement("div");
  chips.className = "chips";
  const set = new Set((current || []).map(Number));
  for (let i = 0; i <= 8; i++) {
    const chip = document.createElement("button");
    chip.className = "chip" + (set.has(i) ? " on" : "");
    chip.textContent = i;
    chip.onclick = () => {
      if (set.has(i)) set.delete(i); else set.add(i);
      chip.classList.toggle("on");
      send({ cmd: "set_param", key: c.key, value: [...set].sort((a, b) => a - b) });
    };
    chips.appendChild(chip);
  }
  wrap.appendChild(chips);
  return wrap;
}

function buildSlider(c, current) {
  const wrap = document.createElement("div");
  wrap.className = "control";
  const val = current != null ? current : (c.min != null ? c.min : 0);

  const label = document.createElement("div");
  label.className = "control-label";
  const span = document.createElement("span"); span.textContent = c.label;
  const valEl = document.createElement("b"); valEl.textContent = fmtNum(val, c.step);
  label.appendChild(span); label.appendChild(valEl);
  wrap.appendChild(label);

  const input = document.createElement("input");
  input.type = "range";
  input.min = c.min; input.max = c.max; input.step = c.step; input.value = val;
  input.oninput = () => {
    const v = c.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
    valEl.textContent = fmtNum(v, c.step);
    send({ cmd: "set_param", key: c.key, value: v });
  };
  wrap.appendChild(input);
  return wrap;
}

// ---- pointer: paint / erase / pan -----------------------------------------
let dragging = false;
let lastPx = null;

function pointerPx(e) {
  const rect = view.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) / rect.width * gridW,
    y: (e.clientY - rect.top) / rect.height * gridH,
  };
}

function paintAt(e) {
  if (!gridW) return;
  const p = pointerPx(e);
  const r = Math.floor(p.y), c = Math.floor(p.x);
  if (r < 0 || c < 0 || r >= gridH || c >= gridW) return;
  send({ cmd: "paint", r, c, value: tool === "erase" ? 0 : 1, radius: brushRadius });
}

function panDrag(e) {
  const p = pointerPx(e);
  if (lastPx) {
    const dx = Math.round(p.x - lastPx.x), dy = Math.round(p.y - lastPx.y);
    if (dx || dy) send({ cmd: "pan", dx: -dx, dy: -dy });   // grab-and-pull-the-world feel
  }
  lastPx = p;
}

view.addEventListener("pointerdown", (e) => {
  dragging = true;
  view.setPointerCapture(e.pointerId);
  lastPx = pointerPx(e);
  if (tool !== "pan") paintAt(e);
});
view.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  if (tool === "pan") panDrag(e); else paintAt(e);
});
view.addEventListener("pointerup", () => { dragging = false; lastPx = null; });
view.addEventListener("pointercancel", () => { dragging = false; lastPx = null; });

view.addEventListener("wheel", (e) => {          // scroll to zoom (infinite only)
  if (!infinite) return;
  e.preventDefault();
  setZoom(e.deltaY < 0 ? zoom / 2 : zoom * 2);
}, { passive: false });

function setZoom(z) {
  send({ cmd: "zoom", zoom: Math.max(1, Math.min(32, Math.round(z))) });
}

function setTool(t) {
  tool = t;
  for (const [id, name] of [["btn-paint", "paint"], ["btn-erase", "erase"], ["btn-pan", "pan"]])
    el(id).classList.toggle("on", name === t);
  view.style.cursor = t === "pan" ? "grab" : "crosshair";
}

// ---- controls --------------------------------------------------------------
el("btn-play").onclick = () => send({ cmd: playing ? "pause" : "play" });
el("btn-step").onclick = () => send({ cmd: "step" });
el("btn-reset").onclick = () => send({ cmd: "reset" });
el("btn-clear").onclick = () => send({ cmd: "clear" });

el("fps").oninput = (e) => {
  el("fps-val").textContent = e.target.value;
  send({ cmd: "set_fps", fps: +e.target.value });
};
el("size").oninput = (e) => { el("size-val").textContent = e.target.value; };
el("size").onchange = (e) => {
  const d = +e.target.value;
  send({ cmd: "set_size", w: d, h: d });
};

el("brush").oninput = (e) => {
  brushRadius = +e.target.value;
  el("brush-val").textContent = e.target.value;
};
el("btn-paint").onclick = () => setTool("paint");
el("btn-erase").onclick = () => setTool("erase");
el("btn-pan").onclick = () => setTool("pan");

// boundary + infinite-grid controls
el("btn-torus").onclick = () => send({ cmd: "set_boundary", mode: "torus" });
el("btn-infinite").onclick = () => { if (canInfinite) send({ cmd: "set_boundary", mode: "infinite" }); };
el("btn-zoomin").onclick = () => setZoom(zoom / 2);
el("btn-zoomout").onclick = () => setZoom(zoom * 2);
el("btn-recenter").onclick = () => send({ cmd: "recenter" });

// keyboard: space = play/pause, s = step
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); send({ cmd: playing ? "pause" : "play" }); }
  else if (e.key === "s") send({ cmd: "step" });
  else if (e.key === "r") send({ cmd: "reset" });
  else if (infinite && e.key === "ArrowLeft") { e.preventDefault(); send({ cmd: "pan", dx: -PAN_KEY_STEP, dy: 0 }); }
  else if (infinite && e.key === "ArrowRight") { e.preventDefault(); send({ cmd: "pan", dx: PAN_KEY_STEP, dy: 0 }); }
  else if (infinite && e.key === "ArrowUp") { e.preventDefault(); send({ cmd: "pan", dx: 0, dy: -PAN_KEY_STEP }); }
  else if (infinite && e.key === "ArrowDown") { e.preventDefault(); send({ cmd: "pan", dx: 0, dy: PAN_KEY_STEP }); }
});

// ---- socket lifecycle ------------------------------------------------------
function connect() {
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    conn.textContent = "live";
    conn.className = "conn live";
  };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") onJson(JSON.parse(e.data));
    else onBinary(e.data);
  };
  ws.onclose = () => {
    conn.textContent = "disconnected";
    conn.className = "conn dead";
    if (!reconnectTimer) reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, 1200);
  };
  ws.onerror = () => ws.close();
}

connect();
