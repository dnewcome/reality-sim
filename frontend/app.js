"use strict";

// ---- connection ------------------------------------------------------------
const proto = location.protocol === "https:" ? "wss" : "ws";
let ws;
let reconnectTimer = null;

// ---- state -----------------------------------------------------------------
const catalog = {};           // id -> lawset meta (incl. palette)
let currentId = null;
let palette32 = new Uint32Array([0, 0xffffffff]); // state -> packed RGBA
let statesCount = 2;
let gridW = 0, gridH = 0, generation = 0;
let latestGrid = null;        // Uint8Array of last frame
let playing = true;

let brushValue = 1;           // paint vs erase
let brushRadius = 2;

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
    if (catalog[currentId]) setPalette(catalog[currentId].palette);
    selectLawsetUI(currentId);
    playing = msg.playing;
    el("btn-play").innerHTML = playing ? "&#10073;&#10073; Pause" : "&#9654; Play";
    el("fps").value = msg.fps;
    el("fps-val").textContent = msg.fps;
    el("size").value = msg.w;
    el("size-val").textContent = msg.w;
    // repaint immediately with the (possibly new) palette
    render();
  }
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

// ---- painting --------------------------------------------------------------
let painting = false;

function cellFromEvent(e) {
  const rect = view.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  const c = Math.floor(x * gridW);
  const r = Math.floor(y * gridH);
  return { r, c };
}

function paintAt(e) {
  if (!gridW) return;
  const { r, c } = cellFromEvent(e);
  if (r < 0 || c < 0 || r >= gridH || c >= gridW) return;
  send({ cmd: "paint", r, c, value: brushValue, radius: brushRadius });
}

view.addEventListener("pointerdown", (e) => {
  painting = true;
  view.setPointerCapture(e.pointerId);
  paintAt(e);
});
view.addEventListener("pointermove", (e) => { if (painting) paintAt(e); });
view.addEventListener("pointerup", () => { painting = false; });
view.addEventListener("pointercancel", () => { painting = false; });

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
el("btn-paint").onclick = () => {
  brushValue = 1;
  el("btn-paint").classList.add("on");
  el("btn-erase").classList.remove("on");
};
el("btn-erase").onclick = () => {
  brushValue = 0;
  el("btn-erase").classList.add("on");
  el("btn-paint").classList.remove("on");
};

// keyboard: space = play/pause, s = step
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); send({ cmd: playing ? "pause" : "play" }); }
  else if (e.key === "s") send({ cmd: "step" });
  else if (e.key === "r") send({ cmd: "reset" });
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
