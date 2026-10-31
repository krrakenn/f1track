// Optimal Lap – static web app. No framework: one index fetch, race data on demand,
// static layers cached on offscreen canvases so each frame only draws what moves.
"use strict";

const $ = (s) => document.querySelector(s);
const COLORS = { purple: "#b14eff", amber: "#f5b83d", tarmac: "#8e8e97", ink: "#ececee",
  rule: "#2a2a2f", asphalt: "#2a2a2e", brake: "#ff4d6a", accel: "#3ddc97" };
const RAMP = [[0, [47, 91, 211]], [0.3, [47, 183, 208]], [0.55, [233, 228, 162]],
  [0.8, [255, 139, 61]], [1, [255, 61, 110]]];
const BINS = 40;
const PALETTE = Array.from({ length: BINS }, (_, i) => ramp(i / (BINS - 1)));
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  index: null, cache: new Map(), race: null, mode: "race", playing: !reduced, rate: 1,
  tau: 0, framePos: 0, progress: 0, intro: 1, last: null, dirty: true, chartsDirty: true,
};

function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1], [t1, c1] = RAMP[i], f = (t - t0) / (t1 - t0);
      return `rgb(${c0.map((v, k) => Math.round(v + f * (c1[k] - v))).join(",")})`;
    }
  }
  return "rgb(255,61,110)";
}
const fmt = (s) => {
  if (!isFinite(s)) return "–";
  const m = Math.floor(s / 60), r = s - 60 * m;
  return `${m}:${r.toFixed(3).padStart(6, "0")}`;
};
const signed = (x, d = 3) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;

function interp(xs, ys, x) {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  return ys[lo] + (x - xs[lo]) / (xs[hi] - xs[lo] || 1) * (ys[hi] - ys[lo]);
}

// ── Data ──────────────────────────────────────────────────────────────────
async function loadRace(slug) {
  if (state.cache.has(slug)) return state.cache.get(slug);
  const p = fetch(`data/${slug}.json`, { cache: "no-cache" }).then((r) => r.json()).then(prep);
  state.cache.set(slug, p);
  return p;
}

function prep(raw) {
  const tr = raw.track, n = tr.x.length;
  raw.n = n;
  raw.E = new Float32Array(n).fill(1);
  raw.Eg = 1;
  // Lateral offsets are drawn scaled by E (track edges and every line alike), so a 15 m
  // track stays wider than the line at full-circuit zoom.
  raw.line = (a) => {
    const x = new Float32Array(n), y = new Float32Array(n), E = raw.E;
    for (let i = 0; i < n; i++) {
      const o = a ? a[i] * E[i] : 0;
      x[i] = tr.x[i] + o * tr.nx[i]; y[i] = tr.y[i] + o * tr.ny[i];
    }
    return { x, y };
  };
  raw.s = Float32Array.from({ length: n }, (_, i) => (i * raw.L) / n);
  raw.histLine = (k) => (raw.histXY[k] ||= raw.line(raw.hist.a[k]));
  setScale(raw, 1);
  const all = raw.opt.v.concat(raw.real.v);
  raw.vmin = Math.min(...all); raw.vmax = Math.max(...all);
  // gap at each distance: pole elapsed time minus model elapsed time (+ = model ahead)
  raw.gap = raw.real.t.map((t, i) => t - raw.opt.t[i]);
  return raw;
}

// Width exaggeration: a global factor Eg so the drawn track is ~TRACK_PX wide, reduced
// locally where the inner edge would fold over itself (offset must stay below the radius).
const TRACK_PX = 13, E_MAX = 5;
function nearSections(race) {
  // distance from each point to the nearest centre-line point >250 m away along the lap
  if (race._near) return race._near;
  const tr = race.track, n = race.n, ds = race.L / n, skip = Math.ceil(250 / ds), out = [];
  for (let i = 0; i < n; i += 2) {
    let best = Infinity;
    for (let j = 0; j < n; j += 2) {
      const dj = Math.abs(i - j), sep = Math.min(dj, n - dj);
      if (sep < skip) continue;
      const d = Math.hypot(tr.x[i] - tr.x[j], tr.y[i] - tr.y[j]);
      if (d < best) best = d;
    }
    if (best < 80) { out.push([i, best]); if (i + 1 < n) out.push([i + 1, best]); }
  }
  return (race._near = out);
}

function setScale(race, Eg) {
  if (race.Eg === Eg && race.opt.xy) return;
  const tr = race.track, n = race.n, E = race.E;
  const cap = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const w = Math.max(tr.wl[i], tr.wr[i], 1);
    cap[i] = Math.max(1, Math.min(Eg, 0.7 / (Math.abs(tr.k[i]) * w + 1e-9)));
  }
  // also where another part of the circuit runs close by (Suzuka's crossover, hairpins,
  // Baku's castle): widened edges of the two sections must not overlap
  for (const [i, gapM] of nearSections(race)) {
    const w = Math.max(tr.wl[i], tr.wr[i], 1);
    cap[i] = Math.max(1, Math.min(cap[i], (0.45 * gapM) / w));
  }
  const W = 12; // running minimum then moving average, so the scale changes gradually
  const mn = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = Infinity;
    for (let j = -W; j <= W; j++) m = Math.min(m, cap[(i + j + n) % n]);
    mn[i] = m;
  }
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = -W; j <= W; j++) sum += mn[(i + j + n) % n];
    E[i] = sum / (2 * W + 1);
  }
  race.Eg = Eg;
  race.opt.xy = race.line(race.opt.a);
  race.real.xy = race.line(race.real.a);
  race.mc.xy = race.line(race.mc.a);
  race.histXY = race.hist.a.map(() => null);
  race._ls = null;
}
function scaleFor(race, k) {
  const widths = Array.from(race.track.wl, (wl, i) => wl + race.track.wr[i]).sort((a, b) => a - b);
  const med = widths[widths.length >> 1];
  return Math.max(1, Math.min(E_MAX, Math.round((TRACK_PX / (k * med)) * 4) / 4));
}

// ── View transform ────────────────────────────────────────────────────────
function fitCanvas(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = cv.getBoundingClientRect();
  const w = Math.max(10, Math.round(r.width * dpr)), h = Math.max(10, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return { w: r.width, h: r.height, dpr };
}

function makeView(race, W, H) {
  const tr = race.track, rot = (race.rotation * Math.PI) / 180, c = Math.cos(rot), s = Math.sin(rot);
  // nothing is drawn over the map any more, so only room for the corner numbers
  const padT = 22, padB = 22, padX = 22;
  // Two passes: frame the true-width track, then re-frame with the widened edges so the
  // exaggerated track never clips at the canvas border.
  let x0, x1, y0, y1, k;
  for (const widen of [1, 0]) {
    const E = widen ? 1 : scaleFor(race, k);
    x0 = Infinity; x1 = -Infinity; y0 = Infinity; y1 = -Infinity;
    for (let i = 0; i < race.n; i++) {
      for (const a of [tr.wl[i] * E, -tr.wr[i] * E]) {
        const px = tr.x[i] + a * tr.nx[i], py = tr.y[i] + a * tr.ny[i];
        const rx = c * px - s * py, ry = s * px + c * py;
        x0 = Math.min(x0, rx); x1 = Math.max(x1, rx); y0 = Math.min(y0, ry); y1 = Math.max(y1, ry);
      }
    }
    k = Math.min((W - 2 * padX) / (x1 - x0), (H - padT - padB) / (y1 - y0));
  }
  const ox = (W - k * (x1 - x0)) / 2, oy = padT + (H - padT - padB - k * (y1 - y0)) / 2;
  return { k, P: (x, y) => [ox + (c * x - s * y - x0) * k, oy + (y1 - (s * x + c * y)) * k] };
}

// ── Track drawing ─────────────────────────────────────────────────────────
function pathOf(ctx, P, xs, ys, i0 = 0, i1 = xs.length, close = false) {
  ctx.beginPath();
  for (let i = i0; i < i1; i++) {
    const [px, py] = P(xs[i], ys[i]);
    i === i0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  if (close) ctx.closePath();
}

function drawBase(ctx, race, view) {
  const tr = race.track, n = race.n, P = view.P, E = race.E;
  const L = [], R = [];
  for (let i = 0; i < n; i++) {
    const l = tr.wl[i] * E[i], r = tr.wr[i] * E[i];
    L.push(P(tr.x[i] + l * tr.nx[i], tr.y[i] + l * tr.ny[i]));
    R.push(P(tr.x[i] - r * tr.nx[i], tr.y[i] - r * tr.ny[i]));
  }
  // one quad per segment, all with the same winding: overlapping parts of the circuit
  // (Suzuka's bridge) stay filled, unlike an even-odd polygon fill
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    ctx.moveTo(...L[i]); ctx.lineTo(...L[j]); ctx.lineTo(...R[j]); ctx.lineTo(...R[i]); ctx.closePath();
  }
  ctx.fillStyle = COLORS.asphalt; ctx.strokeStyle = COLORS.asphalt; ctx.lineWidth = 0.6;
  ctx.fill("nonzero"); ctx.stroke();
  ctx.strokeStyle = "rgba(150,150,158,0.6)";
  ctx.lineWidth = 1;
  for (const E of [L, R]) {
    ctx.beginPath();
    E.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
  }
  // start/finish line
  ctx.strokeStyle = COLORS.ink; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(...L[0]); ctx.lineTo(...R[0]); ctx.stroke();
  // corner numbers, outside the bend
  ctx.font = "600 12px 'Barlow Condensed', sans-serif";
  ctx.fillStyle = COLORS.tarmac; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  // corner numbers outside the bend; a number that would overlap one already placed is
  // tried on the other side of the track, then left out
  const placed = [];
  const free = (x, y, w, h) => placed.every((b) => Math.abs(b.x - x) > (b.w + w) / 2 + 2
    || Math.abs(b.y - y) > (b.h + h) / 2 + 1);
  for (const cn of race.corners) {
    const i = Math.round((cn.s / race.L) * n) % n, text = cn.label.replace(/^T/, "");
    const w = ctx.measureText(text).width, h = 12;
    const outside = tr.k[i] > 0 ? -1 : 1;
    for (const side of [outside, -outside]) {
      const off = (side > 0 ? tr.wl[i] : tr.wr[i]) * race.E[i] + 12 / view.k;
      const [px, py] = P(tr.x[i] + side * off * tr.nx[i], tr.y[i] + side * off * tr.ny[i]);
      if (free(px, py, w, h)) { ctx.fillText(text, px, py); placed.push({ x: px, y: py, w, h }); break; }
    }
  }
}

function speedLine(ctx, P, xy, v, race, width, upto = race.n) {
  // batch segments by colour bin: ~40 strokes instead of one per segment
  const bins = Array.from({ length: BINS }, () => []);
  const span = race.vmax - race.vmin || 1;
  for (let i = 0; i < upto; i++) {
    const j = (i + 1) % race.n;
    const b = Math.max(0, Math.min(BINS - 1, Math.round(((v[i] - race.vmin) / span) * (BINS - 1))));
    bins[b].push(P(xy.x[i], xy.y[i]), P(xy.x[j], xy.y[j]));
  }
  ctx.lineCap = "round"; ctx.lineWidth = width;
  bins.forEach((segs, b) => {
    if (!segs.length) return;
    ctx.beginPath();
    for (let k = 0; k < segs.length; k += 2) { ctx.moveTo(...segs[k]); ctx.lineTo(...segs[k + 1]); }
    ctx.strokeStyle = PALETTE[b];
    ctx.stroke();
  });
  ctx.lineCap = "butt";
}

function staticLayer(key, W, H, dpr, draw) {
  const L = state.layers || (state.layers = new Map());
  const id = `${key}|${W}|${H}`;
  if (!L.has(id)) {
    if (L.size > 24) L.clear();
    const cv = document.createElement("canvas");
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx);
    L.set(id, cv);
  }
  return L.get(id);
}

function posAt(race, xy, sPos) {
  const n = race.n, u = ((((sPos % race.L) + race.L) % race.L) / race.L) * n;
  const i = Math.floor(u) % n, j = (i + 1) % n, f = u - Math.floor(u);
  return [xy.x[i] + f * (xy.x[j] - xy.x[i]), xy.y[i] + f * (xy.y[j] - xy.y[i])];
}

function car(ctx, P, xy, race, sPos, fill, ring, r = 7) {
  const [px, py] = P(...posAt(race, xy, sPos));
  ctx.beginPath(); ctx.arc(px, py, r, 0, 2 * Math.PI);
  ctx.fillStyle = fill; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = ring; ctx.stroke();
}

function drawTrack() {
  const race = state.race; if (!race) return;
  const cv = $("#track"), { w: W, h: H, dpr } = fitCanvas(cv);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const view = makeView(race, W, H), P = view.P, team = race.meta.pole.colour;
  setScale(race, scaleFor(race, view.k));
  setText("#scale-note", race.Eg > 1.05 ? `track width drawn ×${race.Eg.toFixed(1)}` : "");
  const base = staticLayer(`${race.slug}|base`, W, H, dpr, (c) => drawBase(c, race, view));
  ctx.drawImage(base, 0, 0, W, H);

  if (state.mode === "race") {
    if (state.intro < 1) {
      speedLine(ctx, P, race.opt.xy, race.opt.v, race, 2.6, Math.floor(state.intro * race.n));
      return;
    }
    const lines = staticLayer(`${race.slug}|race`, W, H, dpr, (c) => {
      c.globalAlpha = 0.85; c.strokeStyle = team; c.lineWidth = 1.2;
      pathOf(c, P, race.real.xy.x, race.real.xy.y, 0, race.n, true); c.stroke();
      c.globalAlpha = 1;
      speedLine(c, P, race.opt.xy, race.opt.v, race, 2.6);
    });
    ctx.drawImage(lines, 0, 0, W, H);
    const sOpt = state.sOpt, sReal = state.sReal;
    // purple trail: the last ~1.2 s of the model car
    const trailS = interp(race.opt.t, race.s, Math.max(0, state.tauModel - 1.2));
    if (sOpt > trailS) {
      const i0 = Math.floor((trailS / race.L) * race.n), i1 = Math.ceil((sOpt / race.L) * race.n);
      ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 5; ctx.strokeStyle = "rgba(177,78,255,0.3)";
      pathOf(ctx, P, race.opt.xy.x, race.opt.xy.y, i0, Math.min(i1, race.n)); ctx.stroke(); ctx.restore();
    }
    car(ctx, P, race.real.xy, race, sReal, team, "#0b0b0c", 6);
    car(ctx, P, race.opt.xy, race, sOpt, COLORS.purple, "#ffffff", 6.5);
  } else if (state.mode === "line") {
    const k = Math.floor(state.framePos), ls = lineState(race);
    for (let j = Math.max(0, k - 5); j <= k; j++) {
      const xy = race.histLine(j);
      ctx.strokeStyle = `rgba(177,78,255,${0.06 + 0.04 * (j - k + 5)})`; ctx.lineWidth = 1.2;
      pathOf(ctx, P, xy.x, xy.y, 0, race.n, true); ctx.stroke();
    }
    ctx.globalAlpha = 0.5; ctx.strokeStyle = team; ctx.lineWidth = 1.2;
    pathOf(ctx, P, race.real.xy.x, race.real.xy.y, 0, race.n, true); ctx.stroke();
    ctx.globalAlpha = 1;
    speedLine(ctx, P, ls.xy, ls.v, race, 2.6);
  } else {
    const st = stageOf(state.progress);
    const layer = staticLayer(`${race.slug}|speed|${st.key}`, W, H, dpr, (c) =>
      speedLine(c, P, race.opt.xy, st.v(race), race, 2.6));
    ctx.drawImage(layer, 0, 0, W, H);
    car(ctx, P, race.opt.xy, race, state.progress * race.L, COLORS.purple, "#fff", 6);
  }
}

// Line-search replay: morph between consecutive "best so far" lines.
function lineState(race) {
  const m = race.hist.lap.length, k0 = Math.min(m - 1, Math.floor(state.framePos));
  const k1 = Math.min(m - 1, k0 + 1), f = state.framePos - k0;
  const key = `${k0}:${f.toFixed(3)}`;
  if (race._ls && race._ls.key === key) return race._ls;
  const A = race.histLine(k0), B = race.histLine(k1), va = race.hist.v[k0], vb = race.hist.v[k1];
  const x = new Float32Array(race.n), y = new Float32Array(race.n), v = new Array(race.n);
  for (let i = 0; i < race.n; i++) {
    x[i] = A.x[i] + f * (B.x[i] - A.x[i]); y[i] = A.y[i] + f * (B.y[i] - A.y[i]);
    v[i] = va[i] + f * (vb[i] - va[i]);
  }
  race._ls = { key, xy: { x, y }, v };
  return race._ls;
}

// ── Speed-solver stages ───────────────────────────────────────────────────
const STAGES = [
  { key: "corner", until: 0.22, v: (r) => r.stages.corner,
    text: "Step 1: each point gets the highest speed its curvature allows, using grip alone." },
  { key: "accel", until: 0.55, v: (r) => r.stages.accel,
    text: "Step 2: out of every corner, accelerate as hard as engine power and grip allow." },
  { key: "brake", until: 0.88, v: (r) => r.stages.brake,
    text: "Step 3: backwards from every corner, brake as late as the tyres allow." },
  { key: "final", until: 1.01, v: (r) => r.opt.v,
    text: "Step 4: the car's speed is the lowest of the three. That is the lap." },
];
const stageOf = (p) => STAGES.find((s) => p < s.until) || STAGES[3];

// ── Charts ────────────────────────────────────────────────────────────────
function chartPanels(H) {
  if (state.mode === "line") return [["Speed, km/h", 0.55, "speed"], ["Best lap time found so far, s", 0.45, "steps"]];
  if (state.mode === "speed") return [["Speed, km/h", 0.68, "stages"], ["Model braking and acceleration, g", 0.32, "long"]];
  // show as many panels as fit at a readable height, most important first
  const all = [
    ["Speed, km/h", 0.22, "speed", 0],
    ["Speed difference, model minus pole, km/h", 0.13, "dv", 2],
    ["Throttle and brake on the pole lap", 0.1, "pedals", 3],
    ["Cornering force, g", 0.11, "lat", 4],
    ["Grip used by the model car, %", 0.1, "grip", 5],
    ["Braking and acceleration, g", 0.11, "long", 6],
    ["Gap to pole lap, s", 0.14, "gap", 1],
    ["Elevation of the lap, m", 0.09, "elev", 7],
  ];
  const fit = Math.max(2, Math.min(all.length, Math.floor((H - 16) / 86)));
  const keep = all.filter((p) => p[3] < fit);
  const total = keep.reduce((a, p) => a + p[1], 0);
  return keep.map(([t, f, k]) => [t, f / total, k]);
}

function drawCharts() {
  const race = state.race; if (!race) return;
  const cv = $("#charts"), { w: W, h: H, dpr } = fitCanvas(cv);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pad = { l: 40, r: 12, t: 10 }, gap = 24;
  const panels = chartPanels(H), usable = H - pad.t - 16 - gap * panels.length;
  const X = (s) => pad.l + (s / race.L) * (W - pad.l - pad.r);
  const team = race.meta.pole.colour;
  const dynamic = state.mode !== "race";
  const key = `${race.slug}|charts|${state.mode}|${panels.length}|${dynamic ? Math.round(state.framePos * 24) + ":" + Math.floor(state.progress * 200) : ""}`;
  const layer = staticLayer(key, W, H, dpr, (c) => {
    let y = pad.t;
    for (const [title, frac, kind] of panels) {
      const h = usable * frac;
      chartPanel(c, race, kind, title, { x: pad.l, y: y + gap, w: W - pad.l - pad.r, h }, X, team);
      y += h + gap;
    }
  });
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(layer, 0, 0, W, H);
  // cursor
  const sCur = state.mode === "race" ? state.sOpt : state.mode === "speed" ? state.progress * race.L : null;
  if (sCur != null) {
    const px = X(sCur);
    ctx.strokeStyle = "rgba(233,238,246,0.55)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, pad.t + gap - 4); ctx.lineTo(px, H - 16); ctx.stroke();
  }
  state.chartX = X;
}

function axes(c, box, title, y0, y1) {
  const Y = (v) => box.y + box.h - ((v - y0) / (y1 - y0)) * box.h;
  c.font = "500 13px 'Barlow Condensed', sans-serif"; c.fillStyle = COLORS.tarmac;
  c.textAlign = "left"; c.textBaseline = "alphabetic"; c.fillText(title, box.x, box.y - 7);
  const raw = (y1 - y0) / 3, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  c.font = "500 11px 'Barlow Condensed', sans-serif"; c.textAlign = "right"; c.textBaseline = "middle";
  for (let v = Math.ceil(y0 / step) * step; v <= y1 + 1e-9; v += step) {
    c.fillStyle = COLORS.tarmac; c.fillText(step < 1 ? v.toFixed(1) : v.toFixed(0), box.x - 6, Y(v));
    c.strokeStyle = "rgba(48,48,54,0.9)"; c.beginPath(); c.moveTo(box.x, Y(v)); c.lineTo(box.x + box.w, Y(v)); c.stroke();
  }
  return Y;
}

function series(c, X, Y, s, v, color, width, from = 0, to = v.length, dash) {
  c.beginPath();
  for (let i = from; i < to; i++) { const px = X(s[i]), py = Y(v[i]); i === from ? c.moveTo(px, py) : c.lineTo(px, py); }
  c.setLineDash(dash || []); c.strokeStyle = color; c.lineWidth = width; c.stroke(); c.setLineDash([]);
}

function chartPanel(c, race, kind, title, box, X, team) {
  const kmh = (a) => a.map((v) => v * 3.6), s = race.s;
  if (kind === "speed") {
    const Y = axes(c, box, title, Math.floor(race.vmin * 3.6 / 50) * 50, Math.ceil(race.vmax * 3.6 / 50) * 50);
    series(c, X, Y, s, kmh(race.real.v), team, 1.4);
    series(c, X, Y, s, kmh(state.mode === "line" ? lineState(race).v : race.opt.v), COLORS.purple, 1.8);
  } else if (kind === "pedals") {
    const Y = axes(c, box, title, 0, 100);
    const bw = Math.max(1, X(s[1]) - X(s[0]) + 0.6);
    c.fillStyle = team; c.globalAlpha = 0.35;
    race.real.thr.forEach((v, i) => { if (v > 0) c.fillRect(X(s[i]), Y(v), bw, Y(0) - Y(v)); });
    c.globalAlpha = 0.9; c.fillStyle = COLORS.brake;
    race.real.brk.forEach((b, i) => { if (b > 0) c.fillRect(X(s[i]), box.y, bw, box.h * 0.22); });
    // stretches where the official car-data feed froze: no pedal data
    c.globalAlpha = 1; c.fillStyle = "rgba(142,142,151,0.18)";
    race.real.thr.forEach((v, i) => { if (v < 0) c.fillRect(X(s[i]), box.y, bw, box.h); });
    if (race.real.thr.some((v) => v < 0)) {
      c.fillStyle = COLORS.tarmac; c.font = "500 11px 'Barlow Condensed', sans-serif"; c.textAlign = "right";
      c.fillText("grey: feed froze, no pedal data", box.x + box.w - 4, box.y + box.h - 6);
    }
  } else if (kind === "lat") {
    // magnitude: left and right turns both count as cornering force
    const Y = axes(c, box, title, 0, 6);
    series(c, X, Y, s, race.opt.ay.map((a) => Math.abs(a) / 9.81), COLORS.purple, 1.3);
  } else if (kind === "dv") {
    const dv = race.opt.v.map((v, i) => (v - race.real.v[i]) * 3.6);
    const lim = Math.max(20, Math.ceil(Math.max(...dv.map(Math.abs)) / 20) * 20);
    const Y = axes(c, box, title, -lim, lim);
    for (const [sign, col] of [[1, "rgba(177,78,255,0.45)"], [-1, "rgba(245,184,61,0.45)"]]) {
      c.beginPath(); c.moveTo(X(0), Y(0));
      dv.forEach((d, i) => c.lineTo(X(s[i]), Y(sign * d > 0 ? d : 0)));
      c.lineTo(X(s[race.n - 1]), Y(0)); c.fillStyle = col; c.fill();
    }
  } else if (kind === "grip") {
    const Y = axes(c, box, title, 0, 100);
    c.beginPath(); c.moveTo(X(0), Y(0));
    gripUsed(race).forEach((g, i) => c.lineTo(X(s[i]), Y(g)));
    c.lineTo(X(s[race.n - 1]), Y(0)); c.closePath();
    c.fillStyle = "rgba(177,78,255,0.3)"; c.fill();
    series(c, X, Y, s, gripUsed(race), COLORS.purple, 1.1);
  } else if (kind === "elev") {
    const zmax = Math.max(...race.z, 1);
    const Y = axes(c, box, title, 0, Math.ceil(zmax / 5) * 5 || 5);
    c.beginPath(); c.moveTo(X(0), Y(0));
    race.z.forEach((z, i) => c.lineTo(X(s[i]), Y(z)));
    c.lineTo(X(s[race.n - 1]), Y(0)); c.closePath();
    c.fillStyle = "rgba(142,142,151,0.25)"; c.fill();
    series(c, X, Y, s, race.z, COLORS.tarmac, 1.2);
  } else if (kind === "long") {
    const Y = axes(c, box, title, -6, 3);
    series(c, X, Y, s, race.opt.ax.map((a) => a / 9.81), COLORS.purple, 1.3);
  } else if (kind === "gap") {
    const lo = Math.min(0, ...race.gap), hi = Math.max(0.2, ...race.gap);
    const Y = axes(c, box, title, Math.floor(lo * 2) / 2, Math.ceil(hi * 2) / 2);
    c.strokeStyle = COLORS.tarmac; c.beginPath(); c.moveTo(box.x, Y(0)); c.lineTo(box.x + box.w, Y(0)); c.stroke();
    // fill: purple where the model is ahead, amber where behind
    for (const [sign, col] of [[1, "rgba(177,78,255,0.35)"], [-1, "rgba(245,184,61,0.35)"]]) {
      c.beginPath(); c.moveTo(X(0), Y(0));
      race.gap.forEach((g, i) => c.lineTo(X(s[i]), Y(sign * g > 0 ? g : 0)));
      c.lineTo(X(s[race.n - 1]), Y(0)); c.fillStyle = col; c.fill();
    }
    series(c, X, Y, s, race.gap, COLORS.ink, 1.2);
  } else if (kind === "steps") {
    const H_ = race.hist, m = H_.lap.length;
    const lo = Math.min(...H_.lap), hi = Math.max(...H_.lap);
    const Y = axes(c, box, title, Math.floor(lo), Math.ceil(hi));
    const XS = (i) => box.x + (i / (m - 1)) * box.w;
    c.beginPath(); H_.lap.forEach((v, i) => (i ? c.lineTo(XS(i), Y(v)) : c.moveTo(XS(i), Y(v))));
    c.strokeStyle = COLORS.purple; c.lineWidth = 1.6; c.stroke();
    const k0 = Math.floor(state.framePos), k1 = Math.min(m - 1, k0 + 1), f = state.framePos - k0;
    c.fillStyle = COLORS.ink; c.beginPath();
    c.arc(XS(state.framePos), Y(H_.lap[k0] + f * (H_.lap[k1] - H_.lap[k0])), 4.5, 0, 7); c.fill();
    const pole = race.meta.pole.time;
    if (pole >= lo - 1 && pole <= hi + 1) {
      c.setLineDash([4, 4]); c.strokeStyle = team; c.beginPath(); c.moveTo(box.x, Y(pole)); c.lineTo(box.x + box.w, Y(pole)); c.stroke(); c.setLineDash([]);
    }
  } else if (kind === "stages") {
    // corner limits reach the 360 km/h speed cap on straights, so show the full range
    const Y = axes(c, box, title, Math.floor(race.vmin * 3.6 / 50) * 50, 360);
    const p = state.progress, n = race.n, S = race.stages;
    series(c, X, Y, s, kmh(S.corner), COLORS.amber, 1, 0, n, [3, 3]);
    if (p > 0.22) series(c, X, Y, s, kmh(S.accel), COLORS.accel, 1.4, 0, Math.max(2, Math.floor(Math.min(1, (p - 0.22) / 0.33) * n)));
    if (p > 0.55) series(c, X, Y, s, kmh(S.brake), COLORS.brake, 1.4, Math.floor((1 - Math.min(1, (p - 0.55) / 0.33)) * n), n);
    if (p >= 0.88) series(c, X, Y, s, kmh(race.opt.v), COLORS.purple, 2.4);
  }
}

// ── Season chart ──────────────────────────────────────────────────────────
function drawSeason() {
  const idx = state.index; if (!idx) return;
  const cv = $("#season-chart"), { w: W, h: H, dpr } = fitCanvas(cv);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const races = idx.races, n = races.length;
  const gaps = races.map((r) => r.model.time - r.pole.time);
  const lim = Math.ceil(Math.max(...gaps.map(Math.abs)) + 0.5);
  const left = 8, right = 8, top = 10, bottom = 36, bw = (W - left - right) / n;
  const Y = (g) => top + ((lim - g) / (2 * lim)) * (H - top - bottom);
  ctx.strokeStyle = COLORS.rule;
  for (let g = -lim; g <= lim; g += lim > 6 ? 2 : 1) {
    ctx.beginPath(); ctx.moveTo(left, Y(g)); ctx.lineTo(W - right, Y(g)); ctx.stroke();
  }
  ctx.strokeStyle = COLORS.tarmac; ctx.beginPath(); ctx.moveTo(left, Y(0)); ctx.lineTo(W - right, Y(0)); ctx.stroke();
  ctx.font = "600 13px 'Barlow Condensed', sans-serif"; ctx.textAlign = "center";
  races.forEach((r, i) => {
    const x = left + i * bw + bw * 0.18, w = bw * 0.64, g = gaps[i];
    ctx.fillStyle = g < 0 ? COLORS.purple : COLORS.amber;
    ctx.globalAlpha = state.race && state.race.slug === r.slug ? 1 : 0.7;
    ctx.fillRect(x, Math.min(Y(0), Y(g)), w, Math.abs(Y(g) - Y(0)));
    ctx.globalAlpha = 1;
    ctx.fillStyle = COLORS.ink; ctx.textBaseline = g < 0 ? "top" : "bottom";
    ctx.fillText(signed(g, 1), x + w / 2, g < 0 ? Y(g) + 3 : Y(g) - 3);
    ctx.fillStyle = COLORS.tarmac; ctx.textBaseline = "top"; ctx.fillText(r.code, x + w / 2, H - bottom + 10);
  });
  ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = COLORS.tarmac;
  ctx.fillText(`+${lim} s`, left + 2, Y(lim) + 8); ctx.fillText(`−${lim} s`, left + 2, Y(-lim) - 8);
  cv.onclick = (e) => {
    const r = cv.getBoundingClientRect(), i = Math.floor((e.clientX - r.left - left) / bw);
    if (races[i]) { selectRace(races[i].slug); $(".pitwall").scrollIntoView({ behavior: reduced ? "auto" : "smooth" }); }
  };
}

// ── UI ────────────────────────────────────────────────────────────────────
function renderTower() {
  const ol = $("#tower");
  ol.innerHTML = "";
  for (const r of state.index.races) {
    const diff = r.model.time - r.pole.time;
    const pts = r.outline.map(([x, y]) => `${(x * 100).toFixed(1)},${(y * 66).toFixed(1)}`).join(" ");
    const li = document.createElement("li");
    li.innerHTML = `<button data-slug="${r.slug}" style="--row-team:${r.pole.colour}" aria-label="Round ${r.round}, ${r.event}">
      <span class="rnd">${r.round}</span><span class="code">${r.code}</span>
      <svg viewBox="-4 -4 108 74" aria-hidden="true"><polyline points="${pts}"/></svg>
      <span class="times"><span class="pt">${fmt(r.pole.time)}</span>
      <span class="gp ${diff < 0 ? "faster" : "slower"}">${signed(diff, 3)}</span></span></button>`;
    li.firstElementChild.onclick = () => selectRace(r.slug);
    ol.appendChild(li);
  }
}

function setText(sel, txt) { const el = $(sel); if (el.textContent !== txt) el.textContent = txt; }

async function selectRace(slug, first = false) {
  const meta = state.index.races.find((r) => r.slug === slug) || state.index.races.at(-1);
  if (location.hash.slice(1) !== meta.slug) history.replaceState(null, "", `#${meta.slug}`);
  document.querySelectorAll(".tower button").forEach((b) => b.setAttribute("aria-current", String(b.dataset.slug === meta.slug)));
  document.documentElement.style.setProperty("--team", meta.pole.colour);
  setText("#event", `${meta.event}`);
  setText("#circuit", `Round ${meta.round} at ${meta.circuit}, ${(meta.length_m / 1000).toFixed(3)} km. Qualifying weekend of ${new Date(meta.date).toLocaleDateString(undefined, { day: "numeric", month: "long" })}.`);
  setText("#driver", meta.pole.name);
  setText("#team", meta.pole.team);
  setText("#pole-tag", meta.pole.acronym);
  setText("#t-pole", fmt(meta.pole.time));
  setText("#t-model", fmt(meta.model.time));
  const d = meta.model.time - meta.pole.time;
  const diff = $("#t-diff");
  diff.textContent = `${signed(d)} s`;
  diff.className = d < 0 ? "faster" : "slower";
  const ph = meta.model.physics_only, cor = meta.model.correlation;
  setText("#heldout", `${Math.abs(d / meta.pole.time * 100).toFixed(2)} % ${d < 0 ? "faster" : "slower"} than pole.`
    + (cor ? ` Forecast from ${cor.session === "SQ" ? "sprint qualifying" : "final practice"} pace (${fmt(cor.best_practice_s)}), no qualifying data used.` : "")
    + (ph ? ` Physics alone: ${fmt(ph)} (${signed(ph - meta.pole.time)} s).` : ""));
  const race = await loadRace(meta.slug);
  race.meta = meta;
  state.race = race;
  state.tau = 0; state.framePos = 0; state.progress = 0;
  state.intro = reduced || state.mode !== "race" ? 1 : 0;
  state.dirty = true;
  drawSeason();
  // warm the cache for neighbouring races
  const i = state.index.races.indexOf(meta);
  for (const j of [i - 1, i + 1]) if (state.index.races[j]) setTimeout(() => loadRace(state.index.races[j].slug), first ? 1200 : 300);
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  $("#hud").hidden = mode !== "race";
  $("#speed-legend").hidden = false;
  state.tau = 0; state.framePos = 0; state.progress = 0; state.intro = 1; state.playing = true;
  state.dirty = true;
}

function updateHUD() {
  const race = state.race, m = race.meta;
  updateCockpit(race);
  if (state.mode === "race") {
    setText("#clock-model", fmt(state.tauModel));
    setText("#clock-pole", fmt(state.tauPole));
    // gap at the model car's position: how much earlier it got there than the pole lap
    const g = interp(race.s, race.real.t, state.sOpt) - state.tauModel;
    const el = $("#gap");
    el.textContent = `${signed(g)} s`;
    el.className = `gap ${g >= 0 ? "faster" : "slower"}`;
    setText("#gap-note", Math.abs(g) < 0.0005 ? "model against the pole lap"
      : `model ${g > 0 ? "ahead of" : "behind"} ${m.pole.name.split(" ").at(-1)}`);
    setText("#caption", state.intro < 1 ? "Tracing the optimal line…"
      : `Purple car: the model's optimal lap. ${m.pole.team} car: ${m.pole.name}'s real pole lap. The gap says who reached the purple car's position first.`);
  } else if (state.mode === "line") {
    const H_ = race.hist, k = Math.min(H_.lap.length - 1, Math.round(state.framePos));
    const stage = { "centerline": "Start: drive down the middle of the track",
      "min-curvature": "Smoothest line through the corners (first guess)",
      "min-time (coarse)": "Optimizer on a 10 m grid: moves the line and speed together",
      "min-time": "Nonlinear optimizer: moves the line and the speed together", "optimal": "Optimal line" }[H_.stage[k]] || H_.stage[k];
    setText("#caption", `Best line after step ${k + 1} of ${H_.lap.length}. ${stage}. Lap ${fmt(H_.lap[k])}.`);
  } else {
    setText("#caption", stageOf(state.progress).text);
  }
}

// ── Extras: sectors, conditions, top five, cockpit ───────────────────────
function gripUsed(race) {
  // share of the tyres' friction ellipse the model car uses at each point [%]; against the
  // tyre limit only (not engine power), so flat-out straights read low and corners high
  if (race._grip) return race._grip;
  const p = race.params;
  race._grip = race.opt.v.map((v, i) => {
    const lim = limits(p, v), ax = race.opt.ax[i], ay = race.opt.ay[i];
    const lon = ax >= 0 ? p.mu_x_drive * (p.g + p.c_l * v * v) : lim.brake;
    return Math.min(100, 100 * Math.hypot(ay / lim.lat, ax / lon));
  });
  return race._grip;
}

function limits(p, v) {
  const load = p.g + p.c_l * v * v;
  return {
    lat: p.mu * load,
    brake: Math.min(p.mu_x_brake * load + p.c_d * v * v, p.a_brake_cap),
    drive: Math.max(0.5, Math.min(p.p_spec / Math.max(v, 5), p.mu_x_drive * load) - p.c_d * v * v),
  };
}

function updateCockpit(race) {
  const m = race.meta;
  const sM = state.mode === "race" ? state.sOpt
    : state.mode === "speed" ? state.progress * race.L
    : (state.framePos / Math.max(1, race.hist.lap.length - 1)) * race.L;
  const sP = state.mode === "race" ? state.sReal : sM;
  const at = (arr, s) => interp(race.s, arr, ((s % race.L) + race.L) % race.L);
  const vM = at(race.opt.v, sM), vP = at(race.real.v, sP), ax = at(race.opt.ax, sM), ay = at(race.opt.ay, sM);
  setText("#g-model-v", String(Math.round(vM * 3.6)));
  setText("#g-pole-v", String(Math.round(vP * 3.6)));
  setText("#g-pole-who", m.pole.acronym);
  const lim = limits(race.params, vM);
  // model pedals: braking when decelerating, full throttle when accelerating, partial at the limit
  const mThr = ax > 0.3 ? 1 : ax < -1 ? 0 : 0.45, mBrk = ax < -1 ? Math.min(1, -ax / lim.brake) : 0;
  $("#g-model-thr").style.setProperty("--v", mThr);
  $("#g-model-brk").style.setProperty("--v", mBrk);
  $("#g-pole-thr").style.setProperty("--v", Math.max(0, at(race.real.thr, sP)) / 100);
  $("#g-pole-brk").style.setProperty("--v", at(race.real.brk, sP) > 0.5 ? 1 : 0);
  drawGG(race, lim, ax, ay, sM);
}

function drawGG(race, lim, ax, ay, sM) {
  const cv = $("#gg"), { w: W, h: H, dpr } = fitCanvas(cv), ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const g = 9.81, R = Math.min(W, H) / 2 - 12, cx = W / 2, cy = H / 2, k = R / 6; // 6 g at the rim
  ctx.strokeStyle = COLORS.rule; ctx.lineWidth = 1;
  for (const gg of [2, 4, 6]) { ctx.beginPath(); ctx.arc(cx, cy, gg * k, 0, 7); ctx.stroke(); }
  ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
  // grip limit at this speed: cornering sideways, drive up, braking down
  ctx.beginPath();
  for (let a = 0; a <= 2 * Math.PI + 0.01; a += 0.05) {
    const rv = (Math.sin(a) > 0 ? lim.drive : lim.brake) / g;
    const px = cx + Math.cos(a) * (lim.lat / g) * k, py = cy - Math.sin(a) * rv * k;
    a === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.strokeStyle = "rgba(177,78,255,0.55)"; ctx.lineWidth = 1.4; ctx.stroke();
  // the last ~60 m of the lap as a fading trail
  for (let d = 60; d > 0; d -= 6) {
    const s = sM - d, i = Math.round(((((s % race.L) + race.L) % race.L) / race.L) * race.n) % race.n;
    ctx.fillStyle = `rgba(177,78,255,${0.4 - d / 200})`;
    ctx.beginPath(); ctx.arc(cx + (race.opt.ay[i] / g) * k, cy - (race.opt.ax[i] / g) * k, 2.2, 0, 7); ctx.fill();
  }
  ctx.fillStyle = COLORS.purple; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(cx + (ay / g) * k, cy - (ax / g) * k, 5, 0, 7); ctx.fill(); ctx.stroke();
  ctx.fillStyle = COLORS.tarmac; ctx.font = "500 10px 'Barlow Condensed', sans-serif"; ctx.textAlign = "center";
  ctx.fillText("accelerate", cx, cy - R - 3); ctx.fillText("brake", cx, cy + R + 10);
  ctx.textAlign = "right"; ctx.fillText("6 g", cx + R, cy - 3);
}

// ── Main loop ─────────────────────────────────────────────────────────────
function tick(ts) {
  const dt = state.last == null ? 0 : Math.min(0.1, (ts - state.last) / 1000);
  state.last = ts;
  const race = state.race;
  if (race) {
    const m = race.meta, adv = state.playing ? dt * state.rate : 0;
    let moved = state.dirty;
    if (state.mode === "race") {
      if (state.intro < 1) { state.intro = Math.min(1, state.intro + dt / 1.4); moved = true; }
      else if (adv) {
        const cycle = Math.max(m.model.time, m.pole.time) + 1.5;
        state.tau = (state.tau + adv) % cycle; moved = true;
      }
      state.tauModel = Math.min(state.tau, m.model.time);
      state.tauPole = Math.min(state.tau, m.pole.time);
      state.sOpt = interp(race.opt.t, race.s, state.tauModel);
      state.sReal = interp(race.real.t, race.s, state.tauPole);
      $("#scrub").value = Math.round((1000 * state.tau) / (Math.max(m.model.time, m.pole.time) + 1.5));
    } else if (state.mode === "line" && adv) {
      const last = race.hist.lap.length - 1;
      state.framePos = Math.min(last, state.framePos + adv * 1.6); moved = true;
      if (state.framePos >= last) state.playing = false;
      $("#scrub").value = Math.round((1000 * state.framePos) / last);
    } else if (state.mode === "speed" && adv) {
      state.progress = Math.min(1, state.progress + adv / 9); moved = true;
      if (state.progress >= 1) state.playing = false;
      $("#scrub").value = Math.round(1000 * state.progress);
    }
    $("#play").classList.toggle("paused", !state.playing);
    $("#play").setAttribute("aria-label", state.playing ? "Pause" : "Play");
    if (moved) { drawTrack(); drawCharts(); updateHUD(); state.dirty = false; }
  }
  requestAnimationFrame(tick);
}

// ── Wiring ────────────────────────────────────────────────────────────────
function setTowerCollapsed(collapsed) {
  $(".pitwall").classList.toggle("collapsed", collapsed);
  const btn = $("#tower-toggle");
  btn.setAttribute("aria-expanded", String(!collapsed));
  btn.setAttribute("aria-label", collapsed ? "Show the race list" : "Collapse the race list");
  try { localStorage.setItem("towerCollapsed", collapsed ? "1" : "0"); } catch (e) { /* private mode */ }
  state.dirty = true;
}

function wire() {
  let saved = null;
  try { saved = localStorage.getItem("towerCollapsed"); } catch (e) { /* private mode */ }
  setTowerCollapsed(saved == null ? window.innerWidth < 1500 : saved === "1");
  $("#tower-toggle").onclick = () => setTowerCollapsed(!$(".pitwall").classList.contains("collapsed"));
  document.querySelectorAll(".modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $("#play").onclick = () => {
    const r = state.race;
    if (!state.playing && state.mode === "line" && state.framePos >= r.hist.lap.length - 1) state.framePos = 0;
    if (!state.playing && state.mode === "speed" && state.progress >= 1) state.progress = 0;
    state.playing = !state.playing; state.dirty = true;
  };
  $("#scrub").oninput = (e) => {
    const f = e.target.value / 1000, r = state.race; if (!r) return;
    state.playing = false; state.intro = 1;
    if (state.mode === "race") state.tau = f * (Math.max(r.meta.model.time, r.meta.pole.time) + 1.5);
    else if (state.mode === "line") state.framePos = f * (r.hist.lap.length - 1);
    else state.progress = f;
    state.dirty = true;
  };
  document.querySelectorAll(".rates button").forEach((b) => (b.onclick = () => {
    state.rate = +b.dataset.rate;
    document.querySelectorAll(".rates button").forEach((x) => x.classList.toggle("on", x === b));
  }));
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input")) return;
    if (e.key === " ") { e.preventDefault(); $("#play").click(); }
    const races = state.index?.races || [], i = races.findIndex((r) => r.slug === state.race?.slug);
    if (e.key === "ArrowDown" && races[i + 1]) { e.preventDefault(); selectRace(races[i + 1].slug); }
    if (e.key === "ArrowUp" && races[i - 1]) { e.preventDefault(); selectRace(races[i - 1].slug); }
  });
  window.addEventListener("hashchange", () => { const s = location.hash.slice(1); if (s && s !== state.race?.slug) selectRace(s); });
  new ResizeObserver(() => { state.dirty = true; drawSeason(); }).observe(document.body);
}

async function init() {
  wire();
  const idx = await fetch("data/index.json", { cache: "no-cache" }).then((r) => r.json());
  state.index = idx;
  renderTower();
  const st = idx.stats;
  $("#season-stats").textContent = `Forecasting each pole from that weekend's practice pace, the model is typically ${st.median_abs_gap_pct.toFixed(2)} % from the real pole time (average ${st.mean_abs_gap_pct.toFixed(2)} %). Physics alone, with grip fitted on the other races, is typically ${st.physics_median_abs_gap_pct.toFixed(1)} % away.`;
  const groups = Object.entries(idx.grip_by_group || {});
  $("#m-car").textContent = `A point-mass car with downforce, drag and a friction ellipse, fitted to ${idx.laps_used} qualifying laps.`
    + (idx.per_circuit_setup ? " Each weekend's wing level (downforce and drag) is refitted on that weekend's practice, and grip is correlated to its practice pace, as teams do with their simulators." : "")
    + (groups.length > 1
      ? ` A grip factor per official circuit type (${groups.map(([g, v]) => `${g} ${v}`).join(", ")}) covers what a point mass cannot do; every prediction is checked with that race left out.`
      : ` One grip factor (${idx.grip_utilization}) covers what a point mass cannot do, checked by leaving each race out.`);
  $("#generated").textContent = `Data through round ${idx.races.at(-1).round}, generated ${idx.generated}.`;
  await selectRace(location.hash.slice(1) || idx.races.at(-1).slug, true);
  requestAnimationFrame(tick);
}
init().catch((err) => {
  setText("#event", "Race data did not load");
  setText("#circuit", `Serve this folder over HTTP (for example: python -m http.server) so data/ can be fetched. ${err}`);
});
