"""Racing-line simulator component (Streamlit Custom Components v2, inline).

One canvas view: track map on the left, synchronized charts on the right, and three
animation modes driven entirely in the browser:

- ``lap``:   the optimal car and the real pole lap (ghost) drive the lap in real time;
             chart cursors follow the optimal car.
- ``line``:  replay of the optimizer iterates (centerline -> min-curvature sweep ->
             IPOPT iterations -> optimal), line coloured by its own QSS speed.
- ``speed``: how the speed profile is decided: corner limits, acceleration pass,
             braking pass, final profile.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

_HTML = """
<div class="sim">
  <div class="bar">
    <div class="modes">
      <button type="button" data-mode="lap">Lap simulation</button>
      <button type="button" data-mode="line">How the line is found</button>
      <button type="button" data-mode="speed">How speed is decided</button>
    </div>
    <div class="transport">
      <button type="button" class="play" title="Play / pause">&#9654;</button>
      <input class="scrub" type="range" min="0" max="1000" value="0" />
      <select class="rate" title="Playback speed">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="4">4x</option>
      </select>
    </div>
  </div>
  <div class="readout"></div>
  <div class="body">
    <canvas class="map"></canvas>
    <canvas class="charts"></canvas>
  </div>
  <div class="legend"></div>
</div>
"""

_CSS = """
.sim {
  font-family: var(--st-font, sans-serif);
  color: var(--st-text-color);
  background: var(--st-background-color);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
  justify-content: space-between;
}
.modes, .transport {
  display: flex;
  gap: 0.25rem;
  align-items: center;
}
button, select {
  font: inherit;
  font-size: 0.85rem;
  color: var(--st-text-color);
  background: var(--st-secondary-background-color);
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-button-radius, 0.5rem);
  padding: 0.3rem 0.7rem;
  cursor: pointer;
}
button.active {
  background: var(--st-primary-color);
  border-color: var(--st-primary-color);
  color: #fff;
}
button.play {
  min-width: 2.5rem;
}
.scrub {
  width: 14rem;
  accent-color: var(--st-primary-color);
}
.readout {
  font-family: var(--st-code-font, monospace);
  font-size: 0.8rem;
  min-height: 1.2rem;
  opacity: 0.85;
}
.body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 0.75rem;
}
@media (max-width: 900px) {
  .body { grid-template-columns: minmax(0, 1fr); }
}
canvas {
  width: 100%;
  height: 560px;
  background: var(--st-secondary-background-color);
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-base-radius, 0.5rem);
  display: block;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  font-size: 0.8rem;
  opacity: 0.85;
}
.legend span::before {
  content: "";
  display: inline-block;
  width: 1.2rem;
  height: 0.25rem;
  margin-right: 0.35rem;
  vertical-align: middle;
  background: var(--c);
  border-radius: 2px;
}
"""

_JS = """
const STATE = new WeakMap();
const G = 9.81;

function cssVar(el, name, fallback) {
  const v = getComputedStyle(el).getPropertyValue(name).trim();
  return v || fallback;
}

// Turbo-like colormap, t in [0, 1].
function speedColor(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.0, [48, 18, 59]], [0.15, [70, 107, 227]], [0.35, [41, 187, 236]],
    [0.5, [49, 241, 125]], [0.65, [190, 238, 44]], [0.8, [251, 174, 42]],
    [0.92, [234, 79, 13]], [1.0, [122, 4, 3]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const f = (t - t0) / (t1 - t0);
      const c = c0.map((v, k) => Math.round(v + f * (c1[k] - v)));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(122,4,3)";
}

function interp(xs, ys, x) {
  // xs ascending
  let lo = 0, hi = xs.length - 1;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[hi]) return ys[hi];
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const f = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
  return ys[lo] + f * (ys[hi] - ys[lo]);
}

function sizeCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  const w = Math.max(10, Math.round(r.width * dpr));
  const h = Math.max(10, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, r.width, r.height];
}

function makeView(st) {
  const d = st.data, tr = d.track;
  const rot = (tr.rotation || 0) * Math.PI / 180;
  const c = Math.cos(rot), s = Math.sin(rot);
  const R = (x, y) => [c * x - s * y, s * x + c * y];
  // bounds of the rotated boundaries
  let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
  for (let i = 0; i < tr.x.length; i++) {
    for (const a of [tr.wl[i], -tr.wr[i]]) {
      const [px, py] = R(tr.x[i] + a * tr.nx[i], tr.y[i] + a * tr.ny[i]);
      minx = Math.min(minx, px); maxx = Math.max(maxx, px);
      miny = Math.min(miny, py); maxy = Math.max(maxy, py);
    }
  }
  return { R, minx, maxx, miny, maxy };
}

function lineXY(tr, alpha) {
  const n = tr.x.length, xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = alpha ? alpha[i] : 0;
    xs[i] = tr.x[i] + a * tr.nx[i];
    ys[i] = tr.y[i] + a * tr.ny[i];
  }
  return [xs, ys];
}

function posAt(tr, alpha, sPos) {
  const n = tr.x.length, L = tr.L;
  const u = ((sPos % L) + L) % L / (L / n);
  const i = Math.floor(u) % n, j = (i + 1) % n, f = u - Math.floor(u);
  const a0 = alpha ? alpha[i] : 0, a1 = alpha ? alpha[j] : 0;
  const x0 = tr.x[i] + a0 * tr.nx[i], y0 = tr.y[i] + a0 * tr.ny[i];
  const x1 = tr.x[j] + a1 * tr.nx[j], y1 = tr.y[j] + a1 * tr.ny[j];
  return [x0 + f * (x1 - x0), y0 + f * (y1 - y0)];
}

function drawMap(st) {
  const { mapCv, data: d } = st;
  const [ctx, W, H] = sizeCanvas(mapCv);
  const tr = d.track, v = st.view, n = tr.x.length;
  ctx.clearRect(0, 0, W, H);
  const pad = 28;
  const sc = Math.min((W - 2 * pad) / (v.maxx - v.minx), (H - 2 * pad) / (v.maxy - v.miny));
  const ox = (W - sc * (v.maxx - v.minx)) / 2, oy = (H - sc * (v.maxy - v.miny)) / 2;
  const P = (x, y) => {
    const [rx, ry] = v.R(x, y);
    return [ox + (rx - v.minx) * sc, H - (oy + (ry - v.miny) * sc)];
  };
  st.P = P;
  const text = st.colors.text;

  // Asphalt band (car-centre admissible region)
  ctx.beginPath();
  for (let i = 0; i <= n; i++) {
    const k = i % n, a = tr.wl[k];
    const [px, py] = P(tr.x[k] + a * tr.nx[k], tr.y[k] + a * tr.ny[k]);
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  }
  for (let i = n; i >= 0; i--) {
    const k = i % n, a = -tr.wr[k];
    const [px, py] = P(tr.x[k] + a * tr.nx[k], tr.y[k] + a * tr.ny[k]);
    ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(128,128,128,0.28)";
  ctx.fill("evenodd");
  for (const side of [1, -1]) {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const k = i % n, a = side > 0 ? tr.wl[k] : -tr.wr[k];
      const [px, py] = P(tr.x[k] + a * tr.nx[k], tr.y[k] + a * tr.ny[k]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.strokeStyle = "rgba(128,128,128,0.75)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  const strokeLine = (alpha, color, width, dash) => {
    const [xs, ys] = lineXY(tr, alpha);
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const [px, py] = P(xs[i % n], ys[i % n]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.setLineDash(dash || []);
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    ctx.setLineDash([]);
  };
  const speedLine = (alpha, vel, width) => {
    const [xs, ys] = lineXY(tr, alpha);
    const vmin = st.vmin, vmax = st.vmax;
    ctx.lineWidth = width; ctx.lineCap = "round";
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const [ax, ay] = P(xs[i], ys[i]);
      const [bx, by] = P(xs[j], ys[j]);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
      ctx.strokeStyle = speedColor((vel[i] - vmin) / (vmax - vmin));
      ctx.stroke();
    }
    ctx.lineCap = "butt";
  };

  const show = d.show;
  if (show.centerline) strokeLine(null, "rgba(128,128,128,0.9)", 1, [4, 4]);

  if (st.mode === "line") {
    const H_ = d.history, k = st.frame;
    if (show.real) strokeLine(d.lines.real.alpha, st.colors.real, 1.5);
    // faint trail of earlier iterates
    for (let j = Math.max(0, k - 6); j < k; j++) {
      strokeLine(H_.alpha[j], `rgba(128,128,128,${0.1 + 0.05 * (j - k + 6)})`, 1);
    }
    speedLine(H_.alpha[k], H_.v[k], 3);
  } else {
    if (show.real) strokeLine(d.lines.real.alpha, st.colors.real, 1.5);
    if (show.mincurv) strokeLine(d.lines.mincurv.alpha, st.colors.mincurv, 1.5, [6, 3]);
    if (show.optimal) {
      let vel = d.lines.optimal.v;
      if (st.mode === "speed") vel = st.stageSpeed || vel;
      speedLine(d.lines.optimal.alpha, vel, 3);
    }
  }

  if (show.corners) {
    ctx.font = "11px sans-serif"; ctx.fillStyle = text; ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const cn of d.corners) {
      const i = Math.round(cn.s / (tr.L / n)) % n;
      const off = Math.max(tr.wl[i], tr.wr[i]) + 22 / sc;
      // put label on the outside of the corner
      const side = (tr.kappa && tr.kappa[i] > 0) ? -1 : 1;
      const [px, py] = P(tr.x[i] + side * off * tr.nx[i], tr.y[i] + side * off * tr.ny[i]);
      ctx.globalAlpha = 0.8; ctx.fillText(cn.label, px, py); ctx.globalAlpha = 1;
    }
  }

  // start/finish
  {
    const a = tr.wl[0], b = -tr.wr[0];
    const [ax, ay] = P(tr.x[0] + a * tr.nx[0], tr.y[0] + a * tr.ny[0]);
    const [bx, by] = P(tr.x[0] + b * tr.nx[0], tr.y[0] + b * tr.ny[0]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    ctx.strokeStyle = text; ctx.lineWidth = 3; ctx.stroke();
  }

  // cars
  if (st.mode === "lap") {
    const dot = (alpha, sPos, fill, ring) => {
      const [x, y] = posAt(tr, alpha, sPos);
      const [px, py] = P(x, y);
      ctx.beginPath(); ctx.arc(px, py, 6.5, 0, 2 * Math.PI);
      ctx.fillStyle = fill; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = ring; ctx.stroke();
    };
    if (show.real) dot(d.lines.real.alpha, st.sReal, st.colors.real, st.colors.bg);
    dot(d.lines.optimal.alpha, st.sOpt, st.colors.primary, "#fff");
    if (d.lines.live) dot(d.lines.optimal.alpha, st.sLive, st.colors.live, st.colors.bg);
  } else if (st.cursorS != null) {
    const alpha = st.mode === "line" ? d.history.alpha[st.frame] : d.lines.optimal.alpha;
    const [x, y] = posAt(tr, alpha, st.cursorS);
    const [px, py] = P(x, y);
    ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI);
    ctx.fillStyle = st.colors.primary; ctx.fill();
  }

  // speed colour bar
  const bw = 120, bh = 8, bx0 = W - bw - 14, by0 = H - 22;
  for (let i = 0; i < bw; i++) {
    ctx.fillStyle = speedColor(i / bw); ctx.fillRect(bx0 + i, by0, 1, bh);
  }
  ctx.fillStyle = text; ctx.font = "10px sans-serif"; ctx.textBaseline = "bottom";
  ctx.textAlign = "left"; ctx.fillText(`${Math.round(st.vmin * 3.6)}`, bx0, by0 - 2);
  ctx.textAlign = "right"; ctx.fillText(`${Math.round(st.vmax * 3.6)} km/h`, bx0 + bw, by0 - 2);
}

function drawCharts(st) {
  const d = st.data, tr = d.track, L = tr.L;
  const [ctx, W, H] = sizeCanvas(st.chartCv);
  ctx.clearRect(0, 0, W, H);
  const left = 46, right = 10, top = 8, gap = 22;
  const panels = st.mode === "line"
    ? [["Speed [km/h]", 0.42], ["Lateral g", 0.2], ["Lap time by iteration [s]", 0.38]]
    : st.mode === "speed"
      ? [["Speed [km/h]", 0.62], ["Longitudinal g", 0.38]]
      : [["Speed [km/h]", 0.36], ["Throttle / brake (real) [%]", 0.2],
         ["Lateral g", 0.2], ["Gap to real lap [s]  (+ = optimal ahead)", 0.24]];
  const usable = H - top - gap * panels.length - 14;
  let y0 = top;
  const xOf = (sPos) => left + (sPos / L) * (W - left - right);
  st.chartLayout = { left, right, W };
  const text = st.colors.text;

  const axes = (title, h, ymin, ymax, xmax, xlabel) => {
    const yTo = (v) => y0 + gap - 6 + h - ((v - ymin) / (ymax - ymin)) * h;
    ctx.fillStyle = text; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic"; ctx.globalAlpha = 0.85;
    ctx.fillText(title, left, y0 + gap - 10); ctx.globalAlpha = 1;
    ctx.strokeStyle = "rgba(128,128,128,0.35)"; ctx.lineWidth = 1;
    ctx.strokeRect(left, y0 + gap - 6, W - left - right, h);
    // "nice" tick step: 1, 2 or 5 x 10^k giving 3-5 ticks
    const raw = (ymax - ymin) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const dec = step < 1 ? (step < 0.1 ? 2 : 1) : 0;
    ctx.font = "10px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let v = Math.ceil(ymin / step) * step; v <= ymax + 1e-9; v += step) {
      const yy = yTo(v);
      ctx.globalAlpha = 0.6; ctx.fillText(v.toFixed(dec), left - 4, yy);
      ctx.globalAlpha = 0.15; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(W - right, yy);
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    return yTo;
  };
  const series = (xs, ys, yTo, color, width, dash, upto) => {
    ctx.beginPath(); let started = false;
    for (let i = 0; i < ys.length; i++) {
      if (upto != null && !upto(i)) { started = false; continue; }
      const px = xs(i), py = yTo(ys[i]);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    ctx.setLineDash([]);
  };
  const sx = (i) => xOf(tr.s[i]);
  const cursor = (sPos, h) => {
    if (sPos == null) return;
    const px = xOf(((sPos % L) + L) % L);
    ctx.strokeStyle = st.colors.primary; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, y0 + gap - 6); ctx.lineTo(px, y0 + gap - 6 + h); ctx.stroke();
  };
  const kmh = (arr) => arr.map((v) => v * 3.6);
  const cur = st.mode === "lap" ? st.sOpt : st.cursorS;

  for (const [title, frac] of panels) {
    const h = usable * frac;
    if (title.startsWith("Speed")) {
      const vmax = st.vmax * 3.6 + 10, vmin = Math.max(0, st.vmin * 3.6 - 20);
      const yTo = axes(title, h, vmin, vmax);
      if (st.mode === "line") {
        if (d.show.real) series(sx, kmh(d.lines.real.v), yTo, st.colors.real, 1.2);
        series(sx, kmh(d.history.v[st.frame]), yTo, st.colors.primary, 1.6);
      } else if (st.mode === "speed") {
        const S = d.stages, p = st.progress;
        const n = S.v_corner.length;
        series(sx, kmh(S.v_corner), yTo, st.colors.limit, 1, [3, 3]);
        if (p > 0.2) {
          const lim = Math.min(1, (p - 0.2) / 0.35);
          series(sx, kmh(S.v_accel), yTo, st.colors.accel, 1.4, null, (i) => i / n <= lim);
        }
        if (p > 0.55) {
          const lim = Math.min(1, (p - 0.55) / 0.35);
          series(sx, kmh(S.v_brake), yTo, st.colors.brake, 1.4, null, (i) => i / n >= 1 - lim);
        }
        if (p >= 0.9) series(sx, kmh(S.v_final), yTo, st.colors.primary, 2.4);
        if (d.show.real) series(sx, kmh(d.lines.real.v), yTo, st.colors.real, 1, [2, 2]);
      } else {
        if (d.show.real) series(sx, kmh(d.lines.real.v), yTo, st.colors.real, 1.2);
        if (d.show.mincurv) series(sx, kmh(d.lines.mincurv.v), yTo, st.colors.mincurv, 1, [5, 3]);
        if (d.lines.live) series(sx, kmh(d.lines.live.v), yTo, st.colors.live, 1.2);
        series(sx, kmh(d.lines.optimal.v), yTo, st.colors.primary, 1.6);
      }
      cursor(cur, h);
    } else if (title.startsWith("Throttle")) {
      const yTo = axes(title, h, 0, 100);
      const r = d.lines.real;
      ctx.fillStyle = st.colors.accel; ctx.globalAlpha = 0.35;
      ctx.beginPath(); ctx.moveTo(sx(0), yTo(0));
      for (let i = 0; i < r.throttle.length; i++) ctx.lineTo(sx(i), yTo(r.throttle[i]));
      ctx.lineTo(sx(r.throttle.length - 1), yTo(0)); ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 0.8; ctx.fillStyle = st.colors.brake;
      for (let i = 0; i < r.brake.length; i++) {
        if (r.brake[i] > 0.5) ctx.fillRect(sx(i), yTo(100), Math.max(1, sx(1) - sx(0)) + 0.5, h * 0.18);
      }
      ctx.globalAlpha = 1;
      cursor(cur, h);
    } else if (title.startsWith("Lateral")) {
      const yTo = axes(title, h, 0, 7);
      if (st.mode === "line") {
        const v = d.history.v[st.frame], k = st.kappaFrame;
        series(sx, v.map((vi, i) => vi * vi * Math.abs(k[i]) / G), yTo, st.colors.primary, 1.2);
      } else {
        series(sx, d.lines.optimal.alat.map((a) => a / G), yTo, st.colors.primary, 1.2);
      }
      cursor(cur, h);
    } else if (title.startsWith("Longitudinal")) {
      const yTo = axes(title, h, -6, 3);
      series(sx, d.lines.optimal.along.map((a) => a / G), yTo, st.colors.primary, 1.2);
      if (d.show.real) series(sx, d.lines.real.along.map((a) => a / G), yTo, st.colors.real, 1, [2, 2]);
      cursor(cur, h);
    } else if (title.startsWith("Gap")) {
      const gapArr = d.lines.gap;
      let lo = Math.min(0, ...gapArr), hi = Math.max(0.5, ...gapArr);
      const yTo = axes(title, h, lo, hi);
      ctx.strokeStyle = "rgba(128,128,128,0.6)"; ctx.beginPath();
      ctx.moveTo(left, yTo(0)); ctx.lineTo(W - right, yTo(0)); ctx.stroke();
      series(sx, gapArr, yTo, st.colors.primary, 1.4);
      cursor(cur, h);
    } else if (title.startsWith("Lap time by iteration")) {
      const lt = d.history.lap, m = lt.length;
      const lo = Math.min(...lt), hi = Math.max(...lt);
      const xs = (i) => left + (i / Math.max(1, m - 1)) * (W - left - right);
      const yTo = axes(title, h, lo - 0.5, hi + 0.5);
      series(xs, lt, yTo, st.colors.primary, 1.4);
      ctx.fillStyle = st.colors.primary;
      ctx.beginPath(); ctx.arc(xs(st.frame), yTo(lt[st.frame]), 4, 0, 2 * Math.PI); ctx.fill();
      // stage boundaries (labels staggered so neighbours don't overlap)
      ctx.font = "10px sans-serif"; ctx.fillStyle = text; ctx.textAlign = "left";
      let prev = "", row = 0;
      for (let i = 0; i < m; i++) {
        const stg = d.history.stage[i].split(" (")[0];
        if (stg !== prev) {
          ctx.globalAlpha = 0.5; ctx.beginPath();
          ctx.moveTo(xs(i), y0 + gap - 6); ctx.lineTo(xs(i), y0 + gap - 6 + h);
          ctx.strokeStyle = "rgba(128,128,128,0.6)"; ctx.stroke();
          const tx = Math.min(xs(i) + 3, W - right - ctx.measureText(stg).width - 2);
          ctx.globalAlpha = 0.8; ctx.fillText(stg, tx, y0 + gap + 6 + 12 * row);
          ctx.globalAlpha = 1;
          row = (row + 1) % 3;
          prev = stg;
        }
      }
    }
    y0 += h + gap;
  }
  ctx.fillStyle = text; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
  ctx.globalAlpha = 0.6;
  ctx.fillText(st.mode === "line" ? "" : "distance along lap [m]", (left + W - right) / 2, H - 4);
  ctx.globalAlpha = 1;
}

function menger(tr, alpha) {
  const [xs, ys] = lineXY(tr, alpha), n = xs.length, k = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = (i - 1 + n) % n, b = (i + 1) % n;
    const ax = xs[i] - xs[a], ay = ys[i] - ys[a], bx = xs[b] - xs[i], by = ys[b] - ys[i];
    const cx = xs[b] - xs[a], cy = ys[b] - ys[a];
    const den = Math.hypot(ax, ay) * Math.hypot(bx, by) * Math.hypot(cx, cy) + 1e-9;
    k[i] = 2 * (ax * by - ay * bx) / den;
  }
  return k;
}

function readout(st) {
  const d = st.data, el = st.root.querySelector(".readout");
  const L = d.track.L;
  const at = (arr, sPos) => interp(d.track.s, arr, ((sPos % L) + L) % L);
  if (st.mode === "lap") {
    const vO = at(d.lines.optimal.v, st.sOpt) * 3.6;
    const vR = at(d.lines.real.v, st.sReal) * 3.6;
    const gap = at(d.lines.gap, st.sOpt);
    let txt = `t ${st.t.toFixed(2)} s  |  optimal ${st.sOpt.toFixed(0)} m ${vO.toFixed(0)} km/h` +
      `  |  ${d.lines.real.label} ${st.sReal.toFixed(0)} m ${vR.toFixed(0)} km/h` +
      `  |  gap ${gap >= 0 ? "+" : ""}${gap.toFixed(2)} s`;
    if (d.lines.live) txt += `  |  your setup ${st.sLive.toFixed(0)} m`;
    el.textContent = txt;
  } else if (st.mode === "line") {
    const h = d.history;
    el.textContent = `step ${st.frame + 1}/${h.lap.length}  |  ${h.stage[st.frame]}` +
      `${h.stage[st.frame].startsWith("min-time") ? " iteration " + h.iter[st.frame] : ""}` +
      `  |  lap time ${h.lap[st.frame].toFixed(3)} s`;
  } else {
    const p = st.progress;
    const step = p < 0.2 ? "1/4 corner limits: max speed each curvature allows (lateral grip only)"
      : p < 0.55 ? "2/4 acceleration pass: from each corner, accelerate as hard as power + grip allow"
      : p < 0.9 ? "3/4 braking pass: backwards from each corner, brake as late as possible"
      : `4/4 final: the lowest of all three = speed profile, lap ${d.lines.optimal.lap.toFixed(3)} s`;
    el.textContent = step;
  }
}

function frameTick(st, ts) {
  const dt = st.lastTs == null ? 0 : Math.min(0.1, (ts - st.lastTs) / 1000);
  st.lastTs = ts;
  const d = st.data, rate = parseFloat(st.rateEl.value);
  if (st.playing) {
    if (st.mode === "lap") {
      st.t = (st.t + dt * rate) % d.lines.optimal.lap;
    } else if (st.mode === "line") {
      st.frameAcc += dt * rate * 5;
      if (st.frameAcc >= 1) {
        st.frame = Math.min(d.history.lap.length - 1, st.frame + Math.floor(st.frameAcc));
        st.frameAcc = 0;
        if (st.frame >= d.history.lap.length - 1) st.playing = false;
        st.kappaFrame = menger(d.track, d.history.alpha[st.frame]);
      }
    } else {
      st.progress = Math.min(1, st.progress + dt * rate / 8);
      if (st.progress >= 1) st.playing = false;
    }
  }
  update(st);
  st.raf = requestAnimationFrame((t) => frameTick(st, t));
}

function update(st) {
  const d = st.data;
  if (st.mode === "lap") {
    const o = d.lines.optimal, r = d.lines.real;
    st.sOpt = interp(o.t, d.track.s, st.t);
    st.sReal = interp(r.t, d.track.s, Math.min(st.t, r.t[r.t.length - 1]));
    if (d.lines.live) st.sLive = interp(d.lines.live.t, d.track.s, Math.min(st.t, d.lines.live.t[d.lines.live.t.length - 1]));
    st.scrub.value = Math.round(1000 * st.t / o.lap);
  } else if (st.mode === "line") {
    st.scrub.value = Math.round(1000 * st.frame / Math.max(1, d.history.lap.length - 1));
  } else {
    st.scrub.value = Math.round(1000 * st.progress);
    const S = d.stages, p = st.progress;
    st.stageSpeed = p < 0.2 ? S.v_corner : p < 0.55 ? S.v_accel : p < 0.9 ? S.v_brake : S.v_final;
  }
  st.playBtn.innerHTML = st.playing ? "&#10073;&#10073;" : "&#9654;";
  for (const b of st.root.querySelectorAll(".modes button")) {
    b.classList.toggle("active", b.dataset.mode === st.mode);
  }
  drawMap(st);
  drawCharts(st);
  readout(st);
}

function legend(st) {
  const d = st.data, el = st.root.querySelector(".legend"), c = st.colors;
  const items = [];
  if (st.mode === "speed") {
    items.push(["corner limit", c.limit], ["acceleration pass", c.accel],
               ["braking pass", c.brake], ["final speed", c.primary]);
    if (d.show.real) items.push([`real pole lap ${d.lines.real.label}`, c.real]);
  } else {
    items.push(["optimal line (colour = speed)", c.primary]);
    if (d.show.real) items.push([`real pole lap ${d.lines.real.label}`, c.real]);
    if (d.show.mincurv && st.mode === "lap") items.push(["min-curvature line", c.mincurv]);
    if (d.lines.live && st.mode === "lap") items.push(["your setup (sliders)", c.live]);
  }
  el.innerHTML = "";
  for (const [label, col] of items) {
    const sp = document.createElement("span");
    sp.textContent = label; sp.style.setProperty("--c", col); el.appendChild(sp);
  }
}

export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector(".sim");
  if (!root || !data) return;
  let st = STATE.get(parentElement);
  const fresh = !st;
  if (fresh) {
    st = {
      root, mode: "lap", playing: false, t: 0, frame: 0, frameAcc: 0, progress: 0,
      cursorS: null, lastTs: null,
      mapCv: root.querySelector(".map"), chartCv: root.querySelector(".charts"),
      playBtn: root.querySelector(".play"), scrub: root.querySelector(".scrub"),
      rateEl: root.querySelector(".rate"),
    };
    STATE.set(parentElement, st);
  }
  const circuitChanged = !st.data || st.data.circuit !== data.circuit;
  st.data = data;
  st.colors = {
    text: cssVar(root, "--st-text-color", "#ddd"),
    bg: cssVar(root, "--st-background-color", "#111"),
    primary: cssVar(root, "--st-primary-color", "#e10600"),
    real: "rgba(150,150,160,0.95)",
    mincurv: "#b07cff",
    live: "#ffd400",
    accel: "#2bb673",
    brake: "#e74c3c",
    limit: "#f5b942",
  };
  const allV = data.lines.optimal.v.concat(data.lines.real.v);
  st.vmin = Math.min(...allV); st.vmax = Math.max(...allV);
  st.view = makeView(st);
  if (circuitChanged) {
    st.t = 0; st.frame = 0; st.progress = 0; st.playing = false; st.cursorS = null;
  }
  st.frame = Math.min(st.frame, data.history.lap.length - 1);
  st.kappaFrame = menger(data.track, data.history.alpha[st.frame]);
  legend(st);

  if (fresh) {
    for (const b of root.querySelectorAll(".modes button")) {
      b.onclick = () => {
        st.mode = b.dataset.mode; st.playing = true; st.cursorS = null;
        if (st.mode === "line") { st.frame = 0; st.kappaFrame = menger(st.data.track, st.data.history.alpha[0]); }
        if (st.mode === "speed") st.progress = 0;
        legend(st);
      };
    }
    st.playBtn.onclick = () => {
      if (!st.playing) {
        if (st.mode === "line" && st.frame >= st.data.history.lap.length - 1) st.frame = 0;
        if (st.mode === "speed" && st.progress >= 1) st.progress = 0;
      }
      st.playing = !st.playing;
    };
    st.scrub.oninput = () => {
      const f = st.scrub.value / 1000, d = st.data;
      st.playing = false;
      if (st.mode === "lap") st.t = f * d.lines.optimal.lap;
      else if (st.mode === "line") {
        st.frame = Math.round(f * (d.history.lap.length - 1));
        st.kappaFrame = menger(d.track, d.history.alpha[st.frame]);
      } else st.progress = f;
    };
    st.chartCv.onmousemove = (e) => {
      if (st.playing || st.mode === "lap" || !st.chartLayout) return;
      const r = st.chartCv.getBoundingClientRect(), lay = st.chartLayout;
      const f = (e.clientX - r.left - lay.left) / (lay.W - lay.left - lay.right);
      st.cursorS = Math.max(0, Math.min(1, f)) * st.data.track.L;
    };
    st.chartCv.onmouseleave = () => { st.cursorS = null; };
    st.raf = requestAnimationFrame((t) => frameTick(st, t));
  }
  return () => {
    cancelAnimationFrame(st.raf);
    STATE.delete(parentElement);
  };
}
"""

_SIMULATOR = st.components.v2.component(
    "f1line_simulator",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def simulator(data: dict[str, Any], *, key: Optional[str] = None) -> Any:
    """Mount the simulator. ``data`` is built by ``app.data.simulator_payload``."""
    return _SIMULATOR(data=data, key=key)
