"""F1 2026 optimal racing line explorer.

Run with ``make app`` (``streamlit run app/streamlit_app.py``). Reads the artifacts written
by ``f1line build-season``; nothing heavy is recomputed on page load. The only live
computation is the QSS re-simulation when the model sliders move (~ms).
"""

from __future__ import annotations

from dataclasses import replace

import streamlit as st
from simulator import simulator

from data import (
    available_circuits,
    load_artifacts,
    resimulate,
    season_table,
    simulator_payload,
)
from f1line.vehicle.model import VehicleParams

st.set_page_config(
    page_title="F1 2026 racing lines",
    page_icon=":material/sports_motorsports:",
    layout="wide",
)

load = st.cache_data(max_entries=8, show_spinner="Loading circuit...")(load_artifacts)

circuits = available_circuits()
if not circuits:
    st.title("F1 2026 racing lines")
    st.info(
        "No results yet. Build them first:\n\n"
        "```bash\nf1line fetch-season\nf1line build-season\n```",
        icon=":material/info:",
    )
    st.stop()

by_slug = {c["slug"]: c for c in circuits}


def _label(slug: str) -> str:
    c = by_slug[slug]
    rnd = f"R{c['round']}" if c.get("round") else "—"
    tag = {"upcoming": " (upcoming)", "cancelled": " (cancelled)"}.get(c.get("status"), "")
    return f"{rnd} · {c['name']}{tag}"


with st.sidebar:
    st.header("2026 calendar")
    slug = st.selectbox(
        "Circuit", list(by_slug), format_func=_label, key="circuit", bind="query-params"
    )
    overlays = st.pills(
        "Overlays",
        ["Real pole lap", "Min-curvature line", "Center line", "Corner numbers"],
        default=["Real pole lap", "Corner numbers"],
        selection_mode="multi",
    )

art = load(slug)
meta = art["meta"]
p0 = VehicleParams(**meta["params"])

defaults = (round(p0.mu, 2), round(p0.c_l * 1000, 2), int(round(p0.p_spec)))
with st.sidebar:
    st.subheader("Try your own car")
    st.caption("Re-simulates the optimal line with these limits (yellow car).")
    grip = st.slider("Tyre grip μ", 0.5, 5.0, defaults[0], 0.05)
    downforce = st.slider("Downforce c_L ×1000 [1/m]", 0.1, 3.0, defaults[1], 0.05)
    power = st.slider("Power / mass [W/kg]", 300, 1500, defaults[2], 10)

live = None
if (grip, downforce, power) != defaults:
    live = resimulate(art, replace(p0, mu=grip, c_l=downforce / 1000, p_spec=float(power)))

# ── Header ────────────────────────────────────────────────────────────────
lt = meta["lap_times_s"]
geo = meta["geometry"]
st.title(meta["name"])
src = {
    "tum": "TUM racetrack-database",
    "osm": "OpenStreetMap",
    "telemetry": "the drivers' mean line (approximate: no usable map data)",
}.get(geo["source"], geo["source"])
era_note = (
    f" Real lap is from {meta['data_year']} (no {meta['season']} qualifying yet), so it is "
    "an older-generation car." if meta["data_year"] != meta["season"] else ""
)
st.caption(
    f"{meta['season']} season · {meta['length_m'] / 1000:.3f} km · track geometry from {src} · "
    f"real lap: {meta['pole_driver']}, qualifying {meta['data_year']} (OpenF1).{era_note}"
)

with st.container(horizontal=True):
    gap = lt["min_time"] - lt["real_pole"]
    st.metric("Optimal lap (model)", f"{lt['min_time']:.3f} s", f"{gap:+.3f} s vs pole",
              delta_color="inverse", border=True)
    st.metric("Real pole lap", f"{lt['real_pole']:.3f} s", meta["pole_driver"], delta_color="off",
              border=True)
    st.metric("Min-curvature line", f"{lt['min_curvature']:.3f} s",
              f"{lt['min_curvature'] - lt['min_time']:+.3f} s vs optimal", delta_color="inverse",
              border=True)
    st.metric("Center line", f"{lt['centerline']:.3f} s",
              f"{lt['centerline'] - lt['min_time']:+.3f} s vs optimal", delta_color="inverse",
              border=True)
    if live:
        st.metric("Your car", f"{live['lap']:.3f} s", f"{live['lap'] - lt['min_time']:+.3f} s",
                  delta_color="inverse", border=True)

show = {
    "real": "Real pole lap" in overlays,
    "mincurv": "Min-curvature line" in overlays,
    "centerline": "Center line" in overlays,
    "corners": "Corner numbers" in overlays,
    "optimal": True,
}
simulator(simulator_payload(art, show, live), key="sim")

# ── Details ───────────────────────────────────────────────────────────────
tab_laps, tab_season, tab_model = st.tabs(
    [":material/timer: Lap times", ":material/table_chart: Season", ":material/tune: Model & data"]
)
with tab_laps:
    m = meta["metrics"]
    rows = [
        ("Center line (model)", lt["centerline"]),
        ("Min-curvature line (model)", lt["min_curvature"]),
        ("Minimum-time line (model)", lt["min_time"]),
        ("Pole driver's own line (model replay)", lt["replay_pole_line"]),
        (f"Real pole ({meta['pole_driver']} {meta['data_year']})", lt["real_pole"]),
    ]
    if lt.get("min_time_loco"):
        rows.append(("Minimum-time, grip fitted without this circuit", lt["min_time_loco"]))
    st.dataframe(
        {"Lap": [r[0] for r in rows], "Time [s]": [round(r[1], 3) for r in rows],
         "vs real pole [s]": [round(r[1] - lt["real_pole"], 3) for r in rows]},
        hide_index=True,
    )
    st.caption(
        f"Optimal line vs real line: mean lateral distance {m['line_dev_mean_m']:.1f} m, "
        f"max {m['line_dev_max_m']:.1f} m. Optimizer: {meta['solver']['method']} "
        f"({meta['solver'].get('iterations', '?')} iterations, "
        f"{meta['solver'].get('elapsed_s', 0):.1f} s); NLP vs independent QSS lap time "
        f"differ by {meta['solver'].get('nlp_qss_diff_pct') or 0:.2f} %."
    )

with tab_season:
    tbl = season_table()
    if tbl is None:
        st.info("Run `f1line build-season` to produce the season table.")
    else:
        st.dataframe(
            tbl,
            hide_index=True,
            column_config={
                "gap_pct": st.column_config.NumberColumn("Optimal vs pole [%]", format="%+.2f"),
                "gap_loco_pct": st.column_config.NumberColumn(
                    "Held-out optimal vs pole [%]", format="%+.2f"),
                "replay_err_pct": st.column_config.NumberColumn(
                    "Replay of pole line [%]", format="%+.2f"),
                "replay_rmse_kmh": st.column_config.NumberColumn(
                    "Replay speed RMSE [km/h]", format="%.1f"),
            },
        )

with tab_model:
    st.markdown(
        "**Model.** Point mass with aerodynamic downforce and drag and a friction ellipse "
        "(`docs/MODEL.md`). The racing line is optimized in two stages: a minimum-curvature "
        "quadratic program, then a minimum-time nonlinear program (CasADi + IPOPT). Speeds come "
        "from a quasi-steady-state forward/backward simulation."
    )
    st.dataframe(
        {"Parameter": list(meta["params"]),
         "Value": [f"{v:.5g}" for v in meta["params"].values()]},
        hide_index=True,
    )
    st.markdown(
        "**Data.** Laps: [OpenF1](https://openf1.org) (unofficial F1 timing feed, research and "
        "education only). Geometry: [TUM racetrack-database](https://github.com/TUMFTM/"
        "racetrack-database) (LGPL-3.0) or OpenStreetMap contributors (ODbL). Corner numbers: "
        "MultiViewer."
    )
    st.json(geo, expanded=False)
