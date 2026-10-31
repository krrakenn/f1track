.PHONY: setup data season web serve test app report all clean

PYTHON := python
PIP := pip

# ── Environment setup ───────────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv .venv
	.venv\Scripts\pip install --upgrade pip
	.venv\Scripts\pip install -e ".[dev]"
	@echo "Setup complete. Activate with: .venv\Scripts\activate"

# ── Data: TUM tracks, OpenF1 qualifying laps and OSM geometry for the 2026 calendar ──
# (FastF1's upstream answers HTTP 401 since 2026; OpenF1 serves the same feed.)
data:
	.venv\Scripts\f1line fetch-tracks
	.venv\Scripts\f1line fetch-season

# ── Geometry -> calibration (grip utilization, LOCO) -> optimization -> app artifacts ──
season:
	.venv\Scripts\f1line build-season

# ── Static web app: export data, serve locally ───────────────────────────────
web:
	.venv\Scripts\f1line export-web

serve:
	.venv\Scripts\python -m http.server 8600 --directory web

# ── Tests ───────────────────────────────────────────────────────────────────
test:
	.venv\Scripts\pytest --cov=src/f1line --cov-report=term-missing tests/

# ── App ─────────────────────────────────────────────────────────────────────
app:
	.venv\Scripts\streamlit run app/streamlit_app.py

# ── Report ──────────────────────────────────────────────────────────────────
report:
	.venv\Scripts\f1line report

# ── Full pipeline ────────────────────────────────────────────────────────────
all: data season report web

# ── Clean ───────────────────────────────────────────────────────────────────
clean:
	rmdir /s /q data\interim data\processed data\season models reports\figures .pytest_cache htmlcov 2>nul || true
	del /q .coverage 2>nul || true
