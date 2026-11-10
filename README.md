# LOGISTIGO

Streamlit app for planning delivery routes: login, a delivery heatmap, an analytics dashboard and a route planner.

## Setup

This folder doubles as the Python virtualenv (only the app files are tracked in git).

On a fresh clone:

```
python -m venv .
Scripts\activate          # macOS/Linux: source bin/activate
pip install -r requirements.txt
```

## Run

```
Scripts\streamlit.exe run logistigo.py
```

Run it from this folder: Streamlit reads the theme in `.streamlit/config.toml` from the folder you
start it in. Use the environment's own `streamlit` (as above, or after activating). A system-wide
`streamlit` won't have the app's packages and fails with `ModuleNotFoundError: No module named 'folium'`.

The app opens at http://localhost:8501 and needs internet access (OSRM for road routing,
Nominatim for place search).

## Test

```
Scripts\python.exe -m unittest -v test_logistigo
```

The tests run offline (the map services are simulated) against a temporary database, so they never
touch your real `users.db`.

## Data

Users, heatmap points and daily delivery summaries are stored in `users.db` **next to `logistigo.py`**,
whatever folder you launch it from. Set the `LOGISTIGO_DB` environment variable to use a different file.
The database is not committed to git; it is created automatically on first run.

## How routes are planned

- Each stop has a name, a priority and a volume (m³); each vehicle has a capacity (m³).
- Road distances come from the public OSRM server (straight-line distance if it is unreachable);
  place names are geocoded with Nominatim and cached for a day.
- OR-Tools solves a capacitated vehicle-routing problem: minimum total road distance, vehicles start at the
  start location and don't return, high/medium priority stops are pushed earlier in their route, and when
  everything can't fit the low-priority stops are left out first (and listed on screen).
- The "unoptimised" comparison serves the same stops in the order you entered them, filling vehicle 1, then 2, etc.
- Fuel is estimated as optimised distance ÷ the km-per-litre you enter.
- "High priority first" (off by default) makes every vehicle visit all its high-priority stops, then medium,
  then low, even when that means a longer drive. Off, priority only nudges the order.
- Each distinct plan is saved to analytics once, however many times you click the button.

## Importing stops

Under "Import stops from a file", upload a CSV or Excel (.xlsx) file with a `Location` column (or `Address`)
and optional `Priority` (High/Medium/Low) and `Volume (m³)` columns; "Download template" gives an example.
The stops land in the table for review before planning. Missing or invalid values fall back to Medium priority
and 1 m³, with a warning. Up to 200 stops; place lookups are limited to one per second, as Nominatim's
usage policy requires, so a large file takes a little while the first time.
- Analytics and the heatmap are per user: each account only sees its own history.
- Every stop shows the address it was matched to, so a wrong lookup is easy to spot and fix.

## Driver hand-off

- Each vehicle gets an "Open in Google Maps" button with its stops in order. Google Maps links hold at most
  10 stops, so longer routes are split into parts that each start where the previous one ended.
- Download the plan as a CSV run sheet, or as a printable run sheet (open it in a browser and print).
