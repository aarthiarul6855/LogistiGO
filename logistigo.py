import os
import hashlib
import hmac
import html
import secrets
import sqlite3
from datetime import datetime
from urllib.parse import urlencode

import altair as alt
import folium
import pandas as pd
import requests
import streamlit as st
from folium.plugins import HeatMap
from geopy.distance import geodesic
from streamlit_folium import st_folium

st.set_page_config(page_title="Logistigo", page_icon=":material/local_shipping:", layout="wide")

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    HAVE_ORTOOLS = True
except ImportError:  # app still works, with a simpler fallback planner
    HAVE_ORTOOLS = False

OSRM_URL = "http://router.project-osrm.org"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HTTP_TIMEOUT = 20  # seconds
USER_AGENT = "logistigo-app"
PRIORITY_RANK = {"high": 1, "medium": 2, "low": 3}
PRIMARY_COLOR = "#1F5FAD"  # matches .streamlit/config.toml
PRIORITY_COLOR = {"high": "#D64545", "medium": "#E0A030", "low": "#3C9D6B"}
ROUTE_COLORS = ["#1F5FAD", "#C2410C", "#0F766E", "#7C3AED", "#B45309", "#BE185D", "#4D7C0F", "#0369A1"]
VOLUME_SCALE = 100  # OR-Tools needs integers: work in hundredths of a m³

# ---------------------------------------------------------------------------
# Database setup. The path no longer depends on the folder you launch from:
# it is next to this script unless LOGISTIGO_DB points somewhere else.
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get(
    "LOGISTIGO_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
)


@st.cache_resource
def get_connection(path):
    # One shared connection per database file. Every query below is read to the
    # end (fetchall) so no half-finished SELECT keeps the file locked.
    return sqlite3.connect(path, check_same_thread=False, timeout=10)


conn = get_connection(DB_PATH)
conn.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)''')
conn.execute('''CREATE TABLE IF NOT EXISTS heatmap_points (lat REAL, lon REAL, username TEXT)''')
conn.execute('''CREATE TABLE IF NOT EXISTS delivery_summary (
    date TEXT,
    fuel_consumed REAL,
    total_deliveries INTEGER,
    vehicles_used INTEGER,
    priority_high INTEGER,
    priority_medium INTEGER,
    priority_low INTEGER,
    total_distance REAL,
    username TEXT
)''')  # total_distance is stored in metres (kept for compatibility with old rows)
# Databases created before history was per-user lack the username column; add it.
# Rows saved before then have no owner and are not shown to anyone until assigned.
for _table in ("heatmap_points", "delivery_summary"):
    if "username" not in [col[1] for col in conn.execute(f"PRAGMA table_info({_table})").fetchall()]:
        conn.execute(f"ALTER TABLE {_table} ADD COLUMN username TEXT")
conn.commit()


# ---------------------------------------------------------------------------
# Passwords: salted PBKDF2. Old unsalted SHA-256 hashes still log in and are
# upgraded to the new format on the first successful login.
# ---------------------------------------------------------------------------
PBKDF2_ITERATIONS = 200_000


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS).hex()
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password, stored):
    if stored.startswith("pbkdf2$"):
        _, iterations, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations)).hex()
        return hmac.compare_digest(candidate, digest)
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(legacy, stored)


def user_exists(username):
    return bool(conn.execute('SELECT 1 FROM users WHERE username = ?', (username,)).fetchall())


def signup_user(username, password):
    """Returns an error message, or None on success."""
    username = username.strip()
    if not username or not password:
        return "Username and password are required."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if user_exists(username):
        return "That username is already taken."
    conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hash_password(password)))
    conn.commit()
    return None


def login_user(username, password):
    username = username.strip()
    for (stored,) in conn.execute('SELECT password FROM users WHERE username = ?', (username,)).fetchall():
        if verify_password(password, stored):
            if not stored.startswith("pbkdf2$"):
                conn.execute('UPDATE users SET password = ? WHERE username = ? AND password = ?',
                          (hash_password(password), username, stored))
                conn.commit()
            return True
    return False


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _geocode_cached(location_name):
    # Network errors raise, so they are not cached; "not found" is cached.
    response = requests.get(
        NOMINATIM_URL,
        params={"format": "json", "q": location_name, "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json()
    if results:
        top = results[0]
        return float(top["lat"]), float(top["lon"]), top.get("display_name", location_name)
    return None, None, None


def geocode_location(location_name):
    """Returns (lat, lon, matched_address); all None if the place wasn't found."""
    location_name = location_name.strip()
    if not location_name:
        return None, None, None
    try:
        return _geocode_cached(location_name)
    except Exception as e:
        st.error(f"Geocoding error for '{location_name}': {e}")
        return None, None, None


def get_osm_route(locations):
    """Road route through `locations` in order. Returns (geometry, distance_m, duration_s)."""
    coord_string = ";".join(f"{lon},{lat}" for lat, lon in locations)
    url = f"{OSRM_URL}/route/v1/driving/{coord_string}"
    try:
        response = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=HTTP_TIMEOUT)
        data = response.json()
        if response.status_code == 200 and data.get("routes"):
            route = data["routes"][0]
            # Whole route, all legs (previously only the first leg was counted).
            return route["geometry"]["coordinates"], route["distance"], route["duration"]
        st.error(f"OSRM could not build a route: {data.get('message', response.status_code)}")
    except Exception as e:
        st.error(f"Error fetching route from OSRM: {e}")
    return [], 0.0, 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def _osrm_distance_matrix(points):
    coord_string = ";".join(f"{lon},{lat}" for lat, lon in points)
    response = requests.get(f"{OSRM_URL}/table/v1/driving/{coord_string}",
                            params={"annotations": "distance"}, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()["distances"]


def distance_matrix(points):
    """Road distances in metres between all points; straight-line distance where OSRM has no answer."""
    points = tuple(points)
    try:
        road = _osrm_distance_matrix(points)
    except Exception:
        st.warning("Road distance matrix unavailable; planning with straight-line distances.")
        road = None
    matrix = []
    for i, a in enumerate(points):
        row = []
        for j, b in enumerate(points):
            d = road[i][j] if road and road[i][j] is not None else geodesic(a, b).meters
            row.append(d)
        matrix.append(row)
    return matrix


# ---------------------------------------------------------------------------
# Route planning
# ---------------------------------------------------------------------------
def plan_routes(dist, volumes, priorities, capacities, time_limit_s=3):
    """Capacitated vehicle routing from depot (node 0) to stops 1..n, open routes
    (vehicles don't return to the depot). Minimises road distance, nudges
    high/medium priority stops earlier in their route, and drops stops only when
    they cannot fit (low priority dropped first).
    Returns (routes, dropped) as lists of stop indices (0-based, into volumes)."""
    n_stops = len(volumes)
    demands = [0] + [int(round(v * VOLUME_SCALE)) for v in volumes]
    caps = [int(round(cap * VOLUME_SCALE)) for cap in capacities]

    if not HAVE_ORTOOLS:
        return _plan_routes_greedy(dist, demands, priorities, caps)

    manager = pywrapcp.RoutingIndexManager(n_stops + 1, len(caps), 0)
    routing = pywrapcp.RoutingModel(manager)

    def cost_cb(from_index, to_index):
        a, b = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return 0 if b == 0 else int(dist[a][b])  # returning to depot is free -> open routes

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(cost_cb))

    demand_cb = routing.RegisterUnaryTransitCallback(lambda i: demands[manager.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, caps, True, "Capacity")

    # Position of each stop within its route (0 = first stop).
    order_cb = routing.RegisterUnaryTransitCallback(lambda i: 0 if manager.IndexToNode(i) == 0 else 1)
    routing.AddDimension(order_cb, 0, n_stops + 1, True, "Order")
    order_dim = routing.GetDimensionOrDie("Order")
    lateness_cost = {"high": 3000, "medium": 1000, "low": 0}  # metres of detour per position of delay
    drop_penalty = {"high": 3 * 10**9, "medium": 2 * 10**9, "low": 10**9}
    for stop in range(n_stops):
        index = manager.NodeToIndex(stop + 1)
        if lateness_cost[priorities[stop]]:
            order_dim.SetCumulVarSoftUpperBound(index, 0, lateness_cost[priorities[stop]])
        routing.AddDisjunction([index], drop_penalty[priorities[stop]])

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_s)
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return _plan_routes_greedy(dist, demands, priorities, caps)

    routes, served = [], set()
    for v in range(len(caps)):
        index, route = routing.Start(v), []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(node - 1)
                served.add(node - 1)
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    dropped = [s for s in range(n_stops) if s not in served]
    return routes, dropped


def _plan_routes_greedy(dist, demands, priorities, caps):
    """Fallback: priority order, each stop to the vehicle whose last stop is nearest and has room."""
    routes = [[] for _ in caps]
    loads = [0] * len(caps)
    dropped = []
    for stop in sorted(range(len(priorities)), key=lambda s: PRIORITY_RANK[priorities[s]]):
        need = demands[stop + 1]
        fits = [v for v in range(len(caps)) if loads[v] + need <= caps[v]]
        if not fits:
            dropped.append(stop)
            continue
        best = min(fits, key=lambda v: dist[routes[v][-1] + 1 if routes[v] else 0][stop + 1])
        routes[best].append(stop)
        loads[best] += need
    return routes, dropped


def baseline_routes(served, volumes, capacities):
    """Unoptimised plan for comparison: the same stops, in the order they were
    entered, loaded onto vehicle 1 until full, then vehicle 2, and so on."""
    routes = [[] for _ in capacities]
    loads = [0.0] * len(capacities)
    v = 0
    for stop in sorted(served):
        while v < len(capacities) and loads[v] + volumes[stop] > capacities[v]:
            v += 1
        if v == len(capacities):
            break
        routes[v].append(stop)
        loads[v] += volumes[stop]
    return routes


def add_routes_to_map(m, start, stops, routes, names, vehicle_names, layer_name, dashed=False, show=True):
    """Fetch road routes and draw them as one toggleable map layer.
    Returns (per-vehicle info, total_m)."""
    layer = folium.FeatureGroup(name=layer_name, show=show)
    info, total = [], 0.0
    for v, route in enumerate(routes):
        if not route:
            continue
        geometry, distance, duration = get_osm_route([start] + [stops[s] for s in route])
        if geometry:
            folium.PolyLine(
                [(lat, lon) for lon, lat in geometry],
                color="#7A8494" if dashed else ROUTE_COLORS[v % len(ROUTE_COLORS)],
                weight=3 if dashed else 5, opacity=0.9, dash_array="6 8" if dashed else None,
                tooltip=f"{vehicle_names[v]} ({layer_name.lower()})",
            ).add_to(layer)
        total += distance
        info.append({"vehicle": v, "stops": route, "distance_m": distance, "duration_s": duration})
    layer.add_to(m)
    return info, total


def base_map(location, zoom):
    """OpenStreetMap tiles (no API key), kept out of the layer menu."""
    m = folium.Map(location=location, zoom_start=zoom, tiles=None)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m)
    return m


def stop_icon(label, fill, outline, size=26):
    """Round numbered badge: fill = vehicle colour, outline = priority colour."""
    return folium.DivIcon(
        icon_size=(size, size), icon_anchor=(size // 2, size // 2),
        html=(f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{fill};'
              f'border:3px solid {outline};box-sizing:border-box;color:#fff;font:600 12px/20px '
              f'system-ui,sans-serif;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.35)">{label}</div>'),
    )


def build_map(start, stops, names, priorities, routes, vehicle_names):
    """Depot plus every stop, numbered in the order its vehicle visits it.
    Stops that couldn't be served are shown greyed out without a number."""
    m = base_map(list(start), 12)
    folium.Marker(list(start), tooltip="Depot", icon=folium.Icon(color="black", icon="home")).add_to(m)
    stop_layer = folium.FeatureGroup(name="Stops")
    position = {}
    for v, route in enumerate(routes):
        for n, s in enumerate(route, start=1):
            position[s] = (v, n)
    for s, point in enumerate(stops):
        priority = f"{priorities[s].capitalize()} priority"
        if s in position:
            v, n = position[s]
            icon = stop_icon(n, ROUTE_COLORS[v % len(ROUTE_COLORS)], PRIORITY_COLOR[priorities[s]])
            tooltip = f"{vehicle_names[v]} · Stop {n} · {names[s]} · {priority}"
        else:
            icon = stop_icon("–", "#9AA3AF", PRIORITY_COLOR[priorities[s]])
            tooltip = f"Not delivered · {names[s]} · {priority}"
        folium.Marker(point, icon=icon, tooltip=tooltip).add_to(stop_layer)
    stop_layer.add_to(m)
    m.fit_bounds([list(start)] + [list(p) for p in stops])
    return m


def format_duration(seconds):
    minutes = int(round(seconds / 60))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def save_run(result, username):
    """Record a plan in this user's analytics and heatmap (once per distinct plan)."""
    key = (username, result["signature"])
    if key in st.session_state.saved_runs or not result["served"]:
        return
    for lat, lon in result["served_points"]:
        conn.execute('INSERT INTO heatmap_points (lat, lon, username) VALUES (?, ?, ?)', (lat, lon, username))
    pr = result["served_priorities"]
    values = (result["fuel_l"], len(result["served"]), len(result["vehicle_rows"]),
              pr.count("high"), pr.count("medium"), pr.count("low"), result["opt_total"])
    owner = (result["date"], username)
    if conn.execute("SELECT 1 FROM delivery_summary WHERE date = ? AND username = ?", owner).fetchall():
        conn.execute('''UPDATE delivery_summary SET
                    fuel_consumed = fuel_consumed + ?,
                    total_deliveries = total_deliveries + ?,
                    vehicles_used = vehicles_used + ?,
                    priority_high = priority_high + ?,
                    priority_medium = priority_medium + ?,
                    priority_low = priority_low + ?,
                    total_distance = total_distance + ?
                    WHERE date = ? AND username = ?''', values + owner)
    else:
        conn.execute('''INSERT INTO delivery_summary (date, fuel_consumed, total_deliveries, vehicles_used,
                    priority_high, priority_medium, priority_low, total_distance, username)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (result["date"],) + values + (username,))
    conn.commit()
    st.session_state.saved_runs.add(key)


# ---------------------------------------------------------------------------
# Driver hand-off
# ---------------------------------------------------------------------------
MAPS_MAX_WAYPOINTS = 9  # Google Maps URLs accept at most 9 stops between origin and destination


def google_maps_links(start, points):
    """Google Maps driving directions from the depot through `points` in order.
    Long routes are split into consecutive legs, each starting where the last ended."""
    path = [start] + list(points)
    fmt = lambda p: f"{p[0]:.6f},{p[1]:.6f}"
    links, i = [], 0
    while i < len(path) - 1:
        leg = path[i:i + MAPS_MAX_WAYPOINTS + 2]
        params = {"api": "1", "origin": fmt(leg[0]), "destination": fmt(leg[-1]), "travelmode": "driving"}
        if len(leg) > 2:
            params["waypoints"] = "|".join(fmt(p) for p in leg[1:-1])
        links.append("https://www.google.com/maps/dir/?" + urlencode(params))
        i += len(leg) - 1
    return links


def run_sheet_html(result):
    """Printable run sheet: one table per vehicle, with its Google Maps link(s)."""
    esc = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Run sheet</title><style>"
        "body{font:14px/1.45 system-ui,sans-serif;color:#1C2430;margin:32px}"
        "h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 6px}"
        "p.meta{color:#5B6574;margin:0 0 4px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #D5DAE1;padding:6px 8px;text-align:left;vertical-align:top}"
        "th{background:#F3F5F8}a{color:#1F5FAD}@media print{body{margin:12mm}h2{break-after:avoid}"
        "table{break-inside:auto}tr{break-inside:avoid}}</style></head><body>",
        f"<h1>Run sheet · {esc(result['date'])}</h1>",
        f"<p class='meta'>Depot: {esc(result['depot']['name'])} ({esc(result['depot']['matched'])})</p>",
    ]
    for vehicle in result["handoff"]:
        rows = [r for r in result["stop_rows"] if r["Vehicle"] == vehicle["vehicle"]]
        parts.append(f"<h2>{esc(vehicle['vehicle'])} · {len(rows)} stops · {esc(vehicle['summary'])}</h2>")
        for n, link in enumerate(vehicle["links"], start=1):
            label = "Open in Google Maps" + (f" (part {n} of {len(vehicle['links'])})" if len(vehicle["links"]) > 1 else "")
            parts.append(f"<p><a href='{esc(link)}'>{label}</a></p>")
        parts.append("<table><tr><th>#</th><th>Location</th><th>Matched address</th><th>Priority</th>"
                     "<th>Volume (m³)</th><th>Done</th></tr>")
        for r in rows:
            parts.append(f"<tr><td>{r['Stop']}</td><td>{esc(r['Location'])}</td><td>{esc(r['Matched address'])}</td>"
                         f"<td>{esc(r['Priority'])}</td><td>{r['Volume (m³)']:.2f}</td><td>☐</td></tr>")
        parts.append("</table>")
    if result["dropped"]:
        parts.append("<h2>Not delivered (no vehicle had room)</h2><p>" + esc(", ".join(result["dropped"])) + "</p>")
    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
DEFAULT_VEHICLES = pd.DataFrame({"Vehicle": ["Van 1", "Van 2", "Van 3"], "Capacity (m³)": [10.0, 10.0, 10.0]})
DEFAULT_STOPS = pd.DataFrame({"Location": [""] * 3, "Priority": ["Medium"] * 3, "Volume (m³)": [1.0] * 3})


def page_login():
    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        st.title("Logistigo")
        st.caption("Delivery route planning")
        sign_in, create = st.tabs(["Sign in", "Create account"])
        with sign_in:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    if login_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username.strip()
                        st.rerun()
                    st.error("Incorrect username or password.")
        with create:
            with st.form("signup_form"):
                new_user = st.text_input("Username", key="signup_user")
                new_password = st.text_input("Password", type="password", key="signup_password",
                                             help="At least 6 characters.")
                if st.form_submit_button("Create account", use_container_width=True):
                    error = signup_user(new_user, new_password)
                    if error:
                        st.error(error)
                    else:
                        st.success("Account created. You can sign in now.")


def page_plan():
    st.header("Plan routes")
    st.caption("Enter a depot, your vehicles and the stops for the day. Routes are planned when you submit.")

    with st.form("plan_form"):
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.subheader("Trip")
            start_name = st.text_input("Depot / start location", placeholder="e.g. Chennai Central",
                                       key="start_location")
            delivery_date = st.date_input("Delivery date", datetime.today(), key="delivery_date")
            km_per_litre = st.number_input("Fuel efficiency (km/L)", min_value=0.1, value=8.0, step=0.5,
                                           key="km_per_litre")
            st.subheader("Fleet")
            vehicles = st.data_editor(
                st.session_state.get("vehicles_input", DEFAULT_VEHICLES), key="vehicles_editor",
                num_rows="dynamic", hide_index=True, use_container_width=True,
                column_config={
                    "Vehicle": st.column_config.TextColumn(required=True),
                    "Capacity (m³)": st.column_config.NumberColumn(min_value=0.1, step=0.5, format="%.1f",
                                                                    required=True),
                },
            )
        with right:
            st.subheader("Stops")
            stops_df = st.data_editor(
                st.session_state.get("stops_input", DEFAULT_STOPS), key="stops_editor",
                num_rows="dynamic", hide_index=True, use_container_width=True,
                column_config={
                    "Location": st.column_config.TextColumn(width="medium"),
                    "Priority": st.column_config.SelectboxColumn(options=["High", "Medium", "Low"], required=True),
                    "Volume (m³)": st.column_config.NumberColumn(min_value=0.01, step=0.5, format="%.2f",
                                                                  required=True),
                },
            )
            st.caption("Add a stop with + in the table's toolbar, or type in the empty last row. "
                       "Rows without a location are ignored.")
        submitted = st.form_submit_button("Plan routes", type="primary")

    if submitted:
        run_planner(start_name, delivery_date, km_per_litre, vehicles, stops_df)

    result = st.session_state.get("result")
    if result:
        show_result(result)


def run_planner(start_name, delivery_date, km_per_litre, vehicles, stops_df):
    vehicles = vehicles.dropna()
    vehicles = vehicles[vehicles["Vehicle"].astype(str).str.strip() != ""]
    vehicle_names = vehicles["Vehicle"].astype(str).str.strip().tolist()
    capacities = vehicles["Capacity (m³)"].astype(float).tolist()

    problems = []
    if not start_name.strip():
        problems.append("Enter a depot / start location.")
    if not capacities:
        problems.append("Add at least one vehicle.")
    rows = stops_df.dropna(subset=["Location"])
    rows = rows[rows["Location"].astype(str).str.strip() != ""]
    if rows.empty:
        problems.append("Add at least one stop.")
    if problems:
        for p in problems:
            st.error(p)
        return

    with st.spinner("Looking up locations…"):
        start_lat, start_lon, start_matched = geocode_location(start_name)
        start = (start_lat, start_lon)
        names, stops, matched, priorities, volumes, not_found = [], [], [], [], [], []
        for _, row in rows.iterrows():
            name = str(row["Location"]).strip()
            lat, lon, address = geocode_location(name)
            if lat is None:
                not_found.append(name)
                continue
            names.append(name)
            stops.append((lat, lon))
            matched.append(address)
            priorities.append(str(row.get("Priority") or "Medium").lower())
            volumes.append(float(row.get("Volume (m³)") or 1.0))

    if start_lat is None:
        st.error(f"Couldn't find the depot “{start_name.strip()}”. Try a more specific name.")
        return
    if not_found:
        st.warning("Couldn't find: " + ", ".join(not_found) + ". These stops were skipped.")
    if not stops:
        return
    too_big = [n for n, v in zip(names, volumes) if v > max(capacities)]
    if too_big:
        st.warning("Larger than any vehicle: " + ", ".join(too_big) + ".")

    with st.spinner("Planning routes…"):
        dist = distance_matrix([start] + stops)
        routes, dropped = plan_routes(dist, volumes, priorities, capacities)
        m = build_map(start, stops, names, priorities, routes, vehicle_names)
        opt_info, opt_total = add_routes_to_map(m, start, stops, routes, names, vehicle_names, "Optimised")
        served = [s for route in routes for s in route]
        base = baseline_routes(served, volumes, capacities)
        _, base_total = add_routes_to_map(m, start, stops, base, names, vehicle_names, "Entered order",
                                          dashed=True, show=False)
        folium.LayerControl(collapsed=False).add_to(m)

    vehicle_rows = [{
        "Vehicle": vehicle_names[i["vehicle"]],
        "Route": " → ".join(names[s] for s in i["stops"]),
        "Stops": len(i["stops"]),
        "Load (m³)": f"{sum(volumes[s] for s in i['stops']):.1f} / {capacities[i['vehicle']]:.1f}",
        "Distance (km)": round(i["distance_m"] / 1000, 1),
        "Drive time": format_duration(i["duration_s"]),
    } for i in opt_info]

    # Every stop with the address it was matched to, in visit order; unserved stops last.
    stop_rows = []
    for i in opt_info:
        for n, s in enumerate(i["stops"], start=1):
            stop_rows.append({"Vehicle": vehicle_names[i["vehicle"]], "Stop": n, "Location": names[s],
                              "Matched address": matched[s], "Priority": priorities[s].capitalize(),
                              "Volume (m³)": volumes[s], "Latitude": stops[s][0], "Longitude": stops[s][1]})
    for s in dropped:
        stop_rows.append({"Vehicle": "Not delivered", "Stop": None, "Location": names[s],
                          "Matched address": matched[s], "Priority": priorities[s].capitalize(),
                          "Volume (m³)": volumes[s], "Latitude": stops[s][0], "Longitude": stops[s][1]})

    handoff = [{
        "vehicle": vehicle_names[i["vehicle"]],
        "summary": f"{i['distance_m'] / 1000:.1f} km, {format_duration(i['duration_s'])}",
        "links": google_maps_links(start, [stops[s] for s in i["stops"]]),
    } for i in opt_info]

    result = {
        "date": delivery_date.strftime("%Y-%m-%d"),
        "signature": (delivery_date.isoformat(), start, tuple(stops), tuple(priorities), tuple(volumes),
                      tuple(capacities)),
        "map": m, "vehicle_rows": vehicle_rows, "stop_rows": stop_rows, "handoff": handoff,
        "depot": {"name": start_name.strip(), "matched": start_matched},
        "opt_total": opt_total, "base_total": base_total,
        "fuel_l": opt_total / 1000 / km_per_litre,
        "served": served, "total_stops": len(stops),
        "dropped": [names[s] for s in dropped],
        "served_points": [stops[s] for s in served],
        "served_priorities": [priorities[s] for s in served],
    }
    result["run_sheet_csv"] = pd.DataFrame(stop_rows).to_csv(index=False).encode("utf-8")
    result["run_sheet_html"] = run_sheet_html(result)
    st.session_state.result = result
    save_run(result, st.session_state.username)


def show_result(result):
    st.divider()
    st.subheader("Plan")
    opt_km, base_km = result["opt_total"] / 1000, result["base_total"] / 1000
    saved_pct = (base_km - opt_km) / base_km * 100 if base_km > 0 else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total distance", f"{opt_km:.1f} km")
    c2.metric(f"Saved vs entered order ({max(saved_pct, 0):.0f}%)", f"{max(base_km - opt_km, 0):.1f} km")
    c3.metric("Estimated fuel", f"{result['fuel_l']:.1f} L")
    c4.metric("Stops served", f"{len(result['served'])} / {result['total_stops']}")

    if result["dropped"]:
        st.warning("Not enough vehicle capacity for: " + ", ".join(result["dropped"])
                   + ". Lower-priority stops were left out first.")

    vehicles_tab, stops_tab = st.tabs(["Vehicles", "Stops and matched addresses"])
    with vehicles_tab:
        st.dataframe(pd.DataFrame(result["vehicle_rows"]), hide_index=True, use_container_width=True)
    with stops_tab:
        st.caption(f"Depot “{result['depot']['name']}” matched to: {result['depot']['matched']}. "
                   "Check each address below; if one is wrong, make that location more specific "
                   "(add the area, city or postcode) and plan again.")
        st.dataframe(pd.DataFrame(result["stop_rows"]).drop(columns=["Latitude", "Longitude"]),
                     hide_index=True, use_container_width=True,
                     column_config={"Stop": st.column_config.NumberColumn(format="%d"),
                                    "Volume (m³)": st.column_config.NumberColumn(format="%.2f")})

    st.caption("Map: numbers show the order each vehicle visits its stops, in that vehicle's route colour; "
               "the outline shows priority (red high, amber medium, green low). Turn on “Entered order” "
               "in the map’s layer menu to compare with visiting stops in the order you typed them.")
    st_folium(result["map"], key="plan_map", height=520, use_container_width=True, returned_objects=[])

    st.subheader("Driver hand-off")
    st.caption("Send each driver their route. Google Maps opens with the stops in order; routes with more "
               f"than {MAPS_MAX_WAYPOINTS + 1} stops are split into parts, each starting where the last ended.")
    for vehicle in result["handoff"]:
        name_col, link_col = st.columns([1, 2], vertical_alignment="center")
        name_col.markdown(f"**{vehicle['vehicle']}**  \n{vehicle['summary']}")
        with link_col:
            buttons = st.columns(len(vehicle["links"]))
            for n, (col, link) in enumerate(zip(buttons, vehicle["links"]), start=1):
                label = "Open in Google Maps" if len(vehicle["links"]) == 1 else f"Part {n}"
                col.link_button(label, link, use_container_width=True)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Run sheet (CSV)", result["run_sheet_csv"], file_name=f"run-sheet-{result['date']}.csv",
                       mime="text/csv", use_container_width=True)
    d2.download_button("Printable run sheet", result["run_sheet_html"].encode("utf-8"),
                       file_name=f"run-sheet-{result['date']}.html", mime="text/html",
                       use_container_width=True, help="Open in a browser and print (Ctrl+P).")


def integer_ticks(max_value, target=6):
    """Whole-number axis ticks (avoids labels like 0, 0.5, 1 rounding to 0, 1, 1)."""
    step = max(1, -(-int(max_value) // target))
    return list(range(0, int(max_value) + step + 1, step))


def page_analytics():
    st.header("Analytics")
    df = pd.read_sql_query("SELECT * FROM delivery_summary WHERE username = ?", conn,
                           params=(st.session_state.username,))
    if df.empty:
        st.info("No delivery data yet. Plans you create are recorded here.")
        return
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date").sum(numeric_only=True).sort_index()
    daily["Distance (km)"] = daily["total_distance"] / 1000

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deliveries", f"{int(daily['total_deliveries'].sum()):,}")
    c2.metric("Distance", f"{daily['Distance (km)'].sum():,.1f} km")
    c3.metric("Fuel", f"{daily['fuel_consumed'].sum():,.1f} L")
    c4.metric("Days recorded", f"{len(daily)}")

    chart_data = daily.reset_index()
    chart_data["Date"] = chart_data["date"].dt.strftime("%d %b")
    bar_width = max(6, min(36, 420 // len(chart_data)))

    def daily_chart(column, label, whole_numbers=False):
        axis = alt.Axis(format="d", values=integer_ticks(chart_data[column].max())) if whole_numbers else alt.Axis()
        return alt.Chart(chart_data).mark_bar(size=bar_width, color=PRIMARY_COLOR).encode(
            x=alt.X("Date:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{column}:Q", title=None, axis=axis),
            tooltip=[alt.Tooltip("Date:N"), alt.Tooltip(f"{column}:Q", title=label, format=",.1f")],
        ).properties(height=260)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Deliveries per day**")
        st.altair_chart(daily_chart("total_deliveries", "Deliveries", whole_numbers=True), use_container_width=True)
        st.markdown("**Fuel per day (L)**")
        st.altair_chart(daily_chart("fuel_consumed", "Fuel (L)"), use_container_width=True)
    with right:
        st.markdown("**Distance per day (km)**")
        st.altair_chart(daily_chart("Distance (km)", "Distance (km)"), use_container_width=True)
        st.markdown("**Priority mix**")
        mix = pd.DataFrame({"Priority": ["High", "Medium", "Low"],
                            "Deliveries": [int(daily["priority_high"].sum()), int(daily["priority_medium"].sum()),
                                           int(daily["priority_low"].sum())]})
        order = ["High", "Medium", "Low"]
        chart = alt.Chart(mix).mark_bar().encode(
            x=alt.X("Deliveries:Q", axis=alt.Axis(format="d", values=integer_ticks(mix["Deliveries"].max())),
                    title=None),
            y=alt.Y("Priority:N", sort=order, title=None),
            color=alt.Color("Priority:N", scale=alt.Scale(domain=order, range=[PRIORITY_COLOR[p.lower()]
                                                                                for p in order]), legend=None),
            tooltip=["Priority", "Deliveries"],
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)

    with st.expander("Daily records"):
        table = daily.reset_index()[["date", "total_deliveries", "vehicles_used", "Distance (km)", "fuel_consumed",
                                     "priority_high", "priority_medium", "priority_low"]]
        table.columns = ["Date", "Deliveries", "Vehicles", "Distance (km)", "Fuel (L)", "High", "Medium", "Low"]
        table["Date"] = table["Date"].dt.date
        st.dataframe(table.round(1), hide_index=True, use_container_width=True)


def page_heatmap():
    st.header("Delivery heatmap")
    points = conn.execute('SELECT lat, lon FROM heatmap_points WHERE username = ?',
                          (st.session_state.username,)).fetchall()
    st.caption(f"{len(points)} of your delivered stops recorded.")
    m = base_map([20, 78], 4)
    if points:
        # max_zoom=1: leaflet.heat divides each point's weight by 2 for every zoom level below
        # max_zoom (folium's default is 18), so at city zoom a few stops were nearly invisible.
        HeatMap(points, radius=22, blur=16, min_opacity=0.45, max_zoom=1, control=False).add_to(m)
        stops_layer = folium.FeatureGroup(name="Delivery points", show=False)
        for lat, lon in points:
            folium.CircleMarker([lat, lon], radius=3, weight=1, color="#1C2430", fill=True,
                                fill_opacity=0.8).add_to(stops_layer)
        stops_layer.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        m.fit_bounds(points, max_zoom=13)  # a single point would otherwise zoom to street level
    st_folium(m, key="heatmap", height=560, use_container_width=True, returned_objects=[])
    if points:
        with st.expander("Manage data"):
            st.write("Remove all of your recorded points from the heatmap. Analytics and other users are not affected.")
            confirm = st.checkbox("I understand this can't be undone", key="confirm_clear")
            if st.button("Clear heatmap", disabled=not confirm):
                conn.execute('DELETE FROM heatmap_points WHERE username = ?', (st.session_state.username,))
                conn.commit()
                st.session_state.pop("confirm_clear", None)
                st.rerun()


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("saved_runs", set())

if not st.session_state.authenticated:
    page_login()
else:
    with st.sidebar:
        st.markdown("### Logistigo")
        page = st.radio("Navigate", ["Plan routes", "Analytics", "Heatmap"], key="page",
                        label_visibility="collapsed")
        st.divider()
        st.caption(f"Signed in as **{st.session_state.username}**")
        if st.button("Sign out", use_container_width=True):
            for key in ("authenticated", "username", "result"):
                st.session_state.pop(key, None)
            st.rerun()

    {"Plan routes": page_plan, "Analytics": page_analytics, "Heatmap": page_heatmap}[page]()
