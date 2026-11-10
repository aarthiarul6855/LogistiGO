import os
import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime

import folium
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from folium.plugins import HeatMap
from geopy.distance import geodesic
from streamlit_folium import st_folium

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
PRIORITY_COLOR = {"high": "red", "medium": "orange", "low": "green"}
ROUTE_COLORS = ["blue", "red", "green", "purple", "orange", "darkblue", "cadetblue", "darkgreen"]
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
conn.execute('''CREATE TABLE IF NOT EXISTS heatmap_points (lat REAL, lon REAL)''')
conn.execute('''CREATE TABLE IF NOT EXISTS delivery_summary (
    date TEXT,
    fuel_consumed REAL,
    total_deliveries INTEGER,
    vehicles_used INTEGER,
    priority_high INTEGER,
    priority_medium INTEGER,
    priority_low INTEGER,
    total_distance REAL
)''')  # total_distance is stored in metres (kept for compatibility with old rows)
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
        return float(results[0]["lat"]), float(results[0]["lon"])
    return None, None


def geocode_location(location_name):
    location_name = location_name.strip()
    if not location_name:
        return None, None
    try:
        return _geocode_cached(location_name)
    except Exception as e:
        st.error(f"Geocoding error for '{location_name}': {e}")
        return None, None


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


def draw_routes(start, stops, priorities, routes, names, label):
    """Fetch road routes and draw them. Returns (map, per-vehicle info, total_m)."""
    m = folium.Map(location=list(start), zoom_start=12)
    folium.Marker(list(start), popup="Start Point", icon=folium.Icon(color="black")).add_to(m)
    info, total = [], 0.0
    for v, route in enumerate(routes):
        if not route:
            continue
        geometry, distance, duration = get_osm_route([start] + [stops[s] for s in route])
        color = ROUTE_COLORS[v % len(ROUTE_COLORS)]
        if geometry:
            folium.PolyLine([(lat, lon) for lon, lat in geometry], color=color, weight=5,
                            opacity=0.8, tooltip=f"{label} – Vehicle {v + 1}").add_to(m)
        for position, s in enumerate(route, start=1):
            folium.Marker(stops[s], tooltip=f"Vehicle {v + 1}, stop {position}: {names[s]}",
                          popup=f"{names[s]} – priority {priorities[s].capitalize()}",
                          icon=folium.Icon(color=PRIORITY_COLOR[priorities[s]], icon="info-sign")).add_to(m)
        total += distance
        info.append({"vehicle": v + 1, "stops": [names[s] for s in route],
                     "distance_m": distance, "duration_s": duration})
    m.fit_bounds([list(start)] + [list(p) for p in stops])
    return m, info, total


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #ffe6f2; }
        .stButton>button { background-color: #ff66b2; color: white; border-radius: 10px; width: 100%; font-size: 18px; font-weight: bold; }
        .stTextInput>div>div>input { border: 2px solid #ff66b2; border-radius: 10px; padding: 10px; }
        h1, h2, h3 { color: #ff66b2; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💖 LOGISTIGO")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'saved_runs' not in st.session_state:
    st.session_state.saved_runs = set()

if not st.session_state.authenticated:
    menu = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])
    if menu == "Sign Up":
        st.subheader("Create an Account")
        new_user = st.text_input("Username", key="signup_user")
        new_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            error = signup_user(new_user, new_password)
            if error:
                st.error(error)
            else:
                st.success("Account created successfully! Please login. 💖")

    if menu == "Login":
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username.strip()
                st.rerun()
            else:
                st.error("Invalid username or password 💔")
else:
    st.sidebar.write(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        for key in ("authenticated", "username", "result"):
            st.session_state.pop(key, None)
        st.rerun()

    st.subheader("🌸 Choose an Option:")
    choice = st.radio(
        "What would you like to do?",
        ["Heatmap", "Analytics", "Logistics"],
        index=None,
        key="option",
        help="Select one option to proceed."
    )

    if choice == "Heatmap":
        st.subheader("🔥 Delivery Heatmap")

        # A button acts once; the old checkbox wiped the data on every rerun while ticked.
        if st.button("Clear Heatmap Points"):
            conn.execute('DELETE FROM heatmap_points')
            conn.commit()
            st.success("Heatmap points cleared.")

        points = conn.execute('SELECT lat, lon FROM heatmap_points').fetchall()
        m = folium.Map(location=[20, 0], zoom_start=2)
        if points:
            HeatMap(points).add_to(m)
            m.fit_bounds(points)
        else:
            st.info("No delivery points yet. Plan some routes first.")
        st_folium(m, key="heatmap", height=500, use_container_width=True, returned_objects=[])

    if choice == "Analytics":
        st.subheader("📊 Delivery Analytics Dashboard")

        df = pd.read_sql_query("SELECT * FROM delivery_summary", conn)
        if df.empty:
            st.info("No delivery data found.")
        else:
            df["date"] = pd.to_datetime(df["date"])
            grouped = df.groupby("date").sum(numeric_only=True).reset_index().set_index("date")
            grouped["total_distance_km"] = grouped["total_distance"] / 1000

            st.write("### ⛽ Fuel Consumed per Day (litres)")
            st.line_chart(grouped["fuel_consumed"])

            st.write("### 📈 Deliveries Completed per Day")
            st.line_chart(grouped["total_deliveries"])

            st.write("### 🚚 Vehicles Used per Day")
            st.line_chart(grouped["vehicles_used"])

            st.write("### 🛣️ Total Distance per Day (km)")
            st.line_chart(grouped["total_distance_km"])

            st.write("### ⚠️ Overall Priority Distribution")
            priority_totals = df[['priority_high', 'priority_medium', 'priority_low']].sum()
            fig, ax = plt.subplots()
            priority_totals.plot(kind='bar', color=['red', 'orange', 'green'], ax=ax)
            ax.set_title("Overall Priority Distribution")
            ax.set_ylabel("Number of Deliveries")
            ax.set_xlabel("Priority Level")
            ax.set_xticklabels(['High', 'Medium', 'Low'], rotation=0)
            st.pyplot(fig)

    if choice == "Logistics":
        st.subheader("Route Optimization")
        selected_date = st.date_input("Delivery Date", datetime.today())

        num_vehicles = int(st.number_input("Number of Vehicles", min_value=1, value=3))
        vehicle_capacities = []
        cols = st.columns(min(num_vehicles, 4))
        for i in range(num_vehicles):
            with cols[i % len(cols)]:
                vehicle_capacities.append(float(st.number_input(
                    f"Capacity of Vehicle {i + 1} (m³)", min_value=0.1, value=10.0, key=f"cap_{i}")))
        km_per_litre = st.number_input("Fuel efficiency (km per litre)", min_value=0.1, value=8.0)

        num_locations = int(st.number_input("Number of Delivery Locations", min_value=1, value=5))
        start_location_name = st.text_input("Start Location (e.g., Chennai Central)")

        start = None
        if start_location_name.strip():
            lat, lon = geocode_location(start_location_name)
            if lat is None:
                st.warning(f"Start location '{start_location_name}' could not be found.")
            else:
                start = (lat, lon)

        warnings = []
        names, stops, priorities, volumes = [], [], [], []
        for i in range(num_locations):
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                loc_name = st.text_input(f"Delivery Location {i + 1} Name", key=f"loc_name_{i}")
            with col2:
                priority = st.selectbox(f"Priority {i + 1}", ["low", "medium", "high"], key=f"priority_{i}")
            with col3:
                volume = float(st.number_input(f"Volume {i + 1} (m³)", min_value=0.01, value=1.0,
                                               key=f"volume_{i}"))
            if not loc_name.strip():
                continue  # blank rows are simply not used yet
            lat, lon = geocode_location(loc_name)
            if lat is None:
                warnings.append(f"❌ Could not find coordinates for '{loc_name.strip()}'")
                continue
            names.append(loc_name.strip())
            stops.append((lat, lon))
            priorities.append(priority)
            volumes.append(volume)

        total_volume, total_capacity = sum(volumes), sum(vehicle_capacities)
        st.caption(f"Total volume entered: {total_volume:.2f} m³ · total vehicle capacity: {total_capacity:.2f} m³")
        if total_volume > total_capacity:
            warnings.append("Total delivery volume exceeds total vehicle capacity; some stops will be left out "
                            "(low priority first).")
        for s, v in enumerate(volumes):
            if v > max(vehicle_capacities):
                warnings.append(f"'{names[s]}' ({v:.2f} m³) is larger than any vehicle and cannot be delivered.")
        for w in warnings:
            st.warning(w)

        ready = start is not None and len(stops) > 0
        if st.button("Find Optimal Routes 💖", disabled=not ready):
            with st.spinner("Planning routes..."):
                dist = distance_matrix([start] + stops)
                routes, dropped = plan_routes(dist, volumes, priorities, vehicle_capacities)
                opt_map, opt_info, opt_total = draw_routes(start, stops, priorities, routes, names, "Optimised")

                served = [s for route in routes for s in route]
                base = baseline_routes(served, volumes, vehicle_capacities)
                base_map, base_info, base_total = draw_routes(start, stops, priorities, base, names, "Unoptimised")

            st.session_state.result = {
                "date": selected_date.strftime("%Y-%m-%d"),
                "signature": (selected_date.isoformat(), start, tuple(stops), tuple(priorities),
                              tuple(volumes), tuple(vehicle_capacities)),
                "opt_map": opt_map, "opt_info": opt_info, "opt_total": opt_total,
                "base_map": base_map, "base_info": base_info, "base_total": base_total,
                "served": served, "dropped": [names[s] for s in dropped],
                "served_points": [stops[s] for s in served],
                "served_priorities": [priorities[s] for s in served],
                "km_per_litre": km_per_litre,
            }

        result = st.session_state.get("result")
        if result:
            st.write("## 🚛 Optimised Routes")
            for info in result["opt_info"]:
                st.write(f"**Vehicle {info['vehicle']}:** " + " → ".join(info["stops"]))
                st.write(f"Distance: {info['distance_m'] / 1000:.2f} km · Time: {info['duration_s'] / 60:.1f} mins")
            st_folium(result["opt_map"], key="opt_map", height=500, use_container_width=True, returned_objects=[])

            if result["dropped"]:
                st.error("Not delivered (no vehicle had room): " + ", ".join(result["dropped"]))

            st.write("## 🚛 Unoptimised Route (stops in the order entered)")
            st_folium(result["base_map"], key="base_map", height=500, use_container_width=True, returned_objects=[])

            opt_km, base_km = result["opt_total"] / 1000, result["base_total"] / 1000
            fuel = opt_km / result["km_per_litre"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Optimised distance", f"{opt_km:.2f} km")
            c2.metric("Unoptimised distance", f"{base_km:.2f} km")
            if base_km > 0:
                c3.metric("Saved", f"{base_km - opt_km:.2f} km", f"{(base_km - opt_km) / base_km * 100:.1f}%")
            st.write(f"Estimated fuel: {fuel:.2f} litres")

            # Record each distinct plan once, however many times the button is clicked.
            if result["signature"] not in st.session_state.saved_runs and result["served"]:
                for lat, lon in result["served_points"]:
                    conn.execute('INSERT INTO heatmap_points (lat, lon) VALUES (?, ?)', (lat, lon))
                served_priorities = result["served_priorities"]
                values = (fuel, len(result["served"]), len(result["opt_info"]),
                          served_priorities.count("high"), served_priorities.count("medium"),
                          served_priorities.count("low"), result["opt_total"])
                if conn.execute("SELECT 1 FROM delivery_summary WHERE date = ?", (result["date"],)).fetchall():
                    conn.execute('''UPDATE delivery_summary SET
                                fuel_consumed = fuel_consumed + ?,
                                total_deliveries = total_deliveries + ?,
                                vehicles_used = vehicles_used + ?,
                                priority_high = priority_high + ?,
                                priority_medium = priority_medium + ?,
                                priority_low = priority_low + ?,
                                total_distance = total_distance + ?
                                WHERE date = ?''', values + (result["date"],))
                else:
                    conn.execute('''INSERT INTO delivery_summary (date, fuel_consumed, total_deliveries, vehicles_used,
                                priority_high, priority_medium, priority_low, total_distance)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (result["date"],) + values)
                conn.commit()
                st.session_state.saved_runs.add(result["signature"])
                st.success("💖 Route optimization complete and saved to analytics!")
