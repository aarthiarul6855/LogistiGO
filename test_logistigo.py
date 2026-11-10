"""Offline tests for logistigo.py.

Run:  Scripts\\python.exe -m unittest -v test_logistigo

The Nominatim/OSRM servers are replaced with a fake that uses straight-line
distances, and each test uses its own temporary database.
"""
import hashlib
import io
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st
from geopy.distance import geodesic
from streamlit.testing.v1 import AppTest

from stop_import import MAX_STOPS, parse_stops_file, template_csv

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistigo.py")

PLACES = {
    "Chennai Central": (13.0827, 80.2707),
    "Guindy": (13.0067, 80.2206),
    "Tambaram": (12.9249, 80.1000),
    "Anna Nagar": (13.0850, 80.2101),
    "Adyar": (13.0012, 80.2565),
}
# A long line of stops for testing routes that need several Google Maps links.
PLACES.update({f"Point {i}": (13.0 + i * 0.01, 80.20 + i * 0.005) for i in range(1, 13)})


class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.status_code)


class FakeMapServers:
    """Stands in for requests.get: Nominatim search, OSRM route and OSRM table."""

    def __init__(self):
        self.geocode_calls = 0

    @staticmethod
    def _points(url):
        pairs = url.split("/driving/")[1].split(";")
        return [(float(lat), float(lon)) for lon, lat in (p.split(",") for p in pairs)]

    def __call__(self, url, params=None, **kwargs):
        assert "timeout" in kwargs, f"request without timeout: {url}"
        if "nominatim" in url:
            self.geocode_calls += 1
            place = PLACES.get(params["q"])
            return FakeResponse([{"lat": place[0], "lon": place[1],
                                  "display_name": f"{params['q']}, Chennai, Tamil Nadu, India"}] if place else [])
        points = self._points(url)
        if "/table/" in url:
            return FakeResponse({"distances": [[geodesic(a, b).m for b in points] for a in points]})
        legs = [geodesic(points[k], points[k + 1]).m for k in range(len(points) - 1)]
        return FakeResponse({"routes": [{
            "geometry": {"coordinates": [[lon, lat] for lat, lon in points]},
            "distance": sum(legs), "duration": sum(legs) / 10,
            "legs": [{"distance": leg} for leg in legs],
        }]})


def route_km(*names):
    pts = [PLACES[n] for n in names]
    return sum(geodesic(pts[k], pts[k + 1]).km for k in range(len(pts) - 1))


class LogistigoTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test.db")
        env = mock.patch.dict(os.environ, {"LOGISTIGO_DB": self.db_path, "LOGISTIGO_GEOCODE_INTERVAL": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.servers = FakeMapServers()
        net = mock.patch.object(requests, "get", self.servers)
        net.start()
        self.addCleanup(net.stop)
        st.cache_data.clear()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)  # db may still be open on Windows

    # -- helpers ---------------------------------------------------------
    def app(self, logged_in=False, page=None, user="tester"):
        at = AppTest.from_file(APP, default_timeout=60)
        if logged_in:
            at.session_state.authenticated = True
            at.session_state.username = user
        at.run()
        if page:
            at.radio(key="page").set_value(page).run()
        return at

    def db(self, sql, *args):
        con = sqlite3.connect(self.db_path)
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()

    @staticmethod
    def button(at, label):
        return next(b for b in at.button if b.label == label)

    @staticmethod
    def texts(elements):
        return [e.value for e in elements]

    def assertNoExceptions(self, at):
        self.assertEqual([e.value for e in at.exception], [])

    def plan(self, stops, capacities=(10.0,), start="Chennai Central", user="tester", strict=False):
        """stops: list of (name, priority, volume). Fills the planner and submits it."""
        at = AppTest.from_file(APP, default_timeout=60)
        at.session_state.authenticated = True
        at.session_state.username = user
        at.session_state.vehicles_input = pd.DataFrame(
            {"Vehicle": [f"Van {i + 1}" for i in range(len(capacities))], "Capacity (m³)": list(capacities)})
        at.session_state.stops_input = pd.DataFrame(
            {"Location": [n for n, _, _ in stops], "Priority": [p.capitalize() for _, p, _ in stops],
             "Volume (m³)": [v for _, _, v in stops]})
        at.run()
        at.text_input(key="start_location").set_value(start)
        at.toggle(key="strict_priority").set_value(strict)
        self.button(at, "Plan routes").click()
        return at.run()

    # -- accounts --------------------------------------------------------
    def test_signup_validation(self):
        at = self.app()

        def sign_up(user, password):
            at.text_input(key="signup_user").set_value(user)
            at.text_input(key="signup_password").set_value(password)
            self.button(at, "Create account").click().run()

        sign_up("", "secret1")
        self.assertIn("Username and password are required.", self.texts(at.error))
        sign_up("alice", "123")
        self.assertIn("Password must be at least 6 characters.", self.texts(at.error))
        sign_up("alice", "secret1")
        self.assertTrue(at.success)
        sign_up("alice", "another1")
        self.assertIn("That username is already taken.", self.texts(at.error))
        self.assertEqual(self.db("SELECT COUNT(*) FROM users WHERE username='alice'"), [(1,)])
        stored = self.db("SELECT password FROM users WHERE username='alice'")[0][0]
        self.assertTrue(stored.startswith("pbkdf2$"), "password must be salted, not plain SHA-256")

    def test_login_including_legacy_hash_upgrade(self):
        self.app()  # creates the tables
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO users VALUES ('old', ?)", (hashlib.sha256(b"oldpass").hexdigest(),))
        con.commit()
        con.close()

        for password, expected in [("wrong", False), ("oldpass", True)]:
            at = self.app()
            at.text_input(key="login_user").set_value("old")
            at.text_input(key="login_password").set_value(password)
            self.button(at, "Sign in").click().run()
            self.assertNoExceptions(at)
            self.assertEqual(at.session_state.authenticated, expected, password)
        stored = self.db("SELECT password FROM users WHERE username='old'")[0][0]
        self.assertTrue(stored.startswith("pbkdf2$"), "legacy hash should be upgraded after login")

    def test_sign_out(self):
        at = self.app(logged_in=True)
        self.button(at, "Sign out").click().run()
        self.assertFalse(at.session_state.authenticated)
        self.assertTrue(any(b.label == "Sign in" for b in at.button))

    # -- route planner ---------------------------------------------------
    def test_blank_planner_shows_no_errors_until_submitted(self):
        at = self.app(logged_in=True)
        self.assertNoExceptions(at)
        self.assertEqual(self.texts(at.error) + self.texts(at.warning), [])
        self.button(at, "Plan routes").click().run()
        self.assertEqual(self.texts(at.error), ["Enter a depot / start location.", "Add at least one stop."])
        self.assertEqual(self.servers.geocode_calls, 0)

    def test_unknown_place_is_reported(self):
        at = self.plan([("Atlantis", "low", 1.0), ("Guindy", "low", 1.0)])
        self.assertNoExceptions(at)
        self.assertIn("Couldn't find: Atlantis. These stops were skipped.", self.texts(at.warning))
        self.assertEqual(len(at.session_state.result["served"]), 1)

    def test_unknown_depot_is_reported(self):
        at = self.plan([("Guindy", "low", 1.0)], start="Nowhere")
        self.assertTrue(any("Couldn't find the depot" in e for e in self.texts(at.error)))

    def test_geocoding_only_on_submit_and_cached(self):
        at = self.plan([("Guindy", "low", 1.0), ("Adyar", "low", 1.0)])
        calls = self.servers.geocode_calls
        at.run()
        self.assertEqual(self.servers.geocode_calls, calls, "a plain rerun must not geocode")
        self.button(at, "Plan routes").click().run()
        self.assertEqual(self.servers.geocode_calls, calls, "repeat lookups must come from the cache")

    def test_distance_covers_whole_route(self):
        at = self.plan([("Guindy", "low", 1.0), ("Tambaram", "low", 1.0)])
        self.assertNoExceptions(at)
        metrics = {m.label: m.value for m in at.metric}
        shown_km = float(metrics["Total distance"].split()[0])
        self.assertAlmostEqual(shown_km, route_km("Chennai Central", "Guindy", "Tambaram"), delta=0.06)
        self.assertEqual(metrics["Stops served"], "2 / 2")

    def test_optimised_beats_entered_order(self):
        # Entered order zig-zags; the optimiser should not be worse.
        at = self.plan([("Tambaram", "low", 1.0), ("Anna Nagar", "low", 1.0), ("Guindy", "low", 1.0)])
        result = at.session_state.result
        self.assertLessEqual(result["opt_total"], result["base_total"])

    def test_capacity_by_volume_drops_low_priority_first(self):
        stops = [("Guindy", "low", 1.0), ("Tambaram", "medium", 1.0), ("Anna Nagar", "high", 1.0),
                 ("Adyar", "low", 1.0)]
        at = self.plan(stops, capacities=(2.5,))
        self.assertNoExceptions(at)
        self.assertTrue(any(w.startswith("Not enough vehicle capacity for: Guindy, Adyar.")
                            for w in self.texts(at.warning)))
        self.assertEqual({m.label: m.value for m in at.metric}["Stops served"], "2 / 4")
        row = self.db("SELECT total_deliveries, priority_high, priority_medium, priority_low FROM delivery_summary")
        self.assertEqual(row, [(2, 1, 1, 0)])

    def test_oversized_stop_is_flagged(self):
        at = self.plan([("Guindy", "high", 50.0)], capacities=(10.0,))
        self.assertIn("Larger than any vehicle: Guindy.", self.texts(at.warning))

    def test_multiple_vehicles_share_the_load(self):
        stops = [(n, "low", 4.0) for n in ("Guindy", "Tambaram", "Anna Nagar", "Adyar")]
        at = self.plan(stops, capacities=(10.0, 10.0))
        self.assertEqual(self.texts(at.warning), [])
        rows = at.session_state.result["vehicle_rows"]
        self.assertEqual(len(rows), 2)  # 16 m³ can't fit in one 10 m³ vehicle
        self.assertEqual(sum(r["Stops"] for r in rows), 4)
        # fleet + stops editors, plus the per-vehicle and per-stop results tables
        self.assertEqual(len(at.dataframe), 4, "results tables should be shown")

    def test_map_numbers_stops_in_visit_order(self):
        stops = [("Guindy", "high", 1.0), ("Tambaram", "medium", 1.0), ("Anna Nagar", "low", 1.0),
                 ("Adyar", "low", 1.0)]
        at = self.plan(stops, capacities=(2.0, 1.0))  # room for 3 of the 4 stops
        result = at.session_state.result
        stop_layer = next(c for c in result["map"]._children.values()
                          if getattr(c, "layer_name", None) == "Stops")
        tooltips = sorted(child.text for marker in stop_layer._children.values()
                          for child in marker._children.values() if hasattr(child, "text"))
        expected = []
        for row in result["vehicle_rows"]:
            for n, name in enumerate(row["Route"].split(" → "), start=1):
                priority = dict((s[0], s[1]) for s in stops)[name].capitalize()
                expected.append(f"{row['Vehicle']} · Stop {n} · {name} · {priority} priority")
        expected += [f"Not delivered · {name} · Low priority" for name in result["dropped"]]
        self.assertEqual(tooltips, sorted(expected))
        self.assertEqual(len(result["dropped"]), 1)

    def test_repeat_submit_saves_once_and_results_persist(self):
        at = self.plan([("Guindy", "high", 1.0), ("Adyar", "low", 1.0)])
        first = self.db("SELECT total_deliveries, total_distance, fuel_consumed FROM delivery_summary")
        self.button(at, "Plan routes").click().run()
        at.run()
        self.assertEqual(self.db("SELECT total_deliveries, total_distance, fuel_consumed FROM delivery_summary"),
                         first)
        self.assertEqual(self.db("SELECT COUNT(*) FROM heatmap_points"), [(2,)])
        self.assertTrue(at.metric, "results should stay visible after a rerun")
        deliveries, distance_m, fuel = first[0]
        self.assertEqual(deliveries, 2)
        self.assertAlmostEqual(fuel, distance_m / 1000 / 8.0, places=6)  # default 8 km/L

    # -- matched addresses and driver hand-off ------------------------------
    def test_matched_addresses_are_shown(self):
        at = self.plan([("Guindy", "high", 1.0), ("Adyar", "low", 1.0)], capacities=(1.0,))
        result = at.session_state.result
        self.assertEqual(result["depot"]["matched"], "Chennai Central, Chennai, Tamil Nadu, India")
        by_name = {r["Location"]: r for r in result["stop_rows"]}
        self.assertEqual(by_name["Guindy"]["Matched address"], "Guindy, Chennai, Tamil Nadu, India")
        self.assertEqual(by_name["Adyar"]["Vehicle"], "Not delivered")  # dropped stops are still listed

    def test_google_maps_link_follows_visit_order(self):
        at = self.plan([("Guindy", "high", 1.0), ("Tambaram", "low", 1.0), ("Adyar", "low", 1.0)])
        result = at.session_state.result
        [vehicle] = result["handoff"]
        [link] = vehicle["links"]
        query = parse_qs(urlparse(link).query)
        route = [r["Location"] for r in result["stop_rows"] if r["Vehicle"] == "Van 1"]
        fmt = lambda name: "{:.6f},{:.6f}".format(*PLACES[name])
        self.assertEqual(query["origin"], [fmt("Chennai Central")])
        self.assertEqual(query["waypoints"], ["|".join(fmt(n) for n in route[:-1])])
        self.assertEqual(query["destination"], [fmt(route[-1])])
        self.assertEqual(query["travelmode"], ["driving"])

    def test_long_route_is_split_into_connected_maps_links(self):
        stops = [(f"Point {i}", "low", 0.5) for i in range(1, 13)]
        at = self.plan(stops, capacities=(10.0,))
        links = at.session_state.result["handoff"][0]["links"]
        self.assertEqual(len(links), 2)  # 12 stops: 9 waypoints + destination, then the remaining 2
        legs = [parse_qs(urlparse(link).query) for link in links]
        for leg in legs:
            self.assertLessEqual(len(leg.get("waypoints", [""])[0].split("|")), 9)
        self.assertEqual(legs[1]["origin"], legs[0]["destination"], "part 2 must start where part 1 ends")
        visited = sum(len(leg.get("waypoints", [""])[0].split("|")) + 1 for leg in legs)
        self.assertEqual(visited, 12)

    def test_run_sheets(self):
        at = self.plan([("Guindy", "high", 1.0), ("Tambaram", "low", 1.0), ("Adyar", "low", 1.0)],
                       capacities=(2.0,))
        result = at.session_state.result
        sheet = pd.read_csv(io.BytesIO(result["run_sheet_csv"]))
        self.assertEqual(list(sheet.columns), ["Vehicle", "Stop", "Location", "Matched address", "Priority",
                                               "Volume (m³)", "Latitude", "Longitude"])
        self.assertEqual(len(sheet), 3)
        self.assertEqual((sheet["Vehicle"] == "Not delivered").sum(), 1)
        page = result["run_sheet_html"]
        self.assertIn("Van 1 · 2 stops", page)
        self.assertIn("https://www.google.com/maps/dir/?", page)
        self.assertIn("Not delivered", page)
        self.assertEqual(len(at.get("download_button")), 3)  # template + two run sheets

    # -- per-user data ----------------------------------------------------
    def test_history_is_private_to_each_user(self):
        self.plan([("Guindy", "high", 1.0)], user="alice")
        alice = self.app(logged_in=True, page="Analytics", user="alice")
        self.assertEqual({m.label: m.value for m in alice.metric}["Deliveries"], "1")
        bob = self.app(logged_in=True, page="Analytics", user="bob")
        self.assertEqual(self.texts(bob.info), ["No delivery data yet. Plans you create are recorded here."])
        bob.radio(key="page").set_value("Heatmap").run()
        self.assertTrue(any("0 of your delivered stops" in c.value for c in bob.caption))
        # Bob clearing his heatmap must not touch Alice's points.
        self.assertEqual(self.db("SELECT username, COUNT(*) FROM heatmap_points GROUP BY username"), [("alice", 1)])

    def test_old_database_is_migrated(self):
        con = sqlite3.connect(self.db_path)
        con.execute("CREATE TABLE heatmap_points (lat REAL, lon REAL)")
        con.execute("CREATE TABLE delivery_summary (date TEXT, fuel_consumed REAL, total_deliveries INTEGER, "
                    "vehicles_used INTEGER, priority_high INTEGER, priority_medium INTEGER, "
                    "priority_low INTEGER, total_distance REAL)")
        con.execute("INSERT INTO delivery_summary VALUES ('2025-05-03', 1, 5, 1, 1, 2, 2, 1000)")
        con.commit()
        con.close()
        at = self.app(logged_in=True, page="Analytics")
        self.assertNoExceptions(at)
        for table in ("heatmap_points", "delivery_summary"):
            columns = [c[1] for c in self.db(f"PRAGMA table_info({table})")]
            self.assertIn("username", columns)
        self.assertEqual(self.texts(at.info), ["No delivery data yet. Plans you create are recorded here."])

    # -- strict priority ---------------------------------------------------
    # From the depot, Adyar is on the way to Tambaram, so the shortest route visits Adyar first.
    def route_of(self, at, vehicle="Van 1"):
        return [r["Location"] for r in at.session_state.result["stop_rows"] if r["Vehicle"] == vehicle]

    def test_without_strict_priority_distance_wins(self):
        at = self.plan([("Adyar", "low", 1.0), ("Tambaram", "high", 1.0)])
        self.assertEqual(self.route_of(at), ["Adyar", "Tambaram"])
        self.assertTrue(any("nudged earlier" in c.value for c in at.caption))

    def test_strict_priority_puts_high_first(self):
        at = self.plan([("Adyar", "low", 1.0), ("Tambaram", "high", 1.0)], strict=True)
        self.assertNoExceptions(at)
        self.assertEqual(self.route_of(at), ["Tambaram", "Adyar"])
        self.assertTrue(any("high-priority stops first" in c.value for c in at.caption))

    def test_strict_priority_orders_every_vehicle(self):
        stops = [("Adyar", "low", 1.0), ("Point 3", "medium", 1.0), ("Tambaram", "high", 1.0),
                 ("Guindy", "low", 1.0), ("Point 9", "high", 1.0), ("Anna Nagar", "medium", 1.0)]
        at = self.plan(stops, capacities=(3.0, 3.0), strict=True)
        rank = {"High": 1, "Medium": 2, "Low": 3}
        rows = at.session_state.result["stop_rows"]
        for vehicle in ("Van 1", "Van 2"):
            ranks = [rank[r["Priority"]] for r in rows if r["Vehicle"] == vehicle]
            self.assertEqual(ranks, sorted(ranks), f"{vehicle} visits a lower priority before a higher one")
        self.assertEqual(len(at.session_state.result["served"]), 6)

    # -- other pages -----------------------------------------------------
    def test_heatmap_and_analytics_pages(self):
        at = self.plan([("Guindy", "high", 1.0)])
        at.radio(key="page").set_value("Analytics").run()
        self.assertNoExceptions(at)
        self.assertEqual({m.label: m.value for m in at.metric}["Deliveries"], "1")
        at.radio(key="page").set_value("Heatmap").run()
        self.assertNoExceptions(at)
        self.assertEqual(self.db("SELECT COUNT(*) FROM heatmap_points"), [(1,)])
        self.assertTrue(self.button(at, "Clear heatmap").disabled, "clearing must need confirmation")
        at.checkbox(key="confirm_clear").check().run()
        self.button(at, "Clear heatmap").click().run()
        self.assertEqual(self.db("SELECT COUNT(*) FROM heatmap_points"), [(0,)])
        self.assertNoExceptions(at)

    def test_empty_analytics(self):
        at = self.app(logged_in=True, page="Analytics")
        self.assertNoExceptions(at)
        self.assertEqual(self.texts(at.info), ["No delivery data yet. Plans you create are recorded here."])


class StopImportTests(unittest.TestCase):
    @staticmethod
    def csv(*lines):
        return ("\n".join(lines) + "\n").encode("utf-8")

    def test_template_round_trips(self):
        stops, notes = parse_stops_file("template.csv", template_csv())
        self.assertEqual(list(stops.columns), ["Location", "Priority", "Volume (m³)"])
        self.assertEqual(len(stops), 3)
        self.assertEqual(notes, [])

    def test_header_aliases_defaults_and_blank_rows(self):
        data = self.csv("Address,URGENCY,volume_m3", "Guindy,high,2", ",low,1", "  Adyar ,,")
        stops, notes = parse_stops_file("stops.csv", data)
        self.assertEqual(stops["Location"].tolist(), ["Guindy", "Adyar"])
        self.assertEqual(stops["Priority"].tolist(), ["High", "Medium"])
        self.assertEqual(stops["Volume (m³)"].tolist(), [2.0, 1.0])
        self.assertEqual(len(notes), 1)  # Adyar's missing volume

    def test_only_location_column_is_required(self):
        stops, notes = parse_stops_file("stops.csv", self.csv("Location", "Guindy", "Adyar"))
        self.assertEqual(stops["Priority"].tolist(), ["Medium", "Medium"])
        self.assertEqual(stops["Volume (m³)"].tolist(), [1.0, 1.0])
        self.assertEqual(notes, [])

    def test_bad_values_are_reported(self):
        data = self.csv("Location,Priority,Volume", "Guindy,urgent,-3", "Adyar,Low,abc")
        stops, notes = parse_stops_file("stops.csv", data)
        self.assertEqual(stops["Priority"].tolist(), ["Medium", "Low"])
        self.assertEqual(stops["Volume (m³)"].tolist(), [1.0, 1.0])
        self.assertTrue(any("'urgent'" in n for n in notes))
        self.assertTrue(any("Guindy, Adyar" in n for n in notes))

    def test_unusable_files_raise(self):
        with self.assertRaisesRegex(ValueError, "Location"):
            parse_stops_file("stops.csv", self.csv("City,Priority", "Chennai,High"))
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_stops_file("stops.csv", self.csv("Location", "", " "))
        with self.assertRaisesRegex(ValueError, ".csv or .xlsx"):
            parse_stops_file("stops.txt", self.csv("Location", "Guindy"))

    def test_excel_and_windows_encodings(self):
        buffer = io.BytesIO()
        pd.DataFrame({"Location": ["Guindy"], "Priority": ["Low"], "Volume (m³)": [2.5]}).to_excel(buffer, index=False)
        stops, _ = parse_stops_file("stops.xlsx", buffer.getvalue())
        self.assertEqual(stops.iloc[0].tolist(), ["Guindy", "Low", 2.5])
        # Values containing commas are quoted, as Excel does when saving CSV.
        stops, _ = parse_stops_file("stops.csv", 'Location\n"Café Coffee Day, Adyar"\n'.encode("cp1252"))
        self.assertEqual(stops["Location"].tolist(), ["Café Coffee Day, Adyar"])

    def test_large_files_are_capped(self):
        data = self.csv("Location", *[f"Stop {i}" for i in range(MAX_STOPS + 5)])
        stops, notes = parse_stops_file("stops.csv", data)
        self.assertEqual(len(stops), MAX_STOPS)
        self.assertTrue(any(f"first {MAX_STOPS}" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
