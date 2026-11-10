"""Offline tests for logistigo.py.

Run:  Scripts\\python.exe -m unittest -v test_logistigo

The Nominatim/OSRM servers are replaced with a fake that uses straight-line
distances, and each test uses its own temporary database.
"""
import hashlib
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

import pandas as pd
import requests
import streamlit as st
from geopy.distance import geodesic
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logistigo.py")

PLACES = {
    "Chennai Central": (13.0827, 80.2707),
    "Guindy": (13.0067, 80.2206),
    "Tambaram": (12.9249, 80.1000),
    "Anna Nagar": (13.0850, 80.2101),
    "Adyar": (13.0012, 80.2565),
}


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
            return FakeResponse([{"lat": place[0], "lon": place[1]}] if place else [])
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
        env = mock.patch.dict(os.environ, {"LOGISTIGO_DB": self.db_path})
        env.start()
        self.addCleanup(env.stop)
        self.servers = FakeMapServers()
        net = mock.patch.object(requests, "get", self.servers)
        net.start()
        self.addCleanup(net.stop)
        st.cache_data.clear()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)  # db may still be open on Windows

    # -- helpers ---------------------------------------------------------
    def app(self, logged_in=False, page=None):
        at = AppTest.from_file(APP, default_timeout=60)
        if logged_in:
            at.session_state.authenticated = True
            at.session_state.username = "tester"
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

    def plan(self, stops, capacities=(10.0,), start="Chennai Central"):
        """stops: list of (name, priority, volume). Fills the planner and submits it."""
        at = AppTest.from_file(APP, default_timeout=60)
        at.session_state.authenticated = True
        at.session_state.username = "tester"
        at.session_state.vehicles_input = pd.DataFrame(
            {"Vehicle": [f"Van {i + 1}" for i in range(len(capacities))], "Capacity (m³)": list(capacities)})
        at.session_state.stops_input = pd.DataFrame(
            {"Location": [n for n, _, _ in stops], "Priority": [p.capitalize() for _, p, _ in stops],
             "Volume (m³)": [v for _, _, v in stops]})
        at.run()
        at.text_input(key="start_location").set_value(start)
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
        # fleet + stops editors, plus the per-vehicle results table
        self.assertEqual(len(at.dataframe), 3, "per-vehicle table should be shown")

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


if __name__ == "__main__":
    unittest.main()
