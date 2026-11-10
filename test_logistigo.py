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
    def app(self, logged_in=False):
        at = AppTest.from_file(APP, default_timeout=60)
        if logged_in:
            at.session_state.authenticated = True
            at.session_state.username = "tester"
        return at.run()

    def db(self, sql, *args):
        con = sqlite3.connect(self.db_path)
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()

    def assertNoExceptions(self, at):
        self.assertEqual([e.value for e in at.exception], [])

    def logistics_form(self, stops, vehicles=1, capacity=10.0, start="Chennai Central"):
        """stops: list of (name, priority, volume)."""
        at = self.app(logged_in=True)
        at.radio[0].set_value("Logistics").run()
        at.number_input[0].set_value(vehicles).run()
        for i in range(vehicles):
            at.number_input(key=f"cap_{i}").set_value(capacity)
        next(w for w in at.number_input if w.label == "Number of Delivery Locations").set_value(len(stops))
        at.run()
        at.text_input[0].set_value(start)
        for i, (name, priority, volume) in enumerate(stops):
            at.text_input(key=f"loc_name_{i}").set_value(name)
            at.selectbox(key=f"priority_{i}").set_value(priority)
            at.number_input(key=f"volume_{i}").set_value(volume)
        return at.run()

    @staticmethod
    def find_button(at):
        return next(b for b in at.button if "Optimal" in b.label)

    # -- accounts --------------------------------------------------------
    def test_signup_validation(self):
        at = self.app()
        at.sidebar.selectbox[0].set_value("Sign Up").run()

        def sign_up(user, password):
            at.text_input(key="signup_user").set_value(user)
            at.text_input(key="signup_password").set_value(password)
            at.button[0].click().run()

        sign_up("", "secret1")
        self.assertIn("Username and password are required.", [e.value for e in at.error])
        sign_up("alice", "123")
        self.assertIn("Password must be at least 6 characters.", [e.value for e in at.error])
        sign_up("alice", "secret1")
        self.assertTrue(at.success)
        sign_up("alice", "another1")
        self.assertIn("That username is already taken.", [e.value for e in at.error])
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
            at.button[0].click().run()
            self.assertEqual(at.session_state.authenticated, expected, password)
        stored = self.db("SELECT password FROM users WHERE username='old'")[0][0]
        self.assertTrue(stored.startswith("pbkdf2$"), "legacy hash should be upgraded after login")

    # -- route planner ---------------------------------------------------
    def test_blank_form_shows_no_errors_and_button_disabled(self):
        at = self.app(logged_in=True)
        at.radio[0].set_value("Logistics").run()
        self.assertNoExceptions(at)
        self.assertEqual([e.value for e in at.error], [])
        self.assertEqual([w.value for w in at.warning], [])
        self.assertTrue(self.find_button(at).disabled)

    def test_unknown_place_is_reported(self):
        at = self.logistics_form([("Atlantis", "low", 1.0)])
        self.assertIn("❌ Could not find coordinates for 'Atlantis'", [w.value for w in at.warning])

    def test_geocoding_is_cached_between_reruns(self):
        at = self.logistics_form([("Guindy", "low", 1.0), ("Adyar", "low", 1.0)])
        before = self.servers.geocode_calls
        at.run()
        self.assertEqual(self.servers.geocode_calls, before)

    def test_distance_covers_whole_route(self):
        at = self.logistics_form([("Guindy", "low", 1.0), ("Tambaram", "low", 1.0)])
        self.find_button(at).click().run()
        self.assertNoExceptions(at)
        metrics = {m.label: m.value for m in at.metric}
        shown_km = float(metrics["Optimised distance"].split()[0])
        self.assertAlmostEqual(shown_km, route_km("Chennai Central", "Guindy", "Tambaram"), places=1)

    def test_optimised_beats_entered_order(self):
        # Entered order zig-zags; the optimiser should not be worse.
        at = self.logistics_form([("Tambaram", "low", 1.0), ("Anna Nagar", "low", 1.0), ("Guindy", "low", 1.0)])
        self.find_button(at).click().run()
        metrics = {m.label: float(m.value.split()[0]) for m in at.metric if "distance" in m.label}
        self.assertLessEqual(metrics["Optimised distance"], metrics["Unoptimised distance"])

    def test_capacity_by_volume_drops_low_priority_first(self):
        stops = [("Guindy", "low", 1.0), ("Tambaram", "medium", 1.0), ("Anna Nagar", "high", 1.0),
                 ("Adyar", "low", 1.0)]
        at = self.logistics_form(stops, vehicles=1, capacity=2.5)
        self.assertTrue(any("exceeds total vehicle capacity" in w.value for w in at.warning))
        self.find_button(at).click().run()
        self.assertNoExceptions(at)
        self.assertEqual([e.value for e in at.error], ["Not delivered (no vehicle had room): Guindy, Adyar"])
        row = self.db("SELECT total_deliveries, priority_high, priority_medium, priority_low FROM delivery_summary")
        self.assertEqual(row, [(2, 1, 1, 0)])

    def test_oversized_stop_is_flagged(self):
        at = self.logistics_form([("Guindy", "high", 50.0)], capacity=10.0)
        self.assertTrue(any("larger than any vehicle" in w.value for w in at.warning))

    def test_multiple_vehicles_share_the_load(self):
        stops = [(n, "low", 4.0) for n in ("Guindy", "Tambaram", "Anna Nagar", "Adyar")]
        at = self.logistics_form(stops, vehicles=2, capacity=10.0)
        self.find_button(at).click().run()
        self.assertEqual([e.value for e in at.error], [])
        routes = [m.value for m in at.markdown if m.value.startswith("**Vehicle")]
        self.assertEqual(len(routes), 2)  # 16 m³ can't fit in one 10 m³ vehicle

    def test_repeat_click_saves_once_and_results_persist(self):
        at = self.logistics_form([("Guindy", "high", 1.0), ("Adyar", "low", 1.0)])
        self.find_button(at).click().run()
        first = self.db("SELECT total_deliveries, total_distance, fuel_consumed FROM delivery_summary")
        self.find_button(at).click().run()
        at.run()
        self.assertEqual(self.db("SELECT total_deliveries, total_distance, fuel_consumed FROM delivery_summary"),
                         first)
        self.assertEqual(self.db("SELECT COUNT(*) FROM heatmap_points"), [(2,)])
        self.assertTrue(at.metric, "results should stay visible after a rerun")
        deliveries, distance_m, fuel = first[0]
        self.assertEqual(deliveries, 2)
        self.assertAlmostEqual(fuel, distance_m / 1000 / 8.0, places=6)  # default 8 km/l

    # -- other pages -----------------------------------------------------
    def test_heatmap_and_analytics_pages(self):
        at = self.logistics_form([("Guindy", "high", 1.0)])
        self.find_button(at).click().run()
        for page in ("Analytics", "Heatmap"):
            at.radio[0].set_value(page).run()
            self.assertNoExceptions(at)
        self.assertEqual(self.db("SELECT COUNT(*) FROM heatmap_points"), [(1,)])
        next(b for b in at.button if "Clear" in b.label).click().run()
        at.run()
        self.assertEqual(self.db("SELECT COUNT(*) FROM heatmap_points"), [(0,)])
        self.assertNoExceptions(at)

    def test_empty_analytics(self):
        at = self.app(logged_in=True)
        at.radio[0].set_value("Analytics").run()
        self.assertNoExceptions(at)
        self.assertEqual([i.value for i in at.info], ["No delivery data found."])


if __name__ == "__main__":
    unittest.main()
