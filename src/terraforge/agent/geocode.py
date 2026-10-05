"""Place name -> lat/lon via OpenStreetMap Nominatim (free).

Nominatim's usage policy requires: an identifying User-Agent, at most 1 request per second,
and caching so the same query is never sent twice. All three are enforced here. For anything
beyond light interactive use, self-host or use a commercial geocoder instead.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request

URL = "https://nominatim.openstreetmap.org/search"
UA = "terraforge/0.1 (+https://github.com/Professional50coder/terraforge)"


class Geocoder:
    def __init__(self, fetch=None, min_interval: float = 1.0, timeout: float = 10):
        self._fetch = fetch or self._http
        self.min_interval, self.timeout = min_interval, timeout
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._last = 0.0

    def _http(self, place: str) -> list:
        q = urllib.parse.urlencode({"q": place, "format": "jsonv2", "limit": 1})
        req = urllib.request.Request(f"{URL}?{q}", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def __call__(self, place: str) -> dict:
        key = " ".join(place.lower().split())
        if not 2 <= len(key) <= 100:
            raise ValueError("place name must be 2-100 characters")
        with self._lock:  # serialises calls so the rate limit holds across threads
            if key in self._cache:
                return self._cache[key]
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                hits = self._fetch(key)
            finally:
                self._last = time.monotonic()
            if not hits:
                raise LookupError(f"no place found for '{place}'")
            h = hits[0]
            out = {"name": h.get("display_name", place)[:120],
                   "lat": float(h["lat"]), "lon": float(h["lon"])}
            self._cache[key] = out
            return out
