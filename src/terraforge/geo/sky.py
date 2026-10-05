"""Sun position, day/night terminator and moon phase - computed, not faked.

Sun: NOAA's solar-position approximation (Fourier series in the fractional year) gives the
declination (latitude of the subsolar point) and the equation of time; the subsolar
longitude follows from UTC time. Accuracy ~0.1 deg, far finer than a globe needs.

Terminator: the great circle 90 deg from the subsolar point. For each longitude,
    lat = atan(-cos(lon - lon_sun) / tan(decl))
Day side is the hemisphere containing the subsolar point.

Moon phase: the synodic month (29.530588 d) counted from a reference new moon
(2000-01-06 18:14 UTC). Illumination = (1 - cos(2*pi*age/period)) / 2. This ignores the
moon's orbital eccentricity, so phase times can be off by up to ~14 hours - fine for
display, not for ephemeris-grade work, and documented as such.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

SYNODIC_MONTH = 29.530588853
REF_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)


def _utc(dt: datetime | None) -> datetime:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def sun_position(dt: datetime | None = None) -> dict:
    """Subsolar point: {'lat', 'lon'} in degrees (lon in [-180, 180])."""
    dt = _utc(dt)
    doy = dt.timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1 + (dt.hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))  # minutes
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    minutes = dt.hour * 60 + dt.minute + dt.second / 60
    lon = (720 - minutes - eqtime) / 4
    lon = (lon + 180) % 360 - 180
    return {"lat": math.degrees(decl), "lon": lon}


def terminator(dt: datetime | None = None, n: int = 180) -> list[list[float]]:
    """[[lon, lat], ...] along the day/night boundary, west to east."""
    s = sun_position(dt)
    decl = math.radians(s["lat"])
    if abs(decl) < 1e-6:
        decl = 1e-6  # equinox: boundary is the meridians 90 deg from the sun
    pts = []
    for i in range(n + 1):
        lon = -180 + 360 * i / n
        h = math.radians(lon - s["lon"])
        pts.append([lon, math.degrees(math.atan(-math.cos(h) / math.tan(decl)))])
    return pts


def is_daylight(lat: float, lon: float, dt: datetime | None = None) -> bool:
    """True if the sun is above the horizon at (lat, lon)."""
    s = sun_position(dt)
    la, lo, sl, so = map(math.radians, (lat, lon, s["lat"], s["lon"]))
    cos_zenith = math.sin(la) * math.sin(sl) + math.cos(la) * math.cos(sl) * math.cos(lo - so)
    return cos_zenith > 0


def moon_phase(dt: datetime | None = None) -> dict:
    """age_days, illumination in [0,1], and a phase name."""
    age = ((_utc(dt) - REF_NEW_MOON).total_seconds() / 86400) % SYNODIC_MONTH
    frac = age / SYNODIC_MONTH
    illum = (1 - math.cos(2 * math.pi * frac)) / 2
    names = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
             "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
    return {"age_days": age, "illumination": illum, "phase": names[int((frac * 8 + 0.5) % 8)]}
