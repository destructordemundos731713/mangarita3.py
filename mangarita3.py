"""
Mangarita - NASA POWER climate explorer

How to run (in a terminal):
    pip install pandas numpy plotly requests
    python mangarita.py

What it does:
  1. Loads the local RESERVE (reserva_nasa.json): real data downloaded from the NASA POWER API.
  2. If catalog locations are missing, it downloads them from NASA and saves them to the reserve.
     If NASA does not respond, it works with what is already in the reserve (it never makes up data).
  3. Opens a page in your browser with two buttons:
       - View charts and full analysis (comparison dashboard)
       - Search by coordinates:
           a) if the coordinate is in the reserve, it shows it;
           b) if not, it requests it from NASA, shows it and ADDS it to the reserve;
           c) if NASA does not respond, it shows the nearest location in the reserve.
       - After a search, a "See the future" button projects every variable into the future
         (linear trend + seasonal cycle fitted by least squares, with a 95 % range).
       - Above the chart, "Predict future years" adds a 36-month projection computed in the
         browser (linear trend + average monthly anomaly); it needs no internet at all.
  4. The dashboard shows at most 9 locations: it starts with the featured stations and every
     coordinate you search is added to it. When it is full, the oldest featured station is removed
     first and then the oldest search. The reserve itself never loses anything.

Mangarita the cat (home page):
    She reacts to the live temperature of Ipiales, Narino, Colombia (Open-Meteo, no API key needed):
    below 10 C she shivers, wraps herself in a blanket, reads a book and falls asleep; between 10 and
    17 C she is her normal self; above 17 C she puts on a bikini and a snorkel. The page asks the
    program (/api/ipiales-weather) every 60 seconds. With no internet it uses a safe backup of 14.0 C.
    For demos, a small selector under her simulates Cold / Mild / Hot (default: Live).

Cloud deployment (Render / Hugging Face / Railway):
    The server listens on 0.0.0.0 and takes the port from the PORT environment variable.
    The browser is only opened automatically when PORT is not set (i.e. on your own computer).

Options:
    --update       download the whole catalog again (refreshes the reserve)
    --no-download  do not try to download the catalog on startup
To quit the program: press Ctrl + C in the terminal.
"""

import argparse
import json
import math
import os
import threading
import time
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
START_YEAR = 2021
END_YEAR = date.today().year   # always up to today; NASA only returns the months it already has

API_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SYNC_TIMEOUT = 30        # seconds per request when downloading the catalog
SEARCH_TIMEOUT = 20      # seconds per request when the user searches a coordinate
PAUSE_BETWEEN_REQUESTS = 0.4  # to avoid overloading the NASA API

# A coordinate counts as "already in the reserve" if it is within this difference
# (in degrees) of a saved location. 0.05 degrees = about 5 km.
EXACT_MATCH_TOLERANCE_DEG = 0.05

# Critical t value (~95 %) used to decide whether a trend is statistically significant
T_CRITICAL = 2.01

# "See the future": statistical projection settings
FORECAST_DEFAULT_MONTHS = 24   # default horizon shown in the page
FORECAST_MAX_MONTHS = 60       # the server never projects further than this
FORECAST_HARMONICS = 2         # number of seasonal cycles (1 = annual wave, 2 = annual + semi-annual)
FORECAST_MIN_RECORDS = 24      # minimum months of history needed to fit the model
# Physical limits so a projection never leaves what is possible: (minimum, maximum), None = no limit
VARIABLE_LIMITS = {
    "RH2M": (0.0, 100.0),
    "WS10M": (0.0, None),
    "PS": (0.0, None),
    "PRECTOTCORR": (0.0, None),
    "ALLSKY_SFC_SW_DWN": (0.0, None),
}

PREFERRED_PORT = 8765

# Live weather of Ipiales, Narino, Colombia (Open-Meteo: free, no API key)
IPIALES_LAT, IPIALES_LON = 0.8292, -77.6419
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = 5            # seconds: short, so the page never hangs waiting for it
WEATHER_FALLBACK_TEMP = 14.0   # usual cool/mild temperature of Ipiales, used when the API cannot be reached
WEATHER_CACHE_SECONDS = 30     # several open pages share one Open-Meteo request

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # Jupyter / Colab
    BASE_DIR = Path(os.getcwd())
# The file name is kept so the reserve you already downloaded keeps working.
RESERVE_FILE = BASE_DIR / "reserva_nasa.json"

VARIABLES_CONFIG = {
    "T2M": {"name": "Atmospheric temperature", "unit": "°C"},
    "RH2M": {"name": "Relative humidity (2 m)", "unit": "%"},
    "WS10M": {"name": "Wind speed (10 m)", "unit": "m/s"},
    "PS": {"name": "Surface pressure", "unit": "kPa"},
    "PRECTOTCORR": {"name": "Precipitation rate", "unit": "mm/day"},
    "ALLSKY_SFC_SW_DWN": {"name": "Solar downward flux", "unit": "W/m²"},
}
SOLAR_CODE = "ALLSKY_SFC_SW_DWN"

# Stations compared in the dashboard (they must exist in the catalog)
FEATURED = ["Alaska (Arctic)", "Bogota (Equatorial)", "Madrid (Mediterranean)", "Sydney (Southern Hem.)"]

STATION_COLORS = {
    "Alaska (Arctic)": "#00CC96",
    "Bogota (Equatorial)": "#EF553B",
    "Madrid (Mediterranean)": "#636EFA",
    "Sydney (Southern Hem.)": "#FFA15A",
}
EXTRA_COLORS = ["#AB63FA", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "#FFD700", "#7FDBFF", "#F4A6D7"]

# The dashboard never shows more than this many locations at once
DASHBOARD_MAX = 9

# Locations to download: name (latitude, longitude).
# To add more, just add a line here and run the program again.
CATALOG = {
    # The 4 original stations
    "Alaska (Arctic)": (64.8378, -147.7164),
    "Bogota (Equatorial)": (4.6097, -74.0817),
    "Madrid (Mediterranean)": (40.4167, -3.7037),
    "Sydney (Southern Hem.)": (-33.8688, 151.2093),
    # Americas
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Miami": (25.7617, -80.1918),
    "Denver": (39.7392, -104.9903),
    "Anchorage": (61.2181, -149.9003),
    "Honolulu": (21.3099, -157.8581),
    "Vancouver": (49.2827, -123.1207),
    "Toronto": (43.6532, -79.3832),
    "Mexico City": (19.4326, -99.1332),
    "Havana": (23.1136, -82.3666),
    "Panama City": (8.9824, -79.5199),
    "Caracas": (10.4806, -66.9036),
    "Medellín": (6.2442, -75.5812),
    "Cali": (3.4516, -76.5320),
    "Quito": (-0.1807, -78.4678),
    "Lima": (-12.0464, -77.0428),
    "La Paz": (-16.4897, -68.1193),
    "Manaus": (-3.1190, -60.0217),
    "São Paulo": (-23.5505, -46.6333),
    "Santiago": (-33.4489, -70.6693),
    "Buenos Aires": (-34.6037, -58.3816),
    "Ushuaia": (-54.8019, -68.3030),
    # Europe
    "Reykjavik": (64.1466, -21.9426),
    "Oslo": (59.9139, 10.7522),
    "Stockholm": (59.3293, 18.0686),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Lisbon": (38.7223, -9.1393),
    "Rome": (41.9028, 12.4964),
    "Athens": (37.9838, 23.7275),
    "Istanbul": (41.0082, 28.9784),
    "Moscow": (55.7558, 37.6173),
    # Africa
    "Casablanca": (33.5731, -7.5898),
    "Cairo": (30.0444, 31.2357),
    "Dakar": (14.7167, -17.4677),
    "Timbuktu": (16.7735, -3.0074),
    "Lagos": (6.5244, 3.3792),
    "Kinshasa": (-4.4419, 15.2663),
    "Addis Ababa": (9.0300, 38.7400),
    "Nairobi": (-1.2921, 36.8219),
    "Johannesburg": (-26.2041, 28.0473),
    "Cape Town": (-33.9249, 18.4241),
    # Asia
    "Ankara": (39.9334, 32.8597),
    "Baghdad": (33.3152, 44.3661),
    "Tehran": (35.6892, 51.3890),
    "Riyadh": (24.7136, 46.6753),
    "Dubai": (25.2048, 55.2708),
    "Karachi": (24.8607, 67.0011),
    "Tashkent": (41.2995, 69.2401),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kathmandu": (27.7172, 85.3240),
    "Bangkok": (13.7563, 100.5018),
    "Hanoi": (21.0285, 105.8542),
    "Singapore": (1.3521, 103.8198),
    "Jakarta": (-6.2088, 106.8456),
    "Manila": (14.5995, 120.9842),
    "Beijing": (39.9042, 116.4074),
    "Shanghai": (31.2304, 121.4737),
    "Seoul": (37.5665, 126.9780),
    "Tokyo": (35.6762, 139.6503),
    "Ulaanbaatar": (47.8864, 106.9057),
    "Yakutsk": (62.0355, 129.6755),
    # Oceania and Antarctica
    "Darwin": (-12.4634, 130.8456),
    "Alice Springs": (-23.6980, 133.8807),
    "Perth": (-31.9505, 115.8605),
    "Melbourne": (-37.8136, 144.9631),
    "Auckland": (-36.8485, 174.7633),
    "McMurdo Station (Antarctica)": (-77.8460, 166.6680),
}

# Locations saved by the previous (Spanish) version are renamed automatically,
# so nothing has to be downloaded twice.
NAME_MIGRATIONS = {
    "Nueva York": "New York", "Los Ángeles": "Los Angeles", "Ciudad de México": "Mexico City",
    "La Habana": "Havana", "Panamá": "Panama City", "Manaos": "Manaus", "Santiago de Chile": "Santiago",
    "Reikiavik": "Reykjavik", "Estocolmo": "Stockholm", "Londres": "London", "París": "Paris",
    "Berlín": "Berlin", "Lisboa": "Lisbon", "Roma": "Rome", "Atenas": "Athens", "Estambul": "Istanbul",
    "Moscú": "Moscow", "El Cairo": "Cairo", "Tombuctú": "Timbuktu", "Adís Abeba": "Addis Ababa",
    "Johannesburgo": "Johannesburg", "Ciudad del Cabo": "Cape Town", "Bagdad": "Baghdad",
    "Teherán": "Tehran", "Riad": "Riyadh", "Dubái": "Dubai", "Taskent": "Tashkent", "Bombay": "Mumbai",
    "Katmandú": "Kathmandu", "Hanói": "Hanoi", "Singapur": "Singapore", "Yakarta": "Jakarta",
    "Pekín": "Beijing", "Shanghái": "Shanghai", "Seúl": "Seoul", "Tokio": "Tokyo",
    "Ulán Bator": "Ulaanbaatar", "Base McMurdo (Antártida)": "McMurdo Station (Antarctica)",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOCAL RESERVE (only real data downloaded from NASA)
# ──────────────────────────────────────────────────────────────────────────────
reserve_lock = threading.RLock()
RESERVE = {}  # name -> {"lat","lon","downloaded","source","months":[...], "T2M":[...], ...}


def _valid_entry(entry):
    try:
        n = len(entry["months"])
        float(entry["lat"]), float(entry["lon"])
        return n > 0 and all(len(entry[code]) == n for code in VARIABLES_CONFIG)
    except Exception:
        return False


def load_reserve():
    if not RESERVE_FILE.exists():
        return {}
    try:
        locations = json.loads(RESERVE_FILE.read_text(encoding="utf-8"))["locations"]
        return {NAME_MIGRATIONS.get(name, name): e for name, e in locations.items() if _valid_entry(e)}
    except Exception:
        corrupt = RESERVE_FILE.with_name("reserva_nasa_corrupt.json")
        RESERVE_FILE.replace(corrupt)
        print("Warning: the reserve was damaged; a copy was saved as %s and a new one is being started." % corrupt.name)
        return {}


def save_reserve():
    with reserve_lock:
        tmp = RESERVE_FILE.with_name("reserva_nasa.tmp")
        tmp.write_text(json.dumps({"version": 1, "locations": RESERVE}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(RESERVE_FILE)


# Active dashboard list (max DASHBOARD_MAX). It only lives in memory: the RESERVE keeps everything.
DASHBOARD_STATIONS = []      # names, oldest first
DASHBOARD_ORIGINALS = set()  # names in DASHBOARD_STATIONS that came from FEATURED (not from a search)


def init_dashboard():
    """Starts the dashboard with the FEATURED stations that exist in the reserve."""
    with reserve_lock:
        DASHBOARD_STATIONS.clear()
        DASHBOARD_ORIGINALS.clear()
        for name in FEATURED:
            if name in RESERVE and name not in DASHBOARD_STATIONS and len(DASHBOARD_STATIONS) < DASHBOARD_MAX:
                DASHBOARD_STATIONS.append(name)
                DASHBOARD_ORIGINALS.add(name)


def dashboard_seed(name):
    """A featured station that was downloaded after startup joins the dashboard if there is room."""
    with reserve_lock:
        if (name in FEATURED and name in RESERVE and name not in DASHBOARD_STATIONS
                and len(DASHBOARD_STATIONS) < DASHBOARD_MAX):
            DASHBOARD_STATIONS.append(name)
            DASHBOARD_ORIGINALS.add(name)


def dashboard_add(name):
    """
    Adds a searched location to the dashboard (nothing happens if it is already there).
    When the dashboard is full, the oldest original (FEATURED) station is removed first;
    if there are none left, the oldest search is removed (FIFO).
    """
    with reserve_lock:
        if name not in RESERVE or name in DASHBOARD_STATIONS:
            return
        while len(DASHBOARD_STATIONS) >= DASHBOARD_MAX:
            victim = next((n for n in DASHBOARD_STATIONS if n in DASHBOARD_ORIGINALS), DASHBOARD_STATIONS[0])
            DASHBOARD_STATIONS.remove(victim)
            DASHBOARD_ORIGINALS.discard(victim)
        DASHBOARD_STATIONS.append(name)


def entry_to_df(entry):
    data = {"Date": entry["months"]}
    for code in VARIABLES_CONFIG:
        data[code] = entry[code]
    return pd.DataFrame(data)


# ──────────────────────────────────────────────────────────────────────────────
# 3. NASA API
# ──────────────────────────────────────────────────────────────────────────────
def solar_to_w_m2(value, units):
    """Converts solar radiation from the units reported by the API to W/m²."""
    u = (units or "").lower().replace(" ", "").replace("^", "").replace("·", "")
    if u in ("kw-hr/m2/day", "kwh/m2/day", "kw-hr/m2/d"):
        return value * 1000.0 / 24.0          # daily energy -> average power
    if u in ("mj/m2/day", "mj/m2/d"):
        return value * 1e6 / 86400.0
    if u in ("wm-2", "w/m2"):
        return value
    raise ValueError("Unrecognized radiation units: %r" % units)


def fetch_power(lat, lon, retries=0, timeout=SEARCH_TIMEOUT):
    """
    Downloads real monthly NASA POWER data for one point.
    Returns a dict ready for the reserve, or None if something fails.
    """
    params = {
        "start": START_YEAR,
        "end": END_YEAR,
        "latitude": lat,
        "longitude": lon,
        "parameters": ",".join(VARIABLES_CONFIG),
        "community": "RE",
        "format": "JSON",
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(API_URL, params=params, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            parameter_data = payload["properties"]["parameter"]
            solar_units = payload.get("parameters", {}).get(SOLAR_CODE, {}).get("units")
            if solar_units is None:  # the API gave no units: assume the documented monthly/RE ones
                solar_units = "kW-hr/m^2/day"

            months, columns = [], {code: [] for code in VARIABLES_CONFIG}
            for date_key in sorted(parameter_data[next(iter(VARIABLES_CONFIG))]):
                year, month = int(date_key[:4]), int(date_key[4:])
                if not 1 <= month <= 12:  # the API uses month 13 for the annual average
                    continue
                values = {code: parameter_data[code][date_key] for code in VARIABLES_CONFIG}
                if any(v == -999.0 for v in values.values()):  # -999 = invalid data
                    continue
                months.append("%d-%02d" % (year, month))
                for code, v in values.items():
                    v = solar_to_w_m2(v, solar_units) if code == SOLAR_CODE else v
                    columns[code].append(round(float(v), 2))
            if not months:
                raise ValueError("The API returned no valid months")

            entry = {"lat": float(lat), "lon": float(lon), "downloaded": date.today().isoformat(),
                     "end_year": END_YEAR, "source": "NASA POWER", "months": months}
            entry.update(columns)
            return entry
        except Exception as error:  # no internet, API down, unexpected format, etc.
            last_error = error
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    print("    (the API could not be used: %s)" % type(last_error).__name__)
    return None


def sync_catalog(force=False):
    """Downloads into the reserve the catalog locations that are not there yet."""
    this_month = date.today().isoformat()[:7]

    def is_outdated(entry):
        """Saved before this month or with an older end year: newer NASA months may exist now."""
        return entry.get("end_year") != END_YEAR or entry.get("downloaded", "")[:7] != this_month

    with reserve_lock:
        pending = [(n, c) for n, c in CATALOG.items() if force or n not in RESERVE or is_outdated(RESERVE[n])]
        # locations added by coordinate searches are refreshed too
        pending += [(n, (e["lat"], e["lon"])) for n, e in RESERVE.items()
                    if n not in CATALOG and (force or is_outdated(e))]
    if not pending:
        print("The reserve is up to date (%d locations, data until today)." % len(RESERVE))
        return

    print("Downloading %d locations from NASA POWER (this may take a few minutes)..." % len(pending))
    failures = 0
    try:
        for i, (name, (lat, lon)) in enumerate(pending, 1):
            print(" [%d/%d] %s ..." % (i, len(pending), name))
            entry = fetch_power(lat, lon, retries=2, timeout=SYNC_TIMEOUT)
            if entry:
                with reserve_lock:
                    RESERVE[name] = entry
                    save_reserve()
                    dashboard_seed(name)
                failures = 0
            else:
                failures += 1
                if failures >= 3:
                    print("\nIt looks like NASA cannot be reached. Continuing with the current reserve.")
                    break
            time.sleep(PAUSE_BETWEEN_REQUESTS)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Everything downloaded so far is already saved.")
    print("Reserve: %d locations." % len(RESERVE))


# ──────────────────────────────────────────────────────────────────────────────
# 4. COORDINATE SEARCH
# ──────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Distance over the Earth's surface between two coordinates (in km)."""
    earth_radius_km = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return float(2 * earth_radius_km * np.arcsin(np.sqrt(a)))


def nearest_location(lat, lon):
    """Returns (name, distance km, exact match?) or (None, None, False) if the reserve is empty."""
    with reserve_lock:
        best_name, best_dist, best_entry = None, float("inf"), None
        for name, entry in RESERVE.items():
            dist = haversine_km(lat, lon, entry["lat"], entry["lon"])
            if dist < best_dist:
                best_name, best_dist, best_entry = name, dist, entry
        if best_name is None:
            return None, None, False
        is_exact = (abs(lat - best_entry["lat"]) <= EXACT_MATCH_TOLERANCE_DEG
                    and abs(lon - best_entry["lon"]) <= EXACT_MATCH_TOLERANCE_DEG)
        return best_name, best_dist, is_exact


def new_location_name(lat, lon):
    base = "Lat %.2f, Lon %.2f" % (lat, lon)
    name, n = base, 2
    with reserve_lock:
        while name in RESERVE:
            name = "%s (%d)" % (base, n)
            n += 1
    return name


# ──────────────────────────────────────────────────────────────────────────────
# 5. ANALYSIS AND CHARTS (dark theme)
# ──────────────────────────────────────────────────────────────────────────────
TRANSPARENT = "rgba(0,0,0,0)"


def dark_hoverlabel():
    """Dark tooltip with high contrast: the default one can come out white on white with 'x unified'."""
    return dict(bgcolor="#151b2e", bordercolor="#4f46e5", font=dict(color="#f9fafb", size=12))


def color_for(name, position=0):
    return STATION_COLORS.get(name, EXTRA_COLORS[position % len(EXTRA_COLORS)])


def trend_summary(y_vals):
    """Net linear change and whether the trend is statistically significant."""
    n = len(y_vals)
    if n <= 2:
        return None
    x_vals = np.arange(n)
    slope, _ = np.polyfit(x_vals, y_vals, 1)
    total_change = slope * n

    r = np.corrcoef(x_vals, y_vals)[0, 1]
    if np.isnan(r) or abs(r) >= 0.9999:
        return total_change, "N/A"
    t_stat = abs(r) * np.sqrt((n - 2) / (1 - r ** 2))
    return total_change, ("YES (significant) 🚨" if t_stat > T_CRITICAL else "No (seasonal/noise)")


def build_dashboard_figure(frames):
    """6 charts (featured stations overlaid) + summary table. frames: {name: DataFrame}."""
    chart_titles = ["%s (%s)" % (info["name"], info["unit"]) for info in VARIABLES_CONFIG.values()]
    chart_titles.append("📋 NASA summary: what, where, how much changed and is it significant?")

    # With more than 4 stations the summary table has more rows: the page grows so it is not cut off
    base_height = 1350
    row_px = base_height * (1 - 3 * 0.07) / 4
    extra_px = 25 * max(0, len(frames) * len(VARIABLES_CONFIG) - 24)
    total_height = base_height + extra_px

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=chart_titles,
        specs=[[{}, {}], [{}, {}], [{}, {}], [{"type": "table", "colspan": 2}, None]],
        row_heights=[row_px, row_px, row_px, row_px + extra_px],
        vertical_spacing=0.07 * base_height / total_height,
        horizontal_spacing=0.08,
    )

    # fixed colors for the featured stations; the rest take the extra colors one after another
    station_colors, extra_position = {}, 0
    for station_name in frames:
        if station_name in STATION_COLORS:
            station_colors[station_name] = STATION_COLORS[station_name]
        else:
            station_colors[station_name] = EXTRA_COLORS[extra_position % len(EXTRA_COLORS)]
            extra_position += 1

    table_rows = []
    for idx, (param_code, info) in enumerate(VARIABLES_CONFIG.items()):
        row, col = idx // 2 + 1, idx % 2 + 1

        for station_name, df_st in frames.items():
            y_vals = df_st[param_code].astype(float).tolist()
            summary = trend_summary(np.array(y_vals))
            if summary:
                total_change, significant = summary
                table_rows.append([info["name"], station_name,
                                   "%+.2f %s" % (total_change, info["unit"]), significant])

            fig.add_trace(
                go.Scatter(
                    x=df_st["Date"].tolist(), y=y_vals,
                    mode="lines+markers", name=station_name,
                    line=dict(color=station_colors[station_name], width=2),
                    marker=dict(size=4),
                    legendgroup=station_name,
                    showlegend=(idx == 0),
                    hoverlabel=dark_hoverlabel(),
                ),
                row=row, col=col,
            )

    if table_rows:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Variable</b>", "<b>Station</b>",
                            "<b>Net change (%d-%d)</b>" % (START_YEAR, END_YEAR),
                            "<b>Statistically significant?</b>"],
                    fill_color="#312e81", font=dict(color="#f9fafb", size=12), align="left",
                ),
                cells=dict(
                    values=[list(column) for column in zip(*table_rows)],
                    fill_color="#1c2440", font=dict(color="#e5e7eb", size=11), align="left",
                    line_color="#2a3350",
                ),
            ),
            row=4, col=1,
        )

    fig.update_layout(
        height=total_height,
        title_text="🛰️ NASA POWER · Global atmospheric and climate comparison (%d–%d)" % (START_YEAR, END_YEAR),
        template="plotly_dark",
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#151b2e", bordercolor="#4f46e5", font=dict(color="#f9fafb", size=12)),
    )
    return fig


def build_station_figure(df_st, station_name):
    """The 6 variables of a single location."""
    titles = ["%s (%s)" % (info["name"], info["unit"]) for info in VARIABLES_CONFIG.values()]
    fig = make_subplots(rows=3, cols=2, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.09)

    for idx, param_code in enumerate(VARIABLES_CONFIG):
        fig.add_trace(
            go.Scatter(
                x=df_st["Date"].tolist(), y=df_st[param_code].astype(float).tolist(),
                mode="lines+markers", name=VARIABLES_CONFIG[param_code]["name"],
                line=dict(color=color_for(station_name), width=2),
                marker=dict(size=4),
                showlegend=False,
            ),
            row=idx // 2 + 1, col=idx % 2 + 1,
        )

        fig.update_layout(height=800, template="plotly_dark", paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                      hovermode="x unified", hoverlabel=dict(bgcolor="#151b2e", bordercolor="#4f46e5", font=dict(color="#f9fafb")),
                      title_text="Data for %s" % station_name, margin=dict(t=80))
    return fig


def station_stats(df_st):
    """Rows [variable, average, minimum, maximum] for the side table."""
    rows = []
    for param_code, info in VARIABLES_CONFIG.items():
        serie = df_st[param_code].astype(float)
        rows.append(["%s (%s)" % (info["name"], info["unit"]),
                     round(float(serie.mean()), 2), round(float(serie.min()), 2), round(float(serie.max()), 2)])
    return rows


def hex_to_rgba(color, alpha):
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return "rgba(%d,%d,%d,%.2f)" % (r, g, b, alpha)


def _design_matrix(t, month_of_year):
    """Columns: constant, linear trend and FORECAST_HARMONICS seasonal sine/cosine pairs."""
    angle = 2 * np.pi * np.asarray(month_of_year, dtype=float) / 12.0
    columns = [np.ones(len(t)), np.asarray(t, dtype=float)]
    for k in range(1, FORECAST_HARMONICS + 1):
        columns += [np.sin(k * angle), np.cos(k * angle)]
    return np.column_stack(columns)


def forecast_series(months, values, horizon, limits=(None, None)):
    """
    Projects one monthly series into the future.

    Model (least squares):  y(t) = a + b*t + sum_k [ c_k*sin(2*pi*k*m/12) + d_k*cos(2*pi*k*m/12) ]
    where t = months elapsed and m = calendar month. The first part is the long-term trend,
    the second one is the yearly climate cycle. The range is a ~95 % prediction interval that
    widens the further away from the data we go.
    Returns (labels, mean, lower, upper, trend_per_year) or None if there is not enough history.
    """
    periods = pd.PeriodIndex(months, freq="M")
    y = np.asarray(values, dtype=float)
    n = len(y)
    n_params = 2 + 2 * FORECAST_HARMONICS
    if n < max(FORECAST_MIN_RECORDS, n_params + 3):
        return None

    t = np.array([p.ordinal - periods[0].ordinal for p in periods])
    X = _design_matrix(t, [p.month for p in periods])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ coef
    sigma = float(np.sqrt(np.sum(residuals ** 2) / (n - n_params)))
    xtx_inv = np.linalg.pinv(X.T @ X)

    future_periods = [periods[-1] + i for i in range(1, horizon + 1)]
    t_future = np.array([p.ordinal - periods[0].ordinal for p in future_periods])
    X_future = _design_matrix(t_future, [p.month for p in future_periods])
    mean = X_future @ coef
    # prediction standard error = noise of the data + uncertainty of the fitted coefficients
    se = sigma * np.sqrt(1.0 + np.einsum("ij,jk,ik->i", X_future, xtx_inv, X_future))
    lower, upper = mean - T_CRITICAL * se, mean + T_CRITICAL * se

    low_limit, high_limit = limits
    if low_limit is not None or high_limit is not None:
        mean, lower, upper = (np.clip(a, low_limit, high_limit) for a in (mean, lower, upper))

    labels = [str(p) for p in future_periods]
    return labels, mean.tolist(), lower.tolist(), upper.tolist(), float(coef[1] * 12.0)


def build_future_figure(df_st, station_name, horizon):
    """History + projection (dashed line and shaded 95 % range) for the 6 variables, plus a summary table."""
    titles = ["%s (%s)" % (info["name"], info["unit"]) for info in VARIABLES_CONFIG.values()]
    fig = make_subplots(rows=3, cols=2, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.09)
    base_color = color_for(station_name)
    months = df_st["Date"].tolist()
    summary = []

    for idx, (code, info) in enumerate(VARIABLES_CONFIG.items()):
        row, col = idx // 2 + 1, idx % 2 + 1
        history = df_st[code].astype(float).tolist()
        result = forecast_series(months, history, horizon, VARIABLE_LIMITS.get(code, (None, None)))
        first = (idx == 0)

        fig.add_trace(go.Scatter(x=months, y=history, mode="lines+markers", name="History (NASA)",
                                 line=dict(color=base_color, width=2), marker=dict(size=4),
                                 legendgroup="history", showlegend=first), row=row, col=col)
        if result is None:
            continue
        labels, mean, lower, upper, trend_per_year = result

        # every projection trace starts at the last real point so the lines are connected
        x_f = [months[-1]] + labels
        fig.add_trace(go.Scatter(x=x_f, y=[history[-1]] + upper, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False, legendgroup="range"), row=row, col=col)
        fig.add_trace(go.Scatter(x=x_f, y=[history[-1]] + lower, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=hex_to_rgba(base_color, 0.22),
                                 name="95 % range", legendgroup="range", showlegend=first,
                                 hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(x=x_f, y=[history[-1]] + mean, mode="lines+markers", name="Projection",
                                 line=dict(color="#f9fafb", width=2, dash="dash"), marker=dict(size=4),
                                 legendgroup="projection", showlegend=first), row=row, col=col)

        recent = float(np.mean(history[-12:]))
        projected = float(np.mean(mean[:12]))
        summary.append(["%s (%s)" % (info["name"], info["unit"]), round(recent, 2), round(projected, 2),
                        "%+.2f" % (projected - recent), "%+.2f" % trend_per_year])

        fig.update_layout(height=800, template="plotly_dark", paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                      hovermode="x unified", hoverlabel=dict(bgcolor="#151b2e", bordercolor="#4f46e5", font=dict(color="#f9fafb")),
                      margin=dict(t=90),
                      title_text="The future of %s (next %d months)" % (station_name, horizon),
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig, summary



def future_payload(name, horizon):
    """Projection of one reserve location, ready for the page. None if the location does not exist."""
    with reserve_lock:
        if name not in RESERVE:
            return None
        df_st = entry_to_df(RESERVE[name])
    fig, summary = build_future_figure(df_st, name, horizon)
    return {"name": name, "horizon": horizon, "summary": summary,
            "figure": json.loads(fig.to_json()), "available": bool(summary)}


def location_payload(name):
    """Everything the page needs to display one location from the reserve."""
    with reserve_lock:
        entry = RESERVE[name]
        df_st = entry_to_df(entry)
        info = {"lat": entry["lat"], "lon": entry["lon"], "downloaded": entry.get("downloaded", "?"),
                "source": entry.get("source", "NASA POWER")}
    return {
        "name": name,
        "lat": info["lat"],
        "lon": info["lon"],
        "source": "%s (downloaded on %s)" % (info["source"], info["downloaded"]),
        "period": "%s to %s" % (df_st["Date"].min(), df_st["Date"].max()),
        "records": int(len(df_st)),
        "stats": station_stats(df_st),
        "figure": json.loads(build_station_figure(df_st, name).to_json()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. WEB PAGE
# ──────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mangarita</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
         background: #0b1020; color: #e5e7eb; }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .hero { text-align: center; margin-top: 8px; }
  .hero h1 { font-size: 3.2rem; margin: 0; letter-spacing: 0.06em; color: #f9fafb; }
  .cat { width: 130px; height: 130px; display: block; margin: 8px auto 0; overflow: visible; }
  .eye { transform-box: fill-box; transform-origin: center; animation: blink 5s infinite; transition: transform .5s ease; }
  @keyframes blink { 0%, 94%, 100% { transform: scaleY(1); } 97% { transform: scaleY(0.1); } }
  .subtitle { color: #9ca3af; margin: 14px 0 4px; }
  .home-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 24px; }
  .big-btn { padding: 36px 20px; font-size: 1.25rem; border: none; border-radius: 12px;
             background: #4f46e5; color: white; cursor: pointer; width: 100%; }
  .big-btn:hover { background: #6366f1; }
  .home-caption { color: #9ca3af; margin: 8px 4px 0; font-size: 0.95rem; }
  .back-btn { border: 1px solid #2a3350; background: #151b2e; color: #e5e7eb; border-radius: 8px;
              padding: 8px 14px; cursor: pointer; margin-bottom: 16px; }
  .back-btn:hover { background: #1c2440; }
  .banner, .msg { padding: 12px 16px; border-radius: 8px; margin: 12px 0; }
  .ok { background: #064e3b; color: #a7f3d0; }
  .warn { background: #78350f; color: #fde68a; }
  .error { background: #7f1d1d; color: #fecaca; }
  .info { background: #1e3a8a; color: #bfdbfe; }
  form.search { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end; margin: 16px 0; }
  form.search label { display: flex; flex-direction: column; font-size: 0.9rem; gap: 4px; color: #d1d5db; }
  form.search input { padding: 8px 10px; border: 1px solid #2a3350; border-radius: 8px; background: #151b2e;
                      color: #f9fafb; font-size: 1rem; width: 180px; }
  form.search button { padding: 10px 24px; border: none; border-radius: 8px; background: #4f46e5;
                       color: white; font-size: 1rem; cursor: pointer; }
  form.search button:hover { background: #6366f1; }
  form.search button:disabled { background: #3730a3; color: #a5b4fc; cursor: wait; }
  .result-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 24px; align-items: start; }
  .card { background: #151b2e; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.5); }
  table.stats { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 12px; }
  table.stats th, table.stats td { border-bottom: 1px solid #2a3350; padding: 6px 4px; text-align: left; }
  .future-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin: 24px 0 8px; }
  .future-btn { padding: 14px 28px; font-size: 1.1rem; border: none; border-radius: 10px;
                background: #7c3aed; color: white; cursor: pointer; }
  .future-btn:hover { background: #8b5cf6; }
  .future-btn:disabled { background: #4c1d95; color: #c4b5fd; cursor: wait; }
  .future-bar label { color: #d1d5db; font-size: 0.9rem; display: flex; align-items: center; gap: 8px; }
  .future-bar select { padding: 8px 10px; border: 1px solid #2a3350; border-radius: 8px;
                       background: #151b2e; color: #f9fafb; font-size: 1rem; }
  .note { color: #9ca3af; font-size: 0.85rem; margin: 12px 4px 0; }
  .predict-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin-bottom: 12px; }
  .predict-btn-on { background: #475569; }
  .predict-btn-on:hover { background: #64748b; }
  /* ── Mangarita reacts to the weather ── */
  .cat #cat-all { transform-origin: 100px 130px; }
  .cat .layer { opacity: 0; transition: opacity .6s ease; pointer-events: none; }
  .cat.cozy #layer-cold, .cat.st-hot #layer-hot-body, .cat.st-hot #layer-hot-head,
  .cat.asleep #layer-zzz { opacity: 1; }
  /* cold: shiver for 1.5 s, then blanket + book, reading, and finally sleeping */
  .cat.shivering #cat-all { animation: shiver 1.5s ease-in-out 1; }
  @keyframes shiver {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    5%, 15%, 25%, 35%, 45%, 55%, 65%, 75%, 85%, 95% { transform: translate(-2px, 0) rotate(-1.3deg); }
    10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% { transform: translate(2px, -1px) rotate(1.3deg); }
  }
  .cat #cat-head { transform-origin: 100px 140px; transition: transform .8s ease; }
  .cat.reading #cat-head { transform: rotate(4deg) translateY(3px); }
  .cat.asleep #cat-head { transform: rotate(8deg) translateY(5px); }
  .cat.reading #eyes { animation: read 1.7s ease-in-out 3; }
  @keyframes read {
    0%, 100% { transform: translate(-4px, 3px); }
    50% { transform: translate(4px, 3px); }
  }
  .cat.asleep .eye { animation: none; transform: scaleY(0.08); }
  .cat #lids { opacity: 0; transition: opacity .4s ease; }
  .cat.asleep #lids { opacity: 1; }
  .zz { opacity: 0; fill: #93c5fd; font: 700 16px system-ui, sans-serif; }
  .zz2 { font-size: 20px; }
  .zz3 { font-size: 24px; }
  .cat.asleep .zz { animation: floatz 3.2s ease-in-out infinite; }
  .cat.asleep .zz2 { animation-delay: 1s; }
  .cat.asleep .zz3 { animation-delay: 2s; }
  @keyframes floatz {
    0% { opacity: 0; transform: translate(0, 0); }
    25% { opacity: 1; }
    100% { opacity: 0; transform: translate(10px, -26px); }
  }
  .wx-tag { margin: 6px auto 0; font-size: 0.8rem; color: #9ca3af; user-select: none; }
  .wx-tag .wx-mood { display: block; color: #c4b5fd; font-size: 0.78rem; margin-top: 2px; }
  .wx-sim { display: inline-flex; align-items: center; gap: 6px; margin-top: 6px; font-size: 0.7rem;
            color: #6b7280; opacity: 0.55; transition: opacity .2s; }
  .wx-sim:hover, .wx-sim:focus-within { opacity: 1; }
  .wx-sim select { background: #151b2e; color: #9ca3af; border: 1px solid #2a3350; border-radius: 6px;
                   font-size: 0.72rem; padding: 2px 4px; }
  @media (prefers-reduced-motion: reduce) {
    .cat.shivering #cat-all, .cat.reading #eyes, .cat.asleep .zz { animation-duration: .01s; }
  }
  @media (max-width: 900px) { .home-buttons, .result-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">

  <section id="view-home">
    <div class="hero">
      <h1>Mangarita</h1>
      <svg id="mangarita" class="cat" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Mangarita, a cute drawn cat that reacts to the weather">
        <g id="cat-all">
        <!-- tail -->
        <path d="M146 168 C186 168 192 120 168 112" fill="none" stroke="#f59e0b" stroke-width="14" stroke-linecap="round"/>
        <path d="M168 112 C164 108 162 106 160 104" fill="none" stroke="#b45309" stroke-width="14" stroke-linecap="round"/>
        <!-- body -->
        <ellipse cx="100" cy="150" rx="48" ry="40" fill="#f59e0b"/>
        <ellipse cx="100" cy="162" rx="20" ry="24" fill="#fde68a"/>
        <!-- front paws -->
        <ellipse cx="82" cy="186" rx="14" ry="9" fill="#fbbf24"/>
        <ellipse cx="118" cy="186" rx="14" ry="9" fill="#fbbf24"/>
        <!-- COLD layer: cozy blanket + open book (fades in when she is cold) -->
        <g id="layer-cold" class="layer">
          <path d="M50 140 Q100 124 150 140 L158 196 L42 196 Z" fill="#6366f1"/>
          <path d="M50 140 Q100 124 150 140 L152 153 Q100 139 48 153 Z" fill="#818cf8"/>
          <g stroke="#a5b4fc" stroke-width="3" opacity="0.65" stroke-linecap="round">
            <line x1="64" y1="158" x2="60" y2="194"/><line x1="82" y1="158" x2="80" y2="194"/>
            <line x1="100" y1="158" x2="100" y2="194"/><line x1="118" y1="158" x2="120" y2="194"/>
            <line x1="136" y1="158" x2="140" y2="194"/>
            <line x1="50" y1="168" x2="150" y2="168"/><line x1="46" y1="182" x2="154" y2="182"/>
          </g>
          <path d="M58 158 Q80 150 100 158 Q120 150 142 158 L142 184 Q120 176 100 184 Q80 176 58 184 Z" fill="#7c2d12"/>
          <path d="M62 156 Q81 149 100 156 L100 180 Q81 173 62 179 Z" fill="#fef3c7"/>
          <path d="M138 156 Q119 149 100 156 L100 180 Q119 173 138 179 Z" fill="#fffbeb"/>
          <g stroke="#a8a29e" stroke-width="1.5" stroke-linecap="round" fill="none">
            <path d="M67 162 Q80 158 94 162"/><path d="M67 168 Q80 164 94 168"/><path d="M67 174 Q80 170 94 174"/>
            <path d="M106 162 Q120 158 133 162"/><path d="M106 168 Q120 164 133 168"/><path d="M106 174 Q120 170 133 174"/>
          </g>
          <line x1="100" y1="156" x2="100" y2="180" stroke="#d6d3d1" stroke-width="1.5"/>
          <ellipse cx="64" cy="178" rx="10" ry="7" fill="#fbbf24"/>
          <ellipse cx="136" cy="178" rx="10" ry="7" fill="#fbbf24"/>
        </g>
        <!-- HOT layer (body): colorful bikini -->
        <g id="layer-hot-body" class="layer">
          <path d="M72 148 Q60 150 56 154 M128 148 Q140 150 144 154 M97 148 L103 148 M70 176 Q62 174 58 170 M130 176 Q138 174 142 170"
                stroke="#fbbf24" stroke-width="2.5" fill="none" stroke-linecap="round"/>
          <path d="M72 148 L97 148 L84 166 Z" fill="#ec4899"/>
          <path d="M128 148 L103 148 L116 166 Z" fill="#ec4899"/>
          <circle cx="84" cy="153" r="2" fill="#ffffff"/><circle cx="116" cy="153" r="2" fill="#ffffff"/>
          <path d="M70 176 Q100 186 130 176 L114 194 Q100 198 86 194 Z" fill="#ec4899"/>
          <circle cx="90" cy="184" r="2" fill="#ffffff"/><circle cx="110" cy="184" r="2" fill="#ffffff"/><circle cx="100" cy="190" r="2" fill="#ffffff"/>
        </g>
        <g id="cat-head">
        <!-- ears -->
        <polygon points="56,70 58,20 92,52" fill="#f59e0b"/>
        <polygon points="144,70 142,20 108,52" fill="#f59e0b"/>
        <polygon points="63,60 64,36 82,52" fill="#f9a8d4"/>
        <polygon points="137,60 136,36 118,52" fill="#f9a8d4"/>
        <!-- head -->
        <ellipse cx="100" cy="96" rx="52" ry="45" fill="#f59e0b"/>
        <!-- forehead stripes -->
        <path d="M100 52 L100 68 M87 55 L90 67 M113 55 L110 67" stroke="#b45309" stroke-width="4" stroke-linecap="round" fill="none"/>
        <!-- cheeks -->
        <ellipse cx="66" cy="110" rx="9" ry="6" fill="#f9a8d4" opacity="0.55"/>
        <ellipse cx="134" cy="110" rx="9" ry="6" fill="#f9a8d4" opacity="0.55"/>
        <!-- eyes -->
        <g id="eyes">
        <g class="eye"><ellipse cx="78" cy="94" rx="8" ry="10" fill="#111827"/><circle cx="81" cy="90" r="3" fill="#ffffff"/></g>
        <g class="eye"><ellipse cx="122" cy="94" rx="8" ry="10" fill="#111827"/><circle cx="125" cy="90" r="3" fill="#ffffff"/></g>
        </g>
        <!-- closed eyelids (only visible while she sleeps) -->
        <g id="lids" fill="none" stroke="#111827" stroke-width="3" stroke-linecap="round">
          <path d="M69 95 Q78 103 87 95"/><path d="M113 95 Q122 103 131 95"/>
        </g>
        <!-- nose and mouth -->
        <polygon points="94,106 106,106 100,113" fill="#f472b6"/>
        <path d="M100 113 L100 119 M100 119 Q93 126 86 120 M100 119 Q107 126 114 120" stroke="#111827" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <!-- whiskers -->
        <g stroke="#e5e7eb" stroke-width="2" stroke-linecap="round">
          <line x1="68" y1="110" x2="34" y2="102"/><line x1="68" y1="116" x2="32" y2="117"/><line x1="70" y1="121" x2="36" y2="131"/>
          <line x1="132" y1="110" x2="166" y2="102"/><line x1="132" y1="116" x2="168" y2="117"/><line x1="130" y1="121" x2="164" y2="131"/>
        </g>
        <!-- HOT layer (head): diving goggles + snorkel tube -->
        <g id="layer-hot-head" class="layer">
          <path d="M64 92 Q52 90 49 96 M136 92 Q148 90 151 96" stroke="#0369a1" stroke-width="5" fill="none" stroke-linecap="round"/>
          <circle cx="78" cy="94" r="15" fill="#7dd3fc" fill-opacity="0.35" stroke="#0ea5e9" stroke-width="4.5"/>
          <circle cx="122" cy="94" r="15" fill="#7dd3fc" fill-opacity="0.35" stroke="#0ea5e9" stroke-width="4.5"/>
          <path d="M93 92 Q100 88 107 92" stroke="#0ea5e9" stroke-width="4.5" fill="none" stroke-linecap="round"/>
          <path d="M68 87 Q71 83 76 82 M112 87 Q115 83 120 82" stroke="#ffffff" stroke-width="2.5" fill="none" stroke-linecap="round" opacity="0.8"/>
          <path d="M152 90 L158 80 L158 34 Q158 24 170 24" stroke="#f97316" stroke-width="7" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M158 64 L158 50" stroke="#fde047" stroke-width="7"/>
          <circle cx="152" cy="90" r="5" fill="#0369a1"/>
        </g>
        </g>
        <!-- COLD layer: "Zzz" floating up while she sleeps -->
        <g id="layer-zzz" class="layer">
          <text class="zz zz1" x="150" y="62">Z</text>
          <text class="zz zz2" x="160" y="44">Z</text>
          <text class="zz zz3" x="170" y="28">z</text>
        </g>
        </g>
      </svg>
      <div class="wx-tag" id="wx-tag" title="Double-click to cycle the demo weather">
        <span id="wx-line">Ipiales, Colombia: --.- &deg;C</span>
        <span id="wx-mood" class="wx-mood"></span>
      </div>
      <label class="wx-sim"><span id="wx-sim-label">Simulate weather</span> <select id="wx-sim"></select></label>
      <p class="subtitle"><b id="reserve-count">%%RESERVE_COUNT%%</b> locations saved.</p>
    </div>
    <p style="text-align:center">Choose an option to get started:</p>
    <div class="home-buttons">
      <div>
        <button class="big-btn" onclick="show('dashboard')">📊 View charts and full analysis</button>
        <p class="home-caption">Compare temperature, humidity, wind, pressure, rainfall and solar radiation across the featured stations.</p>
      </div>
      <div>
        <button class="big-btn" onclick="show('search')">🔎 Search by coordinates</button>
        <p class="home-caption">Enter a latitude and longitude. If it is not saved yet, it is looked up online and added.</p>
      </div>
    </div>
  </section>

  <section id="view-dashboard" style="display:none">
    <button class="back-btn" onclick="show('home')">⬅ Back to home</button>
    <div id="dash-banner" class="banner"></div>
    <div class="card"><div id="dash-chart"></div></div>
  </section>

  <section id="view-search" style="display:none">
    <button class="back-btn" onclick="show('home')">⬅ Back to home</button>
    <h2>🔎 Search by coordinates</h2>
    <p style="color:#9ca3af">Latitude between -90 and 90 · Longitude between -180 and 180.</p>
    <form class="search" onsubmit="runSearch(); return false;">
      <label>Latitude <input id="lat" type="text" value="4.6097" inputmode="decimal"></label>
      <label>Longitude <input id="lon" type="text" value="-74.0817" inputmode="decimal"></label>
      <button id="search-btn" type="submit">Search</button>
    </form>
    <div id="msg"></div>
    <div id="result" class="result-grid" style="display:none">
      <div class="card" id="info"></div>
      <div class="card">
        <div class="predict-bar">
          <button id="predict-btn" class="future-btn" onclick="predictFuture()">🔮 Predict future years (2025–2027)</button>
        </div>
        <div id="station-chart"></div>
        <div id="predict-note" style="display:none"></div>
      </div>
    </div>
    <div id="future-box" style="display:none">
      <div class="future-bar">
        <button id="future-btn" class="future-btn" onclick="runFuture()">🔮 See the future</button>
        <label>Look ahead
          <select id="future-months" onchange="futureHorizonChanged()">
            <option value="12">12 months</option>
            <option value="24" selected>24 months</option>
            <option value="36">36 months</option>
            <option value="60">60 months</option>
          </select>
        </label>
      </div>
      <div id="future-msg"></div>
      <div id="future-result" style="display:none">
        <div class="card"><div id="future-chart"></div></div>
        <div class="card" style="margin-top:16px">
          <h3 style="margin-top:0">📈 What is expected to change</h3>
          <table class="stats" id="future-table"></table>
          <p class="note">Average of the last 12 recorded months vs. the average of the first 12 projected months.
            "Trend / year" is the long-term slope found in the data.</p>
        </div>
        <p class="note">⚠️ This is a mathematical projection (linear trend + yearly cycle fitted with least squares
          on the saved NASA data), not a weather or climate forecast. The shaded band is an approximate 95 % range:
          the further into the future, the wider it gets, and with only a few years of history long-term trends
          can be misleading.</p>
      </div>
    </div>
  </section>

</div>

<script type="application/json" id="data">%%DATA%%</script>
<script src="/plotly.js"></script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
let dashboardDrawn = false;
let currentLocation = null;
let futureShown = false;
let lastLoc = null;          // location currently drawn in #station-chart
let baseData = null;         // its original traces (history only)
let baseLayoutJson = null;   // its original layout, as text, to redraw without side effects
let projectionShown = false;
let predictLabel = '';
const PREDICT_MONTHS = 36;
const PREDICT_COLOR = '#facc15';

function esc(text) {
  return String(text).replace(/[&<>"']/g, function (c) {
    return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
  });
}

function show(view) {
  ['home', 'dashboard', 'search'].forEach(function (v) {
    document.getElementById('view-' + v).style.display = (v === view) ? 'block' : 'none';
  });
  if (view === 'dashboard') drawDashboard();
  window.scrollTo(0, 0);
}

async function drawDashboard() {
  // ask the server for the current dashboard every time the view opens (it changes with each search)
  try {
    const response = await fetch('/api/dashboard', {cache: 'no-store'});
    if (!response.ok) throw new Error('bad status');
    const fresh = await response.json();
    D.banner = fresh.banner;
    D.banner_class = fresh.banner_class;
    D.dashboard = fresh.dashboard;
    if (fresh.reserve_count !== undefined) {
      document.getElementById('reserve-count').textContent = fresh.reserve_count;
    }
  } catch (error) {
    // no answer from the program: keep drawing the last data we had
  }
  const banner = document.getElementById('dash-banner');
  banner.className = 'banner ' + D.banner_class;
  banner.textContent = D.banner;
  if (!D.dashboard) {
    if (dashboardDrawn) { Plotly.purge('dash-chart'); dashboardDrawn = false; }
    return;
  }
  Plotly.newPlot('dash-chart', D.dashboard.data, D.dashboard.layout, {responsive: true});
  dashboardDrawn = true;
}

function setMessage(kind, text) {
  document.getElementById('msg').innerHTML = '<div class="msg ' + kind + '">' + esc(text) + '</div>';
}

function resetFuture() {
  currentLocation = null;
  futureShown = false;
  document.getElementById('future-box').style.display = 'none';
  document.getElementById('future-result').style.display = 'none';
  document.getElementById('future-msg').innerHTML = '';
  Plotly.purge('future-chart');
}

function futureHorizonChanged() {
  if (futureShown) runFuture();
}

async function runFuture() {
  if (!currentLocation) return;
  const months = document.getElementById('future-months').value;
  const button = document.getElementById('future-btn');
  const msg = document.getElementById('future-msg');
  button.disabled = true;
  msg.innerHTML = '<div class="msg info">Calculating the projection...</div>';
  try {
    const response = await fetch('/api/future?name=' + encodeURIComponent(currentLocation) + '&months=' + encodeURIComponent(months));
    const data = await response.json();
    if (!response.ok || data.status === 'error') {
      msg.innerHTML = '<div class="msg error">' + esc(data.message || 'The projection could not be calculated.') + '</div>';
      return;
    }
    msg.innerHTML = '';
    let rows = '<tr><th>Variable</th><th>Last 12 months</th><th>Next 12 months</th><th>Change</th><th>Trend / year</th></tr>';
    data.future.summary.forEach(function (r) {
      rows += '<tr><td>' + esc(r[0]) + '</td><td>' + r[1] + '</td><td>' + r[2] + '</td><td>' + r[3] + '</td><td>' + r[4] + '</td></tr>';
    });
    document.getElementById('future-table').innerHTML = rows;
    document.getElementById('future-result').style.display = 'block';
    Plotly.react('future-chart', data.future.figure.data, data.future.figure.layout, {responsive: true});
    futureShown = true;
  } catch (error) {
    msg.innerHTML = '<div class="msg error">Could not reach the program. Is the terminal still open?</div>';
  } finally {
    button.disabled = false;
  }
}

// ── "Predict future years": 100 % in the browser, no internet and no server call ──
function monthIndex(label) {            // '2024-03' -> months since year 0
  const parts = label.split('-');
  return parseInt(parts[0], 10) * 12 + parseInt(parts[1], 10) - 1;
}

function monthLabel(index) {            // inverse of monthIndex
  const m = (index % 12) + 1;
  return Math.floor(index / 12) + '-' + (m < 10 ? '0' + m : m);
}

function projectSeries(months, values, horizon, limits) {
  /* value(t) = slope * t + intercept  +  average anomaly of that calendar month
     slope/intercept: least-squares line; anomaly of month m (1..12): mean of (real value - line)
     over every historical month m. The line is fitted on the data WITHOUT its seasonal part
     (a few repetitions of line -> anomalies -> line), otherwise the strong yearly cycle of the
     climate would tilt the trend by itself. */
  const n = values.length;
  if (n < 24) return null;
  const t = months.map(monthIndex);
  let meanT = 0;
  for (let i = 0; i < n; i++) meanT += t[i];
  meanT /= n;
  let sxx = 0;
  for (let i = 0; i < n; i++) sxx += (t[i] - meanT) * (t[i] - meanT);

  let anomaly = new Array(12).fill(0), slope = 0, intercept = 0;
  for (let iter = 0; iter < 30; iter++) {
    let meanY = 0;
    for (let i = 0; i < n; i++) meanY += values[i] - anomaly[t[i] % 12];
    meanY /= n;
    let sxy = 0;
    for (let i = 0; i < n; i++) sxy += (t[i] - meanT) * (values[i] - anomaly[t[i] % 12] - meanY);
    slope = sxy / sxx;
    intercept = meanY - slope * meanT;

    const sums = new Array(12).fill(0), counts = new Array(12).fill(0);
    for (let i = 0; i < n; i++) {
      sums[t[i] % 12] += values[i] - (slope * t[i] + intercept);
      counts[t[i] % 12] += 1;
    }
    anomaly = sums.map(function (sum, k) { return counts[k] ? sum / counts[k] : 0; });
  }

  const labels = [], projected = [];
  for (let h = 1; h <= horizon; h++) {
    const tt = t[n - 1] + h;
    let v = slope * tt + intercept + anomaly[tt % 12];
    if (limits) {                         // physical limits (e.g. humidity 0-100 %)
      if (limits[0] !== null && v < limits[0]) v = limits[0];
      if (limits[1] !== null && v > limits[1]) v = limits[1];
    }
    labels.push(monthLabel(tt));
    projected.push(Math.round(v * 100) / 100);
  }
  return {labels: labels, projected: projected, slopePerYear: slope * 12};
}

function predictRangeText(months) {
  const last = monthIndex(months[months.length - 1]);
  const first = monthLabel(last + 1).slice(0, 4), end = monthLabel(last + PREDICT_MONTHS).slice(0, 4);
  return first === end ? first : first + '–' + end;
}

function resetPredict() {
  lastLoc = null; baseData = null; baseLayoutJson = null; projectionShown = false;
  const button = document.getElementById('predict-btn');
  button.classList.remove('predict-btn-on');
  button.textContent = '🔮 Predict future years (2025–2027)';
  document.getElementById('predict-note').style.display = 'none';
  document.getElementById('predict-note').innerHTML = '';
}

function predictFuture() {
  if (!lastLoc || !baseData) return;
  const button = document.getElementById('predict-btn');
  const note = document.getElementById('predict-note');

  if (projectionShown) {                  // second click: back to history only
    Plotly.react('station-chart', baseData, JSON.parse(baseLayoutJson), {responsive: true});
    note.style.display = 'none';
    button.classList.remove('predict-btn-on');
    button.textContent = predictLabel;
    projectionShown = false;
    return;
  }

  const results = [], traces = [];
  for (let i = 0; i < baseData.length; i++) {
    const src = baseData[i];
    const res = projectSeries(src.x, src.y, PREDICT_MONTHS, D.limits ? D.limits[i] : null);
    if (!res) {
      note.style.display = 'block';
      note.innerHTML = '<div class="msg error">At least 24 months of history are needed to project this location.</div>';
      return;
    }
    results.push(res);
    traces.push({
      type: 'scatter', mode: 'lines+markers', name: 'Projection (' + res.labels[0] + ' to ' + res.labels[res.labels.length - 1] + ')',
      x: [src.x[src.x.length - 1]].concat(res.labels),     // starts at the last real point
      y: [src.y[src.y.length - 1]].concat(res.projected),
      line: {color: PREDICT_COLOR, width: 2, dash: 'dot'}, marker: {size: 4, color: PREDICT_COLOR},
      xaxis: src.xaxis, yaxis: src.yaxis, showlegend: i === 0
    });
  }
  Plotly.react('station-chart', baseData.concat(traces), JSON.parse(baseLayoutJson), {responsive: true});

  // summary table: historical average, trend and the average of each projected calendar year
  const labels = results[0].labels;
  const years = [], monthsPerYear = {};
  labels.forEach(function (l) {
    const y = l.slice(0, 4);
    if (!(y in monthsPerYear)) { monthsPerYear[y] = 0; years.push(y); }
    monthsPerYear[y] += 1;
  });
  let head = '<tr><th>Variable</th><th>History avg</th><th>Trend / year</th>';
  years.forEach(function (y) { head += '<th>' + y + (monthsPerYear[y] < 12 ? ' (' + monthsPerYear[y] + ' mo.)' : '') + '</th>'; });
  let rows = head + '</tr>';
  results.forEach(function (res, i) {
    const hist = baseData[i].y;
    const histAvg = hist.reduce(function (a, b) { return a + b; }, 0) / hist.length;
    let row = '<tr><td>' + esc(lastLoc.stats[i][0]) + '</td><td>' + histAvg.toFixed(2) + '</td><td>' +
              (res.slopePerYear >= 0 ? '+' : '') + res.slopePerYear.toFixed(2) + '</td>';
    years.forEach(function (y) {
      let sum = 0, count = 0;
      res.labels.forEach(function (l, k) { if (l.slice(0, 4) === y) { sum += res.projected[k]; count++; } });
      row += '<td>' + (sum / count).toFixed(2) + '</td>';
    });
    rows += row + '</tr>';
  });
  const months = baseData[0].x;
  note.innerHTML =
    '<table class="stats">' + rows + '</table>' +
    '<p class="note">⚠️ Mathematical projection, not a weather forecast. Each variable = linear trend (least squares, ' +
    'slope · t + intercept, fitted without the seasonal cycle) + the average anomaly of each calendar month (1–12) in the saved history (' +
    esc(months[0]) + ' to ' + esc(months[months.length - 1]) + '). It was calculated in your browser, without internet. ' +
    'With so few years of data, long-term trends can be misleading.</p>';
  note.style.display = 'block';
  button.classList.add('predict-btn-on');
  button.textContent = '🙈 Hide projection';
  projectionShown = true;
}

async function runSearch() {
  const lat = parseFloat(document.getElementById('lat').value.replace(',', '.'));
  const lon = parseFloat(document.getElementById('lon').value.replace(',', '.'));
  const result = document.getElementById('result');
  const button = document.getElementById('search-btn');

  if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    setMessage('error', 'Enter a latitude between -90 and 90 and a longitude between -180 and 180.');
    result.style.display = 'none';
    return;
  }

  button.disabled = true;
  resetFuture();
  resetPredict();
  setMessage('info', 'Searching the reserve and, if needed, asking NASA...');
  try {
    const response = await fetch('/api/search?lat=' + encodeURIComponent(lat) + '&lon=' + encodeURIComponent(lon));
    const data = await response.json();
    if (!response.ok || data.status === 'error') {
      setMessage('error', data.message || 'The search could not be completed.');
      result.style.display = 'none';
      return;
    }
    setMessage(data.status === 'nearest' ? 'warn' : 'ok', data.message);

    const loc = data.location;
    let rows = '';
    loc.stats.forEach(function (r) {
      rows += '<tr><td>' + esc(r[0]) + '</td><td>' + r[1] + '</td><td>' + r[2] + '</td><td>' + r[3] + '</td></tr>';
    });
    document.getElementById('info').innerHTML =
      '<h3 style="margin-top:0">' + esc(loc.name) + '</h3>' +
      '<p>📍 Latitude: <b>' + loc.lat + '</b><br>📍 Longitude: <b>' + loc.lon + '</b></p>' +
      '<p>🗂️ Source: ' + esc(loc.source) + '<br>📅 Period: ' + esc(loc.period) + ' (' + loc.records + ' records)</p>' +
      '<table class="stats"><tr><th>Variable</th><th>Average</th><th>Min</th><th>Max</th></tr>' + rows + '</table>';
    result.style.display = 'grid';
    lastLoc = loc;
    baseData = loc.figure.data.slice();
    baseLayoutJson = JSON.stringify(loc.figure.layout);
    predictLabel = '🔮 Predict future years (' + predictRangeText(baseData[0].x) + ')';
    document.getElementById('predict-btn').textContent = predictLabel;
    Plotly.newPlot('station-chart', loc.figure.data, JSON.parse(baseLayoutJson), {responsive: true});
    currentLocation = loc.name;
    document.getElementById('future-box').style.display = 'block';
  } catch (error) {
    setMessage('error', 'Could not reach the program. Is the terminal still open?');
    result.style.display = 'none';
  } finally {
    button.disabled = false;
  }
}

// ── Mangarita reacts to the live weather of Ipiales, Colombia ──
const WX_LANG = 'en';   // 'en' or 'es': language of the cat's label and selector
const WX_TEXT = {
  en: {place: 'Ipiales, Colombia', live: 'live', backup: 'backup value', simulated: 'simulated',
       waiting: 'checking the weather...',
       cold: 'Mangarita is cold: blanket, book and a nap', mild: 'Mangarita is comfortable',
       hot: 'Mangarita is hot: bikini and snorkel time',
       simLabel: 'Simulate weather', modes: {live: 'Live', cold: 'Cold', mild: 'Mild', hot: 'Hot'}},
  es: {place: 'Ipiales, Colombia', live: 'en vivo', backup: 'valor de respaldo', simulated: 'simulado',
       waiting: 'consultando el clima...',
       cold: 'Mangarita tiene frío: mantita, libro y a dormir', mild: 'Mangarita está a gusto',
       hot: 'Mangarita tiene calor: bikini y snorkel',
       simLabel: 'Simular clima', modes: {live: 'En vivo', cold: 'Frío', mild: 'Templado', hot: 'Calor'}}
};
const WX = WX_TEXT[WX_LANG] || WX_TEXT.en;
const WX_COLD_BELOW = 10.0;      // below this: cold
const WX_HOT_ABOVE = 17.0;       // above this: hot (from 10.0 to 17.0 inclusive: mild)
const WX_POLL_MS = 60000;        // the page asks for the weather every 60 seconds
const WX_SIM_TEMPS = {cold: 6.0, mild: 13.0, hot: 22.0};
const WX_MODES = ['live', 'cold', 'mild', 'hot'];
const COZY_AFTER_MS = 1500;      // length of the shiver
const READING_MS = 5200;         // how long she reads before falling asleep
let wxMode = 'live';
let wxLive = null;               // last answer of /api/ipiales-weather: {temp, live}
let catState = null;
let catTimers = [];

function stateForTemp(temp) {
  if (temp < WX_COLD_BELOW) return 'cold';
  if (temp > WX_HOT_ABOVE) return 'hot';
  return 'mild';
}

function setCatState(state) {
  if (state === catState) return;        // the 60 s polling must not restart the animations
  const cat = document.getElementById('mangarita');
  if (!cat) return;
  catTimers.forEach(clearTimeout);
  catTimers = [];
  cat.classList.remove('st-cold', 'st-hot', 'shivering', 'cozy', 'reading', 'asleep');
  catState = state;
  if (state === 'hot') {
    cat.classList.add('st-hot');
  } else if (state === 'cold') {
    cat.classList.add('st-cold', 'shivering');
    catTimers.push(setTimeout(function () {
      cat.classList.remove('shivering');
      cat.classList.add('cozy', 'reading');
    }, COZY_AFTER_MS));
    catTimers.push(setTimeout(function () {
      cat.classList.remove('reading');
      cat.classList.add('asleep');
    }, COZY_AFTER_MS + READING_MS));
  }
}

function renderWeather() {
  const line = document.getElementById('wx-line');
  const mood = document.getElementById('wx-mood');
  if (!line || !mood) return;
  let temp, tag;
  if (wxMode !== 'live') {
    temp = WX_SIM_TEMPS[wxMode]; tag = WX.simulated;
  } else if (wxLive) {
    temp = wxLive.temp; tag = wxLive.live ? WX.live : WX.backup;
  } else {
    line.textContent = WX.place + ': ' + WX.waiting;
    mood.textContent = '';
    return;
  }
  const state = stateForTemp(temp);
  line.textContent = WX.place + ': ' + temp.toFixed(1) + ' °C (' + tag + ')';
  mood.textContent = WX[state];
  setCatState(state);
}

async function pollWeather() {
  try {
    const response = await fetch('/api/ipiales-weather', {cache: 'no-store'});
    if (!response.ok) throw new Error('bad status');
    const data = await response.json();
    if (typeof data.temp !== 'number' || !isFinite(data.temp)) throw new Error('bad data');
    wxLive = {temp: data.temp, live: !!data.live};
  } catch (error) {
    if (!wxLive) wxLive = {temp: 14.0, live: false};   // the page never breaks without the program or internet
  }
  renderWeather();
}

function initWeather() {
  const select = document.getElementById('wx-sim');
  if (!select) return;
  document.getElementById('wx-sim-label').textContent = WX.simLabel + ':';
  WX_MODES.forEach(function (mode) {
    const option = document.createElement('option');
    option.value = mode;
    option.textContent = WX.modes[mode];
    select.appendChild(option);
  });
  select.value = wxMode;
  select.addEventListener('change', function () { wxMode = select.value; renderWeather(); });
  // hidden shortcut for the demo: double-click the temperature to cycle Live -> Cold -> Mild -> Hot
  document.getElementById('wx-tag').addEventListener('dblclick', function () {
    wxMode = WX_MODES[(WX_MODES.indexOf(wxMode) + 1) % WX_MODES.length];
    select.value = wxMode;
    renderWeather();
  });
  renderWeather();
  pollWeather();
  setInterval(pollWeather, WX_POLL_MS);
}
initWeather();
</script>
</body>
</html>
"""


def build_dashboard_data():
    """Banner + dashboard figure for the locations currently in DASHBOARD_STATIONS (used by the page and /api/dashboard)."""
    with reserve_lock:
        names = [n for n in DASHBOARD_STATIONS if n in RESERVE]
        frames = {n: entry_to_df(RESERVE[n]) for n in names}
        count = len(RESERVE)
        dates = sorted(e.get("downloaded", "") for e in RESERVE.values())
        missing = [n for n in FEATURED if n not in RESERVE]

    if frames:
        if dates[0] == dates[-1]:
            banner = "Real NASA POWER data saved in the reserve (downloaded on %s)." % dates[0]
        else:
            banner = "Real NASA POWER data saved in the reserve (downloaded between %s and %s)." % (dates[0], dates[-1])
        banner += " Showing %d locations on the dashboard (maximum %d)." % (len(frames), DASHBOARD_MAX)
        banner_class = "ok"
        if missing:
            banner += " Missing featured stations: " + ", ".join(missing) + "."
            banner_class = "warn"
        dashboard = json.loads(build_dashboard_figure(frames).to_json())
    else:
        banner = "The reserve does not have the featured stations. Run the program with an internet connection to download them."
        banner_class, dashboard = "error", None

    return {"banner": banner, "banner_class": banner_class, "dashboard": dashboard, "reserve_count": count}


def build_page():
    data = build_dashboard_data()
    count = data.pop("reserve_count")

    # physical limits per variable (same order as the charts), used by the in-browser projection
    data["limits"] = [list(VARIABLE_LIMITS.get(code, (None, None))) for code in VARIABLES_CONFIG]
    data_json = json.dumps(data).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("%%RESERVE_COUNT%%", str(count)).replace("%%DATA%%", data_json)


# ──────────────────────────────────────────────────────────────────────────────
# 7. LOCAL SERVER
# ──────────────────────────────────────────────────────────────────────────────
def search_coordinate(lat, lon):
    """Search logic. Returns (HTTP code, response dict)."""
    name, distance, is_exact = nearest_location(lat, lon)

    if is_exact:
        dashboard_add(name)
        return 200, {"status": "exact", "location": location_payload(name),
                     "message": "Coordinate found in the reserve: %s." % name}

    entry = fetch_power(lat, lon, retries=0, timeout=SEARCH_TIMEOUT)
    if entry:
        new_name = new_location_name(lat, lon)
        with reserve_lock:
            RESERVE[new_name] = entry
            save_reserve()
            dashboard_add(new_name)
            total = len(RESERVE)
        return 200, {"status": "downloaded", "location": location_payload(new_name),
                     "message": "That coordinate was not in the reserve. It was downloaded from NASA and saved "
                                "(the reserve now has %d locations)." % total}

    if name is not None:
        with reserve_lock:
            dashboard_add(name)
            near = RESERVE[name]
            near_lat, near_lon = near["lat"], near["lon"]
        return 200, {"status": "nearest", "location": location_payload(name), "distance_km": distance,
                     "message": "There is no data for (%.4f, %.4f) and NASA could not be reached right now. "
                                "Showing the nearest location in the reserve: %s (%s, %s), %s km away."
                                % (lat, lon, name, near_lat, near_lon, format(round(distance), ","))}

    return 503, {"status": "error",
                 "message": "The reserve is empty and NASA could not be reached. Check your internet connection."}


_weather_lock = threading.Lock()
_weather_cache = {"time": 0.0, "payload": None}


def ipiales_weather():
    """
    Current temperature of Ipiales (Open-Meteo). It never raises: if there is no internet or the API
    fails, it returns the safe backup {"temp": 14.0, "live": False} so the page keeps working.
    """
    now = time.time()
    with _weather_lock:
        cached = _weather_cache["payload"]
        if cached is not None and now - _weather_cache["time"] < WEATHER_CACHE_SECONDS:
            return dict(cached)
    try:
        response = requests.get(
            WEATHER_URL,
            params={"latitude": IPIALES_LAT, "longitude": IPIALES_LON, "current": "temperature_2m",
                    "timezone": "auto"},
            headers=HEADERS, timeout=WEATHER_TIMEOUT,
        )
        response.raise_for_status()
        temp = float(response.json()["current"]["temperature_2m"])
        if not math.isfinite(temp):
            raise ValueError("invalid temperature")
        payload = {"temp": round(temp, 1), "live": True, "source": "Open-Meteo"}
        with _weather_lock:  # only real answers are cached; a failure is retried on the next request
            _weather_cache["time"] = time.time()
            _weather_cache["payload"] = payload
        return dict(payload)
    except Exception as error:  # no internet, timeout, API down, unexpected format...
        print("    (live weather unavailable: %s)" % type(error).__name__)
        return {"temp": WEATHER_FALLBACK_TEMP, "live": False, "source": "backup"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
        pass

    def _send(self, code, body, content_type):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code, payload):
        self._send(code, json.dumps(payload), "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(200, build_page(), "text/html; charset=utf-8")
            elif parsed.path == "/plotly.js":
                self._send(200, get_plotlyjs(), "application/javascript; charset=utf-8")
            elif parsed.path == "/api/dashboard":
                self._send_json(200, build_dashboard_data())
            elif parsed.path == "/api/ipiales-weather":
                self._send_json(200, ipiales_weather())
            elif parsed.path == "/api/search":
                query = parse_qs(parsed.query)
                try:
                    lat, lon = float(query["lat"][0]), float(query["lon"][0])
                    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                        raise ValueError
                except (KeyError, ValueError, IndexError):
                    self._send_json(400, {"status": "error", "message": "Invalid coordinates."})
                    return
                code, payload = search_coordinate(lat, lon)
                self._send_json(code, payload)
            elif parsed.path == "/api/future":
                query = parse_qs(parsed.query)
                try:
                    name = query["name"][0]
                    horizon = int(query.get("months", [FORECAST_DEFAULT_MONTHS])[0])
                    if not 1 <= horizon <= FORECAST_MAX_MONTHS:
                        raise ValueError
                except (KeyError, ValueError, IndexError):
                    self._send_json(400, {"status": "error", "message": "Invalid projection request."})
                    return
                payload = future_payload(name, horizon)
                if payload is None:
                    self._send_json(404, {"status": "error", "message": "That location is not in the reserve."})
                elif not payload["available"]:
                    self._send_json(200, {"status": "error", "message": "Not enough history (at least %d months "
                                          "are needed) to project this location." % FORECAST_MIN_RECORDS})
                else:
                    self._send_json(200, {"status": "ok", "future": payload})
            else:
                self._send(404, "Not found", "text/plain; charset=utf-8")
        except Exception as error:  # a single failure must not bring the server down
            print("Error handling %s: %r" % (self.path, error))
            self._send_json(500, {"status": "error", "message": "Internal program error."})


def run_server():
    in_cloud = "PORT" in os.environ   # Render / Railway / Hugging Face set PORT; a normal computer does not
    try:
        port = int(os.environ.get("PORT", PREFERRED_PORT))
    except ValueError:
        port = PREFERRED_PORT
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    except OSError:  # port in use: pick a free one (only on your own computer)
        if in_cloud:
            raise
        httpd = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = httpd.server_address[1]
    url = "http://127.0.0.1:%d/" % port
    if in_cloud:
        print("\nMangarita is ready and listening on port %d." % port)
    else:
        print("\nMangarita is ready. Open %s in your browser (it will open by itself)." % url)
        print("To quit, press Ctrl + C in this terminal.")
        try:
            webbrowser.open(url)
        except Exception:  # servers without a screen cannot open a browser
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nProgram closed.")
    finally:
        httpd.server_close()


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN PROGRAM
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Mangarita - NASA POWER climate explorer")
    parser.add_argument("--update", action="store_true", help="download the whole catalog again")
    parser.add_argument("--no-download", action="store_true", help="do not download the catalog on startup")
    args, _ = parser.parse_known_args()

    RESERVE.update(load_reserve())
    print("Current reserve: %d locations (%s)" % (len(RESERVE), RESERVE_FILE.name))

    init_dashboard()

    if not args.no_download:
        if RESERVE:
            # the reserve already has data: open the web server right away and update in the background
            threading.Thread(target=sync_catalog, kwargs={"force": args.update}, daemon=True).start()
        else:
            sync_catalog(force=args.update)

    if not RESERVE:
        print("\nThe reserve is empty and there was no connection to NASA.")
        print("Connect to the internet and run the program again: the first run downloads the real data.")
        return

    run_server()


if __name__ == "__main__":
    main()