"""
Download hourly temperature data from the Open-Meteo API.

The original TIPE study used a 7-day window (168 hourly points) in June,
exported to Excel. This script replaces that manual step so the dataset
is reproducible from a single command.

Usage:
    python src/fetch_data.py --start 2024-06-10 --end 2024-06-16 \
        --lat 33.70 --lon -7.40 --out data/temperature.csv
"""

import argparse
import csv
import urllib.request
import urllib.parse

API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat, lon, start, end, archive=True):
    """Return (times, temperatures) for the requested window."""
    base = ARCHIVE_URL if archive else API_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        import json

        payload = json.loads(resp.read().decode())

    hourly = payload["hourly"]
    return hourly["time"], hourly["temperature_2m"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=33.70, help="latitude")
    p.add_argument("--lon", type=float, default=-7.40, help="longitude")
    p.add_argument("--start", default="2024-06-10", help="YYYY-MM-DD")
    p.add_argument("--end", default="2024-06-16", help="YYYY-MM-DD")
    p.add_argument("--out", default="data/temperature.csv")
    p.add_argument(
        "--forecast",
        action="store_true",
        help="use the forecast endpoint instead of the historical archive",
    )
    args = p.parse_args()

    times, temps = fetch(
        args.lat, args.lon, args.start, args.end, archive=not args.forecast
    )

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "temperature"])
        w.writerows(zip(times, temps))

    print(f"Wrote {len(times)} hourly points to {args.out}")


if __name__ == "__main__":
    main()
