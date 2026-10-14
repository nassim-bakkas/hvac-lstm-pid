"""
Download hourly weather data from the Open-Meteo API.

The original TIPE study used a 7-day window (168 hourly points) exported to
Excel by hand. The historical archive goes back decades, so there is no reason
to stay at 7 days: a multi-year pull gives enough data for a genuine
train/validation/test split.

Examples
--------
    # 7-day window, as in the original study
    python src/fetch_data.py --start 2024-06-10 --end 2024-06-16 \
        --out data/temperature.csv

    # three years of hourly data, written to Excel as well
    python src/fetch_data.py --start 2023-01-01 --end 2025-12-31 \
        --out data/temperature_3y.csv --xlsx data/temperature_3y.xlsx

Open-Meteo is free for non-commercial use and needs no API key. The archive
endpoint lags real time by roughly five days.
"""

import argparse
import csv
import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# temperature_2m is what the study models. The others are the main physical
# drivers of HVAC load and are fetched so the model can later be extended from
# univariate to multivariate (see the Limitations section of the README).
DEFAULT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
]


def fetch(lat, lon, start, end, variables, archive=True, timeout=180):
    """Return a dict of {column_name: list_of_values}, including 'time'."""
    base = ARCHIVE_URL if archive else FORECAST_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"

    # The free tier rate-limits; back off and retry rather than failing the run.
    delay = 20
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 4:
                print(f"  rate-limited ({e.code}), retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # Long responses are sometimes truncated in transit; retry.
            if attempt < 4:
                print(f"  transfer failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    else:
        raise RuntimeError("Open-Meteo could not be reached")

    if "error" in payload:
        raise RuntimeError(payload.get("reason", "Open-Meteo returned an error"))

    return payload["hourly"]


def fetch_chunked(lat, lon, start, end, variables, archive=True, chunk_days=365):
    """Fetch a long window in chunks, which the free tier tolerates better."""
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)

    merged = None
    cursor = d0
    while cursor <= d1:
        stop = min(cursor + dt.timedelta(days=chunk_days - 1), d1)
        print(f"  fetching {cursor} to {stop}")
        part = fetch(lat, lon, cursor.isoformat(), stop.isoformat(),
                     variables, archive=archive)
        if merged is None:
            merged = {k: list(v) for k, v in part.items()}
        else:
            for k, v in part.items():
                merged[k].extend(v)
        cursor = stop + dt.timedelta(days=1)
        if cursor <= d1:
            time.sleep(8)  # stay inside the free-tier rate limit

    return merged


def to_rows(hourly, variables):
    """Flatten the API response into a header and row list."""
    header = ["time"] + [v.replace("_2m", "") for v in variables]
    columns = [hourly["time"]] + [hourly[v] for v in variables]
    return header, list(zip(*columns))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ENSET Mohammedia, Morocco. Change for any other site.
    p.add_argument("--lat", type=float, default=33.686)
    p.add_argument("--lon", type=float, default=-7.383)
    p.add_argument("--start", default="2023-01-01", help="YYYY-MM-DD")
    p.add_argument("--end", default="2025-12-31", help="YYYY-MM-DD")
    p.add_argument("--out", default="data/temperature.csv", help="CSV output path")
    p.add_argument("--xlsx", default=None, help="optional .xlsx output path")
    p.add_argument("--vars", nargs="+", default=DEFAULT_VARS,
                   help="Open-Meteo hourly variables to request")
    p.add_argument("--chunk-days", type=int, default=365,
                   help="split the request into chunks of this many days")
    p.add_argument("--forecast", action="store_true",
                   help="use the forecast endpoint instead of the archive")
    args = p.parse_args()

    hourly = fetch_chunked(args.lat, args.lon, args.start, args.end, args.vars,
                           archive=not args.forecast, chunk_days=args.chunk_days)
    header, rows = to_rows(hourly, args.vars)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Wrote {len(rows)} hourly rows to {args.out}")

    if args.xlsx:
        try:
            import pandas as pd
        except ImportError:
            raise SystemExit("pandas and openpyxl are needed for --xlsx")
        df = pd.DataFrame(rows, columns=header)
        df.to_excel(args.xlsx, index=False, sheet_name="hourly")
        print(f"Wrote {len(df)} rows to {args.xlsx}")


if __name__ == "__main__":
    main()
