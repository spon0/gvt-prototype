"""Download one GFS 0.25 deg forecast run from NOMADS as small single-variable GRIB2 files.

Uses the NOMADS GRIB filter, so each file is ~1-2 MB instead of ~500 MB.
NOMADS keeps roughly the last 10 days of runs.

    python fetch_gfs.py --preset pwat --hours 0-120:3 --out C:/data/gfs
    python fetch_gfs.py --preset t2m --date 20260922 --cycle 06 --hours 0-48:1
    python fetch_gfs.py --var TMP --level 850_mb --hours 0-72:6 --out C:/data/gfs_t850

Then in Kit:  start("C:/data/gfs/*.pwat.*.grib2", {"shortName": "pwat"}, ...)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# name -> (NOMADS var, NOMADS level, ecCodes select for GribSequence)
PRESETS = {
    "t2m": ("TMP", "2_m_above_ground", {"shortName": "2t"}),
    "pwat": ("PWAT", "entire_atmosphere_(considered_as_a_single_layer)", {"shortName": "pwat"}),
    "t850": ("TMP", "850_mb", {"shortName": "t", "level": 850}),
    "rh700": ("RH", "700_mb", {"shortName": "r", "level": 700}),
    "prmsl": ("PRMSL", "mean_sea_level", {"shortName": "prmsl"}),
}


def latest_cycle(now: dt.datetime | None = None, lag_hours: int = 6) -> tuple[str, str]:
    """Most recent cycle that is normally fully posted (runs appear ~4-5 h after cycle time)."""
    t = (now or dt.datetime.now(dt.timezone.utc)) - dt.timedelta(hours=lag_hours)
    return t.strftime("%Y%m%d"), f"{t.hour // 6 * 6:02d}"


def parse_hours(spec: str) -> list[int]:
    """'0-120:3' -> [0, 3, ..., 120];  '0,6,12' -> [0, 6, 12]."""
    if "-" in spec:
        rng, _, step = spec.partition(":")
        a, b = (int(x) for x in rng.split("-"))
        return list(range(a, b + 1, int(step or 1)))
    return [int(x) for x in spec.split(",")]


def build_url(date: str, cycle: str, fhour: int, var: str, level: str,
              bbox: tuple[float, float, float, float] | None) -> str:
    lev_key = "lev_" + level.replace("(", r"\(").replace(")", r"\)")
    params = [("file", f"gfs.t{cycle}z.pgrb2.0p25.f{fhour:03d}"), (lev_key, "on"), (f"var_{var}", "on")]
    if bbox:
        left, right, top, bottom = bbox
        params += [("subregion", ""), ("leftlon", left), ("rightlon", right),
                   ("toplat", top), ("bottomlat", bottom)]
    params.append(("dir", f"/gfs.{date}/{cycle}/atmos"))
    query = "&".join(f"{urllib.parse.quote(str(k), safe='_')}={urllib.parse.quote(str(v), safe='')}"
                     for k, v in params)
    return f"{FILTER_URL}?{query}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="pwat")
    p.add_argument("--var", help="NOMADS variable, e.g. TMP (overrides preset)")
    p.add_argument("--level", help="NOMADS level, e.g. 850_mb (overrides preset)")
    p.add_argument("--date", help="run date YYYYMMDD (default: latest posted run)")
    p.add_argument("--cycle", choices=["00", "06", "12", "18"], help="run hour (default: latest)")
    p.add_argument("--hours", default="0-120:3", help="forecast hours, '0-120:3' or '0,6,12'")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
                   help="subregion in degrees (default: global)")
    p.add_argument("--out", default="gfs_data")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests (be kind to NOMADS)")
    args = p.parse_args(argv)

    var, level, select = PRESETS[args.preset]
    var, level = args.var or var, args.level or level
    date, cycle = args.date, args.cycle
    if not date or not cycle:
        d, c = latest_cycle()
        date, cycle = date or d, cycle or c
    tag = args.preset if not (args.var or args.level) else f"{var}_{level}".replace("(", "").replace(")", "")
    os.makedirs(args.out, exist_ok=True)

    hours = parse_hours(args.hours)
    print(f"GFS {date} {cycle}Z  {var} @ {level}  {len(hours)} file(s) -> {os.path.abspath(args.out)}")
    ok = 0
    for i, fh in enumerate(hours):
        dest = os.path.join(args.out, f"gfs.{date}.t{cycle}z.{tag}.f{fh:03d}.grib2")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            ok += 1
            continue
        url = build_url(date, cycle, fh, var, level, tuple(args.bbox) if args.bbox else None)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "gvt-grib-sphere/0.1"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 2:
                    print(f"  f{fh:03d}: failed ({exc})")
                    data = b""
                time.sleep(5 * (attempt + 1))
        if data[:4] != b"GRIB":
            msg = data[:200].decode("utf-8", "replace").strip().replace("\n", " ")
            print(f"  f{fh:03d}: no GRIB returned ({msg or 'empty'}) - run not posted yet, or wrong var/level?")
        else:
            with open(dest, "wb") as f:
                f.write(data)
            ok += 1
            print(f"  f{fh:03d}: {len(data) / 1e6:.1f} MB")
        if i < len(hours) - 1:
            time.sleep(args.delay)

    sel = select if not (args.var or args.level) else "{...}  (run inventory() on a file to pick keys)"
    print(f"{ok}/{len(hours)} files.  In Kit:\n"
          f"  start(r\"{os.path.abspath(args.out)}/*.{tag}.*.grib2\", {sel})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
