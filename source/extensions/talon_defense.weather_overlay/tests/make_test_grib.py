"""Write small synthetic GRIB2 files with ecCodes (no network needed).

    python tests/make_test_grib.py OUT_DIR
"""
from __future__ import annotations

import os
import sys

import eccodes
import numpy as np


def analytic(lat: np.ndarray, lon: np.ndarray, hour: float) -> np.ndarray:
    """Smooth, direction-sensitive test field (K): tells N from S and E from W."""
    return (273.15 + 30.0 * np.cos(np.radians(lat)) + 0.1 * lat
            + 5.0 * np.sin(np.radians(lon - 10.0 * hour)))


def write_regular_ll(path: str, *, step_deg: float, hours: list[int], north_to_south: bool = True,
                     missing_box: tuple[float, float, float, float] | None = None,
                     short_name_params: tuple[int, int, int, int, int] = (0, 0, 0, 103, 2),
                     append: bool = False) -> None:
    ni, nj = int(round(360 / step_deg)), int(round(180 / step_deg)) + 1
    lats = np.linspace(90, -90, nj) if north_to_south else np.linspace(-90, 90, nj)
    lons = np.arange(ni) * step_deg
    lon2, lat2 = np.meshgrid(lons, lats)
    discipline, category, number, surface, level = short_name_params
    with open(path, "ab" if append else "wb") as f:
        for hour in hours:
            gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            try:
                for key, value in [
                    ("Ni", ni), ("Nj", nj),
                    ("latitudeOfFirstGridPointInDegrees", float(lats[0])),
                    ("longitudeOfFirstGridPointInDegrees", 0.0),
                    ("latitudeOfLastGridPointInDegrees", float(lats[-1])),
                    ("longitudeOfLastGridPointInDegrees", float(lons[-1])),
                    ("iDirectionIncrementInDegrees", step_deg), ("jDirectionIncrementInDegrees", step_deg),
                    ("jScansPositively", 0 if north_to_south else 1),
                    ("discipline", discipline), ("parameterCategory", category), ("parameterNumber", number),
                    ("typeOfFirstFixedSurface", surface), ("scaleFactorOfFirstFixedSurface", 0),
                    ("scaledValueOfFirstFixedSurface", level),
                    ("dataDate", 20260922), ("dataTime", 0),
                    ("stepUnits", 1), ("forecastTime", hour),
                    ("packingType", "grid_simple"), ("bitsPerValue", 16),
                ]:
                    eccodes.codes_set(gid, key, value)
                values = analytic(lat2, lon2, hour)
                if missing_box is not None:
                    la0, la1, lo0, lo1 = missing_box
                    box = (lat2 >= la0) & (lat2 <= la1) & (lon2 >= lo0) & (lon2 <= lo1)
                    eccodes.codes_set(gid, "bitmapPresent", 1)
                    eccodes.codes_set(gid, "missingValue", 9999)
                    values = np.where(box, 9999.0, values)
                eccodes.codes_set_values(gid, values.ravel())
                eccodes.codes_write(gid, f)
            finally:
                eccodes.codes_release(gid)


def write_lambert(path: str) -> None:
    """HRRR-like Lambert conformal CONUS patch at 30 km (HRRR itself is 3 km, 1799x1059); value = 2*lat."""
    gid = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(gid, "gridDefinitionTemplateNumber", 30)
        for key, value in [("Nx", 180), ("Ny", 106), ("latitudeOfFirstGridPointInDegrees", 21.138123),
                           ("longitudeOfFirstGridPointInDegrees", 237.280472), ("LaDInDegrees", 38.5),
                           ("LoVInDegrees", 262.5), ("DxInMetres", 30000.0), ("DyInMetres", 30000.0),
                           ("Latin1InDegrees", 38.5), ("Latin2InDegrees", 38.5), ("jScansPositively", 1),
                           ("shapeOfTheEarth", 6), ("dataDate", 20260922), ("dataTime", 1200),
                           ("packingType", "grid_simple"), ("bitsPerValue", 16)]:
            eccodes.codes_set(gid, key, value)
        eccodes.codes_set_values(gid, np.zeros(180 * 106))
        eccodes.codes_set_values(gid, 2.0 * eccodes.codes_get_double_array(gid, "latitudes"))
        with open(path, "wb") as f:
            eccodes.codes_write(gid, f)
    finally:
        eccodes.codes_release(gid)


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    # GFS-like: 0.25 deg global, one file per forecast hour, a missing-data box
    for h in (0, 3, 6, 9, 12):
        write_regular_ll(os.path.join(out, f"gfs_2t_f{h:03d}.grib2"), step_deg=0.25, hours=[h],
                         missing_box=(10, 20, 100, 120))
    # 1 deg, south-to-north scanning, several steps in one file, plus a second variable
    multi = os.path.join(out, "multi_1deg_s2n.grib2")
    write_regular_ll(multi, step_deg=1.0, hours=[0, 6, 12], north_to_south=False)
    write_regular_ll(multi, step_deg=1.0, hours=[0, 6, 12], north_to_south=False,
                     short_name_params=(0, 1, 3, 200, 0), append=True)  # PWAT-like
    write_lambert(os.path.join(out, "lambert_conus.grib2"))
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_grib")
