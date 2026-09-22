"""GRIB access through ecCodes: message inventory, grid lattice, field decoding.

Everything here is plain Python + numpy + ecCodes (no Kit imports), so it can be
unit-tested or used from a notebook.

Supported grids: any *structured* grid ecCodes can give latitudes/longitudes for
(regular_ll, rotated_ll, regular_gg, lambert, polar_stereographic, mercator...).
That covers GFS/GEFS/GDAS/WW3 (global lat/lon) and HRRR/NAM/RAP (Lambert
patches). Reduced (quasi-regular) Gaussian grids are rejected - regrid those
first (e.g. ``wgrib2 -new_grid`` or ``cdo remapbil``).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import eccodes
    _ECCODES_ERROR: Exception | None = None
except ImportError as _exc:  # pragma: no cover - reported when used
    eccodes = None
    _ECCODES_ERROR = _exc

# ecCodes is only guaranteed thread-safe when built with thread support, so all
# calls in this package go through one lock. Decoding happens on a worker thread,
# never on Kit's main thread, so the lock never stalls rendering.
ECCODES_LOCK = threading.RLock()

_HEADER_KEYS = (
    "shortName", "name", "units", "typeOfLevel", "level", "stepType", "stepRange",
    "dataDate", "dataTime", "validityDate", "validityTime", "gridType",
    "md5GridSection", "totalLength", "offset",
)


def _require_eccodes() -> None:
    if eccodes is None:
        raise ImportError(
            "The ecCodes Python bindings are required to read GRIB "
            "(pip install eccodes; the wheels bundle the C library)."
        ) from _ECCODES_ERROR


def _get(gid: Any, key: str, default: Any = None) -> Any:
    """Read a key; None/default when undefined or MISSING."""
    try:
        if not eccodes.codes_is_defined(gid, key):
            return default
        if eccodes.codes_is_missing(gid, key):
            return default
        return eccodes.codes_get(gid, key)
    except Exception:  # noqa: BLE001 - ecCodes raises many error types for "no such key"
        return default


def _valid_time(keys: Mapping[str, Any]) -> _dt.datetime:
    date = int(keys.get("validityDate") or keys.get("dataDate") or 19700101)
    hhmm = int(keys.get("validityTime") or keys.get("dataTime") or 0)
    return _dt.datetime(date // 10000, date // 100 % 100, date % 100,
                        hhmm // 100, hhmm % 100, tzinfo=_dt.timezone.utc)


def _eq(have: Any, want: Any) -> bool:
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            return float(have) == float(want)
        except (TypeError, ValueError):
            return False
    return str(have) == str(want)


def _matches(keys: Mapping[str, Any], select: Mapping[str, Any] | None) -> bool:
    if not select:
        return True
    for key, want in select.items():
        have = keys.get(key)
        if have is None:
            return False
        wants = want if isinstance(want, (list, tuple, set, frozenset)) else (want,)
        if not any(_eq(have, w) for w in wants):
            return False
    return True


# --------------------------------------------------------------------------- messages
@dataclass(frozen=True)
class MessageInfo:
    """One GRIB message: where it lives and what it is. Cheap to create (header only)."""

    path: str
    index: int  # 0-based message number within the file
    offset: int  # byte offset of the message in the file
    length: int  # message length in bytes
    keys: Mapping[str, Any] = field(repr=False)
    valid_time: _dt.datetime = field(default=None)  # type: ignore[assignment]

    @property
    def grid_key(self) -> str:
        return str(self.keys.get("md5GridSection") or "")

    def read_bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            return f.read(self.length)

    def __str__(self) -> str:
        k = self.keys
        return (f"#{self.index:<4d} {str(k.get('shortName')):>8s}  "
                f"{k.get('typeOfLevel')}:{k.get('level')}  step={k.get('stepRange')}  "
                f"valid={self.valid_time:%Y-%m-%d %H:%MZ}  {k.get('name')} [{k.get('units')}]  "
                f"grid={k.get('gridType')}")


def list_messages(path: str, select: Mapping[str, Any] | None = None,
                  extra_keys: Sequence[str] = ()) -> list[MessageInfo]:
    """Scan a GRIB file's headers and return the messages matching ``select``.

    ``select`` maps ecCodes keys to wanted values (a list/tuple means "any of"), e.g.
    ``{"shortName": "t", "typeOfLevel": "isobaricInhPa", "level": 850}``.
    Use :func:`inventory` to discover the keys in a file.
    """
    _require_eccodes()
    wanted = tuple(dict.fromkeys((*_HEADER_KEYS, *(select or {}).keys(), *extra_keys)))
    out: list[MessageInfo] = []
    with ECCODES_LOCK, open(path, "rb") as f:
        index = 0
        while True:
            gid = eccodes.codes_grib_new_from_file(f, headers_only=True)
            if gid is None:
                break
            try:
                keys = {k: _get(gid, k) for k in wanted}
            finally:
                eccodes.codes_release(gid)
            if _matches(keys, select):
                out.append(MessageInfo(path=str(path), index=index, offset=int(keys["offset"]),
                                       length=int(keys["totalLength"]), keys=keys,
                                       valid_time=_valid_time(keys)))
            index += 1
    return out


def inventory(path: str) -> str:
    """Human-readable list of every message in a file (like ``grib_ls``)."""
    return "\n".join(str(m) for m in list_messages(path))


# --------------------------------------------------------------------------- grid
@dataclass(eq=False)
class GribGrid:
    """A structured lat/lon lattice, already subsampled, in mesh vertex order.

    ``lats``/``lons`` have shape (rows, cols). Row-major flattening of that shape
    is the vertex order of the sphere mesh and of every decoded field.
    """

    key: str
    grid_type: str
    lats: np.ndarray
    lons: np.ndarray
    wrap_lon: bool  # connect the last column back to the first (global lat/lon grids)
    stride: int = 1
    # how to turn a raw ecCodes value array into (rows, cols); None for synthetic grids
    ni: int | None = None
    nj: int | None = None
    j_consecutive: bool = False
    alternate_rows: bool = False
    rows: np.ndarray | None = field(default=None, repr=False)
    cols: np.ndarray | None = field(default=None, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.lats.shape  # type: ignore[return-value]

    @property
    def n_points(self) -> int:
        return int(self.lats.size)

    def arrange(self, raw: np.ndarray) -> np.ndarray:
        """Raw ecCodes value order -> flat (rows*cols,) array in mesh vertex order."""
        a = _to_2d(np.asarray(raw), self.ni, self.nj, self.j_consecutive, self.alternate_rows)
        if self.rows is not None:
            a = a[np.ix_(self.rows, self.cols)]
        return np.ascontiguousarray(a).reshape(-1)

    @classmethod
    def regular(cls, lat_step: float = 1.0, lon_step: float = 1.0) -> "GribGrid":
        """Global lat/lon grid (90N->90S, 0->360-step). Handy for synthetic data."""
        lats = np.arange(90.0, -90.0 - lat_step / 2, -lat_step)
        lons = np.arange(0.0, 360.0 - lon_step / 2, lon_step)
        lon2, lat2 = np.meshgrid(lons, lats)
        return cls(key=f"regular:{lat_step}:{lon_step}", grid_type="regular_ll",
                   lats=lat2, lons=lon2, wrap_lon=True)


def _to_2d(flat: np.ndarray, ni: int | None, nj: int | None,
           j_consecutive: bool, alternate_rows: bool) -> np.ndarray:
    if ni is None or nj is None:  # synthetic grid: already (rows*cols)
        return flat
    if j_consecutive:  # columns are contiguous in the file
        a = flat.reshape(ni, nj).copy() if alternate_rows else flat.reshape(ni, nj)
        if alternate_rows:
            a[1::2] = a[1::2, ::-1]
        return a.T
    a = flat.reshape(nj, ni).copy() if alternate_rows else flat.reshape(nj, ni)
    if alternate_rows:  # boustrophedonic scanning
        a[1::2] = a[1::2, ::-1]
    return a


def _pick(n: int, stride: int) -> np.ndarray:
    idx = np.arange(0, n, stride)
    if idx[-1] != n - 1:  # always keep the last row/column (poles, domain edge)
        idx = np.append(idx, n - 1)
    return idx


def auto_stride(ni: int, nj: int, max_points: int | None) -> int:
    if not max_points or ni * nj <= max_points:
        return 1
    return int(math.ceil(math.sqrt(ni * nj / max_points)))


def grid_from_message(gid: Any, stride: int | None = None,
                      max_points: int | None = 400_000) -> GribGrid:
    """Build the (subsampled) lattice for an open ecCodes handle."""
    grid_type = str(_get(gid, "gridType", "unknown"))
    ni = _get(gid, "Ni") or _get(gid, "Nx")
    nj = _get(gid, "Nj") or _get(gid, "Ny")
    if not ni or not nj or grid_type.startswith("reduced"):
        raise ValueError(
            f"GRIB grid '{grid_type}' is not a structured Ni x Nj lattice "
            "(reduced/unstructured grids are not supported - regrid to regular_ll first).")
    ni, nj = int(ni), int(nj)
    j_consecutive = bool(_get(gid, "jPointsAreConsecutive", 0))
    alternate_rows = bool(_get(gid, "alternativeRowScanning", 0))
    stride = stride or auto_stride(ni, nj, max_points)

    lats = eccodes.codes_get_double_array(gid, "latitudes")
    lons = eccodes.codes_get_double_array(gid, "longitudes")
    lat2 = _to_2d(lats, ni, nj, j_consecutive, alternate_rows)
    lon2 = _to_2d(lons, ni, nj, j_consecutive, alternate_rows)
    rows, cols = _pick(lat2.shape[0], stride), _pick(lat2.shape[1], stride)

    wrap = False
    di = _get(gid, "iDirectionIncrementInDegrees")
    if grid_type in ("regular_ll", "rotated_ll", "regular_gg", "rotated_gg") and di:
        # e.g. GFS 0.25: 1440 * 0.25 = 360 -> the seam must be stitched by index.
        # A grid that repeats the seam column (Ni*di = 360+di) closes by itself.
        wrap = abs(ni * float(di) - 360.0) < 0.5 * float(di)
        if wrap and cols[-1] == lat2.shape[1] - 1 and (lat2.shape[1] - 1) % stride:
            cols = cols[:-1]  # keep columns evenly spaced across the stitched seam

    key = str(_get(gid, "md5GridSection") or "")
    if not key:
        key = hashlib.md5(np.concatenate([lats[:4], lons[:4], lats[-4:], lons[-4:]]).tobytes()).hexdigest()
    return GribGrid(key=f"{key}:s{stride}", grid_type=grid_type,
                    lats=np.ascontiguousarray(lat2[np.ix_(rows, cols)]),
                    lons=np.ascontiguousarray(lon2[np.ix_(rows, cols)]),
                    wrap_lon=wrap, stride=stride, ni=ni, nj=nj, j_consecutive=j_consecutive,
                    alternate_rows=alternate_rows, rows=rows, cols=cols)


def decode_values(gid: Any, grid: GribGrid) -> np.ndarray:
    """Decode an open handle's data into mesh vertex order (float32, NaN = missing)."""
    values = eccodes.codes_get_values(gid)
    if _get(gid, "bitmapPresent", 0):
        missing = _get(gid, "missingValue", 9999)
        values = np.where(values == missing, np.nan, values)
    return grid.arrange(values).astype(np.float32, copy=False)


@dataclass
class GribField:
    values: np.ndarray  # (grid.n_points,) float32
    grid: GribGrid
    info: MessageInfo


def read_field(path_or_msg: str | MessageInfo, select: Mapping[str, Any] | None = None, *,
               stride: int | None = None, max_points: int | None = 400_000,
               grid: GribGrid | None = None) -> GribField:
    """Read one field. Pass a path + ``select`` (first match wins) or a MessageInfo."""
    _require_eccodes()
    if isinstance(path_or_msg, MessageInfo):
        msg = path_or_msg
    else:
        found = list_messages(path_or_msg, select)
        if not found:
            raise LookupError(f"No message in {path_or_msg} matches {select}")
        msg = found[0]
    data = msg.read_bytes()
    with ECCODES_LOCK:
        gid = eccodes.codes_new_from_message(data)
        try:
            if grid is None:
                grid = grid_from_message(gid, stride, max_points)
            values = decode_values(gid, grid)
        finally:
            eccodes.codes_release(gid)
    return GribField(values=values, grid=grid, info=msg)


def expand_paths(paths: str | Iterable[str]) -> list[str]:
    """Accept a glob pattern, a single path or an iterable of either."""
    import glob

    items = [paths] if isinstance(paths, str) else list(paths)
    out: list[str] = []
    for item in items:
        hits = sorted(glob.glob(str(item)))
        out.extend(hits if hits else [str(item)])
    return out
