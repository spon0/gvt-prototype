"""Time series of fields with background decoding.

A sequence exposes:
    grid   - the GribGrid every frame shares
    times  - list of valid datetimes (sorted)
    hours  - float64 array, hours since times[0]
    get(i) - decoded values for frame i or None if not decoded yet (never blocks;
             schedules the decode on a worker thread)
    load(i)- blocking version of get
    close()
"""
from __future__ import annotations

import datetime as _dt
import math
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterable, Mapping

import numpy as np

from . import _log
from .grib_io import (ECCODES_LOCK, GribGrid, MessageInfo, _require_eccodes, decode_values,
                      eccodes, expand_paths, grid_from_message, list_messages)


class GribSequence:
    """Frames from one or more GRIB files, one frame per valid time.

    Args:
        paths: glob pattern, path, or list of them. Every message matching
            ``select`` becomes a frame, so one multi-step file works as well as
            one-file-per-forecast-hour.
        select: ecCodes key filter, e.g. ``{"shortName": "2t", "level": 2}``.
        stride: keep every Nth row/column. ``None`` picks the smallest stride that
            keeps the mesh under ``max_points`` vertices (GFS 0.25 deg = 1.04M points).
        cache_frames: decoded frames kept in memory (LRU).
    """

    def __init__(self, paths: str | Iterable[str], select: Mapping[str, Any] | None = None, *,
                 stride: int | None = None, max_points: int | None = 400_000,
                 cache_frames: int = 16):
        _require_eccodes()
        files = expand_paths(paths)
        messages: list[MessageInfo] = []
        for path in files:
            messages.extend(list_messages(path, select))
        if not messages:
            raise LookupError(f"No GRIB messages match {dict(select or {})} in {len(files)} file(s): "
                              f"{files[:3]}{'...' if len(files) > 3 else ''}")

        messages.sort(key=lambda m: (m.valid_time, m.path, m.index))
        # keep the first message per valid time, but tell the user if there were more
        unique: list[MessageInfo] = []
        for m in messages:
            if unique and unique[-1].valid_time == m.valid_time:
                continue
            unique.append(m)
        if len(unique) < len(messages):
            _log.warn(f"{len(messages) - len(unique)} message(s) share a valid time with another and were "
                      f"skipped - narrow `select` (e.g. add typeOfLevel/level/stepType) if that is wrong")
        grids = {m.grid_key for m in unique}
        if len(grids) > 1:
            raise ValueError("Selected messages are on different grids; select one grid per sequence")

        self.messages = unique
        self.times: list[_dt.datetime] = [m.valid_time for m in unique]
        t0 = self.times[0]
        self.hours = np.array([(t - t0).total_seconds() / 3600.0 for t in self.times])
        self.meta = dict(unique[0].keys)

        data = unique[0].read_bytes()
        with ECCODES_LOCK:
            gid = eccodes.codes_new_from_message(data)
            try:
                self.grid: GribGrid = grid_from_message(gid, stride, max_points)
                first = decode_values(gid, self.grid)
            finally:
                eccodes.codes_release(gid)

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict({0: first})
        self._cache_frames = max(2, int(cache_frames))
        self._pending: dict[int, Future] = {}
        self._failed: set[int] = set()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="grib-decode")
        _log.info(f"{len(self)} frame(s) of {self.meta.get('shortName')} on a {self.grid.shape} lattice "
                  f"(stride {self.grid.stride}), {self.times[0]:%Y-%m-%d %HZ} .. {self.times[-1]:%Y-%m-%d %HZ}")

    def __len__(self) -> int:
        return len(self.messages)

    def _decode(self, i: int) -> np.ndarray:
        data = self.messages[i].read_bytes()
        with ECCODES_LOCK:
            gid = eccodes.codes_new_from_message(data)
            try:
                return decode_values(gid, self.grid)
            finally:
                eccodes.codes_release(gid)

    def _work(self, i: int) -> None:
        try:
            values = self._decode(i)
        except Exception as exc:  # noqa: BLE001 - logged once, frame shows as missing
            _log.error(f"Failed to decode frame {i} ({self.messages[i]}): {exc}")
            values = np.full(self.grid.n_points, np.nan, dtype=np.float32)
            with self._lock:
                self._failed.add(i)
        with self._lock:
            self._cache[i] = values
            self._cache.move_to_end(i)
            while len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)
            self._pending.pop(i, None)

    def get(self, i: int) -> np.ndarray | None:
        with self._lock:
            hit = self._cache.get(i)
            if hit is not None:
                self._cache.move_to_end(i)
                return hit
            if i not in self._pending and self._pool is not None:
                self._pending[i] = self._pool.submit(self._work, i)
        return None

    def prefetch(self, indices: Iterable[int]) -> None:
        for i in indices:
            if 0 <= i < len(self):
                self.get(i)

    def load(self, i: int) -> np.ndarray:
        values = self.get(i)
        if values is not None:
            return values
        with self._lock:
            fut = self._pending.get(i)
        if fut is not None:
            fut.result()
        values = self.get(i)
        return values if values is not None else self._decode(i)

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


class SyntheticSequence:
    """Animated fake field on a global grid - checks the rendering path without data.

    Looks like drifting cloud bands: a few Gaussian blobs moving east over a
    latitude-dependent background, values 0..1.
    """

    def __init__(self, n_frames: int = 48, hours_step: float = 1.0, lat_step: float = 1.0,
                 lon_step: float = 1.0, seed: int = 7):
        self.grid = GribGrid.regular(lat_step, lon_step)
        start = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
        self.hours = np.arange(n_frames, dtype=np.float64) * hours_step
        self.times = [start + _dt.timedelta(hours=float(h)) for h in self.hours]
        self.meta = {"shortName": "synthetic", "units": "1"}
        rng = np.random.default_rng(seed)
        self._blobs = [(rng.uniform(-60, 60), rng.uniform(0, 360), rng.uniform(8, 25),
                        rng.uniform(4, 15)) for _ in range(14)]  # lat, lon, size, deg/hour drift
        self._lat = np.radians(self.grid.lats.reshape(-1))
        self._lon = np.radians(self.grid.lons.reshape(-1))

    def __len__(self) -> int:
        return len(self.hours)

    def get(self, i: int) -> np.ndarray:
        h = self.hours[i]
        field = 0.25 * np.cos(3 * self._lat) ** 2
        for lat0, lon0, size, speed in self._blobs:
            lat0r, lon0r = math.radians(lat0), math.radians(lon0 + speed * h)
            cosd = (np.sin(self._lat) * math.sin(lat0r)
                    + np.cos(self._lat) * math.cos(lat0r) * np.cos(self._lon - lon0r))
            dist = np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))
            field = field + np.exp(-(dist / size) ** 2)
        return np.clip(field, 0.0, 1.0).astype(np.float32)

    load = get

    def prefetch(self, indices: Iterable[int]) -> None:
        pass

    def close(self) -> None:
        pass
