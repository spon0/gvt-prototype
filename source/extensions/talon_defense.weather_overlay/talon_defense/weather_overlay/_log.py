"""Logging that goes to carb inside Kit and to the stdlib logger elsewhere."""
from __future__ import annotations

import logging

_py_log = logging.getLogger(__package__ or "weather_overlay")

try:  # inside Kit
    import carb as _carb
except ImportError:  # plain Python (tests, tools)
    _carb = None


def info(msg: str) -> None:
    if _carb is not None:
        _carb.log_info(f"[grib_sphere] {msg}")
    else:
        _py_log.info(msg)


def warn(msg: str) -> None:
    if _carb is not None:
        _carb.log_warn(f"[grib_sphere] {msg}")
    else:
        _py_log.warning(msg)


def error(msg: str) -> None:
    if _carb is not None:
        _carb.log_error(f"[grib_sphere] {msg}")
    else:
        _py_log.error(msg)
