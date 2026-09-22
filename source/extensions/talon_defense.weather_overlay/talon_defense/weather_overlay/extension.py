"""Kit extension entry point: optional autostart from settings, clean shutdown."""
from __future__ import annotations

import asyncio
from typing import Any

import carb.settings
import omni.ext
import omni.kit.app

from . import _log
from .style import OpacityRamp, Style



def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class GribSphereExtension(omni.ext.IExt):
    """Starts playback on launch when ``grib_glob`` (or ``demo``) is set in the .kit settings;
    otherwise it just makes the package importable for scripts.

    Settings are read from ``/exts/<this extension's name>/``, so renaming the
    extension only requires renaming the keys in config/extension.toml."""

    def on_startup(self, ext_id: str) -> None:
        self._task: asyncio.Task | None = None
        settings = carb.settings.get_settings()
        name_of = getattr(omni.ext, "get_extension_name", None)
        ext_name = name_of(ext_id) if callable(name_of) else ext_id.rsplit("-", 1)[0]
        prefix = f"/exts/{ext_name}"

        def get(key: str, default: Any = None) -> Any:
            value = settings.get(f"{prefix}/{key}")
            return default if value is None else value

        grib_glob = get("grib_glob", "") or None
        if grib_glob or get("demo", False):
            self._task = asyncio.ensure_future(self._autostart(get, grib_glob))
        else:
            _log.info(f"{prefix}/grib_glob is empty and demo is off: API only, no autostart")

    async def _autostart(self, get: Any, grib_glob: str | None) -> None:
        import omni.usd

        from .session import start

        app = omni.kit.app.get_app()
        for _ in range(600):  # wait for the app to open a stage (~10 s at 60 fps)
            if omni.usd.get_context().get_stage() is not None:
                break
            await app.next_update_async()
        else:
            _log.error("No stage opened; GRIB sphere autostart skipped")
            return
        await app.next_update_async()

        ramp = list(get("opacity_ramp", []) or [])
        opacity: Any = OpacityRamp(*map(float, ramp)) if len(ramp) in (2, 4, 5) else float(get("opacity", 0.55))
        style = Style(cmap=get("cmap", "turbo"), vmin=_num(get("vmin")), vmax=_num(get("vmax")),
                      opacity=opacity, log_scale=bool(get("log_scale", False)))
        select = dict(get("select", {}) or {}) or None
        stride = int(get("stride", 0) or 0) or None
        try:
            start(grib_glob, select, style=None if grib_glob is None else style,
                  path=get("prim_path", "/World/GribSphere"), radius=float(get("radius", 100.0)),
                  stride=stride, max_points=int(get("max_points", 400_000)),
                  hours_per_second=float(get("hours_per_second", 6.0)), clock=get("clock", "wall"),
                  unlit=bool(get("unlit", True)), double_sided=bool(get("double_sided", False)),
                  emissive_scale=float(get("emissive_scale", 1.0)),
                  material=str(get("material", "mdl")),
                  emissive_intensity=float(get("emissive_intensity", 1000.0)),
                  backend=get("backend", "auto"), fractional_cutout=bool(get("fractional_cutout", True)))
        except Exception as exc:  # noqa: BLE001 - report, don't break app startup
            _log.error(f"GRIB sphere autostart failed: {exc}")

    def on_shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        from .session import stop_all

        stop_all(remove_prims=True)
