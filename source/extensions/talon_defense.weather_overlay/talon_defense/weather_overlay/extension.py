"""Kit extension entry point: optional autostart from settings, clean shutdown."""
from __future__ import annotations

import asyncio
from typing import Any

import carb.settings
import omni.ext
import omni.kit.app

from . import _log
from .layers import WeatherLayers, close_all
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
        if dict(get("layers", {}) or {}):
            self._task = asyncio.ensure_future(self._autostart_layers(get))
        elif grib_glob or get("demo", False):
            self._task = asyncio.ensure_future(self._autostart(get, grib_glob))
        else:
            _log.info(f"{prefix}/grib_glob is empty and demo is off: API only, no autostart")

    async def _wait_for_stage(self) -> Any:
        import omni.usd

        app = omni.kit.app.get_app()
        for _ in range(600):  # wait for the app to open a stage (~10 s at 60 fps)
            if omni.usd.get_context().get_stage() is not None:
                break
            await app.next_update_async()
        else:
            _log.error("No stage opened; GRIB sphere autostart skipped")
            return None
        await app.next_update_async()
        return omni.usd.get_context().get_stage()

    async def _autostart_layers(self, get: Any) -> None:
        """Build one shell per entry in the `layers` settings table, on one shared clock."""
        stage = await self._wait_for_stage()
        if stage is None:
            return
        try:
            manager = WeatherLayers(
                stage, root=get("layers_root", "/World/Weather"),
                base_radius=float(get("radius", 64.0)),
                separation=float(get("layer_separation", 0.3)),
                hours_per_second=float(get("hours_per_second", 6.0)), clock=get("clock", "wall"),
                max_points=int(get("max_points", 400_000)), unlit=bool(get("unlit", True)),
                emissive_intensity=float(get("emissive_intensity", 1000.0)),
                material=str(get("material", "mdl")),
                double_sided=bool(get("double_sided", False)), backend=get("backend", "auto"),
                fractional_cutout=bool(get("fractional_cutout", True)))
            configs = dict(get("layers", {}) or {})
            for name in sorted(configs, key=lambda n: (float(configs[n].get("order", 1e9)), n)):
                cfg = dict(configs[name])
                manager.add(name, cfg.get("glob") or None, dict(cfg.get("select", {}) or {}) or None,
                            cmap=cfg.get("cmap", "turbo"), vmin=_num(cfg.get("vmin")),
                            vmax=_num(cfg.get("vmax")), opacity=float(cfg.get("opacity", 0.55)),
                            opacity_ramp=list(cfg.get("opacity_ramp", []) or []),
                            log_scale=bool(cfg.get("log_scale", False)),
                            stride=int(cfg.get("stride", 0) or 0) or None,
                            radius=_num(cfg.get("radius")), visible=bool(cfg.get("visible", True)))
            manager.start()
            if bool(get("ui", True)):
                from .ui import open_panel

                open_panel(manager)
        except Exception as exc:  # noqa: BLE001 - report, don't break app startup
            _log.error(f"Weather layers autostart failed: {exc}")

    async def _autostart(self, get: Any, grib_glob: str | None) -> None:
        from .session import start

        if await self._wait_for_stage() is None:
            return

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

        try:
            from .ui import close_panels

            close_panels()
        except ImportError:
            pass
        close_all(remove_prims=True)
        stop_all(remove_prims=True)
