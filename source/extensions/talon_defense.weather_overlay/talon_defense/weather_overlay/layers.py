"""Several weather variables on one globe: nested shells, one clock, per-layer visibility.

    from talon_defense.weather_overlay import WeatherLayers, OpacityRamp

    w = WeatherLayers(base_radius=64.0, separation=0.3, hours_per_second=6)
    w.add("pwat", "C:/data/gfs/*.pwat.*.grib2", {"shortName": "pwat"},
          cmap="viridis", opacity=OpacityRamp(20, 60, 0.0, 0.85))
    w.add("t2m", "C:/data/gfs/*.t2m.*.grib2", {"shortName": "2t"},
          cmap="coolwarm", vmin=240, vmax=320, opacity=0.35)
    w.toggle("t2m"); w.solo("pwat"); w.seek(24); w.play()

Each layer is its own shell prim under one parent, at its own radius:

    /World/Weather            Xform - hide this to hide every layer at once
      /World/Weather/pwat     radius base_radius
      /World/Weather/t2m      radius base_radius + separation
      ...

Why not USD instancing: instances share their primvars, and every variable needs
its own per-vertex colors and opacity, so instances would all show the same field.

One clock drives every layer, on absolute valid time, so variables from different
runs or with different step lengths always show the same forecast time. A layer
whose data starts later (or ends earlier) holds its nearest frame.

Hidden layers upload nothing: the per-frame cost (blend + colormap + upload) is
paid per *visible* layer. Overdraw is the other budget - each visible transparent
shell adds an any-hit intersection per ray, so a handful is comfortable, a dozen
is not.
"""
from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from pxr import Sdf, Tf, Usd, UsdGeom

from . import _log
from .player import blended_values, frame_at, subscribe_update, wrap_hours, _release
from .style import OpacityRamp, Style
from .usd_sphere import GribSphere, enable_fractional_cutout_opacity

_MANAGERS: list["WeatherLayers"] = []


def weather_layers() -> "WeatherLayers | None":
    """The most recently created manager (what the extension autostarted, if any)."""
    return _MANAGERS[-1] if _MANAGERS else None


class WeatherLayer:
    """One variable: its own shell, GRIB sequence, style and visibility."""

    def __init__(self, name: str, stage: Usd.Stage, path: str | Sdf.Path, sequence: Any,
                 style: Style, radius: float, *, unlit: bool = True,
                 emissive_intensity: float = 1000.0, material: str = "mdl",
                 double_sided: bool = False, backend: str = "auto",
                 layer: Sdf.Layer | None = None, lookahead: int = 3):
        self.name = name
        self.sequence = sequence
        self.style = style
        self.lookahead = lookahead
        self.sphere = GribSphere(stage, path, sequence.grid, radius=radius, unlit=unlit,
                                 emissive_intensity=emissive_intensity, material=material,
                                 double_sided=double_sided, backend=backend, layer=layer)
        self.times: list[_dt.datetime] = list(sequence.times)
        self.hours_abs = np.asarray(sequence.hours, dtype=np.float64)  # re-based by set_epoch
        self._visible = True
        self._last_key: tuple | None = None
        self.opacity_scale = 1.0

    # ---------------------------------------------------------------- time
    def set_epoch(self, epoch: _dt.datetime) -> None:
        """Re-base this layer's time axis onto the manager's shared epoch."""
        self.hours_abs = np.array([(t - epoch).total_seconds() / 3600.0 for t in self.times])
        self._last_key = None

    @property
    def time_range(self) -> tuple[_dt.datetime, _dt.datetime]:
        return self.times[0], self.times[-1]

    def covers(self, hours_abs: float) -> bool:
        return bool(self.hours_abs[0] <= hours_abs <= self.hours_abs[-1])

    # ---------------------------------------------------------------- visibility
    @property
    def visible(self) -> bool:
        return self._visible

    def set_visible(self, visible: bool) -> None:
        """Show/hide with USD visibility: instant, keeps the geometry loaded."""
        if visible == self._visible:
            return
        self._visible = bool(visible)
        self.sphere.set_visible(self._visible)
        if self._visible:
            self._last_key = None  # re-upload on the next tick, it may be stale

    def set_active(self, active: bool) -> None:
        """Drop the prim from the scene entirely (frees renderer memory, slow to come back)."""
        prim = self.sphere.stage.GetPrimAtPath(self.sphere.path)
        if prim:
            with Usd.EditContext(self.sphere.stage, self.sphere.layer):
                prim.SetActive(bool(active))
        self._last_key = None

    # ---------------------------------------------------------------- per frame
    def render_at(self, hours_abs: float, *, interpolate: bool = True, force: bool = False) -> bool:
        """Show this layer's data at a shared absolute time. Returns True if it uploaded."""
        if not self._visible:
            return False
        h = min(max(hours_abs, self.hours_abs[0]), self.hours_abs[-1])  # hold ends, no looping here
        i, w = frame_at(self.hours_abs, h, interpolate)
        key = (i, round(w, 3))
        if key == self._last_key and not force:
            return False
        values = blended_values(self.sequence, i, w, self.lookahead, loop=False)
        if values is None:
            return False  # still decoding
        if self.style.vmin is None or self.style.vmax is None:
            self.style.autoscale(values)
        self.sphere.set_field(values, self.style)
        self._last_key = key
        return True

    def restyle(self, style: Style | None = None) -> None:
        if style is not None:
            self.style = style
        self._last_key = None

    def set_opacity_scale(self, value: float) -> None:
        """Fade the whole layer without re-uploading (material="mdl")."""
        self.opacity_scale = float(value)
        self.sphere.set_opacity_scale(self.opacity_scale)

    @property
    def units(self) -> str:
        return str(getattr(self.sequence, "meta", {}).get("units", ""))

    def close(self, remove_prim: bool = True) -> None:
        self.sequence.close()
        if remove_prim:
            self.sphere.remove()

    def __repr__(self) -> str:
        return (f"<WeatherLayer {self.name} {len(self.times)} frames "
                f"r={self.sphere.radius:g} {'visible' if self._visible else 'hidden'}>")


class WeatherLayers:
    """Manages several WeatherLayer shells on one shared clock.

    Args:
        root: parent prim for every layer; hiding it hides them all.
        base_radius / separation: first shell's radius and the gap between shells
            (stage units). Separate shells avoid coplanar surfaces fighting.
        hours_per_second: data hours per wall-clock second.
        clock: "wall" or "timeline" (follows the Kit timeline's play/scrub).
        max_upload_hz: cap on colour uploads per second, across all layers.
    """

    RENDER_SETTINGS_WATCH_S = 30.0

    def __init__(self, stage: Any = None, root: str = "/World/Weather", *,
                 base_radius: float = 64.0, separation: float = 0.3,
                 hours_per_second: float = 6.0, interpolate: bool = True, loop: bool = True,
                 clock: str = "wall", max_upload_hz: float | None = 30.0,
                 max_points: int | None = 400_000, unlit: bool = True,
                 emissive_intensity: float = 1000.0, material: str = "mdl",
                 double_sided: bool = False, backend: str = "auto",
                 usd_layer: Sdf.Layer | None = None, fractional_cutout: bool = True):
        if stage is None:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("No USD stage is open")
        self.stage = stage
        self.root = Sdf.Path(root)
        self.usd_layer = usd_layer or stage.GetSessionLayer()
        self.base_radius = float(base_radius)
        self.separation = float(separation)
        self.hours_per_second = float(hours_per_second)
        self.interpolate = interpolate
        self.loop = loop
        self.clock = clock
        self.max_upload_hz = max_upload_hz
        self.defaults = dict(max_points=max_points, unlit=unlit, material=material,
                             emissive_intensity=emissive_intensity, double_sided=double_sided,
                             backend=backend)
        self.fractional_cutout = fractional_cutout
        self.layers: dict[str, WeatherLayer] = {}
        self.epoch: _dt.datetime | None = None
        self.hours = 0.0
        self.playing = False
        self.on_tick: list[Callable[["WeatherLayers"], None]] = []
        self._sub: Any = None
        self._timeline: Any = None
        self._last_upload = 0.0
        self._rs_started = time.perf_counter()
        self._rs_last = 0.0
        self._rs_logged: set[str] = set()
        with Usd.EditContext(self.stage, self.usd_layer):
            UsdGeom.Xform.Define(self.stage, self.root)
        if self.fractional_cutout:
            for line in enable_fractional_cutout_opacity(True, self.stage):
                _log.info(f"render settings: {line}")
        _MANAGERS.append(self)

    # ---------------------------------------------------------------- layers
    def add(self, name: str, grib: str | Iterable[str] | None = None,
            select: Mapping[str, Any] | None = None, *, style: Style | None = None,
            cmap: Any = "turbo", vmin: float | None = None, vmax: float | None = None,
            opacity: Any = 0.55, opacity_ramp: Iterable[float] | None = None,
            log_scale: bool = False, stride: int | None = None, radius: float | None = None,
            visible: bool = True, **sphere_kwargs: Any) -> WeatherLayer:
        """Add a variable. ``grib`` is a glob/paths (None = the synthetic demo field)."""
        if name in self.layers:
            self.remove(name)
        if grib is None:
            from .sequence import SyntheticSequence

            sequence: Any = SyntheticSequence()
        else:
            from .sequence import GribSequence

            sequence = GribSequence(grib, select, stride=stride,
                                    max_points=self.defaults["max_points"])
        if style is None:
            ramp = list(opacity_ramp or [])
            style = Style(cmap=cmap, vmin=vmin, vmax=vmax, log_scale=log_scale,
                          opacity=OpacityRamp(*map(float, ramp)) if len(ramp) in (2, 4, 5)
                          else float(opacity))
        options = {k: v for k, v in self.defaults.items() if k != "max_points"}
        options.update(sphere_kwargs)
        layer = WeatherLayer(name, self.stage, self.root.AppendChild(_prim_name(name)), sequence,
                             style, self._radius_for(radius), layer=self.usd_layer, **options)
        self.layers[name] = layer
        self._rebase()
        layer.set_visible(visible)
        _log.info(f"layer '{name}': {len(sequence)} frames, {sequence.grid.n_points:,} vertices, "
                  f"radius {layer.sphere.radius:g}, {layer.time_range[0]:%Y-%m-%d %HZ}"
                  f" .. {layer.time_range[1]:%Y-%m-%d %HZ}")
        if self._sub is not None:
            layer.render_at(self._wrapped_hours(), interpolate=self.interpolate, force=True)
        return layer

    def _radius_for(self, radius: float | None) -> float:
        if radius is not None:
            return float(radius)
        used = {round(l.sphere.radius, 6) for l in self.layers.values()}
        candidate = self.base_radius
        while round(candidate, 6) in used:
            candidate += self.separation
        return candidate

    def _rebase(self) -> None:
        """Put every layer on one absolute-time axis (hours since the earliest frame)."""
        if not self.layers:
            self.epoch = None
            return
        epoch = min(l.times[0] for l in self.layers.values())
        if epoch != self.epoch:
            self.epoch = epoch
        for l in self.layers.values():
            l.set_epoch(self.epoch)

    def remove(self, name: str) -> None:
        layer = self.layers.pop(name, None)
        if layer is not None:
            layer.close()
            self._rebase()

    def clear(self) -> None:
        for name in list(self.layers):
            self.remove(name)

    def __getitem__(self, name: str) -> WeatherLayer:
        return self.layers[name]

    def __contains__(self, name: str) -> bool:
        return name in self.layers

    def __iter__(self):
        return iter(self.layers.values())

    # ---------------------------------------------------------------- visibility
    def set_visible(self, name: str, visible: bool) -> None:
        self.layers[name].set_visible(visible)

    def show(self, *names: str) -> None:
        for name in names or tuple(self.layers):
            self.layers[name].set_visible(True)

    def hide(self, *names: str) -> None:
        for name in names or tuple(self.layers):
            self.layers[name].set_visible(False)

    def toggle(self, name: str) -> bool:
        layer = self.layers[name]
        layer.set_visible(not layer.visible)
        return layer.visible

    def solo(self, name: str) -> None:
        """Show only this layer."""
        for other, layer in self.layers.items():
            layer.set_visible(other == name)

    @property
    def visible_names(self) -> list[str]:
        return [n for n, l in self.layers.items() if l.visible]

    def set_all_visible(self, visible: bool) -> None:
        """Hide/show the whole overlay at the root prim (layer states are kept)."""
        with Usd.EditContext(self.stage, self.usd_layer):
            img = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.root))
            img.MakeVisible() if visible else img.MakeInvisible()

    # ---------------------------------------------------------------- clock
    @property
    def span_hours(self) -> float:
        if not self.layers:
            return 0.0
        return float(max(l.hours_abs[-1] for l in self.layers.values()))

    @property
    def current_time(self) -> _dt.datetime | None:
        if self.epoch is None:
            return None
        return self.epoch + _dt.timedelta(hours=self._wrapped_hours())

    @property
    def time_range(self) -> tuple[_dt.datetime, _dt.datetime] | None:
        if self.epoch is None:
            return None
        return self.epoch, self.epoch + _dt.timedelta(hours=self.span_hours)

    def _wrapped_hours(self) -> float:
        return wrap_hours(self.hours, 0.0, self.span_hours, self.loop)

    def start(self, play: bool = True) -> "WeatherLayers":
        if self._sub is None:
            if self.clock == "timeline":
                import omni.timeline

                self._timeline = omni.timeline.get_timeline_interface()
            self._sub = subscribe_update(self._on_update, f"{__package__}.layers")
        self.playing = play
        self.render(force=True)
        return self

    def stop(self) -> None:
        self.playing = False
        if self._sub is not None:
            _release(self._sub)
            self._sub = None

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle_play(self) -> bool:
        self.playing = not self.playing
        return self.playing

    def seek(self, hours: float) -> None:
        self.hours = float(hours)
        self.render(force=True)

    def seek_time(self, when: _dt.datetime) -> None:
        if self.epoch is None:
            return
        self.seek((when - self.epoch).total_seconds() / 3600.0)

    # ---------------------------------------------------------------- per frame
    def _on_update(self, dt: float) -> None:
        try:
            self.step(dt)
        except Exception as exc:  # noqa: BLE001 - never spam the log every frame
            _log.error(f"WeatherLayers stopped: {exc!r}")
            self.stop()

    def step(self, dt: float) -> bool:
        if self.fractional_cutout:
            self._watch_render_settings()
        if self.clock == "timeline" and self._timeline is not None:
            self.hours = self._timeline.get_current_time() * self.hours_per_second
        elif self.playing:
            self.hours += dt * self.hours_per_second
        if self.max_upload_hz:
            now = time.perf_counter()
            if now - self._last_upload < 1.0 / self.max_upload_hz:
                return False
        return self.render()

    def render(self, force: bool = False) -> bool:
        hours = self._wrapped_hours()
        uploaded = False
        for layer in list(self.layers.values()):
            if not layer.sphere.is_valid():
                _log.warn(f"layer '{layer.name}' lost its prim; dropping it")
                self.layers.pop(layer.name, None)
                continue
            uploaded |= layer.render_at(hours, interpolate=self.interpolate, force=force)
        if uploaded:
            self._last_upload = time.perf_counter()
        for callback in list(self.on_tick):
            callback(self)
        return uploaded

    def _watch_render_settings(self) -> None:
        now = time.perf_counter()
        if now - self._rs_started > self.RENDER_SETTINGS_WATCH_S or now - self._rs_last < 1.0:
            return
        self._rs_last = now
        for line in enable_fractional_cutout_opacity(True, self.stage):
            if line not in self._rs_logged:
                self._rs_logged.add(line)
                _log.info(f"render settings: {line}")

    # ---------------------------------------------------------------- teardown
    def close(self, remove_prims: bool = True) -> None:
        self.stop()
        for name in list(self.layers):
            self.layers[name].close(remove_prims)
            self.layers.pop(name, None)
        if remove_prims:
            with Usd.EditContext(self.stage, self.usd_layer):
                if self.usd_layer.GetPrimAtPath(self.root):
                    self.stage.RemovePrim(self.root)
        self.on_tick.clear()
        if self in _MANAGERS:
            _MANAGERS.remove(self)

    def __repr__(self) -> str:
        return f"<WeatherLayers {self.root} {list(self.layers)} at {self.current_time}>"


def close_all(remove_prims: bool = True) -> None:
    for manager in list(_MANAGERS):
        manager.close(remove_prims)


def _prim_name(name: str) -> str:
    return Tf.MakeValidIdentifier(name) or "Layer"
