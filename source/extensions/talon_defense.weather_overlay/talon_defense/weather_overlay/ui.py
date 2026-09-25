"""omni.ui panel for WeatherLayers: per-layer toggles, opacity, and a shared time scrubber.

    from talon_defense.weather_overlay import open_panel
    open_panel()          # uses the running manager (weather_layers())
"""
from __future__ import annotations

from typing import Any

import omni.ui as ui

from . import _log
from .layers import WeatherLayers, weather_layers
from .style import COLORMAPS

WINDOW_TITLE = "Weather Layers"
_PANELS: list["WeatherPanel"] = []


class WeatherPanel:
    """A window with one row per layer plus the shared transport controls."""

    def __init__(self, manager: WeatherLayers | None = None, width: int = 420, height: int = 320):
        self.manager = manager or weather_layers()
        if self.manager is None:
            raise RuntimeError("No WeatherLayers manager: create one, or pass it in")
        self._syncing = False
        self._time_label: Any = None
        self._time_slider_model: Any = None
        self._play_button: Any = None
        self._rows: dict[str, dict[str, Any]] = {}
        self.window = ui.Window(WINDOW_TITLE, width=width, height=height)
        self.window.frame.set_build_fn(self._build)
        self.window.frame.rebuild()
        self.manager.on_tick.append(self._on_tick)
        _PANELS.append(self)

    # ---------------------------------------------------------------- build
    def _build(self) -> None:
        self._rows.clear()
        with ui.VStack(spacing=6, height=0):
            self._build_transport()
            ui.Separator(height=6)
            if not self.manager.layers:
                ui.Label("No layers. Add one with WeatherLayers.add(...)", alignment=ui.Alignment.CENTER)
            for name in list(self.manager.layers):
                self._build_layer_row(name)
            ui.Separator(height=6)
            with ui.HStack(height=24, spacing=4):
                ui.Button("Show all", clicked_fn=lambda: self._all(True))
                ui.Button("Hide all", clicked_fn=lambda: self._all(False))
                ui.Button("Rebuild", clicked_fn=self.rebuild)

    def _build_transport(self) -> None:
        with ui.HStack(height=24, spacing=6):
            self._play_button = ui.Button("Pause" if self.manager.playing else "Play", width=70,
                                          clicked_fn=self._toggle_play)
            self._time_label = ui.Label(self._time_text(), width=210)
            ui.Label("h/s", width=24)
            rate = ui.FloatDrag(min=0.0, max=240.0, step=0.5, width=60)
            rate.model.set_value(self.manager.hours_per_second)
            rate.model.add_value_changed_fn(
                lambda m: setattr(self.manager, "hours_per_second", max(0.0, m.get_value_as_float())))
        with ui.HStack(height=24, spacing=6):
            ui.Label("Time", width=40)
            slider = ui.FloatSlider(min=0.0, max=max(self.manager.span_hours, 1e-6), step=0.1)
            self._time_slider_model = slider.model
            self._time_slider_model.set_value(self.manager.hours)
            self._time_slider_model.add_value_changed_fn(self._on_time_slider)

    def _build_layer_row(self, name: str) -> None:
        layer = self.manager.layers[name]
        row: dict[str, Any] = {}
        with ui.HStack(height=24, spacing=6):
            checkbox = ui.CheckBox(width=20)
            checkbox.model.set_value(layer.visible)
            checkbox.model.add_value_changed_fn(
                lambda m, n=name: self._on_visible(n, m.get_value_as_bool()))
            row["visible"] = checkbox.model
            units = f" [{layer.units}]" if layer.units else ""
            ui.Label(f"{name}{units}", width=110, tooltip=str(layer))
            opacity = ui.FloatSlider(min=0.0, max=1.0, step=0.01, width=110)
            opacity.model.set_value(layer.opacity_scale)
            opacity.model.add_value_changed_fn(
                lambda m, n=name: self._on_opacity(n, m.get_value_as_float()))
            row["opacity"] = opacity.model
            names = list(COLORMAPS)
            current = layer.style.cmap if isinstance(layer.style.cmap, str) else None
            combo = ui.ComboBox(names.index(current) if current in names else 0, *names, width=90)
            combo.model.add_item_changed_fn(
                lambda m, _i, n=name, opts=names: self._on_cmap(n, opts[m.get_item_value_model().as_int]))
            row["cmap"] = combo.model
            ui.Button("Solo", width=44, clicked_fn=lambda n=name: self._on_solo(n))
        self._rows[name] = row

    def rebuild(self) -> None:
        """Re-read the manager's layers (call after add/remove)."""
        self.window.frame.rebuild()

    # ---------------------------------------------------------------- events
    def _toggle_play(self) -> None:
        playing = self.manager.toggle_play()
        if self._play_button:
            self._play_button.text = "Pause" if playing else "Play"

    def _on_time_slider(self, model: Any) -> None:
        if self._syncing:
            return
        self.manager.pause()
        self.manager.seek(model.get_value_as_float())
        if self._play_button:
            self._play_button.text = "Play"

    def _on_visible(self, name: str, visible: bool) -> None:
        if not self._syncing and name in self.manager:
            self.manager.set_visible(name, visible)

    def _on_opacity(self, name: str, value: float) -> None:
        if not self._syncing and name in self.manager:
            self.manager[name].set_opacity_scale(value)

    def _on_cmap(self, name: str, cmap: str) -> None:
        if self._syncing or name not in self.manager:
            return
        layer = self.manager[name]
        layer.style.cmap = cmap
        layer.restyle()

    def _on_solo(self, name: str) -> None:
        self.manager.solo(name)
        self._sync()

    def _all(self, visible: bool) -> None:
        self.manager.show() if visible else self.manager.hide()
        self._sync()

    # ---------------------------------------------------------------- refresh
    def _time_text(self) -> str:
        now = self.manager.current_time
        if now is None:
            return "no data"
        return f"{now:%a %d %b %H:%MZ}  (+{self.manager._wrapped_hours():.1f} h)"

    def _on_tick(self, _manager: WeatherLayers) -> None:
        try:
            self._sync()
        except Exception as exc:  # noqa: BLE001 - a dead widget must not kill the clock
            _log.warn(f"weather panel refresh failed: {exc}")
            self.destroy()

    def _sync(self) -> None:
        self._syncing = True
        try:
            if self._time_label:
                self._time_label.text = self._time_text()
            if self._time_slider_model is not None:
                self._time_slider_model.set_value(self.manager._wrapped_hours())
            for name, row in self._rows.items():
                layer = self.manager.layers.get(name)
                if layer is None:
                    continue
                if row["visible"].get_value_as_bool() != layer.visible:
                    row["visible"].set_value(layer.visible)
        finally:
            self._syncing = False

    # ---------------------------------------------------------------- teardown
    def destroy(self) -> None:
        if self._on_tick in self.manager.on_tick:
            self.manager.on_tick.remove(self._on_tick)
        if self.window is not None:
            self.window.destroy()
            self.window = None
        if self in _PANELS:
            _PANELS.remove(self)


def open_panel(manager: WeatherLayers | None = None) -> WeatherPanel:
    """Open (or re-open) the panel for a manager."""
    close_panels()
    return WeatherPanel(manager)


def close_panels() -> None:
    for panel in list(_PANELS):
        panel.destroy()
