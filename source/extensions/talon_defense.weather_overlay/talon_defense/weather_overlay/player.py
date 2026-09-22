"""Drives a GribSphere from a sequence on Kit's per-frame update event."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from . import _log
from .style import Style
from .usd_sphere import GribSphere, enable_fractional_cutout_opacity

UPDATE_OBSERVER = f"{__package__}.player"


def subscribe_update(callback: Callable[[float], None], name: str = UPDATE_OBSERVER) -> Any:
    """Call ``callback(dt_seconds)`` every app update. Keep the returned handle alive.

    Uses Events 2.0 (carb.eventdispatcher, Kit 106.5+/107+/109) and falls back to
    the legacy update event stream on older Kits.
    """
    import omni.kit.app

    def _dt(event: Any) -> float:
        try:
            return float(event["dt"])
        except Exception:  # noqa: BLE001 - legacy IEvent
            return float(event.payload["dt"])

    def _on_event(event: Any) -> None:
        callback(_dt(event))

    event_name = getattr(omni.kit.app, "GLOBAL_EVENT_UPDATE", None)
    if event_name is not None:
        import carb.eventdispatcher

        return carb.eventdispatcher.get_eventdispatcher().observe_event(
            observer_name=name, event_name=event_name, on_event=_on_event)
    return omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        _on_event, name=name)


def _release(handle: Any) -> None:
    for method in ("reset", "unsubscribe"):
        fn = getattr(handle, method, None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
            return


class GribPlayer:
    """Plays a sequence on a sphere, interpolating between GRIB time steps.

    Args:
        hours_per_second: data hours advanced per wall-clock second
            (6 = a 5-day GFS run plays in 20 s).
        interpolate: blend linearly between neighbouring time steps (smooth motion)
            instead of stepping.
        clock: "wall" advances by frame dt while playing; "timeline" follows the Kit
            timeline (play/pause/scrub in the UI), data hour = timeline seconds * hours_per_second.
        max_upload_hz: cap on how often new colors are pushed (the colorize + upload is
            the per-frame cost; 30 Hz is smooth and halves the work at 60 fps).
        lookahead: frames decoded ahead of the play head.
        fractional_cutout: keep fractional cutout opacity on for the viewport's render
            products during the first ``RENDER_SETTINGS_WATCH_S`` seconds (they are created
            after extensions start) and log each render product's mode once.
    """

    RENDER_SETTINGS_WATCH_S = 30.0

    def __init__(self, sphere: GribSphere, sequence: Any, style: Style | None = None, *,
                 hours_per_second: float = 6.0, interpolate: bool = True, loop: bool = True,
                 clock: str = "wall", max_upload_hz: float | None = 30.0, lookahead: int = 3,
                 on_frame: Callable[["GribPlayer"], None] | None = None,
                 fractional_cutout: bool = False):
        if sequence.grid.n_points != sphere.n_points:
            raise ValueError("sphere was built for a different grid than the sequence")
        self.sphere = sphere
        self.sequence = sequence
        self.style = style or Style()
        self.hours_per_second = float(hours_per_second)
        self.interpolate = interpolate
        self.loop = loop
        self.clock = clock
        self.max_upload_hz = max_upload_hz
        self.lookahead = max(1, int(lookahead))
        self.on_frame = on_frame
        self.fractional_cutout = fractional_cutout
        self._rs_started = time.perf_counter()
        self._rs_last = 0.0
        self._rs_logged: set[str] = set()
        self.playing = False
        self.hours = float(sequence.hours[0])
        self._sub: Any = None
        self._timeline: Any = None
        self._last_key: tuple | None = None
        self._last_upload = 0.0
        self.last_upload_ms = 0.0

    # ---------------------------------------------------------------- control
    def start(self, play: bool = True) -> "GribPlayer":
        if self._sub is None:
            if self.clock == "timeline":
                import omni.timeline

                self._timeline = omni.timeline.get_timeline_interface()
            self._sub = subscribe_update(self._on_update)
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

    def seek(self, hours: float) -> None:
        self.hours = float(hours)
        self.render(force=True)

    @property
    def span_hours(self) -> float:
        return float(self.sequence.hours[-1] - self.sequence.hours[0])

    @property
    def current_time(self):
        import datetime as _dt

        return self.sequence.times[0] + _dt.timedelta(hours=self._wrapped_hours() - float(self.sequence.hours[0]))

    # ---------------------------------------------------------------- per frame
    def _on_update(self, dt: float) -> None:
        try:
            self.step(dt)
        except Exception as exc:  # noqa: BLE001 - never spam the log every frame
            _log.error(f"GribPlayer stopped: {exc!r}")
            self.stop()

    def _watch_render_settings(self) -> None:
        now = time.perf_counter()
        if now - self._rs_started > self.RENDER_SETTINGS_WATCH_S or now - self._rs_last < 1.0:
            return
        self._rs_last = now
        for line in enable_fractional_cutout_opacity(True, self.sphere.stage):
            if line not in self._rs_logged:
                self._rs_logged.add(line)
                _log.info(f"render settings: {line}")

    def step(self, dt: float) -> bool:
        """Advance the clock by ``dt`` seconds and render. Returns True if colors were uploaded."""
        if self.fractional_cutout:
            self._watch_render_settings()
        if self.clock == "timeline" and self._timeline is not None:
            self.hours = float(self.sequence.hours[0]) + \
                self._timeline.get_current_time() * self.hours_per_second
        elif self.playing:
            self.hours += dt * self.hours_per_second
        if self.max_upload_hz:
            now = time.perf_counter()
            if now - self._last_upload < 1.0 / self.max_upload_hz:
                return False
        return self.render()

    def _wrapped_hours(self) -> float:
        hs = self.sequence.hours
        h0, span = float(hs[0]), float(hs[-1] - hs[0])
        if span <= 0:
            return h0
        if self.loop:
            return h0 + (self.hours - h0) % span
        return min(max(self.hours, h0), h0 + span)

    def render(self, force: bool = False) -> bool:
        if not self.sphere.is_valid():
            raise RuntimeError(f"{self.sphere.path} was removed from the stage")
        hs = self.sequence.hours
        n = len(hs)
        h = self._wrapped_hours()
        i = int(np.clip(np.searchsorted(hs, h, side="right") - 1, 0, n - 1))
        w = 0.0
        if self.interpolate and i < n - 1:
            w = float((h - hs[i]) / (hs[i + 1] - hs[i]))
        key = (i, round(w, 3))
        if key == self._last_key and not force:
            return False

        ahead = [(i + k) % n if self.loop else i + k for k in range(1, self.lookahead + 1)]
        self.sequence.prefetch(ahead)
        a = self.sequence.get(i)
        b = self.sequence.get(i + 1) if w > 0.0 else None
        if a is None or (w > 0.0 and b is None):
            return False  # still decoding: keep showing the previous frame, retry next update

        values = a if w == 0.0 else a + (b - a) * np.float32(w)
        if self.style.vmin is None or self.style.vmax is None:
            self.style.autoscale(self.sequence.get(0) if self.sequence.get(0) is not None else values)
        t0 = time.perf_counter()
        self.sphere.set_field(values, self.style)
        self._last_upload = time.perf_counter()
        self.last_upload_ms = (self._last_upload - t0) * 1000.0
        self._last_key = key
        if self.on_frame is not None:
            self.on_frame(self)
        return True

    def restyle(self, style: Style | None = None) -> None:
        """Apply a new or edited style immediately."""
        if style is not None:
            self.style = style
        self.render(force=True)
