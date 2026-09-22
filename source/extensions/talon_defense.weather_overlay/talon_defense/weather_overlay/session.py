"""One-call setup for the Script Editor or an extension: sequence + sphere + player."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from . import _log
from .player import GribPlayer
from .style import Style
from .usd_sphere import GribSphere, enable_fractional_cutout_opacity

_ACTIVE: list["Session"] = []


@dataclass
class Session:
    sphere: GribSphere
    sequence: Any
    player: GribPlayer

    def stop(self, remove_prim: bool = False) -> None:
        self.player.stop()
        self.sequence.close()
        if remove_prim:
            self.sphere.remove()
        if self in _ACTIVE:
            _ACTIVE.remove(self)


def active() -> Session | None:
    """The most recently started session (e.g. the one the extension autostarted)."""
    return _ACTIVE[-1] if _ACTIVE else None


def stop_all(remove_prims: bool = False) -> None:
    for session in list(_ACTIVE):
        session.stop(remove_prims)


def start(grib: str | Iterable[str] | None = None, select: Mapping[str, Any] | None = None, *,
          style: Style | None = None, path: str = "/World/GribSphere", radius: float = 100.0,
          stride: int | None = None, max_points: int | None = 400_000,
          hours_per_second: float = 6.0, clock: str = "wall", unlit: bool = True,
          double_sided: bool = False, emissive_scale: float = 1.0, material: str = "mdl",
          emissive_intensity: float = 1000.0, backend: str = "auto", stage: Any = None,
          fractional_cutout: bool = True, replace: bool = True) -> Session:
    """Load GRIB frames, build the shell and start playing.

    ``grib=None`` runs the synthetic demo field (no files or ecCodes needed).
    Re-running replaces the previous session unless ``replace=False``.

        from talon_defense.weather_overlay import start, Style, OpacityRamp
        s = start("C:/data/gfs/*.grib2", {"shortName": "pwat"},
                  style=Style(cmap="viridis", vmin=0, vmax=65, opacity=OpacityRamp(20, 60, 0, 0.85)))
        s.player.pause(); s.player.seek(24); s.stop()
    """
    if replace:
        stop_all()
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD stage is open")
    if fractional_cutout:
        for line in enable_fractional_cutout_opacity(True, stage):
            _log.info(f"render settings: {line}")

    if grib is None:
        from .sequence import SyntheticSequence

        sequence: Any = SyntheticSequence()
        style = style or Style(cmap="blues_r", vmin=0.0, vmax=1.0,
                               opacity=lambda v: (v - 0.25).clip(0, 0.75) / 0.75 * 0.9)
    else:
        from .sequence import GribSequence

        sequence = GribSequence(grib, select, stride=stride, max_points=max_points)
        style = style or Style()

    sphere = GribSphere(stage, path, sequence.grid, radius=radius, unlit=unlit,
                        double_sided=double_sided, emissive_scale=emissive_scale, material=material,
                        emissive_intensity=emissive_intensity, backend=backend)
    player = GribPlayer(sphere, sequence, style, hours_per_second=hours_per_second, clock=clock,
                        fractional_cutout=fractional_cutout)
    player.start()
    session = Session(sphere, sequence, player)
    _ACTIVE.append(session)
    _log.info(f"Playing {len(sequence)} frame(s) on {path} ({sequence.grid.n_points:,} vertices, "
              f"writer={sphere.writer_name})")
    return session
