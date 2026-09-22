"""Transparent lat/lon data shell with per-vertex color + opacity, fed by NOAA GRIB.

Script Editor quick start (inside Kit):

    from talon_defense.weather_overlay import start
    s = start()                                   # synthetic demo field, no data needed
    s = start("C:/data/gfs/*t2m*.grib2", {"shortName": "2t"})
    s.stop()

Building blocks, if you drive the updates yourself:

    seq    = GribSequence(files, select)          # background ecCodes decode
    sphere = GribSphere(stage, "/World/GribSphere", seq.grid, radius=100)
    sphere.set_field(seq.load(0), Style(cmap="turbo"))      # once
    sphere.set_colors(rgb_linear_Nx3, alpha_N)               # any frame, any source
"""
from .grib_io import GribField, GribGrid, MessageInfo, inventory, list_messages, read_field
from .mesh import build_shell_mesh, unit_vectors
from .style import COLORMAPS, OpacityRamp, Style, make_lut

from .sequence import GribSequence, SyntheticSequence

try:
    import pxr  # noqa: F401
except ImportError:  # plain Python without USD: GRIB/mesh/style helpers only
    pxr = None

if pxr is not None:
    from .player import GribPlayer, subscribe_update
    from .session import Session, active, start, stop_all
    from .usd_sphere import GribSphere, enable_fractional_cutout_opacity

try:
    import omni.ext  # noqa: F401
except ImportError:  # not running inside Kit
    pass
else:
    from .extension import GribSphereExtension
