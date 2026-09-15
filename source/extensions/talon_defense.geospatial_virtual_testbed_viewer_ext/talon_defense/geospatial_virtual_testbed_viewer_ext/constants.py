import os
import carb
from pxr import Gf, Sdf

EARTH_PATH = Sdf.Path("/World/Earth")

# earth_medium.usda ships alongside this module (same folder as
# extension.py). Resolving it via __file__ rather than a hardcoded path
# means it keeps working whether Kit loads this module from
# source/extensions/... or from the _build/.../exts/... symlinked copy.
_EXT_MODULE_DIR = os.path.dirname(__file__)
EARTH_ASSET_PATH = os.path.join(_EXT_MODULE_DIR, "earth_medium.usda").replace(os.sep, "/")

# earth_medium.usda is authored at real Earth scale (mean radius ~6,371,000
# stage units, assuming metersPerUnit=1). USD does NOT auto-rescale a
# reference to match the referencing stage's metersPerUnit, and most Kit
# app templates default new stages to centimeter units -- so importing this
# unscaled will look enormous and can put it outside the default camera's
# clipping range. This knocks it down to roughly the same size as the old
# placeholder sphere (radius 50) for now. Set to 1.0 once you're ready to
# work at true geospatial scale (e.g. once Cesium is driving the globe).
EARTH_DISPLAY_SCALE = 1.0e-5

LIGHTS_SCOPE_PATH = Sdf.Path("/World/Lights")
SUN_LIGHT_PATH = LIGHTS_SCOPE_PATH.AppendChild("SunLight")

# DistantLight = parallel rays, like real sunlight at planetary scale.
# `angle` is the light's angular size in degrees (0.53 is the real sun's,
# giving soft-edged shadows); intensity/color are starting points to taste.
SUN_INTENSITY = 3000.0
SUN_COLOR = Gf.Vec3f(1.0, 0.96, 0.9)  # slightly warm
SUN_ANGLE = 0.53

# Real-time day/night cycle speed. 15 deg/s -> a full rotation every 24s,
# fast enough to be obviously "moving" in a live prototype demo. Negative
# so the sun sweeps the same way Earth's real rotation makes the sun
# appear to move (west to east felt backwards -- east to west is correct).
SUN_DEGREES_PER_SECOND = -15.0

BACKGROUND_SCOPE_PATH = Sdf.Path("/World/Background")
STARFIELD_PATH = BACKGROUND_SCOPE_PATH.AppendChild("Starfield")

# starfield.png ships alongside this module in textures/, same as
# earth_diffuse.jpg -- resolved via __file__ for the same reason (works
# whether Kit loads this module from source/extensions/... or the
# _build/.../exts/... symlinked copy).
STARFIELD_TEXTURE_PATH = os.path.join(_EXT_MODULE_DIR, "textures", "starfield.png").replace(os.sep, "/")

# NOT a small number on purpose. `intensity` scales both the light's
# contribution to the scene AND the raw brightness of what's shown as
# background (diffuse/specular=0 below only zeroes out the former) -- and
# this app's exposure/tonemap curve treats "1.0" as essentially black, the
# same reason the sun needed SUN_INTENSITY=3000 instead of 1.0 to read as
# daylight. A dim procedural starfield at intensity 1.0 was very likely
# rendering, just far below the visible floor. Same order of magnitude as
# the sun so it's viewable against this app's exposure setup.
STARFIELD_INTENSITY = 2500.0

# Matches the window name the app's own setup extension (`setup.py`,
# `get_viewport_from_window_name("Viewport")`) already assumes for this app.
VIEWPORT_WINDOW_NAME = "Viewport"


# Simulation events
SIMULATION_NEW_FRAME: int = carb.events.type_from_string("gvt.NEW_FRAME")
SIMULATION_TIME_UPDATED: int = carb.events.type_from_string("gvt.SIMULATION_TIME_UPDATED")

# WGS84 vals in kilometer
WGS84_SEMIMAJOR = 6378.137
WGS84_SEMIMINOR = 6356.752314245
WGS84_RADIUS = 6371.0