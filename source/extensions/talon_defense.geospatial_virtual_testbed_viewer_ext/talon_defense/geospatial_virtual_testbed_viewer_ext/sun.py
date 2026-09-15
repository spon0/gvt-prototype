import carb
from pxr import UsdGeom, UsdLux, Gf, Sdf
import datetime, math, weakref, calendar, enum

from .constants import *
from .utils import *

import omni.kit.pipapi
omni.kit.pipapi.install("skyfield")
from skyfield.api import load
from skyfield.framelib import itrs

class Sun:

    def __init__(self, parent):

        self._gvt_manager = parent

        eph = load('de421.bsp')
        self._sun_eph = eph['sun']
        self._earth_eph = eph['earth']
        self._timescale = load.timescale()
        self._stage = None
        self._time_change_sub = None

        pass

    def _remove_default_lights(self, stage):
        """Strip any light Kit itself put in the stage before we add ours.

        `content.emptyStageOnStart = true` (and the usd_viewer template's own
        sample content) means every new/opened stage already has a built-in
        lighting rig authored by Kit's stage-template machinery -- typically
        a DistantLight at a path like `/Environment/defaultLight`, sometimes
        paired with a sky DomeLight. That rig is *exactly* what was showing
        up as a stray specular highlight on the earth, and a second, brighter
        DomeLight competing with our own starfield's DomeLight is the most
        likely reason the starfield wasn't visible -- multiple dome lights in
        one stage don't blend the way you'd want; one just wins.

        Rather than hardcode a path that could change between Kit/template
        versions, walk the whole stage and remove anything with UsdLux's
        LightAPI applied (every concrete UsdLux light type -- DistantLight,
        DomeLight, SphereLight, RectLight, DiskLight, CylinderLight --
        auto-applies LightAPI, so this catches all of them uniformly),
        except our own sun and starfield once they exist.
        """
        our_scopes = (LIGHTS_SCOPE_PATH, BACKGROUND_SCOPE_PATH)

        def _is_ours(path):
            return any(path == scope or path.HasPrefix(scope) for scope in our_scopes)

        stray_paths = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.HasAPI(UsdLux.LightAPI) and not _is_ours(prim.GetPath())
        ]

        for path in stray_paths:
            # carb.log_warn, not print() -- prints don't reliably show up in
            # Kit's own log file, which makes this stuff hard to confirm
            # after the fact.
            carb.log_warn(
                "[talon_defense.geospatial_virtual_testbed_viewer_ext] removing "
                f"pre-existing default light at {path}"
            )
            stage.RemovePrim(path)

    def create_sun(self, stage):

        self._stage = stage
        self._remove_default_lights(stage)

        if not stage.GetPrimAtPath(LIGHTS_SCOPE_PATH):
            UsdGeom.Scope.Define(stage, LIGHTS_SCOPE_PATH)

        sun_light = UsdLux.DistantLight.Define(stage, SUN_LIGHT_PATH)
        sun_light.CreateIntensityAttr(SUN_INTENSITY)
        sun_light.CreateColorAttr(SUN_COLOR)

        xf = UsdGeom.Xformable(sun_light.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xf.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))

        self._time_change_sub = self._gvt_manager.get_simulation_event_stream().create_subscription_to_pop_by_type(
            event_type=SIMULATION_TIME_UPDATED,
            fn=self._on_time_change
        )

    def _on_time_change(self, event):
        """Advances the sun's position to the simulation clock."""
        dt = self._gvt_manager.get_simulation_utc_datetime()
        sf_time = self._timescale.from_datetime(dt)
        g = self._earth_eph.at(sf_time).observe(self._sun_eph).apparent()
        x, y, z = g.frame_xyz(itrs).km
        r = math.sqrt(x*x + y*y + z*z)
        theta = math.degrees(math.acos(z/r))
        phi = math.degrees(math.atan2(y, x))

        sun_xform = UsdGeom.Xform(self._stage.GetPrimAtPath(SUN_LIGHT_PATH))
        for op in sun_xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                op.Set(Gf.Vec3d(theta, 0, phi + 90.0))