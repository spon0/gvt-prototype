import carb
from pxr import UsdGeom, Gf, Sdf

from .constants import *


class Earth:

    def __init__(self):
        pass

    def create_earth(self, stage):
        # Guard against re-creating it every time OPENED fires (new stage,
        # hot-reload during dev, etc.)
        if stage.GetPrimAtPath(EARTH_PATH):
            return

        self._ensure_world_prim(stage)

        earth_xform = UsdGeom.Xform.Define(stage, EARTH_PATH)
        earth_prim = earth_xform.GetPrim()

        if not earth_prim.GetReferences().AddReference(EARTH_ASSET_PATH):
            carb.log_error(
                f"failed to reference earth model from {EARTH_ASSET_PATH}"
            )
            return

        # Center at the origin and apply the display-scale knockdown
        # documented above (earth_medium.usda is authored at real-world
        # meter scale).
        earth_xform.ClearXformOpOrder()
        earth_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        earth_xform.AddScaleOp().Set(Gf.Vec3d(EARTH_DISPLAY_SCALE, EARTH_DISPLAY_SCALE, EARTH_DISPLAY_SCALE))

        carb.log_warn(
            "referenced earth model "
            f"at {EARTH_PATH} from {EARTH_ASSET_PATH} (scale={EARTH_DISPLAY_SCALE})"
        )

    def _ensure_world_prim(self, stage):
        # The Kit Base Editor template's default stage doesn't always ship a
        # /World xform, so make sure one exists before parenting under it.
        world_path = Sdf.Path("/World")
        if not stage.GetPrimAtPath(world_path):
            UsdGeom.Xform.Define(stage, world_path)
            stage.SetDefaultPrim(stage.GetPrimAtPath(world_path))