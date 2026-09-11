# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import carb
import omni.ext
import omni.kit.app
import omni.usd
from omni.usd import StageEventType
from pxr import UsdGeom, UsdLux, Gf, Sdf

from .stage_loading import LoadingManager
from .stage_management import StageManager

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


# Any class derived from `omni.ext.IExt` in top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when extension
# gets enabled and `on_startup(ext_id)` will be called. Later when extension
# gets disabled on_shutdown() is called.
class Extension(omni.ext.IExt):
    """This extension manages creating the loading and stage
    messaging managers"""
    def on_startup(self, ext_id):
        """This is called every time the extension is activated."""
        # Internal messaging state
        self._loading_manager: LoadingManager = LoadingManager()
        self._stage_manager: StageManager = StageManager()

        # Sun animation state. _sun_rotate_op is set once the light exists;
        # _on_app_update no-ops until then, so subscription order doesn't
        # matter.
        self._sun_rotate_op = None
        self._sun_angle = 0.0

        usd_context = omni.usd.get_context()

        # Subscribe first so we don't miss an OPENED event that fires while
        # this extension is still loading.
        self._stage_event_sub = usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event, name="talon_defense.geospatial_virtual_testbed_viewer_ext stage event"
        )

        # Drives the sun's rotation every rendered frame, independent of
        # whether the Kit timeline is playing -- appropriate for a viewer
        # app that a streaming client may open without ever pressing Play.
        self._app_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_app_update, name="talon_defense.geospatial_virtual_testbed_viewer_ext sun update"
        )

        # In case a stage is already open by the time we get here (common
        # when this extension is enabled manually after the app has booted),
        # try immediately too.
        stage = usd_context.get_stage()
        if stage is not None:
            self._remove_default_lights(stage)
            self._create_earth(stage)
            self._create_lighting(stage)
            self._create_starfield(stage)

    def _on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                # New stage -- the previous light prim (and _sun_rotate_op
                # pointing at it) no longer exists in it. Reset before
                # (re)creating so _on_app_update never ticks a stale op.
                self._sun_rotate_op = None
                self._sun_angle = 0.0
                self._remove_default_lights(stage)
                self._create_earth(stage)
                self._create_lighting(stage)
                self._create_starfield(stage)

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

    def _on_app_update(self, event):
        """Advances the sun's rotation once per rendered frame."""
        if self._sun_rotate_op is None:
            return
        dt = event.payload.get("dt", 0.0) if event.payload else 0.0
        self._sun_angle = (self._sun_angle + SUN_DEGREES_PER_SECOND * dt) % 360.0
        self._sun_rotate_op.Set(self._sun_angle)

    def _ensure_world_prim(self, stage):
        # The Kit Base Editor template's default stage doesn't always ship a
        # /World xform, so make sure one exists before parenting under it.
        world_path = Sdf.Path("/World")
        if not stage.GetPrimAtPath(world_path):
            UsdGeom.Xform.Define(stage, world_path)
            stage.SetDefaultPrim(stage.GetPrimAtPath(world_path))

    def _create_earth(self, stage):
        # Guard against re-creating it every time OPENED fires (new stage,
        # hot-reload during dev, etc.)
        if stage.GetPrimAtPath(EARTH_PATH):
            return

        self._ensure_world_prim(stage)

        earth_xform = UsdGeom.Xform.Define(stage, EARTH_PATH)
        earth_prim = earth_xform.GetPrim()

        if not earth_prim.GetReferences().AddReference(EARTH_ASSET_PATH):
            carb.log_error(
                "[talon_defense.geospatial_virtual_testbed_viewer_ext] "
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
            "[talon_defense.geospatial_virtual_testbed_viewer_ext] referenced earth model "
            f"at {EARTH_PATH} from {EARTH_ASSET_PATH} (scale={EARTH_DISPLAY_SCALE})"
        )

    def _create_lighting(self, stage):
        sun_prim = stage.GetPrimAtPath(SUN_LIGHT_PATH)
        if sun_prim:
            # Lights already exist (e.g. a redundant OPENED event on the
            # same stage) -- just reattach the rotate op to whatever is
            # already there so _on_app_update keeps working, rather than
            # silently going stale.
            xformable = UsdGeom.Xformable(sun_prim)
            ops = xformable.GetOrderedXformOps()
            self._sun_rotate_op = ops[0] if ops else xformable.AddRotateXOp()
            return

        if not stage.GetPrimAtPath(LIGHTS_SCOPE_PATH):
            UsdGeom.Scope.Define(stage, LIGHTS_SCOPE_PATH)

        sun_light = UsdLux.DistantLight.Define(stage, SUN_LIGHT_PATH)
        sun_light.CreateIntensityAttr(SUN_INTENSITY)
        sun_light.CreateColorAttr(SUN_COLOR)
        sun_light.CreateAngleAttr(SUN_ANGLE)

        # You're right that the sun should spin about the up/polar axis --
        # that's the physically correct model (Earth rotates about its own
        # polar axis; the sun's direction is ~fixed in space, so in
        # Earth-fixed coordinates it's equivalent to rotate the sun's
        # direction the other way about that same axis). The previous
        # version rotated about a horizontal axis instead, which happened
        # to produce *a* moving light but not one that matches the mesh's
        # actual poles.
        #
        # The wrinkle: DistantLight emits along its local -Z axis by
        # default, with no rotation applied. On a Z-up stage that raw
        # direction points straight down the same axis we now want to
        # spin it about -- and rotating a vector about an axis it's
        # already aligned with does nothing (a spinning globe's own axis
        # doesn't move either). So on Z-up we first apply a fixed 90deg
        # tilt about a horizontal axis to swing the light's direction out
        # into the equatorial plane, *then* animate the spin about the up
        # axis on top of that. On a Y-up stage no tilt is needed -- the
        # default -Z direction is already perpendicular to Y.
        #
        # Xform op order note: ops are applied local-point-first in the
        # order they're LAST added (innermost), and the FIRST-added op
        # ends up outermost. So the tilt must be added *after* the spin
        # for the tilt to apply first.
        sun_xformable = UsdGeom.Xformable(sun_light)
        sun_xformable.ClearXformOpOrder()
        if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
            self._sun_rotate_op = sun_xformable.AddRotateZOp()  # added first -> outermost -> animated spin
            sun_xformable.AddRotateXOp().Set(90.0)              # added second -> innermost -> fixed tilt
        else:
            self._sun_rotate_op = sun_xformable.AddRotateYOp()
        self._sun_rotate_op.Set(self._sun_angle)

        carb.log_warn(
            "[talon_defense.geospatial_virtual_testbed_viewer_ext] created sun light "
            f"under {LIGHTS_SCOPE_PATH}"
        )

    def _create_starfield(self, stage):
        # Guard against re-creating it every time OPENED fires (new stage,
        # hot-reload during dev, etc.) -- same pattern as _create_earth.
        if stage.GetPrimAtPath(STARFIELD_PATH):
            return

        if not stage.GetPrimAtPath(BACKGROUND_SCOPE_PATH):
            UsdGeom.Scope.Define(stage, BACKGROUND_SCOPE_PATH)

        starfield = UsdLux.DomeLight.Define(stage, STARFIELD_PATH)
        # Explicit Sdf.AssetPath, not a raw string -- inputs:texture:file is
        # asset-typed, and leaving USD to guess the conversion is exactly the
        # kind of thing that can silently leave the attribute unset (dome
        # exists, texture never loads, so nothing renders -- indistinguishable
        # from "no stars"). This mirrors how the earth's diffuse texture is
        # wired up.
        starfield.CreateTextureFileAttr(Sdf.AssetPath(STARFIELD_TEXTURE_PATH))
        starfield.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        starfield.CreateIntensityAttr(STARFIELD_INTENSITY)

        # This DomeLight exists purely to be *seen* as a backdrop -- it
        # should never actually light the scene (that's the sun's and,
        # previously, the ambient light's job). Zeroing both diffuse and
        # specular makes it a visual-only skybox: the renderer still shows
        # its texture wherever the camera sees empty space, but it
        # contributes nothing to how the Earth (or anything else) is lit.
        starfield.CreateDiffuseAttr(0.0)
        starfield.CreateSpecularAttr(0.0)

        carb.log_warn(
            "[talon_defense.geospatial_virtual_testbed_viewer_ext] created starfield "
            f"backdrop at {STARFIELD_PATH} from {STARFIELD_TEXTURE_PATH}"
        )

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used to
        clean up the extension state."""
        # Resetting the state.
        self._stage_event_sub = None
        self._app_update_sub = None
        self._sun_rotate_op = None
        if self._loading_manager:
            self._loading_manager.on_shutdown()
            self._loading_manager = None
        if self._stage_manager:
            self._stage_manager.on_shutdown()
            self._stage_manager = None
