# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import os
import time

import carb
import omni.ext
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import (
    disable_selection,
    frame_viewport_prims,
    get_active_viewport,
    get_viewport_from_window_name,
)
from omni.usd import StageEventType, StageRenderingEventType
from pxr import UsdGeom, UsdLux, Gf, Sdf, Vt

from datetime import datetime, timedelta, UTC

from .constants import *

from .stage_loading import LoadingManager
from .stage_management import StageManager

from .camera_control import CameraControl
from .earth import Earth
from .sun import Sun

# Any class derived from `omni.ext.IExt` in top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when extension
# gets enabled and `on_startup(ext_id)` will be called. Later when extension
# gets disabled on_shutdown() is called.
class GvtManager(omni.ext.IExt):
    """This extension manages creating the loading and stage
    messaging managers"""
    def on_startup(self, ext_id):
        """This is called every time the extension is activated."""

        self._active = True

        # Internal messaging state
        self._loading_manager: LoadingManager = LoadingManager()
        self._stage_manager: StageManager = StageManager()
        self._camera: CameraControl = CameraControl()
        self._earth: Earth = Earth()
        self._sun: Sun = Sun(self)

        usd_context = omni.usd.get_context()

        events_interface = carb.events.acquire_events_interface()
        self._simulation_event_stream = events_interface.create_event_stream()
        self._time_scale = 1000.0
        self._time = time.time()
        self._prev_time = time.time()
        self.utc_datetime = datetime.now(UTC)
        self._clock_manager = None

        # Subscribe first so we don't miss an OPENED event that fires while
        # this extension is still loading.
        self._stage_event_sub = usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event, name="talon_defense.geospatial_virtual_testbed_viewer_ext stage event"
        )

        # Drives the sun's rotation every rendered frame, independent of
        # whether the Kit timeline is playing -- appropriate for a viewer
        # app that a streaming client may open without ever pressing Play.
        self._app_update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_app_update, name="sun update"
        )

        # In case a stage is already open by the time we get here (common
        # when this extension is enabled manually after the app has booted),
        # try immediately too.
        stage = usd_context.get_stage()
        if stage is not None:
            self._initialize_stage_and_environment(stage)

    def get_simulation_event_stream(self):
        return self._simulation_event_stream

    def get_simulation_utc_datetime(self) -> datetime:
        return self.utc_datetime

    def _make_axes(self):
        stage = omni.usd.get_context().get_stage()
        L = 6371.0   # axis length — scale to your stage units

        axes = {
            "X": (Gf.Vec3f(L, 0, 0), Gf.Vec3f(1, 0, 0)),
            "Y": (Gf.Vec3f(0, L, 0), Gf.Vec3f(0, 1, 0)),
            "Z": (Gf.Vec3f(0, 0, L), Gf.Vec3f(0, 0, 1)),
        }

        UsdGeom.Xform.Define(stage, "/World/Axes")

        for name, (end, color) in axes.items():
            curve = UsdGeom.BasisCurves.Define(stage, f"/World/Axes/{name}")
            curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
            curve.CreateCurveVertexCountsAttr().Set(Vt.IntArray([2]))
            curve.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), end]))
            curve.CreateWidthsAttr().Set(Vt.FloatArray([10.0, 10.0]))
            curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)

            cp = curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
            cp.Set(Vt.Vec3fArray([color]))

        print(UsdGeom.GetStageUpAxis(stage))

    def _on_app_update(self, event):
        self._time = time.time()
        delta_time = self._time - self._prev_time

        if event.type == int(StageRenderingEventType.NEW_FRAME) and self._active:

            self.utc_datetime += (delta_time * timedelta(seconds=self._time_scale))
            self._simulation_event_stream.push(SIMULATION_TIME_UPDATED)
            self._simulation_event_stream.pump()

        self._prev_time = self._time

    def _initialize_stage_and_environment(self, stage):
            if stage is not None:
                # Set up-axis to Z
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

                self._earth.create_earth(stage)
                self._sun.create_sun(stage)
                self._create_starfield(stage)
                self._camera.apply_selection_lock_and_pivot(get_active_viewport(), stage)

    def _on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._initialize_stage_and_environment(omni.usd.get_context().get_stage())

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
            "created starfield "
            f"backdrop at {STARFIELD_PATH} from {STARFIELD_TEXTURE_PATH}"
        )

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used to
        clean up the extension state."""
        # Resetting the state.
        self._stage_event_sub = None
        self._app_update_sub = None

        # Dropping the last reference restores normal selection behavior --
        # see disable_selection()'s docs on _setup_camera_and_selection.
        self._selection_disabler = None
        if self._loading_manager:
            self._loading_manager.on_shutdown()
            self._loading_manager = None
        if self._stage_manager:
            self._stage_manager.on_shutdown()
            self._stage_manager = None
