import carb
import omni.ext
import omni.kit.app
import omni.usd
from omni.kit.viewport.utility import (
    disable_selection,
    frame_viewport_prims,
    get_active_viewport
)

from .constants import *

class CameraControl:

    def __init__(self):

        self._selection_disabler = None


    def apply_selection_lock_and_pivot(self, viewport_api, stage):
            # Keeps selection disabled for as long as this extension is alive --
            # disable_selection()'s effect lasts only as long as something holds
            # a reference to the object it returns, so this has to be stashed on
            # self rather than left as a throwaway local.
            self._selection_disabler = disable_selection(viewport_api)
            carb.log_warn(f"self._selection_disabler value {self._selection_disabler}")

            # Programmatic equivalent of selecting the earth and pressing "F":
            # frames it, and -- like a manual focus -- sets it as the camera's
            # orbit pivot ("center of interest"), so orbiting always tumbles
            # around the earth instead of whatever point the camera last
            # happened to be looking at.
            if stage.GetPrimAtPath(EARTH_PATH):
                frame_viewport_prims(viewport_api, [str(EARTH_PATH)])

            carb.log_warn(
                "[talon_defense.geospatial_virtual_testbed_viewer_ext] selection "
                f"disabled; orbit pivot set to {EARTH_PATH}"
            )