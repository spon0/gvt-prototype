"""Kit-side smoke test (runs under `repo test`): builds a small shell on an in-memory
stage from the synthetic field - no GRIB files or ecCodes needed."""
import numpy as np
import omni.kit.test
from pxr import Usd, UsdGeom

import talon_defense.weather_overlay as wo


class TestWeatherOverlay(omni.kit.test.AsyncTestCase):
    async def test_synthetic_shell(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        seq = wo.SyntheticSequence(n_frames=2, lat_step=10, lon_step=10)
        sphere = wo.GribSphere(stage, "/World/GribSphere", seq.grid, radius=64.0, backend="usd")
        style = wo.Style(cmap="blues_r", vmin=0.0, vmax=1.0, opacity=0.5)
        sphere.set_field(seq.get(1), style)

        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/GribSphere/Shell"))
        primvars = UsdGeom.PrimvarsAPI(mesh)
        colors = np.array(primvars.GetPrimvar("dataColor").Get())
        alpha = np.array(primvars.GetPrimvar("dataOpacity").Get())
        self.assertEqual(colors.shape, (seq.grid.n_points, 3))
        self.assertTrue(np.allclose(alpha, 0.5))
        self.assertEqual(len(mesh.GetFaceVertexCountsAttr().Get()), (seq.grid.shape[0] - 1) * seq.grid.shape[1])
