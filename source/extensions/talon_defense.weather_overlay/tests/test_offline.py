"""Offline tests: GRIB decode, mesh winding, USD authoring, player loop (Kit stubbed).

    pip install usd-core eccodes numpy pytest
    python tests/make_test_grib.py tests/_data
    pytest -q tests/test_offline.py        (or: python tests/test_offline.py)
"""
from __future__ import annotations

import os
import sys
import time
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from make_test_grib import analytic, main as make_data  # noqa: E402

from talon_defense.weather_overlay import (GribGrid, GribSequence, OpacityRamp, Style,  # noqa: E402
                                     SyntheticSequence, build_shell_mesh, unit_vectors)

DATA = os.environ.get("GRIB_TEST_DATA", os.path.join(HERE, "_data"))


@pytest.fixture(scope="session", autouse=True)
def data():
    if not os.path.exists(os.path.join(DATA, "lambert_conus.grib2")):
        make_data(DATA)
    return DATA


def _gfs(**kw) -> GribSequence:
    return GribSequence(os.path.join(DATA, "gfs_2t_f*.grib2"), {"shortName": "2t"}, **kw)


# --------------------------------------------------------------------------- GRIB
def test_decode_full_resolution_matches_analytic():
    seq = _gfs(stride=1, max_points=None)
    assert seq.grid.shape == (721, 1440) and seq.grid.wrap_lon
    assert list(seq.hours) == [0, 3, 6, 9, 12]
    for i, h in enumerate(seq.hours):
        v = seq.load(i)
        truth = analytic(seq.grid.lats, seq.grid.lons, h).reshape(-1)
        ok = np.isfinite(v)
        assert np.abs(v[ok] - truth[ok]).max() < 0.01
        lat, lon = seq.grid.lats.reshape(-1), seq.grid.lons.reshape(-1)
        box = (lat >= 10) & (lat <= 20) & (lon >= 100) & (lon <= 120)
        assert np.array_equal(~ok, box), "missing-value bitmap must become NaN exactly in the box"
    seq.close()


@pytest.mark.parametrize("stride,rows,cols", [(2, 361, 720), (3, 241, 480), (7, 104, 206)])
def test_stride_keeps_poles_and_even_seam(stride, rows, cols):
    seq = _gfs(stride=stride)
    g = seq.grid
    assert g.shape == (rows, cols)
    assert g.lats[0, 0] == 90 and g.lats[-1, 0] == -90, "both pole rows kept"
    dlon = np.diff(np.r_[g.lons[0], g.lons[0, 0] + 360])
    assert dlon.max() <= stride * 0.25 + 1e-9
    v = seq.load(2)
    truth = analytic(g.lats, g.lons, 6).reshape(-1)
    ok = np.isfinite(v)
    assert np.abs(v[ok] - truth[ok]).max() < 0.01
    seq.close()


def test_auto_stride_under_budget():
    seq = _gfs(max_points=400_000)
    assert seq.grid.stride == 2 and seq.grid.n_points <= 400_000
    seq.close()


def test_multi_message_file_south_to_north_and_select():
    path = os.path.join(DATA, "multi_1deg_s2n.grib2")
    t2m = GribSequence(path, {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2})
    pwat = GribSequence(path, {"shortName": "pwat"})
    assert len(t2m) == 3 and len(pwat) == 3 and list(t2m.hours) == [0, 6, 12]
    g = t2m.grid
    assert g.lats[0, 0] == -90 and g.lats[-1, 0] == 90 and g.wrap_lon
    truth = analytic(g.lats, g.lons, 12).reshape(-1)
    assert np.abs(t2m.load(2) - truth).max() < 0.01
    with pytest.raises(LookupError):
        GribSequence(path, {"shortName": "nope"})
    t2m.close(), pwat.close()


def test_lambert_patch():
    seq = GribSequence(os.path.join(DATA, "lambert_conus.grib2"))
    g = seq.grid
    assert g.grid_type == "lambert" and g.shape == (106, 180) and not g.wrap_lon
    assert 20 < g.lats.min() < 22 and 50 < g.lats.max() < 54
    assert np.abs(seq.load(0) - 2 * g.lats.reshape(-1)).max() < 0.01
    assert (_face_facing(g, "Y") > 0).all()
    assert len(build_shell_mesh(g).face_counts) == 105 * 179  # open patch, no seam
    seq.close()


# --------------------------------------------------------------------------- mesh
def _face_facing(grid: GribGrid, up: str) -> np.ndarray:
    m = build_shell_mesh(grid, up)
    q = m.face_indices.reshape(-1, 4)
    p = m.points.astype(np.float64)
    n1 = np.cross(p[q[:, 1]] - p[q[:, 0]], p[q[:, 3]] - p[q[:, 0]])
    n2 = np.cross(p[q[:, 3]] - p[q[:, 2]], p[q[:, 1]] - p[q[:, 2]])
    n = np.where(np.linalg.norm(n1, axis=1, keepdims=True) > 1e-12, n1, n2)  # pole quads are triangles
    centre = p[q].mean(axis=1)
    return np.einsum("ij,ij->i", n, centre)


@pytest.mark.parametrize("up", ["Y", "Z"])
def test_winding_outward_both_scan_orders(up):
    for seq in (_gfs(stride=4), GribSequence(os.path.join(DATA, "multi_1deg_s2n.grib2"), {"shortName": "2t"})):
        facing = _face_facing(seq.grid, up)
        assert (facing > 0).all(), f"{(facing <= 0).sum()} faces point inward"
        rows, cols = seq.grid.shape
        m = build_shell_mesh(seq.grid, up)
        assert len(m.face_counts) == (rows - 1) * cols  # seam stitched, no gap
        assert m.face_indices.max() == rows * cols - 1
        seq.close()


def test_axes():
    v = unit_vectors(np.array([0, 0, 90]), np.array([0, 90, 0]), "Y")
    np.testing.assert_allclose(v, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-12)
    v = unit_vectors(np.array([0, 0, 90]), np.array([0, 90, 0]), "Z")
    np.testing.assert_allclose(v, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-12)


# --------------------------------------------------------------------------- style
def test_style():
    s = Style(cmap="viridis", vmin=0, vmax=10, opacity=OpacityRamp(2, 8, 0.0, 0.8), missing_opacity=0.0)
    rgb, a = s.colorize(np.array([0, 5, 10, np.nan, 100], dtype=np.float32))
    assert rgb.dtype == np.float32 and rgb.shape == (5, 3)
    np.testing.assert_allclose(a, [0, 0.4, 0.8, 0, 0.8], atol=1e-6)
    np.testing.assert_allclose(rgb[4], rgb[2])  # clamped above vmax
    lin = Style(cmap="gray", vmin=0, vmax=1, linear_color=True).colorize(np.array([0.5]))[0][0, 0]
    assert abs(lin - 0.214) < 0.01  # sRGB 0.5 -> linear 0.214
    auto = Style(log_scale=True).autoscale(np.array([1e-5, 1e-4, 1e-3, 1e-2], dtype=np.float32))
    assert 1e-5 <= auto.vmin < auto.vmax <= 1e-2
    with pytest.raises(KeyError):
        Style(cmap="nope").colorize(np.zeros(3))


# --------------------------------------------------------------------------- USD
pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from talon_defense.weather_overlay import GribPlayer, GribSphere  # noqa: E402


def _stage(up: str = "Y") -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


@pytest.mark.parametrize("unlit", [True, False])
def test_usd_structure(unlit):
    seq = GribSequence(os.path.join(DATA, "multi_1deg_s2n.grib2"), {"shortName": "2t"})
    stage = _stage()
    sphere = GribSphere(stage, "/World/GribSphere", seq.grid, radius=250, unlit=unlit, material="preview")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/GribSphere/Shell"))
    n = seq.grid.n_points
    assert len(mesh.GetPointsAttr().Get()) == n
    color = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("dataColor")
    alpha = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("dataOpacity")
    assert color.GetInterpolation() == "vertex" and alpha.GetInterpolation() == "vertex"
    assert mesh.GetSubdivisionSchemeAttr().Get() == "none"
    assert UsdGeom.PrimvarsAPI(mesh).GetPrimvar("doNotCastShadows").Get() is True
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(mesh.GetPrim())
    assert abs(bbox.ComputeAlignedRange().GetMax()[1] - 250) < 1e-3  # radius via xformOp:scale

    mat, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
    assert mat.GetPath() == Sdf.Path("/World/GribSphere/Looks/DataShell")
    surf = mat.ComputeSurfaceSource()[0]
    assert surf.GetIdAttr().Get() == "UsdPreviewSurface"

    def src(name):
        s = surf.GetInput(name).GetConnectedSources()[0]
        return (s[0].source.GetPath().name, s[0].sourceName) if s else None

    assert src("opacity") == ("OpacityReader", "result")
    assert src("emissiveColor" if unlit else "diffuseColor") == ("ColorReader", "result")
    assert src("diffuseColor" if unlit else "emissiveColor") is None
    reader = UsdShade.Shader(stage.GetPrimAtPath("/World/GribSphere/Looks/DataShell/ColorReader"))
    assert reader.GetInput("varname").Get() == "dataColor"
    assert surf.GetInput("opacityThreshold").Get() == 0.0

    # everything lives in the session layer; the file the user saves is untouched
    assert stage.GetRootLayer().GetPrimAtPath("/World/GribSphere") is None
    assert stage.GetSessionLayer().GetPrimAtPath("/World/GribSphere/Shell") is not None

    style = Style(cmap="turbo", vmin=240, vmax=310, opacity=0.5)
    values = seq.load(1)
    sphere.set_field(values, style)
    assert sphere.writer_name == "usd (warm-up)"
    for _ in range(2):
        sphere.set_field(values, style)
    assert sphere.writer_name == "usd"  # no FSD outside Kit
    rgb, a = style.colorize(values)
    np.testing.assert_allclose(np.array(color.Get()), rgb, atol=1e-7)
    np.testing.assert_allclose(np.array(alpha.Get()), a, atol=1e-7)

    sphere.radius = 10
    assert stage.GetPrimAtPath("/World/GribSphere").GetAttribute("xformOp:scale").Get()[0] == 10
    # rebuilding at the same path is clean (re-running a script)
    GribSphere(stage, "/World/GribSphere", seq.grid, material="preview")
    sphere.remove()
    assert not stage.GetPrimAtPath("/World/GribSphere")
    seq.close()


def test_mdl_material():
    from talon_defense.weather_overlay.usd_sphere import COLOR_PRIMVAR, DATA_SHELL_MDL, OPACITY_PRIMVAR
    import os, re
    assert os.path.isfile(DATA_SHELL_MDL)
    src = open(DATA_SHELL_MDL).read()
    # primvar names in the MDL must match what the sphere writes
    assert f'data_lookup_float3("{COLOR_PRIMVAR}"' in src and f'data_lookup_float("{OPACITY_PRIMVAR}"' in src
    assert "export material DataShell(" in src and "cutout_opacity:" in src
    code = re.sub(r"//.*", "", src)
    for a, b in ("()", "[]", "{}"):
        assert code.count(a) == code.count(b), (a, b)

    seq = SyntheticSequence(n_frames=1, lat_step=10, lon_step=10)
    stage = _stage("Z")
    sphere = GribSphere(stage, "/World/S", seq.grid, emissive_intensity=500.0)
    mat, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/S/Shell")).ComputeBoundMaterial()
    sh = mat.ComputeSurfaceSource("mdl")[0]
    assert sh.GetImplementationSource() == "sourceAsset"
    assert sh.GetSourceAsset("mdl").path == DATA_SHELL_MDL
    assert sh.GetSourceAssetSubIdentifier("mdl") == "DataShell"
    assert sh.GetInput("unlit").Get() is True and sh.GetInput("emissive_intensity").Get() == 500.0
    for output in (mat.GetSurfaceOutput("mdl"), mat.GetDisplacementOutput("mdl"), mat.GetVolumeOutput("mdl")):
        assert output.GetConnectedSources()[0][0].source.GetPath() == sh.GetPath()
    sphere.set_emissive_intensity(2500)
    sphere.set_opacity_scale(0.5)
    assert sh.GetInput("emissive_intensity").Get() == 2500.0 and sh.GetInput("opacity_scale").Get() == 0.5
    sphere.set_field(seq.get(0), Style(vmin=0, vmax=1, opacity=0.4))
    alpha = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/S/Shell")).GetPrimvar("dataOpacity").Get()
    assert np.allclose(np.array(alpha), 0.4)
    assert stage.GetRootLayer().GetPrimAtPath("/World/S") is None


def test_emissive_scale_only_in_unlit_mode():
    seq = SyntheticSequence(n_frames=1, lat_step=10, lon_step=10)
    style = Style(cmap="turbo", vmin=0, vmax=1, opacity=0.5)
    rgb, _ = style.colorize(seq.get(0))
    for unlit, expect in ((True, rgb * 250.0), (False, rgb)):
        stage = _stage("Z")
        sphere = GribSphere(stage, "/World/S", seq.grid, unlit=unlit, emissive_scale=250.0)
        sphere.set_field(seq.get(0), style)
        got = np.array(UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/S/Shell")).GetPrimvar("dataColor").Get())
        np.testing.assert_allclose(got, expect, rtol=1e-6)


def test_fractional_cutout_on_render_products():
    from talon_defense.weather_overlay.usd_sphere import enable_fractional_cutout_opacity
    stage = _stage("Z")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        vp = stage.DefinePrim("/Render/OmniverseKit/HydraTextures/Viewport0", "RenderProduct")
        vp.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set("RealTimePathTracing")
        vp.CreateAttribute("omni:rtx:rt:fractionalOpacity", Sdf.ValueTypeNames.Bool).Set(False)
        vp.CreateAttribute("omni:rtx:pt:fractionalCutoutOpacity", Sdf.ValueTypeNames.Bool).Set(False)
        rs = stage.DefinePrim("/Render/Settings", "RenderSettings")
        rs.CreateAttribute("omni:rtx:pt:fractionalCutoutOpacity", Sdf.ValueTypeNames.Int).Set(0)
        stage.DefinePrim("/Render/Bare", "RenderProduct")
        stage.DefinePrim("/Render/NotARenderPrim", "Scope").CreateAttribute(
            "omni:rtx:rt:fractionalOpacity", Sdf.ValueTypeNames.Bool).Set(False)
    report = enable_fractional_cutout_opacity(True, stage)
    assert vp.GetAttribute("omni:rtx:rt:fractionalOpacity").Get() is True
    assert vp.GetAttribute("omni:rtx:pt:fractionalCutoutOpacity").Get() is True
    assert rs.GetAttribute("omni:rtx:pt:fractionalCutoutOpacity").Get() == 1
    assert stage.GetPrimAtPath("/Render/NotARenderPrim").GetAttribute("omni:rtx:rt:fractionalOpacity").Get() is False
    assert not stage.GetPrimAtPath("/Render/Bare").GetAttribute("omni:rtx:rt:fractionalOpacity")
    assert len(report) == 3 and "rendermode=RealTimePathTracing" in report[0]
    assert "fractionalCutoutOpacity False->True" in report[0] and "no fractional-opacity attributes" in report[2]
    assert enable_fractional_cutout_opacity(True, stage)[0].count("True->True") == 2  # idempotent
    assert enable_fractional_cutout_opacity(True, _stage()) == []  # no /Render at all


def test_opacity_probe_builds_four_variants():
    from talon_defense.weather_overlay.diagnostics import PROBE_ROOT, opacity_probe, remove_opacity_probe
    seq = SyntheticSequence(n_frames=1, lat_step=10, lon_step=10)
    stage = _stage("Z")
    shell = GribSphere(stage, "/World/GribSphere", seq.grid, radius=64.0)
    shell.set_field(seq.get(0), Style(vmin=0, vmax=1, opacity=OpacityRamp(0.3, 0.8, 0.0, 0.85)))
    report = opacity_probe(stage)
    assert "usd shell opacity (vertex): n=" in report and "zeros=" in report
    assert "usd shell surface: MDL " in report and "(DataShell, file exists: True)" in report
    assert "unlit=True" in report and "emissive_intensity=1000.0" in report
    names = [p.GetName() for p in stage.GetPrimAtPath(PROBE_ROOT).GetChildren()]
    assert names == ["A_red_omnipbr_constant", "B_yellow_mdl_primvar_lit", "C_cyan_mdl_primvar_unlit",
                     "D_white_preview_constant"]
    positions = []
    for name in names:
        mesh = stage.GetPrimAtPath(f"{PROBE_ROOT}/{name}/Shell")
        mat, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        if "omnipbr" in name:
            sh = mat.ComputeSurfaceSource("mdl")[0]
            assert sh.GetSourceAsset("mdl").path == "OmniPBR.mdl"
            assert sh.GetInput("enable_opacity").Get() is True and abs(sh.GetInput("opacity_constant").Get() - 0.3) < 1e-6
        elif "mdl" in name:
            sh = mat.ComputeSurfaceSource("mdl")[0]
            assert sh.GetSourceAssetSubIdentifier("mdl") == "DataShell"
            assert sh.GetInput("unlit").Get() == ("unlit" in name)
        else:
            surf = UsdShade.Shader(stage.GetPrimAtPath(f"{PROBE_ROOT}/{name}/Looks/DataShell/Surface"))
            assert surf.GetIdAttr().Get() == "UsdPreviewSurface"
            assert not surf.GetInput("opacity").GetConnectedSources()[0]
            assert abs(surf.GetInput("opacity").Get() - 0.3) < 1e-6
        alpha = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(f"{PROBE_ROOT}/{name}/Shell")).GetPrimvar("dataOpacity").Get()
        assert np.allclose(np.array(alpha), 0.3)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(f"{PROBE_ROOT}/{name}"))
        positions.append(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
    assert all(64 < p.GetLength() < 192 for p in positions)  # between the (fallback) camera and the globe
    assert len({tuple(round(c, 3) for c in p) for p in positions}) == 4
    assert stage.GetRootLayer().GetPrimAtPath(PROBE_ROOT) is None  # session layer only
    opacity_probe(stage)  # re-running replaces, not duplicates
    assert len(stage.GetPrimAtPath(PROBE_ROOT).GetChildren()) == 4
    remove_opacity_probe(stage)
    assert not stage.GetPrimAtPath(PROBE_ROOT)
    assert stage.GetPrimAtPath("/World/GribSphere/Shell")


def test_fabric_backend_falls_back_without_usdrt():
    seq = SyntheticSequence(n_frames=2, lat_step=5, lon_step=5)
    sphere = GribSphere(_stage("Z"), "/World/S", seq.grid, backend="fabric")
    for _ in range(3):
        sphere.set_field(seq.get(0), Style(vmin=0, vmax=1))
    assert sphere.writer_name == "usd"


def test_fabric_writer_with_fake_usdrt():
    """backend=auto + FSD on: 2 warm-up writes via USD, then usdrt Set() calls with numpy-built arrays."""
    sets = []

    class Attr:
        def __init__(self, name):
            self.name = name

        def IsValid(self):
            return True

        def Set(self, value):
            sets.append((self.name, value))

    class Prim:
        def IsValid(self):
            return True

        def GetAttribute(self, name):
            return Attr(name)

    class RtStage:
        @staticmethod
        def Attach(stage_id):
            assert isinstance(stage_id, int)
            return types.SimpleNamespace(GetPrimAtPath=lambda path: Prim())

    usdrt = types.ModuleType("usdrt")
    usdrt.Usd = types.SimpleNamespace(Stage=RtStage)
    def float_array(a):
        if a.ndim != 2:
            raise ValueError("Incompatible array dimension: 1 - FloatArray requires 2")
        return ("FloatArray", a.shape, a.dtype)

    usdrt.Vt = types.SimpleNamespace(Vec3fArray=lambda a: ("Vec3fArray", a.shape, a.dtype),
                                     FloatArray=float_array)
    carb = types.ModuleType("carb")
    carb.settings = types.ModuleType("carb.settings")
    carb.settings.get_settings = lambda: types.SimpleNamespace(
        get_as_bool=lambda key: key == "/app/useFabricSceneDelegate")
    saved = {k: sys.modules.get(k) for k in ("usdrt", "carb", "carb.settings")}
    sys.modules.update({"usdrt": usdrt, "carb": carb, "carb.settings": carb.settings})
    try:
        seq = SyntheticSequence(n_frames=2, lat_step=5, lon_step=5)
        stage = _stage()
        sphere = GribSphere(stage, "/World/S", seq.grid)
        style = Style(vmin=0, vmax=1)
        for _ in range(4):
            sphere.set_field(seq.get(1), style)
        assert sphere.writer_name == "fabric"
        n = seq.grid.n_points
        assert sets == [("primvars:dataColor", ("Vec3fArray", (n, 3), np.float32)),
                        ("primvars:dataOpacity", ("FloatArray", (n, 1), np.float32))] * 2
        usd_alpha = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/S/Shell")).GetPrimvar("dataOpacity")
        np.testing.assert_allclose(np.array(usd_alpha.Get()), style.colorize(seq.get(1))[1])  # warm-up reached USD
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_upload_timing():
    """Per-frame cost of colorize + USD write at GFS 0.25 deg and at stride 2."""
    stage = _stage()
    for stride in (1, 2):
        seq = _gfs(stride=stride, max_points=None)
        sphere = GribSphere(stage, f"/World/S{stride}", seq.grid)
        style = Style(vmin=240, vmax=310, opacity=OpacityRamp(260, 300, 0.1, 0.8))
        a, b = seq.load(0), seq.load(1)
        times = []
        for k in range(10):
            t0 = time.perf_counter()
            sphere.set_field(a + (b - a) * np.float32(k / 10), style)
            times.append(time.perf_counter() - t0)
        ms = 1000 * np.median(times)
        print(f"\n  stride {stride}: {seq.grid.n_points:,} vertices, colorize+write {ms:.1f} ms")
        assert ms < 250
        seq.close()


# --------------------------------------------------------------------------- player (Kit stubbed)
class _FakeKit:
    """Installs fake omni.kit.app / carb.eventdispatcher modules and pumps updates."""

    def __init__(self):
        self.observers = []
        self.saved = {k: sys.modules.get(k) for k in ("omni", "omni.kit", "omni.kit.app", "carb",
                                                      "carb.eventdispatcher")}
        omni, kit, app, carb, ed = (types.ModuleType(n) for n in
                                    ("omni", "omni.kit", "omni.kit.app", "carb", "carb.eventdispatcher"))
        app.GLOBAL_EVENT_UPDATE = "omni.kit.app:update"
        outer = self

        class Guard:
            def __init__(self, fn):
                self.fn = fn
                outer.observers.append(self)

            def reset(self):
                if self in outer.observers:
                    outer.observers.remove(self)

        class Dispatcher:
            def observe_event(self, observer_name, event_name, on_event, **_):
                assert event_name == app.GLOBAL_EVENT_UPDATE
                return Guard(on_event)

        ed.get_eventdispatcher = lambda: Dispatcher()
        omni.kit, kit.app, carb.eventdispatcher = kit, app, ed
        sys.modules.update({"omni": omni, "omni.kit": kit, "omni.kit.app": app, "carb": carb,
                            "carb.eventdispatcher": ed})

    def pump(self, n: int, dt: float = 1 / 60) -> None:
        for _ in range(n):
            for g in list(self.observers):
                g.fn({"dt": dt})

    def restore(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_player_interpolates_and_prefetches():
    kit = _FakeKit()
    try:
        seq = _gfs(stride=4)
        stage = _stage()
        sphere = GribSphere(stage, "/World/GribSphere", seq.grid)
        style = Style(vmin=240, vmax=310, opacity=0.6)
        uploads = []
        player = GribPlayer(sphere, seq, style, hours_per_second=6.0, max_upload_hz=None,
                            on_frame=lambda p: uploads.append(p.hours)).start()
        assert len(kit.observers) == 1 and uploads == [0.0]  # frame 0 decoded at init, shown at once
        deadline = time.time() + 20
        while player.hours < 7.5 and time.time() < deadline:
            kit.pump(1, dt=1 / 60)
            time.sleep(0.002)  # give the decode thread air, as a real frame would
        assert player.hours >= 7.5 and len(uploads) > 20
        # land exactly between f003 and f006 and check the blended colors
        while seq.get(1) is None or seq.get(2) is None:
            time.sleep(0.01)
        player.seek(4.5)
        expect_rgb, _ = style.colorize(0.5 * (seq.get(1) + seq.get(2)))
        color = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/GribSphere/Shell")).GetPrimvar("dataColor")
        np.testing.assert_allclose(np.array(color.Get()), expect_rgb, atol=1e-6)
        assert str(player.current_time) == "2026-09-22 04:30:00+00:00"
        # loops past the end
        player.seek(12.0 + 1.5)
        assert player._wrapped_hours() == pytest.approx(1.5)
        # stop releases the observer; removing the prim stops the player cleanly
        player.stop()
        assert kit.observers == []
        player.start()
        sphere.remove()
        kit.pump(3)
        assert kit.observers == [] and not player.playing
        seq.close()
    finally:
        kit.restore()


def test_weather_layers_shared_clock_and_visibility():
    """Two variables, different grids and different time ranges, on one clock."""
    import datetime as dt

    from talon_defense.weather_overlay import WeatherLayers, weather_layers

    kit = _FakeKit()
    try:
        stage = _stage("Z")
        w = WeatherLayers(stage, root="/World/Weather", base_radius=64.0, separation=0.5,
                          hours_per_second=6.0, max_upload_hz=None, fractional_cutout=False)
        assert weather_layers() is w
        pwat = w.add("pwat", os.path.join(DATA, "multi_1deg_s2n.grib2"), {"shortName": "pwat"},
                     cmap="viridis", opacity=0.5)
        late = [os.path.join(DATA, f"gfs_2t_f{h:03d}.grib2") for h in (6, 9, 12)]
        t2m = w.add("t2m", late, {"shortName": "2t"}, stride=3, cmap="coolwarm", vmin=240, vmax=310)
        for layer in (pwat, t2m):  # decode up front so render() never waits in this test
            for i in range(len(layer.sequence)):
                layer.sequence.load(i)

        # nested shells under one parent, in insertion order
        assert (pwat.sphere.radius, t2m.sphere.radius) == (64.0, 64.5)
        assert stage.GetPrimAtPath("/World/Weather/pwat/Shell")
        assert stage.GetPrimAtPath("/World/Weather/t2m/Shell")
        assert pwat.sequence.grid.shape != t2m.sequence.grid.shape  # different grids are fine
        assert pwat.sequence.grid.key != t2m.sequence.grid.key

        # one shared absolute-time axis: epoch is the earliest frame of any layer
        assert w.epoch == dt.datetime(2026, 9, 22, 0, tzinfo=dt.timezone.utc)
        assert list(pwat.hours_abs) == [0, 6, 12] and list(t2m.hours_abs) == [6, 9, 12]
        assert w.span_hours == 12

        w.start(play=False)
        assert len(kit.observers) == 1, "one update subscription for all layers"

        # at +9 h both layers show 09Z; t2m's own axis says frame 1
        w.seek(9.0)
        assert w.current_time == dt.datetime(2026, 9, 22, 9, tzinfo=dt.timezone.utc)
        grid = t2m.sequence.grid
        frame = t2m.sequence.load(1)  # its own frame 1 is +9 h on the shared axis
        truth = analytic(grid.lats, grid.lons, 9).reshape(-1)
        ok = np.isfinite(frame)  # the test files carry a missing-value box
        assert np.abs(frame[ok] - truth[ok]).max() < 0.01
        expected, _ = t2m.style.colorize(frame)
        shown = np.array(UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Weather/t2m/Shell"))
                         .GetPrimvar("dataColor").Get())
        np.testing.assert_allclose(shown, expected, atol=1e-6)

        # before its data starts, a layer holds its first frame instead of disappearing
        w.seek(0.0)
        assert t2m._last_key == (0, 0.0) and pwat._last_key == (0, 0.0)

        # hidden layers upload nothing
        t2m.set_visible(False)
        assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/Weather/t2m")).ComputeVisibility() == "invisible"
        frozen = np.array(UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Weather/t2m/Shell"))
                          .GetPrimvar("dataColor").Get())
        pwat_before = np.array(UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Weather/pwat/Shell"))
                               .GetPrimvar("dataColor").Get())
        w.seek(11.0)  # not 12.0: on a 12 h span that wraps back to the start
        assert np.array_equal(np.array(UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Weather/t2m/Shell"))
                                       .GetPrimvar("dataColor").Get()), frozen)
        assert not np.array_equal(np.array(UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Weather/pwat/Shell"))
                                           .GetPrimvar("dataColor").Get()), pwat_before)

        # solo / toggle / show-all
        w.solo("t2m")
        assert w.visible_names == ["t2m"]
        assert w.toggle("pwat") is True and w.visible_names == ["pwat", "t2m"]
        w.set_all_visible(False)
        assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/Weather")).ComputeVisibility() == "invisible"
        w.set_all_visible(True)

        # playing advances every layer together, and the shared span loops
        w.seek(11.9)
        w.play()
        kit.pump(6, dt=1 / 60)
        assert w.hours == pytest.approx(11.9 + 6 * 6.0 / 60)  # 6 h/s at 60 fps = 0.1 h per frame
        assert w.hours > 12 and w._wrapped_hours() == pytest.approx(0.5)

        w.remove("pwat")
        assert not stage.GetPrimAtPath("/World/Weather/pwat") and list(w.layers) == ["t2m"]
        w.close()
        assert kit.observers == [] and not stage.GetPrimAtPath("/World/Weather")
        assert weather_layers() is None
    finally:
        kit.restore()


def test_weather_layers_defaults_and_names():
    from talon_defense.weather_overlay import WeatherLayers

    kit = _FakeKit()
    try:
        stage = _stage()
        w = WeatherLayers(stage, base_radius=10.0, separation=0.25, fractional_cutout=False)
        a = w.add("demo one", None, opacity_ramp=[0.2, 0.8, 0.0, 0.9])  # synthetic field
        b = w.add("demo two", None, radius=20.0)
        c = w.add("demo three", None)
        assert stage.GetPrimAtPath("/World/Weather/demo_one/Shell")
        assert (a.sphere.radius, b.sphere.radius, c.sphere.radius) == (10.0, 20.0, 10.25)
        assert isinstance(a.style.opacity, OpacityRamp) and a.style.opacity(np.array([0.8]))[0] == 0.9
        assert a.sphere.material == "mdl" and a.sphere.unlit is True
        a.set_opacity_scale(0.4)
        shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Weather/demo_one/Looks/DataShell/Surface"))
        assert shader.GetInput("opacity_scale").Get() == pytest.approx(0.4)
        w.add("demo one", None)  # re-adding replaces
        assert len(w.layers) == 3
        w.close()
    finally:
        kit.restore()


def test_extension_autostart_layers_from_settings():
    """`layers` in extension.toml -> one shell per entry, one manager, shared clock."""
    import asyncio
    import importlib

    kit = _FakeKit()
    stage = _stage("Z")
    prefix = "/exts/talon_defense.weather_overlay"
    store = {
        f"{prefix}/radius": 64.0, f"{prefix}/layer_separation": 0.4, f"{prefix}/ui": False,
        f"{prefix}/fractional_cutout": False, f"{prefix}/hours_per_second": 12.0,
        f"{prefix}/layers": {
            "t2m": {"glob": os.path.join(DATA, "gfs_2t_f*.grib2"), "select": {"shortName": "2t"},
                    "cmap": "coolwarm", "vmin": 240.0, "vmax": 310.0, "order": 2, "stride": 4},
            "pwat": {"glob": os.path.join(DATA, "multi_1deg_s2n.grib2"),
                     "select": {"shortName": "pwat"}, "opacity_ramp": [20.0, 60.0, 0.0, 0.85],
                     "order": 1, "visible": False},
        },
    }

    class Settings:
        def get(self, key):
            return store.get(key)

        def set_bool(self, key, value):
            pass

        def get_as_bool(self, key):
            return bool(store.get(key))

    carb = sys.modules["carb"]
    carb.settings = types.ModuleType("carb.settings")
    carb.settings.get_settings = lambda: Settings()
    carb.log_info = carb.log_warn = carb.log_error = lambda msg: None
    omni = sys.modules["omni"]
    omni.ext = types.ModuleType("omni.ext")
    omni.ext.IExt = type("IExt", (), {})
    omni.usd = types.ModuleType("omni.usd")
    omni.usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)

    async def next_update_async():
        await asyncio.sleep(0)

    sys.modules["omni.kit.app"].get_app = lambda: types.SimpleNamespace(next_update_async=next_update_async)
    extra = {"carb.settings": carb.settings, "omni.ext": omni.ext, "omni.usd": omni.usd}
    saved = {k: sys.modules.get(k) for k in extra}
    sys.modules.update(extra)
    try:
        from talon_defense.weather_overlay import weather_layers

        # extension.py binds `carb` at import time, so reload it under this test's stubs
        ext_mod = importlib.reload(importlib.import_module("talon_defense.weather_overlay.extension"))
        ext = ext_mod.GribSphereExtension()

        async def run():
            ext.on_startup("talon_defense.weather_overlay-0.1.0")
            await ext._task

        asyncio.run(run())
        w = weather_layers()
        assert w is not None and list(w.layers) == ["pwat", "t2m"]  # `order` decides, not the dict
        assert (w["pwat"].sphere.radius, w["t2m"].sphere.radius) == (64.0, 64.4)
        assert w.visible_names == ["t2m"] and w.hours_per_second == 12.0
        assert w["t2m"].style.vmin == 240.0 and isinstance(w["pwat"].style.opacity, OpacityRamp)
        assert w["t2m"].sequence.grid.stride == 4
        assert len(kit.observers) == 1 and w.playing
        ext.on_shutdown()
        assert weather_layers() is None and not stage.GetPrimAtPath("/World/Weather")
        assert kit.observers == []
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        kit.restore()


def test_extension_autostart_demo():
    """Extension on_startup with demo=true builds the shell and plays (Kit modules stubbed)."""
    import asyncio
    import importlib

    kit = _FakeKit()
    stage = _stage()
    store = {"/exts/talon_defense.weather_overlay/demo": True,
             "/exts/talon_defense.weather_overlay/radius": 42.0,
             "/exts/talon_defense.weather_overlay/emissive_scale": 3.0,
             "/exts/some.other_ext/demo": True}
    set_calls = {}

    class Settings:
        def get(self, key):
            return store.get(key)

        def set_bool(self, key, value):
            set_calls[key] = value

        def get_as_bool(self, key):
            return bool(store.get(key))

    carb = sys.modules["carb"]
    carb.settings = types.ModuleType("carb.settings")
    carb.settings.get_settings = lambda: Settings()
    carb.log_info = carb.log_warn = carb.log_error = lambda msg: None
    omni = sys.modules["omni"]
    omni.ext = types.ModuleType("omni.ext")
    omni.ext.IExt = type("IExt", (), {})
    omni.usd = types.ModuleType("omni.usd")
    omni.usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)

    async def next_update_async():
        await asyncio.sleep(0)

    sys.modules["omni.kit.app"].get_app = lambda: types.SimpleNamespace(next_update_async=next_update_async)
    extra = {"carb.settings": carb.settings, "omni.ext": omni.ext, "omni.usd": omni.usd}
    saved = {k: sys.modules.get(k) for k in extra}
    sys.modules.update(extra)
    try:
        # extension.py binds `carb` at import time, so reload it under this test's stubs
        ext_mod = importlib.reload(importlib.import_module("talon_defense.weather_overlay.extension"))
        ext = ext_mod.GribSphereExtension()

        async def run():
            ext.on_startup("talon_defense.weather_overlay-0.1.0")
            await ext._task

        asyncio.run(run())
        from talon_defense.weather_overlay import session

        assert len(session._ACTIVE) == 1
        s = session._ACTIVE[0]
        assert s.sphere.radius == 42.0 and s.sphere.emissive_scale == 3.0
        assert s.player.playing and len(kit.observers) == 1
        assert set_calls == {k: True for k in ("/rtx/raytracing/fractionalCutoutOpacity",
                                               "/rtx/pathtracing/fractionalCutoutOpacity")}
        alpha = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/GribSphere/Shell")).GetPrimvar("dataOpacity")
        assert 0 < np.array(alpha.Get()).max() <= 0.9
        kit.pump(30)
        ext.on_shutdown()
        assert session._ACTIVE == [] and kit.observers == []
        assert not stage.GetPrimAtPath("/World/GribSphere")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        kit.restore()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
