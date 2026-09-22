"""Transparency diagnostics for the weather shell. Run from the Script Editor:

    from talon_defense.weather_overlay.diagnostics import opacity_probe, remove_opacity_probe
    print(opacity_probe())
    remove_opacity_probe()

Places four small spheres between the camera and the globe, all at opacity 0.3,
on the same mesh + primvars as the shell:

    A red     OmniPBR, opacity_constant 0.3 (enable_opacity)   reference for RTX cutout
    B yellow  DataShell.mdl, per-vertex primvar opacity, lit
    C cyan    DataShell.mdl, per-vertex primvar opacity, unlit  <- the shell's setup
    D white   UsdPreviewSurface, constant opacity 0.3           control (solid in Kit 110 RT 2.0)

Reading it:
    A solid                      -> cutout opacity is off for this renderer/viewport altogether
    A see-through, B/C solid     -> DataShell.mdl isn't compiling or its primvar lookup fails
                                    (see [rtx.mdltranslator] errors in the Kit log)
    A, B, C see-through          -> the shell should be see-through too
It also reports the renderer settings and the shell's material + opacity values (USD and Fabric).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils

from . import _log
from .grib_io import GribGrid
from .usd_sphere import (DATA_SHELL_MDL, OPACITY_PRIMVAR, RENDER_MODE_ATTR,
                         RENDER_PRODUCT_FRACTIONAL_ATTRS, GribSphere)

PROBE_ROOT = "/World/OpacityProbe"
PROBE_OPACITY = 0.3
PROBES = (  # name, linear rgb, material kind, unlit
    ("A_red_omnipbr_constant", (1.0, 0.05, 0.05), "omnipbr", False),
    ("B_yellow_mdl_primvar_lit", (1.0, 0.8, 0.05), "mdl", False),
    ("C_cyan_mdl_primvar_unlit", (0.05, 0.8, 1.0), "mdl", True),
    ("D_white_preview_constant", (0.9, 0.9, 0.9), "preview", False),
)
CARB_KEYS = ("/rtx/rendermode", "/rtx/raytracing/fractionalCutoutOpacity",
             "/rtx/pathtracing/fractionalCutoutOpacity", "/rtx/material/translucencyAsOpacity",
             "/app/useFabricSceneDelegate")


def _stats(values: Any) -> str:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return "empty"
    return (f"n={a.size:,} min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f} "
            f"zeros={np.mean(a == 0) * 100:.1f}% below0.5={np.mean(a < 0.5) * 100:.1f}%")


def _camera(stage: Usd.Stage) -> tuple[Gf.Vec3d, Gf.Vec3d] | None:
    try:
        from omni.kit.viewport.utility import get_active_viewport

        cam = stage.GetPrimAtPath(get_active_viewport().camera_path)
        xf = UsdGeom.Imageable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return xf.ExtractTranslation(), xf.TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized()
    except Exception:  # noqa: BLE001 - no viewport (tests) -> fixed placement
        return None


def _placements(stage: Usd.Stage, globe_radius: float) -> tuple[list[Gf.Vec3d], float]:
    cam = _camera(stage)
    if cam is None:
        z_up = str(UsdGeom.GetStageUpAxis(stage)) == "Z"
        eye = Gf.Vec3d(globe_radius * 3.0, 0, 0)
        right = Gf.Vec3d(0, 1, 0) if z_up else Gf.Vec3d(0, 0, -1)
    else:
        eye, right = cam
    to_centre = Gf.Vec3d(0, 0, 0) - eye
    gap = max(to_centre.GetLength() - globe_radius, globe_radius * 0.1)  # camera -> globe surface
    centre = eye + to_centre.GetNormalized() * gap * 0.5
    size = gap * 0.03
    return [centre + right * size * 2.6 * (i - 1.5) for i in range(len(PROBES))], size


def _fabric_report(stage: Usd.Stage, shell_mesh: str | None) -> list[str]:
    try:
        from usdrt import Usd as RtUsd
    except ImportError:
        return ["fabric: usdrt not available"]
    lines = []
    try:
        cache = UsdUtils.StageCache.Get()
        sid = cache.GetId(stage)
        rt = RtUsd.Stage.Attach((sid if sid.IsValid() else cache.Insert(stage)).ToLongInt())
        if shell_mesh:
            attr = rt.GetPrimAtPath(shell_mesh).GetAttribute(f"primvars:{OPACITY_PRIMVAR}")
            lines.append(f"fabric shell opacity: {_stats(attr.Get()) if attr and attr.IsValid() else 'missing'}")
        for type_name in ("RenderProduct", "RenderSettings"):
            for path in rt.GetPrimsWithTypeName(type_name):
                prim = rt.GetPrimAtPath(str(path))
                vals = []
                for name in (RENDER_MODE_ATTR, *RENDER_PRODUCT_FRACTIONAL_ATTRS):
                    a = prim.GetAttribute(name)
                    vals.append(f"{name.rsplit(':', 1)[-1]}={a.Get() if a and a.IsValid() else 'n/a'}")
                lines.append(f"fabric {type_name} {path}: " + ", ".join(vals))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"fabric: query failed ({type(exc).__name__}: {exc})")
    return lines


def _describe_surface(mat: UsdShade.Material) -> str:
    surf = mat.ComputeSurfaceSource("mdl")[0]
    if surf and surf.GetSourceAsset("mdl"):
        vals = ", ".join(f"{i.GetBaseName()}={i.Get()}" for i in surf.GetInputs())
        asset = surf.GetSourceAsset("mdl").path
        exists = os.path.isfile(asset) if os.path.isabs(asset) else "search path"
        return f"MDL {asset} ({surf.GetSourceAssetSubIdentifier('mdl')}, file exists: {exists}) {vals}"
    surf = mat.ComputeSurfaceSource()[0]
    if not surf:
        return "none"
    inp = surf.GetInput("opacity")
    src = inp.GetConnectedSources()[0] if inp else []
    return f"{surf.GetIdAttr().Get()} opacity <- {src[0].source.GetPath() if src else (inp.Get() if inp else '-')}"


def _omnipbr(stage: Usd.Stage, path: str, rgb: tuple, opacity: float) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(opacity)
    sh.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(0.0)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"),
                   mat.CreateVolumeOutput("mdl")):
        output.ConnectToSource(out)
    return mat


def opacity_probe(stage: Usd.Stage | None = None, shell_path: str = "/World/GribSphere",
                  globe_radius: float | None = None) -> str:
    """Build the four probe spheres and return (and log) a diagnostics report."""
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    lines = ["=== weather shell opacity probe ==="]
    try:
        import carb.settings

        s = carb.settings.get_settings()
        lines += [f"carb {k} = {s.get(k)!r}" for k in CARB_KEYS]
    except ImportError:
        lines.append("carb: unavailable")

    shell_mesh = None
    shell = stage.GetPrimAtPath(shell_path)
    scale = None
    if shell:
        shell_mesh = f"{shell_path}/Shell"
        mesh = stage.GetPrimAtPath(shell_mesh)
        pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar(OPACITY_PRIMVAR)
        lines.append(f"usd shell opacity ({pv.GetInterpolation() if pv else '-'}): "
                     f"{_stats(pv.Get()) if pv else 'MISSING'}")
        mat, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        lines.append(f"usd shell material: {mat.GetPath() if mat else 'NONE BOUND'}")
        if mat:
            lines.append(f"usd shell surface: {_describe_surface(mat)}")
        ops = UsdGeom.Xformable(shell).GetOrderedXformOps()
        scale = ops[0].Get()[0] if ops else None
    else:
        lines.append(f"no shell at {shell_path}")
    lines += _fabric_report(stage, shell_mesh)
    render_root = stage.GetPrimAtPath("/Render")
    lines.append("usd /Render: " + (", ".join(f"{p.GetPath()}<{p.GetTypeName()}>"
                                            for p in Usd.PrimRange(render_root) if p.GetTypeName())
                                    if render_root else "absent"))

    remove_opacity_probe(stage)
    radius = float(globe_radius or (scale / 1.005 if scale else 64.0))
    positions, size = _placements(stage, radius)
    grid = GribGrid.regular(15.0, 15.0)
    n = grid.n_points
    layer = stage.GetSessionLayer()
    for (name, rgb, kind, unlit), pos in zip(PROBES, positions):
        path = f"{PROBE_ROOT}/{name}"
        sphere = GribSphere(stage, path, grid, radius=size, unlit=unlit, backend="usd", layer=layer,
                            material="preview" if kind == "preview" else "mdl")
        sphere.set_colors(np.tile(np.float32(rgb), (n, 1)), np.full(n, PROBE_OPACITY, np.float32))
        with Usd.EditContext(stage, layer):
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            scale_op = xf.GetOrderedXformOps()[0]
            translate = xf.AddTranslateOp()
            translate.Set(pos)
            xf.SetXformOpOrder([translate, scale_op])
            if kind == "preview":
                surf = UsdShade.Shader(stage.GetPrimAtPath(f"{path}/Looks/DataShell/Surface"))
                surf.GetInput("opacity").DisconnectSource()
                surf.GetInput("opacity").Set(PROBE_OPACITY)
            elif kind == "omnipbr":
                mat = _omnipbr(stage, f"{path}/Looks/OmniPBR", rgb, PROBE_OPACITY)
                UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"{path}/Shell")).Bind(mat)
    lines.append(f"mdl: {DATA_SHELL_MDL} exists={os.path.isfile(DATA_SHELL_MDL)}")
    lines.append(f"probes: 4 spheres at opacity {PROBE_OPACITY} under {PROBE_ROOT}, left to right "
                 "A red (OmniPBR constant), B yellow (DataShell.mdl primvar, lit), "
                 "C cyan (DataShell.mdl primvar, unlit = the shell's setup), D white (UsdPreviewSurface constant)")
    report = "\n".join(lines)
    for line in lines:
        _log.warn(f"probe: {line}")
    return report


def remove_opacity_probe(stage: Usd.Stage | None = None) -> None:
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, layer):
        if layer.GetPrimAtPath(PROBE_ROOT):
            stage.RemovePrim(PROBE_ROOT)
