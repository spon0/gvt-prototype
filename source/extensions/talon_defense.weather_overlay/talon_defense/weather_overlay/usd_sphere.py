"""The data shell on the USD stage: mesh + primvar-driven material + fast per-frame writers.

Prim layout (default path /World/GribSphere):

    GribSphere            Xform, xformOp:scale = radius
      Shell               Mesh on the unit sphere, vertex primvars:
                            primvars:dataColor   color3f[]  (vertex)
                            primvars:dataOpacity float[]    (vertex)
      Looks/DataShell     Material: mdl/DataShell.mdl (default) or UsdPreviewSurface + readers

material="mdl" (default) uses the bundled DataShell.mdl: it reads both primvars with
scene::data_lookup_* and drives MDL cutout_opacity directly. That is the path RTX
renders as smooth alpha (fractional cutout opacity, on by default in Real-Time 2.0).
UsdPreviewSurface opacity is NOT reliably mapped to cutout by RTX (in Kit 110 /
Real-Time 2.0 even a constant opacity of 0.3 renders solid), so material="preview"
is kept only for other renderers.

The preview material reads the primvars through UsdPrimvarReader_float3 / _float. Custom
names are used on purpose: displayColor/displayOpacity are viewport hints that
RTX and the Fabric Scene Delegate treat specially. If the reader can't find a
primvar it returns its fallback: magenta color, opacity 1.0 (an opaque magenta
shell means "primvar lookup failed", an opaque *colored* shell means "fractional
cutout opacity is off in the active render mode").

Per-vertex opacity renders as *cutout* opacity in RTX; fractional cutout opacity
(:func:`enable_fractional_cutout_opacity`) turns that into smooth alpha instead of
a hard threshold.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

from . import _log
from .grib_io import GribGrid
from .mesh import build_shell_mesh
from .style import Style

COLOR_PRIMVAR = "dataColor"      # names are hard-coded in mdl/DataShell.mdl too
OPACITY_PRIMVAR = "dataOpacity"
DATA_SHELL_MDL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdl", "DataShell.mdl").replace(os.sep, "/")

# RTX - Real-Time (legacy) uses the first key; RTX - Real-Time 2.0 and
# Interactive (Path Tracing) use the second. Unknown keys are harmless.
FRACTIONAL_CUTOUT_SETTINGS = (
    "/rtx/raytracing/fractionalCutoutOpacity",
    "/rtx/pathtracing/fractionalCutoutOpacity",
)


# Kit 110 "Render Settings 2.0" mirrors those carb keys onto the viewport's UsdRender
# prims (session layer, under /Render). The renderer reads the USD side, so a carb
# change made after startup may never reach it; set the USD attributes as well.
RENDER_PRODUCT_FRACTIONAL_ATTRS = (
    "omni:rtx:rt:fractionalOpacity",         # OmniRtxSettingsRtAPI_1  (RTX - Real-Time)
    "omni:rtx:pt:fractionalCutoutOpacity",   # OmniRtxDebugSettingsAPI_1 (Real-Time 2.0 / Path Tracing)
)
RENDER_MODE_ATTR = "omni:rtx:rendermode"


def enable_fractional_cutout_opacity(enabled: bool = True, stage: Usd.Stage | None = None) -> list[str]:
    """Let per-vertex opacity blend smoothly instead of cutting out.

    Sets the carb keys and, when ``stage`` is given, the matching attributes on every
    UsdRender product/settings prim under /Render that carries them. Returns one
    line per render prim: its render mode and what changed (for the log).
    """
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        for key in FRACTIONAL_CUTOUT_SETTINGS:
            settings.set_bool(key, enabled)
    except ImportError:
        pass
    if stage is None:
        return []
    render_root = stage.GetPrimAtPath("/Render")
    if not render_root:
        return []
    report = []
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in Usd.PrimRange(render_root, Usd.PrimAllPrimsPredicate):
            if prim.GetTypeName() not in ("RenderProduct", "RenderSettings"):
                continue
            changes = []
            for name in RENDER_PRODUCT_FRACTIONAL_ATTRS:
                attr = prim.GetAttribute(name)
                if not attr:
                    continue  # schema not applied here; the carb key covers it
                before = attr.Get()
                want = int(enabled) if str(attr.GetTypeName()) == "int" else bool(enabled)
                if before != want:
                    try:
                        attr.Set(want)
                    except Exception as exc:  # noqa: BLE001
                        changes.append(f"{name} set failed: {exc}")
                        continue
                changes.append(f"{name.rsplit(':', 1)[-1]} {before}->{want}")
            mode = prim.GetAttribute(RENDER_MODE_ATTR)
            report.append(f"{prim.GetPath()} rendermode={mode.Get() if mode else 'n/a'}; "
                          + (", ".join(changes) or "no fractional-opacity attributes"))
    return report


def _vt_vec3f(a: np.ndarray) -> Vt.Vec3fArray:
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, 3)
    try:
        return Vt.Vec3fArray.FromNumpy(a)
    except AttributeError:  # very old USD builds
        return Vt.Vec3fArray(a)


def _vt_float(a: np.ndarray) -> Vt.FloatArray:
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    try:
        return Vt.FloatArray.FromNumpy(a)
    except AttributeError:
        return Vt.FloatArray(a)


def _fsd_enabled() -> bool:
    try:
        import carb.settings

        return bool(carb.settings.get_settings().get_as_bool("/app/useFabricSceneDelegate"))
    except Exception:  # noqa: BLE001 - outside Kit
        return False


# --------------------------------------------------------------------------- writers
def _valid(obj: Any) -> bool:
    is_valid = getattr(obj, "IsValid", None)
    return bool(is_valid()) if callable(is_valid) else bool(obj)


class _UsdWriter:
    """Writes the two primvars straight into the layer spec inside one Sdf.ChangeBlock.

    Skips UsdAttribute value resolution and sends a single change notice per
    frame. Works with OmniHydra and with the Fabric Scene Delegate (FSD syncs
    USD changes to Fabric in a batch before rendering).
    """

    name = "usd"

    def __init__(self, layer: Sdf.Layer, mesh_path: Sdf.Path):
        self._layer = layer
        self._color_path = mesh_path.AppendProperty(f"primvars:{COLOR_PRIMVAR}")
        self._alpha_path = mesh_path.AppendProperty(f"primvars:{OPACITY_PRIMVAR}")

    def write(self, rgb: np.ndarray, alpha: np.ndarray) -> None:
        color_spec = self._layer.GetAttributeAtPath(self._color_path)
        alpha_spec = self._layer.GetAttributeAtPath(self._alpha_path)
        if color_spec is None or alpha_spec is None:
            raise RuntimeError(f"{self._color_path.GetPrimPath()} no longer exists in {self._layer.identifier}")
        with Sdf.ChangeBlock():
            color_spec.default = _vt_vec3f(rgb)
            alpha_spec.default = _vt_float(alpha)


class _FabricWriter:
    """Writes the primvars directly into Fabric through USDRT (FSD only).

    Skips USD change processing and the USD->Fabric sync entirely - the fastest
    path for large, every-frame updates. The USD layer keeps its initial values.
    """

    name = "fabric"

    def __init__(self, stage: Usd.Stage, mesh_path: Sdf.Path):
        import usdrt  # noqa: F401 - ImportError means "fall back"
        from usdrt import Usd as RtUsd, Vt as RtVt

        cache = UsdUtils.StageCache.Get()
        stage_id = cache.GetId(stage)
        if not stage_id.IsValid():
            stage_id = cache.Insert(stage)
        self._rt_stage = RtUsd.Stage.Attach(stage_id.ToLongInt())
        prim = self._rt_stage.GetPrimAtPath(str(mesh_path))  # populates from USD if needed
        if not _valid(prim):
            raise RuntimeError(f"{mesh_path} is not in Fabric")
        self._color = prim.GetAttribute(f"primvars:{COLOR_PRIMVAR}")
        self._alpha = prim.GetAttribute(f"primvars:{OPACITY_PRIMVAR}")
        if not (_valid(self._color) and _valid(self._alpha)):
            raise RuntimeError(f"{mesh_path} primvars are not in Fabric")
        self._Vt = RtVt

    _alpha_2d: bool | None = None  # usdrt 7.6 wants (N, 1) for FloatArray; older builds take (N,)

    def write(self, rgb: np.ndarray, alpha: np.ndarray) -> None:
        self._color.Set(self._Vt.Vec3fArray(np.ascontiguousarray(rgb, dtype=np.float32).reshape(-1, 3)))
        a = np.ascontiguousarray(alpha, dtype=np.float32).reshape(-1)
        if self._alpha_2d is None:
            try:
                arr = self._Vt.FloatArray(a)
                self._alpha_2d = False
            except (ValueError, TypeError):
                arr = self._Vt.FloatArray(a.reshape(-1, 1))
                self._alpha_2d = True
        else:
            arr = self._Vt.FloatArray(a.reshape(-1, 1) if self._alpha_2d else a)
        self._alpha.Set(arr)


# --------------------------------------------------------------------------- sphere
class GribSphere:
    """A transparent lat/lon shell whose vertices carry color + opacity.

    Args:
        stage: the USD stage (``omni.usd.get_context().get_stage()`` in Kit).
        path: prim path of the shell's Xform.
        grid: lattice from a GRIB file (``GribSequence.grid`` / ``read_field(...).grid``).
        radius: shell radius in stage units (applied as xformOp:scale, points are unit length).
        unlit: drive emissiveColor (colors ignore scene lighting, the usual choice for
            data); False drives diffuseColor so the shell is shaded by lights.
        double_sided: render back faces (see the far side of the shell through the near side).
        cast_shadows: let the shell shadow whatever is inside it (a globe).
        emissive_scale: multiplier applied to the uploaded colors (both materials).
        material: "mdl" (bundled DataShell.mdl, real per-vertex alpha in RTX) or
            "preview" (UsdPreviewSurface + primvar readers, for non-RTX renderers).
        emissive_intensity: brightness of unlit colors with material="mdl" (same scale
            as light intensities); change live with :meth:`set_emissive_intensity`.
        layer: layer to author into. Default: the session layer, so the live overlay
            is never saved into the user's file and never touches undo.
        backend: "auto" (Fabric when FSD is on and usdrt is available, else USD),
            "usd" or "fabric".
    """

    def __init__(self, stage: Usd.Stage, path: str | Sdf.Path, grid: GribGrid, *,
                 radius: float = 100.0, unlit: bool = True, double_sided: bool = False,
                 cast_shadows: bool = False, emissive_scale: float = 1.0,
                 material: str = "mdl", emissive_intensity: float = 1000.0,
                 layer: Sdf.Layer | None = None, backend: str = "auto"):
        self.stage = stage
        self.path = Sdf.Path(str(path))
        self.mesh_path = self.path.AppendChild("Shell")
        self.material_path = self.path.AppendChild("Looks").AppendChild("DataShell")
        self.layer = layer or stage.GetSessionLayer()
        self.unlit = unlit
        self.double_sided = double_sided
        self.cast_shadows = cast_shadows
        self.emissive_scale = float(emissive_scale)
        self.material = material
        if material == "mdl" and not os.path.isfile(DATA_SHELL_MDL):
            _log.warn(f"{DATA_SHELL_MDL} not found; falling back to UsdPreviewSurface")
            self.material = "preview"
        elif material not in ("mdl", "preview"):
            raise ValueError(f"material must be 'mdl' or 'preview', not {material!r}")
        self.emissive_intensity = float(emissive_intensity)
        self.backend = backend
        self._radius = float(radius)
        self._writer: Any = None
        self.grid: GribGrid | None = None
        self.rebuild(grid)

    # ---------------------------------------------------------------- authoring
    def rebuild(self, grid: GribGrid) -> None:
        """(Re)author the mesh for a new grid. Needed only when the grid changes."""
        up_axis = str(UsdGeom.GetStageUpAxis(self.stage))
        shell = build_shell_mesh(grid, up_axis)
        n = len(shell.points)

        with Usd.EditContext(self.stage, self.layer):
            if self.layer.GetPrimAtPath(self.path):
                self.stage.RemovePrim(self.path)  # clean slate in our layer

            xform = UsdGeom.Xform.Define(self.stage, self.path)
            xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(self._radius))

            mesh = UsdGeom.Mesh.Define(self.stage, self.mesh_path)
            mesh.CreatePointsAttr(_vt_vec3f(shell.points))
            mesh.CreateNormalsAttr(_vt_vec3f(shell.normals))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(shell.face_counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(shell.face_indices))
            mesh.CreateExtentAttr(_vt_vec3f(shell.extent))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
            mesh.CreateDoubleSidedAttr(self.double_sided)

            primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
            color = primvars.CreatePrimvar(COLOR_PRIMVAR, Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            color.Set(_vt_vec3f(np.full((n, 3), 0.5, dtype=np.float32)))
            opacity = primvars.CreatePrimvar(OPACITY_PRIMVAR, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
            opacity.Set(_vt_float(np.zeros(n, dtype=np.float32)))  # invisible until data arrives

            primvars.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool,
                                   UsdGeom.Tokens.constant).Set(not self.cast_shadows)

            material = self._define_material()
            binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
            binding.Bind(material)

        self.grid = grid
        self._writer = None  # re-resolved on the next write
        self._usd_warmup = 2

    def _define_material(self) -> UsdShade.Material:
        stage, base = self.stage, self.material_path
        UsdGeom.Scope.Define(stage, base.GetParentPath())
        material = UsdShade.Material.Define(stage, base)
        if self.material == "mdl":
            return self._define_mdl_material(material)
        return self._define_preview_material(material)

    def _define_mdl_material(self, material: UsdShade.Material) -> UsdShade.Material:
        shader = UsdShade.Shader.Define(self.stage, self.material_path.AppendChild("Surface"))
        shader.SetSourceAsset(Sdf.AssetPath(DATA_SHELL_MDL), "mdl")
        shader.SetSourceAssetSubIdentifier("DataShell", "mdl")
        shader.CreateInput("unlit", Sdf.ValueTypeNames.Bool).Set(bool(self.unlit))
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(self.emissive_intensity)
        shader.CreateInput("opacity_scale", Sdf.ValueTypeNames.Float).Set(1.0)
        out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mdl").ConnectToSource(out)
        material.CreateDisplacementOutput("mdl").ConnectToSource(out)
        material.CreateVolumeOutput("mdl").ConnectToSource(out)
        return material

    def _define_preview_material(self, material: UsdShade.Material) -> UsdShade.Material:
        stage, base = self.stage, self.material_path
        color_reader = UsdShade.Shader.Define(stage, base.AppendChild("ColorReader"))
        color_reader.CreateIdAttr("UsdPrimvarReader_float3")
        color_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(COLOR_PRIMVAR)
        color_reader.CreateInput("fallback", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 1.0))
        color_out = color_reader.CreateOutput("result", Sdf.ValueTypeNames.Color3f)

        alpha_reader = UsdShade.Shader.Define(stage, base.AppendChild("OpacityReader"))
        alpha_reader.CreateIdAttr("UsdPrimvarReader_float")
        alpha_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(OPACITY_PRIMVAR)
        alpha_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float).Set(1.0)
        alpha_out = alpha_reader.CreateOutput("result", Sdf.ValueTypeNames.Float)

        surface = UsdShade.Shader.Define(stage, base.AppendChild("Surface"))
        surface.CreateIdAttr("UsdPreviewSurface")
        diffuse = surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        emissive = surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
        if self.unlit:
            diffuse.Set(Gf.Vec3f(0.0))
            emissive.ConnectToSource(color_out)
        else:
            diffuse.ConnectToSource(color_out)
            emissive.Set(Gf.Vec3f(0.0))
        surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(alpha_out)
        surface.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        surface.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)  # no Fresnel glint / refraction
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        return material

    # ---------------------------------------------------------------- per frame
    @property
    def n_points(self) -> int:
        return self.grid.n_points if self.grid is not None else 0

    def is_valid(self) -> bool:
        return bool(self.stage.GetPrimAtPath(self.mesh_path)) and \
            self.layer.GetAttributeAtPath(self.mesh_path.AppendProperty(f"primvars:{COLOR_PRIMVAR}")) is not None

    def _resolve_writer(self) -> Any:
        want = self.backend
        if want == "auto":
            want = "fabric" if _fsd_enabled() else "usd"
        if want == "fabric":
            try:
                writer = _FabricWriter(self.stage, self.mesh_path)
                _log.info(f"{self.mesh_path}: writing primvars through Fabric (usdrt)")
                return writer
            except Exception as exc:  # noqa: BLE001
                level = _log.warn if self.backend == "fabric" else _log.info
                level(f"Fabric writer unavailable ({exc}); using USD writes")
        return _UsdWriter(self.layer, self.mesh_path)

    @property
    def writer_name(self) -> str:
        if self._writer is not None:
            return self._writer.name
        return "usd (warm-up)" if self._usd_warmup > 0 else "unresolved"

    def set_colors(self, rgb: np.ndarray, alpha: np.ndarray) -> None:
        """Upload per-vertex linear RGB (N,3) and opacity (N,), N = grid.n_points."""
        n = self.n_points
        if len(rgb) != n or len(alpha) != n:
            raise ValueError(f"Expected {n} vertices, got rgb={len(rgb)} alpha={len(alpha)}")
        if self.unlit and self.emissive_scale != 1.0:
            rgb = np.asarray(rgb, dtype=np.float32) * np.float32(self.emissive_scale)
        if self._usd_warmup > 0:
            # The first uploads after (re)building go through USD so the USD values and
            # FSD's initial population of the new prim agree; Fabric takes over after that.
            self._usd_warmup -= 1
            _UsdWriter(self.layer, self.mesh_path).write(rgb, alpha)
            return
        if self._writer is None:
            self._writer = self._resolve_writer()
        try:
            self._writer.write(rgb, alpha)
        except Exception as exc:
            if self._writer.name == "fabric":  # e.g. Fabric repopulated: retry once via USD
                _log.warn(f"Fabric write failed ({type(exc).__name__}: {exc}); switching to USD writes")
                self._writer = _UsdWriter(self.layer, self.mesh_path)
                self._writer.write(rgb, alpha)
            else:
                raise

    def set_field(self, values: np.ndarray, style: Style) -> None:
        """Colorize raw field values (mesh vertex order) with ``style`` and upload."""
        rgb, alpha = style.colorize(values)
        self.set_colors(rgb, alpha)

    # ---------------------------------------------------------------- misc
    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = float(value)
        with Usd.EditContext(self.stage, self.layer):
            op = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.path)).GetOrderedXformOps()[0]
            op.Set(Gf.Vec3d(self._radius))

    def _mdl_input(self, name: str) -> UsdShade.Input | None:
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(self.material_path.AppendChild("Surface")))
        return shader.GetInput(name) if self.material == "mdl" and shader else None

    def set_emissive_intensity(self, value: float) -> None:
        """Brightness of unlit colors (material="mdl"); no per-frame cost."""
        self.emissive_intensity = float(value)
        with Usd.EditContext(self.stage, self.layer):
            inp = self._mdl_input("emissive_intensity")
            if inp:
                inp.Set(self.emissive_intensity)

    def set_opacity_scale(self, value: float) -> None:
        """Multiply every vertex's opacity (material="mdl") without re-uploading."""
        with Usd.EditContext(self.stage, self.layer):
            inp = self._mdl_input("opacity_scale")
            if inp:
                inp.Set(float(value))

    def set_visible(self, visible: bool) -> None:
        with Usd.EditContext(self.stage, self.layer):
            img = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.path))
            img.MakeVisible() if visible else img.MakeInvisible()

    def remove(self) -> None:
        with Usd.EditContext(self.stage, self.layer):
            if self.layer.GetPrimAtPath(self.path):
                self.stage.RemovePrim(self.path)
        self._writer = None
