# talon_defense.weather_overlay

Transparent data shells around the globe: one vertex per GRIB grid point, colour and
opacity per vertex, animated from NOAA GRIB files. Several variables can run at once
as separate shells on one shared clock.

## How it works

- **Mesh**: built on the GRIB grid itself, one vertex per grid point (GFS 0.25° is
  1440 × 721). Global lat/lon grids get the seam stitched and both pole rows kept;
  Lambert grids (HRRR, NAM) become a curved patch on the same sphere. Large grids are
  thinned by `stride` (auto: keep under `max_points`).
- **Data**: `primvars:dataColor` (color3f[], vertex) and `primvars:dataOpacity`
  (float[], vertex).
- **Material**: `mdl/DataShell.mdl` reads both primvars with `scene::data_lookup_*` and
  drives MDL `cutout_opacity`. RTX renders that as smooth alpha when fractional cutout
  opacity is on (default in Real-Time 2.0). **UsdPreviewSurface opacity does not work
  for this in Kit 110 / RTX Real-Time 2.0** - even a constant 0.3 renders solid - so
  `material = "preview"` exists only for other renderers.
- **Decode**: ecCodes on a background thread, LRU frame cache, linear interpolation
  between GRIB steps. The main thread only blends two cached frames, applies the
  colormap and uploads.

## Quick start (Script Editor)

```python
from talon_defense.weather_overlay import WeatherLayers, OpacityRamp, open_panel

w = WeatherLayers(base_radius=64.0, separation=0.3, hours_per_second=6)
w.add("pwat", "C:/data/gfs/*.pwat.*.grib2", {"shortName": "pwat"},
      cmap="viridis", opacity=OpacityRamp(20, 60, 0.0, 0.85))
w.add("t2m", "C:/data/gfs/*.t2m.*.grib2", {"shortName": "2t"},
      cmap="coolwarm", vmin=240, vmax=310, opacity=0.35)
w.start()
open_panel(w)          # window with per-layer toggles + time scrubber
```

`tools/fetch_gfs.py --preset pwat --hours 0-120:3 --out C:/data/gfs` downloads GFS
files from NOMADS (1-2 MB each; presets `t2m`, `t850`, `rh700`, `prmsl`).
`inventory(path)` lists a file's messages if you need to find the right `select` keys.

## Multiple variables

Each layer is its own shell prim under one parent, at its own radius:

```
/World/Weather            <- hide this to hide every layer
  /World/Weather/pwat     <- base_radius
  /World/Weather/t2m      <- base_radius + separation
```

Not USD instancing: instances share their primvars, so every variable would show the
same field. Separate shells also let each layer keep its own colormap, opacity ramp,
grid and update rate.

| Call | Effect |
|---|---|
| `w.show("pwat")` / `w.hide("t2m")` / `w.toggle("t2m")` | USD visibility: instant, geometry stays loaded |
| `w.solo("pwat")` | show one, hide the rest |
| `w.set_all_visible(False)` | hide the whole overlay at the parent prim |
| `w["t2m"].set_active(False)` | drop the prim entirely: frees renderer memory, slow to come back |
| `w["t2m"].set_opacity_scale(0.4)` | fade a layer without re-uploading (MDL material) |
| `w.play()` / `w.pause()` / `w.seek(24)` / `w.seek_time(dt)` | shared clock |

**One clock, absolute time.** Layers are aligned on valid time, not on frame index, so
variables from different runs or with different step lengths always show the same
forecast hour. A layer whose data starts later (or ends earlier) holds its nearest
frame instead of disappearing.

**Cost.** The per-frame CPU cost (blend + colormap + upload) is paid per *visible*
layer; hidden layers upload nothing. The GPU cost is overdraw: every visible
transparent shell adds an any-hit intersection per ray, so a handful is fine, a dozen
is not. Measured outside Kit, per layer per upload: ~35-40 ms at full GFS 0.25°
(1.04M vertices), ~8-12 ms at stride 2 (260k, the auto default).

## Settings (config/extension.toml)

Single variable: set `grib_glob` (+ `select`). Several: add `layers.<name>` entries and
the extension builds one shell per entry, starts them on one clock and opens the panel
(`ui = true`).

```toml
exts."talon_defense.weather_overlay".layers.pwat = { glob = "C:/data/gfs/*.pwat.*.grib2", select = { shortName = "pwat" }, cmap = "viridis", opacity_ramp = [20.0, 60.0, 0.0, 0.85], order = 1 }
exts."talon_defense.weather_overlay".layers.t2m  = { glob = "C:/data/gfs/*.t2m.*.grib2", select = { shortName = "2t" }, cmap = "coolwarm", vmin = 240.0, vmax = 310.0, opacity = 0.35, order = 2 }
```

Per-layer keys: `glob`, `select`, `cmap`, `vmin`, `vmax`, `opacity`, `opacity_ramp`,
`log_scale`, `stride`, `radius`, `visible`, `order`. Shared keys: `radius` (the base
radius), `layer_separation`, `layers_root`, `hours_per_second`, `clock`, `max_points`,
`unlit`, `emissive_intensity`, `material`, `backend`, `fractional_cutout`, `ui`.

## Rendering notes

| Topic | What to know |
|---|---|
| Transparency | Fractional cutout opacity must be on. The extension sets both carb keys and the matching attributes on the viewport's render product, and the app's `.kit` sets them at startup. |
| Brightness | `unlit = true` emits the colours (readable on the night side). MDL emission uses the same scale as light intensities, not 0-1: tune `emissive_intensity` (`w["pwat"].sphere.set_emissive_intensity(2500)`). |
| Colours | Built-in colormaps are sRGB and linearised before upload. `cmap` also takes an (N,3) array or a matplotlib colormap. |
| Missing data | GRIB bitmap points become NaN and render at `Style.missing_opacity` (0 = a hole). |
| Shadows | `primvars:doNotCastShadows` is on, so a shell doesn't darken the globe. |
| Session layer | Everything is authored into the session layer: never saved into a .usd, never in undo. |
| Up axis | Follows the stage. Z-up uses ECEF axes (north +Z, 0°/0° at +X), matching `gen_earth_usda.py`. |
| Backends | `backend="auto"`: Fabric (usdrt) when FSD is on, else USD writes inside one `Sdf.ChangeBlock`. |

## Diagnostics

```python
from talon_defense.weather_overlay.diagnostics import opacity_probe, remove_opacity_probe
print(opacity_probe())   # four reference spheres + render settings + shell values, logged too
```

## Tests

Offline, no Kit needed (`pip install usd-core eccodes numpy pytest`):

```
python tests/make_test_grib.py tests/_data
pytest -q tests/test_offline.py
```

They cover GRIB decoding (including Lambert grids, missing values and scan order), mesh
winding, the MDL and preview material networks, the player, the layer manager's shared
clock and visibility rules, and settings-driven autostart with Kit stubbed out.
`talon_defense/weather_overlay/tests/` holds the in-Kit smoke test for `repo test`.

## Limits

- Structured grids only; reduced Gaussian grids must be regridded first.
- One variable per layer, one grid per layer.
- The shell is a sphere of `radius` stage units - not the WGS84 ellipsoid, and with no
  Cesium georeference transform, so it won't line up exactly with a Cesium globe yet.
