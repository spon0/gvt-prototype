#!/usr/bin/env python3
"""Generate a medium-resolution UV-sphere Mesh representing a placeholder
Earth, written out as raw USD ASCII (.usda) text -- no pxr/USD Python
bindings required (this sandbox has no network access to install usd-core).

Topology: standard UV sphere with a single vertex at each pole (triangle
fan caps) and quad bands in between. Points/normals use vertex
interpolation (a sphere's normals have no seam). UVs use faceVarying
interpolation to correctly handle the longitude wrap-around seam.
"""
import math

RADIUS = 6371000.0   # meters -- mean Earth radius (WGS84-ish placeholder)
LON_SEGMENTS = 64     # divisions around the equator ("medium" resolution)
LAT_SEGMENTS = 32     # divisions from pole to pole

# ---------------------------------------------------------------------------
# 1. Geometry generation
# ---------------------------------------------------------------------------
points = []
normals = []
ring_start = {}

# North pole = point index 0
points.append((0.0, RADIUS, 0.0))
normals.append((0.0, 1.0, 0.0))

for i in range(1, LAT_SEGMENTS):
    theta = math.pi * i / LAT_SEGMENTS          # 0 < theta < pi
    y = RADIUS * math.cos(theta)
    r_xz = RADIUS * math.sin(theta)
    ring_start[i] = len(points)
    for j in range(LON_SEGMENTS):
        phi = 2.0 * math.pi * j / LON_SEGMENTS
        x = r_xz * math.cos(phi)
        z = r_xz * math.sin(phi)
        points.append((x, y, z))
        normals.append((x / RADIUS, y / RADIUS, z / RADIUS))

south_pole_index = len(points)
points.append((0.0, -RADIUS, 0.0))
normals.append((0.0, -1.0, 0.0))

face_vertex_counts = []
face_vertex_indices = []
face_uvs = []  # faceVarying -- one (u, v) per face-vertex, same order as face_vertex_indices

first_ring = ring_start[1]

# North cap (triangle fan)
for j in range(LON_SEGMENTS):
    j_next = (j + 1) % LON_SEGMENTS
    v0, v1, v2 = 0, first_ring + j_next, first_ring + j
    face_vertex_counts.append(3)
    face_vertex_indices.extend([v0, v1, v2])
    v_ring = 1.0 - 1.0 / LAT_SEGMENTS
    face_uvs.extend([
        ((j + 0.5) / LON_SEGMENTS, 1.0),
        ((j + 1) / LON_SEGMENTS, v_ring),
        (j / LON_SEGMENTS, v_ring),
    ])

# Middle quad bands
for i in range(1, LAT_SEGMENTS - 1):
    ring_a, ring_b = ring_start[i], ring_start[i + 1]
    v_a = 1.0 - i / LAT_SEGMENTS
    v_b = 1.0 - (i + 1) / LAT_SEGMENTS
    for j in range(LON_SEGMENTS):
        j_next = (j + 1) % LON_SEGMENTS
        v0, v1, v2, v3 = ring_a + j, ring_a + j_next, ring_b + j_next, ring_b + j
        face_vertex_counts.append(4)
        face_vertex_indices.extend([v0, v1, v2, v3])
        u0, u1 = j / LON_SEGMENTS, (j + 1) / LON_SEGMENTS
        face_uvs.extend([(u0, v_a), (u1, v_a), (u1, v_b), (u0, v_b)])

# South cap (triangle fan)
last_ring = ring_start[LAT_SEGMENTS - 1]
v_last = 1.0 / LAT_SEGMENTS
for j in range(LON_SEGMENTS):
    j_next = (j + 1) % LON_SEGMENTS
    v0, v1, v2 = last_ring + j, last_ring + j_next, south_pole_index
    face_vertex_counts.append(3)
    face_vertex_indices.extend([v0, v1, v2])
    face_uvs.extend([
        (j / LON_SEGMENTS, v_last),
        ((j + 1) / LON_SEGMENTS, v_last),
        ((j + 0.5) / LON_SEGMENTS, 0.0),
    ])

# ---------------------------------------------------------------------------
# 2. Sanity checks
# ---------------------------------------------------------------------------
assert len(points) == 2 + (LAT_SEGMENTS - 1) * LON_SEGMENTS
assert len(points) == len(normals)
assert sum(face_vertex_counts) == len(face_vertex_indices) == len(face_uvs)
assert len(face_vertex_counts) == 2 * LON_SEGMENTS + (LAT_SEGMENTS - 2) * LON_SEGMENTS
assert max(face_vertex_indices) == len(points) - 1
assert min(face_vertex_indices) == 0

# Winding-order check: for a sample of faces, the geometric face normal
# (cross product of two edges) should point the same general direction as
# the authored (analytic) vertex normals -- confirms outward-facing CCW
# winding consistent with the normals we authored.
def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

idx = 0
bad = 0
checked = 0
bad_examples = []
n_north_cap = LON_SEGMENTS
n_bands = (LAT_SEGMENTS - 2) * LON_SEGMENTS
for face_i, count in enumerate(face_vertex_counts):
    verts = face_vertex_indices[idx: idx + count]
    idx += count
    p0, p1, p2 = points[verts[0]], points[verts[1]], points[verts[2]]
    face_normal = cross(vsub(p1, p0), vsub(p2, p0))
    avg_normal = tuple(sum(normals[v][k] for v in verts) / count for k in range(3))
    checked += 1
    d = dot(face_normal, avg_normal)
    if d <= 0:
        bad += 1
        if face_i < n_north_cap:
            section = "north_cap"
        elif face_i < n_north_cap + n_bands:
            section = "band"
        else:
            section = "south_cap"
        if len(bad_examples) < 5:
            bad_examples.append((face_i, section, verts, d))
if bad:
    for ex in bad_examples:
        print("BAD FACE:", ex)
    raise SystemExit(f"Winding check FAILED: {bad}/{checked} faces have inward-facing winding")
print(f"Winding check OK: {checked} faces all outward-facing (0 mismatches)")
print(f"points={len(points)} faces={len(face_vertex_counts)} indices={len(face_vertex_indices)}")

# ---------------------------------------------------------------------------
# 3. Write .usda
# ---------------------------------------------------------------------------
def fmt(v, nd):
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if "." in f"{v:.{nd}f}" else f"{v:.{nd}f}"

def fmt_point(p):
    return "(" + ", ".join(fmt(c, 1) for c in p) + ")"

def fmt_normal(n):
    return "(" + ", ".join(fmt(c, 6) for c in n) + ")"

def fmt_uv(uv):
    return "(" + ", ".join(fmt(c, 6) for c in uv) + ")"

# Relative to this .usda file (NOT to whatever stage references it -- USD
# resolves asset paths relative to the layer that authors them). Put the
# texture file at <this folder>/textures/earth_diffuse.jpg to light it up;
# `inputs:fallback` on the texture shader keeps the flat ocean-blue color as
# a safe fallback if that file isn't there yet.
TEXTURE_RELATIVE_PATH = "textures/earth_diffuse.jpg"

out_path = "/tmp/claude-0/-home-claude/a9d55fd3-8dc7-518b-8a5a-f4ac1db989f2/scratchpad/earth_medium.usda"

with open(out_path, "w") as f:
    f.write('#usda 1.0\n')
    f.write('(\n')
    f.write('    defaultPrim = "Earth"\n')
    f.write('    upAxis = "Y"\n')
    f.write('    metersPerUnit = 1\n')
    f.write(f'    doc = "Medium-resolution placeholder Earth sphere ({LON_SEGMENTS}x{LAT_SEGMENTS} UV-sphere mesh, radius={RADIUS:.0f}m). Generated for the GVT prototype; not geospatially accurate -- swap for Cesium\'s real globe later. Material looks for a diffuse texture at {TEXTURE_RELATIVE_PATH} (relative to this file) and falls back to a flat ocean-blue color if that file is missing."\n')
    f.write(')\n\n')

    f.write('def Xform "Earth" (\n')
    f.write('    kind = "component"\n')
    f.write(')\n{\n')
    f.write('    def Mesh "Earth_Geom"\n')
    f.write('    {\n')
    f.write('        uniform bool doubleSided = 0\n')
    f.write(f'        float3[] extent = [{fmt_point((-RADIUS,-RADIUS,-RADIUS))}, {fmt_point((RADIUS,RADIUS,RADIUS))}]\n')

    f.write('        int[] faceVertexCounts = [' + ', '.join(str(c) for c in face_vertex_counts) + ']\n')
    f.write('        int[] faceVertexIndices = [' + ', '.join(str(c) for c in face_vertex_indices) + ']\n')

    f.write('        normal3f[] normals = [' + ', '.join(fmt_normal(n) for n in normals) + '] (\n')
    f.write('            interpolation = "vertex"\n')
    f.write('        )\n')

    f.write('        point3f[] points = [' + ', '.join(fmt_point(p) for p in points) + ']\n')

    f.write('        texCoord2f[] primvars:st = [' + ', '.join(fmt_uv(uv) for uv in face_uvs) + '] (\n')
    f.write('            interpolation = "faceVarying"\n')
    f.write('        )\n')

    f.write('        uniform token subdivisionScheme = "none"\n')
    f.write('        rel material:binding = </Earth/Looks/EarthMat>\n')
    f.write('    }\n\n')

    f.write('    def Scope "Looks"\n')
    f.write('    {\n')
    f.write('        def Material "EarthMat"\n')
    f.write('        {\n')
    f.write('            token outputs:surface.connect = </Earth/Looks/EarthMat/Surface.outputs:surface>\n\n')
    f.write('            def Shader "Surface"\n')
    f.write('            {\n')
    f.write('                uniform token info:id = "UsdPreviewSurface"\n')
    f.write('                color3f inputs:diffuseColor.connect = </Earth/Looks/EarthMat/DiffuseTexture.outputs:rgb>\n')
    f.write('                float inputs:roughness = 0.65\n')
    f.write('                float inputs:metallic = 0\n')
    f.write('                int inputs:useSpecularWorkflow = 0\n')
    f.write('                token outputs:surface\n')
    f.write('            }\n\n')
    f.write('            def Shader "DiffuseTexture"\n')
    f.write('            {\n')
    f.write('                uniform token info:id = "UsdUVTexture"\n')
    f.write(f'                asset inputs:file = @{TEXTURE_RELATIVE_PATH}@\n')
    f.write('                float4 inputs:fallback = (0.09, 0.28, 0.55, 1)\n')
    f.write('                float2 inputs:st.connect = </Earth/Looks/EarthMat/PrimvarReader.outputs:result>\n')
    f.write('                token inputs:wrapS = "repeat"\n')
    f.write('                token inputs:wrapT = "clamp"\n')
    f.write('                color3f outputs:rgb\n')
    f.write('            }\n\n')
    f.write('            def Shader "PrimvarReader"\n')
    f.write('            {\n')
    f.write('                uniform token info:id = "UsdPrimvarReader_float2"\n')
    f.write('                token inputs:varname = "st"\n')
    f.write('                float2 inputs:fallback = (0, 0)\n')
    f.write('                float2 outputs:result\n')
    f.write('            }\n')
    f.write('        }\n')
    f.write('    }\n')
    f.write('}\n')

print(f"wrote {out_path}")
import os
print("file size bytes:", os.path.getsize(out_path))
