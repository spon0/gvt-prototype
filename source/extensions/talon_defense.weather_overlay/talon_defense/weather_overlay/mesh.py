"""Turn a lat/lon lattice into a closed (or patch) quad mesh on the unit sphere."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grib_io import GribGrid


@dataclass
class ShellMesh:
    points: np.ndarray  # (N, 3) float32, unit sphere
    normals: np.ndarray  # (N, 3) float32, outward
    face_counts: np.ndarray  # (F,) int32, all 4
    face_indices: np.ndarray  # (4F,) int32
    extent: np.ndarray  # (2, 3) float32


def unit_vectors(lats_deg: np.ndarray, lons_deg: np.ndarray, up_axis: str = "Y") -> np.ndarray:
    """Geographic -> unit vectors. Z-up uses ECEF axes; Y-up rotates ECEF so +Z(north) -> +Y.

    In both cases lat 0 / lon 0 (Gulf of Guinea) points along +X.
    """
    lat = np.radians(np.asarray(lats_deg, dtype=np.float64)).reshape(-1)
    lon = np.radians(np.asarray(lons_deg, dtype=np.float64)).reshape(-1)
    c = np.cos(lat)
    x, y, z = c * np.cos(lon), c * np.sin(lon), np.sin(lat)
    if up_axis.upper() == "Z":
        return np.stack([x, y, z], axis=-1)
    return np.stack([x, z, -y], axis=-1)  # proper rotation (det=+1), keeps handedness


def build_shell_mesh(grid: GribGrid, up_axis: str = "Y") -> ShellMesh:
    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        raise ValueError(f"Grid {grid.shape} is too small to mesh")
    p = unit_vectors(grid.lats, grid.lons, up_axis)

    idx = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    if grid.wrap_lon:
        right = np.roll(idx, -1, axis=1)  # last column stitches back to the first
        a, b, c, d = idx[:-1, :], right[:-1, :], right[1:, :], idx[1:, :]
    else:
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    quads = np.stack([a, b, c, d], axis=-1).reshape(-1, 4)

    # USD's default rightHanded orientation wants counter-clockwise faces seen from
    # outside. GRIB grids scan N->S or S->N and W->E or E->W, so measure the winding
    # on a sample of faces and flip if it points inward.
    sample = quads[:: max(1, len(quads) // 4096)]
    pa, pb, pd = p[sample[:, 0]], p[sample[:, 1]], p[sample[:, 3]]
    facing = np.einsum("ij,ij->i", np.cross(pb - pa, pd - pa), pa)
    if np.sum(np.sign(facing)) < 0:
        quads = quads[:, ::-1]

    points = p.astype(np.float32)
    return ShellMesh(
        points=points,
        normals=points.copy(),
        face_counts=np.full(len(quads), 4, dtype=np.int32),
        face_indices=np.ascontiguousarray(quads, dtype=np.int32).reshape(-1),
        extent=np.stack([points.min(axis=0), points.max(axis=0)]).astype(np.float32),
    )
