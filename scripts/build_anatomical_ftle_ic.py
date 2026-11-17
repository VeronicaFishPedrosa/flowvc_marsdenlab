#!/usr/bin/env python3
"""
Precompute which Cartesian FTLE-grid points lie inside the vessel lumen and
write a flowVC FTLE_ICFile (use with FTLE_GenerateMesh = 0).

FTLE_MeshBounds and resolution in the .in file must match this IC file.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv

# FTLEPoint is 128 bytes (WSL gcc x86_64): X[3], V[3], vorticity[3],
# ElementIndex, AuxElementIndex, LeftDomain, pad, LeftDomainTime, Scalar,
# HaveFTLE, pad, FTLEwT, FTLEwoT.
FTLEPOINT_FMT = "<" + ("d" * 9) + "iiii" + "dd" + "ii" + "dd"
assert struct.calcsize(FTLEPOINT_FMT) == 128
FTLE_OUTSIDE = -1.0


def read_coordinates(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (n,) = struct.unpack_from("<i", raw, 0)
    pts = np.frombuffer(raw, dtype="<f8", count=n * 3, offset=4).reshape(n, 3).copy()
    return pts


def read_connectivity(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (n,) = struct.unpack_from("<i", raw, 0)
    conn = np.frombuffer(raw, dtype="<i4", count=n * 4, offset=4).reshape(n, 4).copy()
    return conn


def read_adjacency(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    (n,) = struct.unpack_from("<i", raw, 0)
    adj = np.frombuffer(raw, dtype="<i4", count=n * 4, offset=4).reshape(n, 4).copy()
    return adj


def pack_ftle_point(
    x: float, y: float, z: float,
    element_index: int,
    left_domain: int,
    have_ftle: int,
    ftle_wt: float,
    ftle_wot: float,
) -> bytes:
    return struct.pack(
        FTLEPOINT_FMT,
        x, y, z,                 # X[3]
        0.0, 0.0, 0.0,           # V[3]
        0.0, 0.0, 0.0,           # vorticity[3]
        int(element_index),      # ElementIndex
        -1,                      # AuxElementIndex
        int(left_domain),        # LeftDomain
        0,                       # pad
        -1.0 if left_domain else 0.0,  # LeftDomainTime
        0.0,                     # Scalar
        int(have_ftle),          # HaveFTLE
        0,                       # pad
        float(ftle_wt),          # FTLEwT
        float(ftle_wot),         # FTLEwoT
    )


def point_in_tet(q: np.ndarray, coords: np.ndarray, conn_row: np.ndarray, eps: float = 1e-10) -> bool:
    """flowVC-style barycentric containment (mesh.c natural coordinates)."""
    a = coords[conn_row[0]]
    b = coords[conn_row[1]] - a
    c = coords[conn_row[2]] - a
    d = coords[conn_row[3]] - a
    rhs = q - a
    cxd = np.cross(c, d)
    det = float(np.dot(b, cxd))
    if abs(det) < 1e-30:
        return False
    r = float(np.dot(rhs, cxd)) / det
    s = float(np.dot(rhs, np.cross(d, b))) / det
    t = float(np.dot(rhs, np.cross(b, c))) / det
    u = 1.0 - r - s - t
    return r >= -eps and s >= -eps and t >= -eps and u >= -eps


def locate_grid(
    coords: np.ndarray,
    conn: np.ndarray,
    adj,
    bounds: tuple[float, float, float, float, float, float],
    res: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xres, yres, zres = res
    xs = np.linspace(xmin, xmax, xres)
    ys = np.linspace(ymin, ymax, yres)
    zs = np.linspace(zmin, zmax, zres)
    # Match ftle.c: for i, for j, write ZRes points (k varies fastest).
    I, J, K = np.meshgrid(np.arange(xres), np.arange(yres), np.arange(zres), indexing="ij")
    pts = np.column_stack([xs[I.ravel()], ys[J.ravel()], zs[K.ravel()]])

    # VTK cell locator, then barycentric refine using adjacency (VTK and flowVC
    # can disagree on which tet owns a shared-face point).
    t0 = time.time()
    mesh = pv.UnstructuredGrid({pv.CellType.TETRA: conn}, coords)
    print(f"  built UnstructuredGrid in {time.time()-t0:.1f}s")
    t0 = time.time()
    eids = np.asarray(mesh.find_containing_cell(pts), dtype=np.int32)
    print(f"  VTK location done in {time.time()-t0:.1f}s")

    if adj is not None:
        t0 = time.time()
        fixed = demoted = 0
        for idx in np.where(eids >= 0)[0]:
            e = int(eids[idx])
            q = pts[idx]
            if point_in_tet(q, coords, conn[e]):
                continue
            found = -1
            for nb in adj[e]:
                if nb >= 0 and point_in_tet(q, coords, conn[int(nb)]):
                    found = int(nb)
                    break
            if found >= 0:
                eids[idx] = found
                fixed += 1
            else:
                eids[idx] = -1
                demoted += 1
        print(f"  barycentric refine: fixed={fixed}, demoted_to_exterior={demoted} in {time.time()-t0:.1f}s")

    return pts, eids

def apply_neighbor_poison(
    left: np.ndarray, have: np.ndarray, xres: int, yres: int, zres: int
) -> np.ndarray:
    """Match ftle.c: exterior points also set HaveFTLE on their 6 face-neighbors."""
    have = have.copy()
    shape = (xres, yres, zres)
    left3 = left.reshape(shape)
    have3 = have.reshape(shape)
    ii, jj, kk = np.where(left3)
    for i, j, k in zip(ii, jj, kk):
        have3[i, j, k] = 1
        if i > 0:
            have3[i - 1, j, k] = 1
        if i < xres - 1:
            have3[i + 1, j, k] = 1
        if j > 0:
            have3[i, j - 1, k] = 1
        if j < yres - 1:
            have3[i, j + 1, k] = 1
        if k > 0:
            have3[i, j, k - 1] = 1
        if k < zres - 1:
            have3[i, j, k + 1] = 1
    return have3.ravel()


def write_ic(
    path: Path,
    pts: np.ndarray,
    eids: np.ndarray,
    res: tuple[int, int, int],
    poison_neighbors: bool,
) -> dict:
    xres, yres, zres = res
    n = xres * yres * zres
    assert pts.shape[0] == n
    left = (eids < 0).astype(np.int32)
    have = left.copy()  # exterior points do not need FTLE
    if poison_neighbors:
        have = apply_neighbor_poison(left, have, xres, yres, zres)

    with path.open("wb") as f:
        f.write(struct.pack("<iii", xres, yres, zres))
        for idx in range(n):
            if left[idx]:
                f.write(
                    pack_ftle_point(
                        pts[idx, 0], pts[idx, 1], pts[idx, 2],
                        -1, 1, 1, FTLE_OUTSIDE, FTLE_OUTSIDE,
                    )
                )
            else:
                f.write(
                    pack_ftle_point(
                        pts[idx, 0], pts[idx, 1], pts[idx, 2],
                        int(eids[idx]), 0, int(have[idx]), 0.0, 0.0,
                    )
                )

    inside = int((eids >= 0).sum())
    poisoned_interior = int(((eids >= 0) & (have == 1)).sum())
    stats = {
        "total": n,
        "inside": inside,
        "outside": n - inside,
        "hit_rate": inside / n,
        "poisoned_interior": poisoned_interior,
        "computable": inside - poisoned_interior,
    }
    return stats


def read_ic(path: Path):
    data = path.read_bytes()
    xres, yres, zres = struct.unpack_from("<iii", data, 0)
    n = xres * yres * zres
    assert (len(data) - 12) // n == 128
    eids = np.empty(n, dtype=np.int32)
    left = np.empty(n, dtype=np.int32)
    xs = np.empty((n, 3), dtype=np.float64)
    off = 12
    for i in range(n):
        vals = struct.unpack_from(FTLEPOINT_FMT, data, off)
        xs[i] = vals[0:3]
        eids[i] = vals[9]
        left[i] = vals[11]
        off += 128
    return (xres, yres, zres), xs, eids, left


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", type=Path, required=True,
                    help="{prefix}_coordinates.bin from convert_svmultiphysics_to_flowvc.py")
    ap.add_argument("--conn", type=Path, required=True,
                    help="{prefix}_connectivity.bin from convert_svmultiphysics_to_flowvc.py")
    ap.add_argument("--adj", type=Path, required=True,
                    help="{prefix}_adjacency.bin from convert_svmultiphysics_to_flowvc.py")
    ap.add_argument("--out", type=Path, required=True,
                    help="IC file to write; put it in Path_Data and set FTLE_ICFile to its name")
    ap.add_argument("--bounds", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    required=True,
                    help="Must match FTLE_MeshBounds in the .in file that loads this IC")
    ap.add_argument("--res", type=int, nargs=3, metavar=("XRES", "YRES", "ZRES"), required=True,
                    help="Must match FTLE_MeshBounds.XRes/YRes/ZRes in the .in file")
    ap.add_argument("--poison-neighbors", action=argparse.BooleanOptionalAction, default=True,
                    help="Set HaveFTLE=1 on face-neighbors of exterior points (required for production)")
    ap.add_argument("--validate-against", type=Path, default=None,
                    help="Existing flowVC IC file; compare interior ElementIndex agreement and exit")
    args = ap.parse_args()

    print(f"Reading {args.coords}")
    coords = read_coordinates(args.coords)
    print(f"Reading {args.conn}")
    conn = read_connectivity(args.conn)
    adj = read_adjacency(args.adj) if args.adj.is_file() else None
    if adj is None:
        print(f"  WARNING: {args.adj} not found; skipping barycentric refine")
    print(f"  nodes={coords.shape[0]:,}  tets={conn.shape[0]:,}")

    bounds = tuple(args.bounds)
    res = tuple(args.res)

    print(f"Locating Cartesian {res[0]}x{res[1]}x{res[2]} = {res[0]*res[1]*res[2]:,} grid points...")
    pts, eids = locate_grid(coords, conn, adj, bounds, res)
    inside_before = int((eids >= 0).sum())
    print(f"  inside lumen: {inside_before:,} / {pts.shape[0]:,} ({100*inside_before/pts.shape[0]:.2f}%)")

    if args.validate_against is not None:
        print(f"\nValidating against {args.validate_against}")
        (xr, yr, zr), xref, eid_ref, left_ref = read_ic(args.validate_against)
        assert (xr, yr, zr) == res, f"res mismatch: IC {(xr,yr,zr)} vs requested {res}"
        max_coord_err = float(np.max(np.abs(pts - xref)))
        print(f"  max |dX| vs reference IC: {max_coord_err:.3e}")
        both_inside = (eids >= 0) & (left_ref == 0)
        ref_only = (eids < 0) & (left_ref == 0)
        us_only = (eids >= 0) & (left_ref == 1)
        print(f"  both inside: {both_inside.sum()}")
        print(f"  reference-only inside (we missed): {ref_only.sum()}")
        print(f"  us-only inside (reference missed / poisoned?): {us_only.sum()}")
        if both_inside.any():
            agree = (eids[both_inside] == eid_ref[both_inside]).sum()
            print(f"  ElementIndex agreement on overlap: {agree}/{both_inside.sum()}")
        sys.exit(0 if max_coord_err < 1e-9 and ref_only.sum() == 0 else 1)

    stats = write_ic(args.out, pts, eids, res, poison_neighbors=args.poison_neighbors)
    print(f"\nWrote {args.out}")
    print(f"  total={stats['total']:,}  inside={stats['inside']:,}  "
          f"hit_rate={100*stats['hit_rate']:.2f}%")
    print(f"  poisoned_interior={stats['poisoned_interior']:,}  "
          f"computable (HaveFTLE=0 & inside)={stats['computable']:,}")
    print("\n.in file must use the SAME FTLE_CartMesh bounds/res, FTLE_GenerateMesh=0,")
    print(f"and FTLE_ICFile = {args.out.name}")


if __name__ == "__main__":
    main()
