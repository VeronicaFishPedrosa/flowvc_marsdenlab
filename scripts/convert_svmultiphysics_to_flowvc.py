#!/usr/bin/env python3
"""
Convert svMultiPhysics VTU results to flowVC unstructured-mesh binaries.

Writes {prefix}_coordinates.bin, _connectivity.bin, _adjacency.bin, and
{prefix}_vel.{step}.bin. Layouts match mesh.c LoadUnstructMeshData() and
velocity.c LoadUnstructVelDataFrame().

"""

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import meshio
except ImportError:
    sys.exit(
        "This script requires the 'meshio' package (pip install meshio). "
    )


STEP_RE = re.compile(r"result_(\d+)\.vtu$")

# NeighborIndex slot -> local node indices of the opposite face.
FACE_LOCAL_NODES = np.array([
    [0, 2, 3],  # slot 0, opposite local node 1
    [0, 1, 3],  # slot 1, opposite local node 2
    [0, 1, 2],  # slot 2, opposite local node 3
    [1, 2, 3],  # slot 3, opposite local node 0
])


def find_vtu_files(vtu_dir: Path):
    """Return sorted list of (step:int, path:Path) discovered in vtu_dir."""
    files = []
    for p in sorted(vtu_dir.glob("result_*.vtu")):
        m = STEP_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda t: t[0])
    if not files:
        raise FileNotFoundError(f"No result_*.vtu files found in {vtu_dir}")
    return files


def load_step_time_map(flux_file):
    """Build {step: time} from a svMultiPhysics flux file. Returns {} if missing."""
    mapping = {}
    if flux_file is None:
        return mapping
    flux_file = Path(flux_file)
    if not flux_file.exists():
        return mapping

    with open(flux_file, "r") as f:
        past_header = False
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lower = stripped.lower()
            if lower.startswith("step") or lower.startswith("face areas") or lower.startswith("domain volumes"):
                past_header = True
                continue
            if not past_header:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                step = int(float(parts[0]))
                time = float(parts[1])
            except ValueError:
                continue
            mapping[step] = time
    return mapping


def get_tet_cells(mesh) -> np.ndarray:
    """Extract and concatenate all 'tetra' cell blocks from a meshio mesh."""
    tet_blocks = [c.data for c in mesh.cells if c.type == "tetra"]
    if not tet_blocks:
        available = sorted({c.type for c in mesh.cells})
        raise ValueError(
            f"No 'tetra' cell block found (found cell types: {available}). "
            "Is this a volume mesh? 2D/triangular meshes are not supported by this script."
        )
    if len(tet_blocks) > 1:
        print(f"  NOTE: {len(tet_blocks)} separate 'tetra' blocks found; concatenating.")
    return np.vstack(tet_blocks).astype(np.int64) 


def write_coordinates(out_path: Path, points: np.ndarray):
    """Write {prefix}_coordinates.bin: int32 NumNodes, then double x y z per node."""
    n = points.shape[0]
    pts64 = np.ascontiguousarray(points, dtype="<f8")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", n))
        pts64.tofile(f)
    print(f"  Wrote {out_path.name}  ({n} nodes, {out_path.stat().st_size:,} bytes)")


def write_connectivity(out_path: Path, cells: np.ndarray):
    """Write {prefix}_connectivity.bin: int32 NumElements, then int32 n0 n1 n2 n3 per tet."""
    n = cells.shape[0] 
    flat = np.ascontiguousarray(cells, dtype="<i4").reshape(-1) 
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", n))
        flat.tofile(f)
    print(f"  Wrote {out_path.name}  ({n} elements, {out_path.stat().st_size:,} bytes)")


def compute_adjacency(cells: np.ndarray) -> np.ndarray:
    """Neighbor index across each tet face, or -1 on the domain boundary."""
    n_elem = cells.shape[0]
    if n_elem == 0:
        return np.zeros((0, 4), dtype="<i4")

    faces = cells[:, FACE_LOCAL_NODES]  # (n_elem, 4, 3)
    faces = faces.reshape(n_elem * 4, 3)

    elem_idx = np.repeat(np.arange(n_elem, dtype=np.int64), 4)
    slot_idx = np.tile(np.arange(4, dtype=np.int64), n_elem)

    faces_sorted = np.sort(faces, axis=1)
    base = int(cells.max()) + 1
    # Pack (n0, n1, n2) into one int64 key. Needs base**3 < 2^63 (~2 million nodes).
    if base ** 3 >= 2 ** 63:
        raise ValueError(f"Mesh has {base} nodes; too many for the int64 face key.")
    keys = (faces_sorted[:, 0] * base + faces_sorted[:, 1]) * base + faces_sorted[:, 2]

    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    elem_s = elem_idx[order]
    slot_s = slot_idx[order]

    neighbor = np.full((n_elem, 4), -1, dtype="<i4")

    n_faces = keys_s.shape[0]
    n_boundary = 0
    n_interior_pairs = 0
    i = 0
    while i < n_faces:
        j = i + 1
        if j < n_faces and keys_s[j] == keys_s[i]:
            if j + 1 < n_faces and keys_s[j + 1] == keys_s[i]:
                k = j + 1
                while k < n_faces and keys_s[k] == keys_s[i]:
                    k += 1
                raise ValueError(
                    f"Non-manifold mesh: a face is shared by {k - i} elements "
                    f"(expected at most 2). Elements involved: {list(elem_s[i:k])}. "
                    "Adjacency cannot be computed unambiguously."
                )
            e0, s0 = int(elem_s[i]), int(slot_s[i])
            e1, s1 = int(elem_s[j]), int(slot_s[j])
            neighbor[e0, s0] = e1
            neighbor[e1, s1] = e0
            n_interior_pairs += 1
            i = j + 1
        else:
            n_boundary += 1
            i = j

    print(f"  Adjacency: {n_interior_pairs} interior face-pairs, {n_boundary} boundary faces "
          f"(sanity check: {n_interior_pairs * 2 + n_boundary} == {n_faces} total faces)")
    return neighbor


def write_adjacency(out_path: Path, neighbor: np.ndarray):
    """Write {prefix}_adjacency.bin: int32 NumElements, then NeighborIndex[0..3] per element."""
    n = neighbor.shape[0]
    flat = np.ascontiguousarray(neighbor, dtype="<i4").reshape(-1)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", n))
        flat.tofile(f)
    print(f"  Wrote {out_path.name}  ({n} elements, {out_path.stat().st_size:,} bytes)")


def write_velocity_frame(out_path: Path, timestamp: float, velocity: np.ndarray):
    """Write {prefix}_vel.{step}.bin: double timestamp, then double u v w per node."""
    vel64 = np.ascontiguousarray(velocity, dtype="<f8")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<d", timestamp))
        vel64.tofile(f)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--vtu-dir", type=Path, required=True,
                     help="Directory containing result_*.vtu files")
    ap.add_argument("--out-dir", type=Path, required=True,
                     help="Directory to write flowVC binary files to (becomes Path_Data)")
    ap.add_argument("--prefix", required=True,
                     help="Data_InFilePrefix to use for output file names")
    ap.add_argument("--flux-file", type=Path, default=None,
                     help="Flux file for step->time lookup. Default: B_NS_Velocity_flux.txt in --vtu-dir.")
    ap.add_argument("--dt", type=float, default=None,
                     help="CFD time step; time = step * dt for steps missing from --flux-file. "
                          "Required if there is no usable flux file.")
    ap.add_argument("--velocity-field", default="Velocity",
                     help="Point-data array name for velocity.")
    ap.add_argument("--skip-topology", action="store_true",
                     help="Skip writing coordinates/connectivity/adjacency.")
    args = ap.parse_args()

    if not args.vtu_dir.exists():
        sys.exit(f"ERROR: --vtu-dir does not exist: {args.vtu_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    flux_file = args.flux_file
    if flux_file is None:
        candidate = args.vtu_dir / "B_NS_Velocity_flux.txt"
        if candidate.exists():
            flux_file = candidate

    vtu_files = find_vtu_files(args.vtu_dir)
    print(f"Found {len(vtu_files)} VTU files, steps {vtu_files[0][0]} .. {vtu_files[-1][0]}")

    step_time_map = load_step_time_map(flux_file)
    if step_time_map:
        print(f"Loaded {len(step_time_map)} step->time entries from {flux_file}")
        missing = [st for st, _ in vtu_files if st not in step_time_map]
        if missing and args.dt is None:
            sys.exit(f"ERROR: {len(missing)} steps (e.g. {missing[0]}) are not in {flux_file}. "
                     "Pass --dt so their times can be computed as step * dt.")
    else:
        if args.dt is None:
            sys.exit("ERROR: no usable flux file found. Pass --flux-file, or --dt so that "
                     "time = step * dt.")
        print(f"No usable flux file found; falling back to time = step * {args.dt}")

    # Topology from the first frame (mesh is fixed).
    first_step, first_path = vtu_files[0]
    print(f"\nReading topology from {first_path.name} ...")
    mesh0 = meshio.read(first_path)
    points0 = mesh0.points
    cells0 = get_tet_cells(mesh0)
    n_nodes = points0.shape[0]
    n_elems = cells0.shape[0]
    print(f"  {n_nodes} nodes, {n_elems} tetrahedral elements")

    if not args.skip_topology:
        write_coordinates(args.out_dir / f"{args.prefix}_coordinates.bin", points0)
        write_connectivity(args.out_dir / f"{args.prefix}_connectivity.bin", cells0)
        print("  Computing element adjacency...")
        neighbor0 = compute_adjacency(cells0)
        write_adjacency(args.out_dir / f"{args.prefix}_adjacency.bin", neighbor0)
    else:
        print("  --skip-topology set: not writing coordinates/connectivity/adjacency")

    print(f"\nConverting {len(vtu_files)} velocity frames ...")
    n_written = 0
    prev_time = None
    for step, path in vtu_files:
        mesh = meshio.read(path)

        if mesh.points.shape[0] != n_nodes:
            sys.exit(
                f"ERROR: {path.name} has {mesh.points.shape[0]} nodes, expected {n_nodes} "
                f"(from {first_path.name}). Mesh is not fixed across frames -- this converter "
                "cannot handle a remeshed/adaptive simulation."
            )

        if args.velocity_field not in mesh.point_data:
            sys.exit(
                f"ERROR: point data array '{args.velocity_field}' not found in {path.name}. "
                f"Available arrays: {list(mesh.point_data.keys())}"
            )

        velocity = np.asarray(mesh.point_data[args.velocity_field])
        if velocity.shape != (n_nodes, 3):
            sys.exit(
                f"ERROR: velocity array in {path.name} has shape {velocity.shape}, "
                f"expected ({n_nodes}, 3)"
            )

        if step in step_time_map:
            timestamp = step_time_map[step]
        else:
            timestamp = step * args.dt

        if prev_time is not None and timestamp <= prev_time:
            print(f"  WARNING: non-increasing timestamp at step {step} "
                  f"({timestamp:.6g} <= {prev_time:.6g}); check --flux-file / --dt")
        prev_time = timestamp

        out_path = args.out_dir / f"{args.prefix}_vel.{step}.bin"
        write_velocity_frame(out_path, timestamp, velocity)
        n_written += 1
        print(f"  [{n_written}/{len(vtu_files)}] {path.name} -> {out_path.name}  (t={timestamp:.6g})")

    print(f"\nDone. Wrote {n_written} velocity frames to {args.out_dir}")

    step_delta = vtu_files[1][0] - vtu_files[0][0] if len(vtu_files) > 1 else 0
    steps_consistent = all(
        (vtu_files[i + 1][0] - vtu_files[i][0]) == step_delta for i in range(len(vtu_files) - 1)
    )

    def time_of(step):
        return step_time_map[step] if step in step_time_map else step * args.dt

    t0 = time_of(vtu_files[0][0])
    t1 = time_of(vtu_files[-1][0])
    data_tdelta = (t1 - t0) / (len(vtu_files) - 1) if len(vtu_files) > 1 else 0

    print("\nCorresponding flowVC parameter (.in) values:")
    print(f"  Data_MeshType      = 1")
    print(f"  Data_InFilePrefix  = {args.prefix}")
    print(f"  Data_SuffixTMin    = {vtu_files[0][0]}")
    print(f"  Data_SuffixTDelta  = {step_delta}"
          + ("" if steps_consistent else "   *** WARNING: step spacing is NOT constant across all files ***"))
    print(f"  Data_TRes          = {len(vtu_files)}")
    print(f"  Data_TDelta        = {data_tdelta:.6g}")
    print(f"  Data_TMin          = {t0:.6g}")
    print(f"  Data_MeshBounds.XMin/XMax = {points0[:, 0].min():.6f} / {points0[:, 0].max():.6f}")
    print(f"  Data_MeshBounds.YMin/YMax = {points0[:, 1].min():.6f} / {points0[:, 1].max():.6f}")
    print(f"  Data_MeshBounds.ZMin/ZMax = {points0[:, 2].min():.6f} / {points0[:, 2].max():.6f}")
    print("\nNOTE: Int_NormalFlow-related files (_normals.bin, _bflags.bin) were NOT generated. "
          "Only needed if Int_NormalFlow = 1 in the parameter file.")


if __name__ == "__main__":
    main()
