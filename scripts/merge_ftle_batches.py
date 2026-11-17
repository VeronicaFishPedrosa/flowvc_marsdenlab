#!/usr/bin/env python3
"""
Merge batched flowVC FTLE runs into one folder, numbered by launch time.

only needed if memory is insufficient for a single run.

    python3 merge_ftle_batches.py DEST BATCH_DIR [BATCH_DIR ...] \
        --prefix P --t-first T0 --t-delta DT --num-slides N
"""
from __future__ import annotations

import argparse
import re
import shutil
import struct
import sys
from pathlib import Path


def read_t0(path: Path) -> float:
    with open(path, "rb") as fh:
        return struct.unpack("<d", fh.read(8))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dest", type=Path)
    ap.add_argument("batch_dirs", type=Path, nargs="+")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--t-first", type=float, required=True)
    ap.add_argument("--t-delta", type=float, required=True)
    ap.add_argument("--num-slides", type=int, required=True)
    args = ap.parse_args()

    pat = re.compile(rf"^{re.escape(args.prefix)}\.(\d+)\.bin$")
    moves: dict[int, tuple[Path, Path]] = {}
    for bdir in args.batch_dirs:
        for src in sorted(bdir.iterdir()):
            if not pat.match(src.name):
                continue
            t0 = read_t0(src)
            pos = (t0 - args.t_first) / args.t_delta
            n = round(pos)
            if abs(pos - n) > 1e-4 or not 0 <= n < args.num_slides:
                print(f"error: {src} has t0={t0!r}, not on the release grid", file=sys.stderr)
                return 1
            if n in moves:
                print(f"error: slide {n} (t0={t0:.6f}) appears in {moves[n][0].parent} and {bdir}",
                      file=sys.stderr)
                return 1
            src_not = src.with_name(f"{args.prefix}_noT.{pat.match(src.name).group(1)}.bin")
            if not src_not.is_file():
                print(f"error: {src_not} missing", file=sys.stderr)
                return 1
            moves[n] = (src, src_not)

    args.dest.mkdir(parents=True, exist_ok=True)
    for n, (src, src_not) in sorted(moves.items()):
        shutil.move(str(src), args.dest / f"{args.prefix}.{n}.bin")
        shutil.move(str(src_not), args.dest / f"{args.prefix}_noT.{n}.bin")
        print(f"  slide {n:2d}  t0={read_t0(args.dest / f'{args.prefix}.{n}.bin'):.6f}  <- {src.parent.name}/{src.name}")
    # Grid and run-parameter files are identical across batches.
    for extra in (f"{args.prefix}.info", f"{args.prefix}_Cartesian.bin"):
        src = args.batch_dirs[0] / extra
        if src.is_file():
            shutil.copy2(src, args.dest / extra)

    missing = sorted(set(range(args.num_slides)) - set(moves))
    print(f"Merged {len(moves)}/{args.num_slides} slides into {args.dest}")
    if missing:
        print(f"error: missing slides {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
