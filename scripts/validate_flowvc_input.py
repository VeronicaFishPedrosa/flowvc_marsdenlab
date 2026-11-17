"""
Validate a flowVC .in file without running flowVC.

Exit 0 if all checks pass, 1 otherwise.
"""
import argparse
import math
import sys
from pathlib import Path

TINY = 1.0e-10

# (name, type) sequence read by ReadInParameters(). 's' string, 'd' int, 'f' double.
SEQUENCE = [
    ("Path_Data", "s"), ("Path_Output", "s"),
    ("Dimensions", "d"),
    ("Data_MeshType", "d"), ("Data_InFilePrefix", "s"),
    ("Data_SuffixTMin", "d"), ("Data_SuffixTDelta", "d"),
    ("Data_TRes", "d"), ("Data_TDelta", "f"), ("Data_TMin", "f"),
    ("Data_TPeriodic", "d"), ("Data_XPeriodic", "d"),
    ("Data_MeshBounds.XMin", "f"), ("Data_MeshBounds.XMax", "f"),
    ("Data_MeshBounds.YMin", "f"), ("Data_MeshBounds.YMax", "f"),
    ("Data_MeshBounds.ZMin", "f"), ("Data_MeshBounds.ZMax", "f"),
    ("Fluid_Density", "f"), ("Fluid_Viscosity", "f"),
    ("Output_TStart", "f"), ("Output_TRes", "d"), ("Output_TDelta", "f"),
    ("Int_Type", "d"), ("Int_TimeStep", "f"), ("Int_Accuracy", "f"),
    ("Int_MinTimeStep", "f"), ("Int_MaxTimeStep", "f"),
    ("Int_TimeDirection", "d"), ("Int_NormalFlow", "d"),
    ("Int_NormalFlowScaling", "f"), ("Int_Extrapolate", "d"),
    ("Particle_Radius", "f"), ("Particle_Density", "f"), ("Particle_ICType", "d"),
    ("Gravity[0]", "f"), ("Gravity[1]", "f"), ("Gravity[2]", "f"),
    ("LocalSearchChecking", "d"),
    ("FTLE_Compute", "d"), ("FTLE_GenerateMesh", "d"), ("FTLE_ICFile", "s"),
    ("FTLE_CartMesh.XMin", "f"), ("FTLE_CartMesh.XMax", "f"),
    ("FTLE_CartMesh.YMin", "f"), ("FTLE_CartMesh.YMax", "f"),
    ("FTLE_CartMesh.ZMin", "f"), ("FTLE_CartMesh.ZMax", "f"),
    ("FTLE_CartMesh.XRes", "d"), ("FTLE_CartMesh.YRes", "d"), ("FTLE_CartMesh.ZRes", "d"),
    ("FTLE_IntTLength", "f"), ("FTLE_ComputeVariation", "d"),
    ("FTLE_VariationOutFreq", "d"), ("FTLE_OutFilePrefix", "s"),
    ("Trace_Compute", "d"),
    ("Trace_ReleaseStrategy", "d"), ("Trace_ReleaseTMax", "f"),
    ("Trace_GenerateMesh", "d"), ("Trace_InFile", "s"),
    ("Trace_MultipleInFiles", "d"), ("Trace_InFileFormat", "d"),
    ("Trace_OutFilePrefix", "s"), ("Trace_NumLaunchTimes", "d"),
    ("Trace_LaunchTimeSpacing", "f"), ("Trace_IntTLength", "f"),
    ("Trace_AlwaysOutput", "d"),
    ("Trace_CartMesh.XMin", "f"), ("Trace_CartMesh.XMax", "f"),
    ("Trace_CartMesh.YMin", "f"), ("Trace_CartMesh.YMax", "f"),
    ("Trace_CartMesh.ZMin", "f"), ("Trace_CartMesh.ZMax", "f"),
    ("Trace_CartMesh.XRes", "d"), ("Trace_CartMesh.YRes", "d"), ("Trace_CartMesh.ZRes", "d"),
    ("Trace_VorticityCompute", "d"), ("Trace_APCompute", "d"),
    ("Trace_CETCompute", "d"), ("Trace_CETAuxillaryMesh", "d"),
    ("Trace_CETMeshPrefix", "s"), ("Trace_CETSubsteps", "d"),
    ("Trace_RTCompute", "d"), ("Trace_RTOutFilePrefix", "s"),
    ("VelOut_Compute", "d"), ("VelOut_GenerateMesh", "d"),
    ("VelOut_InFile", "s"), ("VelOut_InFileFormat", "d"),
    ("VelOut_FilePrefix", "s"),
    ("VelOut_CartMesh.XMin", "f"), ("VelOut_CartMesh.XMax", "f"),
    ("VelOut_CartMesh.YMin", "f"), ("VelOut_CartMesh.YMax", "f"),
    ("VelOut_CartMesh.ZMin", "f"), ("VelOut_CartMesh.ZMax", "f"),
    ("VelOut_CartMesh.XRes", "d"), ("VelOut_CartMesh.YRes", "d"), ("VelOut_CartMesh.ZRes", "d"),
]


class Tokenizer:
    """Char-by-char scan matching ReadInNextValue() + fscanf token reads."""

    def __init__(self, text):
        self.text = text
        self.pos = 0

    def _getc(self):
        if self.pos >= len(self.text):
            return None
        c = self.text[self.pos]
        self.pos += 1
        return c

    def _ungetc(self):
        self.pos -= 1

    def _skip_ws_and_read_token(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in " \t\r\n":
            self.pos += 1
        return self.text[start:self.pos]

    def read_next_value(self):
        while True:
            c = self._getc()
            if c is None:
                raise EOFError("Unexpected end of file while looking for next parameter")
            if c in (" ", "\n", "\r"):
                continue
            if c == "#":
                while c != "\n":
                    c = self._getc()
                    if c is None:
                        raise EOFError("Unexpected end of file inside comment")
                continue
            else:
                self._ungetc()
                var_name = self._skip_ws_and_read_token()
                eq = self._skip_ws_and_read_token()
                value_tok = self._skip_ws_and_read_token()
                # fscanf "%s\n" eats all following whitespace, including blank lines.
                while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
                    self.pos += 1
                return var_name, eq, value_tok


def parse_file(path: Path):
    """Returns (values_dict, warnings, failures)."""
    text = path.read_text()
    tok = Tokenizer(text)
    values = {}
    warnings = []
    failures = []

    for expected_name, expected_type in SEQUENCE:
        var_name, eq, value_tok = tok.read_next_value()

        if eq != "=":
            failures.append(f"expected '=' after '{var_name}', got '{eq}' "
                             f"(likely a missing/extra parameter upstream)")
            break
        if var_name != expected_name:
            warnings.append(f"label '{var_name}' in file vs '{expected_name}' expected at this "
                             f"position (cosmetic only -- flowVC does not check label text)")

        if expected_type == "d":
            try:
                values[expected_name] = int(value_tok)
            except ValueError:
                failures.append(f"'{expected_name}' (file label '{var_name}') expected an int, "
                                 f"got '{value_tok}'")
                break
        elif expected_type == "f":
            try:
                values[expected_name] = float(value_tok)
            except ValueError:
                failures.append(f"'{expected_name}' (file label '{var_name}') expected a float, "
                                 f"got '{value_tok}'")
                break
        else:
            values[expected_name] = value_tok

    return values, warnings, failures


def check(results, name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    results.append(cond)


def run_cross_field_checks(v: dict, vel_dir: Path | None):
    results = []

    check(results, "Dimensions in {2,3}", v["Dimensions"] in (2, 3))
    check(results, "Data_MeshType in {0,1,2}", v["Data_MeshType"] in (0, 1, 2))
    check(results, "Data_TRes >= 2", v["Data_TRes"] >= 2)
    check(results, "Data_TDelta > 0", v["Data_TDelta"] > 0)
    check(results, "Data_TPeriodic in {0,1}", v["Data_TPeriodic"] in (0, 1))
    check(results, "Data_XPeriodic in {0,1,2}", v["Data_XPeriodic"] in (0, 1, 2))
    check(
        results,
        "Data_XPeriodic == 0 for non-CARTESIAN mesh (CheckParameters hard-errors otherwise)",
        not (v["Data_XPeriodic"] and v["Data_MeshType"] != 0)
    )
    check(results, "Data_MeshBounds.XMin < XMax", v["Data_MeshBounds.XMin"] < v["Data_MeshBounds.XMax"])
    check(results, "Data_MeshBounds.YMin < YMax", v["Data_MeshBounds.YMin"] < v["Data_MeshBounds.YMax"])
    check(results, "Data_MeshBounds.ZMin <= ZMax", v["Data_MeshBounds.ZMin"] <= v["Data_MeshBounds.ZMax"])
    check(results, "Output_TRes >= 1", v["Output_TRes"] >= 1)
    check(
        results, "Output_TDelta > TINY (or Output_TRes == 1)",
        v["Output_TDelta"] >= TINY or v["Output_TRes"] == 1
    )
    check(results, "Int_TimeStep > TINY (Int_Type != 2)", v["Int_Type"] == 2 or v["Int_TimeStep"] >= TINY)
    check(results, "Int_TimeDirection in {1,-1}", v["Int_TimeDirection"] in (1, -1))
    check(
        results, "Int_TimeDirection == -1 requires Particle_Radius <= TINY",
        not (v["Int_TimeDirection"] == -1 and v["Particle_Radius"] > TINY)
    )
    check(
        results, "FTLE_Compute and Trace_Compute are mutually exclusive",
        not (v["FTLE_Compute"] and v["Trace_Compute"])
    )

    if v["FTLE_Compute"]:
        check(results, "FTLE_CartMesh.XMin < XMax", v["FTLE_CartMesh.XMin"] < v["FTLE_CartMesh.XMax"])
        check(results, "FTLE_CartMesh.YMin < YMax", v["FTLE_CartMesh.YMin"] < v["FTLE_CartMesh.YMax"])
        check(results, "FTLE_CartMesh.ZMin <= ZMax", v["FTLE_CartMesh.ZMin"] <= v["FTLE_CartMesh.ZMax"])
        check(results, "FTLE_CartMesh.XRes >= 2", v["FTLE_CartMesh.XRes"] >= 2)
        check(results, "FTLE_CartMesh.YRes >= 2", v["FTLE_CartMesh.YRes"] >= 2)
        check(results, "FTLE_CartMesh.ZRes >= 2 (Dimensions == 3)", v["FTLE_CartMesh.ZRes"] >= 2)
        check(results, "FTLE_IntTLength > TINY", v["FTLE_IntTLength"] > TINY)
        check(
            results,
            "FTLE grid inside Data_MeshBounds (else points start already outside domain)",
            v["Data_MeshBounds.XMin"] <= v["FTLE_CartMesh.XMin"] and v["FTLE_CartMesh.XMax"] <= v["Data_MeshBounds.XMax"]
            and v["Data_MeshBounds.YMin"] <= v["FTLE_CartMesh.YMin"] and v["FTLE_CartMesh.YMax"] <= v["Data_MeshBounds.YMax"]
            and v["Data_MeshBounds.ZMin"] <= v["FTLE_CartMesh.ZMin"] and v["FTLE_CartMesh.ZMax"] <= v["Data_MeshBounds.ZMax"]
        )

    Data_TMax = v["Data_TMin"] + (v["Data_TRes"] - 1) * v["Data_TDelta"]
    print(f"\nDerived Data_TMax = {Data_TMax:.6g}")

    check(
        results, "Output_TStart within [Data_TMin, Data_TMax] (or Data_TPeriodic=1)",
        (v["Data_TMin"] <= v["Output_TStart"] <= Data_TMax) or v["Data_TPeriodic"],
        f"Output_TStart={v['Output_TStart']}, range=[{v['Data_TMin']}, {Data_TMax:.6g}]"
    )

    Output_TEnd = v["Output_TStart"] + (v["Output_TRes"] - 1) * v["Int_TimeDirection"] * v["Output_TDelta"]
    print(f"Derived Output_TEnd = {Output_TEnd:.6g}")
    check(
        results, "Output_TEnd within [Data_TMin, Data_TMax] (or Data_TPeriodic=1)",
        (v["Data_TMin"] <= Output_TEnd <= Data_TMax) or v["Data_TPeriodic"],
        f"Output_TEnd={Output_TEnd:.6g}, range=[{v['Data_TMin']}, {Data_TMax:.6g}]"
    )

    if v["FTLE_Compute"]:
        Data_TMinReq = min(v["Output_TStart"], Output_TEnd + v["Int_TimeDirection"] * v["FTLE_IntTLength"])
        Data_TMaxReq = max(v["Output_TStart"], Output_TEnd + v["Int_TimeDirection"] * v["FTLE_IntTLength"])
        print(f"Derived Data_TMinReq = {Data_TMinReq:.6g}, Data_TMaxReq = {Data_TMaxReq:.6g}")
        check(
            results,
            "Data_TMinReq >= Data_TMin (else flowVC WARNs and clamps -- not fatal, but flagged)",
            Data_TMinReq >= v["Data_TMin"] - TINY or v["Data_TPeriodic"]
        )
        check(
            results,
            "Data_TMaxReq <= Data_TMax (else flowVC WARNs and clamps -- not fatal, but flagged)",
            Data_TMaxReq <= Data_TMax + TINY or v["Data_TPeriodic"]
        )

        # Clamp FTLE time window as flowVC does when Data_TPeriodic = 0.
        if not v["Data_TPeriodic"]:
            Data_TMinReq = max(Data_TMinReq, v["Data_TMin"])
            Data_TMaxReq = min(Data_TMaxReq, Data_TMax)
            print(f"Clamped Data_TMinReq = {Data_TMinReq:.6g}, Data_TMaxReq = {Data_TMaxReq:.6g}")

        if v["Int_TimeDirection"] > 0:
            Data_FirstFrame = math.floor((Data_TMinReq - v["Data_TMin"]) / v["Data_TDelta"])
            Data_LastFrame = math.ceil((Data_TMaxReq - v["Data_TMin"]) / v["Data_TDelta"])
        else:
            Data_FirstFrame = math.ceil((Data_TMaxReq - v["Data_TMin"]) / v["Data_TDelta"])
            Data_LastFrame = math.floor((Data_TMinReq - v["Data_TMin"]) / v["Data_TDelta"])
        print(f"Derived Data_FirstFrame = {Data_FirstFrame}, Data_LastFrame = {Data_LastFrame}")

        # Frame indices must lie in [0, Data_TRes-1]. Requires patched velocity.c
        # (wraparound only when Data_TPeriodic = 1).
        if not v["Data_TPeriodic"]:
            hi = v["Data_TRes"] - 1
            check(
                results,
                f"Data_LastFrame ({Data_LastFrame}) within [0, Data_TRes-1] "
                f"([0, {hi}]) -- requires the patched velocity.c frame guard",
                0 <= Data_LastFrame <= hi,
            )
            check(
                results,
                f"Data_FirstFrame ({Data_FirstFrame}) within [0, Data_TRes-1] "
                f"([0, {hi}]) -- requires the patched velocity.c frame guard",
                0 <= Data_FirstFrame <= hi,
            )

        if vel_dir is not None:
            prefix = v["Data_InFilePrefix"]
            lo, hi = min(Data_FirstFrame, Data_LastFrame), max(Data_FirstFrame, Data_LastFrame)
            resolved, missing = [], []
            for frame in range(lo, hi + 1):
                if v["Data_TPeriodic"]:
                    mod_val = frame % (v["Data_TRes"] - 1)
                else:
                    mod_val = frame
                suffix = v["Data_SuffixTMin"] + mod_val * v["Data_SuffixTDelta"]
                fname = f"{prefix}_vel.{suffix}.bin"
                resolved.append(fname)
                if not (vel_dir / fname).exists():
                    missing.append(fname)
            print(f"\nFrame range [{lo}, {hi}] resolves to files: {resolved}")
            check(
                results, f"All {len(resolved)} resolved velocity files exist in {vel_dir}",
                len(missing) == 0, f"missing: {missing}" if missing else ""
            )

    return all(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_file", type=Path, help="Path to the flowVC .in parameter file")
    ap.add_argument("--vel-dir", type=Path, default=None,
                    help="Directory of <prefix>_vel.*.bin files, to confirm the derived frame range exists")
    args = ap.parse_args()

    values, warnings, failures = parse_file(args.in_file)

    print(f"=== Structural check: {args.in_file.name} ===")
    print(f"Parsed {len(values)} / {len(SEQUENCE)} parameters")
    if warnings:
        print(f"{len(warnings)} cosmetic label mismatch(es) (harmless, not checked by flowVC)")
    if failures:
        print("STRUCTURAL FAILURES (these WILL break parsing):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("Structural parsing OK.\n")

    print(f"=== Cross-field / derived-parameter check: {args.in_file.name} ===")
    ok = run_cross_field_checks(values, args.vel_dir)

    print()
    if ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED -- see [FAIL] lines above")
        sys.exit(1)


if __name__ == "__main__":
    main()
