# flowVC for SimVascular

### 1. Build flowVC

```bash
cd src
make
```

### 2. Convert the svMultiPhysics results

Makes velocity files compatible with flowvc

```bash
python scripts/convert_svmultiphysics_to_flowvc.py \
    --vtu-dir /path/to/simulation/results \
    --out-dir ~/flowvc_run/data \
    --prefix myvessel
```

### 3. Build the FTLE grid (IC file)

```bash
D=~/flowvc_run/data/myvessel
python scripts/build_anatomical_ftle_ic.py \
    --coords ${D}_coordinates.bin --conn ${D}_connectivity.bin --adj ${D}_adjacency.bin \
    --bounds XMIN XMAX YMIN YMAX ZMIN ZMAX --res NX NY NZ \
    --out ~/flowvc_run/data/myvessel_ftle_grid
```

This makes a Cartesian grid and marks which points are inside the vessel.

- `**--bounds`:** for the whole vessel, use the `Data_MeshBounds` values printed
in step 2.
- `**--res`**:** set the grid spacing

Copy the same bounds and resolution into `FTLE_MeshBounds.*` in the `.in`  
file, and set `FTLE_ICFile` to the output file's name.

### 4. Write the `.in` file

```bash
cp examples/ftle_forward.in ~/flowvc_run/myvessel_forward.in
```

### 5. Check the `.in` file

```bash
python scripts/validate_flowvc_input.py ~/flowvc_run/myvessel_forward.in --vel-dir ~/flowvc_run/data
```

This reports errors, mismatched constraints and any missing velocity files.

### 6. Run flowVC

```bash
mkdir -p ~/flowvc_run/output
```

```bash
cd ~/flowvc_run && OMP_NUM_THREADS=8 /path/to/flowvc/src/flowVC myvessel_forward.in
```

## Running out of memory

All `Output_TRes` FTLE fields are kept in memory. If they don't fit, split the
run into batches. Give each batch its own `Output_TStart`, a smaller
`Output_TRes` and its own `Path_Output`. Then put the results back together:

```bash
python scripts/merge_ftle_batches.py ~/flowvc_run/output batch0/ batch1/ batch2/ \
    --prefix forwardFTLE --t-first <first Output_TStart> --t-delta <Output_TDelta> --num-slides <total>
```

What's different from the original flowVC:

- **It is about 2x faster** on 8 threads; FTLE advection runs in parallel with OpenMP.
- **Velocity frames no longer wrap around on non-periodic data.** The original
reused frame 0 when it needed the frame after the last one. 
- **A few crashes and wrong-frame bugs are fixed.**

