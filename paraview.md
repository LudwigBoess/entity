# Plotting Entity ADIOS2 output in ParaView

## TL;DR

There is **no zero-config built-in reader** that opens Entity's ADIOS2 output as a fully
described mesh, because Entity does not embed a VTK / Fides schema in its files. The two
realistic paths with the bundled ParaView readers are:

1. **BPFile + Fides reader** (built into ParaView ≥ 5.10): write a small `data.fides.json`
   sidecar describing the rectilinear mesh and the variables, then open that JSON in
   ParaView. *Recommended for Cartesian runs.*
2. **HDF5 + XDMF sidecar**: generate a one-`.xmf`-per-dump (or a single time-series `.xmf`)
   file with a Python helper, then open the `.xmf` in ParaView. ParaView's native XDMF reader
   is the most painless route and works without any ADIOS2 plugin on the ParaView side.

For Spherical / QSpherical runs the mesh is curvilinear in physical space, so neither of
the above renders a "ready-to-look-at" picture — you also need a coordinate transform
(see *Spherical / QSpherical* below).

---

## What's actually in the output

Output is driven by [src/output/writer.cpp](src/output/writer.cpp). Per-step files land at:

```
<title>/fields/fields.<step:08d>.{bp|h5}
<title>/particles/particles.<step:08d>.{bp|h5}
<title>/spectra/spectra.<step:08d>.{bp|h5}
```

The engine is selected in TOML via `[output] format = "hdf5" | "BPFile" | "disabled"`
(default `hdf5`; see [input.example.toml:506-511](input.example.toml#L506-L511)). ADIOS2 is
v2.11.0 ([cmake/dependencies.cmake:12-14](cmake/dependencies.cmake#L12-L14)).

### Variables in a fields dump (per step)

Defined in [writer.cpp:127-204](src/output/writer.cpp#L127-L204):

- `Step` — `timestep_t` scalar
- `Time` — `simtime_t` scalar
- `X1`, `X2`, `X3` — 1-D `real_t` cell-center coordinate arrays (size `m_flds_g_shape_dwn[i]`)
- `X1e`, `X2e`, `X3e` — 1-D `real_t` cell-edge coordinate arrays (size `m_flds_g_shape_dwn[i] + 1`)
- `N1l`, `N2l`, `N3l` — per-MPI-rank `(offset, size)` bookkeeping (size `2 * #domains`)
- One `real_t` ND array per requested field, e.g. `fE1`, `fE2`, `fE3`, `fB1`, …, `fRho`,
  `fT00`, `fNppc`, `fA3`, etc. Naming is built in
  [fields.h:86-152](src/output/fields.h#L86-L152) — every field name has a leading `f`
  with the first letter of the type capitalized.

### Attributes on the IO

- `NGhosts` — int, 0 unless `output.fields.ghosts = true`
- `Dimension` — 1, 2, or 3
- `Coordinates` — string: `"cart"`, `"sph"`, or `"qsph"`
  ([enums.h:115-134](src/global/enums.h#L115-L134))
- `LayoutRight` — 1 if Kokkos `LayoutRight`, 0 otherwise. **This matters**: when 0, the
  field arrays in the file have their dimension order reversed relative to `(X1, X2, X3)`,
  and the corresponding shape/start/count vectors in the file are stored already reversed
  ([writer.cpp:153-161](src/output/writer.cpp#L153-L161)).

### Mesh topology

The mesh is **rectilinear in the simulation's logical coordinates** — `X1`, `X2`, `X3` are
each 1-D arrays. In Cartesian runs, that's also rectilinear in physical space. In Spherical
/ QSpherical, the logical axes are `(r, θ, φ)`, so the physical mesh is **curvilinear**;
you'll need a coordinate transform on top of the rectilinear (r, θ, φ) grid before
visualizing.

Particle and spectra files have a different schema (1-D arrays per quantity, no mesh).
Particle output is best handled separately as a `vtkPolyData` / point cloud (see
*Particles* below).

---

## ParaView reader options

### Option 1 — Fides reader (BPFile, ParaView's recommended path)

Fides is built into ParaView 5.10+ and reads ADIOS2 BP files via a JSON schema you write
once. It supports rectilinear meshes natively, which matches the Entity layout exactly.

**Steps:**

1. Run with `format = "BPFile"` in your TOML.
2. Write a `data.fides.json` next to the `*.bp` files (one per step, or a templated one
   that uses `%08d` step formatting — Fides supports a `data_sources` block with a
   pattern). Minimum example for a 2-D Cartesian run with `fE1`, `fE2`, `fE3`, `fB3`:

   ```json
   {
     "fides_schema_version": "1.0",
     "data_sources": [
       { "name": "source", "patterns": ["fields.*.bp"] }
     ],
     "coordinate_system": {
       "type": "rectilinear",
       "x_array": { "source": "source", "variable": "X1" },
       "y_array": { "source": "source", "variable": "X2" }
     },
     "cell_set": {
       "cell_set_type": "structured",
       "dimensions": { "source": "source", "variable_dimensions": "fE1" }
     },
     "fields": [
       { "name": "E1",  "association": "points", "source": "source", "variable": "fE1"  },
       { "name": "E2",  "association": "points", "source": "source", "variable": "fE2"  },
       { "name": "E3",  "association": "points", "source": "source", "variable": "fE3"  },
       { "name": "B3",  "association": "points", "source": "source", "variable": "fB3"  }
     ],
     "step_information": {
       "variable": "Step",
       "source": "source"
     }
   }
   ```

   Add a `z_array` entry for 3-D. **Watch the `LayoutRight` attribute**: if it's 0,
   you may have to swap the `x_array`/`y_array`/`z_array` order so that the Fides axes
   line up with the variable's stored dimension order.

3. In ParaView: `File → Open → data.fides.json`. Pick the `ADIOS2VTXReader` /
   `Fides Reader` if prompted.

**Pros:** purely built-in, handles the whole time series, plays well with `paraview-server`
on a remote system. **Cons:** requires hand-writing the schema, and the schema must be
kept in sync with whichever fields you've enabled in TOML. The `fields` array can be
generated from the TOML `output.fields.quantities` list.

### Option 2 — VTX reader / `vtk.xml` attribute (built-in, but needs a code change)

ParaView's `ADIOS2VTXReader` can open a `.bp` directly **only if the file embeds a
`vtk.xml` attribute** containing a VTK XML schema. Entity does not currently write that
attribute (see the `DefineAttribute` calls in
[writer.cpp:127-161](src/output/writer.cpp#L127-L161)). Adding it would mean emitting an
ImageData/RectilinearGrid VTK XML string at `defineMeshLayout` time. It's the most
"plug-and-play" outcome (drag `.bp` into ParaView, it just works), but it's a
non-trivial output-side change. Worth considering if you expect heavy ParaView use.

### Option 3 — XDMF sidecar (HDF5 path, simplest right now)

If you keep `format = "hdf5"`, the cleanest route is an XDMF sidecar. ParaView's native
XDMF reader handles rectilinear meshes via `<Topology TopologyType="3DRectMesh">` plus
`<Geometry GeometryType="VXVYVZ">`, which maps 1:1 to `X1`/`X2`/`X3`. A short Python
helper (using `h5py` only — no ADIOS2 dependency on the analysis side) can scan a `fields/`
directory and emit either:

- one `.xmf` per dump (point ParaView at the directory; it groups them as a time series), or
- a single `.xmf` with `<Grid GridType="Collection" CollectionType="Temporal">` referencing
  every dump — easier to load, single click in ParaView.

Per-dump XDMF skeleton (2-D Cartesian):

```xml
<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="entity" GridType="Uniform">
      <Time Value="<Time scalar from h5>"/>
      <Topology TopologyType="2DRectMesh" Dimensions="<NX2> <NX1>"/>
      <Geometry GeometryType="VXVY">
        <DataItem Format="HDF" Dimensions="<NX1>" NumberType="Float" Precision="8">
          fields.00000000.h5:/X1
        </DataItem>
        <DataItem Format="HDF" Dimensions="<NX2>" NumberType="Float" Precision="8">
          fields.00000000.h5:/X2
        </DataItem>
      </Geometry>
      <Attribute Name="E1" AttributeType="Scalar" Center="Node">
        <DataItem Format="HDF" Dimensions="<NX2> <NX1>" NumberType="Float" Precision="8">
          fields.00000000.h5:/fE1
        </DataItem>
      </Attribute>
      <!-- repeat <Attribute> for each fXxx variable -->
    </Grid>
  </Domain>
</Xdmf>
```

Notes:

- The order of `Dimensions` is *slowest-to-fastest*. With `LayoutRight = 1` and
  `(X1, X2)` defined as the row/column axes, the field array is shaped `(NX2, NX1)` in
  XDMF order — i.e. dimension reversed from how `X1`, `X2` are listed. With
  `LayoutRight = 0`, the field is already stored with the reversed axis order on disk;
  the XDMF `Dimensions` then matches `X1 X2`. Read the `LayoutRight` attribute and
  branch on it.
- Use `Precision="4"` if Entity was compiled with single-precision `real_t`.
- Cell-centered vs node-centered: the `X1`, `X2` arrays are *cell centers*. If you want
  exact cell-centered visualization, use `Center="Cell"` on the `<Attribute>` and
  switch the geometry to use `X1e`/`X2e` (edges, length `NX+1`) with
  `TopologyType="2DRectMesh" Dimensions="<NX2+1> <NX1+1>"`.

**Pros:** works with the default HDF5 engine, no schema-language learning curve, ParaView
opens `.xmf` natively. **Cons:** sidecar files to keep in sync; XDMF is also slightly
finicky about dimension order (above).

### Option 4 — Skip ParaView's ADIOS2 path entirely, write VTK files

For one-off plots or curvilinear (Spherical) meshes, a Python script that reads the BP/HDF5
dump and writes a `.vts` (structured grid) or `.vtr` (rectilinear grid) per step is often
the path of least resistance, especially because you can pre-compute the Cartesian
coordinates from `(r, θ, φ)` and ParaView shows the warped sphere immediately. Use
`pyvista`, `meshio`, or `vtk` directly. This is the same data ParaView would see via
options 1-3, just materialized into VTK's native containers.

---

## Spherical / QSpherical specifically

`X1 = r`, `X2 = θ`, `X3 = φ`. ParaView won't render a recognizable spherical wedge from a
rectilinear `(r, θ, φ)` grid on its own. Two options:

1. **Python preprocess to `vtkStructuredGrid` (.vts):** compute
   `x = r sin θ cos φ; y = r sin θ sin φ; z = r cos θ` for each node, write a structured
   grid with explicit Cartesian coordinates and the field arrays as point data. This is
   the simplest viewer experience (paraview opens the `.vts` and you see the sphere).
2. **In-ParaView transform:** open via Fides/XDMF as a rectilinear `(r, θ, φ)` grid, then
   apply `Calculator` filters to compute `(x, y, z)` and a `WarpByVector` /
   `Transform`. Doable but tedious and you'll redo it every session.

For QSpherical the only difference is that `r` (and possibly `θ`) are stretched
non-uniformly — `X1`, `X2` already contain the stretched physical values, so the same
formulas above work directly; no additional QSpherical-aware decoding is needed.

---

## Particles

Particle dumps are 1-D `real_t` arrays per quantity (`writeParticleQuantity` in
[writer.cpp:359-370](src/output/writer.cpp#L359-L370)) with no mesh. They're not rendered
by Fides/XDMF as a mesh; treat them as a point cloud. The pragmatic route is a Python
sidecar that emits a `.vtp` (PolyData) or `.csv` per dump, then load that in ParaView and
use a `Glyph` filter. Same logic as for Spherical fields: a small preprocess is cheaper
than fighting the readers.

---

## Recommendation

- **If you can switch to BPFile**: Fides reader + a generated `data.fides.json` is the
  cleanest pure-ParaView path. Write a small Python helper that reads the TOML
  `output.fields.quantities` list and emits the JSON. Cartesian only — for spherical
  add a preprocess step.
- **If you want to stay on the default HDF5 engine**: an XDMF sidecar generator is the
  least invasive option; ParaView's native XDMF reader is rock-solid.
- **For Spherical / QSpherical or particles**: just write a Python `pyvista` /
  `meshio` script that emits `.vts` / `.vtp`. The readers all expect Cartesian
  coordinates, so a 30-line preprocess saves more time than wiring up filters in
  ParaView every session.

If long-term you want **drag-and-drop opens of the `.bp` directly**, the right fix is to
add a `vtk.xml` attribute in `Writer::defineMeshLayout` so ADIOS2VTXReader recognizes the
mesh without any sidecar. That's an output-side change; the schemas above are pure
post-processing.
