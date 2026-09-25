# Intermediate three-dimensional DARTS model

This model adapts a natural-gas depletion workflow to hydrogen. It is the second
research model, between the uniform tank and the compositional Bourakébougou model.
Open [White_hydrogen_DARTS.ipynb](White_hydrogen_DARTS.ipynb) for an explained example.
The runner and fluid/reservoir definitions are separate Python files in this folder.

## Run

From this repository's root, using the existing DARTS environment:

```bash
DARTS_PY="/Users/sanderbertdacosta/.local/share/python-envs/darts-py311/bin/python"
"$DARTS_PY" darts/run.py --suite
"$DARTS_PY" darts/run.py --config my_case.json --output my_results
"$DARTS_PY" darts/validate.py
"$DARTS_PY" -m unittest darts.test_properties darts.test_model_3d -v
```

Select `darts-local` in the notebook. The suite runs pure hydrogen, a methane comparison,
and hydrogen with a prescribed external supply of 300 kg/day. A separate config JSON
can override individual inputs. `--days 730` changes runtime. Each output folder must
have `status.json` set to `complete` before its results are used.

## Assumptions and formulation

The default native grid has 21 × 21 × 5 cells covering 1 km × 1 km × 30 m. Porosity
is 0.15, horizontal permeability 10 mD and vertical permeability 1 mD. The reservoir
mid-depth is 1000 m. Both the initial pressure of 100 bar absolute and producer BHP
of 30 bar absolute are referenced to this depth. Temperature is fixed at 330 K.
The well is completed through all five layers. Refinement preserves this physical
completion and total reservoir volume.

Initial gas pressure follows the gas hydrostatic gradient. Composition is calculated
at each depth to give water saturation 0.2. Water is initially at its immobile endpoint;
subsequent water motion is allowed. TPFA flow includes gravity, phase mobility and
rock/water compressibility. Corey relative-permeability curves are assumptions, and
outer boundaries are sealed. The reservoir geometry and rock properties are illustrative.

The two phases are pure gas and water, with no dissolution or compositional mixing.
H₂ density uses the corrected [NIST correlation](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=832233).
Hydrogen viscosity is interpolated from dilute-gas NIST values at reservoir temperature;
pressure corrections are omitted. The methane comparison uses Peng–Robinson density
and an assumed viscosity. Water properties use engineering constants. The optional
source adds a known mass of gas; an optional first-order loss removes a separately
recorded mass. These terms do not solve geological generation or reaction chemistry.

All pressures are absolute bar; other native units are metres, days, mD, cP and kmol.
Standard gas volumes in the main examples use 1.01325 bar and 288.15 K. The minor's
tank reconstruction uses a different declared standard convention, so standard volumes
must be converted before comparison. The temperature screen is 273.15–373.15 K and
BHP must be at least 1.1 bar. Supply viscosity explicitly outside the H₂ table's
300–350 K interval. The NIST density fit is not a thermal-property model.

## Output and verification

Each scenario retains accepted-step histories, 3D cell snapshots, native HDF5 output,
a summary, log, production plots, `reservoir_3d.png` and initial/final `.vtk` grids.
The middle horizontal layer is shown in `reservoir_maps.png`. The three-dimensional
figure displays cell-centred orthogonal slices with the vertical scale expanded.
The thick red well segment shows the completed interval and the star marks its BHP datum.
VTK depth coordinates increase downward.

Native material balance uses interpolated accumulation and independently reconstructed
perforation fluxes, integrated at accepted backward-Euler steps. The installed build's
legacy component-rate export omits a mobility operator, so it is not used as production.
Exact NIST/PR inventories are also exported to expose property-interpolation effects.

The tests check published density values, mixture-coordinate conversion, hydrostatic
initialization, five actual well segments, closed gravity equilibrium, sources/losses,
and production balances. `output/validation.json` records analytical ideal-gas limits
and time, property-interpolation, horizontal-grid and vertical-grid refinement. It is
numerical verification, not calibration to a field. The separate Bourakébougou model
adds gas composition, dissolution, capillary structure and a water-bearing zone.
