# Bourakébougou compositional hydrogen model

Open [Bourakebougou_DARTS.ipynb](Bourakebougou_DARTS.ipynb), or run the scripts below.
This is the third research model and the second DARTS implementation. Its separate
folder contains the model, fluid properties, configuration, scientific evidence,
verification and 3D results.

## Running and changing the case

From this repository's root, using the existing DARTS environment:

```bash
export DARTS_PY="/Users/sanderbertdacosta/.local/share/python-envs/darts-py311/bin/python"
"$DARTS_PY" Bourakebougou/run.py
"$DARTS_PY" Bourakebougou/run.py --suite
"$DARTS_PY" Bourakebougou/run.py --config my_case.json --output my_results
"$DARTS_PY" Bourakebougou/validate.py
"$DARTS_PY" Bourakebougou/plot_properties.py
"$DARTS_PY" -m unittest Bourakebougou.test_properties -v
```

The notebook reads completed outputs by default and checks that their parameters
match its inputs. If the baseline is absent it calculates it automatically.
Set `rerun = True` to recalculate after changing inputs. The optional suite and
refinement sections explain how to generate their outputs if they are absent.
Use the existing `darts-local` kernel. `run.py --days 730` changes the run length.
Do not infer success from an old figure: each output folder has a completion status.

The base grid is 9 × 9 × 9 over an assumed 1 km × 1 km × 45 m domain, with top depth
100 m and contact depth 125 m. The assumed gas pressure is 6 bar absolute at 110 m,
temperature 303.15 K, porosity 0.10 and horizontal permeability 10 mD. Vertical
permeability is one tenth of horizontal permeability. A central well completes
105–110 m and operates at 3 bar absolute referenced to 107.5 m. These engineering
inputs are not presented as measured Bourakébougou values. The published approximate
dry-gas proportions are 0.98 H₂, 0.01 N₂ and 0.01 CH₄.

## Physics

The native finite-volume model conserves four components in two mobile phases. A
compositional flash predicts gas–water exchange; PR gas-mixture density and IAPWS
water properties provide phase densities and viscosities. Gas viscosity uses a
dilute-mixture approximation. Relative permeability and capillary parameters are
assumptions. Hydrostatic initialization, gravity, a capillary transition and a
resolved water-bearing zone distinguish this model from the intermediate model.
The fixed physical completion is preserved when the vertical grid changes.

The temperature range is restricted to 290–350 K and accepted cell pressures to
1.1–100 bar. The base case is fresh water. Unvalidated salt parameters and unsupported
H₂ caloric data are not used. Molecular diffusion is optional, conservative and
zero by default; the grid does not resolve metre-scale diffusive boundary layers.
Hysteresis, explicit karst conduits, heat flow and reaction kinetics are omitted.
[SCIENCE.md](SCIENCE.md) explains these choices and primary references.

The installed flash requires explicit aqueous evaluators for H₂. Its gas phase uses
the stable-root label with a hybrid aqueous preference: the maximum-root label can
drop a gas phase after an auxiliary critical-point search fails on an off-solution
Newton trial. A regression test retains that trial state and requires strict phase
and component closure. Native critical-point diagnostics may appear in logs even
when the flash converges. The runner rejects failed flashes and material balances.

## Scenarios and output

The suite compares closed depletion, prescribed water influx of 100,000 kg/day,
prescribed H₂ supply of 100 kg/day, and a lower gas-relative-permeability endpoint
of 0.22. Inputs occur in cells below the initial contact. Water influx and H₂ supply
are independent sensitivities, not a geological generation law or calibrated aquifer.
The lower endpoint is informed by a carbonate laboratory analogue; other curve
parameters remain assumptions. The archived fitted laboratory curve and attribution
are in `reference_data/`.

Each case exports `history.csv`, `spatial_snapshots.csv`, `summary.json`, native HDF5,
a native log, `production.png`, `vertical_sections.png`, `reservoir_3d.png`, and
initial/final VTK volumes. VTK coordinates use metres with depth positive downward.
The 3D figures expand the vertical scale and display cell-centred orthogonal slices.

The runner records all accepted timesteps and integrates independent perforation
fluxes using the same backward-Euler time convention as DARTS. `validation.json`
records component conservation, known inputs, closed-reservoir relaxation and
refinement differences. `properties_validation.json` contains independent Henry-law
and NIST property comparisons. The distinction between gas-phase H₂, dissolved H₂
and cumulative production is preserved in all outputs.
