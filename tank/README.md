# Improved hydrogen tank model

Start with **[Tank_model_improved.ipynb](Tank_model_improved.ipynb)**. It is an
executed, explained replacement for the experimental script in your original
`Tank_model.ipynb`. The original notebook has been overwritten with the corrected,
self-contained version in its original explanatory style. The earlier notebook and
paper are retained in the parent folder’s recovery ZIP. The reusable
solver is [tank_model.py](tank_model.py), and [run_tank.py](run_tank.py)
reproduces the worked scenarios and saved results.

## Run

From this repository's root:

```bash
export DARTS_PY="/Users/sanderbertdacosta/.local/share/python-envs/darts-py311/bin/python"
"$DARTS_PY" tank/run_tank.py
"$DARTS_PY" -m unittest tank.test_tank_model -v
```

Select the existing **darts-local** notebook kernel. Its explicit interpreter is
`/Users/sanderbertdacosta/.local/share/python-envs/darts-py311/bin/python`.
The notebook checks `sys.executable`. No new environment or package installation
was needed; the project environment and central Python registry were checked.
The solver requires NumPy/SciPy; figures use Matplotlib. The existing tested
versions were NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2, nbformat 5.11.1,
and Python 3.11.16. See the parent project's `PYTHON_ENVIRONMENT.md` and shared
environment snapshots for full reproducibility.

## What changed

The core state is hydrogen **mass**, with separate cumulative production,
external supply, leakage, and consumption. Pressure is recovered from the
current inventory using a verified normal-H2 density correlation. All input and
output names carry units. A source raises pressure instead of being discarded
at an arbitrary storage cap. Controls change at declared schedule boundaries;
time reporting does not control numerical accuracy. Invalid inputs and
out-of-domain thermodynamic states fail explicitly.

Two production options are available: a simple pressure-squared deliverability
coefficient or a radial gas well. The latter integrates real-gas density over
pressure, using explicit permeability, thickness, drainage radius, well radius,
skin, well count and prescribed viscosity. The constant-Z limit is independently
checked against the analytical radial gas-flow equation.
This option uses the steady radial resistance `ln(re/rw) + skin`; it does not
apply the minor's `-3/4` correction for a pseudo-steady mean-pressure formulation.
The tank pressure represents the drainage pressure in this approximation.

Specific original errors corrected (zero-based cells):

- Cells 3–4 copy NIST coefficients 8 and 9 incorrectly, by factors 100 and
  1000. At 50 MPa and 400 K, those cells give **Z = 1.01815361**, whereas
  the corrected equation and published check give **Z = 1.24304763**.
  At 200 MPa and 500 K, the original gives 24.386782 versus 1.74461629.
- Conflicting function definitions make cell 10 pass scalar inputs to a
  tuple-based function. The new notebook has one property implementation and
  no regression coefficients or viscosity values hidden in global state.
- Cells 15/18 use `10**8` for MPa-to-Pa conversion instead of `10**6`.
  Their pressure updates ignore Z even while displaying changing Z.
- Storage and flow volumes lack reference conditions; porosity is not used.
  The alleged mass-flow formula is dimensionally wrong and omits both molar
  mass and year-to-second conversion. Recharge clipping loses mass.
- The reaction examples have unbalanced stoichiometry and invented rates.
  They are replaced by an explicitly empirical first-order H2 loss option,
  not presented as validated reaction chemistry.
- The nearby `tank model shortened.ipynb` was also reviewed. Its radial-flow
  version uses fixed hydrostatic pressure inside the rate calculation instead
  of current reservoir pressure, applies three wells while the report specifies
  five, and estimates initial pore pressure from rock overburden. Its stated
  storage geometry is also inconsistent with a 50 m drainage radius. The new
  inputs expose these choices and use evolving reservoir pressure in the rate.

The full original notebook has no valid clean-run physical baseline. The
comparison retains its identifiable scenarios and compares its well-defined
property calculation; success is not defined as reproducing its erroneous
pressure or production curves.

## Research scenario and volume ambiguity

The reconstructed minor scenario retains the notebook's **500 bar, 600 K,
0.6 km³ initial storage, 0.235 km³/year target, and optional 0.001 km³/year
source**. It uses report assumptions of **30 m thickness, five wells,
porosity 0.2, 0.1 m well radius**, and **1 bar / 298.14 K standard conditions**.
The report's 40%-of-hydrostatic pressure estimate at 4 km gives a **160 bar
BHP**; the 500 bar initial pressure is the explicit notebook input. Lithostatic
overburden is not substituted for pore pressure.

The original storage volume is ambiguous. The primary reconstruction interprets
0.6 km³ as standard gas inventory, obtaining **2,807,186 m³ gas pore volume**
and **14,035,931 m³ bulk rock** at gas saturation 1. Five equal circular drainage
areas imply a **172.58 m radius**. The report's 50 m radius is retained as a
separate well-deliverability sensitivity at the same storage volume, so it is
not asserted to be a self-consistent alternative reservoir geometry.

A second interpretation treats 0.6 km³ as physical gas pore space; its much
larger inventory gives a very different answer. This makes the uncertainty
visible rather than assigning an unsupported geometry. The notebook and figure
show both. Permeability 10⁻¹⁵ m² and constant viscosity 2.79×10⁻⁵ Pa·s are
illustrative original inputs. Report source rates 0.0022–0.0066 kg/s are
prescribed sensitivities, not independently measured generation rates.

The additional **100 bar, 330 K** scenario shares initial geometry with DARTS:
1 km × 1 km × 30 m, porosity 0.15, gas saturation 0.8, giving **3.6 million m³**
gas pore volume. Its standard conditions are **1.01325 bar / 288.15 K**.
It uses an uncalibrated deliverability coefficient and should not generally
match a spatial DARTS well. The fixed-volume tank and mobile-water DARTS model
have different physical assumptions.

## Verification and retained outputs

**Eleven tests pass:** published NIST Z values, monotonic density and inverse
EOS, domain rejection, a closed tank, ideal-gas constant inflow/outflow,
first-order exponential loss, the producer BHP limit, ideal radial flow,
control changes, competing-sink conservation and timestep refinement.

The 100 bar example reaches **36.373 bar** after ten years and produces
**15.620 million kg** under the declared illustrative controls. The refined
pressure differs by **2.14×10⁻⁸ bar**. The largest relative mass residual is
below **6×10⁻¹⁵** in those scenarios and **1.3×10⁻¹⁴** in the reconstructed
minor scenarios. These are numerical checks, not field validation.

- [Baseline figure](output/tank_baseline.png) and [sensitivity figure](output/tank_sensitivity.png)
- [Minor-research scenario figure](output/minor_research_sensitivity.png)
- [Baseline table](output/tank_baseline.csv), [sensitivity summary](output/tank_sensitivity.csv), and [minor baseline table](output/minor_research_baseline.csv)
- [Verification record](output/verification.json) and [minor assumptions/comparison record](output/minor_research_verification.json)

## Limits and sources

Pure H2, fixed temperature, rigid rock and uniform tank pressure are assumed.
Water is immobile and gas pore volume is fixed. There is no dissolution,
diffusion, capillary flow, multiphase mobility, aquifer support, geomechanics,
heat transfer, or product chemistry. A first-order loss records hydrogen
removed from the modeled gas; it does not predict where that hydrogen goes.
Radial wells assume steady independent drainage with prescribed viscosity.
The density correlation is not a viscosity or thermal-properties model.

Sources: the user's `minor formation of hydrogen.docx`, §5 and Appendix 2;
the user's original `programming for tank model/Tank_model.ipynb`; and
[Lemmon, Huber & Leachman (2008), Eq. 3, Tables 1–2](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=832233).
The NIST correlation is restricted to 200–1000 K and at most 200 MPa. Its density
agreement does not validate reservoir volume, source rates, well properties,
or sustainable production.
