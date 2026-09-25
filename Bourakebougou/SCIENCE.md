# Scientific basis for the Bourakébougou model

This third model represents an assumed shallow accumulation informed by Bourakébougou,
Mali. It extends the intermediate pure-hydrogen DARTS model to gas mixtures, hydrogen
dissolved in water, gas–water exchange and vertical segregation. Numerical verification
and comparison with reference properties establish how the implementation behaves.
They do not determine the field's connected volume, reserves or sustainable production.

## What the field evidence constrains

Maiga et al. (2023) describe a karstified dolomitic carbonate accumulation beneath
sealing dolerite. Their approximate shallow dry-gas composition, 98% H₂, 1% N₂ and
1% CH₄, sets the model's initial dry-gas proportions. The shallow interval in Bougou-19
is reported at 110–150 m. Carbonate plug porosities range from 0.21% to 14.32%, but
plug measurements do not characterize the connected karst void space. The model's
area, effective porosity, permeability, contact depth, temperature and well controls
are therefore declared assumptions. The published pressure and flow descriptions
lack the reference conditions and production time series required for history matching.
The reported increase from 4.5 to 5 bar is not entered as measured downhole absolute
pressure. A machine-readable record is in `field_evidence.json`.

Source: [Maiga et al. (2023)](https://doi.org/10.1038/s41598-023-38977-y).
Earlier field report: [Prinzhofer et al. (2018)](https://doi.org/10.1016/j.ijhydene.2018.08.193).

## Phase equilibrium and fluid properties

The component conservation equations track H₂, N₂, CH₄ and H₂O separately. At each
pressure, temperature and overall composition, a two-phase flash partitions the
components between gas and aqueous phases. Hydrogen may dissolve or exsolve; this
transfers hydrogen between phases without creating or destroying hydrogen mass.
The DARTSflash aqueous implementation is selected explicitly with the Ziabakhsh
solute and water models. Its H₂ extension is checked independently against dilute
hydrogen solubility from the IAPWS Henry correlation. The base publication is
[Ziabakhsh-Ganji and Kooi (2012)](https://doi.org/10.1016/j.ijggc.2012.07.025);
the installed source, rather than this citation alone, defines the implemented
H₂ coefficients.

Gas-mixture density is evaluated from the gas root of the Peng–Robinson EOS using
the actual wet-gas composition. The pure-H₂ limit is compared with the independent
[NIST density correlation](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=832233).
Liquid density and viscosity use IAPWS water properties, neglecting the small density
increment of dilute dissolved gases. This is a freshwater case: unverified H₂ salting
coefficients in the installed library are not used to claim a brine prediction.
Gas viscosity is a dilute-gas approximation appropriate for the low-pressure scenario;
mixture and pressure corrections must be reconsidered for a substantially different case.

Independent equilibrium reference: [IAPWS Henry guideline](https://iapws.org/technical-guidance/release/HenGuide).
Low-temperature experimental reference: [Wiesenburg and Guinasso (1979)](https://doi.org/10.1021/je60083a006).
For more saline or high-pressure scenarios, measured solubilities and a separately
validated activity/fugacity model are needed, for example
[the 2024 water/NaCl measurements](https://doi.org/10.1016/j.ijhydene.2023.10.290).

The calculation is isothermal. The installed cubic-EOS database does not supply valid
H₂ ideal heat-capacity coefficients, so enabling a thermal solve would add unsupported
physics. Density accuracy alone cannot validate enthalpy or energy conservation.

## Flow and unresolved carbonate properties

Darcy flow includes phase mobility, gravity and capillary pressure in three dimensions.
Initial gas and water pressures depend on depth. Gas saturation follows the assumed
capillary relation above the gas–water contact; water below the contact is initially
slightly undersaturated with dissolved gas. A pressure-controlled producer removes
components through the native well connection. Closed external boundaries represent
a sealed compartment, not an independently calculated dolerite leakage threshold.

Corey mobility and Brooks–Corey capillary parameters are effective-continuum assumptions.
They are not measured Bourakébougou functions. The archived carbonate drainage fit
from [Rezaei et al. (2022)](https://doi.org/10.1029/2022GL099433) is retained as a
comparison dataset, with its [CC BY 4.0 provenance](https://doi.org/10.6084/m9.figshare.19722520.v1).
Its test used 206.8 bar, 353.15 K and 35 ppt brine, so direct transfer to shallow
karst would be unjustified. The retained data are fitted curves, not independent raw
measurements. A lower gas-mobility endpoint scenario tests sensitivity without claiming
to reproduce the full experiment. Hysteresis and separate fracture/matrix flow are
not resolved; laboratory imbibition data and connected-karst characterization would
be needed to add those mechanisms credibly.

Molecular diffusion, if enabled, must redistribute components conservatively. Its
coefficient is an effective pore-fluid diffusivity in m²/s converted to m²/day; the
native operator separately accounts for porosity, phase saturation and molar density.
Experiments place dissolved-H₂ diffusion near several ×10⁻⁹ m²/s around room temperature:
[Wang et al. (2023)](https://doi.org/10.1021/acs.jced.3c00085).
At 4×10⁻⁹ m²/s the free-fluid length √(2Dt) is only about 1.6 m over ten years,
before tortuosity. A coarse reservoir grid cannot resolve this thin diffusion scale.
The main scenarios therefore do not treat diffusion as a large-scale recharge mechanism.

## Supply and interpretation

Hydrogen supply and water influx are independent prescribed sensitivity inputs.
Positive water influx can support pressure without adding H₂. Prescribed H₂ supply
adds exactly the recorded component mass; it is not a fitted serpentinization rate.
The model does not resolve deeper source rocks, microbial reactions or geological
recharge pathways. The closed case is the reference, followed by separately labelled
water-influx and hydrogen-supply cases. A similar pressure response would not identify
which mechanism operates in the field.

Before a field forecast, obtain pressure measurements with a depth and absolute/gauge
reference, metered production and uptime, fluid temperature and salinity, well tests,
connected pore-volume information, gas/water samples over time, and capillary and
relative-permeability measurements representative of the matrix and connected karst.
