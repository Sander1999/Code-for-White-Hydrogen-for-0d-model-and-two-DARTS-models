"""Pure property/volume calculations for the immiscible gas–water model.

DARTS units are bar, day, m, mD, cP, kg/m³, kg/kmol and kmol.  This module
has no DARTS dependency. All default reservoir inputs are illustrative.

H2: Lemmon, Huber & Leachman (2008), Eq. 3, Tables 1–2,
https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=832233 . This density
correlation is not a thermal/caloric EOS. CH4: standard Peng–Robinson vapour
root, using constants from open-DARTS' DARTS-flash unit-test component data.
The methane case is a pure-CH4 engineering baseline, not a gas mixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp

R_J_MOL_K = 8.314472  # Value used in Lemmon et al. Table 1.
GRAVITY_BAR_M_KG_M3 = 9.80665e-5  # Native DARTS conn_mesh gravity constant.
MOLAR_MASS_KG_KMOL = {"H2": 2.01588, "CH4": 16.043, "H2O": 18.01528}
_A = np.array([0.05888460, -0.06136111, -0.002650473, 0.002731125,
               0.001802374, -0.001150707, 0.9588528e-4, -0.1109040e-6,
               0.1264403e-9])
_B = np.array([1.325, 1.87, 2.5, 2.8, 2.938, 3.14, 3.37, 3.75, 4.0])
_C = np.array([1.0, 1.0, 2.0, 2.0, 2.42, 2.63, 3.0, 4.0, 5.0])


def load_config(parameters: Mapping | str | Path | None = None) -> dict:
    """Read the shipped flat JSON defaults and apply a checked partial override."""
    cfg = json.loads(Path(__file__).with_name("config.json").read_text())
    if isinstance(parameters, (str, Path)):
        parameters = json.loads(Path(parameters).read_text())
    if parameters is not None:
        unknown = set(parameters) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown DARTS configuration keys: {sorted(unknown)}")
        cfg.update(parameters)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping) -> None:
    """Reject units/ranges that cannot represent the layered 3D screening model."""
    if cfg["species"] not in ("H2", "CH4"):
        raise ValueError("species must be H2 or CH4; no gas-mixture flash is implemented")
    expected = {"H2": ("auto", "nist", "ideal"), "CH4": ("auto", "peng_robinson", "ideal")}
    if cfg["eos"] not in expected[cfg["species"]]:
        raise ValueError(f"Unsupported EOS for {cfg['species']}: {cfg['eos']}")
    for key, value in cfg.items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
    for key in ("nx", "ny", "nz"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("length_x_m", "length_y_m", "thickness_m", "permeability_md",
                "permeability_y_ratio", "permeability_z_ratio", "initial_pressure_bar",
                "producer_bhp_bar", "well_radius_m", "water_density_kg_m3",
                "water_viscosity_cp", "gas_relperm_exponent", "water_relperm_exponent",
                "runtime_days", "first_timestep_days", "max_timestep_days",
                "nonlinear_tolerance", "linear_tolerance", "obl_pressure_step_bar",
                "obl_composition_step"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not isinstance(cfg["producer_enabled"], bool):
        raise ValueError("producer_enabled must be true or false")
    if cfg["depth_m"] <= cfg["thickness_m"] / 2:
        raise ValueError("depth_m is the mid-depth; the reservoir top must have positive depth")
    if cfg["producer_bhp_bar"] > cfg["initial_pressure_bar"]:
        raise ValueError("Producer BHP cannot exceed initial reservoir pressure")
    if cfg["producer_bhp_bar"] <= cfg["obl_pressure_step_bar"]:
        raise ValueError("Use a positive pressure interpolation grid below the producer BHP")
    if cfg["producer_bhp_bar"] < 1.1:
        raise ValueError("Keep BHP >= 1.1 bar to retain liquid water throughout the supported temperature range")
    if not 0 < cfg["porosity"] < 1:
        raise ValueError("porosity must lie strictly between 0 and 1")
    if not 0 < cfg["initial_water_saturation"] < 1:
        raise ValueError("This two-phase model requires 0 < initial_water_saturation < 1")
    swr, sgr = cfg["residual_water_saturation"], cfg["residual_gas_saturation"]
    if min(swr, sgr) < 0 or swr + sgr >= 1:
        raise ValueError("Residual saturations must be nonnegative and sum to less than one")
    for key in ("gas_relperm_endpoint", "water_relperm_endpoint"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for key in ("recharge_kg_day", "loss_rate_per_day", "rock_compressibility_per_bar",
                "water_compressibility_per_bar"):
        if cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if cfg["gas_viscosity_cp"] is not None and cfg["gas_viscosity_cp"] <= 0:
        raise ValueError("gas_viscosity_cp must be positive or null")
    if cfg["timestep_multiplier"] <= 1:
        raise ValueError("timestep_multiplier must exceed one")
    if cfg["first_timestep_days"] > cfg["max_timestep_days"]:
        raise ValueError("first_timestep_days must not exceed max_timestep_days")
    # Water is represented by fixed-temperature engineering constants, not a phase flash.
    if not 273.15 <= cfg["temperature_k"] <= 373.15:
        raise ValueError("This liquid-water screening model supports 273.15–373.15 K")
    if cfg["species"] == "H2":
        maximum_pressure = float(np.max(initial_pressure_profile(cfg)))
        upper_node = .01 + np.ceil((maximum_pressure-.01) /
                                   cfg["obl_pressure_step_bar"])*cfg["obl_pressure_step_bar"]
        if upper_node > 2000:
            raise ValueError("All initial layer pressures and their upper interpolation nodes must stay within the NIST 2000-bar limit")
    if cfg["rock_compressibility_per_bar"] * cfg["initial_pressure_bar"] >= 1:
        raise ValueError("Rock pore-volume multiplier would become nonpositive on depletion")
    if cfg["water_compressibility_per_bar"] * cfg["initial_pressure_bar"] >= 1:
        raise ValueError("Water density would become nonpositive on depletion")


def _result(value):
    return float(value) if np.ndim(value) == 0 else value


def gas_z_factor(pressure_bar, temperature_k: float, species="H2", eos="auto"):
    """Dimensionless gas Z. H2 uses the pressure-explicit NIST density fit."""
    p = np.asarray(pressure_bar, dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p <= 0) or not np.isfinite(temperature_k):
        raise ValueError("Pressure and temperature must be finite, with pressure positive")
    if temperature_k <= 0 or species not in ("H2", "CH4"):
        raise ValueError("Temperature must be positive and species H2 or CH4")
    if eos == "auto":
        eos = "nist" if species == "H2" else "peng_robinson"
    if eos == "ideal":
        z = np.ones_like(p)
    elif eos == "nist" and species == "H2":
        if not 200 <= temperature_k <= 1000 or np.any(p > 2000):
            raise ValueError("NIST H2 density correlation requires 200–1000 K and p <= 2000 bar")
        p_mpa = p[..., None] / 10.0
        z = 1.0 + np.sum(_A * (100.0 / temperature_k) ** _B * p_mpa ** _C, axis=-1)
    elif eos == "peng_robinson" and species == "CH4":
        # Standard PR alpha, cubic Z polynomial; largest real root is vapour.
        # Constants agree with DARTS-flash tests/cpp/unit/test_eos.cpp.
        tc, pc, omega = 190.564, 45.992, 0.01141
        if temperature_k <= tc:
            raise ValueError("The CH4 baseline must be above its critical temperature")
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + kappa * (1 - np.sqrt(temperature_k / tc))) ** 2
        aa = 0.45724 * alpha * (p / pc) * (tc / temperature_k) ** 2
        bb = 0.07780 * (p / pc) * tc / temperature_k
        z = np.empty(p.size)
        for i, (a, b) in enumerate(zip(aa.ravel(), bb.ravel())):
            roots = np.roots([1, b - 1, a - 3 * b**2 - 2 * b,
                              -a * b + b**2 + b**3])
            real = roots.real[np.abs(roots.imag) < 1e-9]
            z[i] = max(real[real > b])
        z = z.reshape(p.shape)
    else:
        raise ValueError(f"Unsupported gas/EOS pair: {species}/{eos}")
    return _result(z)


def gas_density_kg_m3(pressure_bar, temperature_k: float, species="H2", eos="auto"):
    """Mass density from p/(ZRT); MW in kg/kmol is converted to kg/mol."""
    z = gas_z_factor(pressure_bar, temperature_k, species, eos)
    rho = np.asarray(pressure_bar) * 1e5 * MOLAR_MASS_KG_KMOL[species] / (
        1000.0 * z * R_J_MOL_K * temperature_k)
    return _result(rho)


def gas_viscosity_cp(cfg: Mapping) -> float:
    """Approximate constant viscosity at reservoir T, or an explicit user value.

    H2 low-density values are linearly interpolated from NIST's gas table;
    no high-density correction is claimed. CH4 uses a 0.012 cP screening value.
    https://www.nist.gov/pml/sensor-science/fluid-metrology/database-thermophysical-properties-gases-used-semiconductor-11
    """
    if cfg["gas_viscosity_cp"] is not None:
        return float(cfg["gas_viscosity_cp"])
    if cfg["species"] == "CH4":
        return 0.012
    temperature = cfg["temperature_k"]
    if not 300 <= temperature <= 350:
        raise ValueError("Specify gas_viscosity_cp explicitly for H2 outside 300–350 K")
    return float(np.interp(temperature, [300, 310, 320, 350],
                           [0.00895, 0.00915, 0.00935, 0.00994]))


def water_density_kg_m3(pressure_bar, cfg: Mapping):
    """Linear liquid density relative to the configured initial pressure."""
    rho = cfg["water_density_kg_m3"] * (1 + cfg["water_compressibility_per_bar"] * (
        np.asarray(pressure_bar) - cfg["initial_pressure_bar"]))
    if np.any(rho <= 0):
        raise ValueError("Water density is nonpositive outside the model's range")
    return _result(rho)


def bulk_volume_m3(cfg: Mapping) -> float:
    return cfg["length_x_m"] * cfg["length_y_m"] * cfg["thickness_m"]


def cell_depths_m(cfg: Mapping) -> np.ndarray:
    """Positive-down cell centres in native KJI order (x index changes fastest).

    ``depth_m`` is the mid-reservoir datum, shared by initial pressure and BHP.
    Increasing nz refines the same physical thickness; it never enlarges storage.
    """
    dz = cfg["thickness_m"] / cfg["nz"]
    layers = cfg["depth_m"] - cfg["thickness_m"] / 2 + (np.arange(cfg["nz"]) + 0.5) * dz
    return np.repeat(layers, cfg["nx"] * cfg["ny"])


def initial_pressure_profile(cfg: Mapping, depths_m=None) -> np.ndarray:
    """Isothermal gas hydrostatics, dp/dz=rho_g(p,T)*g, with p in bar.

    The initial gas pressure is specified at ``depth_m``. With zero capillary
    pressure, water shares that pressure and is not separately hydrostatic.
    The default Sw equals residual Sw, so water is immobile initially. A user
    choosing mobile initial water deliberately starts gravity redistribution.
    """
    depths = cell_depths_m(cfg) if depths_m is None else np.asarray(depths_m, dtype=float)
    if np.any(~np.isfinite(depths)):
        raise ValueError("Hydrostatic depths must be finite")
    offsets = depths - cfg["depth_m"]
    result = np.full(offsets.shape, cfg["initial_pressure_bar"], dtype=float)
    for sign in (-1, 1):
        selected = offsets * sign > 0
        if not np.any(selected):
            continue
        end = float(np.max(offsets[selected]) if sign == 1 else np.min(offsets[selected]))
        solution = solve_ivp(
            lambda _depth, p: [GRAVITY_BAR_M_KG_M3 * gas_density_kg_m3(
                p[0], cfg["temperature_k"], cfg["species"], cfg["eos"])],
            (0, end), [cfg["initial_pressure_bar"]], method="DOP853",
            rtol=1e-12, atol=1e-12, dense_output=True,
        )
        if not solution.success:
            raise RuntimeError("Gas hydrostatic initialization failed")
        result[selected] = solution.sol(offsets[selected])[0]
    return result


def initial_gas_mole_fraction(cfg: Mapping, pressure_bar=None):
    """Convert volume saturation to mole fraction at the local pressure.

    The optional pressure array preserves the same initial water saturation
    through the vertically varying density field. Omission retains the datum
    value for property-only callers.
    """
    p = cfg["initial_pressure_bar"] if pressure_bar is None else pressure_bar
    sg = 1 - cfg["initial_water_saturation"]
    rg = gas_density_kg_m3(p, cfg["temperature_k"],
                         cfg["species"], cfg["eos"]) / MOLAR_MASS_KG_KMOL[cfg["species"]]
    rw = water_density_kg_m3(p, cfg) / MOLAR_MASS_KG_KMOL["H2O"]
    return sg * rg / (sg * rg + (1 - sg) * rw)


def phase_state(pressure_bar, gas_mole_fraction, cfg: Mapping) -> dict:
    """Physical phase saturation, density and rock multiplier for each state."""
    p, z = np.broadcast_arrays(np.asarray(pressure_bar), np.asarray(gas_mole_fraction))
    if np.any((z < 0) | (z > 1)):
        raise ValueError("Gas mole fraction must lie between zero and one")
    rho_g = np.asarray(gas_density_kg_m3(p, cfg["temperature_k"], cfg["species"], cfg["eos"]))
    rho_w = np.asarray(water_density_kg_m3(p, cfg))
    v_g = z / (rho_g / MOLAR_MASS_KG_KMOL[cfg["species"]])
    v_w = (1 - z) / (rho_w / MOLAR_MASS_KG_KMOL["H2O"])
    sg = v_g / (v_g + v_w)
    rock = 1 + cfg["rock_compressibility_per_bar"] * (p - cfg["initial_pressure_bar"])
    return {"gas_saturation": sg, "water_saturation": 1 - sg,
            "gas_density_kg_m3": rho_g, "water_density_kg_m3": rho_w,
            "rock_multiplier": rock}


def inventory_from_state(pressure_bar, gas_mole_fraction, cfg: Mapping,
                         reference_pore_volume_m3=None) -> dict:
    """Sum component inventory over equally sized cells or supplied pore volumes.

    Default total reference pore volume is the configured domain volume times
    porosity. DARTS interpolated-operator inventory may differ slightly from this
    exact-property inventory; use Model.inventory(interpolated=True) for balance.
    """
    phase = phase_state(pressure_bar, gas_mole_fraction, cfg)
    n = np.size(phase["gas_saturation"])
    if reference_pore_volume_m3 is None:
        reference_pore_volume_m3 = bulk_volume_m3(cfg) * cfg["porosity"] / n
    pv = np.asarray(reference_pore_volume_m3) * phase["rock_multiplier"]
    gas = float(np.sum(pv * phase["gas_saturation"] * phase["gas_density_kg_m3"]))
    water = float(np.sum(pv * phase["water_saturation"] * phase["water_density_kg_m3"]))
    return {"gas_kg": gas, "gas_kmol": gas / MOLAR_MASS_KG_KMOL[cfg["species"]],
            "water_kg": water, "water_kmol": water / MOLAR_MASS_KG_KMOL["H2O"],
            "pore_volume_m3": float(np.sum(np.broadcast_to(pv, np.shape(phase["gas_saturation"]))))}


def initial_inventory(cfg: Mapping) -> dict:
    p = initial_pressure_profile(cfg)
    return inventory_from_state(p, initial_gas_mole_fraction(cfg, p), cfg)


class GasDensity:
    """DARTS density evaluator, returning kg/m³."""
    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, pressure, temperature=None, x=None):
        return gas_density_kg_m3(pressure, self.cfg["temperature_k"],
                               self.cfg["species"], self.cfg["eos"])


class CoreyRelativePermeability:
    """Explicit phase residuals avoid ambiguity in generic phase-name dispatch."""
    def __init__(self, residual_self, residual_other, endpoint, exponent):
        self.residual_self = residual_self
        self.residual_other = residual_other
        self.endpoint = endpoint
        self.exponent = exponent

    def evaluate(self, saturation):
        effective = np.clip((saturation - self.residual_self) /
                            (1 - self.residual_self - self.residual_other), 0, 1)
        return self.endpoint * effective ** self.exponent
