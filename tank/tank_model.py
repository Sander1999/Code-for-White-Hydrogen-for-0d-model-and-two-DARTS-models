"""Conservative, isothermal hydrogen tank screening model.

The state is gas mass, not pressure. Every source and sink has a cumulative
inventory, so a gas-law change cannot create or destroy hydrogen. Public
pressures are absolute bar; time is days; masses are kg; temperatures are K.
Standard m3 (Sm3) refer explicitly to ``StandardConditions``.

The NIST correlation is for normal-H2 density only (not enthalpy or mixtures).
Lemmon, Huber & Leachman (2008), J. Res. NIST 113, 341–350, Eq. 3/Table 1.
https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=832233
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence
import csv

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

R_GAS = 8.314472  # J/(mol K), as used in the published NIST correlation
H2_MOLAR_MASS = 0.00201588  # kg/mol
BAR_TO_PA = 1.0e5


def _positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}; got {value!r}")


class GasEOS(Protocol):
    """The density and inverse-density interface needed by the tank."""

    def density_kg_m3(self, pressure_bar: float, temperature_k: float) -> float: ...

    def pressure_bar(self, density_kg_m3: float, temperature_k: float) -> float: ...


@dataclass(frozen=True)
class HydrogenEOS:
    """Normal-H2 pressure-based density correlation, 200–1000 K, 0–2000 bar.

    Vectorized ``z_factor`` and ``density_kg_m3`` also accept NumPy arrays.
    Inversion is bracketed over the validated pressure range; out-of-domain
    states fail explicitly rather than being extrapolated or clipped.
    """

    def z_factor(self, pressure_bar, temperature_k):
        p, t = np.broadcast_arrays(
            np.asarray(pressure_bar, dtype=float), np.asarray(temperature_k, dtype=float)
        )
        if not (np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 2000))):
            raise ValueError("H2 density correlation requires 0 <= pressure_bar <= 2000")
        if not (np.all(np.isfinite(t)) and np.all((t >= 200) & (t <= 1000))):
            raise ValueError("H2 density correlation requires 200 <= temperature_k <= 1000")
        a = np.array([0.05888460, -0.06136111, -0.002650473, 0.002731125,
                      0.001802374, -0.001150707, 0.9588528e-4,
                      -0.1109040e-6, 0.1264403e-9])
        b = np.array([1.325, 1.87, 2.5, 2.8, 2.938, 3.14, 3.37, 3.75, 4.0])
        c = np.array([1.0, 1.0, 2.0, 2.0, 2.42, 2.63, 3.0, 4.0, 5.0])
        z = 1 + np.sum(a * (100 / t[..., None]) ** b * (p[..., None] / 10) ** c, axis=-1)
        return z.item() if z.ndim == 0 else z

    def density_kg_m3(self, pressure_bar, temperature_k):
        return (np.asarray(pressure_bar) * BAR_TO_PA * H2_MOLAR_MASS
                / (self.z_factor(pressure_bar, temperature_k) * R_GAS * np.asarray(temperature_k)))

    def pressure_bar(self, density_kg_m3: float, temperature_k: float) -> float:
        _positive("density_kg_m3", density_kg_m3, allow_zero=True)
        rho_max = float(self.density_kg_m3(2000, temperature_k))
        if density_kg_m3 > rho_max:
            raise ValueError("Hydrogen inventory exceeds the 2000-bar density correlation limit")
        if density_kg_m3 == 0:
            return 0.0
        return brentq(lambda p: self.density_kg_m3(p, temperature_k) - density_kg_m3,
                      0, 2000, xtol=1e-10, rtol=1e-13)


@dataclass(frozen=True)
class ConstantZEOS:
    """Explicit constant-Z approximation; Z=1 is the ideal gas law."""

    z: float = 1.0
    molar_mass_kg_mol: float = H2_MOLAR_MASS

    def __post_init__(self):
        _positive("z", self.z)
        _positive("molar_mass_kg_mol", self.molar_mass_kg_mol)

    def density_kg_m3(self, pressure_bar: float, temperature_k: float) -> float:
        _positive("pressure_bar", pressure_bar, allow_zero=True)
        _positive("temperature_k", temperature_k)
        return pressure_bar * BAR_TO_PA * self.molar_mass_kg_mol / (self.z * R_GAS * temperature_k)

    def pressure_bar(self, density_kg_m3: float, temperature_k: float) -> float:
        _positive("density_kg_m3", density_kg_m3, allow_zero=True)
        _positive("temperature_k", temperature_k)
        return density_kg_m3 * self.z * R_GAS * temperature_k / (BAR_TO_PA * self.molar_mass_kg_mol)


@dataclass(frozen=True)
class StandardConditions:
    """Sm3 is defined here as 15 °C and 1.01325 absolute bar by default."""

    temperature_k: float = 288.15
    pressure_bar: float = 1.01325

    def __post_init__(self):
        _positive("standard temperature_k", self.temperature_k)
        _positive("standard pressure_bar", self.pressure_bar)


@dataclass(frozen=True)
class TankConfig:
    """Gas-filled pore volume is bulk volume × porosity × gas saturation.

    A constant gas volume assumes immobile water and rigid rock. The model
    does not predict saturation movement, water production, or aquifer influx.
    """

    gas_pore_volume_m3: float = 3_600_000.0
    temperature_k: float = 330.0
    initial_pressure_bar: float = 100.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            _positive(key, value)


@dataclass(frozen=True)
class RadialWell:
    """Steady radial single-gas Darcy flow with constant specified viscosity.

    q_mass = N 2πkh/[mu (ln(re/rw)+skin)] integral(rho(p) dp).
    Integrating density includes changing real-gas Z and converts all pressure
    differences to Pa. Independent wells and uniform gas mobility are assumed;
    multiphase flow and transient drainage require the spatial DARTS model.
    """

    permeability_m2: float = 1e-15
    thickness_m: float = 30.0
    drainage_radius_m: float = 50.0
    well_radius_m: float = 0.1
    viscosity_pa_s: float = 2.79e-5
    well_count: int = 5
    skin: float = 0.0

    def __post_init__(self):
        for key in ["permeability_m2", "thickness_m", "drainage_radius_m",
                    "well_radius_m", "viscosity_pa_s", "well_count"]:
            _positive(key, getattr(self, key))
        if not isinstance(self.well_count, int):
            raise ValueError("well_count must be an integer")
        if self.drainage_radius_m <= self.well_radius_m:
            raise ValueError("drainage_radius_m must exceed well_radius_m")
        if not np.isfinite(self.skin) or np.log(self.drainage_radius_m / self.well_radius_m) + self.skin <= 0:
            raise ValueError("Radial well logarithmic resistance must be positive")

    def mass_rate_kg_day(self, pressure_bar, bottomhole_pressure_bar, temperature_k, eos):
        if pressure_bar <= bottomhole_pressure_bar:
            return 0.0
        half_range = (pressure_bar - bottomhole_pressure_bar) / 2
        p = half_range * _QUADRATURE_NODES + (pressure_bar + bottomhole_pressure_bar) / 2
        # Scalar calls retain compatibility with any GasEOS implementation.
        density = np.array([eos.density_kg_m3(float(value), temperature_k) for value in p])
        integrated_density = half_range * BAR_TO_PA * np.dot(_QUADRATURE_WEIGHTS, density)
        resistance = self.viscosity_pa_s * (np.log(self.drainage_radius_m / self.well_radius_m) + self.skin)
        return float(self.well_count * 2 * np.pi * self.permeability_m2 * self.thickness_m
                     / resistance * integrated_density * 86400)


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(12)


@dataclass(frozen=True)
class OperatingStage:
    """One constant-control interval.

    Production is min(target, C max(p² - p_bhp², 0)). C is a screening
    deliverability coefficient [Sm3/day/bar²], not a measured permeability.
    This pressure-squared law is approximate for real gas. Finite C makes
    extraction vanish continuously at BHP; no inventory or pressure clipping
    is used. A source is externally supplied H2, not predicted geochemistry.

    Leakage is L max(p - p_threshold, 0), with L [Sm3/day/bar]. It is an
    optional empirical sensitivity, not a capillary/multiphase seal model.
    """

    duration_days: float
    production_target_sm3_day: float = 0.0
    bottomhole_pressure_bar: float = 30.0
    deliverability_sm3_day_bar2: float = 25.0
    source_sm3_day: float = 0.0
    leakage_sm3_day_bar: float = 0.0
    leakage_threshold_bar: float = 1.01325
    loss_rate_day: float = 0.0
    radial_well: RadialWell | None = None
    label: str = "stage"

    def __post_init__(self):
        _positive("duration_days", self.duration_days)
        _positive("bottomhole_pressure_bar", self.bottomhole_pressure_bar)
        _positive("leakage_threshold_bar", self.leakage_threshold_bar)
        for key in ["production_target_sm3_day", "deliverability_sm3_day_bar2",
                    "source_sm3_day", "leakage_sm3_day_bar", "loss_rate_day"]:
            _positive(key, getattr(self, key), allow_zero=True)


@dataclass
class TankResult:
    """Output arrays share a time axis, including all control changes."""

    time_days: np.ndarray
    pressure_bar: np.ndarray
    mass_kg: np.ndarray
    cumulative_source_kg: np.ndarray
    cumulative_produced_kg: np.ndarray
    cumulative_leaked_kg: np.ndarray
    cumulative_consumed_kg: np.ndarray
    production_sm3_day: np.ndarray
    source_sm3_day: np.ndarray
    leakage_sm3_day: np.ndarray
    consumption_kg_day: np.ndarray
    stage_index: np.ndarray
    initial_mass_kg: float
    standard_density_kg_m3: float
    rhs_evaluations: int

    @property
    def balance_residual_kg(self) -> np.ndarray:
        return (self.mass_kg - self.initial_mass_kg - self.cumulative_source_kg
                + self.cumulative_produced_kg + self.cumulative_leaked_kg
                + self.cumulative_consumed_kg)

    @property
    def relative_balance_error(self) -> float:
        scale = max(self.initial_mass_kg + float(self.cumulative_source_kg[-1]), 1.0)
        return float(np.max(np.abs(self.balance_residual_kg)) / scale)

    def summary(self) -> dict:
        """Plain numbers suitable for JSON and notebook display."""
        return {
            "duration_days": float(self.time_days[-1]),
            "initial_mass_kg": float(self.initial_mass_kg),
            "final_mass_kg": float(self.mass_kg[-1]),
            "final_pressure_bar": float(self.pressure_bar[-1]),
            "produced_kg": float(self.cumulative_produced_kg[-1]),
            "source_kg": float(self.cumulative_source_kg[-1]),
            "leaked_kg": float(self.cumulative_leaked_kg[-1]),
            "consumed_kg": float(self.cumulative_consumed_kg[-1]),
            "produced_fraction_of_initial_inventory": float(self.cumulative_produced_kg[-1] / self.initial_mass_kg),
            "max_balance_residual_kg": float(np.max(np.abs(self.balance_residual_kg))),
            "relative_balance_error": self.relative_balance_error,
            "rhs_evaluations": self.rhs_evaluations,
        }

    def to_csv(self, path: str | Path) -> None:
        """Write unit-labelled numerical results; parent directory must exist."""
        arrays = {key: value for key, value in vars(self).items() if isinstance(value, np.ndarray)}
        arrays["balance_residual_kg"] = self.balance_residual_kg
        with Path(path).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(arrays)
            writer.writerows(zip(*arrays.values()))


def simulate(
    config: TankConfig,
    stages: Sequence[OperatingStage],
    *,
    eos: GasEOS | None = None,
    standard: StandardConditions | None = None,
    sample_interval_days: float = 5.0,
    max_step_days: float = 10.0,
    rtol: float = 1e-9,
    atol_kg: float = 1e-7,
) -> TankResult:
    """Integrate mass balance with adaptive DOP853 and bracketed EOS inversion.

    Each schedule stage is solved separately so the solver never steps across
    an undisclosed control discontinuity. Display interval is independent of
    integration step. Output rates at a stage boundary use the *new* control;
    cumulative quantities remain continuous. Source and all sinks are tracked.
    """
    if not stages:
        raise ValueError("At least one operating stage is required")
    for name, value in [("sample_interval_days", sample_interval_days),
                        ("max_step_days", max_step_days), ("rtol", rtol), ("atol_kg", atol_kg)]:
        _positive(name, value)
    eos = HydrogenEOS() if eos is None else eos
    standard = StandardConditions() if standard is None else standard
    rho_standard = float(eos.density_kg_m3(standard.pressure_bar, standard.temperature_k))
    mass_initial = float(eos.density_kg_m3(config.initial_pressure_bar, config.temperature_k)) * config.gas_pore_volume_m3
    state = np.array([mass_initial, 0.0, 0.0, 0.0, 0.0])
    all_time, all_state, all_stage = [], [], []
    start, evaluations = 0.0, 0

    def rates(pressure, control):
        if control.radial_well is None:
            capacity = control.deliverability_sm3_day_bar2 * max(pressure**2 - control.bottomhole_pressure_bar**2, 0)
        else:
            capacity = control.radial_well.mass_rate_kg_day(pressure, control.bottomhole_pressure_bar,
                                                           config.temperature_k, eos) / rho_standard
        production = min(control.production_target_sm3_day, capacity)
        leakage = control.leakage_sm3_day_bar * max(pressure - control.leakage_threshold_bar, 0)
        return np.array([control.source_sm3_day, production, leakage])

    for index, stage in enumerate(stages):
        end = start + stage.duration_days

        def rhs(_time, values):
            # Negative trial states can occur inside an adaptive solver stage;
            # use zero pressure only for that trial. Accepted states are checked
            # below and are never corrected by clipping.
            pressure = eos.pressure_bar(max(float(values[0]), 0.0) / config.gas_pore_volume_m3,
                                        config.temperature_k)
            source, produced, leaked = rho_standard * rates(pressure, stage)
            consumed = stage.loss_rate_day * max(float(values[0]), 0.0)
            return [source - produced - leaked - consumed, source, produced, leaked, consumed]

        count = max(1, int(np.ceil(stage.duration_days / sample_interval_days)))
        times = np.linspace(start, end, count + 1)
        solution = solve_ivp(rhs, (start, end), state, method="DOP853", t_eval=times,
                             rtol=rtol, atol=atol_kg, max_step=max_step_days)
        if not solution.success:
            raise RuntimeError(f"Tank integration failed in stage {index}: {solution.message}")
        if np.any(~np.isfinite(solution.y)) or np.any(solution.y[0] < 0):
            raise RuntimeError("Tank integration produced an invalid inventory; reduce maximum step")
        # Keep the last endpoint only for the final stage, avoiding duplicated
        # boundary times and giving the next stage ownership of boundary rates.
        selected = slice(None) if index == len(stages) - 1 else slice(None, -1)
        all_time.append(solution.t[selected])
        all_state.append(solution.y[:, selected])
        all_stage.append(np.full(len(solution.t[selected]), index, dtype=int))
        state = solution.y[:, -1]
        start = end
        evaluations += solution.nfev

    time = np.concatenate(all_time)
    values = np.concatenate(all_state, axis=1)
    indices = np.concatenate(all_stage)
    pressure = np.array([eos.pressure_bar(float(m) / config.gas_pore_volume_m3, config.temperature_k)
                         for m in values[0]])
    actual_rates = np.array([rates(p, stages[i]) for p, i in zip(pressure, indices)])
    consumed_rates = np.array([stages[i].loss_rate_day * m for m, i in zip(values[0], indices)])
    result = TankResult(time, pressure, *values, actual_rates[:, 1], actual_rates[:, 0],
                        actual_rates[:, 2], consumed_rates, indices, mass_initial, rho_standard, evaluations)
    if result.relative_balance_error > max(1e-8, 10 * rtol):
        raise RuntimeError("Tank calculation failed its mass-conservation check")
    return result
