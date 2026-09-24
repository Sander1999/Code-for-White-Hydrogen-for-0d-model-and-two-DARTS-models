"""Run the worked tank scenarios and save figures, tables, and verification."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

if __package__:
    from .tank_model import ConstantZEOS, HydrogenEOS, OperatingStage, RadialWell, StandardConditions, TankConfig, simulate
else:
    from tank_model import ConstantZEOS, HydrogenEOS, OperatingStage, RadialWell, StandardConditions, TankConfig, simulate


def original_z_audit():
    """Compare the original notebook's cells 3–4 against the published table.

    This reproduces only its well-defined coefficient calculation; it is not
    a reference simulation. The full notebook has conflicting definitions and
    does not have a valid single clean-run baseline.
    """
    a = np.array([0.05888460, -0.06136111, -0.002650473, 0.002731125,
                  0.001802374, -0.001150707, 0.00009588528, -0.00001109040, 0.0000001264403])
    b = np.array([1.325, 1.87, 2.5, 2.8, 2.938, 3.14, 3.37, 3.75, 4.0])
    c = np.array([1, 1, 2, 2, 2.42, 2.63, 3, 4, 5])
    rows = []
    for temperature, pressure_mpa, reference in [(300, 10, 1.05985282), (400, 50, 1.24304763), (500, 200, 1.74461629)]:
        rows.append({"temperature_k": temperature, "pressure_mpa": pressure_mpa,
                     "published_z": reference,
                     "original_cells_3_4_z": float(1 + np.sum(a * (100 / temperature)**b * pressure_mpa**c)),
                     "corrected_z": HydrogenEOS().z_factor(pressure_mpa * 10, temperature)})
    return rows


def run_minor_scenarios():
    """Preserve original numbers with explicit, defensible volume assumptions.

    The original 0.6 km3 storage did not declare reference conditions. The
    primary reconstruction treats it as *initial standard gas inventory* at
    the report's 298.14 K and 1 bar; a separate sensitivity interprets it as
    gas pore volume. The resulting curves must not be called a same-input
    validation of the physically inconsistent original model.
    """
    eos = HydrogenEOS()
    standard = StandardConditions(temperature_k=298.14, pressure_bar=1)
    rho_std = float(eos.density_kg_m3(standard.pressure_bar, standard.temperature_k))
    rho_initial = float(eos.density_kg_m3(500, 600))
    gas_volume = 0.6e9 * rho_std / rho_initial
    # Preserve porosity, thickness and 5 wells; derive re from gas PV rather
    # than silently assigning a conflicting 50 m drainage volume.
    porosity, gas_saturation, thickness, wells = 0.2, 1.0, 30.0, 5
    drainage_radius = np.sqrt(gas_volume / (porosity * gas_saturation * thickness * wells * np.pi))
    radial = RadialWell(drainage_radius_m=float(drainage_radius))
    config = TankConfig(gas_pore_volume_m3=gas_volume, temperature_k=600, initial_pressure_bar=500)
    stage = OperatingStage(100 * 365.25, production_target_sm3_day=0.235e9 / 365.25,
                           bottomhole_pressure_bar=160, radial_well=radial)
    variants = {
        "No replenishment": (config, stage),
        "Notebook source 0.001 km³/year": (config, replace(stage, source_sm3_day=0.001e9 / 365.25)),
        "Report source 0.0022 kg/s": (config, replace(stage, source_sm3_day=0.0022 * 86400 / rho_std)),
        "Report source 0.0066 kg/s": (config, replace(stage, source_sm3_day=0.0066 * 86400 / rho_std)),
        "Report drainage radius 50 m": (config, replace(stage, radial_well=replace(radial, drainage_radius_m=50))),
        "0.6 km³ interpreted as gas pore volume": (replace(config, gas_pore_volume_m3=0.6e9),
            replace(stage, radial_well=replace(radial, drainage_radius_m=float(np.sqrt(0.6e9 / (porosity * thickness * wells * np.pi))))))
    }
    results = {name: simulate(cfg, [control], eos=eos, standard=standard,
                              max_step_days=90, sample_interval_days=365.25)
               for name, (cfg, control) in variants.items()}
    summary = {
        "original_z_reference_comparison": original_z_audit(),
        "interpretation": "0.6 km3 initial gas at 298.14 K and 1 bar; pure-H2 gas saturation = 1",
        "gas_pore_volume_m3": gas_volume,
        "derived_bulk_volume_m3": gas_volume / porosity,
        "derived_drainage_radius_m": float(drainage_radius),
        "porosity": porosity,
        "temperature_k": 600,
        "initial_pressure_bar": 500,
        "bottomhole_pressure_bar": 160,
        "radial_well": asdict(radial),
        "viscosity_status": "2.79e-5 Pa.s retained from original illustrative constant-viscosity model; sensitivity input, not calibrated",
        "reference_conditions": asdict(standard),
        "source_status": "Prescribed source sensitivities from user's report/notebook; not independently established generation kinetics",
        "scenarios": {label: result.summary() for label, result in results.items()},
    }
    return results, summary


def make_minor_figure(scenarios):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    volume_case = "0.6 km³ interpreted as gas pore volume"
    for label, result in scenarios.items():
        if label == volume_case:
            continue
        years = result.time_days / 365.25
        axes[0, 0].plot(years, result.pressure_bar, label=label)
        axes[0, 1].plot(years, result.cumulative_produced_kg / 1e6, label=label)
    axes[0, 0].axhline(160, color="gray", ls="--", lw=1)
    axes[0, 0].set(xlim=(0, 15), ylabel="Pressure [bar absolute]", title="First 15 years: radial-flow depletion")
    axes[0, 1].set(ylabel="Cumulative H₂ produced [10⁶ kg]", title="Prescribed-source sensitivities")
    for label, color in [("No replenishment", "#176b87"), (volume_case, "#ad5b28")]:
        result = scenarios[label]
        x = result.time_days / 365.25
        axes[1, 0].plot(x, result.pressure_bar, color=color, label=label)
        axes[1, 1].plot(x, result.cumulative_produced_kg / 1e6, color=color, label=label)
    axes[1, 0].set(ylabel="Pressure [bar absolute]", title="Ambiguous original volume: pressure")
    axes[1, 1].set(ylabel="Cumulative H₂ produced [10⁶ kg]", title="Ambiguous original volume: production")
    axes[1, 1].legend(fontsize=8, loc="upper left")
    for axis in axes.flat:
        axis.set_xlabel("Time [years]")
        axis.grid(alpha=0.2)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8)
    figure.suptitle("Reconstructed minor-research scenario • 500 bar, 600 K", fontsize=14)
    return figure


def run_examples():
    """Illustrative inputs; no field-calibrated production forecast is implied."""
    config = TankConfig()
    baseline_stage = OperatingStage(3652.5, production_target_sm3_day=100_000,
                                    label="10-year production")
    baseline = simulate(config, [baseline_stage], sample_interval_days=15)
    scenarios = {"No replenishment": baseline}
    for source in [10_000, 30_000]:
        scenarios[f"Replenishment {source:,.0f} Sm³/day"] = simulate(
            config, [replace(baseline_stage, source_sm3_day=source)], sample_interval_days=15)
    scenarios["First-order loss 0.02/year"] = simulate(
        config, [replace(baseline_stage, loss_rate_day=0.02 / 365.25)], sample_interval_days=15)
    scenarios["Ideal gas (Z = 1)"] = simulate(
        config, [baseline_stage], eos=ConstantZEOS(), sample_interval_days=15)
    scenarios["Half gas pore volume"] = simulate(
        replace(config, gas_pore_volume_m3=config.gas_pore_volume_m3 / 2),
        [baseline_stage], sample_interval_days=15)
    # Deliberate shut-in with recharge demonstrates source accounting and
    # pressure recovery, not a claim that natural H2 reservoirs regenerate.
    schedule = [replace(baseline_stage, duration_days=3 * 365.25, source_sm3_day=10_000),
                replace(baseline_stage, duration_days=365.25, production_target_sm3_day=0, source_sm3_day=10_000),
                replace(baseline_stage, duration_days=6 * 365.25, source_sm3_day=10_000)]
    scenarios["One-year shut-in + replenishment"] = simulate(config, schedule, sample_interval_days=15)

    # Independent refinement against the baseline; report data at same times.
    refined = simulate(config, [baseline_stage], sample_interval_days=15,
                       max_step_days=2, rtol=1e-11, atol_kg=1e-9)
    eos = HydrogenEOS()
    verification = {
        "config": asdict(config),
        "baseline_control": asdict(baseline_stage),
        "standard_conditions": {"pressure_bar": 1.01325, "temperature_k": 288.15},
        "assumption_status": "Illustrative homogeneous screening model; inputs are not field calibrated",
        "initial_h2_z": eos.z_factor(config.initial_pressure_bar, config.temperature_k),
        "initial_h2_density_kg_m3": float(eos.density_kg_m3(config.initial_pressure_bar, config.temperature_k)),
        "max_pressure_refinement_difference_bar": float(np.max(np.abs(baseline.pressure_bar - refined.pressure_bar))),
        "produced_mass_refinement_relative_difference": float(abs(baseline.cumulative_produced_kg[-1] - refined.cumulative_produced_kg[-1]) / refined.cumulative_produced_kg[-1]),
        "maximum_scenario_relative_mass_balance_error": max(result.relative_balance_error for result in scenarios.values()),
        "scenarios": {label: result.summary() for label, result in scenarios.items()},
    }
    if verification["max_pressure_refinement_difference_bar"] > 1e-4:
        raise AssertionError("Pressure refinement difference exceeds 0.0001 bar")
    if verification["maximum_scenario_relative_mass_balance_error"] > 1e-10:
        raise AssertionError("Mass balance did not close to the specified tolerance")
    return config, scenarios, verification


def make_figures(scenarios):
    """Create two figures; caller may display and/or save them."""
    import matplotlib.pyplot as plt

    baseline = scenarios["No replenishment"]
    years = baseline.time_days / 365.25
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
    axes[0, 0].plot(years, baseline.pressure_bar, color="#176b87", lw=2)
    axes[0, 0].axhline(30, color="#777777", ls="--", label="Producer BHP")
    axes[0, 0].set(ylabel="Pressure [bar absolute]", title="Pressure declines as gas leaves")
    axes[0, 0].legend()
    axes[0, 1].plot(years, baseline.production_sm3_day / 1000, color="#ad5b28", lw=2)
    axes[0, 1].set(ylabel="Production [10³ Sm³/day]", title="Well deliverability limits the target")
    axes[1, 0].plot(years, baseline.mass_kg / 1e6, label="Remaining", lw=2)
    axes[1, 0].plot(years, baseline.cumulative_produced_kg / 1e6, label="Produced", lw=2)
    axes[1, 0].axhline(baseline.initial_mass_kg / 1e6, color="#777777", ls="--", label="Initial inventory")
    axes[1, 0].set(ylabel="Hydrogen mass [10⁶ kg]", title="Inventory and cumulative production")
    axes[1, 0].legend()
    axes[1, 1].plot(years, baseline.balance_residual_kg, color="#31764c", lw=1.5)
    axes[1, 1].set(ylabel="Mass balance residual [kg]", title="All source and sink inventories included")
    for axis in axes.flat:
        axis.set_xlabel("Time [years]")
        axis.grid(alpha=0.2)
    figure.suptitle("Hydrogen tank • illustrative 10-year extraction", fontsize=15)

    sensitivity, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for label, result in scenarios.items():
        x = result.time_days / 365.25
        axs[0].plot(x, result.pressure_bar, label=label)
        axs[1].plot(x, result.cumulative_produced_kg / 1e6, label=label)
    axs[0].set(ylabel="Pressure [bar absolute]", title="Pressure sensitivity")
    axs[1].set(ylabel="Cumulative H₂ produced [10⁶ kg]", title="Production sensitivity")
    for axis in axs:
        axis.set_xlabel("Time [years]")
        axis.grid(alpha=0.2)
    sensitivity.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=9)
    sensitivity.suptitle("Uncalibrated scenarios • source rates are assumptions", fontsize=14)
    return figure, sensitivity


def save_results(output_dir, scenarios, verification, figures):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios["No replenishment"].to_csv(output_dir / "tank_baseline.csv")
    with (output_dir / "tank_sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario", *next(iter(scenarios.values())).summary()])
        writer.writeheader()
        for label, result in scenarios.items():
            writer.writerow({"scenario": label, **result.summary()})
    (output_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    for figure, name in zip(figures, ["tank_baseline", "tank_sensitivity"]):
        figure.savefig(output_dir / f"{name}.png", dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    arguments = parser.parse_args()
    _, scenarios, verification = run_examples()
    figures = make_figures(scenarios)
    save_results(arguments.output, scenarios, verification, figures)
    minor_results, minor_summary = run_minor_scenarios()
    minor_figure = make_minor_figure(minor_results)
    minor_figure.savefig(arguments.output / "minor_research_sensitivity.png", dpi=170)
    minor_results["No replenishment"].to_csv(arguments.output / "minor_research_baseline.csv")
    (arguments.output / "minor_research_verification.json").write_text(json.dumps(minor_summary, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
    print(f"Saved results to {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
