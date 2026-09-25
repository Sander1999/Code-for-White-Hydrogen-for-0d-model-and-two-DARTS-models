"""A natural-gas depletion model adapted to native hydrogen extraction.

Two immiscible phases (pure gas and liquid water), a 3D Cartesian reservoir,
TPFA finite-volume fluxes, closed outer boundaries, and one central BHP producer.
No H2 is injected. Optional recharge is a prescribed distributed geological gas
input; optional loss is a first-order removal from free-gas inventory. Neither
claims to solve serpentinization, microbiology, dissolution, or caprock transport.
"""

from __future__ import annotations

import numpy as np

from darts.engines import value_vector, well_control_iface
from darts.models.darts_model import DartsModel
from darts.nonlinear_solvers import NewtonSolver
from darts.physics.base.physics import PhysicsBase
from darts.physics.dead_oil import DeadOilProperties
from darts.physics.properties.basic import ConstFunc, RockCompactionEvaluator
from darts.physics.properties.density import DensityBasic
from darts.reservoirs.struct_reservoir import StructReservoir

try:  # Package import and direct script execution.
    from .properties import (CoreyRelativePermeability, GasDensity, MOLAR_MASS_KG_KMOL,
                             bulk_volume_m3, gas_viscosity_cp, initial_gas_mole_fraction,
                             initial_pressure_profile, inventory_from_state, load_config)
except ImportError:
    from properties import (CoreyRelativePermeability, GasDensity, MOLAR_MASS_KG_KMOL,
                            bulk_volume_m3, gas_viscosity_cp, initial_gas_mole_fraction,
                            initial_pressure_profile, inventory_from_state, load_config)


class NativeGasWaterProperties(DeadOilProperties):
    """Immiscible pure-phase flash with a dimensionally explicit native source."""
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(phases_name=["gas", "water"],
                         components_name=[cfg["species"], "H2O"],
                         Mw=np.array([MOLAR_MASS_KG_KMOL[cfg["species"]],
                                      MOLAR_MASS_KG_KMOL["H2O"]]),
                         eps_z=1e-10, rock_comp=cfg["rock_compressibility_per_bar"],
                         temperature=cfg["temperature_k"])
        self.rock_compr_ev = RockCompactionEvaluator(
            pref=cfg["initial_pressure_bar"], compres=cfg["rock_compressibility_per_bar"])
        self.density_ev = {
            "gas": GasDensity(cfg),
            "water": DensityBasic(cfg["water_density_kg_m3"],
                                  cfg["water_compressibility_per_bar"],
                                  cfg["initial_pressure_bar"]),
        }
        self.viscosity_ev = {"gas": ConstFunc(gas_viscosity_cp(cfg)),
                             "water": ConstFunc(cfg["water_viscosity_cp"])}
        swr, sgr = cfg["residual_water_saturation"], cfg["residual_gas_saturation"]
        self.rel_perm_ev = {
            "gas": CoreyRelativePermeability(sgr, swr, cfg["gas_relperm_endpoint"],
                                              cfg["gas_relperm_exponent"]),
            "water": CoreyRelativePermeability(swr, sgr, cfg["water_relperm_endpoint"],
                                                cfg["water_relperm_exponent"]),
        }
        self.output_props = {"gas_saturation": lambda: self.sat[0],
                             "water_saturation": lambda: self.sat[1],
                             "gas_density_kg_m3": lambda: self.dens[0],
                             "water_density_kg_m3": lambda: self.dens[1]}

    def evaluate_mass_source(self, pressure, temperature, zc):
        # engine_super_cpu.tpp adds bulk_volume*dt*mass_source to the residual:
        # POSITIVE removes component; NEGATIVE adds component. Units kmol/bulk-m³/day.
        # The engine only applies this term to reservoir cells, not well segments.
        gas_kmol_bulk_m3 = (self.cfg["porosity"] * self.rock_compr_ev.evaluate(pressure)
                            * self.sat[0] * self.dens_m[0])
        recharge = self.cfg["recharge_kg_day"] / self.Mw[0] / bulk_volume_m3(self.cfg)
        self.mass_source[:] = [self.cfg["loss_rate_per_day"] * gas_kmol_bulk_m3 - recharge, 0]
        return self.mass_source


class Model(DartsModel):
    """Importable model factory: Model(), Model({...}), or Model('config.json')."""
    def __init__(self, parameters=None):
        self.cfg = load_config(parameters)
        super().__init__()
        self.timer.node["initialization"].start()
        self.set_reservoir()
        self.set_physics()
        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        c = self.cfg
        self.reservoir = StructReservoir(
            self.timer, nx=c["nx"], ny=c["ny"], nz=c["nz"],
            dx=c["length_x_m"] / c["nx"], dy=c["length_y_m"] / c["ny"],
            dz=c["thickness_m"] / c["nz"], permx=c["permeability_md"],
            permy=c["permeability_md"] * c["permeability_y_ratio"],
            permz=c["permeability_md"] * c["permeability_z_ratio"],
            poro=c["porosity"], start_z=c["depth_m"] - c["thickness_m"] / 2, cache=False)

    def set_wells(self):
        if self.cfg["producer_enabled"]:
            self.reservoir.add_well("PROD", well_diameter=2 * self.cfg["well_radius_m"])
            for layer in range(1, self.cfg["nz"] + 1):
                self.reservoir.add_perforation(
                    "PROD", res_cell_idx=((self.cfg["nx"] + 1) // 2,
                                          (self.cfg["ny"] + 1) // 2, layer),
                    well_diameter=2 * self.cfg["well_radius_m"], skin=self.cfg["well_skin"],
                    ms_epm=True)
            # Native controls act at the head depth. This is a virtual BHP
            # datum at reservoir mid-depth, not a surface wellhead pressure.
            # Each actual EPM segment is aligned with its perforated layer.
            well = self.reservoir.wells[-1]
            well.well_head_depth = self.cfg["depth_m"]

    def set_physics(self):
        c = self.cfg
        self.property_container = NativeGasWaterProperties(c)
        self.physics = PhysicsBase(
            components=[c["species"], "H2O"], phases=["gas", "water"], timer=self.timer,
            state_spec=PhysicsBase.StateSpecification.P,
            axes_step=[c["obl_pressure_step_bar"], c["obl_composition_step"]],
            axes_origin=[0.01, 1e-10], epsilon_z=1e-10,
            extrapolation_flag=True, cache=False)
        self.physics.add_property_region(self.property_container)

    def set_solver(self):
        super().set_solver()
        c = self.cfg
        self.ts_control.dt_first = c["first_timestep_days"]
        self.ts_control.dt_min = 1e-10
        self.ts_control.dt_max = c["max_timestep_days"]
        self.ts_control.dt_mult = c["timestep_multiplier"]
        self.ts_control.runtime = c["runtime_days"]
        self.nonlinear_solver = NewtonSolver(tolerance=c["nonlinear_tolerance"])
        self.linear_solver.spec.tolerance = c["linear_tolerance"]

    def set_initial_conditions(self):
        mesh = self.reservoir.mesh
        pressure = initial_pressure_profile(self.cfg, np.asarray(mesh.depth)[:mesh.n_res_blocks])
        self.physics.set_initial_conditions_from_array(
            mesh=mesh,
            input_distribution={"pressure": pressure,
                                self.cfg["species"]: initial_gas_mole_fraction(self.cfg, pressure)})

    def set_well_controls(self):
        for well in self.reservoir.wells:
            self.physics.set_well_controls(wctrl=well.control,
                                          control_type=well_control_iface.BHP,
                                          is_inj=False, target=self.cfg["producer_bhp_bar"])

    def reservoir_state(self) -> np.ndarray:
        """Return independent copies of [pressure_bar, gas_mole_fraction] per cell."""
        nb = self.reservoir.mesh.n_res_blocks
        return np.asarray(self.physics.engine.X)[:2 * nb].reshape(nb, 2).copy()

    def perforation_rates(self) -> dict:
        """Native component flux across perforations, positive for production.

        The installed engine's legacy ``time_data`` component rates omit the
        separate LAMBDA mobility operator. Reconstruct the *same* TPFA stencil
        used by ``engine_super_cpu.tpp`` from native interpolated FLUX and LAMBDA
        operators, mesh transmissibilities, gravity, capillary pressure and
        phase upwinding. This is a flux calculation, independent of inventory.
        No diffusion or dispersion is enabled by this model.
        """
        mesh = self.reservoir.mesh
        states = np.asarray(self.physics.engine.X).reshape(mesh.n_blocks, 2)
        block_m, block_p = np.asarray(mesh.block_m), np.asarray(mesh.block_p)
        tran, gravity = np.asarray(mesh.tran), np.asarray(mesh.grav_coef)
        ev = self.physics.reservoir_operators[0]
        kmol_day = np.zeros(2)
        for well in self.reservoir.wells:
            for segment, reservoir_cell, _wi, _wid in well.perforations:
                well_cell = well.well_body_idx + segment
                conn = np.flatnonzero((block_m == reservoir_cell) & (block_p == well_cell))
                if conn.size != 1:
                    raise RuntimeError("Expected one native reservoir-to-well connection per perforation")
                conn = int(conn[0])
                reservoir_values = value_vector(np.zeros(ev.n_ops))
                well_values = value_vector(np.zeros(ev.n_ops))
                self.physics.acc_flux_itor[0].evaluate(value_vector(states[reservoir_cell]), reservoir_values)
                self.physics.acc_flux_w_itor.evaluate(value_vector(states[well_cell]), well_values)
                res, wel = np.asarray(reservoir_values), np.asarray(well_values)
                for phase in range(2):
                    pressure_difference = (states[well_cell, 0] - states[reservoir_cell, 0]
                        + 0.5 * (res[ev.GRAV_OP + phase] + wel[ev.GRAV_OP + phase]) * gravity[conn]
                        - wel[ev.PC_OP + phase] + res[ev.PC_OP + phase])
                    upstream = res if pressure_difference < 0 else wel
                    phase_volume_rate = (tran[conn] * pressure_difference
                                         * upstream[ev.LAMBDA_OP + phase])
                    start = ev.FLUX_OP + phase * 2
                    kmol_day -= phase_volume_rate * upstream[start:start + 2]
        return {"gas_kmol_day": float(kmol_day[0]), "water_kmol_day": float(kmol_day[1]),
                "gas_kg_day": float(kmol_day[0] * MOLAR_MASS_KG_KMOL[self.cfg["species"]]),
                "water_kg_day": float(kmol_day[1] * MOLAR_MASS_KG_KMOL["H2O"])}

    def inventory(self, interpolated=False, include_well=False) -> dict:
        """Component inventory in kg/kmol, optionally using native OBL accumulation.

        Native accumulation is the appropriate basis for numerical balance checks.
        Physical EOS inventory is more useful for reporting. Well storage, when
        requested, uses the engine's well accumulation without rock compression.
        """
        mesh = self.reservoir.mesh
        nb = mesh.n_res_blocks
        n = mesh.n_blocks if include_well else nb
        states = np.asarray(self.physics.engine.X)[:2 * n].reshape(n, 2)
        pore_volume = np.asarray(mesh.volume)[:n] * np.asarray(mesh.poro)[:n]
        if not interpolated:
            if include_well:
                raise ValueError("include_well requires interpolated=True to match native storage")
            return inventory_from_state(states[:, 0], states[:, 1], self.cfg, pore_volume)
        kmol = np.zeros(2)
        for i, state in enumerate(states):
            itor = self.physics.acc_flux_itor[0] if i < nb else self.physics.acc_flux_w_itor
            evaluator = self.physics.reservoir_operators[0] if i < nb else self.physics.well_operators
            values = value_vector(np.zeros(evaluator.n_ops))
            itor.evaluate(value_vector(state), values)
            kmol += pore_volume[i] * np.asarray(values)[evaluator.ACC_OP:evaluator.ACC_OP + 2]
        return {"gas_kmol": float(kmol[0]), "water_kmol": float(kmol[1]),
                "gas_kg": float(kmol[0] * MOLAR_MASS_KG_KMOL[self.cfg["species"]]),
                "water_kg": float(kmol[1] * MOLAR_MASS_KG_KMOL["H2O"])}
