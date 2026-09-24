"""Scientific checks of units, reference data, initialization and native source.

Run with the registered DARTS interpreter:
    python -m unittest white_hydrogen.darts.test_properties
"""
import unittest

import numpy as np

from .model import NativeGasWaterProperties
from .properties import (CoreyRelativePermeability, MOLAR_MASS_KG_KMOL,
                         GRAVITY_BAR_M_KG_M3, R_J_MOL_K,
                         bulk_volume_m3, cell_depths_m, gas_density_kg_m3, gas_z_factor,
                         initial_gas_mole_fraction, initial_inventory,
                         initial_pressure_profile, load_config, phase_state)


class GasWaterPropertyTests(unittest.TestCase):
    def test_nist_published_compressibility_points(self):
        # Lemmon et al. 2008 Table 2; pressure given there in MPa.
        self.assertAlmostEqual(gas_z_factor(100, 300), 1.05985282, places=8)
        self.assertAlmostEqual(gas_z_factor(500, 400), 1.24304763, places=8)

    def test_ideal_gas_has_correct_molar_and_mass_units(self):
        h2 = gas_density_kg_m3(1, 300, "H2", "ideal")
        methane = gas_density_kg_m3(1, 300, "CH4", "ideal")
        self.assertAlmostEqual(h2, 0.08082, places=5)
        self.assertAlmostEqual(methane / h2, 16.043 / 2.01588, places=12)

    def test_real_hydrogen_density_is_positive_and_monotone(self):
        rho = gas_density_kg_m3(np.geomspace(0.01, 2000, 100), 330)
        self.assertTrue(np.all(np.isfinite(rho)))
        self.assertTrue(np.all(np.diff(rho) > 0))
        self.assertLess(gas_density_kg_m3(100, 330), gas_density_kg_m3(100, 330, eos="ideal"))

    def test_out_of_validity_input_is_rejected(self):
        with self.assertRaises(ValueError):
            gas_z_factor(2100, 330)
        with self.assertRaises(ValueError):
            load_config({"temperature_k": 600})
        with self.assertRaises(ValueError):
            load_config({"unexpected_pressure_Pa": 1e7})
        with self.assertRaises(ValueError):
            load_config({"initial_pressure_bar": 2000})
        with self.assertRaises(ValueError):
            load_config({"producer_bhp_bar": 1.0})
        with self.assertRaises(ValueError):
            load_config({"nz": 0})
        with self.assertRaises(ValueError):
            load_config({"depth_m": 10})

    def test_saturation_is_not_mole_fraction(self):
        cfg = load_config()
        z = initial_gas_mole_fraction(cfg)
        self.assertNotAlmostEqual(z, 1 - cfg["initial_water_saturation"])
        self.assertAlmostEqual(float(phase_state(100, z, cfg)["water_saturation"]), 0.2)
        inventory = initial_inventory(cfg)
        p = initial_pressure_profile(cfg)
        expected_mass = (bulk_volume_m3(cfg) * cfg["porosity"] * 0.8 * np.mean(
            (1 + cfg["rock_compressibility_per_bar"] * (p - 100))
            * gas_density_kg_m3(p, cfg["temperature_k"])))
        self.assertAlmostEqual(inventory["gas_kg"], expected_mass, places=7)

    def test_hydrostatic_initialization_matches_ideal_gas_analytical_profile(self):
        cfg = load_config({"eos": "ideal", "nx": 2, "ny": 3, "nz": 7, "thickness_m": 300})
        depth = cell_depths_m(cfg)
        pressure = initial_pressure_profile(cfg)
        exponent = (MOLAR_MASS_KG_KMOL["H2"] / 1000 * 9.80665
                    / (R_J_MOL_K * cfg["temperature_k"]))
        expected = cfg["initial_pressure_bar"] * np.exp(exponent * (depth - cfg["depth_m"]))
        np.testing.assert_allclose(pressure, expected, rtol=2e-12, atol=1e-10)
        self.assertEqual(len(np.unique(depth)), cfg["nz"])
        self.assertEqual(depth.size, cfg["nx"] * cfg["ny"] * cfg["nz"])
        self.assertLess(pressure.min(), cfg["initial_pressure_bar"])
        self.assertGreater(pressure.max(), cfg["initial_pressure_bar"])

    def test_layer_compositions_preserve_residual_water_saturation(self):
        cfg = load_config({"nz": 9})
        p = initial_pressure_profile(cfg)
        state = phase_state(p, initial_gas_mole_fraction(cfg, p), cfg)
        np.testing.assert_allclose(state["water_saturation"], cfg["initial_water_saturation"], atol=2e-15)
        # Continuous hydrostatics also agrees with the native trapezoidal
        # gravity stencil to far below a microbar over this 30-m interval.
        rho = state["gas_density_kg_m3"].reshape(cfg["nz"], -1)[:, 0]
        layers_p = p.reshape(cfg["nz"], -1)[:, 0]
        expected_difference = .5 * (rho[1:] + rho[:-1]) * GRAVITY_BAR_M_KG_M3 * cfg["thickness_m"] / cfg["nz"]
        np.testing.assert_allclose(np.diff(layers_p), expected_difference, rtol=1e-8, atol=1e-11)

    def test_vertical_refinement_preserves_total_volume_and_initial_inventory(self):
        single = load_config({"nz": 1})
        multi = load_config({"nz": 9})
        self.assertEqual(bulk_volume_m3(single), bulk_volume_m3(multi))
        # Small physical second-order change is due to resolving gas hydrostatics.
        self.assertLess(abs(initial_inventory(single)["gas_kg"] / initial_inventory(multi)["gas_kg"] - 1), 2e-8)

    def test_phase_residual_endpoints(self):
        water = CoreyRelativePermeability(0.2, 0.05, 0.7, 3)
        gas = CoreyRelativePermeability(0.05, 0.2, 1.0, 2)
        self.assertEqual(water.evaluate(0.2), 0)
        self.assertEqual(water.evaluate(0.95), 0.7)
        self.assertEqual(gas.evaluate(0.05), 0)
        self.assertEqual(gas.evaluate(0.8), 1.0)

    def test_native_recharge_units_and_sign(self):
        cfg = load_config({"recharge_kg_day": 25})
        props = NativeGasWaterProperties(cfg)
        props.evaluate(np.array([cfg["initial_pressure_bar"], initial_gas_mole_fraction(cfg)]))
        # Native residual contains +dt*bulk_volume*source, so recharge is negative.
        self.assertAlmostEqual(-props.mass_source[0] * bulk_volume_m3(cfg)
                               * MOLAR_MASS_KG_KMOL["H2"], 25)
        self.assertEqual(props.mass_source[1], 0)

    def test_first_order_sink_uses_inventory_not_flow_area(self):
        cfg = load_config({"loss_rate_per_day": 0.001})
        props = NativeGasWaterProperties(cfg)
        props.evaluate(np.array([cfg["initial_pressure_bar"], initial_gas_mole_fraction(cfg)]))
        loss = props.mass_source[0] * bulk_volume_m3(cfg) * MOLAR_MASS_KG_KMOL["H2"]
        self.assertAlmostEqual(loss / initial_inventory(cfg)["gas_kg"], 0.001)
        self.assertEqual(props.mass_source[1], 0)


if __name__ == "__main__":
    unittest.main()
