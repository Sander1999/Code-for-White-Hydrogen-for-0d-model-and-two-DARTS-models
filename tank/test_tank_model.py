"""Physical reference tests; run with the project DARTS Python interpreter."""

from dataclasses import replace
import unittest

import numpy as np

try:
    from .tank_model import ConstantZEOS, HydrogenEOS, OperatingStage, RadialWell, StandardConditions, TankConfig, simulate
except ImportError:
    from tank_model import ConstantZEOS, HydrogenEOS, OperatingStage, RadialWell, StandardConditions, TankConfig, simulate


class EquationOfStateTests(unittest.TestCase):
    def test_nist_published_reference_values(self):
        # Lemmon et al. (2008), Table 2. Values are independent of this code.
        eos = HydrogenEOS()
        for temperature, pressure_mpa, expected in [
            (200, 1, 1.00675450), (300, 10, 1.05985282),
            (400, 50, 1.24304763), (500, 200, 1.74461629),
            (200, 200, 2.85953449),
        ]:
            self.assertAlmostEqual(eos.z_factor(10 * pressure_mpa, temperature), expected, delta=5e-9)

    def test_inverse_and_positive_compressibility(self):
        eos = HydrogenEOS()
        pressures = np.geomspace(0.01, 2000, 100)
        for temperature in [200, 330, 1000]:
            density = eos.density_kg_m3(pressures, temperature)
            self.assertTrue(np.all(np.diff(density) > 0))
            recovered = np.array([eos.pressure_bar(float(rho), temperature) for rho in density])
            np.testing.assert_allclose(recovered, pressures, rtol=1e-8, atol=1e-9)

    def test_domain_rejected(self):
        for pressure, temperature in [(-1, 330), (2001, 330), (10, 199), (10, 1001), (np.nan, 330)]:
            with self.assertRaises(ValueError):
                HydrogenEOS().density_kg_m3(pressure, temperature)
        with self.assertRaises(ValueError):
            HydrogenEOS().pressure_bar(1e9, 330)


class MassBalanceTests(unittest.TestCase):
    def setUp(self):
        self.config = TankConfig(gas_pore_volume_m3=12_000, initial_pressure_bar=100, temperature_k=330)
        self.ideal = ConstantZEOS()

    def test_radial_flow_matches_ideal_gas_closed_form(self):
        well = RadialWell()
        p, pw, temperature = 500, 160, 600
        # Integral rho dp = MW/(2 R T) (p_res² - p_well²), p in Pa.
        expected = (well.well_count * np.pi * well.permeability_m2 * well.thickness_m
                    * self.ideal.molar_mass_kg_mol * ((p * 1e5)**2 - (pw * 1e5)**2)
                    / (well.viscosity_pa_s * (np.log(well.drainage_radius_m / well.well_radius_m) + well.skin)
                       * 8.314472 * temperature) * 86400)
        self.assertAlmostEqual(well.mass_rate_kg_day(p, pw, temperature, self.ideal) / expected, 1, delta=1e-13)
        self.assertEqual(well.mass_rate_kg_day(10, 20, temperature, self.ideal), 0)

    def test_closed_tank_stays_at_initial_state(self):
        for eos in [self.ideal, HydrogenEOS()]:
            result = simulate(self.config, [OperatingStage(1000)], eos=eos)
            np.testing.assert_allclose(result.pressure_bar, 100, atol=1e-9)
            self.assertLess(result.relative_balance_error, 1e-12)

    def test_constant_source_exact_ideal_gas_pressure(self):
        stage = OperatingStage(120, source_sm3_day=100)
        result = simulate(self.config, [stage], eos=self.ideal)
        std = StandardConditions()
        expected_dp_day = 100 * std.pressure_bar * self.config.temperature_k / (std.temperature_k * self.config.gas_pore_volume_m3)
        np.testing.assert_allclose(result.pressure_bar, 100 + expected_dp_day * result.time_days, rtol=1e-12)

    def test_constant_withdrawal_exact_ideal_material_balance(self):
        # Large finite deliverability keeps the target active over this test.
        stage = OperatingStage(120, production_target_sm3_day=100, deliverability_sm3_day_bar2=100)
        result = simulate(self.config, [stage], eos=self.ideal)
        std = StandardConditions()
        expected_dp_day = 100 * std.pressure_bar * self.config.temperature_k / (std.temperature_k * self.config.gas_pore_volume_m3)
        np.testing.assert_allclose(result.pressure_bar, 100 - expected_dp_day * result.time_days, rtol=1e-12)
        np.testing.assert_allclose(result.production_sm3_day, 100)

    def test_first_order_loss_exact_exponential(self):
        stage = OperatingStage(1000, loss_rate_day=0.005)
        result = simulate(self.config, [stage], eos=self.ideal, max_step_days=7)
        np.testing.assert_allclose(result.mass_kg, result.initial_mass_kg * np.exp(-0.005 * result.time_days), rtol=1e-9)
        np.testing.assert_allclose(result.pressure_bar, 100 * np.exp(-0.005 * result.time_days), rtol=1e-9)
        self.assertGreater(result.mass_kg[-1], 0)

    def test_producer_cannot_extract_below_bottomhole_pressure(self):
        stages = [OperatingStage(10_000, production_target_sm3_day=1e6, deliverability_sm3_day_bar2=0.5)]
        result = simulate(self.config, stages, eos=self.ideal, sample_interval_days=20, max_step_days=5)
        self.assertGreaterEqual(result.pressure_bar.min(), 30 - 1e-8)
        self.assertAlmostEqual(result.pressure_bar[-1], 30, delta=1e-6)
        self.assertLess(result.relative_balance_error, 1e-12)

    def test_piecewise_controls_and_competing_sinks(self):
        stage = OperatingStage(100, production_target_sm3_day=1000, source_sm3_day=250,
                               leakage_sm3_day_bar=0.2, loss_rate_day=0.0001)
        stages = [stage, replace(stage, duration_days=75, production_target_sm3_day=0),
                  replace(stage, duration_days=125, production_target_sm3_day=500)]
        coarse = simulate(self.config, stages, max_step_days=10, sample_interval_days=5)
        fine = simulate(self.config, stages, max_step_days=2, sample_interval_days=5, rtol=1e-11, atol_kg=1e-9)
        np.testing.assert_allclose(coarse.mass_kg, fine.mass_kg, rtol=2e-8)
        self.assertEqual(len(np.unique(coarse.time_days)), len(coarse.time_days))
        self.assertEqual(coarse.time_days[-1], 300)
        self.assertEqual(coarse.production_sm3_day[np.flatnonzero(coarse.time_days == 100)[0]], 0)
        self.assertGreater(coarse.cumulative_source_kg[-1], 0)
        self.assertGreater(coarse.cumulative_leaked_kg[-1], 0)
        self.assertGreater(coarse.cumulative_consumed_kg[-1], 0)
        self.assertLess(coarse.relative_balance_error, 1e-12)

    def test_invalid_controls_fail_early(self):
        with self.assertRaises(ValueError):
            TankConfig(gas_pore_volume_m3=0)
        with self.assertRaises(ValueError):
            OperatingStage(1, source_sm3_day=-1)
        with self.assertRaises(ValueError):
            simulate(self.config, [])
        with self.assertRaises(ValueError):
            simulate(self.config, [OperatingStage(1)], max_step_days=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
