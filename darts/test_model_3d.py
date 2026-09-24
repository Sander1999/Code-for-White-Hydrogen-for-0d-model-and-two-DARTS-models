"""Native 3D gravity and multi-perforation material-balance regression tests.

All generated native output is removed with the temporary directory. These
checks verify discretization and controls, not real-reservoir calibration.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from darts.engines import redirect_darts_output, set_num_threads

from .model import Model
from .properties import cell_depths_m
from .run import run_case


class NativeThreeDimensionalTests(unittest.TestCase):
    def test_closed_gas_hydrostatics_remains_at_rest(self):
        with tempfile.TemporaryDirectory(prefix="hydrogen_3d_test_", dir="/private/tmp") as tmp:
            summary, records, snapshots = run_case(
                dict(nx=3, ny=3, nz=5, producer_enabled=False, runtime_days=10),
                Path(tmp) / "closed", plots=False)
            initial, final = snapshots[0][1], snapshots[-1][1]
            self.assertEqual(initial.shape, (45, 2))
            self.assertGreater(np.ptp(initial[:, 0]), 0.01)
            self.assertLess(np.max(np.abs(final[:, 0] - initial[:, 0])), 1e-7)
            self.assertLess(summary["max_relative_gas_balance_error"], 1e-10)
            self.assertLess(summary["max_relative_water_balance_error"], 1e-10)
            self.assertEqual(records[-1]["produced_kg"], 0)

    def test_every_layer_has_a_native_well_segment_and_shared_bhp_datum(self):
        with tempfile.TemporaryDirectory(prefix="hydrogen_3d_test_", dir="/private/tmp") as tmp:
            redirect_darts_output(str(Path(tmp) / "native.log"))
            set_num_threads(2)
            model = Model(dict(nx=3, ny=3, nz=5))
            model.init()
            mesh, well = model.reservoir.mesh, model.reservoir.wells[0]
            self.assertEqual(mesh.n_res_blocks, 45)
            np.testing.assert_allclose(np.asarray(mesh.depth)[:45], cell_depths_m(model.cfg))
            self.assertEqual(len(well.perforations), 5)
            self.assertEqual(len({int(perf[0]) for perf in well.perforations}), 5)
            self.assertEqual(np.asarray(mesh.depth)[well.well_head_idx], model.cfg["depth_m"])
            for segment, reservoir_cell, _wi, _wid in well.perforations:
                self.assertEqual(np.asarray(mesh.depth)[well.well_body_idx + segment],
                                 np.asarray(mesh.depth)[reservoir_cell])

    def test_multilayer_production_and_source_sink_balance(self):
        with tempfile.TemporaryDirectory(prefix="hydrogen_3d_test_", dir="/private/tmp") as tmp:
            summary, records, snapshots = run_case(
                dict(nx=3, ny=3, nz=5, runtime_days=10,
                     recharge_kg_day=200, loss_rate_per_day=0.0001),
                Path(tmp) / "producing", plots=False)
            self.assertGreater(summary["produced_gas_kg"], 0)
            self.assertLess(summary["final_pressure_mean_bar"], 100)
            self.assertAlmostEqual(summary["recharged_gas_kg"], 2000, places=7)
            self.assertGreater(summary["lost_gas_kg"], 0)
            self.assertLess(summary["max_relative_gas_balance_error"], 1e-7)
            self.assertLess(summary["max_relative_water_balance_error"], 1e-7)
            self.assertLess(max(abs(r["bhp_bar"] - 30) for r in records[1:]), 1e-5)
            self.assertEqual(snapshots[-1][1].shape, (45, 2))


if __name__ == "__main__":
    unittest.main()
