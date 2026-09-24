"""Conservation, analytical-limit and refinement checks for the native model.

All intermediate simulations are automatically removed. A compact JSON report
is retained next to the delivered outputs. No field-validation claim is made.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

try:
    from .run import run_case
except ImportError:
    from run import run_case


def validate(output=None):
    output = Path(output or Path(__file__).with_name('output') / 'validation.json')
    results, checks = {}, {}
    with tempfile.TemporaryDirectory(prefix='hydrogen_model_validation_', dir='/private/tmp') as scratch:
        def case(name, overrides):
            summary, rows, snapshots = run_case(overrides, Path(scratch) / name, plots=False)
            results[name] = {key: value for key, value in summary.items() if key != 'parameters'}
            results[name]['parameters'] = summary['parameters']
            return summary, rows, snapshots

        closed_base = dict(nx=3, ny=3, nz=1, producer_enabled=False, runtime_days=10.,
                           rock_compressibility_per_bar=0., water_compressibility_per_bar=0., eos='ideal')
        closed, rows, _ = case('closed_equilibrium', closed_base)
        checks['closed_pressure_constant'] = max(abs(r['pressure_mean_bar']-100) for r in rows) < 1e-6
        for name, recharge, loss in [('source', 1000., 0.), ('sink', 0., .001),
                                     ('source_and_sink', 1000., .001)]:
            summary, rows, _ = case(name, closed_base | {'recharge_kg_day': recharge, 'loss_rate_per_day': loss})
            # Independent discrete backward-Euler inventory recurrence on actual accepted steps.
            expected = rows[0]['native_gas_kg']
            errors = []
            for row in rows[1:]:
                dt = row['dt_days']
                expected = (expected + recharge*dt)/(1+loss*dt)
                errors.append(abs(row['native_gas_kg']-expected)/rows[0]['native_gas_kg'])
            error = max(errors)
            results[name]['analytical_discrete_inventory_relative_error'] = error
            checks[name+'_inventory'] = error < 1e-6
            m0 = rows[0]['native_gas_kg']
            expected_continuous = (recharge/loss+(m0-recharge/loss)*np.exp(-loss*10)
                                   if loss else m0+recharge*10)
            continuous_error = abs(rows[-1]['native_gas_kg']-expected_continuous)/m0
            results[name]['continuous_inventory_relative_error'] = continuous_error
            checks[name+'_continuous_solution'] = continuous_error < 1e-4
            # Rigid volume, incompressible immobile water, ideal gas: p/p0=M/M0.
            pressure_error = max(abs(r['pressure_mean_bar']/100-r['gas_kg']/rows[0]['gas_kg']) for r in rows)
            results[name]['ideal_pressure_inventory_ratio_error'] = pressure_error
            checks[name+'_pressure'] = pressure_error < 1e-6

        control, rows, _ = case('zero_drawdown', {'nx': 3, 'ny': 3,
                                                'producer_bhp_bar': 100., 'runtime_days': 10.})
        checks['zero_drawdown_stationary'] = (abs(control['produced_gas_kg']) < .01
                                              and abs(control['final_pressure_mean_bar']-100) < 1e-6)
        fine_time, _, _ = case('time_1day', {'max_timestep_days': 1.})
        base, _, _ = case('baseline', {})
        fine_obl, _, _ = case('interpolation_refinement', {'obl_pressure_step_bar': .125,
                                                           'obl_composition_step': .00005})
        coarse_grid, _, _ = case('grid_11', {'nx': 11, 'ny': 11})
        fine_grid, _, _ = case('grid_41', {'nx': 41, 'ny': 41})
        vertical, _, _ = case('vertical_grid_15', {'nz': 15})
        metrics = {}
        for name, other in [('time', fine_time), ('interpolation', fine_obl), ('grid', fine_grid), ('vertical_grid', vertical)]:
            metrics[name+'_production_relative_difference'] = abs(base['produced_gas_kg']-other['produced_gas_kg'])/other['produced_gas_kg']
            metrics[name+'_mean_pressure_difference_bar'] = abs(base['final_pressure_mean_bar']-other['final_pressure_mean_bar'])
        metrics['coarse_to_base_production_relative_difference'] = abs(coarse_grid['produced_gas_kg']-base['produced_gas_kg'])/base['produced_gas_kg']
        checks['time_refinement_within_1percent'] = metrics['time_production_relative_difference'] < .01
        checks['interpolation_refinement_within_0_1percent'] = metrics['interpolation_production_relative_difference'] < .001
        checks['grid_refinement_within_2percent'] = metrics['grid_production_relative_difference'] < .02
        checks['vertical_grid_within_0_1percent'] = metrics['vertical_grid_production_relative_difference'] < .001
        checks['grid_change_decreases'] = metrics['grid_production_relative_difference'] < metrics['coarse_to_base_production_relative_difference']
        methane, _, _ = case('methane_baseline', {'species': 'CH4', 'eos': 'peng_robinson'})
        checks['methane_produces_and_depletes'] = methane['produced_gas_kg'] > 0 and methane['final_pressure_mean_bar'] < 100
        results['refinement_metrics'] = metrics
    report = {'passed': all(bool(v) for v in checks.values()), 'checks': {k: bool(v) for k, v in checks.items()},
              'scope': 'Numerical verification and property/analytical limits; not field validation', 'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'checks': report['checks'], 'refinement': metrics}, indent=2))
    if not report['passed']:
        raise AssertionError(f'Validation failed; see {output}')
    return report


if __name__ == '__main__':
    validate()
