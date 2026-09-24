"""Run the native DARTS extraction model and export interpretable results.

From the reservoir-simulation workspace: ./run_darts.sh white_hydrogen/darts/run.py
Use --methane for the same geometry and well with a pure-CH4 gas baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
from darts.engines import set_num_threads, redirect_darts_output

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from white_hydrogen.visualize_3d import geometry, export_vtk, plot_reservoir_3d

try:
    from .model import Model
    from .properties import MOLAR_MASS_KG_KMOL, gas_density_kg_m3, phase_state
except ImportError:
    from model import Model
    from properties import MOLAR_MASS_KG_KMOL, gas_density_kg_m3, phase_state


class RecordedModel(Model):
    """Record each accepted timestep for backward-Euler material balances."""
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.records = []

    def record(self):
        time = float(self.physics.engine.t)
        state = self.reservoir_state()
        if not np.isfinite(state).all() or np.any(state[:, 0] < 1.1):
            raise RuntimeError('Accepted state left the finite liquid-water screening range (p >= 1.1 bar)')
        physical = self.inventory()
        native = self.inventory(interpolated=True)
        phase = phase_state(state[:, 0], state[:, 1], self.cfg)
        data = self.physics.engine.time_data
        gas_rate = water_rate = bhp = 0.0
        if self.reservoir.wells and time > 0:
            # Use reservoir perforation rates: wellhead rates also include well storage.
            # Legacy engine.time_data component rates omit the separate mobility
            # operator in this installed version. Reconstruct native TPFA fluxes.
            rates = self.perforation_rates()
            gas_rate = rates['gas_kmol_day']
            water_rate = rates['water_kmol_day']
            bhp = float(data['PROD : BHP (bar)'][-1])
        mw = MOLAR_MASS_KG_KMOL[self.cfg['species']]
        previous = self.records[-1] if self.records else None
        dt = time - previous['time_days'] if previous else 0.0
        row = {
            'time_days': time,
            'dt_days': dt,
            'pressure_mean_bar': float(state[:, 0].mean()),
            'pressure_min_bar': float(state[:, 0].min()),
            'pressure_max_bar': float(state[:, 0].max()),
            'gas_saturation_min': float(np.min(phase['gas_saturation'])),
            'gas_saturation_max': float(np.max(phase['gas_saturation'])),
            'gas_kg': physical['gas_kg'],
            'native_gas_kg': native['gas_kg'],
            'water_kg': physical['water_kg'],
            'native_water_kg': native['water_kg'],
            'production_kg_day': gas_rate * mw,
            'water_production_kg_day': water_rate * MOLAR_MASS_KG_KMOL['H2O'],
            'bhp_bar': bhp if time > 0 else self.cfg['initial_pressure_bar'],
        }
        for name, rate in [('produced_kg', row['production_kg_day']),
                           ('water_produced_kg', row['water_production_kg_day']),
                           ('recharged_kg', self.cfg['recharge_kg_day']),
                           ('lost_kg', self.cfg['loss_rate_per_day'] * native['gas_kg'])]:
            row[name] = (previous[name] if previous else 0.0) + dt * rate
        first = self.records[0] if self.records else row
        row['gas_balance_residual_kg'] = (native['gas_kg'] - first['native_gas_kg']
                                         + row['produced_kg'] - row['recharged_kg'] + row['lost_kg'])
        row['water_balance_residual_kg'] = (native['water_kg'] - first['native_water_kg']
                                           + row['water_produced_kg'])
        row['physical_gas_balance_residual_kg'] = (physical['gas_kg'] - first['gas_kg']
                                                  + row['produced_kg'] - row['recharged_kg'] + row['lost_kg'])
        self.records.append(row)

    def after_converged_timestep(self):
        super().after_converged_timestep()
        self.record()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(parameters=None, output=None, report_days=None, *, plots=True):
    """Run a case with an explicit completion marker, including failed reruns."""
    output = Path(output or Path(__file__).with_name('output')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'status.json'
    marker.write_text(json.dumps({'status': 'running'})+'\n')
    try:
        result = _run_case(parameters, output, report_days, plots=plots)
    except Exception as error:
        marker.write_text(json.dumps({'status': 'failed', 'error': str(error),
                                      'notice': 'Outputs in this folder may be incomplete or from a previous run.'}, indent=2)+'\n')
        raise
    marker.write_text(json.dumps({'status': 'complete', 'days': result[0]['days'],
                                  'species': result[0]['parameters']['species']}, indent=2)+'\n')
    return result


def _run_case(parameters, output, report_days, *, plots):
    """Return (summary, accepted-step records, snapshots); write reusable outputs."""
    redirect_darts_output(str(output / 'native.log'))
    set_num_threads(2)
    model = RecordedModel(parameters)
    model.init()
    model.set_output(output_folder=str(output))
    model.record()
    end = model.cfg['runtime_days']
    if report_days is None:
        report_days = sorted(set([min(x, end) for x in [30., 90., 180., end]]))
    times = np.asarray(report_days, dtype=float)
    if (times.size == 0 or not np.isfinite(times).all() or times[0] <= 0
            or np.any(np.diff(times) <= 0) or not np.isclose(times[-1], end)):
        raise ValueError('report_days must increase from above zero and end at runtime_days')
    snapshots = [(0.0, model.reservoir_state())]
    for day in times:
        model.run(float(day - model.physics.engine.t), verbose=0,
                  save_well_data=bool(model.reservoir.wells),
                  save_well_data_after_run=bool(model.reservoir.wells))
        if not np.isclose(model.physics.engine.t, day, rtol=0, atol=1e-8):
            raise RuntimeError(f'DARTS did not reach day {day}')
        state = model.reservoir_state()
        if not np.isfinite(state).all():
            raise RuntimeError('Non-finite DARTS state')
        snapshots.append((float(day), state))
    records = model.records
    first, final = records[0], records[-1]
    scale = first['native_gas_kg'] + final['recharged_kg']
    balance = max(abs(r['gas_balance_residual_kg']) for r in records) / scale
    water_balance = max(abs(r['water_balance_residual_kg']) for r in records) / first['native_water_kg']
    physical_balance = max(abs(r['physical_gas_balance_residual_kg']) for r in records) / scale
    rho_std = gas_density_kg_m3(1.01325, 288.15, model.cfg['species'], model.cfg['eos'])
    for row in records:
        row['production_sm3_day'] = row['production_kg_day'] / rho_std
        row['produced_sm3'] = row['produced_kg'] / rho_std
    summary = {
        'open_darts_version': version('open-darts'),
        'model_scope': 'Illustrative 3D isothermal immiscible pure-gas/water depletion; no field calibration',
        'standard_conditions': {'pressure_bar_absolute': 1.01325, 'temperature_k': 288.15},
        'parameters': model.cfg,
        'reservoir_cells': int(model.reservoir.mesh.n_res_blocks),
        'accepted_timesteps': len(records) - 1,
        'days': final['time_days'],
        'initial_gas_kg': first['gas_kg'],
        'remaining_gas_kg': final['gas_kg'],
        'produced_gas_kg': final['produced_kg'],
        'produced_standard_m3': final['produced_sm3'],
        'recharged_gas_kg': final['recharged_kg'],
        'lost_gas_kg': final['lost_kg'],
        'produced_fraction_of_initial_inventory': final['produced_kg'] / first['gas_kg'],
        'final_pressure_mean_bar': final['pressure_mean_bar'],
        'final_pressure_range_bar': [final['pressure_min_bar'], final['pressure_max_bar']],
        'final_production_kg_day': final['production_kg_day'],
        'max_relative_gas_balance_error': balance,
        'max_relative_water_balance_error': water_balance,
        'max_relative_physical_gas_balance_error': physical_balance,
        'finite_states': bool(all(np.isfinite(list(r.values())).all() for r in records)),
        'balance_method': 'Native interpolated accumulation and reservoir perforation rates; accepted-step right-endpoint integration',
    }
    if not summary['finite_states'] or balance > 2e-5 or water_balance > 2e-5:
        raise RuntimeError(f'Conservation/finite-state check failed: {summary}')
    if min(r['pressure_min_bar'] for r in records) <= 0:
        raise RuntimeError('Nonpositive absolute pressure')
    if model.cfg['producer_enabled']:
        if min(r['production_kg_day'] for r in records) < -1e-5:
            raise RuntimeError('Producer reversed flow; use an appropriate operating schedule')
        if max(abs(r['bhp_bar']-model.cfg['producer_bhp_bar']) for r in records[1:]) > 1e-4:
            raise RuntimeError('Producer BHP not maintained')
    write_csv(output / 'history.csv', records)
    snapshot_rows = []
    cfg = model.cfg
    xyz, _, _ = geometry(cfg)
    for day, state in snapshots:
        phase = phase_state(state[:, 0], state[:, 1], cfg)
        for cell, (pressure, zgas) in enumerate(state):
            snapshot_rows.append({'time_days': day, 'cell': cell,
                'x_m': xyz[cell, 0], 'y_m': xyz[cell, 1], 'depth_m': xyz[cell, 2],
                'pressure_bar': pressure, 'gas_mole_fraction': zgas,
                'gas_saturation': phase['gas_saturation'][cell],
                'gas_density_kg_m3': phase['gas_density_kg_m3'][cell]})
    write_csv(output / 'spatial_snapshots.csv', snapshot_rows)
    for label, (day, state) in [('initial', snapshots[0]), ('final', snapshots[-1])]:
        phase = phase_state(state[:, 0], state[:, 1], cfg)
        export_vtk(output / f'reservoir_{label}.vtk', cfg,
                   {'pressure_bar': state[:, 0], 'gas_saturation': phase['gas_saturation']})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if plots:
        plot_results(summary, records, snapshots, output)
    return summary, records, snapshots


def plot_results(summary, records, snapshots, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg = summary['parameters']
    t = np.array([r['time_days'] for r in records])
    col = lambda name: np.array([r[name] for r in records])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    axes[0, 0].plot(t, col('pressure_mean_bar'), label='Mean reservoir')
    axes[0, 0].fill_between(t, col('pressure_min_bar'), col('pressure_max_bar'), alpha=.2, label='Cell range')
    axes[0, 0].axhline(cfg['producer_bhp_bar'], color='gray', ls='--', label='Producer BHP')
    axes[0, 0].set_ylabel('Absolute pressure (bar)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(t[1:], col('production_kg_day')[1:]/1000)
    axes[0, 1].set_ylabel('Gas production (tonne/day)')
    axes[1, 0].plot(t, col('produced_kg')/1e6, label='Produced')
    axes[1, 0].plot(t, col('gas_kg')/1e6, label='Remaining')
    if cfg['recharge_kg_day']:
        axes[1, 0].plot(t, col('recharged_kg')/1e6, label='Recharge')
    axes[1, 0].set_ylabel('Gas mass (million kg)')
    axes[1, 0].legend(fontsize=8)
    scale = records[0]['native_gas_kg']+records[-1]['recharged_kg']
    axes[1, 1].plot(t, col('gas_balance_residual_kg')/scale, label='Native balance')
    axes[1, 1].plot(t, col('physical_gas_balance_residual_kg')/scale, label='Exact EOS inventory')
    axes[1, 1].set_ylabel('Relative gas balance residual')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Time (days)')
        ax.grid(alpha=.2)
    fig.suptitle(f"{cfg['species']} extraction | illustrative closed-boundary reservoir")
    fig.savefig(output / 'production.png', dpi=160)
    plt.close(fig)
    day, state = snapshots[-1]
    sg = phase_state(state[:, 0], state[:, 1], cfg)['gas_saturation']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, values, title, label in zip(axes, [state[:, 0], sg],
                                      ['Reservoir pressure', 'Gas saturation'], ['bar absolute', 'fraction']):
        im = ax.imshow(values.reshape(cfg.get('nz', 1), cfg['ny'], cfg['nx'])[cfg.get('nz', 1)//2], origin='lower',
                       extent=[0, cfg['length_x_m'], 0, cfg['length_y_m']], aspect='equal', cmap='viridis')
        if cfg['producer_enabled']:
            i, j = (cfg['nx']+1)//2-1, (cfg['ny']+1)//2-1
            ax.plot((i+.5)*cfg['length_x_m']/cfg['nx'], (j+.5)*cfg['length_y_m']/cfg['ny'],
                    'r+', ms=10, label='Producer')
            ax.legend(fontsize=8)
        fig.colorbar(im, ax=ax, label=label, shrink=.8)
        ax.set(xlabel='x (m)', ylabel='y (m)', title=title)
    fig.suptitle(f"{cfg['species']} at day {day:g} | middle horizontal layer")
    fig.savefig(output / 'reservoir_maps.png', dpi=160)
    plt.close(fig)
    plot_reservoir_3d(output / 'reservoir_3d.png', cfg, state[:, 0], sg,
                       title=f"{cfg['species']} intermediate 3D DARTS model", day=day)


def run_suite(output=None, overrides=None):
    """Run hydrogen, methane and the prescribed-H2-supply sensitivity."""
    output = Path(output or Path(__file__).with_name('output'))
    base = overrides or {}
    cases = {'hydrogen': {}, 'methane': {'species': 'CH4', 'eos': 'peng_robinson'},
             'recharge': {'recharge_kg_day': 300.}}
    results = {}
    histories = {}
    for name, settings in cases.items():
        results[name], histories[name], _ = run_case(base | settings, output / name)
        print(f"Completed {name}: {results[name]['produced_gas_kg']:.3f} kg", flush=True)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for name, rows in histories.items():
        time = [r['time_days'] for r in rows]
        axes[0].plot(time, [r['pressure_mean_bar'] for r in rows], label=name)
        axes[1].plot(time, [100*r['produced_kg']/rows[0]['gas_kg'] for r in rows], label=name)
    axes[0].set_ylabel('Mean pressure (bar absolute)')
    axes[1].set_ylabel('Produced gas / initial gas mass (%)')
    for ax in axes:
        ax.set_xlabel('Time (days)'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Intermediate 3D DARTS model | gas and prescribed supply comparisons')
    fig.savefig(output / 'scenario_comparison.png', dpi=170)
    plt.close(fig)
    (output / 'scenario_summary.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--days', type=float)
    parser.add_argument('--methane', action='store_true')
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    if args.days is not None:
        overrides['runtime_days'] = args.days
    if args.suite:
        run_suite(args.output, overrides)
        return
    if args.methane:
        overrides.update(species='CH4', eos='peng_robinson')
    output = args.output or Path(__file__).with_name('output') / ('methane' if args.methane else 'hydrogen')
    summary, _, _ = run_case(overrides, output)
    print(f"Completed {summary['days']:g} days in real DARTS: {output.resolve()}")
    print(f"Produced {summary['produced_gas_kg']/1000:,.3f} tonnes; mean pressure {summary['final_pressure_mean_bar']:.3f} bar")
    print(f"Native gas balance error: {summary['max_relative_gas_balance_error']:.3g}")


if __name__ == '__main__':
    main()
