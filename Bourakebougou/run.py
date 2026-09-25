"""Run the third model: a Bourakébougou-inspired 3D compositional reservoir.

From the repository root: "$DARTS_PY" Bourakebougou/run.py
Use --suite for closed depletion, water influx, H2 supply and mobility scenarios.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
from importlib.metadata import version
import numpy as np
from darts.engines import redirect_darts_output,set_num_threads

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visualize_3d import geometry,export_vtk,plot_reservoir_3d
try:
    from .model import Model
    from .properties import MW,COMPONENT_LABELS
except ImportError:
    from Bourakebougou.model import Model
    from Bourakebougou.properties import MW,COMPONENT_LABELS


class RecordedModel(Model):
    def __init__(self,parameters=None):
        super().__init__(parameters)
        self.records=[]

    def record(self):
        state=self.reservoir_state();time=float(self.physics.engine.t)
        if not np.isfinite(state).all() or state[:,0].min()<1.1 or state[:,0].max()>100:
            raise RuntimeError('Accepted states left the 1.1–100 bar property screening range')
        composition=np.column_stack((state[:,1:],1-state[:,1:].sum(axis=1)))
        if composition.min()<-1e-8 or composition.max()>1+1e-8:
            raise RuntimeError('Accepted component fractions left the composition simplex')
        exact=self.inventory();native=self.inventory(interpolated=True)
        rates=self.perforation_rates() if time>0 and self.reservoir.wells else None
        q=np.asarray(rates['component_kg_day']) if rates else np.zeros(4)
        phase_q=np.asarray(rates['phase_component_kg_day']) if rates else np.zeros((2,4))
        sources=np.asarray(self.source_rates()['component_kg_day'])
        previous=self.records[-1] if self.records else None
        dt=time-previous['time_days'] if previous else 0.
        dry_gas=phase_q[0,:3]/MW[:3]
        row={'time_days':time,'dt_days':dt,'pressure_mean_bar':float(state[:,0].mean()),
             'pressure_min_bar':float(state[:,0].min()),'pressure_max_bar':float(state[:,0].max()),
             'free_H2_kg':exact['free_hydrogen_kg'],'dissolved_H2_kg':exact['dissolved_hydrogen_kg'],
             'gas_phase_production_kg_day':float(phase_q[0].sum()),
             'water_phase_production_kg_day':float(phase_q[1].sum()),
             'produced_dry_gas_H2_mole_fraction':float(dry_gas[0]/dry_gas.sum()) if dry_gas.sum()>0 else 0.,
             'bhp_bar':float(self.physics.engine.time_data['PROD : BHP (bar)'][-1]) if rates else self.cfg['datum_pressure_bar']}
        for j,name in enumerate(COMPONENT_LABELS):
            row[f'{name}_kg']=exact['component_kg'][j]
            row[f'native_{name}_kg']=native['component_kg'][j]
            row[f'{name}_rate_kg_day']=float(q[j])
            row[f'{name}_produced_kg']=(previous[f'{name}_produced_kg'] if previous else 0.)+dt*q[j]
            row[f'{name}_supplied_kg']=(previous[f'{name}_supplied_kg'] if previous else 0.)+dt*sources[j]
            first=self.records[0] if self.records else row
            row[f'{name}_balance_residual_kg']=(row[f'native_{name}_kg']-first[f'native_{name}_kg']
                                                +row[f'{name}_produced_kg']-row[f'{name}_supplied_kg'])
            row[f'{name}_physical_balance_residual_kg']=(row[f'{name}_kg']-first[f'{name}_kg']
                                                +row[f'{name}_produced_kg']-row[f'{name}_supplied_kg'])
        self.records.append(row)

    def after_converged_timestep(self):
        super().after_converged_timestep();self.record()


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_case(parameters=None,output=None,report_days=None,*,plots=True):
    output=Path(output or Path(__file__).with_name('output')/'closed').resolve();output.mkdir(parents=True,exist_ok=True)
    status=output/'status.json';status.write_text('{"status":"running"}\n')
    try:result=_run_case(parameters,output,report_days,plots=plots)
    except Exception as error:
        status.write_text(json.dumps({'status':'failed','error':str(error),'notice':'Results may be incomplete or left over from an earlier run.'},indent=2)+'\n');raise
    status.write_text(json.dumps({'status':'complete','days':result[0]['days']})+'\n')
    return result


def _run_case(parameters,output,report_days,*,plots):
    redirect_darts_output(str(output/'native.log'));set_num_threads(2)
    model=RecordedModel(parameters);model.init();model.set_output(output_folder=str(output));model.record()
    end=model.cfg['runtime_days']
    times=np.asarray(report_days if report_days is not None else sorted(set(min(t,end) for t in [30.,90.,180.,end])),float)
    if not len(times) or times[0]<=0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or abs(times[-1]-end)>1e-8:
        raise ValueError('report_days must increase above zero and finish at runtime_days')
    snapshots=[(0.,model.reservoir_state())]
    for day in times:
        model.run(float(day-model.physics.engine.t),verbose=0,save_well_data=bool(model.reservoir.wells),save_well_data_after_run=bool(model.reservoir.wells))
        if abs(model.physics.engine.t-day)>1e-8:raise RuntimeError(f'DARTS stopped before day {day}')
        snapshots.append((float(day),model.reservoir_state()))
    rows=model.records;first,last=rows[0],rows[-1]
    balance={};physical_balance={}
    for name in COMPONENT_LABELS:
        scale=max(first[f'native_{name}_kg']+last[f'{name}_supplied_kg'],1e-10)
        balance[name]=max(abs(r[f'{name}_balance_residual_kg']) for r in rows)/scale
        physical_balance[name]=max(abs(r[f'{name}_physical_balance_residual_kg']) for r in rows)/scale
    summary={'model_scope':'Bourakebougou-inspired 3D compositional scenario; no field history match',
        'open_darts_version':version('open-darts'),'parameters':model.cfg,'days':last['time_days'],
        'reservoir_cells':len(snapshots[0][1]),'accepted_timesteps':len(rows)-1,
        'initial_component_kg':{n:first[f'{n}_kg'] for n in COMPONENT_LABELS},
        'remaining_component_kg':{n:last[f'{n}_kg'] for n in COMPONENT_LABELS},
        'produced_component_kg':{n:last[f'{n}_produced_kg'] for n in COMPONENT_LABELS},
        'supplied_component_kg':{n:last[f'{n}_supplied_kg'] for n in COMPONENT_LABELS},
        'initial_free_H2_kg':first['free_H2_kg'],'initial_dissolved_H2_kg':first['dissolved_H2_kg'],
        'remaining_free_H2_kg':last['free_H2_kg'],'remaining_dissolved_H2_kg':last['dissolved_H2_kg'],
        'final_pressure_mean_bar':last['pressure_mean_bar'],
        'final_pressure_range_bar':[last['pressure_min_bar'],last['pressure_max_bar']],
        'final_H2_rate_kg_day':last['H2_rate_kg_day'],
        'final_produced_dry_gas_H2_mole_fraction':last['produced_dry_gas_H2_mole_fraction'],
        'max_relative_component_balance_error':balance,'max_relative_physical_component_balance_error':physical_balance,
        'finite_states':bool(all(np.isfinite(list(r.values())).all() for r in rows)),
        'balance_method':'Native OBL accumulation and independent TPFA perforation flux, integrated at accepted backward-Euler timesteps; reservoir storage excludes well storage'}
    if not summary['finite_states'] or max(balance.values())>2e-5:raise RuntimeError(f'Component conservation failed: {balance}')
    if model.cfg['producer_enabled']:
        if min(r['H2_rate_kg_day'] for r in rows)<-1e-6:raise RuntimeError('H2 producer reversed flow')
        if max(abs(r['bhp_bar']-model.cfg['producer_bhp_bar']) for r in rows[1:])>1e-4:raise RuntimeError('BHP control not maintained')
    write_csv(output/'history.csv',rows)
    xyz,_,_=geometry(model.cfg);spatial=[];snapshot_fields=[]
    for day,state in snapshots:
        phase=model.cell_properties(state)
        fields={'pressure_bar':state[:,0],'gas_saturation':phase['saturation'][:,0],
                'gas_H2_mole_fraction':phase['composition'][:,0,0],
                'aqueous_H2_mole_fraction':phase['composition'][:,1,0]}
        snapshot_fields.append((day,state,fields))
        for j,point in enumerate(xyz):
            spatial.append({'time_days':day,'cell':j,'x_m':point[0],'y_m':point[1],'depth_m':point[2],
                            **{name:float(values[j]) for name,values in fields.items()}})
    write_csv(output/'spatial_snapshots.csv',spatial)
    export_vtk(output/'reservoir_initial.vtk',model.cfg,snapshot_fields[0][2])
    export_vtk(output/'reservoir_final.vtk',model.cfg,snapshot_fields[-1][2])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if plots:plot_results(summary,rows,snapshot_fields,output)
    return summary,rows,snapshots


def plot_results(summary,rows,snapshots,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([r['time_days'] for r in rows]);col=lambda name:np.array([r[name] for r in rows])
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    axes[0,0].plot(t,col('pressure_mean_bar'));axes[0,0].fill_between(t,col('pressure_min_bar'),col('pressure_max_bar'),alpha=.2)
    axes[0,0].set_ylabel('Pressure (bar absolute)')
    axes[0,1].plot(t,col('H2_produced_kg')/1000,label='Produced');axes[0,1].plot(t,col('H2_supplied_kg')/1000,label='Supplied');axes[0,1].legend();axes[0,1].set_ylabel('Cumulative H₂ (tonne)')
    axes[0,2].plot(t,col('free_H2_kg')/1000,label='Gas phase');axes[0,2].plot(t,col('dissolved_H2_kg')/1000,label='Aqueous phase');axes[0,2].legend();axes[0,2].set_ylabel('Remaining H₂ (tonne)')
    axes[1,0].plot(t[1:],col('H2_rate_kg_day')[1:]);axes[1,0].set_ylabel('H₂ production (kg/day)')
    axes[1,1].plot(t[1:],100*col('produced_dry_gas_H2_mole_fraction')[1:]);axes[1,1].set_ylabel('Produced dry-gas H₂ (mol%)')
    for name in COMPONENT_LABELS:
        scale=rows[0][f'native_{name}_kg']+rows[-1][f'{name}_supplied_kg']
        axes[1,2].plot(t,col(f'{name}_balance_residual_kg')/scale,label=name)
    axes[1,2].set_ylabel('Relative component balance residual');axes[1,2].legend(fontsize=8)
    for ax in axes.flat:ax.set_xlabel('Time (days)');ax.grid(alpha=.2)
    fig.suptitle('Bourakébougou-inspired compositional scenario | assumed geometry and operating conditions')
    fig.savefig(output/'production.png',dpi=170);plt.close(fig)
    day,state,fields=snapshots[-1]
    plot_reservoir_3d(output/'reservoir_3d.png',summary['parameters'],state[:,0],fields['gas_saturation'],title='Bourakébougou-inspired 3D model',day=day)
    cfg=summary['parameters'];nx,ny,nz=cfg['nx'],cfg['ny'],cfg['nz'];top=cfg['top_depth_m']
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for ax,name,title in zip(axes,['pressure_bar','gas_saturation','aqueous_H2_mole_fraction'],['Pressure (bar absolute)','Gas saturation','Dissolved H₂ mole fraction']):
        image=fields[name].reshape(nz,ny,nx)[:,ny//2,:]
        im=ax.imshow(image,origin='upper',extent=[0,cfg['length_x_m'],top+cfg['thickness_m'],top],aspect='auto')
        fig.colorbar(im,ax=ax,shrink=.8);ax.set(xlabel='x (m)',ylabel='Depth (m)',title=title)
    fig.suptitle(f'Central vertical section at day {day:g}; vertical scale expanded')
    fig.savefig(output/'vertical_sections.png',dpi=170);plt.close(fig)


def run_suite(output=None,overrides=None):
    output=Path(output or Path(__file__).with_name('output'));base=overrides or {}
    cases={'closed':{},'water_influx':{'water_influx_kg_day':100000.},
           'hydrogen_supply':{'hydrogen_recharge_kg_day':100.},
           'lower_gas_mobility':{'gas_relperm_endpoint':.22}}
    results={}
    for name,settings in cases.items():
        results[name]=run_case(base|settings,output/name)[0]
        print(f'{name}: {results[name]["produced_component_kg"]["H2"]:.3f} kg H2 produced',flush=True)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for name,summary in results.items():
        with (output/name/'history.csv').open() as f:rows=list(csv.DictReader(f))
        t=np.array([float(r['time_days']) for r in rows])
        for ax,key,scale in zip(axes,['pressure_mean_bar','H2_produced_kg','dissolved_H2_kg'],[1,1000,1000]):
            ax.plot(t,[float(r[key])/scale for r in rows],label=name.replace('_',' '))
    for ax,label in zip(axes,['Mean pressure (bar absolute)','Produced H₂ (tonne)','Dissolved H₂ remaining (tonne)']):
        ax.set(xlabel='Time (days)',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Independent assumed supply and mobility scenarios; not a field history match')
    fig.savefig(output/'scenario_comparison.png',dpi=170);plt.close(fig)
    (output/'scenario_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--days',type=float);parser.add_argument('--suite',action='store_true')
    args=parser.parse_args();overrides=json.loads(args.config.read_text()) if args.config else {}
    if args.days is not None:overrides['runtime_days']=args.days
    if args.suite:run_suite(args.output,overrides)
    else:
        summary,_,_=run_case(overrides,args.output)
        print(f"Completed {summary['days']:g} days; H2 production {summary['produced_component_kg']['H2']:,.3f} kg")
        print('Native component balance errors:',summary['max_relative_component_balance_error'])

if __name__=='__main__':main()
