"""Native component conservation, source and discretization checks.

Runs real DARTS in a temporary folder and retains a compact validation.json.
Tests assess numerical reliability, not a calibrated Bourakébougou prediction.
"""
from pathlib import Path
import contextlib
import json
import tempfile
import numpy as np
try:
    from .run import run_case
    from .properties import load_config,COMPONENT_LABELS
except ImportError:
    from run import run_case
    from properties import load_config,COMPONENT_LABELS


def extended_refinement_checks(results,checks):
    """Compare successive lateral refinement and a fixed-completion vertical run."""
    base=results['baseline'];fine=results['fine_horizontal_grid']
    finer=results['finer_horizontal_grid'];vertical=results['fine_vertical_grid']
    metrics=results['refinement_metrics']
    horizontal=abs(fine['produced_component_kg']['H2']-finer['produced_component_kg']['H2'])/finer['produced_component_kg']['H2']
    vertical_diff=abs(base['produced_component_kg']['H2']-vertical['produced_component_kg']['H2'])/vertical['produced_component_kg']['H2']
    metrics['further_horizontal_grid_H2_production_relative_difference']=horizontal
    metrics['further_horizontal_grid_mean_pressure_difference_bar']=abs(fine['final_pressure_mean_bar']-finer['final_pressure_mean_bar'])
    metrics['vertical_grid_H2_production_relative_difference']=vertical_diff
    metrics['vertical_grid_mean_pressure_difference_bar']=abs(base['final_pressure_mean_bar']-vertical['final_pressure_mean_bar'])
    checks['successive_horizontal_grid_difference_decreases']=horizontal<metrics['horizontal_grid_H2_production_relative_difference']
    checks['vertical_grid_production_tolerance']=vertical_diff<.01


def validate(output=None):
    output=Path(output or Path(__file__).with_name('output')/'validation.json')
    cfg=load_config();results={};checks={}
    with tempfile.TemporaryDirectory(prefix='hydrogen_3d_validation_',dir='/private/tmp') as scratch:
        def case(name,overrides):
            with (Path(scratch)/f'{name}.stdout.log').open('w') as log,contextlib.redirect_stdout(log):
                result=run_case(overrides,Path(scratch)/name,plots=False)
            results[name]=result[0]
            print(f'Completed verification case {name}',flush=True)
            return result
        small={'nx':3,'ny':3,'producer_enabled':False,'runtime_days':10.}
        closed,closed_rows,closed_states=case('closed_hydrostatic',small)
        drift=float(np.max(np.abs(closed_states[-1][1][:,0]-closed_states[0][1][:,0])))
        results['closed_hydrostatic']['max_pressure_drift_bar']=drift
        # Continuous initial hydrostatics and cell-averaged capillary operators
        # need not be exactly stationary; bound and report their relaxation.
        checks['hydrostatic_relaxation_under_0_02_bar']=drift<.02
        checks['closed_all_components_conserved']=max(closed['max_relative_component_balance_error'].values())<2e-5
        for name,source in [('hydrogen_source',{'hydrogen_recharge_kg_day':100.}),
                            ('water_source',{'water_influx_kg_day':100000.}),
                            ('combined_source',{'hydrogen_recharge_kg_day':100.,'water_influx_kg_day':100000.})]:
            summary,rows,_=case(name,small|source)
            for component in COMPONENT_LABELS:
                source_rate=100. if component=='H2' and 'hydrogen_recharge_kg_day' in source else (100000. if component=='H2O' and 'water_influx_kg_day' in source else 0.)
                expected=rows[0][f'native_{component}_kg']+source_rate*10.
                error=abs(rows[-1][f'native_{component}_kg']-expected)/expected
                checks[f'{name}_{component}_independent_inventory']=error<2e-5
        base,_,_=case('baseline',{})
        timestep,_,_=case('half_timestep',{'max_timestep_days':cfg['max_timestep_days']/2})
        interpolation,_,_=case('fine_interpolation',{'obl_pressure_step_bar':cfg['obl_pressure_step_bar']/2,
                              'obl_hydrogen_step':cfg['obl_hydrogen_step']/2,'obl_trace_step':cfg['obl_trace_step']/2})
        fine_grid,_,_=case('fine_horizontal_grid',{'nx':2*cfg['nx']-1,'ny':2*cfg['ny']-1})
        metrics={}
        for name,other,tolerance in [('timestep',timestep,.01),('interpolation',interpolation,.01),('horizontal_grid',fine_grid,.05)]:
            diff=abs(base['produced_component_kg']['H2']-other['produced_component_kg']['H2'])/other['produced_component_kg']['H2']
            metrics[name+'_H2_production_relative_difference']=diff
            metrics[name+'_mean_pressure_difference_bar']=abs(base['final_pressure_mean_bar']-other['final_pressure_mean_bar'])
            checks[name+'_production_tolerance']=diff<tolerance
        results['refinement_metrics']=metrics
        # Odd lateral grids retain the same central well coordinates. Fixed
        # physical completion length and BHP datum are retained by Model in z.
        case('finer_horizontal_grid',{'nx':3*cfg['nx']-2,'ny':3*cfg['ny']-2})
        case('fine_vertical_grid',{'nz':3*cfg['nz']})
        extended_refinement_checks(results,checks)
    report={'passed':all(bool(x) for x in checks.values()),'checks':{k:bool(v) for k,v in checks.items()},
            'scope':'Four-component native conservation; isothermal freshwater property closure; successive horizontal grid, fixed-completion vertical grid, time and OBL refinement. Not field validation.',
            'results':results}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'checks':report['checks'],'refinement':metrics},indent=2))
    if not report['passed']:raise AssertionError(f'Validation failed; see {output}')
    return report

if __name__=='__main__':validate()
