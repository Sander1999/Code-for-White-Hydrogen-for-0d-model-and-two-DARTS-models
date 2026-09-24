"""Independent property checks; run ``python -m unittest ...test_properties``.

Henry's law is a low-pressure reference, not identical to the flash model:
its comparison omits pressure corrections and uses ideal gas fugacity. A 2%
model-comparison tolerance at 1.1–5 bar is therefore distinct from the much
stricter component conservation tolerance. PR density is compared with NIST's
hydrogen density correlation only in the shallow 1.1–30 bar, 298–313 K range.
These checks do not validate field geometry, permeability or supply rates.
"""
from __future__ import annotations
import unittest
import numpy as np
from iapws._iapws import _Henry
from .properties import (load_config, FlashSystem, CapillaryPressure,
                         GasWaterProperties, RelativePermeability, MW)
from white_hydrogen.darts.properties import gas_density_kg_m3


def property_benchmarks():
    """Return reproducible numerical evidence, with all deviations dimensionless."""
    henry=[];density=[];closure=[]
    dry=np.array([.98,.01,.01])
    for temperature in [298.15,303.15,313.15]:
        system=FlashSystem(load_config({'temperature_k':temperature}))
        for pressure in [1.1,5.,10.,30.]:
            rho=system.gas_density(pressure,np.array([1.,0.,0.,0.]))
            rho_ref=float(gas_density_kg_m3(pressure,temperature,'H2','nist'))
            density.append({'temperature_k':temperature,'pressure_bar':pressure,
                            'PR_kg_m3':rho,'NIST_kg_m3':rho_ref,
                            'relative_deviation':rho/rho_ref-1})
            if pressure<=5.:
                _,x=system.split(pressure,[.2,1e-12,1e-12,.8])
                reference=x[0,0]*pressure*.1/_Henry(temperature,'H2')
                henry.append({'temperature_k':temperature,'pressure_bar':pressure,
                              'flash_aqueous_H2':float(x[1,0]),'Henry_aqueous_H2':float(reference),
                              'relative_deviation':float(x[1,0]/reference-1)})
            for gas_fraction in [1e-6,1e-4,.001,.02,.2,.8,.999999]:
                z=np.r_[gas_fraction*dry,1-gas_fraction]
                q=system.state_properties(pressure,z)
                closure.append(float(np.max(np.abs(q['nu']@q['x']-z))))
                if not np.isclose(q['saturation'].sum(),1) or np.min(q['saturation'])<0:
                    raise AssertionError('Invalid phase saturation')
                if np.any(q['rho_kg_m3'][q['nu']>0]<=0):
                    raise AssertionError('Nonpositive density')
    return {'Henry_reference':henry,'density_reference':density,
            'max_flash_component_closure':max(closure),
            'flash_states_checked':len(closure)}


class TestReservoirProperties(unittest.TestCase):
    def test_independent_benchmarks(self):
        result=property_benchmarks()
        self.assertLess(max(abs(row['relative_deviation']) for row in result['Henry_reference']),.02)
        self.assertLess(max(abs(row['relative_deviation']) for row in result['density_reference']),.01)
        self.assertLess(result['max_flash_component_closure'],1e-10)

    def test_phase_appearance_with_dissolved_hydrogen(self):
        system=FlashSystem(load_config())
        dilute=system.state_properties(6.,[1e-6,1e-8,1e-8,1-1.02e-6])
        rich=system.state_properties(6.,[.02,.0002,.0002,.9796])
        self.assertAlmostEqual(dilute['saturation'][0],0.,places=12)
        self.assertGreater(rich['saturation'][0],0.1)
        self.assertGreater(rich['x'][1,0],0.)

    def test_hydrogen_mixture_critical_point_mapping_regression(self):
        # In the installed native version MAX-labelled mapping drops a valid
        # gas phase for this Newton-trial composition despite error_code=0.
        system=FlashSystem(load_config())
        z=np.array([.753315000001,.076371750001,.076349500001,.093963749997])
        nu,x=system.split(2.951,z)
        np.testing.assert_allclose(nu@x,z,atol=1e-12,rtol=0.)
        self.assertGreater(nu[0],.9)
        self.assertGreater(nu[1],.05)

    def test_capillary_and_mobility_endpoints(self):
        cfg=load_config();pc=CapillaryPressure(cfg)
        self.assertEqual(pc.evaluate([0.,1.])[1],0.)
        for pressure in [.01,.1,1.]:
            sw=cfg['residual_water_saturation']+(1-cfg['residual_water_saturation'])*(1+pressure/cfg['capillary_entry_bar'])**(-cfg['capillary_exponent'])
            self.assertAlmostEqual(pc.evaluate([1-sw,sw])[1],pressure,places=10)
        mobility=RelativePermeability(.02,.25,2.,.22)
        self.assertEqual(mobility.evaluate(.02),0.)
        self.assertAlmostEqual(mobility.evaluate(.75),.22)

    def test_prescribed_sources_are_bulk_volume_scaled(self):
        cfg=load_config({'hydrogen_recharge_kg_day':12.3,'water_influx_kg_day':456.})
        volume=2e6;pc=GasWaterProperties(cfg,FlashSystem(cfg),volume)
        native=pc.evaluate_mass_source(6.,303.15,np.array([.01,.0001,.0001,.9898]))
        np.testing.assert_allclose(-native*volume*MW,[12.3,0.,0.,456.],rtol=1e-14)


if __name__=='__main__':
    unittest.main()
