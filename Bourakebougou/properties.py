"""Isothermal gas–water properties for a Bourakébougou-inspired scenario.

Components H2/N2/C1/H2O; public labels H2/N2/CH4/H2O. Equilibrium uses the
installed DARTS-flash PR/Ziabakhsh fugacity models. Density is evaluated
separately: direct PR gas volume, IAPWS-97 dilute aqueous density. Stock
EoSDensity(flash, phase_idx) and the default mixed Jager AQ constructor are
intentionally avoided after independent API checks found incorrect densities
and an H2-related constructor crash in the installed versions.
"""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from iapws import IAPWS97
from dartsflash.components import CompData
from dartsflash.mixtures import Mixture
from dartsflash.libflash import EoS, AQEoS
from darts.physics.base.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc, RockCompactionEvaluator

COMPONENTS = ["H2", "N2", "C1", "H2O"]
COMPONENT_LABELS = ["H2", "N2", "CH4", "H2O"]
MW = np.array([2.01588, 28.0134, 16.04246, 18.01528])  # kg/kmol
GRAVITY_BAR_PER_M_PER_KG_M3 = 9.80665e-5


def load_config(parameters=None):
    cfg = json.loads(Path(__file__).with_name("config.json").read_text())
    if isinstance(parameters, (str, Path)):
        parameters = json.loads(Path(parameters).read_text())
    if parameters:
        unknown = set(parameters) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        cfg.update(parameters)
    for key in ("nx", "ny", "nz"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, value in cfg.items():
        if isinstance(value, (float, int)) and not np.isfinite(value):
            raise ValueError(f"{key} must be finite")
    positive = ("length_x_m", "length_y_m", "thickness_m", "datum_pressure_bar",
                "producer_bhp_bar", "porosity", "permeability_md", "vertical_permeability_ratio",
                "well_radius_m", "gas_relperm_exponent", "water_relperm_exponent", "capillary_entry_bar",
                "capillary_exponent", "capillary_max_bar", "runtime_days", "first_timestep_days",
                "max_timestep_days", "nonlinear_tolerance", "linear_tolerance",
                "obl_pressure_step_bar", "obl_hydrogen_step", "obl_trace_step")
    if any(cfg[key] <= 0 for key in positive):
        raise ValueError("Geometry, flow/solver scales and pressures must be positive")
    if not 0 < cfg["porosity"] < 1 or not 0 <= cfg["residual_water_saturation"] < 1:
        raise ValueError("Invalid porosity or residual water saturation")
    if not 0 <= cfg["residual_gas_saturation"] < 1-cfg["residual_water_saturation"]:
        raise ValueError("Residual phase saturations must sum to less than one")
    if not 0 < cfg["gas_relperm_endpoint"] <= 1:
        raise ValueError("Gas relative-permeability endpoint must be in (0,1]")
    if not 0 < cfg["aqueous_initial_saturation_fraction"] <= 1:
        raise ValueError("Aqueous initial dissolved-gas saturation fraction must be in (0,1]")
    if not 290 <= cfg["temperature_k"] <= 350:
        raise ValueError("This shallow isothermal scenario supports 290–350 K")
    if not 1.1 <= cfg["producer_bhp_bar"] <= cfg["datum_pressure_bar"] <= 100:
        raise ValueError("Require 1.1 <= producer BHP <= datum pressure <= 100 bar")
    top, bottom = cfg["top_depth_m"], cfg["top_depth_m"] + cfg["thickness_m"]
    if not top < cfg["gas_water_contact_depth_m"] < bottom:
        raise ValueError("Gas-water contact must lie within the model")
    if not top <= cfg["datum_depth_m"] < cfg["gas_water_contact_depth_m"]:
        raise ValueError("Pressure datum must be inside the gas cap")
    if not top <= cfg["producer_completion_top_m"] < cfg["producer_completion_bottom_m"] <= bottom:
        raise ValueError("Producer completion must lie within the layer")
    if not cfg["producer_completion_top_m"] <= cfg["producer_depth_m"] <= cfg["producer_completion_bottom_m"]:
        raise ValueError("Producer pressure-datum depth must lie within its completion")
    for key in ("hydrogen_recharge_kg_day", "water_influx_kg_day", "aqueous_effective_diffusion_m2_s",
                "rock_compressibility_per_bar"):
        if cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if cfg["timestep_multiplier"] <= 1 or cfg["first_timestep_days"] > cfg["max_timestep_days"]:
        raise ValueError("Invalid timestep controls")
    gas = np.asarray(cfg["dry_gas_mole_fractions"])
    if gas.shape != (3,) or not np.isfinite(gas).all() or np.any(gas <= 0) or not np.isclose(gas.sum(), 1):
        raise ValueError("Provide positive normalized dry H2/N2/CH4 mole fractions")
    return cfg


@lru_cache(maxsize=8192)
def water_properties(pressure_bar, temperature_k):
    water = IAPWS97(P=float(pressure_bar)*0.1, T=float(temperature_k))
    if water.phase != "Liquid":
        raise ValueError("The aqueous density model requires liquid water")
    return float(water.rho), float(water.mu*1000)  # kg/m³ and cP


class FlashSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.comp_data = CompData(COMPONENTS, setprops=True)
        self.comp_data.Mw = list(MW)
        self.flash = Mixture(self.comp_data)
        # STABLE labels the PR phase without requiring the native auxiliary
        # critical-point classifier to label it MAX. That classifier can return
        # NaN for H2 mixtures and otherwise silently omit a valid gas phase from
        # mapped flash output. The hybrid aqueous root preference is retained.
        self.flash.set_vl_eos("V", hybrid_aq_eos_name="Aq", root_order=[EoS.STABLE])
        aq = AQEoS(self.comp_data, {AQEoS.water: AQEoS.Ziabakhsh2012,
                                   AQEoS.solute: AQEoS.Ziabakhsh2012})
        self.flash.set_aq_eos("Aq", aq_eos=aq)
        self.flash.init_ptflash(eos_order=["V", "Aq"], min_z=1e-12)
        self.gas_eos = self.flash.eos["V"]
        self.gas_eos.set_root_flag(EoS.STABLE)

    def split(self, pressure_bar, composition):
        if pressure_bar <= 0:
            raise ValueError("Absolute flash pressure must be positive")
        z = np.maximum(np.asarray(composition, float), 1e-12)
        z /= z.sum()
        error = self.flash.evaluate(float(pressure_bar), self.cfg["temperature_k"], z)
        result = self.flash.get_flash_results()
        nu = np.asarray(result.nu).copy()
        x = np.asarray(result.X).reshape(2, 4).copy()
        if (error or not np.isfinite(x).all() or not np.isfinite(nu).all()
                or abs(nu.sum()-1)>1e-10 or np.min(nu)<-1e-12
                or np.max(np.abs(nu@x-z))>1e-7):
            raise RuntimeError(f"Gas-water flash failed at {pressure_bar} bar, z={z.tolist()}, nu={nu.tolist()}: error {error}")
        return nu, x

    def gas_density(self, pressure_bar, composition):
        self.gas_eos.set_root_flag(EoS.STABLE)
        volume = self.gas_eos.V(float(pressure_bar), self.cfg["temperature_k"], np.asarray(composition))
        return float(np.dot(MW, composition)*1e-3/volume)

    def equilibrium_endmembers(self, pressure_bar):
        # Reported dry composition is used as a total dry-gas ratio; aqueous
        # partitioning produces a very small difference in equilibrium gas ratio.
        feed = np.r_[0.2*np.asarray(self.cfg["dry_gas_mole_fractions"]), 0.8]
        _, x = self.split(pressure_bar, feed)
        return x

    def state_properties(self, pressure_bar, z):
        nu, x = self.split(pressure_bar, z)
        rho = np.zeros(2)
        molar_rho = np.zeros(2)
        for phase in range(2):
            if nu[phase] > 0:
                rho[phase] = (self.gas_density(pressure_bar, x[phase]) if phase == 0 else
                              water_properties(float(pressure_bar), self.cfg["temperature_k"])[0])
                molar_rho[phase] = rho[phase]/np.dot(MW,x[phase])
        volume = np.divide(nu, molar_rho, out=np.zeros(2), where=molar_rho>0)
        saturation = volume/volume.sum()
        return {"nu": nu, "x": x, "rho_kg_m3": rho, "rho_kmol_m3": molar_rho,
                "saturation": saturation}


class GasDensity:
    def __init__(self, system): self.system = system
    def evaluate(self, pressure, temperature, x): return self.system.gas_density(pressure,x)


class AqueousDensity:
    def evaluate(self, pressure, temperature, x): return water_properties(float(pressure),float(temperature))[0]


class AqueousViscosity:
    def evaluate(self, pressure, temperature, x, rho): return water_properties(float(pressure),float(temperature))[1]


class DiluteGasViscosity:
    """Wilke dilute-mixture viscosity; approximate pure-component reference values.

    H2 uses NIST's 300 K value (8.95 µPa·s). N2/CH4/H2O reference values
    agree to tabulated precision with Huber, Viscosity of Gases (NIST):
    https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=926439
    T^0.7 scaling is an approximation over this shallow range; density-dependent
    transport and dissolved-gas effects on water are omitted.
    """
    def evaluate(self, pressure, temperature, x, rho):
        mu = np.array([.00895,.01781,.01112,.00977])*(temperature/300.)**0.7
        matrix = (1+np.sqrt(mu[:,None]/mu[None,:])*(MW[None,:]/MW[:,None])**.25)**2 / np.sqrt(8*(1+MW[:,None]/MW[None,:]))
        return float(np.sum(np.asarray(x)*mu/(matrix@np.asarray(x))))


class RelativePermeability:
    def __init__(self, residual_self, residual_other, exponent, endpoint=1.):
        self.sr,self.other,self.exponent,self.endpoint=residual_self,residual_other,exponent,endpoint
    def evaluate(self,saturation):
        return float(self.endpoint*np.clip((saturation-self.sr)/(1-self.sr-self.other),0,1)**self.exponent)


class CapillaryPressure:
    """Modified Brooks-Corey: zero at Sw=1; positive Pc=pg-pw above contact."""
    def __init__(self,cfg): self.cfg=cfg
    def evaluate(self,saturation):
        c=self.cfg
        effective=np.clip((saturation[1]-c["residual_water_saturation"])/(1-c["residual_water_saturation"]),1e-8,1)
        pc=min(c["capillary_max_bar"], c["capillary_entry_bar"]*(effective**(-1/c["capillary_exponent"])-1))
        return np.array([0.,pc])


class GasWaterProperties(PropertyContainer):
    def __init__(self,cfg,system,source_bulk_volume_m3=0.):
        self.cfg,self.system=cfg,system
        self.source_bulk_volume_m3=source_bulk_volume_m3
        super().__init__(["gas","water"],COMPONENTS,MW,eps_z=1e-12,
                         rock_comp=cfg["rock_compressibility_per_bar"],temperature=cfg["temperature_k"])
        self.rock_compr_ev=RockCompactionEvaluator(pref=cfg["datum_pressure_bar"],compres=cfg["rock_compressibility_per_bar"])
        self.density_ev={"gas":GasDensity(system),"water":AqueousDensity()}
        self.viscosity_ev={"gas":DiluteGasViscosity(),"water":AqueousViscosity()}
        swr,sgr=cfg["residual_water_saturation"],cfg["residual_gas_saturation"]
        self.rel_perm_ev={"gas":RelativePermeability(sgr,swr,cfg["gas_relperm_exponent"],cfg["gas_relperm_endpoint"]),
                          "water":RelativePermeability(swr,sgr,cfg["water_relperm_exponent"])}
        self.capillary_pressure_ev=CapillaryPressure(cfg)
        # A common effective molar Fick coefficient gives zero summed molar
        # diffusion flux. Native operators already supply phi, saturation, rho.
        self.diffusion_ev={"gas":ConstFunc(np.zeros(4)),"water":ConstFunc(np.ones(4)*cfg["aqueous_effective_diffusion_m2_s"]*86400)}
        self.output_props={"gas_saturation":lambda:self.sat[0],"dissolved_H2_mole_fraction":lambda:self.x[1,0]}

    def run_flash(self,pressure,temperature,zc,evaluate_PT=False):
        self.nu,self.x=self.system.split(pressure,zc)
        self.temperature=temperature
        return np.flatnonzero(self.nu>0)

    def evaluate_mass_source(self,pressure,temperature,zc):
        self.mass_source[:]=0
        if self.source_bulk_volume_m3>0:
            self.mass_source[0]=-self.cfg["hydrogen_recharge_kg_day"]/MW[0]/self.source_bulk_volume_m3
            self.mass_source[3]=-self.cfg["water_influx_kg_day"]/MW[3]/self.source_bulk_volume_m3
        return self.mass_source


def initial_profile(cfg,system):
    """Layer-centre gas/water hydrostatic pressures and capillary saturations.

    Continuous hydrostatics is discretized onto finite volumes; flash capillary
    pressure corrections to chemical equilibrium are neglected. A no-well run
    measures resulting discretization/initial-equilibration transients.
    """
    depths=cfg["top_depth_m"]+(np.arange(cfg["nz"])+.5)*cfg["thickness_m"]/cfg["nz"]
    contact=cfg["gas_water_contact_depth_m"]
    datum=cfg["datum_depth_m"]
    def gas_rhs(z,p):
        x=system.equilibrium_endmembers(float(p[0]))[0]
        return [GRAVITY_BAR_PER_M_PER_KG_M3*system.gas_density(float(p[0]),x)]
    def integrate(rhs,z0,p0,z):
        if z==z0:return p0
        return float(solve_ivp(rhs,[z0,z],[p0],rtol=1e-9,atol=1e-10).y[0,-1])
    pcontact=integrate(gas_rhs,datum,cfg["datum_pressure_bar"],contact)
    def water_rhs(z,p):
        return [GRAVITY_BAR_PER_M_PER_KG_M3*water_properties(float(p[0]),cfg["temperature_k"])[0]]
    states=[];sat=[]
    for depth in depths:
        pw=integrate(water_rhs,contact,pcontact,depth)
        if depth<contact:
            p=integrate(gas_rhs,datum,cfg["datum_pressure_bar"],depth)
            pc=max(p-pw,0.)
            if pc>=cfg["capillary_max_bar"]:raise ValueError("Hydrostatic capillary pressure exceeds configured cap")
            sw=cfg["residual_water_saturation"]+(1-cfg["residual_water_saturation"])*(1+pc/cfg["capillary_entry_bar"])**(-cfg["capillary_exponent"])
            x=system.equilibrium_endmembers(p)
            rhog=system.gas_density(p,x[0])/np.dot(MW,x[0])
            rhow=water_properties(p,cfg["temperature_k"])[0]/np.dot(MW,x[1])
            beta=(1-sw)*rhog/((1-sw)*rhog+sw*rhow)
            z=beta*x[0]+(1-beta)*x[1]
        else:
            p=pw
            z=system.equilibrium_endmembers(p)[1].copy()
            z[:3]*=cfg["aqueous_initial_saturation_fraction"]
            z[3]=1-z[:3].sum()
            sw=1.
        states.append(np.r_[p,z[:3]]);sat.append(1-sw)
    return depths,np.asarray(states),np.asarray(sat)
