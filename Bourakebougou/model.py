"""Native 3-D compositional flow inspired by Bourakébougou's shallow gas system.

This is a scenario model, not a history match. It separates initial gas/water
storage, production, optional prescribed H2 supply, and optional prescribed
water influx. No geological generation or recharge rate is inferred from the
reported field pressure. All boundaries are sealed except the producer; source
terms represent explicit external mass inputs into cells below the contact.
"""
from __future__ import annotations
import numpy as np
from darts.engines import value_vector, well_control_iface
from darts.models.darts_model import DartsModel
from darts.nonlinear_solvers import NewtonSolver
from darts.physics.base.physics import PhysicsBase
from darts.reservoirs.struct_reservoir import StructReservoir
try:
    from .properties import (COMPONENTS, COMPONENT_LABELS, MW, FlashSystem,
                             GasWaterProperties, initial_profile, load_config)
except ImportError:
    from properties import (COMPONENTS, COMPONENT_LABELS, MW, FlashSystem,
                            GasWaterProperties, initial_profile, load_config)


class Model(DartsModel):
    """Model(parameters=None), compatible with DARTS init/set_output/run workflow."""
    def __init__(self,parameters=None):
        self.cfg=load_config(parameters)
        self.component_labels=COMPONENT_LABELS
        super().__init__()
        self.timer.node["initialization"].start()
        self.system=FlashSystem(self.cfg)
        self.layer_depths,self.layer_states,self.initial_layer_gas_saturation=initial_profile(self.cfg,self.system)
        c=self.cfg
        self.cell_volume_m3=c["length_x_m"]*c["length_y_m"]*c["thickness_m"]/(c["nx"]*c["ny"]*c["nz"])
        self.depths=np.repeat(self.layer_depths,c["nx"]*c["ny"])
        self.source_region=np.where(self.depths>=c["gas_water_contact_depth_m"],1,0)
        self.source_bulk_volume_m3=float(np.count_nonzero(self.source_region)*self.cell_volume_m3)
        if not self.source_bulk_volume_m3:raise ValueError("Grid must contain cells below the gas-water contact")
        self.set_reservoir();self.set_physics()
        self.timer.node["initialization"].stop()

    def set_reservoir(self):
        c=self.cfg
        self.reservoir=StructReservoir(self.timer,nx=c["nx"],ny=c["ny"],nz=c["nz"],
            dx=c["length_x_m"]/c["nx"],dy=c["length_y_m"]/c["ny"],dz=c["thickness_m"]/c["nz"],
            permx=c["permeability_md"],permy=c["permeability_md"],
            permz=c["permeability_md"]*c["vertical_permeability_ratio"],
            poro=c["porosity"],depth=self.depths,start_z=c["top_depth_m"],
            op_num=self.source_region,cache=False)

    def set_physics(self):
        c=self.cfg
        self.physics=PhysicsBase(COMPONENTS,["gas","water"],self.timer,
            axes_step=[c["obl_pressure_step_bar"],c["obl_hydrogen_step"],c["obl_trace_step"],c["obl_trace_step"]],
            axes_origin=[0.101,1e-12,1e-12,1e-12],epsilon_z=1e-12,
            # Wet gas remains away from the sum(z)=1 face in the supported
            # shallow p/T range. Disable simplex extrapolation to permit fine
            # trace-component spacing without forcing the H2 axis equally fine.
            state_spec=PhysicsBase.StateSpecification.P,extrapolation_flag=False,cache=False)
        self.property_containers={}
        for region,volume in [(0,0.),(1,self.source_bulk_volume_m3)]:
            pc=GasWaterProperties(c,self.system,volume)
            self.property_containers[region]=pc
            self.physics.add_property_region(pc,region)
        self.property_container=self.property_containers[0]

    def set_wells(self):
        if self.cfg["producer_enabled"]:
            c=self.cfg
            self.reservoir.add_well("PROD")
            dz=c["thickness_m"]/c["nz"]
            i,j=(c["nx"]+1)//2,(c["ny"]+1)//2
            # Preserve the physical 105–110 m completion when refining z.
            # Fractional-cell completion scales the Peaceman WI by open length.
            for layer,depth in enumerate(self.layer_depths,1):
                opened=max(0.,min(depth+dz/2,c["producer_completion_bottom_m"])-
                           max(depth-dz/2,c["producer_completion_top_m"]))
                if opened<=1e-10:continue
                _,wi,_=self.reservoir.discretizer.calc_well_index(
                    i,j,layer,well_diameter=2*c["well_radius_m"],skin=c["well_skin"])
                self.reservoir.add_perforation("PROD",res_cell_idx=(i,j,layer),
                    well_diameter=2*c["well_radius_m"],skin=c["well_skin"],
                    well_index=wi*opened/dz,well_indexD=0.,ms_epm=False)
            # Single pressure node: native gravity connects every completion to
            # the same fixed BHP datum; resolved wellbore friction is omitted.
            well=self.reservoir.wells[-1]
            well.well_head_depth=well.well_body_depth=c["producer_depth_m"]
            length=c["producer_completion_bottom_m"]-c["producer_completion_top_m"]
            well.segment_volume*=length/dz
            well.segment_depth_increment=length

    def set_initial_conditions(self):
        states=np.repeat(self.layer_states,self.cfg["nx"]*self.cfg["ny"],axis=0)
        distribution={"pressure":states[:,0]}
        distribution.update({name:states[:,j+1] for j,name in enumerate(COMPONENTS[:3])})
        self.physics.set_initial_conditions_from_array(self.reservoir.mesh,distribution)

    def set_well_controls(self):
        for well in self.reservoir.wells:
            self.physics.set_well_controls(wctrl=well.control,control_type=well_control_iface.BHP,
                                           is_inj=False,target=self.cfg["producer_bhp_bar"])

    def set_solver(self):
        super().set_solver()
        c=self.cfg
        self.ts_control.dt_first=c["first_timestep_days"]
        self.ts_control.dt_min=1e-12
        self.ts_control.dt_max=c["max_timestep_days"]
        self.ts_control.dt_mult=c["timestep_multiplier"]
        self.ts_control.runtime=c["runtime_days"]
        self.nonlinear_solver=NewtonSolver(tolerance=c["nonlinear_tolerance"],max_iterations=20)
        self.linear_solver.spec.tolerance=c["linear_tolerance"]

    def reservoir_state(self):
        nb=self.reservoir.mesh.n_res_blocks
        return np.asarray(self.physics.engine.X)[:4*nb].reshape(nb,4).copy()

    def cell_properties(self,states=None):
        """Exact flash/phase properties per reservoir cell (copies, no history)."""
        states=self.reservoir_state() if states is None else np.asarray(states)
        fields={"saturation":[],"composition":[],"density_kg_m3":[],"density_kmol_m3":[],"capillary_pressure_bar":[]}
        for state in states:
            z=np.r_[state[1:],1-state[1:].sum()]
            q=self.system.state_properties(float(state[0]),z)
            fields["saturation"].append(q["saturation"])
            fields["composition"].append(q["x"])
            fields["density_kg_m3"].append(q["rho_kg_m3"])
            fields["density_kmol_m3"].append(q["rho_kmol_m3"])
            fields["capillary_pressure_bar"].append(self.property_container.capillary_pressure_ev.evaluate(q["saturation"])[1])
        return {name:np.asarray(values) for name,values in fields.items()}

    def inventory(self,interpolated=False):
        """Four component inventories [H2,N2,CH4,H2O], in kg and kmol.

        Exact mode also reports free/dissolved H2 and phase inventories. Native
        interpolation mode is the independent mass-balance reference and should
        not be mixed with exact flash phase inventories in a single balance.
        """
        mesh=self.reservoir.mesh
        states=self.reservoir_state()
        pv=np.asarray(mesh.volume)[:len(states)]*np.asarray(mesh.poro)[:len(states)]
        if interpolated:
            kmol=np.zeros(4)
            regions=np.asarray(mesh.op_num)[:len(states)]
            for i,state in enumerate(states):
                region=int(regions[i]);ev=self.physics.reservoir_operators[region]
                values=value_vector(np.zeros(ev.n_ops))
                self.physics.acc_flux_itor[region].evaluate(value_vector(state),values)
                kmol+=pv[i]*np.asarray(values)[ev.ACC_OP:ev.ACC_OP+4]
            return {"component_kmol":kmol.tolist(),"component_kg":(kmol*MW).tolist(),"hydrogen_kg":float(kmol[0]*MW[0])}
        q=self.cell_properties(states)
        rock=1+self.cfg["rock_compressibility_per_bar"]*(states[:,0]-self.cfg["datum_pressure_bar"])
        phase_kmol=pv[:,None,None]*rock[:,None,None]*q["saturation"][:,:,None]*q["density_kmol_m3"][:,:,None]*q["composition"]
        component_kmol=phase_kmol.sum(axis=(0,1));phase_component_kg=phase_kmol.sum(axis=0)*MW
        return {"component_kmol":component_kmol.tolist(),"component_kg":(component_kmol*MW).tolist(),
            "hydrogen_kg":float(component_kmol[0]*MW[0]),"free_hydrogen_kg":float(phase_component_kg[0,0]),
            "dissolved_hydrogen_kg":float(phase_component_kg[1,0]),"free_H2_kg":float(phase_component_kg[0,0]),
            "dissolved_H2_kg":float(phase_component_kg[1,0]),"phase_component_kg":phase_component_kg.tolist(),
            "gas_phase_kg":float(phase_component_kg[0].sum()),"water_phase_kg":float(phase_component_kg[1].sum())}

    def source_rates(self):
        """Prescribed external influx into below-contact cells, positive addition."""
        kg=np.array([self.cfg["hydrogen_recharge_kg_day"],0.,0.,self.cfg["water_influx_kg_day"]])
        return {"component_kg_day":kg.tolist(),"component_kmol_day":(kg/MW).tolist()}

    def phase_fields(self):
        q=self.cell_properties()
        return {"gas_saturation":q["saturation"][:,0],"gas_H2_mole_fraction":q["composition"][:,0,0],
                "aqueous_H2_mole_fraction":q["composition"][:,1,0],
                "gas_density_kg_m3":q["density_kg_m3"][:,0],"water_density_kg_m3":q["density_kg_m3"][:,1]}

    def perforation_rates(self):
        """Current native TPFA component fluxes; positive production into well.

        Uses FLUX × LAMBDA, real mesh WI/gravity, capillary pressure and phase
        upwinding. Legacy engine.time_data component rates omit mobility and must
        not be used for this model's material balances. Molecular diffusion acts
        only between reservoir cells, so it contributes no perforation flux.
        """
        mesh=self.reservoir.mesh
        states=np.asarray(self.physics.engine.X).reshape(mesh.n_blocks,4)
        regions=np.asarray(mesh.op_num)
        block_m,block_p=np.asarray(mesh.block_m),np.asarray(mesh.block_p)
        tran,gravity=np.asarray(mesh.tran),np.asarray(mesh.grav_coef)
        phase_component=np.zeros((2,4));phase_volume=np.zeros(2)
        for well in self.reservoir.wells:
            for segment,cell,_wi,_wid in well.perforations:
                well_cell=well.well_body_idx+segment
                indices=np.flatnonzero((block_m==cell)&(block_p==well_cell))
                if len(indices)!=1:raise RuntimeError("Expected a single native perforation connection")
                conn=int(indices[0]);region=int(regions[cell]);ev=self.physics.reservoir_operators[region]
                a,b=value_vector(np.zeros(ev.n_ops)),value_vector(np.zeros(ev.n_ops))
                self.physics.acc_flux_itor[region].evaluate(value_vector(states[cell]),a)
                self.physics.acc_flux_w_itor.evaluate(value_vector(states[well_cell]),b)
                res,wel=np.asarray(a),np.asarray(b)
                for phase in range(2):
                    dp=states[well_cell,0]-states[cell,0]+.5*(res[ev.GRAV_OP+phase]+wel[ev.GRAV_OP+phase])*gravity[conn]-wel[ev.PC_OP+phase]+res[ev.PC_OP+phase]
                    up=res if dp<0 else wel
                    q=-tran[conn]*dp*up[ev.LAMBDA_OP+phase]
                    phase_volume[phase]+=q
                    start=ev.FLUX_OP+4*phase
                    phase_component[phase]+=q*up[start:start+4]
        rates=phase_component.sum(axis=0)
        return {"component_kmol_day":rates.tolist(),"component_kg_day":(rates*MW).tolist(),
            "gas_component_kmol_day":phase_component[0].tolist(),
            "phase_component_kg_day":(phase_component*MW).tolist(),"phase_m3_day":phase_volume.tolist(),
            "hydrogen_kg_day":float(rates[0]*MW[0]),"gas_phase_kg_day":float(np.dot(phase_component[0],MW)),
            "water_phase_kg_day":float(np.dot(phase_component[1],MW))}
