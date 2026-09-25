"""Plot assumed constitutive curves and the attributed carbonate analogue."""
from pathlib import Path
import numpy as np
try:
    from .properties import load_config,RelativePermeability,CapillaryPressure
except ImportError:
    from properties import load_config,RelativePermeability,CapillaryPressure


def plot_properties(output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(__file__).resolve().parent
    output=Path(output or folder/'output');output.mkdir(parents=True,exist_ok=True)
    cfg=load_config();swr,sgr=cfg['residual_water_saturation'],cfg['residual_gas_saturation']
    gas=RelativePermeability(sgr,swr,cfg['gas_relperm_exponent'])
    water=RelativePermeability(swr,sgr,cfg['water_relperm_exponent'])
    pc=CapillaryPressure(cfg);sg=np.linspace(0,1,400)
    reference=np.genfromtxt(folder/'reference_data'/'rezaei_2022_carbonate_drainage_fit.csv',delimiter=',',names=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,model,name,title in [(axes[0],gas,'gas_relative_permeability','Gas relative permeability'),(axes[1],water,'water_relative_permeability','Water relative permeability')]:
        values=[model.evaluate(s if name.startswith('gas') else 1-s) for s in sg]
        ax.plot(sg,values,label='Assumed model curve')
        ax.plot(reference['gas_saturation_fraction'],reference[name],ls='--',label='Carbonate laboratory fit')
        ax.set(xlabel='Gas saturation',ylabel=title,xlim=(0,1),ylim=(0,1.02));ax.grid(alpha=.2)
    axes[0].plot(sg,.22*np.array([gas.evaluate(s) for s in sg]),ls=':',label='Lower endpoint sensitivity')
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    axes[2].plot(sg,[pc.evaluate([s,1-s])[1] for s in sg]);axes[2].set(xlabel='Gas saturation',ylabel='Assumed gas–water capillary pressure (bar)',xlim=(0,1));axes[2].grid(alpha=.2)
    fig.suptitle('Assumed shallow-reservoir curves and a carbonate analogue\nLaboratory fit: Rezaei et al. (2022), 206.8 bar, 353.15 K, 35 ppt brine; not Mali calibration',fontsize=11)
    fig.savefig(output/'constitutive_curves.png',dpi=170);plt.close(fig)
    return output/'constitutive_curves.png'

if __name__=='__main__':print(plot_properties())
