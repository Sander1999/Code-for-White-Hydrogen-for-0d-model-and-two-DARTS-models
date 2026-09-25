"""Scientific views and portable VTK export of DARTS cell-centred 3D results."""
from pathlib import Path
import numpy as np


def geometry(cfg):
    nx,ny,nz = cfg['nx'],cfg['ny'],cfg.get('nz',1)
    top = cfg.get('top_depth_m',cfg.get('depth_m',0.)-cfg['thickness_m']/2)
    dx,dy,dz=cfg['length_x_m']/nx,cfg['length_y_m']/ny,cfg['thickness_m']/nz
    k,j,i=np.indices((nz,ny,nx))
    return np.column_stack(((i.ravel()+.5)*dx,(j.ravel()+.5)*dy,top+(k.ravel()+.5)*dz)),(nx,ny,nz),top


def export_vtk(path,cfg,fields):
    """Legacy VTK rectilinear-grid cell data, ordered x fastest; metres/bar/fractions."""
    _,(nx,ny,nz),top=geometry(cfg)
    with Path(path).open('w') as f:
        f.write('# vtk DataFile Version 3.0\nHydrogen reservoir; coordinates in metres, z positive down\nASCII\nDATASET RECTILINEAR_GRID\n')
        f.write(f'DIMENSIONS {nx+1} {ny+1} {nz+1}\n')
        for axis,values in [('X',np.linspace(0,cfg['length_x_m'],nx+1)),('Y',np.linspace(0,cfg['length_y_m'],ny+1)),('Z',np.linspace(top,top+cfg['thickness_m'],nz+1))]:
            f.write(f'{axis}_COORDINATES {len(values)} double\n')
            f.write(' '.join(f'{v:.12g}' for v in values)+'\n')
        f.write(f'CELL_DATA {nx*ny*nz}\n')
        for name,values in fields.items():
            values=np.asarray(values).ravel()
            if values.size != nx*ny*nz or not np.isfinite(values).all():
                raise ValueError(f'Invalid VTK field {name}')
            f.write(f'SCALARS {name} double 1\nLOOKUP_TABLE default\n')
            f.write('\n'.join(f'{v:.12g}' for v in values)+'\n')


def plot_reservoir_3d(path,cfg,pressure,gas_saturation,*,title='Hydrogen reservoir',day=0):
    """Orthogonal cell slices in a 3D box; no interpolation or fabricated geometry."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz,(nx,ny,nz),top=geometry(cfg)
    pressure=np.asarray(pressure).ravel();sg=np.asarray(gas_saturation).ravel()
    if pressure.size!=len(xyz) or sg.size!=len(xyz):raise ValueError('Grid/field size mismatch')
    k,j,i=np.indices((nz,ny,nx))
    visible=((i==nx//2)|(j==ny//2)|(k==0)).ravel()
    fig=plt.figure(figsize=(12,5.6),layout='constrained')
    for panel,values,label,cmap in [(1,pressure,'Pressure (bar absolute)','viridis'),(2,sg,'Gas saturation (fraction)','cividis')]:
        ax=fig.add_subplot(1,2,panel,projection='3d',computed_zorder=False)
        args={'vmin':0,'vmax':1} if panel==2 else {}
        points=ax.scatter(*xyz[visible].T,c=values[visible],s=max(12,1800/max(nx,ny)),marker='s',cmap=cmap,depthshade=False,zorder=2,**args)
        xmax,ymax,bottom=cfg['length_x_m'],cfg['length_y_m'],top+cfg['thickness_m']
        for x in [0,xmax]:
            for y in [0,ymax]:ax.plot([x,x],[y,y],[top,bottom],c='.6',lw=.6,zorder=1)
        for z in [top,bottom]:
            ax.plot([0,xmax,xmax,0,0],[0,0,ymax,ymax,0],[z]*5,c='.6',lw=.6,zorder=1)
        if cfg.get('producer_enabled',True):
            completion=cfg.get('producer_depth_m',cfg.get('depth_m',(top+bottom)/2))
            completion_top=cfg.get('producer_completion_top_m',top+.5*cfg['thickness_m']/nz)
            completion_bottom=cfg.get('producer_completion_bottom_m',bottom-.5*cfg['thickness_m']/nz)
            ax.plot([xmax/2]*2,[ymax/2]*2,[top-0.12*cfg['thickness_m'],completion_bottom],c='crimson',lw=2,label='Producer',zorder=8)
            ax.plot([xmax/2]*2,[ymax/2]*2,[completion_top,completion_bottom],c='crimson',lw=5,zorder=9)
            ax.scatter([xmax/2],[ymax/2],[completion],c='crimson',marker='*',s=60,zorder=9)
            ax.legend(loc='upper left',fontsize=8)
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='Depth (m)',zlim=(bottom,top-.12*cfg['thickness_m']))
        ax.set_box_aspect((1,1,.65));ax.view_init(elev=23,azim=-58)
        fig.colorbar(points,ax=ax,shrink=.6,pad=.06,label=label)
    fig.suptitle(f'{title} at day {day:g}\nCell-centred orthogonal slices; vertical scale expanded',fontsize=12)
    fig.savefig(path,dpi=180);plt.close(fig)
