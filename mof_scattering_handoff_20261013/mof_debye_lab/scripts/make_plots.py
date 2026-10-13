"""Plot saved numerical results using Matplotlib defaults, with separate figures."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'

def make_plots(show=True):
    rows=np.genfromtxt(OUT/'benchmark.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8.6,4.8))
    for method,label in [('upstream_direct_float64','Direct sum, float64'),
                         ('upstream_grid_float64','Grid kernel, float64'),
                         ('upstream_grid_float32','Grid kernel, float32')]:
        r=rows[rows['method']==method]
        ax.plot(r['n_atoms'],r['seconds_median'],marker='o',label=label)
    ax.set_yscale('log');ax.set_xticks([192,648,1536])
    ax.set_xlabel('Number of atoms in synthetic finite host')
    ax.set_ylabel('Runtime (seconds; logarithmic axis)')
    ax.set_title('Adapted DebyeCalculator kernels: CPU, 301 q points',pad=14)
    ax.legend();ax.grid(True,alpha=.25)
    fig.text(.5,.015,'One CPU thread. Median of 3 runs. This is not a full-package or GPU benchmark.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'benchmark_runtime.png',dpi=180)
    fig.savefig(OUT/'benchmark_runtime.svg')
    if show:plt.show()
    else:plt.close(fig)

    d=np.genfromtxt(OUT/'ensemble_validation.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(8.6,4.8))
    ax.plot(d['q_Ainv'],d['analytic_total'],label='Analytical independent-pore average')
    ax.plot(d['q_Ainv'][::5],d['mc_mean_total'][::5],linestyle='none',marker='o',markersize=3.5,
            label='Mean of 8,192 explicit configurations')
    ax.fill_between(d['q_Ainv'],d['mc_mean_total']-3*d['mc_sem_total'],
                     d['mc_mean_total']+3*d['mc_sem_total'],alpha=.18,label='Monte Carlo mean ± 3 SEM')
    ax.plot(d['q_Ainv'],d['analytic_host']+d['analytic_guest_internal'],linestyle='--',
            label='Host + isolated particles only (cross terms omitted)')
    ax.set_yscale('log');ax.set_xlabel(r'$q$ ($\mathrm{\AA}^{-1}$)')
    ax.set_ylabel(r'Full Debye intensity (electron$^2$; logarithmic axis)')
    ax.set_title('Nanoparticles in pores: analytical vs explicit Debye average',pad=14)
    ax.legend(fontsize=8,loc='upper right');ax.grid(True,alpha=.2)
    fig.text(.5,.015,'Synthetic host; 13/19-atom Au templates; independent pores; uniform rotations; Gaussian centre jitter.',ha='center',fontsize=8.3)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'ensemble_validation.png',dpi=180)
    fig.savefig(OUT/'ensemble_validation.svg')
    if show:plt.show()
    else:plt.close(fig)

    records=np.genfromtxt(OUT/'bounded_placements.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(8.6,4.8))
    bins=np.linspace(-2.2,4.0,45)
    for count in (13,19,43):
        selected=records['n_atoms']==count
        ax.hist(records['dx_A'][selected],bins=bins,density=True,histtype='step',linewidth=1.8,
                label=f'{count}-atom template (n={int(selected.sum())})')
    ax.set_xlabel(r'Particle centre displacement $\Delta x$ ($\mathrm{\AA}$)')
    ax.set_ylabel('Conditional probability density')
    ax.set_title('Pore confinement couples particle size and centre position',pad=14)
    ax.legend();ax.grid(True,alpha=.2)
    fig.text(.5,.015,'Same proposed Gaussian centre distribution, conditioned on each rigid template fitting inside the pore.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'bounded_displacement.png',dpi=180)
    fig.savefig(OUT/'bounded_displacement.svg');plt.close(fig)

if __name__=='__main__':make_plots()
