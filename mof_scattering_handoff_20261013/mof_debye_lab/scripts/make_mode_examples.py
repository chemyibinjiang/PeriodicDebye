"""Generate versioned examples from one deliberately synthetic cell/guest model."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np


def main():
    output=Path(__file__).resolve().parents[1]/'examples'/'modes'
    output.mkdir(exist_ok=True)
    (output/'host_cell.xyz').write_text('4\nSynthetic C/O/Zn basis; not a validated MOF; angstrom\nC 0.3 0.4 0.2\nO 4.0 1.0 2.0\nZn 2.0 5.0 3.0\nC 7.0 8.0 9.0\n',encoding='utf-8')
    cell=[[12.,0.,0.],[1.,11.,0.],[.5,1.5,13.]]
    base={'schema_version':'1.0','description':'Synthetic shared triclinic C/O/Zn cell and illustrative Au guest; no material fit.',
        'scattering_case':'single_crystal_periodic','boundary':'bulk','guest_estimator':'analytic','backend':'reference',
        'device':'cpu','host':{'xyz':'host_cell.xyz','cell_vectors_A':cell},
        'pores':{'centres_fractional':[[.5,.5,.5]]},'particles':[{'xyz':'../Au_13.xyz','weight':.65},{'xyz':'../Au_19.xyz','weight':.35}],
        'population':{'occupancy':.6,'displacement_sigma_A':.25,'orientation':'uniform'},
        'statistics':{'seed':20261013,'n_samples':512},
        'sampling':{'reflection_q_max':2.,'q_vectors':[[float(q),0.,0.] for q in np.linspace(0,2,81)]}}
    configs={'periodic_single_crystal':deepcopy(base)}
    finite=deepcopy(base)
    finite.update(scattering_case='single_crystal_finite',boundary='finite')
    finite['host']['cell_mask']=np.ones((3,2,2),dtype=int).tolist()
    finite['sampling']={'q_vectors':[[float(q),.1,.05] for q in np.linspace(0,2,121)]}
    configs['finite_single_crystal']=finite
    powder=deepcopy(finite)
    powder.update(scattering_case='pxrd',backend='lattice_grouped')
    powder['sampling']={'q':{'min':0.,'max':2.,'points':101}}
    configs['pxrd_finite']=powder
    bulk=deepcopy(base)
    bulk['scattering_case']='pxrd'
    bulk['sampling']={'q':{'min':0.,'max':2.,'points':81}}
    bulk['angular']={'n_mu':12,'n_phi':24}
    configs['pxrd_bulk']=bulk
    bounded=deepcopy(finite)
    bounded['host'].pop('cell_mask')
    bounded['guest_estimator']='moments'
    bounded['population']={'occupancy':.7,'displacement_covariance_A2':[[.25,0,0],[0,.09,0],[0,0,.09]],
        'orientation':'wobble','wobble_std_deg':15.,'pore_radius_A':4.5,'wall_clearance_A':.1}
    bounded['statistics']['n_samples']=512
    bounded['statistics']['moment_method']='sampled'
    configs['finite_bounded_moments']=bounded
    for name,config in configs.items():
        (output/f'{name}.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(configs)} configurations in {output}')


if __name__=='__main__':
    main()
