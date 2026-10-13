"""Versioned three-case configuration, capability checks and reproducible outputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
from .core import AtomCloud, concatenate, direct_iq, upstream_kernel_iq, validate_q
from . import __version__
from .directional import amplitude_components, cell_shape_amplitude, angular_grid, validate_q_vectors
from .distributions import ParticleLibrary, PorePopulation
from .ensemble import analytic_ensemble
from .lattice import LatticePairCache
from .moments import guest_moments, finite_ensemble_intensity
from .periodic import (validate_cell, reciprocal_lattice, bragg_reflections,
                       cell_guest_moments, powder_shells, validate_periodic_population)

CASES = {'pxrd', 'single_crystal_finite', 'single_crystal_periodic'}
BACKENDS = {'reference', 'adapted_kernel', 'installed_package', 'lattice_grouped'}


def _keys(mapping, allowed, label):
    if not isinstance(mapping, dict):
        raise ValueError(f'{label} must be an object.')
    extra = set(mapping)-set(allowed)
    if extra:
        raise ValueError(f'Unsupported {label} fields: {sorted(extra)}')


def validate_config(config):
    _keys(config, ('schema_version','description','scattering_case','boundary','guest_estimator',
        'backend','device','host','pores','particles','population','sampling','statistics','angular'), 'configuration')
    if config.get('schema_version') not in ('1.0','1.1'):
        raise ValueError('schema_version must be "1.0" or "1.1".')
    case, boundary = config.get('scattering_case'), config.get('boundary')
    if case not in CASES or boundary not in ('finite','bulk'):
        raise ValueError('Explicit scattering_case and finite/bulk boundary are required.')
    if (case == 'single_crystal_finite' and boundary != 'finite') or (case == 'single_crystal_periodic' and boundary != 'bulk'):
        raise ValueError('Single-crystal case and boundary are incompatible.')
    estimator, backend = config.get('guest_estimator'), config.get('backend')
    if estimator not in ('explicit','analytic','moments') or backend not in BACKENDS:
        raise ValueError('Select a supported guest_estimator and backend explicitly.')
    if boundary == 'bulk' and (backend != 'reference' or estimator == 'explicit'):
        raise ValueError('Bulk supports reference amplitude moments; one repeated explicit random cell is unsupported.')
    if case != 'pxrd' and backend not in ('reference','lattice_grouped'):
        raise ValueError('Scalar Debye backends cannot evaluate directional scattering.')
    if estimator == 'moments' and backend != 'reference':
        raise ValueError('Moment angular/directional evaluation currently uses the reference backend.')
    if estimator == 'analytic' and backend == 'installed_package':
        raise ValueError('Installed native factors cannot be mixed with Gemmi analytical guest factors.')
    if config.get('device','cpu') != 'cpu' and backend not in ('adapted_kernel','installed_package'):
        raise ValueError('This backend executes on CPU; a CUDA label would be incorrect.')
    if config.get('device','cpu') not in ('cpu','cuda'):
        raise ValueError('device must be cpu or cuda.')
    host = config.get('host',{})
    _keys(host, ('xyz','cif','cif_options','cell_vectors_A','cell_mask','atom_cut','guest_retention'), 'host')
    if len(set(host)&{'xyz','cif'})!=1 or not isinstance(host.get('xyz',host.get('cif')),str):
        raise ValueError('Specify exactly one host.xyz or host.cif file.')
    if config['schema_version']=='1.0' and set(host)&{'cif','cif_options','atom_cut','guest_retention'}:
        raise ValueError('CIF and surface inputs require schema_version "1.1".')
    if 'cif_options' in host:
        if 'cif' not in host:
            raise ValueError('cif_options requires host.cif.')
        _keys(host['cif_options'],('block','occupancy_policy','exclude_labels','exclude_elements','duplicate_tolerance_A','duplicate_policy'),'cif_options')
    if 'cif' in host and 'cell_vectors_A' in host:
        raise ValueError('CIF cell vectors are read from the source; a lattice override is unsupported.')
    has_cell='cell_vectors_A' in host or 'cif' in host
    if boundary == 'bulk' and (not has_cell or set(host)&{'cell_mask','atom_cut','guest_retention'}):
        raise ValueError('Bulk requires a cell basis/CIF, with no finite cell_mask, atom_cut or guest_retention.')
    if 'cell_mask' in host or backend == 'lattice_grouped':
        if not has_cell or ('cell_mask' not in host and 'cif' not in host):
            raise ValueError('Full-cell shape evaluation requires a CIF or cell_vectors_A and cell_mask.')
        mask=np.asarray(host.get('cell_mask',[[[1]]]))
        if mask.ndim!=3 or not np.all(np.isin(mask,[0,1])) or not mask.any():
            raise ValueError('cell_mask must be a nonempty 3D binary mask.')
    if 'atom_cut' in host and backend=='lattice_grouped':
        raise ValueError('Atom cuts break complete-cell factorization; select reference or an explicit-coordinate backend.')
    retention=host.get('guest_retention','site')
    if retention not in ('site','centre','whole_particle'):
        raise ValueError('guest_retention must be site, centre or whole_particle.')
    if 'atom_cut' in host and config.get('particles') and retention=='site':
        raise ValueError('Atom-cut guests require an explicit centre or whole_particle guest_retention rule.')
    if retention!='site' and 'atom_cut' not in host and 'cell_mask' not in host and 'cif' not in host:
        raise ValueError('Guest containment requires a cell mask/CIF or atom_cut external boundary.')
    if case == 'single_crystal_finite' and backend == 'lattice_grouped' and config.get('particles'):
        raise ValueError('Directional cell factorization currently supports a deterministic host only.')
    _keys(config.get('statistics',{}),('n_samples','seed','moment_method'),'statistics')
    stats=config.get('statistics',{})
    if stats.get('moment_method','sampled') not in ('analytic','sampled'):
        raise ValueError('moment_method must be analytic or sampled.')
    n=stats.get('n_samples',512)
    if isinstance(n,bool) or not isinstance(n,int) or n<2:
        raise ValueError('n_samples must be an integer >= 2.')
    seed=stats.get('seed',20261013)
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('seed must be a nonnegative integer.')
    _keys(config.get('angular',{}),('n_mu','n_phi'),'angular')
    sampling=config.get('sampling',{})
    _keys(sampling,('q','q_vectors','hkl','reflection_q_max','reflection_q_min'),'sampling')
    if case=='pxrd':
        if 'q' not in sampling or set(sampling)-{'q'}:
            raise ValueError('PXRD requires scalar sampling.q only.')
    elif case=='single_crystal_finite':
        if len(set(sampling)&{'q_vectors','hkl'})!=1 or set(sampling)-{'q_vectors','hkl'}:
            raise ValueError('Finite single crystal requires exactly q_vectors or hkl, never scalar q.')
    else:
        if 'reflection_q_max' not in sampling or 'q' in sampling or ('q_vectors' in sampling and 'hkl' in sampling):
            raise ValueError('Periodic single crystal requires reflection_q_max and optional directional diffuse samples.')
    if 'hkl' in sampling and not has_cell:
        raise ValueError('hkl sampling requires cell_vectors_A or CIF.')
    return config


def _scalar_q(spec):
    if isinstance(spec,dict):
        _keys(spec,('min','max','points'),'q')
        points=spec.get('points')
        if isinstance(points,bool) or not isinstance(points,int) or points<1:
            raise ValueError('q.points must be a positive integer.')
        if not 0<=spec.get('min',-1)<=spec.get('max',-1):
            raise ValueError('Require 0 <= q.min <= q.max.')
        return validate_q(np.linspace(spec['min'],spec['max'],points))
    return validate_q(spec)


def _inputs(config, root):
    hashes={}
    def read_xyz(name):
        path=(root/name).resolve()
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return AtomCloud.read_xyz(path)
    settings=config['host']
    crystal=None
    processing={}
    if 'cif' in settings:
        from .crystal import read_cif
        path=(root/settings['cif']).resolve()
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        crystal=read_cif(path,**settings.get('cif_options',{}))
        basis=crystal.basis
        cell=crystal.cell_vectors_A
        processing['crystallography']=crystal.metadata
    else:
        basis=read_xyz(settings['xyz'])
        cell=validate_cell(settings['cell_vectors_A']) if 'cell_vectors_A' in settings else None
    mask=np.asarray(settings['cell_mask'],dtype=int) if 'cell_mask' in settings else None
    if crystal is not None and config['boundary']=='finite' and mask is None:
        mask=np.ones((1,1,1),dtype=int)
    shifts=np.argwhere(mask)@cell if mask is not None else np.zeros((1,3))
    particles=config.get('particles',[])
    lazy_shape=mask is not None and config['backend']=='lattice_grouped' and not particles
    host=concatenate([basis.moved(shift) for shift in shifts]) if mask is not None and not lazy_shape else basis
    processing.update(cell_vectors_A=None if cell is None else cell.tolist(),basis_atoms=len(basis),
        retained_cells=int(mask.sum()) if mask is not None else None,
        host_atoms=len(basis)*int(mask.sum()) if lazy_shape else len(host),host_atoms_materialized=not lazy_shape)
    external_boundary=None
    if mask is not None:
        external_boundary={'kind':'cell_mask','cell_vectors_A':cell.tolist(),'mask':mask.tolist()}
    if 'atom_cut' in settings:
        from .geometry import apply_atom_cut
        host,cut_metadata=apply_atom_cut(host,settings['atom_cut'])
        processing['atom_cut']=cut_metadata
        processing['host_atoms']=len(host)
        external_boundary={'kind':'intersection','boundaries':[external_boundary,settings['atom_cut']]} if external_boundary else settings['atom_cut']
    retention=settings.get('guest_retention','site')
    processing['guest_retention']=retention
    processing['surface_policy']='unpassivated_atom_cut' if 'atom_cut' in settings else ('complete cells' if mask is not None else 'explicit supplied atoms')
    if not particles:
        if config.get('pores') or config.get('population'):
            raise ValueError('Pore/population fields require particle templates.')
        return host,basis,cell,mask,None,hashes,processing
    for particle in particles:
        _keys(particle,('xyz','weight'),'particle')
    library=ParticleLibrary(tuple(read_xyz(p['xyz']) for p in particles),[p.get('weight',1.) for p in particles])
    pores=config.get('pores',{})
    _keys(pores,('centres_A','centres_fractional','expand_symmetry'),'pores')
    coordinate_keys=set(pores)&{'centres_A','centres_fractional'}
    if len(coordinate_keys)!=1:
        raise ValueError('Specify exactly one pore coordinate frame: centres_A or centres_fractional.')
    key=next(iter(coordinate_keys))
    centres=np.asarray(pores[key],dtype=float)
    if centres.ndim!=2 or centres.shape[1]!=3 or len(centres)==0 or not np.isfinite(centres).all():
        raise ValueError('Pore centres must be a nonempty K-by-3 array.')
    processing['input_pore_sites']=len(centres)
    if 'centres_fractional' in pores:
        if cell is None:
            raise ValueError('Fractional pore centres require cell_vectors_A.')
        centres=centres@cell
    params=dict(config.get('population',{}))
    allowed=('occupancy','displacement_sigma_A','displacement_covariance_A2','mean_displacement_A',
        'position_mode','uniform_ball_radius_A','orientation','reference_rotation','wobble_std_deg',
        'pore_radius_A','wall_clearance_A','min_host_distance_A','max_attempts')
    _keys(params,allowed,'population')
    if not isinstance(pores.get('expand_symmetry',False),bool):
        raise ValueError('pores.expand_symmetry must be a boolean.')
    if 'expand_symmetry' in pores and config['schema_version']=='1.0':
        raise ValueError('Pore symmetry expansion requires schema_version "1.1".')
    if pores.get('expand_symmetry',False):
        if crystal is None:
            raise ValueError('Pore symmetry expansion requires host.cif symmetry operations.')
        for name in ('occupancy','pore_radius_A'):
            if name in params and params[name] is not None and np.asarray(params[name]).ndim!=0:
                raise ValueError('Symmetry-expanded pores require shared scalar occupancy/radius; supply fully expanded sites for per-site arrays.')
        if np.asarray(params.get('mean_displacement_A',[0,0,0])).shape!=(3,):
            raise ValueError('Symmetry-expanded pores require one shared mean displacement vector.')
        from .crystal import expand_fractional_sites
        fractional=centres@np.linalg.inv(cell)
        centres=expand_fractional_sites(fractional,cell,crystal.symops)@cell
    if config['schema_version']=='1.1':
        if cell is not None and (config['boundary']=='bulk' or mask is not None):
            from .crystal import expand_fractional_sites
            unique=expand_fractional_sites(centres@np.linalg.inv(cell),cell,('x,y,z',))
            if len(unique)!=len(centres):
                raise ValueError('Duplicate periodic-equivalent pore centres would populate one site more than once.')
        elif len(centres)>1:
            from scipy.spatial import cKDTree
            distances,_=cKDTree(centres).query(centres,k=2)
            if np.any(distances[:,1]<=1e-5):
                raise ValueError('Duplicate pore centres would populate one site more than once.')
    processing['pore_sites_per_cell']=len(centres)
    processing['pore_symmetry_expanded']=pores.get('expand_symmetry',False)
    if mask is not None:
        k=len(centres)
        for key in ('occupancy','pore_radius_A'):
            if key in params and params[key] is not None:
                params[key]=np.tile(np.broadcast_to(np.asarray(params[key]),(k,)),len(shifts))
        if 'mean_displacement_A' in params:
            params['mean_displacement_A']=np.tile(np.broadcast_to(params['mean_displacement_A'],(k,3)),(len(shifts),1))
        centres=np.concatenate([centres+s for s in shifts])
    processing['pore_sites_before_surface_filter']=len(centres)
    if retention!='site':
        from .geometry import contains_points
        keep=contains_points(centres,external_boundary)
        for name in ('occupancy','pore_radius_A'):
            if name in params and params[name] is not None:
                params[name]=np.broadcast_to(np.asarray(params[name]),(len(centres),))[keep]
        if 'mean_displacement_A' in params:
            params['mean_displacement_A']=np.broadcast_to(params['mean_displacement_A'],(len(centres),3))[keep]
        centres=centres[keep]
    processing['retained_pore_sites']=len(centres)
    processing['pore_centres_A']=centres.tolist()
    processing['external_containment']='nominal sites filtered, then placement conditioned at fixed size' if retention!='site' else 'retain whole guests belonging to retained cell sites; no external-boundary clipping'
    if not len(centres):
        return host,basis,cell,mask,None,hashes,processing
    if config['boundary']=='bulk' and params.get('min_host_distance_A') is not None and config['schema_version']=='1.0':
        raise ValueError('Periodic host clearance requires schema_version "1.1" for periodic neighbor handling.')
    population=PorePopulation(centres,library,host=host,
        periodic_cell_A=cell if config['boundary']=='bulk' else None,
        external_boundary=external_boundary if retention!='site' else None,guest_retention=retention,**params)
    if config['boundary']=='bulk':
        validate_periodic_population(population,cell)
    if population._host_tree is not None and getattr(population._host_tree,'is_periodic',False):
        processing['host_clearance']=population._host_tree.metadata
    return host,basis,cell,mask,population,hashes,processing


def environment_record():
    import torch
    versions={}
    for name in ('numpy','scipy','gemmi','torch','pytest','debyecalculator','mof-debye-lab'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    venv_config=Path(sys.prefix)/'pyvenv.cfg'
    venv_settings={}
    if venv_config.is_file():
        for line in venv_config.read_text(encoding='utf-8').splitlines():
            if '=' in line:
                key,value=line.split('=',1)
                venv_settings[key.strip()]=value.strip()
    return {'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
        'versions':versions,'cuda_available':torch.cuda.is_available(),'torch_threads':torch.get_num_threads(),
        'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,
        'venv_settings':venv_settings}


def run_configuration(config, root=Path('.')):
    """Return tables and JSON-compatible metadata; perform no output-file mutation."""
    validate_config(config)
    stamp=time.perf_counter()
    host,basis,cell,mask,population,hashes,processing=_inputs(config,Path(root))
    case,boundary=config['scattering_case'],config['boundary']
    estimator,backend=config['guest_estimator'],config['backend']
    stats=config.get('statistics',{}); seed=stats.get('seed',20261013); samples=stats.get('n_samples',512)
    method='analytic' if estimator=='analytic' else stats.get('moment_method','sampled')
    angular=config.get('angular',{}); sampling=config['sampling']; actual=[]; tables={}; extra={}
    metadata={'schema_version':config['schema_version'],'software_version':__version__,'config':config,'scattering_case':case,
        'boundary':boundary,'guest_estimator':estimator,'backend_requested':backend,
        'device_actual':config.get('device','cpu'),'seed':seed,'input_sha256':hashes,
        'units':{'positions':'angstrom','q':'inverse angstrom','finite_intensity':'electron^2',
            'diffuse_per_cell':'electron^2','bragg_weight':'electron^2 angstrom^-3',
            'powder_radial_weight':'electron^2 angstrom^-1'},
        'form_factors':'neutral Gemmi IT92','output_statistic':'realization' if estimator=='explicit' else 'ensemble_mean',
        'physical_assumptions':['kinematic elastic scattering','fixed host','independent pores',
            'independent identically distributed cells for bulk','rigid guests','number-weighted template probabilities'],
        'instrument_corrections':False,'surface_policy':processing['surface_policy'],'input_processing':processing,
        'uncertainty':'Numerical mean-amplitude variance is reported separately from physical disorder variance. No intensity SEM is claimed; refine angular grid and sample count.',
        'sampled_moment_policy':'Unbiased variance and mean-square correction; finite-sample negative estimates are retained.'}
    def moments_at(vectors):
        gm=guest_moments(population,vectors,method=method,n_samples=samples,seed=seed)
        metadata['moment_metadata']=gm.metadata
        return gm
    def vectors_at():
        if 'hkl' in sampling:
            return validate_q_vectors(sampling['hkl'])@reciprocal_lattice(cell)
        return validate_q_vectors(sampling['q_vectors'])
    def powder_backend(cloud,q):
        if backend=='reference':
            actual.append('numpy_scipy_direct_debye')
            return direct_iq(cloud,q)
        if backend=='adapted_kernel':
            values,info=upstream_kernel_iq(cloud,q,device=config.get('device','cpu'),return_info=True)
            actual.append(info); return values
        if backend=='installed_package':
            from .installed import installed_iq
            values,info=installed_iq(cloud,q,device=config.get('device','cpu'),return_info=True)
            metadata['form_factors']='installed package native; see actual_method metadata'
            actual.append(info); return values
        if cloud is not host:
            raise ValueError('Full-cell lattice grouping cannot represent a sampled guest realization.')
        cache=LatticePairCache(basis,cell,mask)
        values=cache.iq(q)
        actual.append({'method':'linear_fft_full_cell_debye',**cache.metadata})
        return values
    def powder_moments(q, bulk=False):
        directions,weights=angular_grid(**angular)
        names=('diffuse','angular_mean_amplitude_variance') if bulk else ('total','diffuse','angular_mean_amplitude_variance')
        columns={name:np.zeros(len(q)) for name in names}
        for start in range(0,len(q),8):
            scalar=q[start:start+8]
            vectors=(scalar[:,None,None]*directions[None,:,:]).reshape(-1,3)
            gm=cell_guest_moments(population,vectors,method,samples,seed,cell) if bulk else moments_at(vectors)
            values={'diffuse':gm.variance,'angular_mean_amplitude_variance':gm.mean_variance}
            if not bulk:
                values['total']=finite_ensemble_intensity(host,vectors,gm)['total']
            for name in names:
                columns[name][start:start+len(scalar)]=values[name].reshape(len(scalar),-1)@weights
        return columns
    if boundary=='bulk':
        q=_scalar_q(sampling['q']) if case=='pxrd' else None
        qmax=float(q.max()) if q is not None else sampling['reflection_q_max']
        qmin=0. if q is not None else sampling.get('reflection_q_min',0.)
        reflections=bragg_reflections(host,cell,qmax,qmin,population,method,samples,seed)
        tables['reflections']={**{k:reflections['hkl'][:,i] for i,k in enumerate(('h','k','l'))},
            **{k:reflections['q_vectors'][:,i] for i,k in enumerate(('qx','qy','qz'))},
            **{k:reflections[k] for k in ('q','f_squared','bragg_weight','diffuse_at_reflections','mean_variance')}}
        extra={'forward_weight':reflections['forward_weight'],'bragg_measure_prefactor':reflections['measure_prefactor'],
            'bragg_convention':'Per-cell weight multiplying delta^3(q-G), includes (2*pi)^3/V. Never a peak height.',
            'moment_metadata':reflections['moment_metadata']}
        if case=='pxrd':
            tables['powder_shells']=powder_shells(reflections)
            tables['diffuse']={'q':q,**powder_moments(q,bulk=True)}
            extra['powder_convention']='Radial density weight multiplying delta(q-|G|); sum over all G once / (4*pi*|G|^2). Forward weight separate.'
        else:
            vectors=vectors_at() if 'q_vectors' in sampling or 'hkl' in sampling else np.zeros((1,3))
            gm=cell_guest_moments(population,vectors,method,samples,seed,cell)
            tables['diffuse']={**{k:vectors[:,i] for i,k in enumerate(('qx','qy','qz'))},
                'diffuse':gm.variance,'mean_amplitude_variance':gm.mean_variance}
        actual.append('directional_cell_moments_and_reciprocal_enumeration')
    elif case=='single_crystal_finite':
        vectors=vectors_at()
        if population is not None and estimator!='explicit':
            result=finite_ensemble_intensity(host,vectors,moments_at(vectors))
            parts={k:np.asarray(v) for k,v in result.items() if isinstance(v,np.ndarray) and v.shape==(len(vectors),) and not np.iscomplexobj(v)}
            actual.append('tiled_directional_independent_pore_moments')
        elif backend=='lattice_grouped':
            a=cell_shape_amplitude(basis,cell,mask,vectors)
            parts={'total':abs(a)**2,'amplitude_real':a.real,'amplitude_imag':a.imag}
            actual.append('full_cell_amplitude_factorization')
        else:
            guests=[] if population is None else [d.cloud for d in population.sample(np.random.default_rng(seed))]
            result=amplitude_components(host,guests,vectors)
            parts={k:np.asarray(v) for k,v in result.items() if isinstance(v,np.ndarray) and v.shape==(len(vectors),) and not np.iscomplexobj(v)}
            actual.append('tiled_directional_amplitude')
        tables['intensity']={**{k:vectors[:,i] for i,k in enumerate(('qx','qy','qz'))},**parts}
    else:
        q=_scalar_q(sampling['q'])
        if population is None:
            parts={'total':powder_backend(host,q)}
        elif estimator=='explicit':
            full=concatenate([host]+[d.cloud for d in population.sample(np.random.default_rng(seed))])
            parts={'total':powder_backend(full,q)}
        elif estimator=='analytic':
            population.analytic_sigma()
            parts=analytic_ensemble(host,population,q,host_iq=powder_backend(host,q))
        else:
            parts=powder_moments(q)
            actual.append('angular_average_of_directional_moments')
        tables['intensity']={'q':q,**parts}
    metadata.update(extra)
    metadata.update(actual_method=actual,seconds=time.perf_counter()-stamp,environment=environment_record())
    return {'tables':tables,'metadata':metadata}


def write_result(result, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for name,table in result['tables'].items():
        np.savetxt(output/f'{name}.csv',np.column_stack(list(table.values())),delimiter=',',
            header=','.join(table),comments='')
    (output/'metadata.json').write_text(json.dumps(result['metadata'],indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    result=run_configuration(config,args.config.resolve().parent)
    write_result(result,args.out)
    print(json.dumps({'output':str(args.out.resolve()),'tables':list(result['tables']),
        'seconds':result['metadata']['seconds'],'actual_method':result['metadata']['actual_method']},indent=2))
