#!/usr/bin/env python3
"""Generate synthetic geometries and check scattering against known physical laws.

No timing or performance conclusions: this is an executable correctness gallery.
Every reference below is a closed-form physical equation, not a second call to
the production scattering engine. Neutral IT92 atomic tables are shared inputs.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

import gemmi
import numpy as np

from mofdebye.core import AtomCloud
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.moments import guest_moments, finite_ensemble_intensity
from mofdebye.runner import run_configuration, write_result, environment_record


SEED = 20261014


def factor(element, q):
    """Evaluate the published table independently, one scalar at a time."""
    table = gemmi.Element(element).it92
    return np.array([sum(float(a) * np.exp(-float(b) * (float(x) / (4*np.pi))**2)
                         for a, b in zip(table.a, table.b)) + float(table.c) for x in q])


def sphere_characteristic(x):
    """Characteristic function of a position uniformly distributed in a ball."""
    x = np.asarray(x, dtype=float)
    small = abs(x) < 1e-3
    result = np.empty_like(x)
    result[small] = 1 - x[small]**2/10 + x[small]**4/280 - x[small]**6/15120
    result[~small] = 3*(np.sin(x[~small])-x[~small]*np.cos(x[~small]))/x[~small]**3
    return result


def write_cif(path, lengths, group, symbol='C', position=(0, 0, 0), multiplicity=1):
    path.write_text(
        'data_synthetic_validation\n'
        + ''.join(f'_cell_length_{axis} {value}\n' for axis, value in zip('abc', lengths))
        + '_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n'
        + f"_space_group_name_H-M_alt '{group}'\n"
        + 'loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n'
        + '_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n'
        + '_atom_site_symmetry_multiplicity\n'
        + f'{symbol}1 {symbol} {position[0]} {position[1]} {position[2]} 1 {multiplicity}\n',
        encoding='utf-8')


def make_config(host, q, case='single_crystal_finite', estimator='explicit', **fields):
    bulk = case == 'single_crystal_periodic'
    sampling = ({'reflection_q_max': float(q)} if bulk else
                {'q': np.asarray(q).tolist()} if case == 'pxrd' else
                {'q_vectors': np.asarray(q).tolist()})
    return dict(schema_version='1.1', scattering_case=case,
                boundary='bulk' if bulk else 'finite', guest_estimator=estimator,
                backend='reference', host=host, sampling=sampling, **fields)


def validate(output, samples=8192):
    output = Path(output).resolve()
    inputs = output/'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    checks = []
    curves = {}

    def check(name, actual, expected, equation, tolerance=2e-11):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(abs(actual-expected)))
        scale = max(1., float(np.max(abs(expected))))
        record = dict(name=name, equation=equation, points=actual.size,
                      max_absolute_error=error, max_error_divided_by_reference_scale=error/scale,
                      reference_scale=scale, allowed_scaled_error=tolerance, passed=error <= tolerance*scale)
        checks.append(record)
        if not record['passed']:
            raise AssertionError(record)

    def run(name, config):
        (inputs/(name+'.json')).write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
        result = run_configuration(config, inputs)
        write_result(result, output/name)
        return result

    def table(name, **columns):
        np.savetxt(output/(name+'.csv'), np.column_stack(list(columns.values())), delimiter=',',
                   header=','.join(columns), comments='')
        curves[name] = columns

    q = np.linspace(0, 6, 241)
    vectors = np.column_stack([q, np.zeros((len(q), 2))])
    fc, fa = factor('C', q), factor('Au', q)

    # One atom at a nonzero origin: its phase cannot change its intensity.
    write_cif(inputs/'single.cif', [10]*3, 'P 1', position=(.17, .23, .31))
    single = run('single_atom', make_config({'cif': 'single.cif'}, vectors))['tables']['intensity']['total']
    single_powder = run('single_atom_powder', make_config({'cif': 'single.cif'}, q, 'pxrd'))['tables']['intensity']['total']
    check('single_atom_directional', single, fc**2, 'I(q) = f_C(q)^2, independent of position and direction')
    check('single_atom_powder', single_powder, fc**2, 'I_powder(q) = f_C(q)^2')
    table('single_atom', q=q, calculated=single/fc**2, expected=np.ones(len(q)))

    # Two equal atoms: a cosine for a direction, a sinc only after powder averaging.
    d = 2.5
    AtomCloud(('C', 'C'), [[-d/2, 0, 0], [d/2, 0, 0]], 'synthetic dimer').write_xyz(inputs/'dimer.xyz')
    directional = run('dimer_directional', make_config({'xyz': 'dimer.xyz'}, vectors))['tables']['intensity']['total']
    powder = run('dimer_powder', make_config({'xyz': 'dimer.xyz'}, q, 'pxrd'))['tables']['intensity']['total']
    transverse = run('dimer_transverse', make_config({'xyz': 'dimer.xyz'}, vectors[:, [1, 0, 2]]))['tables']['intensity']['total']
    directional_exact = 2*fc**2*(1+np.cos(q*d))
    powder_exact = 2*fc**2*(1+np.sinc(q*d/np.pi))
    check('dimer_directional', directional, directional_exact, 'I_parallel = 2 f^2 [1 + cos(q d)]')
    check('dimer_powder', powder, powder_exact, 'I_powder = 2 f^2 [1 + sin(q d)/(q d)]')
    check('dimer_transverse', transverse, 4*fc**2, 'I_perpendicular = 4 f^2')
    nulls = np.array([np.pi/d, 3*np.pi/d, 5*np.pi/d])
    extinct = run('dimer_zeros', make_config({'xyz': 'dimer.xyz'}, np.column_stack([nulls, np.zeros((3, 2))])))['tables']['intensity']['total']
    check('dimer_exact_interference_zeros', extinct, np.zeros(3), 'I(q d = odd pi) = 0')
    table('dimer', q=q, directional=directional/fc**2, directional_exact=directional_exact/fc**2,
          powder=powder/fc**2, powder_exact=powder_exact/fc**2)

    # Nine-atom finite chain, with a stable removable-singularity limit.
    n, spacing = 9, 2.1
    chain_q = np.unique(np.r_[q, 2*np.pi/spacing, 4*np.pi/spacing])
    chain_vectors = np.column_stack([chain_q, np.zeros((len(chain_q), 2))])
    chain_f = factor('C', chain_q)
    write_cif(inputs/'chain_cell.cif', [spacing, 8, 8], 'P 1')
    chain_host = {'cif': 'chain_cell.cif', 'cell_mask': np.ones((n, 1, 1), int).tolist()}
    chain_config = make_config(chain_host, chain_vectors)
    chain = run('chain_explicit', chain_config)['tables']['intensity']['total']
    chain_config = {**chain_config, 'backend': 'lattice_grouped'}
    grouped = run('chain_factorized', chain_config)['tables']['intensity']['total']
    phase = (chain_q*spacing+np.pi) % (2*np.pi)-np.pi
    chain_exact = chain_f**2*(n*np.sinc(n*phase/(2*np.pi))/np.sinc(phase/(2*np.pi)))**2
    check('finite_chain_explicit', chain, chain_exact, 'I = f^2 [sin(N q d/2)/sin(q d/2)]^2; reciprocal-node limit N^2 f^2')
    check('finite_chain_factorized', grouped, chain_exact, 'Same finite-chain grating law through complete-cell factorization')
    chain_powder = run('chain_powder', make_config(chain_host, chain_q, 'pxrd'))['tables']['intensity']['total']
    powder_chain_exact = chain_f**2*(n+2*sum((n-m)*np.sinc(chain_q*spacing*m/np.pi) for m in range(1, n)))
    check('finite_chain_powder', chain_powder, powder_chain_exact, 'I/f^2 = N + 2 sum_{m=1}^{N-1} (N-m) sinc(q m d)')
    table('chain', q=chain_q, calculated=chain/chain_f**2, factorized=grouped/chain_f**2, expected=chain_exact/chain_f**2)

    # Cubic motifs imported from one CIF special-position site, not expanded XYZ.
    for name, group, atoms in [('bcc', 'I m -3 m', 2), ('fcc', 'F m -3 m', 4)]:
        write_cif(inputs/(name+'.cif'), [4]*3, group, multiplicity=atoms)
        periodic = make_config({'cif': name+'.cif'}, 6, 'single_crystal_periodic', 'analytic')
        reflections = run(name, periodic)['tables']['reflections']
        hkl = np.column_stack([reflections[key].astype(int) for key in ('h', 'k', 'l')])
        oracle_hkl = np.array([h for h in product(range(-4,5),repeat=3)
                               if 0 < sum(x*x for x in h) <= (6*4/(2*np.pi))**2+1e-12])
        check(name+'_complete_reflection_set', sorted(map(tuple,hkl)), sorted(map(tuple,oracle_hkl)),
              'Independent integer sphere enumeration: 0 < h^2+k^2+l^2 <= (q_max a/2pi)^2', tolerance=0)
        check(name+'_reciprocal_vectors', np.column_stack([reflections[k] for k in ('qx','qy','qz')]),
              hkl*(2*np.pi/4), 'Cubic G = (2pi/a)(h,k,l)')
        check(name+'_reciprocal_radii', reflections['q'], np.sqrt((hkl*hkl).sum(axis=1))*(2*np.pi/4),
              '|G| = (2pi/a) sqrt(h^2+k^2+l^2)')
        allowed = (hkl.sum(axis=1) % 2 == 0) if name == 'bcc' else np.all(hkl % 2 == hkl[:, :1] % 2, axis=1)
        f = factor('C', reflections['q'])
        exact = atoms**2*f**2*allowed
        check(name+'_structure_factor', reflections['f_squared'], exact,
              'BCC: F/f=1+(-1)^(h+k+l)' if name == 'bcc' else 'FCC: F/f=4 for all h,k,l same parity; otherwise 0')
        check(name+'_bragg_measure', reflections['bragg_weight'], (2*np.pi)**3/4**3*exact, 'W_G = (2 pi)^3/V |F_G|^2')
        table(name, h=hkl[:, 0], k=hkl[:, 1], l=hkl[:, 2], q=reflections['q'],
              calculated=reflections['f_squared']/f**2, expected=atoms**2*allowed)
        bulk_powder = run(name+'_powder', {**periodic, 'scattering_case': 'pxrd', 'sampling': {'q': q.tolist()}})['tables']['powder_shells']
        squared_h = (oracle_hkl*oracle_hkl).sum(axis=1)
        oracle_shells, multiplicities = np.unique(squared_h, return_counts=True)
        radii = np.sqrt(oracle_shells)*(2*np.pi/4)
        oracle_allowed = (oracle_hkl.sum(axis=1)%2==0) if name=='bcc' else np.all(oracle_hkl%2==oracle_hkl[:,:1]%2,axis=1)
        check(name+'_powder_shell_radii', bulk_powder['q'], radii, 'Independent integer-shell radii')
        check(name+'_powder_multiplicity', bulk_powder['multiplicity'], multiplicities,
              'Independent count of all signed/permuted reciprocal vectors in each shell', tolerance=0)
        expected_weights = []
        for shell_number,radius,f_shell in zip(oracle_shells,radii,factor('C',radii)):
            selected = squared_h == shell_number
            expected_weights.append(atoms**2*f_shell**2*np.sum(oracle_allowed[selected])*(2*np.pi)**3/4**3/(4*np.pi*radius**2))
        check(name+'_powder_radial_weights', bulk_powder['radial_weight'], expected_weights,
              'Each cubic h^2+k^2+l^2 shell: sum W_G/(4 pi |G|^2), counting every reciprocal vector once')

    # Occupancy average at q=0: enumerate absent/present states analytically.
    host = AtomCloud(('C',), [[0, 0, 0]])
    guest = AtomCloud(('Au',), [[0, 0, 0]])
    host.write_xyz(inputs/'host.xyz'); guest.write_xyz(inputs/'guest.xyz')
    probabilities = np.linspace(0, 1, 51)
    actual = []
    for probability in probabilities:
        population = PorePopulation([[2, 0, 0]], ParticleLibrary((guest,), [1]), occupancy=probability)
        moments = guest_moments(population, [[0, 0, 0]], method='analytic')
        actual.append(finite_ensemble_intensity(host, [[0, 0, 0]], moments)['total'][0])
    exact = fc[0]**2+probabilities*fa[0]**2+2*probabilities*fc[0]*fa[0]
    mean_only = (fc[0]+probabilities*fa[0])**2
    check('bernoulli_occupancy_q0', actual, exact, 'E I = (1-p) f_C^2 + p (f_C+f_Au)^2, including p(1-p) f_Au^2 variance')
    table('occupancy', probability=probabilities, calculated=np.asarray(actual)/fa[0]**2,
          expected=exact/fa[0]**2, incorrect_mean_only=mean_only/fa[0]**2)

    # Unbounded Gaussian displacement: characteristic function is exact.
    p, sigma, centre = .37, .45, [1.2, .5, .8]
    write_cif(inputs/'host.cif', [8]*3, 'P 1')
    gaussian_config = make_config({'cif': 'host.cif'}, vectors, estimator='analytic',
        particles=[{'xyz': 'guest.xyz', 'weight': 1}], pores={'centres_A': [centre]},
        population={'occupancy': p, 'displacement_sigma_A': sigma})
    gaussian = run('gaussian_finite', gaussian_config)['tables']['intensity']
    phi = np.exp(-.5*(sigma*q)**2)
    mean = p*fa*phi*np.exp(1j*q*centre[0])
    variance = fa**2*(p-p*p*phi**2)
    check('gaussian_finite_total', gaussian['total'], abs(fc+mean)**2+variance, 'E I = |f_C + p f_Au exp(-sigma^2 q^2/2) exp(i q.r)|^2 + diffuse')
    check('gaussian_physical_diffuse', gaussian['diffuse'], variance, 'diffuse = f_Au^2 [p - p^2 exp(-sigma^2 q^2)]')
    periodic_config = {**gaussian_config, 'scattering_case': 'single_crystal_periodic', 'boundary': 'bulk',
                       'sampling': {'reflection_q_max': 3, 'q_vectors': vectors.tolist()}}
    periodic = run('gaussian_bulk', periodic_config)['tables']
    check('gaussian_bulk_diffuse', periodic['diffuse']['diffuse'], variance, 'Per-cell diffuse equals the same independent-disorder variance')
    refl = periodic['reflections']; rq = refl['q']; rf, rg = factor('C', rq), factor('Au', rq)
    rphase = np.column_stack([refl[k] for k in ('qx', 'qy', 'qz')]) @ centre
    weight = (2*np.pi)**3/8**3*abs(rf+p*rg*np.exp(-.5*(sigma*rq)**2)*np.exp(1j*rphase))**2
    check('gaussian_bulk_bragg_weight', refl['bragg_weight'], weight, 'W_G=(2pi)^3/V |f_C+p f_Au exp(-sigma^2 G^2/2) exp(i G.r)|^2')
    table('gaussian', q=q, calculated=gaussian['diffuse']/fa**2, expected=variance/fa**2,
          coherent_guest=p*p*phi**2, total_guest=np.full(len(q), p))

    # Uniform ball: actual sampling versus exact characteristic function and SEM.
    p, radius, centre = .65, 1.25, [2., 1., .5]
    ball_config = make_config({'cif': 'host.cif'}, vectors, estimator='moments',
        particles=[{'xyz': 'guest.xyz', 'weight': 1}], pores={'centres_A': [centre]},
        population={'occupancy': p, 'position_mode': 'uniform_ball', 'uniform_ball_radius_A': radius,
                    'pore_radius_A': radius}, statistics={'n_samples': samples, 'seed': SEED})
    ball = run('uniform_ball', ball_config)['tables']['intensity']
    c1 = sphere_characteristic(q*radius)*np.cos(q*centre[0])
    c2 = (1+sphere_characteristic(2*q*radius)*np.cos(2*q*centre[0]))/2
    extra_mean = p*(fa**2+2*fc*fa*c1)
    exact = fc**2+extra_mean
    # For an occupied draw X=f_Au^2+2 f_C f_Au cos(q.(r+u)); otherwise X=0.
    second = p*(fa**4+4*fc*fa**3*c1+4*fc**2*fa**2*c2)
    exact_sem = np.sqrt(np.maximum(second-extra_mean**2, 0)/samples)
    z = abs(ball['total']-exact)/np.maximum(exact_sem, 1e-10)
    checks.append(dict(name='uniform_ball_sampled_intensity', equation='phi(q)=3[sin(qR)-qR cos(qR)]/(qR)^3; E I=f_C^2+p f_Au^2+2p f_C f_Au phi cos(q.r)',
                       points=len(q), samples=samples, seed=SEED, max_error_in_exact_intensity_SEM=float(z.max()),
                       allowed_exact_SEM=6., passed=bool(z.max()<6),
                       SEM_source='Independent exact second moment of absent/present single-guest configuration intensity; not production amplitude SEM'))
    if not checks[-1]['passed']:
        raise AssertionError(checks[-1])
    cross_sem = np.sqrt(np.maximum(p*c2-p*p*c1**2, 0)/samples)/p
    cross_actual = ball['host_guest']/(2*p*fc*fa)
    cross_z = abs(cross_actual-c1)/np.maximum(cross_sem, 1e-10)
    checks.append(dict(name='uniform_ball_sampled_interference', equation='E cross/(2 p f_C f_Au) = phi(qR) cos(q.r)',
                       points=len(q), samples=samples, seed=SEED, max_error_in_exact_intensity_SEM=float(cross_z.max()),
                       allowed_exact_SEM=6., passed=bool(cross_z.max()<6),
                       SEM_source='Exact variance of occupancy times cos(q.(r+u)), divided by sample count'))
    if not checks[-1]['passed']:
        raise AssertionError(checks[-1])
    table('uniform_ball', q=q, calculated=ball['host_guest']/(2*p*fc*fa), expected=c1,
          exact_cross_SEM=cross_sem, total=ball['total'], total_expected=exact, exact_intensity_SEM=exact_sem)

    summary = dict(status='passed', purpose='Known-solution correctness tests, not performance benchmarks',
                   physical_examples=8, checks=checks, environment=environment_record(),
                   oracle_independence='Scattering references are explicit equations; neutral IT92 table is the shared physical input and evaluated independently.',
                   geometry_scope='Synthetic point-atom geometries and probability laws; no real material or instrument claim.',
                   source_hashes={name: hashlib.sha256((PROJECT/'src/mofdebye'/name).read_bytes()).hexdigest()
                                  for name in ['crystal.py', 'core.py', 'directional.py', 'moments.py', 'periodic.py', 'runner.py', 'distributions.py']})
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return summary, curves


def plot(curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(4, 2, figsize=(13, 16), constrained_layout=True)
        fig.suptitle('Known physical solutions: equations versus computed scattering', fontsize=16)
        def lines(ax, x, calculated, exact, title, xlabel, ylabel):
            ax.plot(x, exact, color='#1A6B87', label='Exact equation', linewidth=1.8)
            ax.plot(x[::6], calculated[::6], 'o', color='#B86824', ms=4, mfc='none', label='Code')
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.grid(alpha=.18)
            ax.legend(frameon=False, fontsize=8)
        c=curves['single_atom']
        lines(axes[0,0], c['q'], c['calculated'], c['expected'], '1. One atom: position-independent intensity', 'q / inverse angstrom', 'I / f(q)^2')
        axes[0,0].set_ylim(.95,1.05)
        c=curves['dimer']
        lines(axes[0,1],c['q'],c['directional'],c['directional_exact'],'2. Dimer: directional cosine and powder sinc','q / inverse angstrom','I / f(q)^2')
        axes[0,1].plot(c['q'],c['powder_exact'],color='#5E4585',label='Exact powder')
        axes[0,1].plot(c['q'][::9],c['powder'][::9],'s',mfc='none',color='#4F8752',ms=3,label='Code powder')
        axes[0,1].legend(frameon=False,fontsize=8)
        c=curves['chain']
        lines(axes[1,0],c['q']*2.1/(2*np.pi),c['calculated'],c['expected'],'3. Nine-atom chain: fringes and N^2 maxima','q d / (2 pi)','I / f(q)^2')
        c=curves['occupancy']
        lines(axes[2,1],c['probability'],c['calculated'],c['expected'],'6. Random occupancy: average intensity','Occupancy probability p','I / f_Au(0)^2')
        axes[2,1].plot(c['probability'],c['incorrect_mean_only'],'--',color='#A34B63',label='Mean amplitude squared only (incorrect)')
        axes[2,1].legend(frameon=False,fontsize=8)
        for ax,name,title in [(axes[1,1],'bcc','4. BCC: h+k+l must be even'),(axes[2,0],'fcc','5. FCC: h,k,l must share parity')]:
            c=curves[name]; section=c['l']==0; allowed=c['expected']>0
            ax.scatter(c['h'][section & ~allowed],c['k'][section & ~allowed],marker='x',color='#AAA',s=35,label='Equation: forbidden')
            ax.scatter(c['h'][section & allowed],c['k'][section & allowed],facecolors='none',edgecolors='#1A6B87',s=95,label='Equation: allowed')
            positive=section & (c['calculated']>1e-20)
            ax.scatter(c['h'][positive],c['k'][positive],color='#B86824',s=18,label='Code: nonzero')
            ax.set(title=title,xlabel='h (l=0)',ylabel='k',aspect='equal')
            ax.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3);ax.grid(alpha=.15)
        c=curves['gaussian']
        lines(axes[3,0],c['q'],c['calculated'],c['expected'],'7. Gaussian disorder: coherent to diffuse','q / inverse angstrom','Guest contribution / f_Au(q)^2')
        axes[3,0].plot(c['q'],c['coherent_guest'],'--',label='Coherent guest',color='#5E4585')
        axes[3,0].plot(c['q'],c['total_guest'],':',label='Total guest = p',color='#888')
        axes[3,0].legend(frameon=False,fontsize=8)
        c=curves['uniform_ball']
        lines(axes[3,1],c['q'],c['calculated'],c['expected'],'8. Uniform-ball guest: sampled versus exact','q / inverse angstrom','Host-guest cross term / (2 p f_C f_Au)')
        axes[3,1].fill_between(c['q'],c['expected']-3*c['exact_cross_SEM'],c['expected']+3*c['exact_cross_SEM'],color='#1A6B87',alpha=.15,label='Exact 3 SEM band')
        axes[3,1].legend(frameon=False,fontsize=8)
        fig.savefig(Path(output)/'physical_examples.png',dpi=180)
        fig.savefig(Path(output)/'physical_examples.pdf')
        plt.close(fig)


def plot_geometries(output):
    """Display the actual generated sites and clearly labelled distribution guides."""
    from itertools import product, combinations
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mofdebye.crystal import read_cif
    inputs = Path(output)/'inputs'
    with plt.rc_context({'font.size': 9}):
        fig = plt.figure(figsize=(18, 10))
        fig.subplots_adjust(left=.025,right=.965,bottom=.07,top=.9,wspace=.35,hspace=.38)
        fig.suptitle('Validation geometries | blue: fixed carbon | gold: guest site or mean position', fontsize=15)
        geometries = [
            ('1. Single atom', read_cif(inputs/'single.cif').basis.xyz, None, None),
            ('2. Dimer: d = 2.5 A', AtomCloud.read_xyz(inputs/'dimer.xyz').xyz, None, None),
            ('3. Nine-atom chain: d = 2.1 A', np.column_stack([np.arange(9)*2.1,np.zeros((9,2))]), None, None),
            ('4. BCC: 2 unique cell sites', read_cif(inputs/'bcc.cif').basis.xyz, None, 4),
            ('5. FCC: 4 unique cell sites', read_cif(inputs/'fcc.cif').basis.xyz, None, 4),
            ('6. Bernoulli guest: absent or present', np.zeros((1,3)), np.array([2.,0,0]), None),
            ('7. Gaussian: sigma = 0.45 A', np.zeros((1,3)), np.array([1.2,.5,.8]), None),
            ('8. Uniform ball: R = 1.25 A', np.zeros((1,3)), np.array([2.,1,.5]), None),
        ]
        for index,(title,xyz,guest,edge) in enumerate(geometries):
            if index in (1,2,5):
                ax=fig.add_subplot(2,4,index+1)
                ax.scatter(xyz[:,0],xyz[:,1],color='#1A6B87',s=65,zorder=3)
                extent=xyz[:,0]
                if guest is not None:
                    ax.scatter(guest[0],guest[1],color='#CA9223',s=95,zorder=3)
                    extent=np.r_[extent,guest[0]]
                ax.axhline(0,color='#AAA',lw=.8)
                ax.set(title=title,xlabel='x / A',ylabel='y / A (z = 0)',
                       xlim=(extent.min()-1,extent.max()+1),ylim=(-1,1),yticks=[-1,0,1])
                ax.grid(alpha=.15)
                continue
            ax=fig.add_subplot(2,4,index+1,projection='3d')
            ax.scatter(*xyz.T,color='#1A6B87',s=65,depthshade=False)
            bounds=xyz.copy()
            if guest is not None:
                ax.scatter(*guest,color='#CA9223',s=95,depthshade=False)
                bounds=np.vstack([bounds,guest])
            if edge:
                corners=np.array(list(product((0.,float(edge)),repeat=3)))
                for a,b in combinations(corners,2):
                    if np.count_nonzero(a!=b)==1:
                        ax.plot(*np.vstack([a,b]).T,color='#888',alpha=.65,lw=.8)
                bounds=corners
            if index in (6,7):
                radius=.45 if index==6 else 1.25
                theta,phi=np.meshgrid(np.linspace(0,2*np.pi,25),np.linspace(0,np.pi,13))
                x=guest[0]+radius*np.sin(phi)*np.cos(theta)
                y=guest[1]+radius*np.sin(phi)*np.sin(theta)
                z=guest[2]+radius*np.cos(phi)
                ax.plot_wireframe(x,y,z,color='#CA9223',alpha=.3,linewidth=.6)
                bounds=np.vstack([bounds,guest-radius,guest+radius])
                note='Wire: 1-sigma guide; distribution unbounded' if index==6 else 'Wire: allowed displacement boundary'
                ax.text2D(.5,-.04,note,transform=ax.transAxes,ha='center',fontsize=8)
            low=bounds.min(axis=0);high=bounds.max(axis=0);span=np.maximum(high-low,.8)
            middle=(low+high)/2
            low=middle-.65*span;high=middle+.65*span
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),
                   title=title,xlabel='x / A',ylabel='y / A',zlabel='z / A')
            ax.set_box_aspect(high-low)
            ax.view_init(elev=23,azim=-58)
        fig.savefig(Path(output)/'physical_geometries.png',dpi=160)
        fig.savefig(Path(output)/'physical_geometries.pdf')
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=PROJECT.parent/'local_runs/physical_examples')
    parser.add_argument('--samples',type=int,default=8192)
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    if args.samples<2:
        parser.error('--samples must be at least 2')
    summary,curves=validate(args.out,args.samples)
    if not args.no_plot:
        plot(curves,args.out)
        plot_geometries(args.out)
    exact=[c['max_error_divided_by_reference_scale'] for c in summary['checks'] if 'max_error_divided_by_reference_scale' in c]
    print(json.dumps({'status':summary['status'],'physical_examples':summary['physical_examples'],'checks':len(summary['checks']),
                      'worst_exact_scaled_error':max(exact),'sampled_max_exact_SEM':max(c['max_error_in_exact_intensity_SEM'] for c in summary['checks'] if 'max_error_in_exact_intensity_SEM' in c),
                      'output':str(args.out.resolve())},indent=2))


if __name__=='__main__':
    main()
