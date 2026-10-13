"""Synthetic geometry for numerical validation, NOT a chemically validated MOF."""
from __future__ import annotations
import numpy as np
from .core import AtomCloud, concatenate

def cage_basis(n=24, radius_A=8.0):
    """A mixed-element spherical point cage. Does not represent real bonds or chemistry."""
    k = np.arange(n)
    z = 1 - 2 * (k + .5) / n
    phi = np.pi * (3 - np.sqrt(5)) * k
    xyz = radius_A * np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
    elements = tuple('Zn' if i % 12 == 0 else ('O' if i % 3 == 0 else 'C') for i in k)
    return AtomCloud(elements, xyz, 'SYNTHETIC cage: numerical test, not a real MOF')

def finite_host(shape=(2, 2, 2), spacing_A=22.0, n_basis=24, mask=None):
    basis = cage_basis(n_basis)
    mask = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    cells = np.argwhere(mask)
    centres = cells * spacing_A
    host = concatenate([basis.moved(c) for c in centres], 'SYNTHETIC finite porous lattice')
    return host, centres, basis, np.eye(3) * spacing_A, mask

def fcc_particle(radius_A=3.2, axes=(1.0, 1.0, 1.0), element='Au', lattice_A=4.08):
    """Fixed-lattice fcc cut. 4.08 A is an illustrative input, not a refined value."""
    if radius_A <= 0 or lattice_A <= 0 or min(axes) <= 0:
        raise ValueError('Positive radius, axes and lattice constant are required.')
    axes = np.asarray(axes)
    extent = int(np.ceil(radius_A * max(axes) / lattice_A)) + 1
    cells = np.stack(np.meshgrid(*[np.arange(-extent, extent+1)]*3, indexing='ij'), axis=-1).reshape(-1, 3)
    motif = np.array([[0,0,0], [0,.5,.5], [.5,0,.5], [.5,.5,0]])
    points = ((cells[:, None, :] + motif[None, :, :]) * lattice_A).reshape(-1, 3)
    points = points[(np.sum((points / axes) ** 2, axis=1) <= radius_A ** 2 + 1e-12)]
    return AtomCloud((element,) * len(points), points, f'SYNTHETIC fcc {element} cut R={radius_A:g} A')
