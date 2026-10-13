"""Explicit probability distributions for rigid nanoparticles around pore centres.
Size is represented by a number-weighted catalogue of atomic templates, not by
stretching atomic coordinates. Pore occupancies are Bernoulli random variables.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.stats import lognorm
from .core import AtomCloud, _positive_integer

@dataclass(frozen=True)
class ParticleLibrary:
    templates: tuple[AtomCloud, ...]
    weights: np.ndarray

    def __post_init__(self):
        templates = tuple(self.templates)
        w = np.array(self.weights, dtype=float, copy=True)
        if not templates or w.shape != (len(templates),) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('A nonempty catalogue and finite nonnegative number weights are required.')
        if any(len(t) == 0 for t in templates):
            raise ValueError('Empty templates are not allowed; model empty pores with occupancy.')
        w /= w.sum(); w.setflags(write=False)
        object.__setattr__(self, 'templates', templates)
        object.__setattr__(self, 'weights', w)

    @property
    def mean_atom_count(self):
        return sum(w * len(t) for w, t in zip(self.weights, self.templates))


def lognormal_catalogue_weights(radius_nodes_A, median_radius_A, log_sigma, lower_A=None, upper_A=None):
    """Probability masses in midpoint bins of a truncated lognormal R distribution.
    Weights are per PARTICLE, not by mass or scattering intensity. Radius nodes
    label already-constructed atomic templates; no atom positions are rescaled.
    log_sigma is the standard deviation of ln(R / median_radius_A).
    """
    r = np.asarray(radius_nodes_A, dtype=float)
    if r.ndim != 1 or len(r) < 1 or np.any(r <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError('Radius nodes must be positive and strictly increasing.')
    if median_radius_A <= 0 or log_sigma <= 0:
        raise ValueError('Positive lognormal median and log_sigma required.')
    lo = 0.0 if lower_A is None else float(lower_A)
    hi = np.inf if upper_A is None else float(upper_A)
    if not 0 <= lo <= r[0] or hi < r[-1] or hi <= lo:
        raise ValueError('Truncation bounds must contain the catalogue nodes.')
    edges = np.concatenate(([lo], .5*(r[:-1] + r[1:]), [hi]))
    cdf = lognorm.cdf(edges, s=log_sigma, scale=median_radius_A)
    w = np.diff(cdf)
    if w.sum() <= 0:
        raise ValueError('The requested truncated size distribution has numerically zero mass.')
    return w / w.sum()

@dataclass(frozen=True)
class GuestDraw:
    pore_index: int
    template_index: int
    displacement_A: np.ndarray
    rotation_matrix: np.ndarray
    cloud: AtomCloud
    attempts: int

class PorePopulation:
    """Independent pore occupancies and template draws, with conditional placements.

    Optional hard-wall and host-clearance rejection are performed AFTER selecting
    occupancy and a template. They condition position/orientation at fixed size,
    preserving the requested occupied-pore size weights. They are not an energetic
    model. There is no inter-pore correlation model in this version.
    With spherical hard walls, pore spheres must be disjoint.
    """
    def __init__(self, centres_A, library: ParticleLibrary, occupancy=1.0,
                 displacement_sigma_A=0.0, displacement_covariance_A2=None,
                 mean_displacement_A=(0., 0., 0.), position_mode='gaussian',
                 uniform_ball_radius_A=0.0, orientation='uniform',
                 reference_rotation=None, wobble_std_deg=10.0,
                 pore_radius_A=None, wall_clearance_A=0.0,
                 host: Optional[AtomCloud]=None, min_host_distance_A=None,
                 max_attempts=10_000, periodic_cell_A=None,
                 external_boundary=None, guest_retention='site'):
        centres = np.asarray(centres_A, dtype=float)
        if centres.ndim != 2 or centres.shape[1] != 3 or not len(centres) or not np.isfinite(centres).all():
            raise ValueError('centres_A must be a finite K-by-3 array.')
        self.centres = centres.copy(); self.library = library
        self.occupancy = np.broadcast_to(np.asarray(occupancy, dtype=float), (len(centres),)).copy()
        if not np.isfinite(self.occupancy).all() or np.any((self.occupancy < 0) | (self.occupancy > 1)):
            raise ValueError('Occupancy probabilities must be in [0, 1].')
        self.mean_displacement = np.broadcast_to(np.asarray(mean_displacement_A, dtype=float), centres.shape).copy()
        if not np.isfinite(self.mean_displacement).all():
            raise ValueError('Mean displacements must be finite.')
        if displacement_covariance_A2 is None:
            if not np.isfinite(displacement_sigma_A) or displacement_sigma_A < 0:
                raise ValueError('displacement_sigma_A must be finite and nonnegative.')
            cov = np.eye(3) * float(displacement_sigma_A)**2
        else:
            cov = np.asarray(displacement_covariance_A2, dtype=float)
        if cov.shape != (3, 3) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() < -1e-12:
            raise ValueError('Displacement covariance must be a positive-semidefinite 3-by-3 matrix.')
        self.covariance = cov.copy()
        eig, vec = np.linalg.eigh(cov)
        self._cov_sqrt = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
        if position_mode not in ('gaussian', 'uniform_ball'):
            raise ValueError('position_mode must be gaussian or uniform_ball.')
        self.position_mode = position_mode
        if not np.isfinite(uniform_ball_radius_A) or uniform_ball_radius_A < 0:
            raise ValueError('uniform_ball_radius_A must be finite and nonnegative.')
        self.uniform_ball_radius = float(uniform_ball_radius_A)
        if orientation not in ('uniform', 'fixed', 'wobble'):
            raise ValueError('orientation must be uniform, fixed or wobble.')
        self.orientation = orientation
        self.reference_rotation = np.eye(3) if reference_rotation is None else np.asarray(reference_rotation, dtype=float)
        if self.reference_rotation.shape != (3, 3) or not np.allclose(self.reference_rotation.T @ self.reference_rotation, np.eye(3), atol=1e-10) or np.linalg.det(self.reference_rotation) < 0:
            raise ValueError('reference_rotation must be a proper rotation matrix.')
        if not np.isfinite(wobble_std_deg) or wobble_std_deg < 0:
            raise ValueError('wobble_std_deg must be finite and nonnegative.')
        self.wobble_std_rad = np.deg2rad(wobble_std_deg)
        self.pore_radius = None if pore_radius_A is None else np.broadcast_to(np.asarray(pore_radius_A, dtype=float), (len(centres),)).copy()
        if not np.isfinite(wall_clearance_A) or wall_clearance_A < 0:
            raise ValueError('Wall clearance must be finite and nonnegative.')
        self.wall_clearance = float(wall_clearance_A)
        if self.pore_radius is not None:
            if not np.isfinite(self.pore_radius).all() or np.any(self.pore_radius <= self.wall_clearance):
                raise ValueError('Pore radii must exceed the requested wall clearance.')
            for i in range(len(centres)):
                for j in range(i):
                    if np.linalg.norm(centres[i]-centres[j]) < self.pore_radius[i]+self.pore_radius[j]-1e-10:
                        raise ValueError('Overlapping hard-wall pores require an inter-particle exclusion model, not implemented.')
        self.min_host_distance = min_host_distance_A
        if min_host_distance_A is not None:
            if not np.isfinite(min_host_distance_A) or min_host_distance_A <= 0 or host is None or len(host) == 0:
                raise ValueError('Host clearance requires a positive finite distance and nonempty host coordinates.')
            if periodic_cell_A is None:
                self._host_tree = cKDTree(host.xyz)
            else:
                from .geometry import PeriodicHostTree
                self._host_tree = PeriodicHostTree(host, periodic_cell_A)
        else:
            self._host_tree = None
        if guest_retention not in ('site', 'centre', 'whole_particle'):
            raise ValueError('guest_retention must be site, centre or whole_particle.')
        if (guest_retention == 'site') != (external_boundary is None):
            raise ValueError('A centre/whole_particle retention rule requires an explicit external boundary.')
        if periodic_cell_A is not None and external_boundary is not None:
            raise ValueError('A bulk periodic population cannot have a finite external boundary.')
        if external_boundary is not None:
            from .geometry import contains_points
            contains_points(np.zeros((1,3)), external_boundary)
        self.external_boundary = external_boundary
        self.guest_retention = guest_retention
        self.max_attempts = _positive_integer(max_attempts, 'max_attempts')

    def _position(self, pore, rng):
        if self.position_mode == 'gaussian':
            delta = self._cov_sqrt @ rng.normal(size=3)
        else:
            direction = rng.normal(size=3)
            while np.linalg.norm(direction) == 0:
                direction = rng.normal(size=3)
            delta = direction / np.linalg.norm(direction) * self.uniform_ball_radius * rng.random()**(1/3)
        return self.mean_displacement[pore] + delta

    def _rotation(self, rng):
        if self.orientation == 'uniform':
            # Haar-uniform rotations, NOT independently uniform Euler angles.
            return Rotation.random(random_state=rng).as_matrix()
        if self.orientation == 'fixed':
            return self.reference_rotation.copy()
        perturbation = Rotation.from_rotvec(rng.normal(scale=self.wobble_std_rad, size=3)).as_matrix()
        return perturbation @ self.reference_rotation

    def sample(self, rng: np.random.Generator):
        draws = []
        for pore, centre in enumerate(self.centres):
            if rng.random() >= self.occupancy[pore]:
                continue
            # Keep this selection fixed throughout rejection: avoid unintended size reweighting.
            t = int(rng.choice(len(self.library.templates), p=self.library.weights))
            template = self.library.templates[t]
            for attempt in range(1, self.max_attempts + 1):
                u, rotation = self._position(pore, rng), self._rotation(rng)
                xyz = template.xyz @ rotation.T + u
                if self.pore_radius is not None and np.any(np.linalg.norm(xyz, axis=1) > self.pore_radius[pore]-self.wall_clearance):
                    continue
                xyz = xyz + centre
                if self.external_boundary is not None:
                    from .geometry import contains_points
                    selected = xyz if self.guest_retention == 'whole_particle' else (centre + u)[None,:]
                    if not np.all(contains_points(selected, self.external_boundary)):
                        continue
                if self._host_tree is not None and self._host_tree.query(xyz, k=1)[0].min() < self.min_host_distance:
                    continue
                cloud = AtomCloud(template.elements, xyz, f'pore {pore}, template {t}')
                draws.append(GuestDraw(pore, t, u, rotation, cloud, attempt))
                break
            else:
                raise RuntimeError(f'No valid placement for pore {pore}, template {t} after {self.max_attempts} attempts. '
                                   'This template/geometry/distribution is incompatible or rejection is inefficient.')
        return draws

    def analytic_sigma(self):
        """Check assumptions rather than silently applying an invalid analytical formula."""
        if self.external_boundary is not None:
            raise ValueError('External-boundary conditioning is not included in the analytical path; use sampled moments or explicit sampling.')
        if self.orientation != 'uniform' or self.position_mode != 'gaussian':
            raise ValueError('Analytical path requires Haar-uniform particle orientation and Gaussian displacements.')
        if self.pore_radius is not None or self._host_tree is not None:
            raise ValueError('Hard-wall/host-clearance conditioning is not included in the analytical path; use explicit sampling.')
        variance = np.trace(self.covariance) / 3
        if not np.allclose(self.covariance, np.eye(3)*variance, atol=1e-12, rtol=1e-12):
            raise ValueError('Analytical path currently requires isotropic displacement covariance.')
        return np.sqrt(max(variance, 0))
