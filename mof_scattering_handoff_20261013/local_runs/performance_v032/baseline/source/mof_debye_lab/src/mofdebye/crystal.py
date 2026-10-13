"""Strict small-molecule CIF import into a fully occupied, static unit-cell basis.

Gemmi parses CIF syntax, uncertainty suffixes and chemical element symbols. We
validate symmetry and site policies before expanding explicitly; no occupancies,
thermal factors, solvent masks or missing hydrogens are silently reconstructed.
Lattice vectors are rows: ``cartesian = fractional @ cell_vectors_A``.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import re

import gemmi
import numpy as np

from .core import AtomCloud


@dataclass(frozen=True)
class CrystalStructure:
    basis: AtomCloud
    cell_vectors_A: np.ndarray
    fractional: np.ndarray
    metadata: dict
    symops: tuple[str, ...]

    def __post_init__(self):
        for name in ('cell_vectors_A', 'fractional'):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _column(block, tag):
    return list(block.find_values(tag))


def _scalar(block, tag):
    values = _column(block, tag)
    if len(values) > 1:
        raise ValueError(f'CIF {tag} must be a scalar.')
    return gemmi.cif.as_string(values[0]) if values and values[0] not in ('.', '?') else None


def _number(value, description):
    try:
        result = gemmi.cif.as_number(value)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f'CIF {description} must be a finite number.') from exc
    if not np.isfinite(result):
        raise ValueError(f'CIF {description} must be a finite number, not {value!r}.')
    return float(result)


def _cell_matrix(cell):
    array = np.asarray(cell, dtype=float)
    if (array.shape != (3, 3) or not np.isfinite(array).all()
            or abs(np.linalg.det(array)) < 1e-12):
        raise ValueError('cell_vectors_A must be a nonsingular finite 3-by-3 matrix.')
    return array


def _tolerance(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError('Duplicate tolerance must be positive and finite in angstrom.')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Duplicate tolerance must be positive and finite in angstrom.')
    return value


def _operations(values, description):
    """Require a complete finite crystallographic group modulo cell translations."""
    try:
        operations = tuple(gemmi.Op(value).wrap() for value in values)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f'Invalid symmetry operation in {description}: {exc}') from exc
    keys = {op.triplet() for op in operations}
    if not operations or len(keys) != len(operations):
        raise ValueError(f'{description} has an empty or duplicate symmetry-operation list.')
    if 'x,y,z' not in keys:
        raise ValueError(f'{description} must include the identity symmetry operation.')
    for op in operations:
        # A crystallographic basis maps the integer lattice bijectively to itself.
        if (abs(op.det_rot()) != op.DEN ** 3
                or np.any(np.asarray(op.rot, dtype=int) % op.DEN)):
            raise ValueError(f'{description} contains a non-crystallographic rotation.')
    if len(operations) > 192:
        raise ValueError(f'{description} has more than 192 crystallographic operations.')
    for a in operations:
        for b in operations:
            if (a * b).wrap().triplet() not in keys:
                raise ValueError(f'{description} is not a complete, closed symmetry-operation list.')
    return tuple(op.triplet() for op in operations)


def _resolve_symmetry(block, cell_parameters):
    declarations = []
    number = None
    for tag in ('_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz'):
        values = _column(block, tag)
        if values:
            ops = _operations([gemmi.cif.as_string(v) for v in values], tag)
            declarations.append((tag, [ops]))
    for tag in ('_space_group_name_Hall', '_symmetry_space_group_name_Hall'):
        value = _scalar(block, tag)
        if value:
            try:
                ops = tuple(op.triplet() for op in gemmi.symops_from_hall(value))
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f'Unknown or invalid Hall symmetry symbol {value!r}.') from exc
            declarations.append((tag, [_operations(ops, tag)]))
    for tag in ('_space_group_name_H-M_alt', '_symmetry_space_group_name_H-M'):
        value = _scalar(block, tag)
        if value:
            candidates = []
            for origin in ('1', '2'):
                sg = gemmi.find_spacegroup_by_name(value, cell_parameters[3], cell_parameters[5], origin)
                if sg is None:
                    raise ValueError(f'Unknown H-M symmetry symbol {value!r}.')
                ops = tuple(op.triplet() for op in sg.operations())
                if set(ops) not in [set(candidate) for candidate in candidates]:
                    candidates.append(ops)
            declarations.append((tag, candidates))
    for tag in ('_space_group_IT_number', '_symmetry_Int_Tables_number'):
        value = _scalar(block, tag)
        if value:
            parsed = _number(value, tag)
            if not parsed.is_integer() or not 1 <= parsed <= 230:
                raise ValueError(f'{tag} must be an International Tables integer from 1 to 230.')
            if number is not None and number != int(parsed):
                raise ValueError('Conflicting CIF space-group numbers.')
            number = int(parsed)
    source = None
    if declarations:
        candidates = list(declarations[0][1])
        for tag, alternatives in declarations[1:]:
            candidates = [ops for ops in candidates if any(set(ops) == set(other) for other in alternatives)]
            if not candidates:
                raise ValueError(f'CIF symmetry declarations disagree (including {tag}); check setting/origin.')
        if len(candidates) != 1:
            raise ValueError('Ambiguous space-group origin: supply explicit symmetry operations or an H-M :1/:2 choice.')
        ops = candidates[0]
        source = declarations[0][0]
    elif number is not None:
        sg = gemmi.get_spacegroup_reference_setting(number)
        choices = [gemmi.find_spacegroup_by_name(sg.hm, cell_parameters[3], cell_parameters[5], origin)
                   for origin in ('1', '2')]
        if {op.triplet() for op in choices[0].operations()} != {op.triplet() for op in choices[1].operations()}:
            raise ValueError('Ambiguous space-group origin for IT number; supply explicit operations or an H-M :1/:2 choice.')
        ops = tuple(op.triplet() for op in sg.operations())
        source = 'IT_number_reference_setting'
    else:
        raise ValueError('CIF is missing symmetry operations or a resolvable space-group declaration; specify P 1 explicitly.')
    resolved = gemmi.find_spacegroup_by_ops(gemmi.GroupOps([gemmi.Op(op) for op in ops]))
    if number is not None and (resolved is None or resolved.number != number):
        raise ValueError('CIF symmetry operations and space-group number disagree.')
    return ops, source, resolved


def _validate_metric(operations, cell):
    # Test dimensionless Cartesian rotations, so a very long axis cannot hide
    # incompatible lengths/angles in the other axes. Allow cell rounding at 1e-4.
    inverse_columns = np.linalg.inv(cell.T)
    for op in operations:
        rotation = np.asarray(gemmi.Op(op).rot, dtype=float) / gemmi.Op.DEN
        cartesian_rotation = cell.T @ rotation @ inverse_columns
        if not np.allclose(cartesian_rotation.T @ cartesian_rotation, np.eye(3), rtol=0, atol=1e-4):
            raise ValueError('CIF unit-cell metric is incompatible with its symmetry operations.')


def _periodic_matches(point, previous, cell, tolerance, fractional_bounds):
    """Find *all* sites within a Cartesian periodic tolerance, even in skew cells.

    If ||(delta-n) A|| <= tol then |delta_i-n_i| <= tol ||A^-1[:,i]||.
    Enumerating exactly these integer ranges avoids assuming that componentwise
    rounding gives the nearest periodic image in a nonorthogonal cell.
    """
    if not previous:
        return []
    delta = np.asarray(previous) - point
    centered = delta - np.rint(delta)
    possible = np.flatnonzero(np.all(np.abs(centered) <= fractional_bounds + 1e-14, axis=1))
    matches = []
    for index in possible:
        lower = np.ceil(delta[index] - fractional_bounds - 1e-14).astype(np.int64)
        upper = np.floor(delta[index] + fractional_bounds + 1e-14).astype(np.int64)
        if np.prod(upper - lower + 1, dtype=float) > 1_000_000:
            raise ValueError('Cell is too ill-conditioned for this duplicate tolerance; reduce the cell or tolerance.')
        for shift in product(*(range(a, b + 1) for a, b in zip(lower, upper))):
            if np.linalg.norm((delta[index] - shift) @ cell) <= tolerance * (1 + 1e-12):
                matches.append(int(index))
                break
    return matches


def expand_fractional_sites(centres, cell, symops, tolerance_A=1e-5):
    """Expand fractional pore seeds to unique [0,1) positions in the same cell.

    Returned coordinates remain fractional; this neither infers pores from atoms
    nor assigns pore sizes. All input seeds describe one equivalent pore family.
    """
    cell = _cell_matrix(cell)
    tolerance = _tolerance(tolerance_A)
    positions = np.asarray(centres, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError('Fractional sites must be a finite N-by-3 array.')
    operations = _operations(symops, 'pore symmetry')
    _validate_metric(operations, cell)
    bounds = tolerance * np.linalg.norm(np.linalg.inv(cell), axis=0)
    expanded = []
    for position in positions:
        for operation in operations:
            point = np.mod(gemmi.Op(operation).apply_to_xyz(position.tolist()), 1.0)
            if not _periodic_matches(point, expanded, cell, tolerance, bounds):
                expanded.append(point)
    return np.asarray(expanded, dtype=float).reshape(-1, 3)


def read_cif(path, *, block=None, occupancy_policy='require_full', exclude_labels=(),
             exclude_elements=(), duplicate_tolerance_A=1e-5, duplicate_policy='merge_same_species'):
    """Read a small-molecule CIF and expand its asymmetric sites into one cell.

    Partial occupancy and disorder are rejected by default. ``drop_partial``
    explicitly discards sites with 0 <= occupancy < 1 and records every removal;
    it never rescales occupancy to one. Marked disorder still requires explicit
    label/element exclusion. Solvent and hydrogens are retained by default; CIF
    labels/elements are the only exclusions (no connectivity-based guessing).
    Missing occupancy tags imply one, but unknown occupancy values are errors.
    ``merge_same_species`` merges equivalent distinct input sites and records
    this; ``error`` rejects them. Special-position symmetry copies always merge.
    Different species at the same position always fail. B/U values and ionic
    charges are recorded but the existing neutral-atom static model ignores them.
    """
    if occupancy_policy not in ('require_full', 'drop_partial'):
        raise ValueError('occupancy_policy must be require_full or drop_partial.')
    if duplicate_policy not in ('merge_same_species', 'error'):
        raise ValueError('duplicate_policy must be merge_same_species or error.')
    tolerance = _tolerance(duplicate_tolerance_A)
    if isinstance(exclude_labels, str) or isinstance(exclude_elements, str):
        raise ValueError('CIF exclusions must be sequences of labels/elements, not strings.')
    excluded_labels = set(exclude_labels)
    excluded_elements = set()
    for name in exclude_elements:
        if not isinstance(name, str):
            raise ValueError('Excluded CIF elements must be chemical symbols.')
        element = gemmi.Element(name)
        if element.atomic_number == 0 or element.name.lower() != name.lower():
            raise ValueError(f'Unknown excluded element {name!r}.')
        excluded_elements.add(element.name)
    path = Path(path).resolve()
    data = path.read_bytes()
    try:
        document = gemmi.cif.read_string(data.decode('utf-8-sig'))
    except (UnicodeError, ValueError, RuntimeError) as exc:
        raise ValueError(f'Cannot parse CIF {path.name}: {exc}') from exc
    if block is None:
        if len(document) != 1:
            raise ValueError('CIF must have one data block or an explicit block name.')
        selected = document[0]
    else:
        if not isinstance(block, str):
            raise ValueError('CIF block selector must be a data-block name.')
        selected = document.find_block(block)
        if selected is None:
            raise ValueError(f'CIF data block {block!r} was not found.')
    parameters = []
    for tag in ('_cell_length_a', '_cell_length_b', '_cell_length_c',
                '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma'):
        value = _scalar(selected, tag)
        if value is None:
            raise ValueError(f'CIF requires {tag}.')
        parameters.append(_number(value, tag))
    if any(value <= 0 for value in parameters[:3]) or any(not 0 < angle < 180 for angle in parameters[3:]):
        raise ValueError('CIF cell lengths must be positive and angles must lie strictly between 0 and 180 degrees.')
    unit_cell = gemmi.UnitCell(*parameters)
    cell = _cell_matrix(np.asarray(unit_cell.orth.mat).T)
    if not np.isfinite(unit_cell.volume) or unit_cell.volume <= 1e-12:
        raise ValueError('CIF unit cell has invalid or zero volume.')
    operations, symmetry_source, sg = _resolve_symmetry(selected, parameters)
    _validate_metric(operations, cell)
    labels = [gemmi.cif.as_string(v) for v in _column(selected, '_atom_site_label')]
    if not labels or any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError('CIF requires nonempty, unique atom-site labels.')
    if excluded_labels - set(labels):
        raise ValueError(f'Excluded CIF labels do not exist: {sorted(excluded_labels - set(labels))}.')
    columns = {}
    for suffix in ('fract_x', 'fract_y', 'fract_z', 'occupancy', 'disorder_group', 'disorder_assembly',
                   'symmetry_multiplicity', 'site_symmetry_multiplicity'):
        column = _column(selected, '_atom_site_' + suffix)
        if (suffix.startswith('fract_') or column) and len(column) != len(labels):
            raise ValueError(f'CIF atom-site {suffix} column must match the atom-site labels.')
        columns[suffix] = column
    try:
        small = gemmi.make_small_structure_from_block(selected)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f'Cannot parse CIF atom sites: {exc}') from exc
    if len(small.sites) != len(labels):
        raise ValueError('CIF did not produce one fractional atom site per label.')
    type_symbols = _column(selected, '_atom_site_type_symbol')
    if type_symbols and len(type_symbols) != len(labels):
        raise ValueError('CIF atom-site type symbols must match the atom-site labels.')
    bounds = tolerance * np.linalg.norm(np.linalg.inv(cell), axis=0)
    expanded, elements, origins = [], [], []
    exclusions, merged_sites, input_sites = [], [], []
    candidates = 0
    for index, site in enumerate(small.sites):
        label = site.label
        element = site.element.name
        record = {'label': label, 'element': element}
        if label in excluded_labels or element in excluded_elements:
            record['reason'] = 'exclude_labels' if label in excluded_labels else 'exclude_elements'
            exclusions.append(record)
            continue
        if site.element.atomic_number == 0:
            raise ValueError(f'Unknown chemical element at CIF site {label!r}.')
        if type_symbols:
            symbol = gemmi.cif.as_string(type_symbols[index])
            match = re.fullmatch(r'([A-Za-z]{1,2})(?:[0-9]*[+-])?', symbol)
            if match is None or match.group(1).lower() != element.lower():
                raise ValueError(f'Invalid chemical type symbol {symbol!r} at CIF site {label!r}; '
                                 'use an element symbol with an optional standard charge suffix (e.g. Zn2+).')
        occupancy = _number(columns['occupancy'][index], f'occupancy of {label}') if columns['occupancy'] else 1.0
        record['occupancy'] = occupancy
        if occupancy < 0 or occupancy > 1:
            raise ValueError(f'CIF site {label!r} has occupancy outside [0,1].')
        if occupancy != 1.0:
            if occupancy_policy == 'drop_partial':
                record['reason'] = 'drop_partial'
                exclusions.append(record)
                continue
            raise ValueError(f'CIF site {label!r} has partial occupancy {occupancy}; require_full rejects it. '
                             'Use explicit exclusions or the drop_partial policy.')
        for suffix in ('disorder_group', 'disorder_assembly'):
            column = columns[suffix]
            if column and gemmi.cif.as_string(column[index]) not in ('', '0'):
                raise ValueError(f'CIF site {label!r} is marked with {suffix}; explicitly exclude unresolved disorder.')
        fractional = np.asarray([_number(columns['fract_' + axis][index], f'{label} fract_{axis}')
                                 for axis in ('x', 'y', 'z')])
        multiplicities = []
        for suffix in ('symmetry_multiplicity', 'site_symmetry_multiplicity'):
            if columns[suffix]:
                value = _number(columns[suffix][index], f'{label} {suffix}')
                if not value.is_integer() or value < 1:
                    raise ValueError(f'CIF site {label!r} {suffix} must be a positive integer.')
                multiplicities.append(int(value))
        input_sites.append({**record, 'fractional': fractional.tolist(), 'charge': site.charge})
        own_orbit = []
        for operation in operations:
            point = np.mod(gemmi.Op(operation).apply_to_xyz(fractional.tolist()), 1.0)
            candidates += 1
            if not _periodic_matches(point, own_orbit, cell, tolerance, bounds):
                own_orbit.append(point)
            matches = _periodic_matches(point, expanded, cell, tolerance, bounds)
            for match in matches:
                if elements[match] != element:
                    raise ValueError(f'Conflicting species at equivalent CIF sites: {origins[match]} '
                                     f'({elements[match]}) and {label} ({element}).')
                if origins[match] != label:
                    if duplicate_policy == 'error':
                        raise ValueError(f'Duplicate equivalent CIF input sites: {origins[match]} and {label}.')
                    pair = {'kept_label': origins[match], 'merged_label': label, 'element': element}
                    if pair not in merged_sites:
                        merged_sites.append(pair)
            if not matches:
                expanded.append(point)
                elements.append(element)
                origins.append(label)
        if any(multiplicity != len(own_orbit) for multiplicity in multiplicities):
            raise ValueError(f'CIF site {label!r} declares symmetry multiplicity {multiplicities}, '
                             f'but coordinates produce {len(own_orbit)} sites at tolerance {tolerance:g} A; '
                             'verify the coordinates, symmetry and duplicate tolerance (no automatic snapping).')
        input_sites[-1]['expanded_orbit_count'] = len(own_orbit)
        input_sites[-1]['declared_symmetry_multiplicity'] = multiplicities[0] if multiplicities else None
    if not expanded:
        raise ValueError('CIF exclusions leave no fully occupied atoms.')
    fractional = np.asarray(expanded, dtype=float)
    thermal_tags = [item.pair[0] for item in selected if item.pair is not None
                    and item.pair[0].lower().startswith(('_atom_site_u_', '_atom_site_b_', '_atom_site_aniso_'))]
    for item in selected:
        if item.loop is not None:
            thermal_tags.extend(tag for tag in item.loop.tags
                                if tag.lower().startswith(('_atom_site_u_', '_atom_site_b_', '_atom_site_aniso_')))
    metadata = {
        'source_path': str(path), 'source_sha256': sha256(data).hexdigest(), 'block_name': selected.name,
        'parser': 'gemmi', 'parser_version': gemmi.__version__, 'cell_parameters': parameters,
        'cell_vectors_A': cell.tolist(), 'cell_volume_A3': float(unit_cell.volume),
        'spacegroup_hm': sg.xhm() if sg else None, 'spacegroup_number': sg.number if sg else None,
        'symmetry_source': symmetry_source, 'symmetry_operations': list(operations),
        'spacegroup_setting_assumption': ('IT number uses the standard/reference setting'
                                         if symmetry_source == 'IT_number_reference_setting' else None),
        'symmetry_operation_count': len(operations), 'input_site_count': len(labels),
        'retained_input_site_count': len(input_sites), 'expanded_candidate_count': candidates,
        'expanded_atom_count': len(expanded), 'duplicates_removed': candidates - len(expanded),
        'merged_input_sites': merged_sites, 'expanded_atom_labels': origins, 'retained_input_sites': input_sites,
        'excluded_sites': exclusions, 'occupancy_policy': occupancy_policy,
        'occupancy_missing_tag_assumption': 'one' if not columns['occupancy'] else None,
        'duplicate_policy': duplicate_policy, 'duplicate_tolerance_A': tolerance,
        'exclude_labels': sorted(excluded_labels), 'exclude_elements': sorted(excluded_elements),
        'hydrogen_policy': 'retain unless explicitly excluded; missing hydrogens are not added',
        'solvent_policy': 'retain unless explicitly excluded by label/element; no solvent inference',
        'thermal_displacements': {'treatment': 'ignored; static coordinates', 'present_tags': sorted(set(thermal_tags))},
        'scattering_factors': 'neutral-atom IT92; reported ionic charges ignored',
        'element_source': '_atom_site_type_symbol' if type_symbols else 'Gemmi inference from atom-site labels',
        'coordinate_convention': 'fractional in [0,1); Cartesian angstrom = fractional @ cell_vectors_A',
    }
    basis = AtomCloud(tuple(elements), fractional @ cell, f'{path.name}:{selected.name}')
    return CrystalStructure(basis, cell, fractional, metadata, operations)
