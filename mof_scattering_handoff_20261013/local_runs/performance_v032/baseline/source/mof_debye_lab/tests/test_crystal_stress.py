"""Independent ASE references and adversarial crystallographic input tests.

ASE is an optional external validation dependency. The 230-group sweep compares
every standard space group with ASE's separate symmetry database, including
coordinate orbits, cell geometry, and complex structure factors. It does not
use importer symmetry operations to construct the expected coordinates.
"""
from itertools import product

import gemmi
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from mofdebye.crystal import expand_fractional_sites, read_cif


SEED = 20261013
GENERAL_POSITION = np.random.default_rng(SEED).uniform(.071, .929, 3)
ORIGIN_CHOICES = [48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130,
                  133, 134, 137, 138, 141, 142, 201, 203, 222, 224, 227, 228]


def _write_cif(path, parameters, declaration, sites=None):
    if sites is None:
        sites = [('C1', 'C', GENERAL_POSITION)]
    tags = ('length_a', 'length_b', 'length_c', 'angle_alpha', 'angle_beta', 'angle_gamma')
    header = '\n'.join(f'_cell_{tag} {value:.16g}' for tag, value in zip(tags, parameters))
    atoms = '\n'.join(f'{label} {element} ' + ' '.join(f'{float(v):.16g}' for v in position)
                      for label, element, position in sites)
    path.write_text(f'data_stress\n{header}\n{declaration}\nloop_\n_atom_site_label\n'
                    '_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n'
                    f'_atom_site_fract_z\n{atoms}\n', encoding='utf-8')
    return path


def _standard_cell(number):
    # Conventional crystal-system constraints, independent of the importer.
    if number <= 2:
        return [7.1, 8.3, 9.7, 76, 83, 69]
    if number <= 15:
        return [7.1, 8.3, 9.7, 90, 106, 90]
    if number <= 74:
        return [7.1, 8.3, 9.7, 90, 90, 90]
    if number <= 142:
        return [7.1, 7.1, 9.7, 90, 90, 90]
    if number <= 194:
        return [7.1, 7.1, 9.7, 90, 90, 120]
    return [7.1, 7.1, 7.1, 90, 90, 90]


def _ase_setting(number, reference):
    """Align database origin choices, without deriving expected sites from Gemmi."""
    ase = pytest.importorskip('ase.spacegroup')
    expected_ops = {(tuple(np.asarray(op.rot).ravel()), tuple(np.asarray(op.tran) % op.DEN))
                    for op in reference.operations()}
    for setting in (1, 2):
        try:
            group = ase.Spacegroup(number, setting=setting)
        except ase.SpacegroupNotFoundError:
            continue
        actual_ops = {(tuple((rotation * 24).ravel()),
                       tuple(np.rint(np.mod(translation, 1) * 24).astype(int) % 24))
                      for rotation, translation in group.get_symop()}
        if actual_ops == expected_ops:
            return group
    raise AssertionError(f'ASE has no matching standard setting for {number}: {reference.xhm()}')


def _orbit_errors(actual, expected, cell):
    assert len(actual) == len(expected)
    delta = np.asarray(actual)[:, None, :] - np.asarray(expected)[None, :, :]
    delta -= np.rint(delta)
    costs = np.linalg.norm(delta @ cell, axis=2)
    rows, columns = linear_sum_assignment(costs)
    assert len(set(columns)) == len(expected)
    return float(np.max(np.abs(delta[rows, columns]))), float(np.max(costs[rows, columns]))


def compare_reference_group(number, directory):
    """Return measured independent-reference errors for a standard space group."""
    reference = gemmi.get_spacegroup_reference_setting(number)
    ase_group = _ase_setting(number, reference)
    parameters = _standard_cell(number)
    path = _write_cif(directory / f'sg_{number:03d}.cif', parameters,
                      f"_space_group_name_H-M_alt '{reference.xhm()}'\n_space_group_IT_number {number}")
    imported = read_cif(path)
    expected, kinds = ase_group.equivalent_sites([GENERAL_POSITION], onduplicates='error', symprec=1e-10)
    assert set(kinds) == {0}
    fractional_error, cartesian_error = _orbit_errors(imported.fractional, expected, imported.cell_vectors_A)
    np.testing.assert_allclose(imported.cell_vectors_A,
                               pytest.importorskip('ase.geometry').cellpar_to_cell(parameters), atol=3e-14)
    assert imported.metadata['spacegroup_number'] == number
    assert len(imported.basis) == len(ase_group.get_symop())
    assert fractional_error < 2e-14
    assert cartesian_error < 2e-13
    hkls = np.random.default_rng(SEED + number).integers(-7, 8, (31, 3))
    actual_f = np.exp(2j * np.pi * imported.fractional @ hkls.T).sum(axis=0)
    expected_f = np.exp(2j * np.pi * expected @ hkls.T).sum(axis=0)
    scattering_error = float(np.max(np.abs(actual_f - expected_f)))
    assert scattering_error < 2e-11
    return {'number': number, 'symbol': reference.xhm(), 'ase_setting': ase_group.setting,
            'atoms': len(imported.basis), 'max_fractional_error': fractional_error,
            'max_cartesian_error_A': cartesian_error, 'max_structure_factor_error': scattering_error}


@pytest.mark.parametrize('number', range(1, 231))
def test_all_230_standard_groups_against_ase(tmp_path, number, record_property):
    result = compare_reference_group(number, tmp_path)
    for key, value in result.items():
        record_property(key, value)


@pytest.mark.parametrize('number', ORIGIN_CHOICES)
@pytest.mark.parametrize('declaration', ['number', 'bare_hm'])
def test_all_24_ambiguous_origin_choices_require_explicit_selection(tmp_path, number, declaration):
    if declaration == 'number':
        symmetry = f'_space_group_IT_number {number}'
    else:
        symmetry = f"_space_group_name_H-M_alt '{gemmi.get_spacegroup_reference_setting(number).hm}'"
    path = _write_cif(tmp_path / 'ambiguous.cif', _standard_cell(number), symmetry)
    with pytest.raises(ValueError, match='Ambiguous.*origin'):
        read_cif(path)


@pytest.mark.parametrize('symbol,cell,expected', [
    ('P 21 1 1', [8, 9, 10, 106, 90, 90], [[.13, .27, .41], [.63, .73, .59]]),
    ('P 1 1 21', [8, 9, 10, 90, 90, 106], [[.13, .27, .41], [.87, .73, .91]]),
    ('R 3:R', [8, 8, 8, 76, 76, 76], [[.13, .27, .41], [.41, .13, .27], [.27, .41, .13]]),
])
def test_nonstandard_axis_and_rhombohedral_settings_have_manual_orbits(tmp_path, symbol, cell, expected):
    path = _write_cif(tmp_path / 'alternate.cif', cell, f"_space_group_name_H-M_alt '{symbol}'",
                      [('C1', 'C', [.13, .27, .41])])
    imported = read_cif(path)
    assert _orbit_errors(imported.fractional, expected, imported.cell_vectors_A)[1] < 1e-12


def test_hexagonal_rhombohedral_setting_has_manual_centering_orbit(tmp_path):
    seed = np.array([.13, .27, .41])
    x, y, z = seed
    rotations = np.array([[x, y, z], [-y, x-y, z], [y-x, -x, z]])
    centering = np.array([[0, 0, 0], [2/3, 1/3, 1/3], [1/3, 2/3, 2/3]])
    expected = np.mod((rotations[:, None, :] + centering).reshape(-1, 3), 1)
    path = _write_cif(tmp_path / 'hexagonal.cif', [8, 8, 10, 90, 90, 120],
                      "_space_group_name_H-M_alt 'R 3:H'", [('C1', 'C', seed)])
    imported = read_cif(path)
    assert _orbit_errors(imported.fractional, expected, imported.cell_vectors_A)[1] < 1e-12


@pytest.mark.parametrize('number,position', [(225, [0, 0, 0]), (225, [.5, .5, .5]),
    (227, [.125, .125, .125]), (221, [0, .5, .5]), (194, [1/3, 2/3, .25]),
    (166, [0, 0, 0]), (139, [0, 0, .5]), (62, [.13, .25, .27]), (14, [0, 0, 0])])
def test_special_position_multiplicity_against_ase(tmp_path, number, position):
    reference = gemmi.get_spacegroup_reference_setting(number)
    group = _ase_setting(number, reference)
    path = _write_cif(tmp_path / 'special.cif', _standard_cell(number),
                      f"_space_group_name_H-M_alt '{reference.xhm()}'", [('C1', 'C', position)])
    imported = read_cif(path)
    expected, _ = group.equivalent_sites([position], onduplicates='keep', symprec=1e-10)
    assert _orbit_errors(imported.fractional, expected, imported.cell_vectors_A)[1] < 1e-12
    assert len(imported.basis) < len(group.get_symop())


@pytest.mark.parametrize('number', [3, 16, 75, 143, 168, 195, 225, 230])
@pytest.mark.parametrize('long_axis', [1, 1e6])
def test_invalid_metric_not_hidden_by_length_scale(tmp_path, number, long_axis):
    cell = _standard_cell(number)
    cell[3] = 83
    cell[2] *= long_axis
    path = _write_cif(tmp_path / 'metric.cif', cell,
                      f"_space_group_name_H-M_alt '{gemmi.get_spacegroup_reference_setting(number).xhm()}'")
    with pytest.raises(ValueError, match='metric is incompatible'):
        read_cif(path)


@pytest.mark.parametrize('trial', range(25))
def test_skew_periodic_tolerance_against_exhaustive_image_search(trial):
    rng = np.random.default_rng(SEED + 1000 + trial)
    shear = rng.integers(2, 6)
    cell = np.array([[1, 0, 0], [shear + .23, .12, 0], [.4, .3, 2.]])
    delta = rng.uniform(-.5, .5, 3)
    extents = np.array([80, 18, 2])
    images = np.array(list(product(*(range(-n, n+1) for n in extents))))
    distances = np.linalg.norm((delta - images) @ cell, axis=1)
    distance = float(np.min(distances))
    assert distance > 1e-6
    # Certify that every image potentially closer than this result lies inside
    # the independently enumerated box, including images far outside +/- 1.
    assert np.all(np.abs(delta) + distance * np.linalg.norm(np.linalg.inv(cell), axis=0) < extents)
    for tolerance, count in [(distance * .9999, 2), (distance * 1.0001, 1)]:
        expanded = expand_fractional_sites([[0, 0, 0], delta], cell, ['x,y,z'], tolerance_A=tolerance)
        assert len(expanded) == count


@pytest.mark.parametrize('scale', [.01, 1, 1000])
def test_duplicate_tolerance_is_cartesian_and_respects_boundary(tmp_path, scale):
    tolerance = 1e-5
    cell = np.array([7.1, 8.3, 9.7]) * scale
    for fraction, count in [(.999, 1), (1.001, 2)]:
        offset = tolerance * fraction / cell[0]
        path = _write_cif(tmp_path / 'boundary.cif', [*cell, 90, 90, 90],
                          "_space_group_name_H-M_alt 'P 1'",
                          [('C1', 'C', [0, .2, .3]), ('C2', 'C', [1-offset, .2, .3])])
        imported = read_cif(path, duplicate_tolerance_A=tolerance)
        assert len(imported.basis) == count


@pytest.mark.parametrize('integer_shift', [[0, 0, 0], [7, -9, 12], [-102, 30, 51]])
def test_integer_translation_invariance_of_multispecies_orbits(tmp_path, integer_shift):
    base = [('C1', 'C', np.array([.137, .281, .419])), ('Zn1', 'Zn', np.array([.037, .117, .331]))]
    shifted = [(label, element, position + integer_shift) for label, element, position in base]
    declaration = "_space_group_name_H-M_alt 'P 21 21 21'"
    reference = read_cif(_write_cif(tmp_path / 'base.cif', [7, 8, 9, 90, 90, 90], declaration, base))
    actual = read_cif(_write_cif(tmp_path / 'shifted.cif', [7, 8, 9, 90, 90, 90], declaration, shifted))
    assert actual.basis.elements == reference.basis.elements
    np.testing.assert_allclose(actual.fractional, reference.fractional, atol=2e-14, rtol=0)


@pytest.mark.parametrize('angles', [[10, 10, 170], [179, 179, 179], [30, 30, 90]])
def test_impossible_cell_angles_fail_instead_of_producing_a_basis(tmp_path, angles):
    path = _write_cif(tmp_path / 'impossible.cif', [7, 8, 9, *angles], "_space_group_name_H-M_alt 'P 1'")
    with pytest.raises(ValueError, match='nonsingular finite|invalid or zero volume'):
        read_cif(path)


@pytest.mark.parametrize('lattice', ['bcc', 'fcc', 'diamond'])
def test_imported_lattice_structure_factors_match_closed_form_extinction_rules(tmp_path, lattice):
    from mofdebye.core import form_factors
    from mofdebye.directional import amplitude

    symbol, position, atom_count = {
        'bcc': ('I m -3 m', [0, 0, 0], 2),
        'fcc': ('F m -3 m', [0, 0, 0], 4),
        'diamond': ('F d -3 m:2', [.125, .125, .125], 8),
    }[lattice]
    path = _write_cif(tmp_path / f'{lattice}.cif', [7, 7, 7, 90, 90, 90],
                      f"_space_group_name_H-M_alt '{symbol}'", [('C1', 'C', position)])
    crystal = read_cif(path)
    assert len(crystal.basis) == atom_count
    hkl = np.array(list(product(range(-4, 5), repeat=3)))
    h, k, l = hkl.T
    parity = lambda integers: np.where(integers % 2 == 0, 1, -1)
    fcc_factor = 1 + parity(h+k) + parity(h+l) + parity(k+l)
    if lattice == 'bcc':
        lattice_factor = 1 + parity(h+k+l)
    elif lattice == 'fcc':
        lattice_factor = fcc_factor
    else:
        lattice_factor = 2 * fcc_factor * np.cos(np.pi * (h+k+l) / 4)
    vectors = hkl * (2*np.pi/7)
    atomic_factor = form_factors(['C'], np.linalg.norm(vectors, axis=1))['C']
    expected = atomic_factor * lattice_factor
    measured = amplitude(crystal.basis, vectors)
    np.testing.assert_allclose(measured, expected, atol=2e-12, rtol=2e-13)
    forbidden = np.abs(lattice_factor) < 1e-12
    assert np.count_nonzero(forbidden) > 100
    assert np.max(np.abs(measured[forbidden])**2) < 1e-23
    assert measured[np.all(hkl == 0, axis=1)][0].real == pytest.approx(atom_count * atomic_factor[np.all(hkl == 0, axis=1)][0])
