"""CIF contracts and independent geometry checks for R07."""
from hashlib import sha256
import json

import gemmi
import numpy as np
import pytest

from mofdebye.crystal import expand_fractional_sites, read_cif


CELL = '''_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 90
'''
ATOM_TAGS = '''loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
'''


def cif(tmp_path, rows='C1 C 0.1 0.2 0.3 1', symmetry="_space_group_name_H-M_alt 'P 1'",
        cell=CELL, extra='', atom_tags=ATOM_TAGS, name='test'):
    text = f'data_{name}\n{cell}{symmetry}\n{extra}\n{atom_tags}{rows}\n'
    path = tmp_path / 'structure.cif'
    path.write_text(text, encoding='utf-8')
    return path


def test_p1_uncertainty_nonorthogonal_coordinates_and_provenance(tmp_path):
    cell = CELL.replace('10\n', '10.0(2)\n', 1).replace('alpha 90', 'alpha 80').replace('gamma 90', 'gamma 75')
    path = cif(tmp_path, rows='Zn1 Zn2+ -0.1(2) 1.2 0.300(4) 1.0(1)', cell=cell,
               extra='_atom_site_B_iso_or_equiv 0.4')
    crystal = read_cif(path)
    np.testing.assert_allclose(crystal.fractional, [[0.9, 0.2, 0.3]])
    expected = gemmi.UnitCell(10, 11, 12, 80, 100, 75).orthogonalize(gemmi.Fractional(.9, .2, .3))
    np.testing.assert_allclose(crystal.basis.xyz[0], list(expected), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(crystal.cell_vectors_A, axis=1), [10, 11, 12])
    assert crystal.basis.elements == ('Zn',)
    assert crystal.metadata['source_sha256'] == sha256(path.read_bytes()).hexdigest()
    assert crystal.metadata['block_name'] == 'test'
    assert crystal.metadata['retained_input_sites'][0]['charge'] == 2
    assert crystal.metadata['thermal_displacements']['present_tags'] == ['_atom_site_B_iso_or_equiv']
    assert crystal.metadata['expanded_atom_count'] == 1
    assert crystal.symops == ('x,y,z',)
    json.dumps(crystal.metadata)
    assert not crystal.fractional.flags.writeable
    assert not crystal.cell_vectors_A.flags.writeable


def test_inversion_special_position_full_occupancy_is_not_multiplied(tmp_path):
    crystal = read_cif(cif(tmp_path, rows='C1 C 0.1 0.2 0.3 1\nZn1 Zn 0 0 0 1',
                           symmetry="_symmetry_space_group_name_H-M 'P -1'"))
    np.testing.assert_allclose(crystal.fractional, [[.1, .2, .3], [.9, .8, .7], [0, 0, 0]])
    assert crystal.basis.elements == ('C', 'C', 'Zn')
    assert crystal.metadata['expanded_candidate_count'] == 4
    assert crystal.metadata['duplicates_removed'] == 1
    assert crystal.metadata['merged_input_sites'] == []


@pytest.mark.parametrize('tag', ['_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz'])
def test_explicit_symmetry_loops_without_group_symbol(tmp_path, tag):
    symmetry = f"loop_\n{tag}\n'x,y,z'\n'-x,y+1/2,-z'"
    crystal = read_cif(cif(tmp_path, symmetry=symmetry))
    np.testing.assert_allclose(crystal.fractional, [[.1, .2, .3], [.9, .7, .7]])
    assert crystal.metadata['spacegroup_number'] == 4
    assert crystal.metadata['symmetry_source'] == tag


@pytest.mark.parametrize('declaration', [
    "_space_group_name_H-M_alt 'P21/c'",
    "_symmetry_space_group_name_H-M 'P 1 21/c 1'",
    "_space_group_name_Hall '-P 2ybc'",
    "_symmetry_space_group_name_Hall '-P 2ybc'",
    '_space_group_IT_number 14',
    '_symmetry_Int_Tables_number 14',
])
def test_spacegroup_aliases_and_number_reference_setting(tmp_path, declaration):
    crystal = read_cif(cif(tmp_path, symmetry=declaration))
    assert crystal.metadata['spacegroup_number'] == 14
    assert len(crystal.basis) == 4
    if 'number' in declaration:
        assert crystal.metadata['spacegroup_setting_assumption']


def test_explicit_setting_and_redundant_symmetry_declarations_agree(tmp_path):
    ops = gemmi.find_spacegroup_by_name('P n n n:2').operations()
    symmetry = "_space_group_name_H-M_alt 'P n n n'\n_space_group_IT_number 48\nloop_\n_space_group_symop_operation_xyz\n"
    symmetry += '\n'.join(f"'{op.triplet()}'" for op in ops)
    crystal = read_cif(cif(tmp_path, symmetry=symmetry, cell=CELL.replace('beta 100', 'beta 90')))
    assert crystal.metadata['spacegroup_hm'] == 'P n n n:2'


@pytest.mark.parametrize('symmetry', ["_space_group_name_H-M_alt 'P n n n'", '_space_group_IT_number 48'])
def test_origin_ambiguity_is_rejected(tmp_path, symmetry):
    with pytest.raises(ValueError, match='Ambiguous.*origin'):
        read_cif(cif(tmp_path, symmetry=symmetry))


@pytest.mark.parametrize('symmetry', [
    "_space_group_name_H-M_alt 'P 1'\nloop_\n_space_group_symop_operation_xyz\n'x,y,z'\n'-x,-y,-z'",
    "_space_group_name_H-M_alt 'P 1'\n_space_group_name_Hall '-P 1'",
    "_space_group_name_H-M_alt 'P -1'\n_space_group_IT_number 1",
    "_space_group_IT_number 1\n_symmetry_Int_Tables_number 2",
])
def test_conflicting_symmetry_sources_fail(tmp_path, symmetry):
    with pytest.raises(ValueError, match='disagree|Conflicting'):
        read_cif(cif(tmp_path, symmetry=symmetry))


@pytest.mark.parametrize('symmetry,match', [
    ('', 'missing symmetry'),
    ("_space_group_name_H-M_alt 'UNKNOWN'", 'Unknown H-M'),
    ("loop_\n_space_group_symop_operation_xyz\n'-x,-y,-z'", 'identity'),
    ("loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n'x+1,y,z'", 'duplicate'),
    ("loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n'-y,x,z'", 'closed'),
    ("loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n'x/2,y,z'", 'non-crystallographic'),
    ("loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n'bad'", 'Invalid symmetry'),
    ('_space_group_IT_number 231', 'integer from 1 to 230'),
])
def test_invalid_or_incomplete_symmetry_fails(tmp_path, symmetry, match):
    with pytest.raises(ValueError, match=match):
        read_cif(cif(tmp_path, symmetry=symmetry))


def test_cell_metric_symmetry_mismatch_fails(tmp_path):
    with pytest.raises(ValueError, match='metric is incompatible'):
        read_cif(cif(tmp_path, symmetry="_space_group_name_H-M_alt 'P 4'"))


def test_long_cell_axis_cannot_mask_incompatible_perpendicular_metric(tmp_path):
    cell = CELL.replace('length_c 12', 'length_c 1000').replace('beta 100', 'beta 90')
    with pytest.raises(ValueError, match='metric is incompatible'):
        read_cif(cif(tmp_path, symmetry="_space_group_name_H-M_alt 'P 4'", cell=cell))


@pytest.mark.parametrize('tag', ['symmetry_multiplicity', 'site_symmetry_multiplicity'])
def test_declared_multiplicity_detects_rounded_special_positions(tmp_path, tag):
    cell = CELL.replace('length_b 11', 'length_b 10').replace('length_c 12', 'length_c 10')
    cell = cell.replace('beta 100', 'beta 90').replace('gamma 90', 'gamma 120')
    path = cif(tmp_path, symmetry="_space_group_name_H-M_alt 'P 3'", cell=cell,
               atom_tags=ATOM_TAGS + f'_atom_site_{tag}\n', rows='C1 C .3333 .6667 0 1 1')
    with pytest.raises(ValueError, match='declares symmetry multiplicity.*coordinates produce 3'):
        read_cif(path)
    # An explicit larger tolerance is a user modeling choice; no coordinates are snapped.
    crystal = read_cif(path, duplicate_tolerance_A=.002)
    assert len(crystal.basis) == 1
    assert crystal.metadata['retained_input_sites'][0]['declared_symmetry_multiplicity'] == 1


@pytest.mark.parametrize('multiplicity', ['0', '-1', '1.5', '?'])
def test_declared_multiplicity_must_be_known_positive_integer(tmp_path, multiplicity):
    path = cif(tmp_path, atom_tags=ATOM_TAGS + '_atom_site_symmetry_multiplicity\n',
               rows=f'C1 C .1 .2 .3 1 {multiplicity}')
    with pytest.raises(ValueError, match='positive integer|finite number'):
        read_cif(path)


def test_exact_special_and_general_position_multiplicities(tmp_path):
    path = cif(tmp_path, symmetry="_space_group_name_H-M_alt 'P -1'",
               atom_tags=ATOM_TAGS + '_atom_site_symmetry_multiplicity\n',
               rows='C1 C .1 .2 .3 1 2\nZn1 Zn 0 0 0 1 1')
    crystal = read_cif(path)
    assert len(crystal.basis) == 3
    assert [site['expanded_orbit_count'] for site in crystal.metadata['retained_input_sites']] == [2, 1]


def test_duplicate_sites_deduplicate_across_boundary_with_metadata(tmp_path):
    path = cif(tmp_path, rows='C1 C 0.1 0.2 0.3 1\nC2 C 1.1 0.2 0.3 1')
    crystal = read_cif(path)
    assert len(crystal.basis) == 1
    assert crystal.metadata['merged_input_sites'] == [{'kept_label': 'C1', 'merged_label': 'C2', 'element': 'C'}]
    with pytest.raises(ValueError, match='Duplicate equivalent'):
        read_cif(path, duplicate_policy='error')


def test_different_species_same_position_is_never_merged(tmp_path):
    path = cif(tmp_path, rows='C1 C 0.1 0.2 0.3 1\nN1 N 0.9 0.8 0.7 1',
               symmetry="_space_group_name_H-M_alt 'P -1'")
    with pytest.raises(ValueError, match='Conflicting species'):
        read_cif(path)


@pytest.mark.parametrize('occupancy', ['0.5', '0', '0.999999'])
def test_partial_occupancy_must_be_rejected_or_explicitly_dropped(tmp_path, occupancy):
    path = cif(tmp_path, rows=f'C1 C 0.1 0.2 0.3 1\nO1 O 0.4 0.5 0.6 {occupancy}')
    with pytest.raises(ValueError, match='partial occupancy'):
        read_cif(path)
    crystal = read_cif(path, occupancy_policy='drop_partial')
    assert len(crystal.basis) == 1
    assert crystal.metadata['excluded_sites'][0] == {
        'label': 'O1', 'element': 'O', 'occupancy': float(occupancy), 'reason': 'drop_partial'}
    assert crystal.metadata['retained_input_site_count'] == 1


@pytest.mark.parametrize('occupancy,match', [('.', 'finite number'), ('?', 'finite number'),
                                          ('-0.1', 'outside'), ('1.1', 'outside')])
def test_invalid_occupancy_cannot_be_silently_dropped(tmp_path, occupancy, match):
    path = cif(tmp_path, rows=f'C1 C 0.1 0.2 0.3 {occupancy}')
    with pytest.raises(ValueError, match=match):
        read_cif(path, occupancy_policy='drop_partial')


def test_missing_occupancy_column_has_documented_full_occupancy_assumption(tmp_path):
    crystal = read_cif(cif(tmp_path, rows='C1 C .1 .2 .3',
                           atom_tags=ATOM_TAGS.replace('_atom_site_occupancy\n', '')))
    assert crystal.metadata['occupancy_missing_tag_assumption'] == 'one'


@pytest.mark.parametrize('suffix', ['disorder_group', 'disorder_assembly'])
def test_disorder_requires_explicit_exclusion_even_for_full_occupancy(tmp_path, suffix):
    path = cif(tmp_path, rows='C1 C .1 .2 .3 1 .\nO1 O .2 .3 .4 1 1',
               atom_tags=ATOM_TAGS + f'_atom_site_{suffix}\n')
    with pytest.raises(ValueError, match=suffix):
        read_cif(path)
    crystal = read_cif(path, exclude_labels=['O1'])
    assert len(crystal.basis) == 1
    assert crystal.metadata['excluded_sites'][0]['reason'] == 'exclude_labels'


def test_hydrogens_solvent_retained_by_default_and_only_explicitly_removed(tmp_path):
    path = cif(tmp_path, rows='C1 C .1 .2 .3 1\nH1 H .2 .3 .4 1\nOsol O .3 .4 .5 1')
    assert read_cif(path).basis.elements == ('C', 'H', 'O')
    stripped = read_cif(path, exclude_labels=['Osol'], exclude_elements=['H'])
    assert stripped.basis.elements == ('C',)
    assert [item['label'] for item in stripped.metadata['excluded_sites']] == ['H1', 'Osol']
    with pytest.raises(ValueError, match='do not exist'):
        read_cif(path, exclude_labels=['typo'])


def test_multiblock_requires_explicit_selection(tmp_path):
    path = cif(tmp_path, name='first')
    text = path.read_text(encoding='utf-8')
    path.write_text(text + text.replace('data_first', 'data_second'), encoding='utf-8')
    with pytest.raises(ValueError, match='explicit block'):
        read_cif(path)
    assert read_cif(path, block='second').metadata['block_name'] == 'second'
    with pytest.raises(ValueError, match='not found'):
        read_cif(path, block='absent')


@pytest.mark.parametrize('changes,match', [
    ({'cell': CELL.replace('_cell_length_a 10\n', '')}, 'requires _cell_length_a'),
    ({'cell': CELL.replace('length_a 10', 'length_a -1')}, 'lengths must be positive'),
    ({'cell': CELL.replace('angle_beta 100', 'angle_beta 0')}, 'angles must lie'),
    ({'rows': 'C1 C ? .2 .3 1'}, 'finite number'),
    ({'rows': 'C1 C .1 .2 .3 1\nC1 C .3 .2 .1 1'}, 'unique atom-site labels'),
    ({'rows': 'X1 Xx .1 .2 .3 1'}, 'Unknown chemical element'),
    ({'rows': 'X1 Crap .1 .2 .3 1'}, 'Invalid chemical type symbol'),
])
def test_missing_or_malformed_cell_and_site_fields_fail(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        read_cif(cif(tmp_path, **changes))


def test_malformed_cif_syntax_is_contextual_error(tmp_path):
    path = tmp_path / 'broken.cif'
    path.write_text("data_test\nloop_\n_label\n_coord\nC1\n", encoding='utf-8')
    with pytest.raises(ValueError, match='Cannot parse CIF broken.cif'):
        read_cif(path)


def test_expansion_of_pore_seeds_wraps_and_deduplicates_special_positions():
    positions = expand_fractional_sites([[.1, .2, .3], [.9, .8, .7], [1, 0, 0]],
                                       np.diag([10, 11, 12]), ['x,y,z', '-x,-y,-z'])
    np.testing.assert_allclose(positions, [[.1, .2, .3], [.9, .8, .7], [0, 0, 0]])


def test_periodic_dedup_uses_cartesian_metric_in_a_skew_cell():
    # Fractional rounding chooses (0,0,0), whose Cartesian residual is ~0.98 A;
    # another image, subtracting b, has distance ~0.02 A. Exact metric search
    # must find it even though neither fractional component is near a boundary.
    cell = np.array([[1., 0, 0], [1., .01, 0], [0, 0, 1.]])
    delta = np.array([.49, .49, 0.])
    assert np.linalg.norm((delta - np.rint(delta)) @ cell) > .9
    images = np.asarray([delta - shift for shift in [(i, j, 0) for i in range(-2, 3) for j in range(-2, 3)]])
    assert np.min(np.linalg.norm(images @ cell, axis=1)) < .03
    positions = expand_fractional_sites([[0, 0, 0], delta], cell, ['x,y,z'], tolerance_A=.03)
    assert len(positions) == 1


@pytest.mark.parametrize('option', [{'occupancy_policy': 'promote'}, {'duplicate_policy': 'discard'},
                                  {'duplicate_tolerance_A': 0}, {'duplicate_tolerance_A': float('nan')},
                                  {'exclude_labels': 'C1'}, {'exclude_elements': ['nope']}])
def test_bad_import_policy_parameters_are_rejected(tmp_path, option):
    with pytest.raises(ValueError):
        read_cif(cif(tmp_path), **option)


def test_all_excluded_sites_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match='leave no fully occupied atoms'):
        read_cif(cif(tmp_path), exclude_elements=['C'])
