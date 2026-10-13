"""Sourced IRMOF-1 regressions with independent crystallographic expectations."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np
import pytest

from mofdebye.core import AtomCloud, concatenate, direct_iq
from mofdebye.crystal import read_cif
from mofdebye.directional import amplitude, intensity
from mofdebye.periodic import reciprocal_lattice
from mofdebye.runner import run_configuration, validate_config


DATA = Path(__file__).resolve().parents[1] / 'examples' / 'real_mof'
CIF_SHA256 = 'e5d0b3d72445c15cd2ed02a36fa2fdf3d5d13f6c21441984475449d48d3f0ad2'
COMPOSITION = {'C': 192, 'H': 96, 'O': 104, 'Zn': 32}
CONFIGURATIONS = ('periodic_single_crystal', 'finite_single_crystal',
                  'pxrd_finite', 'pxrd_bulk', 'pxrd_atom_cut')


@pytest.fixture(scope='module')
def crystal():
    return read_cif(DATA / 'IRMOF-1.cif')


def config(name):
    return json.loads((DATA / f'{name}.json').read_text(encoding='utf-8'))


def test_bundled_cif_and_license_match_pinned_source_hashes():
    provenance = json.loads((DATA / 'provenance.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((DATA / 'IRMOF-1.cif').read_bytes()).hexdigest() == CIF_SHA256
    assert provenance['cif']['sha256'] == CIF_SHA256
    license_record = provenance['license']
    assert hashlib.sha256((DATA / license_record['file']).read_bytes()).hexdigest() == license_record['sha256']


def test_real_mof_expansion_matches_known_cell_composition_and_special_positions(crystal):
    assert len(crystal.basis) == 424
    assert Counter(crystal.basis.elements) == COMPOSITION
    assert crystal.metadata['symmetry_operation_count'] == 192
    assert len(crystal.symops) == 192
    assert crystal.metadata['input_site_count'] == 7
    assert crystal.metadata['expanded_atom_count'] == 424
    np.testing.assert_allclose(crystal.cell_vectors_A, np.eye(3) * 25.832, atol=1e-12)
    assert crystal.metadata['cell_volume_A3'] == pytest.approx(17237.492730368)
    assert np.all((crystal.fractional >= 0) & (crystal.fractional < 1))

    # Gemmi's separate crystallographic expansion uses its own special-position
    # merging, rather than the project's merge code or the project's symop list.
    block = gemmi.cif.read_file(str(DATA / 'IRMOF-1.cif')).sole_block()
    structure = gemmi.make_small_structure_from_block(block)
    independent = structure.get_all_unit_cell_sites()
    expected = sorted((site.element.name, *np.round(np.asarray(site.fract.tolist()) % 1, 8))
                      for site in independent)
    actual = sorted((element, *np.round(position, 8))
                    for element, position in zip(crystal.basis.elements, crystal.fractional))
    assert actual == expected


def test_f_centering_systematic_absences_follow_imported_crystal(crystal):
    # F-centering permits integer hkl only when all indices have the same parity.
    # These mixed-parity reflections must cancel, independently of form factors.
    forbidden = np.array([[1, 0, 0], [1, 1, 0], [2, 1, 0], [1, 2, 3], [2, 2, 1]])
    permitted = np.array([[2, 0, 0], [2, 2, 0], [1, 1, 1]])
    reciprocal = reciprocal_lattice(crystal.cell_vectors_A)
    absent = amplitude(crystal.basis, forbidden @ reciprocal)
    allowed = amplitude(crystal.basis, permitted @ reciprocal)
    assert np.max(np.abs(allowed)) > 1
    assert np.max(np.abs(absent)) < 1e-10 * np.max(np.abs(allowed))


def test_finite_real_mof_factorization_matches_explicit_two_cell_atom_cloud(crystal):
    specification = config('finite_single_crystal')
    hkl = np.array([[0, 0, 0], [0.37, 0.12, -0.19], [1.9, 0, 0], [2, 0, 0], [2.1, 0, 0]])
    specification['sampling'] = {'hkl': hkl.tolist()}
    result = run_configuration(specification, DATA)
    explicit = concatenate([crystal.basis, crystal.basis.moved(crystal.cell_vectors_A[0])])
    expected = intensity(explicit, hkl @ reciprocal_lattice(crystal.cell_vectors_A))
    np.testing.assert_allclose(result['tables']['intensity']['total'], expected, rtol=2e-12, atol=1e-8)
    assert result['metadata']['input_processing']['host_atoms'] == 848
    assert result['metadata']['input_processing']['retained_cells'] == 2
    assert result['metadata']['actual_method'] == ['full_cell_amplitude_factorization']


def test_real_mof_atom_cut_has_expected_atoms_and_matches_independent_debye(crystal):
    specification = config('pxrd_atom_cut')
    specification['sampling'] = {'q': [0, 0.5, 1.7]}
    selected = np.linalg.norm(crystal.basis.xyz - [12.916, 12.916, 12.916], axis=1) <= 12.0
    cut = AtomCloud(tuple(e for e, keep in zip(crystal.basis.elements, selected) if keep),
                    crystal.basis.xyz[selected])
    assert len(cut) == 232
    assert Counter(cut.elements) == {'C': 96, 'H': 48, 'O': 56, 'Zn': 32}
    result = run_configuration(specification, DATA)
    np.testing.assert_allclose(result['tables']['intensity']['total'],
                               direct_iq(cut, specification['sampling']['q']), rtol=2e-12)
    assert result['metadata']['input_processing']['host_atoms'] == 232
    assert result['metadata']['surface_policy'] == 'unpassivated_atom_cut'


@pytest.mark.parametrize('name', CONFIGURATIONS)
def test_all_real_mof_examples_run_with_small_scattering_samples(name):
    specification = config(name)
    validate_config(specification)
    if specification['scattering_case'] == 'pxrd':
        specification['sampling'] = {'q': [0, 0.4, 1.0]}
    elif specification['boundary'] == 'bulk':
        specification['sampling'] = {'reflection_q_max': 1.0, 'hkl': [[0, 0, 0], [1, 0, 0]]}
    else:
        specification['sampling'] = {'hkl': [[0, 0, 0], [1.7, 0, 0], [2, 0, 0]]}
    result = run_configuration(specification, DATA)
    assert result['metadata']['schema_version'] == '1.1'
    assert result['metadata']['input_processing']['basis_atoms'] == 424
    assert result['metadata']['input_sha256'][str((DATA / 'IRMOF-1.cif').resolve())] == CIF_SHA256
    for table in result['tables'].values():
        for values in table.values():
            assert np.isfinite(values).all()
    if specification['boundary'] == 'bulk':
        # An ordered empty host has no static-disorder diffuse contribution.
        np.testing.assert_array_equal(result['tables']['diffuse']['diffuse'], 0)
        assert len(result['tables']['reflections']['q']) > 0
    else:
        assert np.all(result['tables']['intensity']['total'] >= 0)
    json.dumps(result['metadata'], allow_nan=False)
