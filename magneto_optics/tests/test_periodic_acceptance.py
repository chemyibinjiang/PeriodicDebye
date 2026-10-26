"""Independent periodic physics oracles, shared with the reproducible report.

The report implements static zeta and dynamic damped-real-space references,
open-channel radiation identities, circular sheet solutions, independent
material heating, and primitive/supercell equivalence. No stored golden output.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from magnetooptics.materials import gyrotropic_permittivity
from magnetooptics.periodic_geometry import RectangularCell

_script = Path(__file__).resolve().parents[1] / "scripts" / "validate_periodic.py"
_spec = importlib.util.spec_from_file_location("periodic_acceptance_oracles", _script)
oracles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracles)


@pytest.fixture(scope="module")
def acceptance():
    return oracles.acceptance_report()


@pytest.mark.parametrize("name", [
    "square_static_zeta_reference", "independent_Abel_refinement",
    "dynamic_Ewald_vs_independent_Abel", "radiative_self_all_open_channels",
    "closed_circular_sheet_fields", "Faraday_Kerr_eigenchannel_angles_deg",
    "Onsager_reversal_Jones", "g_zero_circular_degeneracy",
    "all_order_energy_conservation", "independent_material_absorption",
    "diffracted_orders_carry_nonzero_power", "primitive_supercell_field_invariance",
    "primitive_supercell_power_invariance", "folded_orders_cancel",
])
def test_independent_physical_acceptance(acceptance, name):
    check = next(item for item in acceptance["criteria"] if item["name"] == name)
    assert check["passed"], check


def test_translation_through_cell_boundary_preserves_power_and_correct_order_phase():
    periods = np.array([900, 760])
    original = np.array([[80, 93, 0], [400, 350, 0]])
    displacement = np.array([421, 503])
    translated = original.copy()
    translated[:, :2] = (translated[:, :2] + displacement) % periods
    epsilon = gyrotropic_permittivity(-1.8 + .2j, .05 + .01j)
    ports, solution = oracles.solve_ports(RectangularCell(periods, original, 14), 500, epsilon, 14, [1, 0, 0])
    shifted, shifted_solution = oracles.solve_ports(RectangularCell(periods, translated, 14), 500, epsilon, 14, [1, 0, 0])
    assert_allclose(shifted_solution.dipoles, solution.dipoles, rtol=2e-11, atol=1e-9)
    transverse = 2 * np.pi * ports["order_indices"] / periods
    phases = np.exp(-1j * (transverse @ displacement))
    for key in ("transmitted_fields", "reflected_fields"):
        assert_allclose(shifted[key], phases[:, None] * ports[key], rtol=2e-11, atol=1e-12)
    for key in ("transmitted_power", "reflected_power", "total_transmission", "total_reflection", "absorption"):
        assert_allclose(shifted[key], ports[key], rtol=2e-11, atol=1e-12)


def test_nonidentical_basis_permutation_preserves_physical_scattering():
    positions = np.array([[17, 21, 0], [90, 33, 0], [62, 142, 0]])
    radii = np.array([6.0, 8.0, 7.0])
    epsilons = np.array([gyrotropic_permittivity(epsilon, g) for epsilon, g in [(2.2 + .1j, .02), (-1.8 + .2j, .05 + .01j), (3 + .1j, .03)]])
    cell = RectangularCell([220, 240], positions, radii)
    permutation = np.array([2, 0, 1])
    permuted = RectangularCell(cell.periods_nm, positions[permutation], radii[permutation])
    ports, solution = oracles.solve_ports(cell, 620, epsilons, radii, [1, 0, 0])
    reordered, reordered_solution = oracles.solve_ports(permuted, 620, epsilons[permutation], radii[permutation], [1, 0, 0])
    assert_allclose(reordered_solution.dipoles, solution.dipoles[permutation], rtol=2e-11, atol=1e-9)
    for key in ("transmitted_fields", "reflected_fields", "total_transmission", "total_reflection", "absorption"):
        assert_allclose(reordered[key], ports[key], rtol=2e-11, atol=1e-12)
