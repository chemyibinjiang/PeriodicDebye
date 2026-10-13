"""Optional integration tests of the real installed package, never the vendor."""
import importlib.metadata
import numpy as np
import pytest
import torch
from mofdebye.core import AtomCloud, form_factors
from mofdebye.installed import installed_iq, native_form_factors

try:
    _version = importlib.metadata.version('debyecalculator')
except importlib.metadata.PackageNotFoundError:
    _version = None
pytestmark = pytest.mark.skipif(_version != '1.0.14', reason='verified optional debyecalculator 1.0.14 is not installed')


@pytest.mark.parametrize('elements,xyz', [
    (('Au',), [[1, 2, 3]]),
    (('Au', 'O'), [[0, 0, 0], [3, 0, 0]]),
    (('C', 'C'), [[0, 0, 0], [0, 0, 0]]),
])
def test_public_package_full_scale_from_tabulated_factors(elements, xyz):
    q = np.linspace(0, 8, 81)
    cloud = AtomCloud(elements, xyz)
    f = native_form_factors(elements, q)
    expected = sum(f[el] ** 2 for el in elements)
    if len(elements) == 2:
        r = np.linalg.norm(cloud.xyz[1] - cloud.xyz[0])
        expected += 2 * f[elements[0]] * f[elements[1]] * np.sinc(q * r / np.pi)
    result, info = installed_iq(cloud, q, return_info=True)
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=1e-10)
    assert info['native_scale'] == .5
    assert info['actual_method'] == 'batched_direct'
    assert info['grid_available'] is False
    assert info['device'] == 'cpu'


def test_native_factor_table_has_distinct_known_forward_coefficients():
    # Au coefficients in installed Waasmaier--Kirfel table, independent of iq.
    expected = sum([16.77739, 19.317156, 32.979683, 5.595453, 10.576854]) - 6.279078
    native = native_form_factors(('Au',), [0])['Au'][0]
    assert native == pytest.approx(expected, abs=1e-12)
    assert abs(native - form_factors(('Au',), [0])['Au'][0]) > 1e-4


def test_mixed_species_against_direct_pair_enumeration():
    rng = np.random.default_rng(371)
    cloud = AtomCloud(tuple(['Au', 'C', 'Zn', 'O'] * 4), rng.normal(size=(16, 3)))
    q = np.linspace(0, 8, 81)
    f = native_form_factors(cloud.elements, q)
    expected = np.zeros_like(q)
    for i in range(len(cloud)):
        for j in range(len(cloud)):
            distance = np.linalg.norm(cloud.xyz[i] - cloud.xyz[j])
            expected += f[cloud.elements[i]] * f[cloud.elements[j]] * np.sinc(q * distance / np.pi)
    np.testing.assert_allclose(installed_iq(cloud, q), expected, rtol=2e-12, atol=1e-9)


@pytest.mark.parametrize('q', [[0, .1, .3], [.2, .1], [1, 1], [[1, 2]], [-1]])
def test_incompatible_q_rejected(q):
    with pytest.raises(ValueError):
        installed_iq(AtomCloud(('C',), [[0, 0, 0]]), q)


def test_empty_single_q_and_dtype_restoration():
    original = torch.get_default_dtype()
    result = installed_iq(AtomCloud((), np.empty((0, 3))), [.375])
    np.testing.assert_array_equal(result, [0])
    carbon = AtomCloud(('C',), [[0, 0, 0]])
    np.testing.assert_allclose(installed_iq(carbon, [.375]), native_form_factors(('C',), [.375])['C'] ** 2)
    assert torch.get_default_dtype() == original


def test_unavailable_cuda_rejected_without_fallback():
    if torch.cuda.is_available():
        pytest.skip('CUDA is available on this machine')
    with pytest.raises(RuntimeError, match='no silent CPU fallback'):
        installed_iq(AtomCloud(('C',), [[0, 0, 0]]), [0, .1], device='cuda')
