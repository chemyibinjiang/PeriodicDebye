"""Independent field checks and exact finite-grid convolution acceptance."""
import numpy as np
import pytest

from magnetooptics.interaction import DenseGreenOperator, FFTGreenOperator, green_tensor


def reference_field(positions, sources, k):
    output = np.zeros_like(sources, dtype=complex)
    for i, observation in enumerate(positions):
        for j, source in enumerate(positions):
            if i == j:
                continue
            r = observation - source
            distance = np.linalg.norm(r)
            unit = r / distance
            p = sources[j]
            output[i] += np.exp(1j*k*distance)/(4*np.pi) * (
                k*k/distance*np.cross(np.cross(unit, p), unit)
                + (1/distance**3 - 1j*k/distance**2)*(3*unit*np.dot(unit, p)-p))
    return output


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (4, 1, 3), (3, 2, 1)])
@pytest.mark.parametrize("sparse", [False, True])
def test_fft_is_finite_linear_convolution(shape, sparse):
    index = np.indices(shape).reshape(3, -1).T
    if sparse and len(index) > 2:
        index = index[::2]
    index = index - [4, 2, 1]
    spacing = np.array([17., 29., 41.])
    positions = index * spacing
    rng = np.random.default_rng(73)
    p = rng.normal(size=(len(index), 3)) + 1j*rng.normal(size=(len(index), 3))
    k = .013
    expected = reference_field(positions, p, k)
    dense = DenseGreenOperator(positions, k)
    fast = FFTGreenOperator(index, spacing, k, shape=shape)
    np.testing.assert_allclose(dense.apply(p), expected, rtol=8e-15, atol=2e-20)
    np.testing.assert_allclose(fast.apply(p), expected, rtol=1e-13, atol=2e-20)
    assert fast.metadata["boundary"] == "finite_open"
    assert np.all(np.asarray(fast.padded_shape) >= 2*np.asarray(shape)-1)


def test_green_longitudinal_transverse_and_zero():
    radius, k = 23., .021
    tensor = green_tensor([radius, 0, 0], k)
    phase = np.exp(1j*k*radius)/(4*np.pi*radius**3)
    np.testing.assert_allclose(np.diag(tensor), phase*np.array([2*(1-1j*k*radius),
        (k*radius)**2-1+1j*k*radius, (k*radius)**2-1+1j*k*radius]), rtol=2e-15)
    np.testing.assert_array_equal(tensor - np.diag(np.diag(tensor)), np.zeros((3, 3)))
    np.testing.assert_array_equal(green_tensor([0, 0, 0], k), np.zeros((3, 3)))


def test_corner_sources_do_not_wrap_as_periodic():
    indices = np.array([[0, 0, 0], [9, 0, 0]])
    p = np.array([[1., 2., 3.], [0., 0., 0.]], dtype=complex)
    fast = FFTGreenOperator(indices, 31., .01)
    actual = fast.apply(p)
    np.testing.assert_allclose(actual[1], green_tensor([279., 0, 0], .01) @ p[0], atol=1e-21)
    assert not np.allclose(actual[1], green_tensor([31., 0, 0], .01) @ p[0], atol=1e-10)


@pytest.mark.parametrize("positions", [[], [[0, 0]], [[0, 0, np.nan]], [[0, 0, 0], [0, 0, 0]]])
def test_dense_rejects_bad_positions(positions):
    with pytest.raises(ValueError):
        DenseGreenOperator(positions, .01)


@pytest.mark.parametrize("indices", [[], [[0, 0]], [[0, 0, .5]], [[0, 0, np.inf]],
    [[0, 0, 0], [0, 0, 0]], [[True, False, True]], [[0, 0, 2**51]]])
def test_fft_rejects_bad_indices(indices):
    with pytest.raises(ValueError):
        FFTGreenOperator(indices, 10., .01)


@pytest.mark.parametrize("spacing", [0, -1, np.nan, [1, 2], [1, 0, 2]])
def test_fft_rejects_bad_spacing(spacing):
    with pytest.raises(ValueError):
        FFTGreenOperator([[0, 0, 0]], spacing, .01)


@pytest.mark.parametrize("k", [0, -1, np.inf, 1j, True])
def test_rejects_bad_wavenumber(k):
    with pytest.raises(ValueError):
        green_tensor([1, 0, 0], k)


@pytest.mark.parametrize("shape", [[0, 1, 1], [1., 1., 1.], [True, True, True], [1, 1], [1, 1, 1]])
def test_fft_rejects_bad_shape(shape):
    with pytest.raises(ValueError):
        FFTGreenOperator([[0, 0, 0], [2, 0, 0]], 10., .01, shape=shape)


def test_explicit_memory_guards_precede_large_allocations():
    with pytest.raises(MemoryError):
        DenseGreenOperator(np.arange(300).reshape(100, 3), .01, max_memory_bytes=10000)
    with pytest.raises(MemoryError):
        FFTGreenOperator([[0, 0, 0], [1_000_000, 1_000_000, 1_000_000]], 10., .01)


@pytest.mark.parametrize("operator", [DenseGreenOperator([[0, 0, 0]], .01), FFTGreenOperator([[0, 0, 0]], 10., .01)])
def test_input_copy_and_application_validation(operator):
    with pytest.raises(ValueError):
        operator.apply([[1, 2]])
    with pytest.raises(ValueError):
        operator.apply([[1, 2, np.nan]])
    assert not operator.positions_nm.flags.writeable


def test_reject_complex_coordinates_and_boolean_lengths():
    with pytest.raises(ValueError, match="real"):
        DenseGreenOperator([[1j, 0, 0]], .01)
    with pytest.raises(ValueError, match="real"):
        green_tensor([1j, 0, 0], .01)
    for spacing in (True, 1+1j):
        with pytest.raises(ValueError):
            FFTGreenOperator([[0, 0, 0]], spacing, .01)
