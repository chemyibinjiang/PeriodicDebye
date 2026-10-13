"""Reject ambiguous sampling limits rather than silently changing the model."""
import numpy as np
import pytest

from mofdebye.core import AtomCloud
from mofdebye.distributions import ParticleLibrary, PorePopulation


@pytest.mark.parametrize('limit', [True, False, 1.9, 2.0, float('nan'), float('inf'), '2', 0, -2])
def test_rejection_attempt_limit_requires_a_positive_integer(limit):
    library = ParticleLibrary((AtomCloud(('C',), [[0, 0, 0]]),), [1])
    with pytest.raises(ValueError, match='max_attempts must be a positive integer'):
        PorePopulation([[0, 0, 0]], library, max_attempts=limit)


def test_numpy_integer_rejection_limit_is_supported():
    library = ParticleLibrary((AtomCloud(('C',), [[0, 0, 0]]),), [1])
    population = PorePopulation([[0, 0, 0]], library, max_attempts=np.int64(3))
    assert population.max_attempts == 3
