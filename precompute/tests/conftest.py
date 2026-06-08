"""Shared test fixtures (written — plumbing)."""
import pytest
import mlx.core as mx


@pytest.fixture
def sharp_logits():
    """Confident distribution — most mass on token 0."""
    return mx.array([8.0, 1.0, 0.5, 0.0, -2.0])


@pytest.fixture
def flat_logits():
    """Uncertain distribution — mass spread out (a 'fork')."""
    return mx.array([1.2, 1.0, 0.9, 0.8, 0.6])
