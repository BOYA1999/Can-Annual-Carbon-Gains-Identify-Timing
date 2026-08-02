import numpy as np

from scripts.run_adversarial_signals import signal_source_indices
from scripts.run_budget_ablation import relative_cap
from scripts.run_failure_boundary import LEVELS, SCENARIOS, SEEDS


def test_equal_information_signal_replacements_preserve_annual_indices():
    permutation = np.random.default_rng(20260802).permutation(365)
    dynamic = np.concatenate([signal_source_indices(day, "DYNAMIC", permutation) for day in range(365)])
    shifted = np.concatenate([signal_source_indices(day, "SHIFT_7D", permutation) for day in range(365)])
    permuted = np.concatenate([signal_source_indices(day, "PERMUTED_DAYS", permutation) for day in range(365)])
    expected = np.arange(8760)
    assert np.array_equal(np.sort(dynamic), expected)
    assert np.array_equal(np.sort(shifted), expected)
    assert np.array_equal(np.sort(permuted), expected)


def test_budget_cap_handles_positive_and_negative_references():
    assert relative_cap(10.0, 0.01) == 10.1
    assert relative_cap(-10.0, 0.08) == -9.2


def test_failure_boundary_grid_is_frozen():
    assert LEVELS == (0.125, 0.15, 0.175)
    assert len(SCENARIOS) == len(LEVELS) * len(SEEDS) == 9
