from src.evaluate import distinct_nondominated_count, exact_shapley, harsanyi_dividends, pareto_status, select_knee, select_practical_policy
from scripts.run_shapley import _relative_cap, _resource_key


def _metrics(costs, emissions):
    return {
        policy: {"annual_operating_cost_usd": cost, "annual_emissions_kgco2e": emissions[policy]}
        for policy, cost in costs.items()
    }


def test_practical_policy_uses_only_highest_cost_eligible_level():
    costs = {"F0": 100.0, "F1": 101.0, "F2": 102.0, "F3": 103.0, "F4": 104.0, "F5": 105.1, "F6": 120.0}
    emissions = {policy: 100.0 - index for index, policy in enumerate(costs)}
    assert select_practical_policy(_metrics(costs, emissions)) == "F4"


def test_pareto_and_knee_selection_are_deterministic():
    costs = {"F0": 100.0, "F1": 101.0, "F2": 102.0, "F3": 103.0, "F4": 104.0, "F5": 105.0, "F6": 110.0}
    emissions = {"F0": 100.0, "F1": 95.0, "F2": 91.0, "F3": 88.0, "F4": 86.0, "F5": 85.0, "F6": 84.0}
    metrics = _metrics(costs, emissions)
    status = pareto_status(metrics)
    assert all(status.values())
    assert distinct_nondominated_count(metrics, status) == 7
    assert select_knee(metrics, status) == "F4"


def test_exact_shapley_and_harsanyi_close_for_additive_game():
    features = ("A", "B", "C")
    weights = {"A": 2.0, "B": 3.0, "C": 5.0}
    values = {
        frozenset(coalition): sum(weights[feature] for feature in coalition)
        for size in range(4)
        for coalition in __import__("itertools").combinations(features, size)
    }
    shapley = exact_shapley(values)
    dividends = harsanyi_dividends(values)
    assert shapley == weights
    assert dividends["AB"] == 0
    assert dividends["ABC"] == 0
    assert sum(shapley.values()) == values[frozenset(features)] - values[frozenset()]


def test_relative_budget_is_shared_only_within_resource_pairs():
    assert _resource_key("A0B0C1") == _resource_key("A1B0C1") == "B0C1"
    assert _resource_key("A0B1C0") != _resource_key("A1B1C1")
    assert _relative_cap(10.0) == 10.4
    assert _relative_cap(-10.0) == -9.6
