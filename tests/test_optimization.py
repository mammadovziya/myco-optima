"""Tests for deterministic FBA/FVA, sensitivity and medium search."""

from __future__ import annotations

import math

import pytest

from myco_optima.models import build_model, effective_medium
from myco_optima.optimization import (
    analyse_sensitivities,
    medium_cost,
    optimise_medium,
    rank_sensitivities,
    run_fba,
    run_fva,
)


def _bounds(model) -> dict[str, tuple[float, float]]:
    return {reaction.id: reaction.bounds for reaction in model.reactions}


def test_fba_is_deterministic_and_does_not_mutate_input() -> None:
    model = build_model("trichoderma_reesei")
    before = _bounds(model)

    first = run_fba(model, {"Glucose": 9, "Ammonium sulfate": 7, "oxygen": 18})
    second = run_fba(model, {"Glucose": 9, "Ammonium sulfate": 7, "oxygen": 18})

    assert first == second
    assert first.status == "optimal"
    assert first.objective_value == pytest.approx(first.biomass_flux)
    assert 0 < first.yield_coefficient < 1
    assert _bounds(model) == before


def test_fba_reports_transport_capacity_capping() -> None:
    result = run_fba("aspergillus_niger", {"glucose": 10_000})
    assert result.medium["glucose"] == 40.0
    assert any("capped" in warning for warning in result.warnings)


def test_fva_is_finite_ordered_and_retains_growth_fraction() -> None:
    model = build_model("aspergillus_oryzae")
    before = _bounds(model)
    fba = run_fba(model)
    fva = run_fva(model, fraction_of_optimum=0.95)

    assert fva.status == "optimal"
    assert fva.ranges["BIOMASS"].minimum >= 0.95 * fba.growth_rate - 1e-7
    assert fva.ranges["BIOMASS"].maximum == pytest.approx(fba.growth_rate)
    for interval in fva.ranges.values():
        assert math.isfinite(interval.minimum)
        assert math.isfinite(interval.maximum)
        assert interval.minimum <= interval.maximum + 1e-9
    assert _bounds(model) == before


def test_fva_validates_fraction_and_reactions() -> None:
    with pytest.raises(ValueError, match="interval"):
        run_fva("aspergillus_niger", fraction_of_optimum=0)
    with pytest.raises(KeyError, match="Unknown FVA"):
        run_fva("aspergillus_niger", reaction_ids=["NOT_A_REACTION"])


def test_sensitivity_ranking_is_stable_and_table_friendly() -> None:
    first = rank_sensitivities("aspergillus_niger")
    second = rank_sensitivities("aspergillus_niger")
    analysis = analyse_sensitivities("aspergillus_niger")

    assert first == second == analysis.rankings
    assert [row.rank for row in first] == list(range(1, len(first) + 1))
    assert [row.impact for row in first] == sorted((row.impact for row in first), reverse=True)
    assert first[0].nutrient_id == "ammonium"
    assert analysis.evaluations == 1 + 2 * len(first)
    assert {"parameter", "elasticity", "impact"}.issubset(first[0].to_dict())


def test_sensitivity_does_not_mutate_model() -> None:
    model = build_model("fusarium_venenatum")
    before = _bounds(model)
    rank_sensitivities(model, {"glucose": 12, "ammonium": 9})
    assert _bounds(model) == before


def test_medium_optimisation_is_reproducible_and_budget_feasible() -> None:
    baseline = effective_medium(build_model("aspergillus_niger"))
    budget = medium_cost(baseline)

    first = optimise_medium("aspergillus_niger", budgets={"maximum_cost": budget})
    second = optimise_medium("aspergillus_niger", budgets={"maximum_cost": budget})

    assert first == second
    assert first.status == "optimal"
    assert first.estimated_cost <= budget + 1e-7
    assert (
        first.growth_rate
        >= first.target_fraction * max(row.growth_rate for row in first.candidates) - 1e-7
    )
    assert first.evaluations <= 28
    assert len(first.candidates) > 1


def test_mapping_candidates_match_streamlit_call_shape() -> None:
    result = optimise_medium(
        "aspergillus_niger",
        candidates={"Glucose": 16, "Ammonium sulfate": 6, "oxygen": 15, "phosphate": 1},
        budgets={"maximum_cost": 30},
    )
    assert result.baseline_medium["glucose"] == 16
    assert result.baseline_medium["ammonium"] == 6
    assert result.budget == 30
    assert result.objective_value == result.growth_rate


def test_nitrate_does_not_outperform_equal_ammonium_by_construction() -> None:
    ammonium = run_fba("aspergillus_niger", {"ammonium": 7}).growth_rate
    nitrate = run_fba("aspergillus_niger", {"nitrate": 7}).growth_rate
    assert nitrate <= ammonium + 1e-8
