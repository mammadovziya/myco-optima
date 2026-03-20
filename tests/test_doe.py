import pandas as pd
import pytest

from myco_optima.doe import Factor, build_sensitivity_guided_design


@pytest.fixture
def four_factors():
    return [
        Factor("Carbon", 10, 30, "g/L"),
        Factor("Nitrogen", 1, 7, "g/L"),
        Factor("Oxygen", 2, 18, "mmol/gDW/h"),
        Factor("Phosphate", 0.2, 1.2, "g/L"),
    ]


def test_reduces_81_candidates_to_15_runs(four_factors):
    design = build_sensitivity_guided_design(
        four_factors,
        {"Carbon": 0.8, "Nitrogen": 0.6, "Oxygen": 0.4, "Phosphate": 0.1},
    )

    assert design.candidate_count == 81
    assert design.selected_count == 15
    assert design.retained_factors == ("Carbon", "Nitrogen", "Oxygen")
    assert design.reduction_percent == pytest.approx(81.48148, rel=1e-5)


def test_box_behnken_has_12_edges_and_three_centres(four_factors):
    design = build_sensitivity_guided_design(
        four_factors, ["Nitrogen", "Oxygen", "Carbon", "Phosphate"]
    )
    centres = design.runs.loc[design.runs["design_role"] == "centre replicate"]
    edges = design.runs.loc[design.runs["design_role"] == "interaction edge"]

    assert len(centres) == 3
    assert len(edges) == 12
    assert edges.drop(columns=["run", "design_role"]).drop_duplicates().shape[0] == 12
    assert set(centres["Nitrogen"]) == {4}
    assert set(centres["Oxygen"]) == {10}
    assert set(centres["Carbon"]) == {20}


def test_absolute_sensitivity_sets_rank_and_output_bounds(four_factors):
    design = build_sensitivity_guided_design(
        four_factors,
        {"Carbon": 0.2, "Nitrogen": -0.9, "Oxygen": 0.4, "Phosphate": 0.1},
    )

    assert design.retained_factors[0] == "Nitrogen"
    for factor in four_factors:
        if factor.name in design.retained_factors:
            assert design.runs[factor.name].between(factor.low, factor.high).all()


def test_selection_is_deterministic(four_factors):
    ranking = {"Carbon": 0.8, "Nitrogen": 0.6, "Oxygen": 0.4, "Phosphate": 0.1}
    first = build_sensitivity_guided_design(four_factors, ranking).runs
    second = build_sensitivity_guided_design(four_factors, ranking).runs
    pd.testing.assert_frame_equal(first, second)


def test_rejects_invalid_factor_bounds():
    invalid = [Factor("A", 1, 1), Factor("B", 0, 1), Factor("C", 0, 1)]
    with pytest.raises(ValueError, match="low < high"):
        build_sensitivity_guided_design(invalid, ["A", "B", "C"])
