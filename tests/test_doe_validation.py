"""Mathematical checks and synthetic limits for the 15-run design."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from myco_optima.doe import Factor, build_sensitivity_guided_design
from myco_optima.doe_validation import (
    run_synthetic_quadratic_benchmark,
    validate_box_behnken_design,
)


@pytest.fixture
def four_factors() -> tuple[Factor, ...]:
    return (
        Factor("Carbon", 10, 30, "g/L"),
        Factor("Nitrogen", 1, 7, "g/L"),
        Factor("Oxygen", 2, 18, "mmol/gDW/h"),
        Factor("Phosphate", 0.2, 1.2, "g/L"),
    )


@pytest.fixture
def sensitivities() -> dict[str, float]:
    return {"Carbon": 0.8, "Nitrogen": 0.6, "Oxygen": 0.4, "Phosphate": 0.1}


def test_default_design_is_balanced_orthogonal_and_quadratic_full_rank(
    four_factors: tuple[Factor, ...], sensitivities: dict[str, float]
) -> None:
    design = build_sensitivity_guided_design(four_factors, sensitivities)
    report = validate_box_behnken_design(design, four_factors)

    assert report.is_valid
    assert report.run_count == 15
    assert report.edge_count == 12
    assert report.unique_edge_count == 12
    assert report.centre_count == 3
    assert report.level_counts == ((4, 7, 4),) * 3
    assert report.linear_column_sums == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert report.maximum_absolute_linear_cross_product == pytest.approx(0.0, abs=1e-12)
    assert report.quadratic_parameter_count == 10
    assert report.quadratic_rank == 10
    assert report.residual_degrees_of_freedom == 5
    assert report.conditional_pure_error_degrees_of_freedom == 2


def test_validation_detects_a_corrupted_edge(
    four_factors: tuple[Factor, ...], sensitivities: dict[str, float]
) -> None:
    design = build_sensitivity_guided_design(four_factors, sensitivities)
    corrupted_runs = design.runs.copy()
    corrupted_runs.loc[0, design.retained_factors[0]] = 999.0
    corrupted = replace(design, runs=corrupted_runs)

    report = validate_box_behnken_design(corrupted, four_factors)

    assert not report.is_valid
    assert not report.coded_levels_valid
    assert not report.edge_structure_valid
    assert not report.level_balance_valid


def test_noise_free_quadratic_is_recovered_on_all_81_reference_conditions(
    four_factors: tuple[Factor, ...], sensitivities: dict[str, float]
) -> None:
    result = run_synthetic_quadratic_benchmark(four_factors, sensitivities)

    assert result.training_runs == 15
    assert result.reference_grid_conditions == 81
    assert result.retained_factors == ("Carbon", "Nitrogen", "Oxygen")
    assert result.dropped_factor == "Phosphate"
    assert result.exact_aligned_recovery
    assert result.coefficient_maximum_absolute_error < 1e-12
    assert result.aligned_grid_rmse < 1e-12
    assert result.aligned_grid_maximum_absolute_error < 1e-12


def test_excluded_factor_negative_control_is_not_recoverable(
    four_factors: tuple[Factor, ...], sensitivities: dict[str, float]
) -> None:
    result = run_synthetic_quadratic_benchmark(
        four_factors,
        sensitivities,
        dropped_factor_linear_effect=2.0,
    )

    assert result.omitted_factor_limitation_detected
    assert result.omitted_factor_grid_rmse == pytest.approx(2 * np.sqrt(2 / 3))
    assert result.omitted_factor_grid_maximum_absolute_error == pytest.approx(2.0)
    assert "not biological validation" in result.to_dict()["interpretation"]


def test_benchmark_requires_four_factors_and_a_nonzero_finite_negative_control(
    four_factors: tuple[Factor, ...], sensitivities: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="exactly four"):
        run_synthetic_quadratic_benchmark(four_factors[:3], sensitivities)
    with pytest.raises(ValueError, match="finite and non-zero"):
        run_synthetic_quadratic_benchmark(
            four_factors, sensitivities, dropped_factor_linear_effect=0
        )
