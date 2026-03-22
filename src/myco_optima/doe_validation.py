"""Reproducible mathematical checks for the three-factor Box-Behnken plan.

These checks validate design construction and a deliberately synthetic response
surface. They do not validate a biological model, establish statistical power,
or show that 15 physical experiments universally replace an 81-condition grid.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

from myco_optima.doe import Factor, ScreeningDesign, build_sensitivity_guided_design

_CODED_LEVELS = (-1.0, 0.0, 1.0)
_QUADRATIC_PARAMETER_COUNT = 10


@dataclass(frozen=True)
class BoxBehnkenValidation:
    """Structural and estimability properties of one generated design."""

    retained_factors: tuple[str, ...]
    run_count: int
    edge_count: int
    unique_edge_count: int
    centre_count: int
    level_counts: tuple[tuple[int, int, int], ...]
    linear_column_sums: tuple[float, ...]
    maximum_absolute_linear_cross_product: float
    quadratic_parameter_count: int
    quadratic_rank: int
    residual_degrees_of_freedom: int
    conditional_pure_error_degrees_of_freedom: int
    candidate_count_valid: bool
    selected_count_valid: bool
    coded_levels_valid: bool
    edge_structure_valid: bool
    centre_structure_valid: bool
    role_labels_valid: bool
    level_balance_valid: bool
    linear_orthogonality_valid: bool
    quadratic_full_rank: bool

    @property
    def is_valid(self) -> bool:
        """Whether all claimed three-factor Box-Behnken properties hold."""

        return all(
            (
                self.candidate_count_valid,
                self.selected_count_valid,
                self.coded_levels_valid,
                self.edge_structure_valid,
                self.centre_structure_valid,
                self.role_labels_valid,
                self.level_balance_valid,
                self.linear_orthogonality_valid,
                self.quadratic_full_rank,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe validation evidence."""

        return {**asdict(self), "is_valid": self.is_valid}


@dataclass(frozen=True)
class SyntheticQuadraticBenchmark:
    """Noise-free software benchmark plus an excluded-factor negative control."""

    training_runs: int
    reference_grid_conditions: int
    retained_factors: tuple[str, ...]
    dropped_factor: str
    quadratic_parameter_count: int
    fitted_quadratic_rank: int
    coefficient_maximum_absolute_error: float
    aligned_grid_rmse: float
    aligned_grid_maximum_absolute_error: float
    dropped_factor_linear_effect: float
    omitted_factor_grid_rmse: float
    omitted_factor_grid_maximum_absolute_error: float

    @property
    def exact_aligned_recovery(self) -> bool:
        """Whether the deliberately well-specified benchmark is numerically exact."""

        return (
            self.fitted_quadratic_rank == self.quadratic_parameter_count
            and self.coefficient_maximum_absolute_error < 1e-10
            and self.aligned_grid_rmse < 1e-10
            and self.aligned_grid_maximum_absolute_error < 1e-10
        )

    @property
    def omitted_factor_limitation_detected(self) -> bool:
        """Whether the negative control exposes information absent from the design."""

        return self.omitted_factor_grid_rmse > 0 and (
            self.omitted_factor_grid_maximum_absolute_error > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe benchmark evidence."""

        return {
            **asdict(self),
            "exact_aligned_recovery": self.exact_aligned_recovery,
            "omitted_factor_limitation_detected": self.omitted_factor_limitation_detected,
            "interpretation": (
                "Noise-free quadratic software verification only; not biological validation, "
                "a power analysis, or evidence that 15 experiments universally replace 81."
            ),
        }


def _factor_lookup(factors: tuple[Factor, ...]) -> dict[str, Factor]:
    if len({factor.name for factor in factors}) != len(factors):
        raise ValueError("Factor names must be unique.")
    return {factor.name: factor for factor in factors}


def _coded_design(
    design: ScreeningDesign,
    factors: tuple[Factor, ...],
) -> np.ndarray:
    if len(design.retained_factors) != 3:
        raise ValueError("Validation requires exactly three retained factors.")
    factor_by_name = _factor_lookup(factors)
    missing_factors = set(design.retained_factors) - set(factor_by_name)
    if missing_factors:
        raise ValueError(f"Design references unknown factors: {sorted(missing_factors)}")
    missing_columns = set(design.retained_factors) - set(design.runs.columns)
    if missing_columns:
        raise ValueError(f"Design is missing factor columns: {sorted(missing_columns)}")

    try:
        decoded = design.runs.loc[:, design.retained_factors].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Design factor columns must be numeric.") from exc
    if decoded.ndim != 2 or decoded.shape[1] != 3 or not np.isfinite(decoded).all():
        raise ValueError("Design factor columns must be a finite three-column matrix.")

    centres = np.asarray(
        [factor_by_name[name].centre for name in design.retained_factors], dtype=float
    )
    half_ranges = np.asarray(
        [
            (factor_by_name[name].high - factor_by_name[name].low) / 2
            for name in design.retained_factors
        ],
        dtype=float,
    )
    if (
        not np.isfinite(centres).all()
        or not np.isfinite(half_ranges).all()
        or np.any(half_ranges <= 0)
    ):
        raise ValueError("Factors must have finite bounds with low < high.")
    return (decoded - centres) / half_ranges


def _quadratic_matrix(coded: np.ndarray) -> np.ndarray:
    if coded.ndim != 2 or coded.shape[1] != 3:
        raise ValueError("A three-factor coded matrix is required.")
    first, second, third = coded.T
    return np.column_stack(
        (
            np.ones(len(coded)),
            first,
            second,
            third,
            first**2,
            second**2,
            third**2,
            first * second,
            first * third,
            second * third,
        )
    )


def validate_box_behnken_design(
    design: ScreeningDesign,
    factors: tuple[Factor, ...] | list[Factor],
    *,
    tolerance: float = 1e-7,
) -> BoxBehnkenValidation:
    """Check balance, first-order orthogonality and quadratic estimability.

    The conditional pure-error degrees of freedom become real pure-error
    information only if the centre rows are run as independent experimental
    replicates under the same conditions.
    """

    if not isinstance(design, ScreeningDesign):
        raise TypeError("design must be a ScreeningDesign instance")
    factors = tuple(factors)
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be a positive finite number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite number")

    coded = _coded_design(design, factors)
    close_to_level = np.stack(
        [np.isclose(coded, level, atol=tolerance, rtol=0) for level in _CODED_LEVELS],
        axis=-1,
    )
    coded_levels_valid = bool(np.all(np.any(close_to_level, axis=-1)))
    nearest_indices = np.argmin(np.abs(coded[..., None] - np.asarray(_CODED_LEVELS)), axis=-1)
    snapped = np.asarray(_CODED_LEVELS)[nearest_indices]

    zero_counts = np.count_nonzero(np.isclose(coded, 0, atol=tolerance, rtol=0), axis=1)
    extreme_counts = np.count_nonzero(np.isclose(np.abs(coded), 1, atol=tolerance, rtol=0), axis=1)
    centre_mask = zero_counts == 3
    edge_mask = (zero_counts == 1) & (extreme_counts == 2)
    centre_count = int(centre_mask.sum())
    edge_count = int(edge_mask.sum())
    unique_edge_count = int(np.unique(snapped[edge_mask], axis=0).shape[0]) if edge_count else 0

    roles = (
        design.runs["design_role"].astype(str).to_numpy()
        if "design_role" in design.runs
        else np.full(len(coded), "missing")
    )
    expected_roles = np.where(
        centre_mask,
        "centre replicate",
        np.where(edge_mask, "interaction edge", "invalid"),
    )
    role_labels_valid = bool(np.array_equal(roles, expected_roles))

    level_counts = tuple(
        tuple(
            int(np.count_nonzero(np.isclose(coded[:, axis], level, atol=tolerance, rtol=0)))
            for level in _CODED_LEVELS
        )
        for axis in range(3)
    )
    expected_counts = (4, 4 + centre_count, 4)
    level_balance_valid = all(counts == expected_counts for counts in level_counts)

    linear_sums_array = coded.sum(axis=0)
    linear_information = coded.T @ coded
    off_diagonal = linear_information - np.diag(np.diag(linear_information))
    maximum_cross_product = float(np.max(np.abs(off_diagonal)))
    linear_orthogonality_valid = bool(
        np.allclose(linear_sums_array, 0, atol=tolerance, rtol=0)
        and np.allclose(off_diagonal, 0, atol=tolerance, rtol=0)
        and np.allclose(np.diag(linear_information), 8, atol=tolerance, rtol=0)
    )

    quadratic = _quadratic_matrix(coded)
    quadratic_rank = int(np.linalg.matrix_rank(quadratic, tol=tolerance))
    residual_df = len(coded) - quadratic_rank

    return BoxBehnkenValidation(
        retained_factors=design.retained_factors,
        run_count=len(coded),
        edge_count=edge_count,
        unique_edge_count=unique_edge_count,
        centre_count=centre_count,
        level_counts=level_counts,
        linear_column_sums=tuple(float(value) for value in linear_sums_array),
        maximum_absolute_linear_cross_product=maximum_cross_product,
        quadratic_parameter_count=_QUADRATIC_PARAMETER_COUNT,
        quadratic_rank=quadratic_rank,
        residual_degrees_of_freedom=residual_df,
        conditional_pure_error_degrees_of_freedom=max(centre_count - 1, 0),
        candidate_count_valid=design.candidate_count == 3 ** len(factors),
        selected_count_valid=design.selected_count == len(coded),
        coded_levels_valid=coded_levels_valid,
        edge_structure_valid=edge_count == 12 and unique_edge_count == 12,
        centre_structure_valid=centre_count >= 1 and edge_count + centre_count == len(coded),
        role_labels_valid=role_labels_valid,
        level_balance_valid=level_balance_valid,
        linear_orthogonality_valid=linear_orthogonality_valid,
        quadratic_full_rank=quadratic_rank == _QUADRATIC_PARAMETER_COUNT,
    )


def run_synthetic_quadratic_benchmark(
    factors: tuple[Factor, ...] | list[Factor],
    sensitivities: dict[str, float] | tuple[str, ...] | list[str],
    *,
    dropped_factor_linear_effect: float = 2.0,
) -> SyntheticQuadraticBenchmark:
    """Fit a known quadratic and compare predictions with the 81-point grid.

    The aligned response is noise-free, exactly quadratic in the three retained
    factors, and independent of the fourth factor. Exact recovery is therefore
    an algebraic software check under favourable assumptions. The negative
    control adds a linear effect for the excluded factor after fitting; its
    resulting error demonstrates what the 15-run design cannot recover.
    """

    factors = tuple(factors)
    if len(factors) != 4:
        raise ValueError("The 81-condition benchmark requires exactly four factors.")
    if not isinstance(dropped_factor_linear_effect, (int, float)) or isinstance(
        dropped_factor_linear_effect, bool
    ):
        raise TypeError("dropped_factor_linear_effect must be a finite number")
    dropped_factor_linear_effect = float(dropped_factor_linear_effect)
    if not math.isfinite(dropped_factor_linear_effect) or dropped_factor_linear_effect == 0:
        raise ValueError("dropped_factor_linear_effect must be finite and non-zero")

    design = build_sensitivity_guided_design(factors, sensitivities, centre_replicates=3)
    validation = validate_box_behnken_design(design, factors)
    if not validation.is_valid:
        raise ValueError("The generated Box-Behnken design failed structural validation.")

    retained = design.retained_factors
    dropped_names = tuple(factor.name for factor in factors if factor.name not in retained)
    if len(dropped_names) != 1:
        raise ValueError("The benchmark requires exactly one excluded factor.")
    dropped_name = dropped_names[0]

    training_coded = _coded_design(design, factors)
    training_matrix = _quadratic_matrix(training_coded)
    # Fixed, non-degenerate coefficients in the documented matrix order:
    # intercept, linear, pure quadratic, then pairwise interactions.
    expected_coefficients = np.asarray((12.0, 2.5, -1.75, 1.25, -0.8, -0.4, -0.6, 0.7, -0.3, 0.5))
    training_response = training_matrix @ expected_coefficients
    fitted_coefficients, _, fitted_rank, _ = np.linalg.lstsq(
        training_matrix, training_response, rcond=None
    )

    full_grid = np.asarray(tuple(product(_CODED_LEVELS, repeat=4)), dtype=float)
    factor_positions = {factor.name: index for index, factor in enumerate(factors)}
    retained_grid = full_grid[:, [factor_positions[name] for name in retained]]
    reference_matrix = _quadratic_matrix(retained_grid)
    aligned_reference = reference_matrix @ expected_coefficients
    predicted = reference_matrix @ fitted_coefficients
    aligned_error = predicted - aligned_reference

    omitted_reference = aligned_reference + (
        dropped_factor_linear_effect * full_grid[:, factor_positions[dropped_name]]
    )
    omitted_error = predicted - omitted_reference

    return SyntheticQuadraticBenchmark(
        training_runs=len(training_matrix),
        reference_grid_conditions=len(full_grid),
        retained_factors=retained,
        dropped_factor=dropped_name,
        quadratic_parameter_count=_QUADRATIC_PARAMETER_COUNT,
        fitted_quadratic_rank=int(fitted_rank),
        coefficient_maximum_absolute_error=float(
            np.max(np.abs(fitted_coefficients - expected_coefficients))
        ),
        aligned_grid_rmse=float(np.sqrt(np.mean(aligned_error**2))),
        aligned_grid_maximum_absolute_error=float(np.max(np.abs(aligned_error))),
        dropped_factor_linear_effect=dropped_factor_linear_effect,
        omitted_factor_grid_rmse=float(np.sqrt(np.mean(omitted_error**2))),
        omitted_factor_grid_maximum_absolute_error=float(np.max(np.abs(omitted_error))),
    )


def _default_evidence() -> dict[str, Any]:
    factors = (
        Factor("Carbon", 10, 30, "g/L"),
        Factor("Nitrogen", 1, 7, "g/L"),
        Factor("Oxygen", 2, 18, "mmol/gDW/h"),
        Factor("Phosphate", 0.2, 1.2, "g/L"),
    )
    sensitivities = {"Carbon": 0.8, "Nitrogen": 0.6, "Oxygen": 0.4, "Phosphate": 0.1}
    design = build_sensitivity_guided_design(factors, sensitivities)
    return {
        "design_validation": validate_box_behnken_design(design, factors).to_dict(),
        "synthetic_benchmark": run_synthetic_quadratic_benchmark(factors, sensitivities).to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(_default_evidence(), indent=2, allow_nan=False))
